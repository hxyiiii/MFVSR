from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
import torch

from mfvsr_eval.metrics import (
    DISTSNonFiniteError,
    PerceptualMetrics,
    masked_map_mean,
    patch_dists_l2pool_channel_chunks,
    ssim_scalar_and_map,
)
from mfvsr_eval.record_compatibility import runtime_record_is_compatible
from mfvsr_eval.temporal import TorchvisionRAFT, temporal_change_map, temporal_maps


class ZeroFlow:
    def flow(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.zeros((*first.shape[:2], 2), dtype=np.float32)


class MeanAbsoluteModel(torch.nn.Module):
    def forward(self, first, second):
        return torch.abs(first - second).mean().reshape(1)


class MetricBackendTests(unittest.TestCase):
    def test_flow_free_temporal_change_matches_original_definition(self):
        previous = np.zeros((4, 4, 3), dtype=np.uint8)
        current = np.full((4, 4, 3), 10, dtype=np.uint8)
        valid = np.ones((4, 4), dtype=bool)
        change, support = temporal_change_map(
            previous, current, previous, previous, valid, valid
        )
        self.assertAlmostEqual(float(change.mean()), 10 / 255, places=7)
        self.assertTrue(support.all())

    def test_chunked_dists_l2pool_is_mathematically_equivalent(self):
        class L2pooling(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stride = 2
                self.padding = 1
                kernel = torch.tensor([
                    [0.0625, 0.1250, 0.0625],
                    [0.1250, 0.2500, 0.1250],
                    [0.0625, 0.1250, 0.0625],
                ])
                self.register_buffer("filter", kernel[None, None].repeat(8, 1, 1, 1))

            def forward(self, value):
                output = torch.nn.functional.conv2d(
                    value.square(), self.filter, stride=2, padding=1, groups=8
                )
                return (output + 1e-12).sqrt()

        reference = L2pooling()
        patched = L2pooling()
        patched.load_state_dict(reference.state_dict())
        patch_dists_l2pool_channel_chunks(patched, 3)
        value = torch.rand((1, 8, 32, 48))
        torch.testing.assert_close(patched(value), reference(value), rtol=0, atol=0)

    def test_ssim_identity_and_regional_map(self):
        image = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3) % 256
        image = image.astype(np.uint8)
        scalar, score_map = ssim_scalar_and_map(image, image, {
            "win_size": 11, "gaussian_weights": True, "sigma": 1.5
        })
        self.assertAlmostEqual(scalar, 1.0, places=7)
        support = np.zeros((32, 32), dtype=bool)
        support[10:20, 10:20] = True
        regional, count = masked_map_mean(score_map, support)
        self.assertEqual(count, 100)
        self.assertAlmostEqual(regional, 1.0, places=7)

    def test_lpips_mask_mass_and_tensor_ranges(self):
        metric = PerceptualMetrics.__new__(PerceptualMetrics)
        metric.device = "cpu"
        spatial = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True
        value, mass = metric.masked_lpips(spatial, mask)
        self.assertAlmostEqual(value, 1.0)
        self.assertAlmostEqual(mass, 1.0)
        black = np.zeros((2, 2, 3), dtype=np.uint8)
        white = np.full((2, 2, 3), 255, dtype=np.uint8)
        self.assertEqual(float(metric._tensor(black, True).min()), -1.0)
        self.assertEqual(float(metric._tensor(white, True).max()), 1.0)
        self.assertEqual(float(metric._tensor(black, False).min()), 0.0)
        self.assertEqual(float(metric._tensor(white, False).max()), 1.0)

    def test_conditioned_dists_removes_nonregion_difference(self):
        metric = PerceptualMetrics.__new__(PerceptualMetrics)
        metric.device = "cpu"
        metric.dists_model = MeanAbsoluteModel()
        metric.runtime_config = {}
        metric.config = {"dists": {"context_margin": 0, "minimum_crop_size": 8}}
        target = np.zeros((16, 16, 3), dtype=np.uint8)
        prediction = target.copy()
        prediction[4:8, 4:8] = 255
        prediction[8:12, 8:12] = 255
        region = np.zeros((16, 16), dtype=bool)
        region[4:8, 4:8] = True
        context = np.ones((16, 16), dtype=bool)
        conditioned = metric.dists(prediction, target, region, context, conditioned=True)
        unconditioned = metric.dists(prediction, target, region, context, conditioned=False)
        self.assertAlmostEqual(conditioned, 16 / 256)
        self.assertAlmostEqual(unconditioned, 32 / 256)

    def test_overall_dists_caps_only_long_side_with_area_downsampling(self):
        metric = PerceptualMetrics.__new__(PerceptualMetrics)
        metric.device = "cpu"
        metric.dists_model = MeanAbsoluteModel()
        metric.runtime_config = {"dists_overall_max_input_long_side": 8}
        metric.config = {"dists": {"context_margin": 0, "minimum_crop_size": 8}}
        target = np.zeros((8, 16, 3), dtype=np.uint8)
        prediction = np.full_like(target, 255)
        region = np.ones((8, 16), dtype=bool)
        result = metric.dists_with_diagnostics(
            prediction, target, region, None, conditioned=False
        )
        self.assertAlmostEqual(result["value"], 1.0)
        self.assertEqual(result["input_size"], [8, 4])
        self.assertTrue(result["input_resized"])

    def test_conditioned_dists_keeps_native_context_above_overall_cap(self):
        metric = PerceptualMetrics.__new__(PerceptualMetrics)
        metric.device = "cpu"
        metric.dists_model = MeanAbsoluteModel()
        metric.runtime_config = {"dists_overall_max_input_long_side": 8}
        metric.config = {"dists": {"context_margin": 0, "minimum_crop_size": 8}}
        target = np.zeros((16, 16, 3), dtype=np.uint8)
        prediction = np.full_like(target, 255)
        region = np.ones((16, 16), dtype=bool)
        result = metric.dists_with_diagnostics(
            prediction, target, region, region, conditioned=True
        )
        self.assertEqual(result["input_size"], [16, 16])
        self.assertFalse(result["input_resized"])

    def test_high_resolution_runtime_policy_selectively_invalidates_old_records(self):
        config = {"runtime": {"dists_overall_max_input_long_side": 1920}}
        low_resolution = {"target_probe": {"width": 1920, "height": 1080}}
        high_resolution_old = {"target_probe": {"width": 4096, "height": 2160}}
        high_resolution_current = {
            **high_resolution_old,
            "dists_execution": {
                "overall_max_input_long_side": 1920,
                "overall_downsampling": "pytorch_area",
                "conditioned_context_resized": False,
            },
        }
        self.assertTrue(runtime_record_is_compatible(low_resolution, config, {"dists"}))
        self.assertFalse(runtime_record_is_compatible(high_resolution_old, config, {"dists"}))
        self.assertTrue(runtime_record_is_compatible(high_resolution_current, config, {"dists"}))
        self.assertTrue(runtime_record_is_compatible(high_resolution_old, config, {"psnr"}))

    def test_nonfinite_gpu_dists_uses_same_weight_cpu_fallback(self):
        metric = PerceptualMetrics.__new__(PerceptualMetrics)
        metric.device = "cuda:0"
        metric.dists_model = object()
        metric.dists_cpu_model = object()
        metric.runtime_config = {}
        metric.config = {"dists": {
            "context_margin": 0,
            "minimum_crop_size": 8,
            "nonfinite_policy": "cpu_fallback_same_weights",
        }}
        metric._dists_on_device = Mock(side_effect=[None, 0.25])
        metric._get_cpu_dists_model = Mock(return_value=metric.dists_cpu_model)
        target = np.zeros((16, 16, 3), dtype=np.uint8)
        prediction = target.copy()
        prediction[4:8, 4:8] = 255
        region = np.zeros((16, 16), dtype=bool)
        region[4:8, 4:8] = True
        result = metric.dists_with_diagnostics(
            prediction, target, region, np.ones((16, 16), dtype=bool), True
        )
        self.assertEqual(result["value"], 0.25)
        self.assertEqual(result["backend"], "cpu_fallback")
        self.assertEqual(result["fallback_count"], 1)
        self.assertEqual(result["gpu_retry_count"], 0)
        self.assertEqual(result["nonfinite_count"], 0)
        self.assertIsNone(result["null_reason"])
        self.assertEqual(metric._dists_on_device.call_count, 2)

    def test_nonfinite_gpu_dists_retries_native_gpu_without_cpu_fallback(self):
        metric = PerceptualMetrics.__new__(PerceptualMetrics)
        metric.device = "cuda:0"
        metric.dists_model = object()
        metric.dists_cpu_model = None
        metric.runtime_config = {"dists_nonfinite_action": "gpu_retry"}
        metric.config = {"dists": {
            "context_margin": 0,
            "minimum_crop_size": 8,
            "nonfinite_policy": "cpu_fallback_same_weights",
        }}
        metric._dists_on_device = Mock(side_effect=[None, 0.25])
        metric._get_cpu_dists_model = Mock()
        target = np.zeros((16, 16, 3), dtype=np.uint8)
        region = np.ones((16, 16), dtype=bool)
        result = metric.dists_with_diagnostics(
            target, target, region, region, True
        )
        self.assertEqual(result["value"], 0.25)
        self.assertEqual(result["backend"], "cuda:0")
        self.assertEqual(result["fallback_count"], 0)
        self.assertEqual(result["gpu_retry_count"], 1)
        metric._get_cpu_dists_model.assert_not_called()
        self.assertIs(metric._dists_on_device.call_args_list[0].args[0], metric.dists_model)
        self.assertIs(metric._dists_on_device.call_args_list[1].args[0], metric.dists_model)

    def test_nonfinite_gpu_dists_fails_after_single_gpu_retry(self):
        metric = PerceptualMetrics.__new__(PerceptualMetrics)
        metric.device = "cuda:0"
        metric.dists_model = object()
        metric.dists_cpu_model = None
        metric.runtime_config = {"dists_nonfinite_action": "gpu_retry"}
        metric.config = {"dists": {
            "context_margin": 0,
            "minimum_crop_size": 8,
            "nonfinite_policy": "cpu_fallback_same_weights",
        }}
        metric._dists_on_device = Mock(side_effect=[None, None])
        metric._get_cpu_dists_model = Mock()
        target = np.zeros((16, 16, 3), dtype=np.uint8)
        region = np.ones((16, 16), dtype=bool)
        with self.assertRaises(DISTSNonFiniteError):
            metric.dists_with_diagnostics(target, target, region, region, True)
        metric._get_cpu_dists_model.assert_not_called()

    def test_temporal_zero_flow_values_and_region_warp(self):
        shape = (16, 16, 3)
        previous_target = np.zeros(shape, dtype=np.uint8)
        target = previous_target.copy()
        previous_prediction = previous_target.copy()
        prediction = np.full(shape, 10, dtype=np.uint8)
        valid = np.ones(shape[:2], dtype=bool)
        region = np.zeros(shape[:2], dtype=bool)
        region[4:12, 4:12] = True
        result = temporal_maps(
            previous_prediction, prediction, previous_target, target,
            valid, valid, {"bbox": region}, ZeroFlow()
        )
        expected_twe = (10 / 255) ** 2
        expected_tc = 10 / 255
        self.assertTrue(result.warp_valid.all())
        self.assertAlmostEqual(float(result.prediction_warp_error.mean()), expected_twe, places=7)
        self.assertAlmostEqual(float(result.target_warp_error.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(result.change_error.mean()), expected_tc, places=7)
        self.assertTrue(np.array_equal(result.warped_previous_regions["bbox"], region))

    def test_raft_input_stays_uint8_until_official_transform(self):
        estimator = TorchvisionRAFT.__new__(TorchvisionRAFT)
        estimator.torch = torch
        estimator.device = "cpu"
        tensor = estimator._tensor(np.full((8, 8, 3), 255, dtype=np.uint8))
        self.assertEqual(tensor.dtype, torch.uint8)
        self.assertEqual(int(tensor.max()), 255)

    def test_raft_input_cap_preserves_aspect_ratio_and_native_1080p(self):
        estimator = TorchvisionRAFT.__new__(TorchvisionRAFT)
        estimator.max_input_long_side = 960
        estimator.resize_trigger_long_side = 1920
        self.assertEqual(estimator.input_size(1080, 1920), (1080, 1920))
        self.assertEqual(estimator.input_size(2160, 3840), (540, 960))
        self.assertEqual(estimator.input_size(3840, 2160), (960, 540))

    def test_raft_flow_releases_model_owned_correlation_pyramid(self):
        class FakeCorrBlock:
            corr_pyramid = [torch.ones(2)]

        class FakeModel:
            corr_block = FakeCorrBlock()

            def __call__(self, first, second):
                self.corr_block.corr_pyramid = [torch.ones(1024)]
                height, width = first.shape[-2:]
                return [torch.zeros((1, 2, height, width))]

        estimator = TorchvisionRAFT.__new__(TorchvisionRAFT)
        estimator.torch = torch
        estimator.device = "cpu"
        estimator.max_input_long_side = 0
        estimator.resize_trigger_long_side = 0
        estimator.transforms = lambda first, second: (first.float(), second.float())
        estimator.model = FakeModel()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        result = estimator.flow(image, image)
        self.assertEqual(result.shape, (8, 8, 2))
        self.assertEqual(len(estimator.model.corr_block.corr_pyramid), 1)
        self.assertEqual(estimator.model.corr_block.corr_pyramid[0].numel(), 1)


if __name__ == "__main__":
    unittest.main()
