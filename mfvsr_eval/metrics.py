from __future__ import annotations

import math
import types
from collections import defaultdict

import numpy as np

from .regions import interior_mask, support_box


def finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


class DISTSNonFiniteError(RuntimeError):
    """Raised when both accelerated and native GPU DISTS are non-finite."""


def _chunked_l2pool_forward(module, value):
    """DISTS L2 pooling with mathematically identical channel-block dispatch."""
    import torch

    squared = value.square()
    outputs = []
    chunk_size = module._mfvsr_channel_chunk
    for start in range(0, squared.shape[1], chunk_size):
        stop = min(start + chunk_size, squared.shape[1])
        block = torch.nn.functional.conv2d(
            squared[:, start:stop],
            module.filter[start:stop],
            stride=module.stride,
            padding=module.padding,
            groups=stop - start,
        )
        if not bool(torch.isfinite(block).all()):
            # Some ROCm/MIOpen versions return NaN for particular shapes even
            # after moderate group splitting. Per-channel convolution is the
            # same depthwise operation and avoids that backend kernel path.
            block = torch.cat([
                torch.nn.functional.conv2d(
                    squared[:, channel:channel + 1],
                    module.filter[channel:channel + 1],
                    stride=module.stride,
                    padding=module.padding,
                    groups=1,
                )
                for channel in range(start, stop)
            ], dim=1)
        outputs.append(block)
    return (torch.cat(outputs, dim=1) + 1e-12).sqrt()


def patch_dists_l2pool_channel_chunks(model, chunk_size: int) -> None:
    """Avoid a ROCm depthwise-convolution NaN without changing DISTS weights."""
    if chunk_size <= 0:
        return
    for module in model.modules():
        if module.__class__.__name__ == "L2pooling":
            module._mfvsr_channel_chunk = chunk_size
            module.forward = types.MethodType(_chunked_l2pool_forward, module)


def masked_psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    """Return standard data_range=255 PSNR and the pooling statistics."""
    if prediction.shape != target.shape or prediction.shape[:2] != mask.shape:
        raise ValueError("PSNR inputs have incompatible shapes")
    value_count = int(mask.sum()) * prediction.shape[2]
    if value_count == 0:
        return {"value": None, "sse": 0.0, "value_count": 0, "exact_match": False}
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    sse = float(np.square(difference[mask]).sum())
    if sse == 0:
        return {"value": None, "sse": 0.0, "value_count": value_count, "exact_match": True}
    mse = sse / value_count
    return {
        "value": 10.0 * math.log10((255.0**2) / mse),
        "sse": sse,
        "value_count": value_count,
        "exact_match": False,
    }


def ssim_scalar_and_map(prediction: np.ndarray, target: np.ndarray, config: dict):
    """Wang-style SSIM through scikit-image with explicit parameters."""
    from skimage.metrics import structural_similarity

    score, score_map = structural_similarity(
        target.astype(np.float32) / 255.0,
        prediction.astype(np.float32) / 255.0,
        data_range=1.0,
        channel_axis=2,
        full=True,
        win_size=int(config.get("win_size", 11)),
        gaussian_weights=bool(config.get("gaussian_weights", True)),
        sigma=float(config.get("sigma", 1.5)),
        use_sample_covariance=bool(config.get("use_sample_covariance", False)),
    )
    if score_map.ndim == 3:
        score_map = score_map.mean(axis=2)
    return float(score), score_map


def masked_map_mean(score_map: np.ndarray, mask: np.ndarray) -> tuple[float | None, int]:
    if score_map.shape != mask.shape:
        raise ValueError(f"map/mask shape mismatch: {score_map.shape} != {mask.shape}")
    count = int(mask.sum())
    return (float(score_map[mask].mean()), count) if count else (None, 0)


def _expand_box_to_minimum(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    minimum: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    height, width = shape
    need_width = max(0, minimum - (x1 - x0))
    need_height = max(0, minimum - (y1 - y0))
    x0 -= need_width // 2
    x1 += need_width - need_width // 2
    y0 -= need_height // 2
    y1 += need_height - need_height // 2
    if x0 < 0:
        x1 = min(width, x1 - x0)
        x0 = 0
    if y0 < 0:
        y1 = min(height, y1 - y0)
        y0 = 0
    if x1 > width:
        x0 = max(0, x0 - (x1 - width))
        x1 = width
    if y1 > height:
        y0 = max(0, y0 - (y1 - height))
        y1 = height
    return x0, y0, x1, y1


def _dists_input_geometry(
    height: int,
    width: int,
    maximum_long_side: int,
) -> tuple[list[int], bool]:
    """Return DISTS [width, height] after the optional long-side cap."""
    if maximum_long_side <= 0 or max(height, width) <= maximum_long_side:
        return [width, height], False
    scale = maximum_long_side / max(height, width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return [resized_width, resized_height], True


class PerceptualMetrics:
    """Official LPIPS and DISTS backends with explicit regional conditioning."""

    def __init__(
        self,
        enabled: set[str],
        config: dict,
        device: str,
        runtime_config: dict | None = None,
    ):
        self.enabled = enabled
        self.config = config
        self.device = device
        self.runtime_config = runtime_config or {}
        self.lpips_spatial = None
        self.lpips_scalar = None
        self.dists_model = None
        self.dists_cpu_model = None
        if "lpips" in enabled:
            import lpips

            backbone = config["lpips"].get("backbone", "alex")
            version = config["lpips"].get("version", "0.1")
            self.lpips_spatial = lpips.LPIPS(net=backbone, version=version, spatial=True).to(device).eval()
            self.lpips_scalar = lpips.LPIPS(net=backbone, version=version, spatial=False).to(device).eval()
        if "dists" in enabled:
            from DISTS_pytorch import DISTS

            self.dists_model = DISTS().to(device).eval()
            if str(device).split(":", 1)[0] != "cpu":
                patch_dists_l2pool_channel_chunks(
                    self.dists_model,
                    int(self.runtime_config.get("dists_l2pool_channel_chunk", 0)),
                )

    def _tensor(self, image: np.ndarray, minus_one_to_one: bool, device: str | None = None):
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()[None] / 255.0
        if minus_one_to_one:
            tensor = tensor * 2.0 - 1.0
        return tensor.to(device or self.device)

    def lpips_outputs(self, prediction: np.ndarray, target: np.ndarray):
        if self.lpips_spatial is None:
            return None, None
        import torch

        pred = self._tensor(prediction, minus_one_to_one=True)
        gt = self._tensor(target, minus_one_to_one=True)
        with torch.inference_mode():
            spatial = self.lpips_spatial(pred, gt, normalize=False).detach()
            scalar = float(self.lpips_scalar(pred, gt, normalize=False).item())
        return scalar, spatial

    def masked_lpips(self, spatial_map, mask: np.ndarray) -> tuple[float | None, float]:
        if spatial_map is None or not mask.any():
            return None, 0.0
        import torch

        with torch.inference_mode():
            mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None].to(self.device)
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor,
                size=spatial_map.shape[-2:],
                mode="area",
            )
            mass = float(mask_tensor.sum().item())
            if mass == 0:
                return None, 0.0
            return float((spatial_map * mask_tensor).sum().item() / mass), mass

    def _dists_on_device(
        self,
        model,
        prediction: np.ndarray,
        target: np.ndarray,
        device: str,
        resize_to: list[int] | None = None,
    ) -> float | None:
        import torch

        pred_tensor = self._tensor(prediction, minus_one_to_one=False, device=device)
        target_tensor = self._tensor(target, minus_one_to_one=False, device=device)
        with torch.inference_mode():
            if resize_to is not None:
                output_size = (resize_to[1], resize_to[0])
                pred_tensor = torch.nn.functional.interpolate(
                    pred_tensor, size=output_size, mode="area"
                )
                target_tensor = torch.nn.functional.interpolate(
                    target_tensor, size=output_size, mode="area"
                )
            return finite_or_none(model(pred_tensor, target_tensor).item())

    def _get_cpu_dists_model(self):
        if self.dists_cpu_model is None:
            from DISTS_pytorch import DISTS

            # A separate lazy CPU copy keeps the primary GPU model resident and
            # uses the identical released architecture and weights.
            self.dists_cpu_model = DISTS().to("cpu").eval()
        return self.dists_cpu_model

    def dists_with_diagnostics(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        region_mask: np.ndarray,
        context_mask: np.ndarray | None,
        conditioned: bool,
    ) -> dict:
        if self.dists_model is None or not region_mask.any():
            return {
                "value": None,
                "backend": None,
                "input_size": None,
                "input_resized": False,
                "fallback_count": 0,
                "gpu_retry_count": 0,
                "nonfinite_count": 0,
                "null_reason": "empty_support_or_disabled",
            }
        cfg = self.config["dists"]
        if context_mask is None:
            pred_crop = prediction
            target_crop = target
            mask_crop = region_mask
        else:
            box = support_box(context_mask, int(cfg.get("context_margin", 16)))
            if box is None:
                return {
                    "value": None,
                    "backend": None,
                    "input_size": None,
                    "input_resized": False,
                    "fallback_count": 0,
                    "gpu_retry_count": 0,
                    "nonfinite_count": 0,
                    "null_reason": "empty_context",
                }
            box = _expand_box_to_minimum(box, region_mask.shape, int(cfg.get("minimum_crop_size", 64)))
            x0, y0, x1, y1 = box
            pred_crop = prediction[y0:y1, x0:x1].copy()
            target_crop = target[y0:y1, x0:x1].copy()
            mask_crop = region_mask[y0:y1, x0:x1]
        if conditioned:
            pred_crop[~mask_crop] = target_crop[~mask_crop]
        input_resized = False
        if context_mask is None:
            input_size, input_resized = _dists_input_geometry(
                int(pred_crop.shape[0]),
                int(pred_crop.shape[1]),
                int(self.runtime_config.get("dists_overall_max_input_long_side", 0) or 0),
            )
        else:
            input_size = [int(pred_crop.shape[1]), int(pred_crop.shape[0])]
        resize_to = input_size if input_resized else None
        primary_is_cpu = str(self.device).split(":", 1)[0] == "cpu"
        native_conditioned = (
            conditioned
            and not primary_is_cpu
            and bool(self.runtime_config.get("dists_native_gpu_for_conditioned", False))
        )
        if native_conditioned:
            import torch

            with torch.backends.cudnn.flags(enabled=False):
                value = self._dists_on_device(
                    self.dists_model, pred_crop, target_crop, self.device, resize_to
                )
        else:
            value = self._dists_on_device(
                self.dists_model, pred_crop, target_crop, self.device, resize_to
            )
        if value is not None:
            return {
                "value": value,
                "backend": self.device,
                "input_size": input_size,
                "input_resized": input_resized,
                "fallback_count": 0,
                "gpu_retry_count": 0,
                "nonfinite_count": 0,
                "null_reason": None,
            }

        action = str(self.runtime_config.get("dists_nonfinite_action", "legacy_policy"))
        if action == "gpu_retry" and not primary_is_cpu and not native_conditioned:
            import torch

            # ROCm/MIOpen can select a convolution solver that returns NaN for
            # particular high-resolution or small-crop shapes. PyTorch's native
            # GPU convolution evaluates the same released model and weights.
            with torch.backends.cudnn.flags(enabled=False):
                retry = self._dists_on_device(
                    self.dists_model, pred_crop, target_crop, self.device, resize_to
                )
            if retry is not None:
                return {
                    "value": retry,
                    "backend": self.device,
                    "input_size": input_size,
                    "input_resized": input_resized,
                    "fallback_count": 0,
                    "gpu_retry_count": 1,
                    "nonfinite_count": 0,
                    "null_reason": None,
                }
            raise DISTSNonFiniteError(
                "DISTS returned a non-finite value on GPU both before and after "
                "a native GPU convolution retry; CPU fallback is disabled by runtime policy"
            )
        if action == "gpu_retry" and native_conditioned:
            raise DISTSNonFiniteError(
                "DISTS remained non-finite on the native GPU convolution path; "
                "CPU fallback is disabled by runtime policy"
            )

        policy = str(cfg.get("nonfinite_policy", "null"))
        if policy == "cpu_fallback_same_weights" and not primary_is_cpu:
            fallback = self._dists_on_device(
                self._get_cpu_dists_model(), pred_crop, target_crop, "cpu", resize_to
            )
            if fallback is not None:
                return {
                    "value": fallback,
                    "backend": "cpu_fallback",
                    "input_size": input_size,
                    "input_resized": input_resized,
                    "fallback_count": 1,
                    "gpu_retry_count": 0,
                    "nonfinite_count": 0,
                    "null_reason": None,
                }
            return {
                "value": None,
                "backend": "cpu_fallback",
                "input_size": input_size,
                "input_resized": input_resized,
                "fallback_count": 1,
                "gpu_retry_count": 0,
                "nonfinite_count": 1,
                "null_reason": "nonfinite_after_cpu_fallback",
            }
        return {
            "value": None,
            "backend": self.device,
            "input_size": input_size,
            "input_resized": input_resized,
            "fallback_count": 0,
            "gpu_retry_count": 0,
            "nonfinite_count": 1,
            "null_reason": "nonfinite_primary",
        }

    def dists(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        region_mask: np.ndarray,
        context_mask: np.ndarray | None,
        conditioned: bool,
    ) -> float | None:
        """Backward-compatible value-only DISTS interface."""
        return self.dists_with_diagnostics(
            prediction, target, region_mask, context_mask, conditioned
        )["value"]


class RegionAccumulator:
    def __init__(self):
        self.totals = defaultdict(lambda: defaultdict(float))
        self.series = defaultdict(lambda: defaultdict(list))

    def add(self, region: str, row: dict) -> None:
        total = self.totals[region]
        for key in (
            "effective_pixels", "target_pixels", "sse", "psnr_value_count",
            "ssim_sum", "ssim_count", "lpips_sum", "lpips_count",
            "dists_sum", "dists_count", "dists_fallback_count",
            "dists_gpu_retry_count", "dists_nonfinite_count", "twe_sum", "twe_count",
            "twe_target_sum", "twe_target_count", "tc_sum", "tc_count",
        ):
            total[key] += float(row.get(key) or 0.0)
        for key in ("psnr", "ssim", "lpips", "dists", "twe", "twe_target", "twe_excess", "tc"):
            value = row.get(key)
            if value is not None:
                self.series[region][key].append(float(value))

    @staticmethod
    def _mean(total: dict, name: str) -> float | None:
        count = total[f"{name}_count"]
        return finite_or_none(total[f"{name}_sum"] / count) if count else None

    def summary(self, region: str) -> dict:
        total = self.totals[region]
        sse = total["sse"]
        count = total["psnr_value_count"]
        pooled_psnr = None
        if count and sse:
            pooled_psnr = 10.0 * math.log10((255.0**2) / (sse / count))
        metrics = {
            "psnr": finite_or_none(pooled_psnr),
            "infinite_psnr": bool(count and sse == 0),
            "ssim": self._mean(total, "ssim"),
            "lpips": self._mean(total, "lpips"),
            "dists": self._mean(total, "dists"),
            "twe": self._mean(total, "twe"),
            "twe_target_baseline": self._mean(total, "twe_target"),
            "tc": self._mean(total, "tc"),
        }
        if metrics["twe"] is not None and metrics["twe_target_baseline"] is not None:
            metrics["twe_excess"] = metrics["twe"] - metrics["twe_target_baseline"]
        else:
            metrics["twe_excess"] = None
        distributions = {}
        for metric, values in self.series[region].items():
            array = np.asarray(values, dtype=np.float64)
            if len(array):
                distributions[metric] = {
                    "mean": finite_or_none(array.mean()),
                    "std": finite_or_none(array.std()),
                    "p05": finite_or_none(np.percentile(array, 5)),
                    "p50": finite_or_none(np.percentile(array, 50)),
                    "p95": finite_or_none(np.percentile(array, 95)),
                    "frame_count": int(len(array)),
                }
        target_pixels = total["target_pixels"]
        sufficient = {key: finite_or_none(value) for key, value in total.items()}
        return {
            "effective_pixels": int(total["effective_pixels"]),
            "target_pixels": int(target_pixels),
            "coverage": finite_or_none(total["effective_pixels"] / target_pixels) if target_pixels else None,
            "metrics": metrics,
            "sufficient_statistics": sufficient,
            "frame_distributions": distributions,
        }


def ssim_support(shape: tuple[int, int], config: dict) -> np.ndarray:
    return interior_mask(shape, int(config.get("win_size", 11)) // 2)
