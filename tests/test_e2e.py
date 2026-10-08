from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mfvsr_eval.run import main as run_main
from mfvsr_eval.evaluator import _coverage_classification, _validate_fps_pair
from mfvsr_eval.config import load_config
from mfvsr_eval.io import load_manifest_rows, sample_from_row
from mfvsr_eval.inventory import inventory_sample
from mfvsr_eval.profiles import apply_evaluation_profile
from mfvsr_eval.reuse_records import reuse_sample
from mfvsr_eval.schema import VideoProbe


def encode_rgb(path: Path, frames: np.ndarray, fps: int = 2) -> None:
    height, width = frames.shape[1:3]
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast", str(path),
    ]
    subprocess.run(command, input=frames.tobytes(), check=True)


class EndToEndTests(unittest.TestCase):
    def test_fps_comparison_uses_exact_fraction_boundary(self):
        target = VideoProbe(64, 64, 25.0, 25, 1, 2, 0.08)
        boundary = VideoProbe(64, 64, 25.01, 2501, 100, 2, 0.08)
        details = _validate_fps_pair("prediction", boundary, target, 0.01)
        self.assertTrue(details["matched"])
        self.assertEqual(details["delta_fraction"], "1/100")
        ntsc = VideoProbe(64, 64, 30000 / 1001, 30000, 1001, 2, 0.08)
        with self.assertRaisesRegex(ValueError, "FPS mismatch"):
            _validate_fps_pair("prediction", ntsc, target, 0.01)

    def test_completeness_requires_both_spatial_and_temporal_coverage(self):
        self.assertEqual(_coverage_classification(1.0, 1.0), ("complete", []))
        self.assertEqual(
            _coverage_classification(0.99, 1.0),
            ("partial", ["partial_spatial_coverage"]),
        )
        self.assertEqual(
            _coverage_classification(1.0, 0.99),
            ("partial", ["partial_temporal_coverage"]),
        )
        self.assertEqual(
            _coverage_classification(0.99, 0.99),
            ("partial", ["partial_spatial_coverage", "partial_temporal_coverage"]),
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video_id = "case_Fixed-bottom_Static_Plain-white_Monolingual-English"
        input_root = self.root / "benchmark"
        sample_dir = input_root / "AI_gen" / "monolingual" / self.video_id
        sample_dir.mkdir(parents=True)
        target = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        prediction = target.copy()
        prediction[:, 20:30, 10:25] = 20
        glyph = np.zeros_like(target)
        glyph[:, 20:30, 10:25] = 255
        self.input_path = sample_dir / f"{self.video_id}.mp4"
        self.gt_path = self.root / "gt.mp4"
        self.mask_path = sample_dir / f"{self.video_id}_mask.mp4"
        self.bbox_path = sample_dir / f"{self.video_id}_mask.json"
        encode_rgb(self.input_path, target)
        encode_rgb(self.gt_path, target)
        encode_rgb(self.mask_path, glyph)
        model_root = self.root / "model"
        output_dir = model_root / "AI_gen" / "monolingual" / self.video_id / self.video_id
        output_dir.mkdir(parents=True)
        self.prediction_path = output_dir / f"{self.video_id}_cleaned.mp4"
        encode_rgb(self.prediction_path, prediction)
        (output_dir / "info.json").write_text(json.dumps({
            "status": "success",
            "output_raw": {"video_path": str(self.prediction_path), "num_frames": 2},
            "performance": {"total_inference_time_seconds": 1.0, "peak_gpu_memory_MB": 100},
            "environment": {"gpu": "test"},
        }), encoding="utf-8")
        self.bbox_path.write_text(json.dumps([
            {"frame": frame, "x": 10, "y": 20, "width": 15, "height": 10}
            for frame in range(2)
        ]), encoding="utf-8")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"samples": [{
            "video_id": self.video_id,
            "split": "AI_gen/monolingual",
            "input_video_path": str(self.input_path),
            "gt_video_path": str(self.gt_path),
            "glyph_mask_path": str(self.mask_path),
            "bbox_json_path": str(self.bbox_path),
        }]}), encoding="utf-8")
        self.output_root = self.root / "evaluation"
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({
            "manifest": str(self.manifest),
            "input_root": str(input_root),
            "output_root": str(self.output_root),
            "runtime": {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "fps_tolerance": 0.01},
            "models": {"fake": {
                "result_root": str(model_root), "geometry": "full_frame_resize_to_gt", "temporal": "exact"
            }},
            "metrics": {
                "enabled": ["psnr"], "mask_threshold": 127,
                "max_glyph_outside_bbox_pixels": 0,
                "region_matrix": {"psnr": ["overall", "bbox", "subtitle", "bbox_bg", "outside"]}
            },
            "records": {"write_frames_parquet": True}
        }), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, *extra):
        with redirect_stdout(StringIO()):
            return run_main([
                "--config", str(self.config), "--model", "fake", "--gpus", "cpu", *extra
            ])

    def test_run_resume_and_annotation_invalidation(self):
        self.assertEqual(self._run(), 0)
        result_path = (
            self.output_root / "records" / "fake" / "AI_gen" / "monolingual"
            / self.video_id / "result.json"
        )
        record = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["completeness"], "complete")
        self.assertEqual(
            record["fps_validation"]["prediction_vs_target"]["delta_fraction"], "0"
        )
        self.assertEqual(record["fps_validation"]["mask_vs_target"]["delta_fraction"], "0")
        self.assertTrue(record["regions"]["outside"]["metrics"]["infinite_psnr"])
        import pandas as pd
        frame_rows = pd.read_parquet(result_path.parent / "frames.parquet")
        self.assertEqual(len(frame_rows), 2 * 5)
        self.assertEqual(set(frame_rows["region"]), {"overall", "bbox", "subtitle", "bbox_bg", "outside"})
        first_mtime = result_path.stat().st_mtime_ns
        self.assertEqual(self._run(), 0)
        self.assertEqual(result_path.stat().st_mtime_ns, first_mtime)
        time.sleep(0.01)
        os.utime(self.bbox_path, None)
        self.assertEqual(self._run(), 0)
        self.assertGreater(result_path.stat().st_mtime_ns, first_mtime)

    def test_core_inventory_requires_a_current_evaluation_record(self):
        self.assertEqual(self._run(), 0)
        config = load_config(self.config)
        row = load_manifest_rows(self.manifest)[0]
        sample = sample_from_row(config, row, "fake")
        current = inventory_sample(
            sample, "ffprobe", False, config, self.output_root
        )
        self.assertEqual(current["eligible"], 1)
        self.assertEqual(current["reason"], "current_evaluation_success")

        time.sleep(0.01)
        os.utime(self.bbox_path, None)
        stale = inventory_sample(
            sample, "ffprobe", False, config, self.output_root
        )
        self.assertEqual(stale["eligible"], 0)
        self.assertEqual(stale["reason"], "missing_or_stale_success_evaluation")

    def test_legacy_frame_records_are_projected_without_overwriting_source(self):
        self.assertEqual(self._run(), 0)
        source_result = (
            self.output_root / "records" / "fake" / "AI_gen" / "monolingual"
            / self.video_id / "result.json"
        )
        source_mtime = source_result.stat().st_mtime_ns
        raw = load_config(self.config)
        raw["evaluation_profiles"] = {"lite": {
            "output_root": str(self.root / "lite"),
            "sampling": {
                "mode": "stratified_fixed_budget",
                "subtitle_active_frames": 1, "global_frames": 0,
                "subtitle_active_pairs": 1, "global_pairs": 0,
            },
            "metrics": {
                "enabled": ["psnr"],
                "region_matrix": {"psnr": ["overall", "bbox", "subtitle", "bbox_bg", "outside"]},
            },
        }}
        profiled = apply_evaluation_profile(raw, "lite")
        row = load_manifest_rows(self.manifest)[0]
        sample = sample_from_row(profiled, row, "fake")
        action, reason = reuse_sample(
            sample, profiled, self.output_root, Path(profiled["output_root"])
        )
        self.assertEqual((action, reason), ("reused", ""))
        self.assertEqual(source_result.stat().st_mtime_ns, source_mtime)
        target = json.loads((
            Path(profiled["output_root"]) / "records" / "fake" / "AI_gen"
            / "monolingual" / self.video_id / "result.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(target["evaluation_profile"], "lite")
        self.assertEqual(target["sampling"]["spatial_frame_count"], 1)
        self.assertEqual(
            target["reuse_provenance"]["mode"],
            "lossless_projection_of_existing_frame_statistics",
        )

    def test_mask_fps_mismatch_is_rejected(self):
        glyph = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        glyph[:, 20:30, 10:25] = 255
        encode_rgb(self.mask_path, glyph, fps=3)
        self.assertEqual(self._run("--strict"), 1)
        result_path = (
            self.output_root / "records" / "fake" / "AI_gen" / "monolingual"
            / self.video_id / "result.json"
        )
        record = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failure")
        self.assertIn("glyph_mask", record["error"])

    def test_multi_gpu_plan_without_starting_workers(self):
        output = StringIO()
        with redirect_stdout(output):
            code = run_main([
                "--config", str(self.config), "--model", "fake", "--gpus", "0,3",
                "--workers-per-gpu", "2", "--dry-run",
            ])
        self.assertEqual(code, 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["world_size"], 4)
        self.assertEqual(sum(plan["tasks_per_worker"].values()), 1)

    def test_rocm_worker_uses_one_visibility_filter(self):
        with (
            patch.dict(os.environ, {
                "ROCR_VISIBLE_DEVICES": "7",
                "HIP_VISIBLE_DEVICES": "7",
                "CUDA_VISIBLE_DEVICES": "7",
            }),
            patch("mfvsr_eval.run.subprocess.Popen") as popen,
            redirect_stdout(StringIO()),
        ):
            popen.return_value.wait.return_value = 0
            code = run_main([
                "--config", str(self.config), "--model", "fake", "--gpus", "2,5",
                "--limit", "0", "--output-root", str(self.root / "launcher-test"),
            ])
        self.assertEqual(code, 0)
        environments = [call.kwargs["env"] for call in popen.call_args_list]
        self.assertEqual([env["ROCR_VISIBLE_DEVICES"] for env in environments], ["2", "5"])
        for environment in environments:
            self.assertNotIn("HIP_VISIBLE_DEVICES", environment)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)


if __name__ == "__main__":
    unittest.main()
