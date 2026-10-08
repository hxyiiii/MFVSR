from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from mfasr_eval.analyze import (
    _score_track_rows,
    analyze,
    is_complete,
    is_evaluable,
    main as analyze_main,
    validate_analysis_records,
)
from mfasr_eval.score import SCORE_VERSION


def successful_record(video_id: str, completeness: str, coverage: float, offset: float = 0.0) -> dict:
    return {
        "status": "success",
        "completeness": completeness,
        "video_id": video_id,
        "spatiotemporal_coverage": coverage,
        "spatial_coverage": 1.0,
        "temporal_mapping": {"temporal_coverage": coverage},
        "regions": {
            "overall": {"metrics": {"psnr": 30.0 + offset}},
            "subtitle": {"metrics": {"psnr": 30.0 + offset, "lpips": 0.2, "dists": 0.2}},
            "bbox_bg": {"metrics": {"lpips": 0.2, "dists": 0.2}},
            "outside": {"metrics": {"psnr": 30.0, "lpips": 0.2, "twe_excess": 0.1, "tc": 0.2}},
            "bbox": {"metrics": {"twe_excess": 0.1, "tc": 0.2}},
        },
        "efficiency": {
            "reported_seconds_per_output_frame": 1.0,
            "reported_peak_gpu_memory_mb": 100.0,
        },
    }


class AnalysisTests(unittest.TestCase):
    def test_evaluable_requires_positive_verified_support(self):
        complete = successful_record("v1", "complete", 1.0)
        self.assertTrue(is_evaluable(complete))
        self.assertTrue(is_complete(complete))
        incomplete_support = successful_record("v2", "complete", 0.5)
        self.assertTrue(is_evaluable(incomplete_support))
        self.assertFalse(is_complete(incomplete_support))
        zero_support = successful_record("v3", "partial", 0.0)
        self.assertFalse(is_evaluable(zero_support))
        missing_support = {"status": "success", "completeness": "partial"}
        self.assertFalse(is_evaluable(missing_support))
        invalid_completeness = successful_record("v4", "unknown", 1.0)
        self.assertFalse(is_evaluable(invalid_completeness))

    def test_final_record_gate_rejects_missing_or_stale_records(self):
        config = {
            "models": {"m1": {}},
            "metrics": {"enabled": []},
            "analysis": {"require_terminal_records": True},
        }
        with self.assertRaisesRegex(ValueError, "missing record"):
            validate_analysis_records(config, ["m1"], {"m1": {"v1": None}})
        config["analysis"] = {"require_current_records": True}
        stale = {
            "status": "failure",
            "protocol": "old",
            "evaluator_version": "old",
        }
        with self.assertRaisesRegex(ValueError, "stale protocol"):
            validate_analysis_records(config, ["m1"], {"m1": {"v1": stale}})

    def test_missing_sr_component_withholds_summary_means(self):
        record = successful_record("v1", "complete", 1.0)
        del record["regions"]["outside"]["metrics"]["lpips"]
        anchors = {}
        for key in (
            "subtitle.psnr", "subtitle.lpips", "subtitle.dists",
            "outside.psnr", "outside.lpips",
            "bbox.twe_excess", "bbox.tc", "outside.twe_excess", "outside.tc",
        ):
            anchors[key] = {"p05": 0.0, "p95": 40.0 if key.endswith("psnr") else 1.0}
        rows = _score_track_rows(
            "common_evaluable",
            "m1",
            {"v1"},
            {"v1": record},
            {"v1": {}},
            [],
            {"score_version": SCORE_VERSION, "anchors": anchors},
            10,
        )
        self.assertEqual(rows[0]["missing_success_score_count"], 1)
        self.assertIsNone(rows[0]["sr_q_mean"])
        self.assertIsNone(rows[0]["restoration_mean"])

    def test_shared_missing_sr_component_reports_supported_subset(self):
        valid = successful_record("v1", "complete", 1.0)
        unsupported = successful_record("v2", "complete", 1.0)
        del unsupported["regions"]["bbox"]["metrics"]["twe_excess"]
        anchors = {}
        for key in (
            "subtitle.psnr", "subtitle.lpips", "subtitle.dists",
            "outside.psnr", "outside.lpips",
            "bbox.twe_excess", "bbox.tc", "outside.twe_excess", "outside.tc",
        ):
            anchors[key] = {"p05": 0.0, "p95": 40.0 if key.endswith("psnr") else 1.0}
        rows = _score_track_rows(
            "common_evaluable",
            "m1",
            {"v1", "v2"},
            {"v1": valid, "v2": unsupported},
            {"v1": {}, "v2": {}},
            [],
            {"score_version": SCORE_VERSION, "anchors": anchors},
            10,
        )
        self.assertEqual(rows[0]["sample_count"], 2)
        self.assertEqual(rows[0]["missing_success_score_count"], 1)
        self.assertEqual(rows[0]["sr_q_n"], 1)
        self.assertIsNotNone(rows[0]["sr_q_mean"])
        self.assertEqual(rows[0]["restoration_n"], 1)

    def test_six_model_intersection_excludes_any_single_model_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"samples": [
                {"video_id": "v1", "split": "s"},
                {"video_id": "v2", "split": "s"},
            ]}), encoding="utf-8")
            models = [f"m{index}" for index in range(6)]
            for model in models:
                for video_id in ("v1", "v2"):
                    path = root / "records" / model / "s" / video_id / "result.json"
                    path.parent.mkdir(parents=True)
                    record = successful_record(video_id, "complete", 1.0)
                    if model == "m5" and video_id == "v2":
                        record = {"status": "failure", "failure_category": "inference_failure"}
                    path.write_text(json.dumps(record), encoding="utf-8")
            result = analyze(
                {"output_root": str(root), "analysis": {"groupings": [], "interactions": []}},
                metadata,
                models,
                None,
            )
            self.assertEqual(result["common_evaluable_ids"], ["v1"])
            self.assertEqual(result["native_counts"]["m5"], 1)
            self.assertTrue(all(result["native_counts"][model] == 2 for model in models[:-1]))

    def test_common_native_and_full_reliability_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_rows = []
            for index in range(1, 4):
                metadata_rows.append({
                    "video_id": f"v{index}", "split": "AI_gen/monolingual",
                    "source": "AI-Generated", "resolution_class": "720p-class",
                    "duration_class": "short_le_10s", "difficulty_tier": "Easy",
                })
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"samples": metadata_rows}), encoding="utf-8")
            records_root = root / "records"

            def write(model, video_id, record):
                path = records_root / model / "AI_gen" / "monolingual" / video_id / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(record), encoding="utf-8")

            write("m1", "v1", successful_record("v1", "complete", 1.0))
            write("m1", "v2", successful_record("v2", "complete", 1.0, 1.0))
            write("m1", "v3", {"status": "failure", "failure_category": "inference_failure"})
            write("m2", "v1", successful_record("v1", "complete", 1.0, 2.0))
            write("m2", "v2", successful_record("v2", "partial", 0.5, 3.0))

            anchors = root / "anchors.json"
            anchor_values = {}
            for key in (
                "subtitle.psnr", "subtitle.lpips", "subtitle.dists",
                "outside.psnr", "outside.lpips",
                "bbox.twe_excess", "bbox.tc", "outside.twe_excess", "outside.tc",
            ):
                anchor_values[key] = {"p05": 0.0, "p95": 40.0 if key.endswith("psnr") else 1.0}
            anchors.write_text(json.dumps({
                "score_version": SCORE_VERSION,
                "anchors": anchor_values,
            }), encoding="utf-8")
            config = {
                "output_root": str(root),
                "analysis": {"groupings": ["source"], "interactions": [], "bootstrap_samples": 10},
            }
            result = analyze(config, metadata, ["m1", "m2"], anchors)
            self.assertEqual(result["score_version"], SCORE_VERSION)
            self.assertEqual(result["common_evaluable_count"], 2)
            ballot = result["common_evaluable_ids"]
            self.assertEqual(ballot, ["v1", "v2"])
            self.assertEqual(result["strict_common_complete_count"], 1)
            self.assertEqual(result["native_counts"], {"m1": 2, "m2": 2})
            reliability = {row["model"]: row for row in result["reliability"]}
            self.assertEqual(reliability["m1"]["failure_or_missing"], 1)
            self.assertEqual(reliability["m2"]["evaluable_partial"], 1)
            self.assertEqual(reliability["m2"]["failure_or_missing"], 1)
            summary = {row["model"]: row for row in result["score_summary"]}
            self.assertIsNone(summary["m1"]["effective_sr_all_3000"])
            self.assertIsNone(summary["m2"]["effective_sr_all_3000"])
            self.assertEqual(summary["m2"]["common_evaluable_n"], 2)
            self.assertIsNone(summary["m2"]["common_evaluable_effective_sr_mean"])
            profile = next(
                row for row in result["scores_by_dimension"]
                if row["track"] == "common_evaluable"
                and row["model"] == "m1"
                and row["grouping"] == "all"
            )
            self.assertIsNotNone(profile["restoration_mean"])
            self.assertIsNotNone(profile["preservation_mean"])
            self.assertIsNotNone(profile["temporal_mean"])
            flattened = next(
                row for row in result["per_sample_metrics"]
                if row["model"] == "m1" and row["video_id"] == "v1"
            )
            self.assertEqual(flattened["subtitle.psnr"], 30.0)
            self.assertEqual(flattened["bbox_bg.dists"], 0.2)
            tracks = {row["track"] for row in result["quality"]}
            self.assertEqual(tracks, {
                "common_evaluable", "model_native", "strict_common_complete"
            })
            common_coverage = next(
                row for row in result["coverage"]
                if row["track"] == "common_evaluable"
                and row["model"] == "m2"
                and row["grouping"] == "all"
            )
            self.assertEqual(common_coverage["sample_count"], 2)
            self.assertEqual(common_coverage["partial_count"], 1)
            self.assertAlmostEqual(common_coverage["spatiotemporal_coverage_mean"], 0.75)
            source_distribution = next(
                row for row in result["dataset_dimensions"]
                if row["grouping"] == "source" and row["group"] == "AI-Generated"
            )
            self.assertEqual(source_distribution["sample_count"], 3)
            source_retention = next(
                row for row in result["retention"]
                if row["grouping"] == "source" and row["group"] == "AI-Generated"
            )
            self.assertEqual(source_retention["common_evaluable_count"], 2)
            self.assertEqual(source_retention["strict_common_complete_count"], 1)

            analysis_dir = root / "analysis"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "output_root": str(root),
                "models": {"m1": {}, "m2": {}},
                "analysis": {
                    "metadata_path": str(metadata),
                    "output_dir": str(analysis_dir),
                    "groupings": ["source"],
                    "interactions": [],
                    "bootstrap_samples": 10,
                },
            }), encoding="utf-8")
            with redirect_stdout(StringIO()):
                self.assertEqual(analyze_main([
                    "--config", str(config_path), "--anchors", str(anchors)
                ]), 0)
            for filename in (
                "dataset_dimensions.csv", "per_sample_metrics.csv",
                "quality_common_and_native.csv",
                "coverage_common_and_native.csv", "reliability.csv",
                "reliability_by_dimension.csv", "intersection_retention.csv",
                "efficiency_native_logs.csv", "comparison_sets.json",
                "sr_per_sample.csv", "sr_summary.csv", "sr_by_dimension.csv",
                "analysis_summary.json",
            ):
                self.assertTrue((analysis_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
