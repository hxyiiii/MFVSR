from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mfasr_eval.fit_anchors import percentile
from mfasr_eval.metadata import (
    difficulty,
    duration_class,
    metadata_signature,
    parse_factors,
    resolution_class,
    source_class,
    validate_dimension_coverage,
)
from mfasr_eval.metrics import RegionAccumulator, masked_psnr
from mfasr_eval.regions import load_bboxes, make_regions, rasterize_bboxes
from mfasr_eval.score import SCORE_VERSION, score_record
from mfasr_eval.worker import shard_for


WEIGHTS = {
    "spatial_layout": {"Fixed": 0, "Dynamic": 2},
    "temporal_behavior": {"Static": 0, "Animated": 1},
    "visual_style": {"Plain": 0, "Decorated": 1, "Stylized": 2},
    "language_mode": {"Monolingual": 0, "Bilingual": 1},
    "cutoffs": {"easy_max": 1, "medium_max": 3},
}


class MetadataTests(unittest.TestCase):
    def test_parse_all_factors(self):
        value = "002_Dynamic-circle_Animated-typewriter_Plain-white_Bilingual-Portuguese-Russian"
        self.assertEqual(parse_factors(value), {
            "spatial_layout": "Dynamic",
            "temporal_behavior": "Animated",
            "visual_style": "Plain",
            "language_mode": "Bilingual",
        })
        value = "id_Fixed-bottom_left_Static_Decorated-subtitle_box_Monolingual-Chinese_Simp"
        self.assertEqual(parse_factors(value)["temporal_behavior"], "Static")
        self.assertEqual(parse_factors(value)["visual_style"], "Decorated")

    def test_source_duration_resolution_boundaries(self):
        self.assertEqual(source_class("AI_gen/bilingual"), "AI-Generated")
        self.assertEqual(source_class("origin_clean/1080P_above"), "Real-HQ")
        self.assertEqual(source_class("origin_clean/512P_to_1080P/monolingual"), "Real-LQ")
        self.assertEqual(duration_class(10.0), "short_le_10s")
        self.assertEqual(duration_class(10.0001), "medium_10_to_30s")
        self.assertEqual(duration_class(30.0), "medium_10_to_30s")
        self.assertEqual(duration_class(30.0001), "long_gt_30s")
        self.assertEqual(resolution_class(1360, 768), "720p-class")
        self.assertEqual(resolution_class(1920, 1080), "1080p-class")

    def test_difficulty_examples_have_correct_bins(self):
        easy = {
            "spatial_layout": "Fixed", "temporal_behavior": "Animated",
            "visual_style": "Plain", "language_mode": "Monolingual",
        }
        medium = {
            "spatial_layout": "Dynamic", "temporal_behavior": "Static",
            "visual_style": "Decorated", "language_mode": "Monolingual",
        }
        hard = {
            "spatial_layout": "Dynamic", "temporal_behavior": "Animated",
            "visual_style": "Stylized", "language_mode": "Bilingual",
        }
        self.assertEqual(difficulty(easy, WEIGHTS), (1, "Easy"))
        self.assertEqual(difficulty(medium, WEIGHTS), (3, "Medium"))
        self.assertEqual(difficulty(hard, WEIGHTS), (6, "Hard"))

    def test_dimension_coverage_validation(self):
        samples = [
            {"source": "Real-HQ", "difficulty_tier": "Easy"},
            {"source": "Real-LQ", "difficulty_tier": "Medium"},
            {"source": "AI-Generated", "difficulty_tier": "Hard"},
        ]
        config = {
            "expected_sample_count": 3,
            "expected_categories": {
                "source": ["Real-HQ", "Real-LQ", "AI-Generated"],
                "difficulty_tier": ["Easy", "Medium", "Hard"],
            },
        }
        summary = validate_dimension_coverage(samples, config)
        self.assertEqual(summary["difficulty_tier"], {"Easy": 1, "Hard": 1, "Medium": 1})
        with self.assertRaisesRegex(ValueError, "sample count mismatch"):
            validate_dimension_coverage(samples, {**config, "expected_sample_count": 4})

    def test_metadata_signature_tracks_semantics_not_worker_count(self):
        base = {
            "resolution_short_side_threshold": 900,
            "short_duration_max_seconds": 10,
            "medium_duration_max_seconds": 30,
            "difficulty_weights": WEIGHTS,
            "probe_workers": 8,
            "checkpoint_every": 50,
        }
        operational_change = {**base, "probe_workers": 16, "checkpoint_every": 100}
        semantic_change = {**base, "short_duration_max_seconds": 12}
        self.assertEqual(metadata_signature(base), metadata_signature(operational_change))
        self.assertNotEqual(metadata_signature(base), metadata_signature(semantic_change))


class RegionMetricTests(unittest.TestCase):
    @staticmethod
    def _bbox_file(rows):
        temp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(rows, temp)
        temp.close()
        return Path(temp.name)

    def test_atomic_partition(self):
        bbox = rasterize_bboxes([(1, 1, 5, 5)], 6, 6)
        glyph = np.zeros((6, 6), dtype=bool)
        glyph[2:4, 2:4] = True
        regions = make_regions(6, 6, bbox, glyph)
        self.assertTrue(np.array_equal(regions["bbox"], regions["subtitle"] | regions["bbox_bg"]))
        total = sum(int(regions[name].sum()) for name in ("subtitle", "bbox_bg", "outside"))
        self.assertEqual(total, 36)

    def test_outside_glyph_fails_in_strict_mode(self):
        bbox = np.zeros((4, 4), dtype=bool)
        glyph = np.zeros((4, 4), dtype=bool)
        glyph[0, 0] = True
        with self.assertRaisesRegex(ValueError, "glyph/BBox inconsistency"):
            make_regions(4, 4, bbox, glyph)

    def test_canonical_empty_bbox_is_valid_absent_subtitle(self):
        path = self._bbox_file([
            {"frame": 0, "x": 0, "y": 0, "width": 0, "height": 0}
        ])
        try:
            boxes = load_bboxes(path)
        finally:
            path.unlink()
        self.assertEqual(boxes, {0: []})
        bbox = rasterize_bboxes(boxes[0], 4, 4)
        glyph = np.zeros((4, 4), dtype=bool)
        regions = make_regions(4, 4, bbox, glyph)
        self.assertEqual(int(regions["bbox"].sum()), 0)
        self.assertEqual(int(regions["subtitle"].sum()), 0)
        self.assertEqual(int(regions["bbox_bg"].sum()), 0)
        self.assertTrue(regions["overall"].all())
        self.assertTrue(regions["outside"].all())

    def test_empty_and_nonempty_bbox_cannot_coexist(self):
        path = self._bbox_file([
            {"frame": 0, "x": 0, "y": 0, "width": 0, "height": 0},
            {"frame": 0, "x": 1, "y": 1, "width": 2, "height": 2},
        ])
        try:
            with self.assertRaisesRegex(ValueError, "coexist"):
                load_bboxes(path)
        finally:
            path.unlink()

    def test_partial_zero_bbox_is_invalid(self):
        path = self._bbox_file([
            {"frame": 0, "x": 0, "y": 0, "width": 0, "height": 2}
        ])
        try:
            with self.assertRaisesRegex(ValueError, "invalid non-empty BBox"):
                load_bboxes(path)
        finally:
            path.unlink()

    def test_psnr_and_pooled_summary(self):
        target = np.zeros((2, 2, 3), dtype=np.uint8)
        prediction = np.full((2, 2, 3), 10, dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)
        result = masked_psnr(prediction, target, mask)
        self.assertAlmostEqual(result["value"], 20 * math.log10(255 / 10), places=10)
        accumulator = RegionAccumulator()
        accumulator.add("overall", {
            "sse": result["sse"], "psnr_value_count": result["value_count"],
            "effective_pixels": 4, "target_pixels": 4,
        })
        self.assertAlmostEqual(accumulator.summary("overall")["metrics"]["psnr"], result["value"], places=10)


class ScoreTests(unittest.TestCase):
    def _record(self):
        regions = {
            "subtitle": {"metrics": {"psnr": 30.0, "lpips": 0.2, "dists": 0.2}},
            "bbox_bg": {"metrics": {"lpips": 0.2, "dists": 0.2}},
            "outside": {"metrics": {"psnr": 30.0, "lpips": 0.2, "twe_excess": -0.1, "tc": 0.2}},
            "bbox": {"metrics": {"twe_excess": 0.2, "tc": 0.2}},
        }
        return {"regions": regions, "spatiotemporal_coverage": 0.5}

    def test_score_directions_and_coverage(self):
        anchors = {}
        for region, metric in (
            ("subtitle", "psnr"), ("outside", "psnr"),
            ("subtitle", "lpips"), ("subtitle", "dists"),
            ("outside", "lpips"), ("bbox", "twe_excess"),
            ("bbox", "tc"), ("outside", "twe_excess"), ("outside", "tc"),
        ):
            anchors[f"{region}.{metric}"] = {"p05": 0.0, "p95": 40.0 if metric == "psnr" else 1.0}
        result = score_record(self._record(), {"score_version": SCORE_VERSION, "anchors": anchors})
        self.assertIsNotNone(result["sr_q"])
        self.assertIsNone(result["effective_sr"])
        self.assertEqual(result["coverage"], 0.5)
        self.assertGreater(result["global_temporal"], result["local_temporal"])

    def test_missing_component_is_not_silently_renormalized(self):
        result = score_record(self._record(), {"score_version": SCORE_VERSION, "anchors": {}})
        self.assertIsNone(result["sr_q"])
        self.assertTrue(result["missing_components"])

    def test_exact_match_psnr_maps_to_best_quality(self):
        record = self._record()
        record["regions"]["outside"]["metrics"]["psnr"] = None
        record["regions"]["outside"]["metrics"]["infinite_psnr"] = True
        anchors = {}
        for region, metric in (
            ("subtitle", "psnr"), ("outside", "psnr"),
            ("subtitle", "lpips"), ("subtitle", "dists"),
            ("outside", "lpips"), ("bbox", "twe_excess"),
            ("bbox", "tc"), ("outside", "twe_excess"), ("outside", "tc"),
        ):
            anchors[f"{region}.{metric}"] = {"p05": 0.0, "p95": 40.0 if metric == "psnr" else 1.0}
        result = score_record(record, {"score_version": SCORE_VERSION, "anchors": anchors})
        self.assertEqual(result["details"]["global_preservation"]["outside.psnr"]["quality"], 1.0)

    def test_bbox_non_glyph_diagnostic_does_not_change_sr(self):
        anchors = {}
        for region, metric in (
            ("subtitle", "psnr"), ("outside", "psnr"),
            ("subtitle", "lpips"), ("subtitle", "dists"),
            ("outside", "lpips"), ("bbox", "twe_excess"),
            ("bbox", "tc"), ("outside", "twe_excess"), ("outside", "tc"),
        ):
            anchors[f"{region}.{metric}"] = {
                "p05": 0.0,
                "p95": 40.0 if metric == "psnr" else 1.0,
            }
        anchor_file = {"score_version": SCORE_VERSION, "anchors": anchors}
        first = self._record()
        second = self._record()
        second["regions"]["bbox_bg"]["metrics"] = {}
        first_score = score_record(first, anchor_file)
        second_score = score_record(second, anchor_file)
        self.assertEqual(first_score["sr_q"], second_score["sr_q"])
        self.assertEqual(second_score["missing_components"], [])

    def test_old_anchor_definition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "score_version"):
            score_record(self._record(), {"anchors": {}})


class UtilityTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(percentile([0, 10], 5), 0.5)
        self.assertEqual(percentile([0, 10], 95), 9.5)

    def test_shard_is_deterministic_and_bounded(self):
        self.assertEqual(shard_for("video", 8), shard_for("video", 8))
        self.assertIn(shard_for("video", 8), range(8))


if __name__ == "__main__":
    unittest.main()
