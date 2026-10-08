from __future__ import annotations

import json
from pathlib import Path

from mfvsr_eval.profiles import apply_evaluation_profile
from mfvsr_eval.sampling import build_sampling_plan, evenly_spaced
from mfvsr_eval.selection_io import load_video_ids
from mfvsr_eval.signature import evaluation_signature_for_model


def test_evenly_spaced_and_sampling_are_deterministic():
    assert evenly_spaced(list(range(10)), 3) == [0, 4, 9]
    bboxes = {index: [{"x": 1}] for index in range(10, 40)}
    config = {
        "mode": "stratified_fixed_budget",
        "subtitle_active_frames": 8, "global_frames": 4,
        "subtitle_active_pairs": 6, "global_pairs": 2,
    }
    first = build_sampling_plan(50, bboxes, config)
    second = build_sampling_plan(50, bboxes, config)
    assert first == second
    assert first["spatial_frame_count"] == 12
    assert first["temporal_pair_count"] == 8
    assert all(0 <= value < 50 for value in first["spatial_frame_indices"])
    assert all(1 <= value < 50 for value in first["temporal_pair_end_indices"])


def test_profile_does_not_mutate_config_and_legacy_signature_is_preserved():
    config = {
        "models": {"m": {"rule": "x"}},
        "metrics": {"enabled": ["psnr"], "region_matrix": {"psnr": ["overall"]}},
        "evaluation_profiles": {"lite": {
            "output_root": "/new", "sampling": {"mode": "stratified_fixed_budget"},
            "metrics": {"enabled": ["lpips"], "region_matrix": {"lpips": ["overall"]}},
        }},
    }
    legacy = evaluation_signature_for_model("m", config, {"psnr"})
    defaulted = apply_evaluation_profile(config, None)
    assert evaluation_signature_for_model("m", defaulted, {"psnr"}) == legacy
    profiled = apply_evaluation_profile(config, "lite")
    assert config["metrics"]["enabled"] == ["psnr"]
    assert profiled["metrics"]["enabled"] == ["lpips"]
    assert evaluation_signature_for_model("m", profiled, {"lpips"}) != legacy


def test_load_video_ids_text_and_csv(tmp_path: Path):
    text = tmp_path / "ids.txt"
    text.write_text("# frozen\na\nb\na\n", encoding="utf-8")
    assert load_video_ids(text) == {"a", "b"}
    csv_path = tmp_path / "ids.csv"
    csv_path.write_text("video_id,extra\nc,1\nd,2\n", encoding="utf-8")
    assert load_video_ids(csv_path) == {"c", "d"}

