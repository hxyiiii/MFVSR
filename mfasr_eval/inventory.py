from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .atomic import atomic_csv, atomic_json
from .cli import parse_models
from .config import load_config
from .evaluator import _fingerprints, evaluation_signature, is_current_record, record_paths
from .io import iter_samples, load_manifest_rows, probe_video
from .profiles import apply_evaluation_profile


BOX_STYLE_PATTERN = re.compile(r"(?:^|[-_])(subtitle[_-]?box|boxed)(?:[-_]|$)", re.I)


def is_box_style(video_id: str) -> bool:
    return BOX_STYLE_PATTERN.search(video_id) is not None


def inventory_sample(
    sample, ffprobe: str, deep: bool,
    config: dict | None = None,
    evaluation_root: Path | None = None,
) -> dict:
    row = {
        "model": sample.model, "video_id": sample.video_id, "split": sample.split,
        "box_style": int(is_box_style(sample.video_id)),
        "eligible": 0, "reason": "", "info_path": str(sample.info_path),
        "prediction_path": str(sample.prediction_path), "prediction_frames": None,
        "evaluation_record": None, "evaluation_current": None,
    }
    if not sample.info_path.is_file():
        row["reason"] = "missing_info"
        return row
    try:
        info = json.loads(sample.info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        row["reason"] = f"malformed_info:{type(exc).__name__}"
        return row
    if info.get("status") != "success":
        row["reason"] = f"inference_{info.get('status', 'unknown')}"
        return row
    if not sample.prediction_path.is_file():
        row["reason"] = "missing_prediction"
        return row
    if deep:
        try:
            probe = probe_video(sample.prediction_path, ffprobe)
            row["prediction_frames"] = probe.frame_count
            if probe.frame_count < 1:
                row["reason"] = "empty_prediction"
                return row
        except Exception as exc:
            row["reason"] = f"prediction_probe_failed:{type(exc).__name__}"
            return row
    if evaluation_root is not None:
        assert config is not None
        path = record_paths(evaluation_root, sample)["result"]
        row["evaluation_record"] = str(path)
        enabled = set(config["metrics"]["enabled"])
        try:
            fingerprints = _fingerprints(sample, False)
            signature = evaluation_signature(sample, config, enabled)
            current = is_current_record(
                path, signature, fingerprints, config, enabled
            )
        except Exception as exc:
            row["reason"] = f"evaluation_check_failed:{type(exc).__name__}"
            row["evaluation_current"] = 0
            return row
        row["evaluation_current"] = int(current)
        if not current:
            row["reason"] = "missing_or_stale_success_evaluation"
            return row
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("completeness") not in {"complete", "partial"}:
            row["reason"] = "invalid_evaluation_completeness"
            return row
        if float(record.get("spatiotemporal_coverage") or 0.0) <= 0:
            row["reason"] = "zero_evaluation_support"
            return row
        row["eligible"] = 1
        row["reason"] = "current_evaluation_success"
        return row
    row["eligible"] = 1
    row["reason"] = "success_artifact_ready"
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory immutable inference outputs for core-set selection")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--require-evaluation", action="store_true")
    parser.add_argument("--evaluation-root")
    parser.add_argument("--deep", action="store_true", help="ffprobe every prediction")
    args = parser.parse_args(argv)
    config = apply_evaluation_profile(load_config(args.config), args.profile)
    models = parse_models(args.model, config["models"])
    rows = load_manifest_rows(args.manifest or config["manifest"])
    ffprobe = config.get("runtime", {}).get("ffprobe", "ffprobe")
    if args.evaluation_root and not args.require_evaluation:
        parser.error("--evaluation-root requires --require-evaluation")
    evaluation_root = (
        Path(args.evaluation_root or config["output_root"])
        if args.require_evaluation else None
    )
    report = [
        inventory_sample(sample, ffprobe, args.deep, config, evaluation_root)
        for sample in iter_samples(config, rows, models)
    ]
    output = Path(args.output)
    atomic_csv(output, report)
    counts = {
        model: {
            "total": sum(row["model"] == model for row in report),
            "eligible": sum(row["model"] == model and row["eligible"] for row in report),
        }
        for model in models
    }
    atomic_json(output.with_suffix(".json"), {
        "schema_version": "1.0", "deep_probe": args.deep,
        "require_evaluation": args.require_evaluation,
        "evaluation_root": str(evaluation_root) if evaluation_root else None,
        "models": models, "counts": counts, "csv": str(output),
    })
    print(json.dumps({"output": str(output), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
