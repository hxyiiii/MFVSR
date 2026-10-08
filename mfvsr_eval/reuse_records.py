from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

from . import (
    COMPATIBLE_EVALUATOR_VERSIONS,
    EVALUATOR_VERSION,
    EXECUTION_REVISION,
    PROTOCOL_VERSION,
    REGIONS,
    SCHEMA_VERSION,
)
from .atomic import atomic_csv, atomic_json, atomic_parquet
from .cli import parse_models
from .config import load_config, validate_protocol_config
from .evaluator import (
    _fingerprints,
    _required_input_check,
    evaluation_signature,
    is_current_record,
    record_paths,
    sample_lock,
)
from .io import iter_samples, load_manifest_rows
from .metrics import RegionAccumulator
from .profiles import apply_evaluation_profile
from .record_compatibility import runtime_record_is_compatible
from .regions import load_bboxes
from .sampling import build_sampling_plan
from .selection_io import load_video_ids


METRIC_COLUMNS = {
    "psnr": {"psnr", "psnr_exact_match", "sse", "psnr_value_count"},
    "ssim": {"ssim", "ssim_sum", "ssim_count"},
    "lpips": {"lpips", "lpips_sum", "lpips_count", "lpips_lowres_mask_mass"},
    "dists": {
        "dists", "dists_sum", "dists_count", "dists_backend",
        "dists_input_width", "dists_input_height", "dists_input_resized",
        "dists_fallback_count", "dists_gpu_retry_count",
        "dists_nonfinite_count", "dists_null_reason",
    },
    "twe": {
        "twe", "twe_target", "twe_excess", "twe_sum", "twe_count",
        "twe_target_sum", "twe_target_count",
    },
    "tc": {"tc", "tc_sum", "tc_count"},
}
TEMPORAL_METRICS = {"twe", "tc"}


def _clean_value(value):
    try:
        if value is not None and not isinstance(value, (str, bool)) and math.isnan(value):
            return None
    except TypeError:
        pass
    return value


def _read_frame_rows(base: Path, source_record: dict) -> tuple[list[dict], Path]:
    parquet = base / "frames.parquet"
    jsonl = base / "frames.jsonl.gz"
    if parquet.is_file():
        import pandas as pd

        rows = pd.read_parquet(parquet).to_dict(orient="records")
        return [{key: _clean_value(value) for key, value in row.items()} for row in rows], parquet
    if jsonl.is_file():
        with gzip.open(jsonl, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()], jsonl
    declared = Path(str(source_record.get("frame_record_path") or ""))
    raise FileNotFoundError(
        f"source frame record missing (checked {parquet}, {jsonl}, declared {declared})"
    )


def _project_rows(rows: list[dict], config: dict, sampling: dict) -> list[dict]:
    enabled = set(config["metrics"]["enabled"])
    region_matrix = config["metrics"]["region_matrix"]
    spatial = set(sampling["spatial_frame_indices"])
    temporal = set(sampling["temporal_pair_end_indices"])
    columns = set().union(*(row.keys() for row in rows)) if rows else set()
    for metric in enabled:
        required = METRIC_COLUMNS[metric]
        primary = metric
        if primary not in columns:
            raise ValueError(f"source frame record does not contain metric column {primary}")
        if not (required & columns):
            raise ValueError(f"source frame record lacks columns for {metric}")
    output = []
    for source in rows:
        row = dict(source)
        frame_index = int(row["frame_index"])
        region = str(row["region"])
        for metric, names in METRIC_COLUMNS.items():
            keep = (
                metric in enabled
                and region in set(region_matrix.get(metric, []))
                and (
                    frame_index in temporal
                    if metric in TEMPORAL_METRICS else frame_index in spatial
                )
            )
            if not keep:
                for name in names:
                    row.pop(name, None)
        output.append(row)
    return output


def reuse_sample(sample, config: dict, source_root: Path, output_root: Path) -> tuple[str, str]:
    enabled = set(config["metrics"]["enabled"])
    _required_input_check(sample)
    source_paths = record_paths(source_root, sample)
    if not source_paths["result"].is_file():
        return "not_reusable", "missing_source_result"
    source_record = json.loads(source_paths["result"].read_text(encoding="utf-8"))
    if source_record.get("status") != "success":
        return "not_reusable", f"source_status_{source_record.get('status')}"
    if source_record.get("protocol") != PROTOCOL_VERSION:
        return "not_reusable", "source_protocol_mismatch"
    if source_record.get("evaluator_version") not in COMPATIBLE_EVALUATOR_VERSIONS:
        return "not_reusable", "source_evaluator_incompatible"
    if not enabled <= set(source_record.get("requested_metrics") or []):
        return "not_reusable", "source_metrics_incomplete"
    if not runtime_record_is_compatible(source_record, config, enabled):
        return "not_reusable", "source_runtime_policy_incompatible"
    with_hash = any(
        (payload or {}).get("sha256") is not None
        for payload in (source_record.get("inputs") or {}).values()
    )
    fingerprints = _fingerprints(sample, with_hash)
    if source_record.get("inputs") != fingerprints:
        return "not_reusable", "input_fingerprint_mismatch"
    signature = evaluation_signature(sample, config, enabled)
    target_paths = record_paths(output_root, sample)
    if is_current_record(target_paths["result"], signature, fingerprints, config, enabled):
        return "already_current", ""
    target_probe = source_record.get("target_probe") or {}
    frame_count = int(target_probe.get("frame_count") or 0)
    if frame_count < 1:
        return "not_reusable", "source_missing_target_probe"
    sampling = build_sampling_plan(
        frame_count, load_bboxes(sample.bbox_json_path), config.get("sampling")
    )
    rows, source_frame_path = _read_frame_rows(source_paths["base"], source_record)
    if "dists" in enabled:
        if any(float(row.get("dists_fallback_count") or 0) > 0 for row in rows):
            return "not_reusable", "source_dists_used_cpu_fallback"
        if any(float(row.get("dists_nonfinite_count") or 0) > 0 for row in rows):
            return "not_reusable", "source_dists_nonfinite"
    if "twe" in enabled:
        estimator = source_record.get("temporal_estimator") or {}
        width = int(target_probe.get("width") or 0)
        height = int(target_probe.get("height") or 0)
        limit = int(config.get("runtime", {}).get("raft_max_input_long_side", 0) or 0)
        trigger = int(config.get("runtime", {}).get("raft_resize_trigger_long_side", 0) or 0)
        expected = [width, height]
        if limit > 0 and max(width, height) > max(trigger, limit):
            scale = limit / max(width, height)
            expected = [max(1, round(width * scale)), max(1, round(height * scale))]
        if estimator.get("flow_input_size") != expected:
            return "not_reusable", "source_raft_input_policy_mismatch"
    projected = _project_rows(rows, config, sampling)
    accumulator = RegionAccumulator()
    for row in projected:
        accumulator.add(str(row["region"]), row)
    stale_seconds = float(config.get("runtime", {}).get("lock_stale_hours", 2.0)) * 3600.0
    with sample_lock(target_paths["lock"], stale_seconds):
        if is_current_record(target_paths["result"], signature, fingerprints, config, enabled):
            return "already_current", ""
        atomic_parquet(target_paths["frames_parquet"], projected)
        record = deepcopy(source_record)
        record.update({
            "schema_version": SCHEMA_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
            "execution_revision": EXECUTION_REVISION,
            "evaluation_signature": signature,
            "evaluation_profile": config.get("evaluation_profile", "legacy_full"),
            "requested_metrics": sorted(enabled),
            "sampling": sampling,
            "inputs": fingerprints,
            "regions": {region: accumulator.summary(region) for region in REGIONS},
            "temporal_estimator": (
                source_record.get("temporal_estimator") if "twe" in enabled else None
            ),
            "dists_execution": (
                source_record.get("dists_execution") if "dists" in enabled else None
            ),
            "frame_record_path": str(target_paths["frames_parquet"]),
            "frame_record_format": "parquet",
            "evaluation_seconds": 0.0,
            "reuse_provenance": {
                "mode": "lossless_projection_of_existing_frame_statistics",
                "source_result": str(source_paths["result"]),
                "source_frames": str(source_frame_path),
                "source_evaluation_signature": source_record.get("evaluation_signature"),
                "projected_unix": time.time(),
            },
        })
        atomic_json(target_paths["result"], record)
    return "reused", ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project compatible legacy frame records into a lean profile without recomputation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--manifest")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--ids-file")
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    config = apply_evaluation_profile(load_config(args.config), args.profile)
    validate_protocol_config(config)
    models = parse_models(args.model, config["models"])
    rows = load_manifest_rows(args.manifest or config["manifest"])
    allowed = load_video_ids(args.ids_file) if args.ids_file else None
    if allowed is not None:
        rows = [row for row in rows if str(row["video_id"]) in allowed]
    output_root = Path(args.output_root or config["output_root"])
    report = []
    for sample in iter_samples(config, rows, models):
        try:
            action, reason = reuse_sample(
                sample, config, Path(args.source_root), output_root
            )
        except Exception as exc:
            action, reason = "not_reusable", f"{type(exc).__name__}: {exc}"
        report.append({
            "model": sample.model,
            "video_id": sample.video_id,
            "split": sample.split,
            "action": action,
            "reason": reason,
        })
        print(json.dumps(report[-1], ensure_ascii=False), flush=True)
    report_path = Path(args.report or output_root / "summaries" / "reuse_report.csv")
    atomic_csv(report_path, report)
    counts = {}
    for row in report:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
    atomic_json(report_path.with_suffix(".json"), {
        "schema_version": "1.0", "profile": args.profile,
        "source_root": str(args.source_root), "output_root": str(output_root),
        "counts": counts, "report_csv": str(report_path),
    })
    print(json.dumps({"counts": counts, "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
