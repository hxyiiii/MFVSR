from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .atomic import atomic_json
from .config import load_config
from .profiles import apply_evaluation_profile
from .score import SCORE_COMPONENTS, SCORE_VERSION, metric_key, metric_value


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        values = data if isinstance(data, list) else data.get("video_ids")
        if not isinstance(values, list):
            raise ValueError("calibration JSON must be a list or contain video_ids")
        return {str(value) for value in values}
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _support_count(record: dict, region: str, metric: str) -> float | None:
    """Return the sufficient-statistic support for one score metric."""
    stats = (((record.get("regions") or {}).get(region) or {}).get(
        "sufficient_statistics"
    ) or {})
    name = {
        "psnr": "psnr_value_count",
        "lpips": "lpips_count",
        "dists": "dists_count",
        "twe_excess": "twe_count",
        "tc": "tc_count",
    }.get(metric)
    if name is None:
        return None
    value = stats.get(name)
    return float(value) if value is not None else None


def fit(records_root: str | Path, models: list[str], calibration_ids: set[str]) -> dict:
    records_root = Path(records_root)
    if not models or not calibration_ids:
        raise ValueError("models and calibration_ids must both be non-empty")
    required = {
        (region, metric, direction)
        for components in SCORE_COMPONENTS.values()
        for region, metric, direction in components
    }
    values = {metric_key(region, metric): [] for region, metric, _ in required}
    records = []
    incomplete = []
    for model in models:
        model_records = {}
        for path in records_root.glob(f"{model}/**/result.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            video_id = str(record.get("video_id"))
            if video_id not in calibration_ids:
                continue
            if video_id in model_records:
                raise ValueError(f"duplicate calibration record for {model}/{video_id}")
            model_records[video_id] = record
        missing = sorted(calibration_ids - set(model_records))
        non_success = sorted(
            video_id for video_id, record in model_records.items()
            if record.get("status") != "success"
        )
        if missing or non_success:
            incomplete.append(
                f"{model}: missing={len(missing)}, non_success={len(non_success)}"
            )
        records.extend(
            (model, video_id, model_records[video_id])
            for video_id in sorted(model_records)
            if model_records[video_id].get("status") == "success"
        )
    if incomplete:
        raise ValueError(
            "incomplete calibration record set; refusing to freeze anchors ("
            + "; ".join(incomplete) + ")"
        )

    missing_metrics: dict[tuple[str, str], list[tuple[str, float | None]]] = {}
    for model, video_id, record in records:
        for region, metric, _ in required:
            value = metric_value(record, region, metric)
            if value is None or (not math.isfinite(value) and metric != "psnr"):
                key = metric_key(region, metric)
                missing_metrics.setdefault((video_id, key), []).append(
                    (model, _support_count(record, region, metric))
                )
                continue
            # Exact-match PSNR is +inf and maps to quality 1 after finite
            # anchors are frozen; it must not make P95 itself infinite.
            if math.isfinite(value):
                values[metric_key(region, metric)].append(value)
    structural_missing = []
    invalid_missing = []
    expected_models = set(models)
    for (video_id, key), cells in sorted(missing_metrics.items()):
        missing_models = {model for model, _ in cells}
        supports = [support for _, support in cells]
        if missing_models == expected_models and all(support == 0.0 for support in supports):
            structural_missing.append({
                "video_id": video_id,
                "metric": key,
                "reason": "zero_support_for_all_models",
                "models": sorted(missing_models),
            })
        else:
            invalid_missing.extend(
                f"{model}/{video_id}:{key}[support={support}]"
                for model, support in cells
            )
    if invalid_missing:
        preview = ", ".join(invalid_missing[:5])
        suffix = "" if len(invalid_missing) <= 5 else f", ... (+{len(invalid_missing) - 5})"
        raise ValueError(
            "missing or non-finite calibration metrics; refusing to freeze anchors: "
            + preview + suffix
        )
    anchors = {}
    for key, metric_values in sorted(values.items()):
        if len(metric_values) < 2:
            raise ValueError(f"insufficient calibration values for {key}: {len(metric_values)}")
        low = percentile(metric_values, 5)
        high = percentile(metric_values, 95)
        if not high > low:
            raise ValueError(f"degenerate calibration anchors for {key}: {low}, {high}")
        anchors[key] = {"p05": low, "p95": high, "count": len(metric_values)}
    return {
        "schema_version": "1.2",
        "score_version": SCORE_VERSION,
        "method": "pooled_model_outputs_on_frozen_human_disjoint_audit300",
        "percentiles": [5, 95],
        "models": models,
        "calibration_video_count": len(calibration_ids),
        "calibration_record_count": len(records),
        "expected_calibration_record_count": len(models) * len(calibration_ids),
        "complete_model_video_grid": True,
        "structurally_undefined_metric_cells": structural_missing,
        "structurally_undefined_video_count": len({
            row["video_id"] for row in structural_missing
        }),
        "score_eligible_video_count": len(calibration_ids) - len({
            row["video_id"] for row in structural_missing
        }),
        "anchors": anchors,
        "dimension_weights": {"restoration": 1.0, "preservation": 1.0, "temporal": 1.0},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze SR-Score normalization anchors")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default="audit300")
    parser.add_argument("--calibration-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", action="append")
    args = parser.parse_args(argv)
    config = apply_evaluation_profile(load_config(args.config), args.profile)
    models = args.model or list(config["models"])
    result = fit(
        Path(config["output_root"]) / "records",
        models,
        load_ids(args.calibration_ids),
    )
    atomic_json(args.output, result)
    print(json.dumps({"output": args.output, "anchors": len(result["anchors"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
