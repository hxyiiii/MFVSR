from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import COMPATIBLE_EVALUATOR_VERSIONS, PROTOCOL_VERSION
from .atomic import atomic_csv, atomic_json
from .cli import parse_models
from .config import load_config
from .score import score_record
from .record_compatibility import runtime_record_is_compatible
from .signature import evaluation_signature_for_model
from .profiles import apply_evaluation_profile
from .selection_io import load_video_ids
from .inventory import is_box_style


DATA_DIMENSIONS = {"source", "resolution_class", "duration_class", "difficulty_tier"}
SUBTITLE_FACTORS = {"spatial_layout", "temporal_behavior", "visual_style", "language_mode"}
SCORE_PROFILE_FIELDS = ("restoration", "preservation", "temporal")


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(path: str | Path) -> dict[str, dict]:
    data = _read_json(Path(path))
    rows = data if isinstance(data, list) else data.get("samples")
    if not isinstance(rows, list):
        raise ValueError("metadata must be a list or contain samples")
    output = {}
    for row in rows:
        video_id = str(row["video_id"])
        if video_id in output:
            raise ValueError(f"duplicate metadata video_id: {video_id}")
        output[video_id] = row
    return output


def load_records(
    records_root: str | Path,
    metadata: dict[str, dict],
    models: list[str],
) -> dict[str, dict[str, dict | None]]:
    records_root = Path(records_root)
    output = {model: {} for model in models}
    for model in models:
        for video_id, meta in metadata.items():
            path = records_root / model / meta["split"] / video_id / "result.json"
            if path.is_file():
                try:
                    output[model][video_id] = _read_json(path)
                except (OSError, json.JSONDecodeError):
                    output[model][video_id] = {
                        "status": "failure", "failure_category": "malformed_record"
                    }
            else:
                output[model][video_id] = None
    return output


def is_evaluable(record: dict | None) -> bool:
    if not record or record.get("status") != "success":
        return False
    if record.get("completeness") not in {"complete", "partial"}:
        return False
    try:
        coverage = (
            float(record["spatial_coverage"]),
            float(record["temporal_mapping"]["temporal_coverage"]),
            float(record["spatiotemporal_coverage"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) and 0.0 < value <= 1.0 for value in coverage)


def _input_fingerprints_are_current(record: dict) -> tuple[bool, str | None]:
    inputs = record.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        return False, "missing input fingerprints"
    for name, fingerprint in inputs.items():
        if not isinstance(fingerprint, dict) or not fingerprint.get("path"):
            return False, f"invalid {name} fingerprint"
        path = Path(fingerprint["path"])
        try:
            stat = path.stat()
        except OSError as exc:
            return False, f"cannot stat {name}: {exc}"
        if stat.st_size != fingerprint.get("size_bytes"):
            return False, f"{name} size changed"
        if stat.st_mtime_ns != fingerprint.get("mtime_ns"):
            return False, f"{name} mtime changed"
        expected_hash = fingerprint.get("sha256")
        if expected_hash is not None and _sha256_file(path) != expected_hash:
            return False, f"{name} SHA-256 changed"
    return True, None


def validate_analysis_records(
    config: dict,
    models: list[str],
    records: dict[str, dict[str, dict | None]],
) -> dict:
    analysis_cfg = config.get("analysis", {})
    require_terminal = bool(analysis_cfg.get("require_terminal_records", False))
    require_current = bool(analysis_cfg.get("require_current_records", False))
    issues = []
    checked = 0
    enabled = set(config.get("metrics", {}).get("enabled", []))
    expected_signatures = (
        {model: evaluation_signature_for_model(model, config, enabled) for model in models}
        if require_current else {}
    )
    for model in models:
        for video_id, record in records[model].items():
            checked += 1
            if record is None:
                if require_terminal:
                    issues.append(f"{model}/{video_id}: missing record")
                continue
            if record.get("failure_category") == "malformed_record":
                if require_terminal or require_current:
                    issues.append(f"{model}/{video_id}: malformed record")
                continue
            if not require_current:
                continue
            if record.get("protocol") != PROTOCOL_VERSION:
                issues.append(f"{model}/{video_id}: stale protocol")
                continue
            if record.get("evaluator_version") not in COMPATIBLE_EVALUATOR_VERSIONS:
                issues.append(f"{model}/{video_id}: stale evaluator version")
                continue
            status = record.get("status")
            if status == "success":
                if record.get("evaluation_signature") != expected_signatures[model]:
                    issues.append(f"{model}/{video_id}: stale evaluation signature")
                    continue
                if not runtime_record_is_compatible(record, config, enabled):
                    issues.append(f"{model}/{video_id}: stale runtime metric policy")
                    continue
                if record.get("completeness") not in {"complete", "partial"}:
                    issues.append(f"{model}/{video_id}: invalid completeness")
                    continue
                current, reason = _input_fingerprints_are_current(record)
                if not current:
                    issues.append(f"{model}/{video_id}: {reason}")
            elif status != "failure":
                issues.append(f"{model}/{video_id}: invalid terminal status {status!r}")
    if issues:
        examples = "; ".join(issues[:8])
        raise ValueError(
            f"analysis record readiness check failed for {len(issues)} of {checked} "
            f"model-sample records; examples: {examples}. "
            "Run mfvsr_eval.run for the affected model(s) and repeat analysis."
        )
    return {
        "require_terminal_records": require_terminal,
        "require_current_records": require_current,
        "checked_model_sample_records": checked,
        "issues": 0,
    }


def is_complete(record: dict | None) -> bool:
    if not is_evaluable(record) or record.get("completeness") != "complete":
        return False
    coverage = (
        float(record["spatial_coverage"]),
        float(record["temporal_mapping"]["temporal_coverage"]),
        float(record["spatiotemporal_coverage"]),
    )
    return all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12) for value in coverage)


def _stats(values: list[float], bootstrap_samples: int, seed_text: str) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    if len(values) == 1 or bootstrap_samples <= 0:
        low = high = mean
    else:
        seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        boot = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(bootstrap_samples)]
        boot.sort()
        low = boot[math.floor(0.025 * (len(boot) - 1))]
        high = boot[math.ceil(0.975 * (len(boot) - 1))]
    return {"n": len(values), "mean": mean, "std": std, "ci95_low": low, "ci95_high": high}


def _prefixed_stats(prefix: str, values: list[float], bootstrap_samples: int, seed: str) -> dict:
    return {
        f"{prefix}_{name}": value
        for name, value in _stats(values, bootstrap_samples, seed).items()
    }


def _group_value(meta: dict, grouping: str) -> str:
    names = grouping.split("+")
    return " × ".join(str(meta[name]) for name in names)


def _grouped_ids(
    video_ids: set[str],
    metadata: dict[str, dict],
    groupings: list[str],
):
    yield "all", "all", set(video_ids)
    for grouping in groupings:
        groups = defaultdict(set)
        for video_id in sorted(video_ids):
            groups[_group_value(metadata[video_id], grouping)].add(video_id)
        for group, ids in sorted(groups.items()):
            yield grouping, group, ids


def _axis_family(grouping: str) -> str:
    if grouping == "all":
        return "overall"
    if grouping in DATA_DIMENSIONS:
        return "data_dimension"
    if grouping in SUBTITLE_FACTORS:
        return "subtitle_factor"
    return "interaction"


def _dataset_dimension_rows(
    metadata: dict[str, dict],
    groupings: list[str],
) -> list[dict]:
    all_ids = set(metadata)
    rows = []
    for grouping, group, ids in _grouped_ids(all_ids, metadata, groupings):
        rows.append({
            "axis_family": _axis_family(grouping),
            "grouping": grouping,
            "group": group,
            "sample_count": len(ids),
            "sample_fraction": len(ids) / len(all_ids) if all_ids else None,
        })
    return rows


def _quality_rows(
    track: str,
    model: str,
    video_ids: set[str],
    records: dict[str, dict | None],
    metadata: dict[str, dict],
    groupings: list[str],
    bootstrap_samples: int,
) -> list[dict]:
    output = []
    for grouping, group, ids in _grouped_ids(video_ids, metadata, groupings):
        group_records = [records[video_id] for video_id in sorted(ids)]
        metrics = defaultdict(list)
        exact_psnr = Counter()
        for record in group_records:
            for region, payload in (record.get("regions") or {}).items():
                region_metrics = payload.get("metrics") or {}
                if region_metrics.get("infinite_psnr") is True:
                    exact_psnr[(region, "psnr")] += 1
                    metrics[(region, "psnr")]
                for metric, value in region_metrics.items():
                    if value is not None and not isinstance(value, bool):
                        metrics[(region, metric)].append(float(value))
        for (region, metric), values in sorted(metrics.items()):
            if track.startswith("box_stress") and region not in {"overall", "subtitle"}:
                continue
            if track.startswith("box_stress") and metric in {"twe", "twe_excess", "tc"}:
                continue
            summary = _stats(
                values,
                bootstrap_samples,
                f"quality|{track}|{model}|{grouping}|{group}|{region}|{metric}",
            )
            available = summary["n"] + exact_psnr[(region, metric)]
            output.append({
                "track": track,
                "model": model,
                "axis_family": _axis_family(grouping),
                "grouping": grouping,
                "group": group,
                "track_sample_count": len(ids),
                "region": region,
                "metric": metric,
                "available_video_count": available,
                "metric_availability_rate": available / len(ids) if ids else None,
                "exact_match_count": exact_psnr[(region, metric)],
                **summary,
            })
    return output


def _per_sample_metric_rows(
    metadata: dict[str, dict],
    models: list[str],
    records: dict[str, dict[str, dict | None]],
    records_root: str | Path,
) -> list[dict]:
    """Flatten auditable per-video results without replacing the raw records."""
    output = []
    for model in models:
        for video_id in sorted(metadata):
            meta = metadata[video_id]
            record = records[model][video_id]
            fps_validation = (record or {}).get("fps_validation") or {}
            prediction_fps = fps_validation.get("prediction_vs_target") or {}
            mask_fps = fps_validation.get("mask_vs_target") or {}
            row = {
                "model": model,
                "video_id": video_id,
                "split": meta.get("split"),
                "source": meta.get("source"),
                "resolution_class": meta.get("resolution_class"),
                "duration_class": meta.get("duration_class"),
                "difficulty_tier": meta.get("difficulty_tier"),
                "spatial_layout": meta.get("spatial_layout"),
                "temporal_behavior": meta.get("temporal_behavior"),
                "visual_style": meta.get("visual_style"),
                "language_mode": meta.get("language_mode"),
                "status": record.get("status") if record else "missing_record",
                "completeness": record.get("completeness") if record else None,
                "failure_category": record.get("failure_category") if record else "missing_record",
                "evaluator_version": record.get("evaluator_version") if record else None,
                "evaluation_signature": record.get("evaluation_signature") if record else None,
                "prediction_fps": prediction_fps.get("candidate_fps"),
                "target_fps": prediction_fps.get("target_fps"),
                "mask_fps": mask_fps.get("candidate_fps"),
                "prediction_target_fps_delta": prediction_fps.get("delta_fps"),
                "mask_target_fps_delta": mask_fps.get("delta_fps"),
                "fps_tolerance": prediction_fps.get("tolerance_fps"),
                "spatial_coverage": record.get("spatial_coverage") if record else None,
                "temporal_coverage": (
                    (record.get("temporal_mapping") or {}).get("temporal_coverage")
                    if record else None
                ),
                "spatiotemporal_coverage": record.get("spatiotemporal_coverage") if record else None,
                "record_path": str(
                    Path(records_root) / model / str(meta.get("split")) / video_id / "result.json"
                ),
                "frame_record_path": record.get("frame_record_path") if record else None,
            }
            if record:
                for input_name, payload in (record.get("inputs") or {}).items():
                    row[f"input.{input_name}.path"] = (payload or {}).get("path")
                for region, payload in (record.get("regions") or {}).items():
                    row[f"{region}.effective_pixels"] = payload.get("effective_pixels")
                    row[f"{region}.target_pixels"] = payload.get("target_pixels")
                    row[f"{region}.coverage"] = payload.get("coverage")
                    for metric, value in (payload.get("metrics") or {}).items():
                        row[f"{region}.{metric}"] = value
                    sufficient = payload.get("sufficient_statistics") or {}
                    for diagnostic in (
                        "dists_count", "dists_fallback_count", "dists_nonfinite_count"
                    ):
                        row[f"{region}.{diagnostic}"] = sufficient.get(diagnostic)
            output.append(row)
    return output


def _coverage_rows(
    track: str,
    model: str,
    video_ids: set[str],
    records: dict[str, dict | None],
    metadata: dict[str, dict],
    groupings: list[str],
    bootstrap_samples: int,
) -> list[dict]:
    output = []
    for grouping, group, ids in _grouped_ids(video_ids, metadata, groupings):
        group_records = [records[video_id] for video_id in sorted(ids)]
        spatial = [float(record["spatial_coverage"]) for record in group_records]
        temporal = [
            float(record["temporal_mapping"]["temporal_coverage"]) for record in group_records
        ]
        joint = [float(record["spatiotemporal_coverage"]) for record in group_records]
        complete = sum(record.get("completeness") == "complete" for record in group_records)
        output.append({
            "track": track,
            "model": model,
            "axis_family": _axis_family(grouping),
            "grouping": grouping,
            "group": group,
            "sample_count": len(ids),
            "complete_count": complete,
            "partial_count": len(ids) - complete,
            "partial_rate": (len(ids) - complete) / len(ids) if ids else None,
            **_prefixed_stats(
                "spatial_coverage", spatial, bootstrap_samples,
                f"coverage|{track}|{model}|{grouping}|{group}|spatial",
            ),
            **_prefixed_stats(
                "temporal_coverage", temporal, bootstrap_samples,
                f"coverage|{track}|{model}|{grouping}|{group}|temporal",
            ),
            **_prefixed_stats(
                "spatiotemporal_coverage", joint, bootstrap_samples,
                f"coverage|{track}|{model}|{grouping}|{group}|joint",
            ),
        })
    return output


def _reliability_row(
    model: str,
    video_ids: set[str],
    records: dict[str, dict | None],
    grouping: str,
    group: str,
) -> dict:
    category = Counter()
    temporal_coverage = []
    spatial_coverage = []
    joint_coverage = []
    for video_id in video_ids:
        record = records[video_id]
        if record is None:
            category["missing_record"] += 1
        elif not is_evaluable(record):
            if record.get("status") == "success":
                category["invalid_success_record"] += 1
            else:
                category[record.get("failure_category", "failure")] += 1
        elif not is_complete(record):
            category["evaluable_partial"] += 1
        else:
            category["complete_success"] += 1
        if is_evaluable(record):
            temporal_coverage.append(float(record["temporal_mapping"]["temporal_coverage"]))
            spatial_coverage.append(float(record["spatial_coverage"]))
            joint_coverage.append(float(record["spatiotemporal_coverage"]))
    total = len(video_ids)
    evaluable_total = category["complete_success"] + category["evaluable_partial"]
    return {
        "model": model,
        "axis_family": _axis_family(grouping),
        "grouping": grouping,
        "group": group,
        "total": total,
        "complete_success": category["complete_success"],
        "evaluable_partial": category["evaluable_partial"],
        "evaluable_total": evaluable_total,
        "failure_or_missing": total - evaluable_total,
        "failure_rate": (total - evaluable_total) / total if total else None,
        "incomplete_rate": category["evaluable_partial"] / total if total else None,
        "mean_temporal_coverage": (
            statistics.fmean(temporal_coverage) if temporal_coverage else None
        ),
        "mean_spatial_coverage": statistics.fmean(spatial_coverage) if spatial_coverage else None,
        "mean_spatiotemporal_coverage": (
            statistics.fmean(joint_coverage) if joint_coverage else None
        ),
        "failure_breakdown_json": json.dumps(dict(category), sort_keys=True),
    }


def _score_track_rows(
    track: str,
    model: str,
    video_ids: set[str],
    records: dict[str, dict | None],
    metadata: dict[str, dict],
    groupings: list[str],
    anchors: dict,
    bootstrap_samples: int,
) -> list[dict]:
    output = []
    for grouping, group, ids in _grouped_ids(video_ids, metadata, groupings):
        sr_q = []
        profile_values = {name: [] for name in SCORE_PROFILE_FIELDS}
        evaluable_count = 0
        missing_success = 0
        for video_id in sorted(ids):
            record = records[video_id]
            if is_evaluable(record):
                evaluable_count += 1
                score = score_record(record, anchors)
                if score["sr_q"] is None:
                    missing_success += 1
                else:
                    sr_q.append(float(score["sr_q"]))
                    for name in SCORE_PROFILE_FIELDS:
                        profile_values[name].append(float(score[name]))
        sr_summary = _stats(
            sr_q, bootstrap_samples,
            f"score|{track}|{model}|{grouping}|{group}|sr_q",
        )
        effective_summary = {
            "n": 0, "mean": None, "std": None,
            "ci95_low": None, "ci95_high": None,
        }
        row = {
            "track": track,
            "model": model,
            "axis_family": _axis_family(grouping),
            "grouping": grouping,
            "group": group,
            "sample_count": len(ids),
            "evaluable_count": evaluable_count,
            "missing_success_score_count": missing_success,
            **{f"sr_q_{name}": value for name, value in sr_summary.items()},
            **{f"effective_sr_{name}": value for name, value in effective_summary.items()},
        }
        for name, values in profile_values.items():
            profile_summary = _stats(
                values,
                bootstrap_samples,
                f"score|{track}|{model}|{grouping}|{group}|{name}",
            )
            row.update({
                f"{name}_{key}": value for key, value in profile_summary.items()
            })
        output.append(row)
    return output


def analyze(
    config: dict,
    metadata_path: str | Path,
    models: list[str],
    anchor_path: str | Path | None,
    include_ids: set[str] | None = None,
) -> dict:
    metadata = load_metadata(metadata_path)
    if include_ids is not None:
        missing = include_ids - set(metadata)
        if missing:
            raise ValueError(f"requested IDs missing metadata: {sorted(missing)[:8]}")
        metadata = {video_id: metadata[video_id] for video_id in sorted(include_ids)}
    records = load_records(Path(config["output_root"]) / "records", metadata, models)
    record_audit = validate_analysis_records(config, models, records)
    all_ids = set(metadata)
    box_style_ids = {
        video_id for video_id, row in metadata.items()
        if bool(row.get("box_style", is_box_style(video_id)))
    }
    primary_quality_ids = all_ids - box_style_ids
    complete_sets = {
        model: {video_id for video_id, record in records[model].items() if is_complete(record)}
        for model in models
    }
    evaluable_sets = {
        model: {video_id for video_id, record in records[model].items() if is_evaluable(record)}
        for model in models
    }
    common_evaluable_ids = (
        set.intersection(*(evaluable_sets[model] for model in models)) if models else set()
    ) & primary_quality_ids
    box_stress_common_ids = (
        set.intersection(*(evaluable_sets[model] for model in models)) if models else set()
    ) & box_style_ids
    strict_common_complete_ids = (
        set.intersection(*(complete_sets[model] for model in models)) if models else set()
    ) & primary_quality_ids
    analysis_cfg = config.get("analysis", {})
    groupings = list(analysis_cfg.get("groupings", [])) + list(
        analysis_cfg.get("interactions", [])
    )
    bootstrap_samples = int(analysis_cfg.get("bootstrap_samples", 1000))

    dataset_dimensions = _dataset_dimension_rows(metadata, groupings)
    per_sample_metrics = _per_sample_metric_rows(
        metadata,
        models,
        records,
        Path(config["output_root"]) / "records",
    )
    quality = []
    coverage = []
    tracks = (
        ("common_evaluable", common_evaluable_ids),
        ("model_native", None),
        ("strict_common_complete", strict_common_complete_ids),
        ("box_stress_common", box_stress_common_ids),
    )
    for model in models:
        for track, fixed_ids in tracks:
            ids = (
                evaluable_sets[model] & primary_quality_ids
                if fixed_ids is None else fixed_ids
            )
            quality.extend(_quality_rows(
                track, model, ids, records[model], metadata, groupings, bootstrap_samples
            ))
            coverage.extend(_coverage_rows(
                track, model, ids, records[model], metadata, groupings, bootstrap_samples
            ))

    reliability = [
        _reliability_row(model, all_ids, records[model], "all", "all") for model in models
    ]
    reliability_by_dimension = []
    for model in models:
        for grouping, group, ids in _grouped_ids(all_ids, metadata, groupings):
            reliability_by_dimension.append(
                _reliability_row(model, ids, records[model], grouping, group)
            )

    retention = []
    for grouping, group, ids in _grouped_ids(all_ids, metadata, groupings):
        if grouping == "all":
            continue
        common_count = len(ids & common_evaluable_ids)
        strict_count = len(ids & strict_common_complete_ids)
        retention.append({
            "axis_family": _axis_family(grouping),
            "grouping": grouping,
            "group": group,
            "full_count": len(ids),
            "common_evaluable_count": common_count,
            "common_evaluable_retention_rate": common_count / len(ids) if ids else None,
            "strict_common_complete_count": strict_count,
            "strict_common_complete_retention_rate": strict_count / len(ids) if ids else None,
        })

    efficiency = []
    for model in models:
        rows = [records[model][video_id] for video_id in evaluable_sets[model]]
        seconds = [
            float(row["efficiency"]["reported_seconds_per_output_frame"])
            for row in rows
            if (row.get("efficiency") or {}).get("reported_seconds_per_output_frame") is not None
        ]
        memory = [
            float(row["efficiency"]["reported_peak_gpu_memory_mb"])
            for row in rows
            if (row.get("efficiency") or {}).get("reported_peak_gpu_memory_mb") is not None
        ]
        efficiency.append({
            "model": model,
            "video_count": len(rows),
            "reported_seconds_per_frame_mean": statistics.fmean(seconds) if seconds else None,
            "reported_peak_memory_mb_mean": statistics.fmean(memory) if memory else None,
            "comparable_across_models": False,
        })

    score_rows = []
    scores_by_dimension = []
    score_summary = []
    score_version = None
    common_score_eligible_count = None
    if anchor_path is not None:
        anchors = _read_json(Path(anchor_path))
        score_version = anchors.get("score_version")
        common_score_sets = {
            model: {
                video_id for video_id in common_evaluable_ids
                if score_record(records[model][video_id], anchors)["sr_q"] is not None
            }
            for model in models
        }
        if common_score_sets:
            reference_model = models[0]
            reference = common_score_sets[reference_model]
            common_score_eligible_count = len(reference)
            asymmetric = {
                model: sorted(reference ^ values)[:8]
                for model, values in common_score_sets.items()
                if values != reference
            }
            if asymmetric:
                raise ValueError(
                    "SR-Score support differs across models on the common-evaluable "
                    f"set; refusing a biased comparison: {asymmetric}"
                )
        for model in models:
            for video_id in sorted(all_ids):
                record = records[model][video_id]
                if is_evaluable(record):
                    score = score_record(record, anchors)
                    effective = None
                else:
                    score = {"sr_q": None, "missing_components": []}
                    effective = None
                score_rows.append({
                    "score_version": score_version,
                    "model": model,
                    "video_id": video_id,
                    "status": record.get("status") if record else "missing_record",
                    "completeness": record.get("completeness") if record else None,
                    "spatiotemporal_coverage": (
                        record.get("spatiotemporal_coverage") if is_evaluable(record) else 0.0
                    ),
                    "sr_q": score.get("sr_q"),
                    "effective_sr": effective,
                    **{
                        name: score.get(name)
                        for name in SCORE_PROFILE_FIELDS
                    },
                    "missing_components": ";".join(score.get("missing_components", [])),
                })
            score_tracks = (
                ("common_evaluable", common_evaluable_ids),
                ("model_native", evaluable_sets[model]),
                ("strict_common_complete", strict_common_complete_ids),
            )
            for track, ids in score_tracks:
                scores_by_dimension.extend(_score_track_rows(
                    track, model, ids, records[model], metadata, groupings,
                    anchors, bootstrap_samples,
                ))

        overall_scores = {
            (row["model"], row["track"]): row
            for row in scores_by_dimension
            if row["grouping"] == "all"
        }
        for model in models:
            common = overall_scores[(model, "common_evaluable")]
            native = overall_scores[(model, "model_native")]
            strict = overall_scores[(model, "strict_common_complete")]
            score_summary.append({
                "model": model,
                "effective_sr_all_3000": None,
                "common_evaluable_sr_q_mean": common["sr_q_mean"],
                "common_evaluable_effective_sr_mean": common["effective_sr_mean"],
                "common_evaluable_n": common["sr_q_n"],
                "common_evaluable_sample_count": common["sample_count"],
                "native_sr_q_mean": native["sr_q_mean"],
                "native_effective_sr_mean": native["effective_sr_mean"],
                "native_n": native["sr_q_n"],
                "native_sample_count": native["sample_count"],
                "strict_common_sr_q_mean": strict["sr_q_mean"],
                "strict_common_effective_sr_mean": strict["effective_sr_mean"],
                "strict_common_n": strict["sr_q_n"],
                "strict_common_sample_count": strict["sample_count"],
                "missing_success_scores": native["missing_success_score_count"],
            })

    return {
        "metadata_count": len(metadata),
        "models": models,
        "common_evaluable_count": len(common_evaluable_ids),
        "common_evaluable_ids": sorted(common_evaluable_ids),
        "strict_common_complete_count": len(strict_common_complete_ids),
        "strict_common_complete_ids": sorted(strict_common_complete_ids),
        "box_style_count": len(box_style_ids),
        "box_stress_common_count": len(box_stress_common_ids),
        "native_counts": {model: len(evaluable_sets[model]) for model in models},
        "native_primary_quality_counts": {
            model: len(evaluable_sets[model] & primary_quality_ids) for model in models
        },
        "score_version": score_version,
        "common_score_eligible_count": common_score_eligible_count,
        "record_audit": record_audit,
        "dataset_dimensions": dataset_dimensions,
        "per_sample_metrics": per_sample_metrics,
        "quality": quality,
        "coverage": coverage,
        "reliability": reliability,
        "reliability_by_dimension": reliability_by_dimension,
        "retention": retention,
        "efficiency": efficiency,
        "scores": score_rows,
        "scores_by_dimension": scores_by_dimension,
        "score_summary": score_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build common-evaluable/native/reliability MFVSR tables"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--ids-file")
    parser.add_argument("--metadata")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--anchors")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    config = apply_evaluation_profile(load_config(args.config), args.profile)
    models = parse_models(args.model or ["all"], config["models"])
    metadata_path = args.metadata or config["analysis"]["metadata_path"]
    output_dir = Path(args.output_dir or Path(config["output_root"]) / "analysis")
    result = analyze(
        config, metadata_path, models, args.anchors,
        load_video_ids(args.ids_file) if args.ids_file else None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "dataset_dimensions.csv", result["dataset_dimensions"])
    atomic_csv(output_dir / "per_sample_metrics.csv", result["per_sample_metrics"])
    atomic_csv(output_dir / "quality_common_and_native.csv", result["quality"])
    atomic_csv(output_dir / "coverage_common_and_native.csv", result["coverage"])
    atomic_csv(output_dir / "reliability.csv", result["reliability"])
    atomic_csv(output_dir / "reliability_by_dimension.csv", result["reliability_by_dimension"])
    atomic_csv(output_dir / "intersection_retention.csv", result["retention"])
    atomic_csv(output_dir / "efficiency_native_logs.csv", result["efficiency"])
    atomic_json(output_dir / "comparison_sets.json", {
        "common_evaluable_ids": result["common_evaluable_ids"],
        "strict_common_complete_ids": result["strict_common_complete_ids"],
        "box_style_count": result["box_style_count"],
        "box_stress_common_count": result["box_stress_common_count"],
        "native_counts": result["native_counts"],
        "native_primary_quality_counts": result["native_primary_quality_counts"],
    })
    if args.anchors:
        atomic_csv(output_dir / "sr_per_sample.csv", result["scores"])
        atomic_csv(output_dir / "sr_summary.csv", result["score_summary"])
        atomic_csv(output_dir / "sr_by_dimension.csv", result["scores_by_dimension"])
    excluded = {
        "dataset_dimensions", "per_sample_metrics", "quality", "coverage", "reliability",
        "reliability_by_dimension", "retention", "efficiency", "scores",
        "scores_by_dimension", "common_evaluable_ids", "strict_common_complete_ids",
    }
    atomic_json(output_dir / "analysis_summary.json", {
        key: value for key, value in result.items() if key not in excluded
    })
    print(json.dumps({
        "output_dir": str(output_dir),
        "common_evaluable_count": result["common_evaluable_count"],
        "strict_common_complete_count": result["strict_common_complete_count"],
        "box_style_count": result["box_style_count"],
        "box_stress_common_count": result["box_stress_common_count"],
        "native_counts": result["native_counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
