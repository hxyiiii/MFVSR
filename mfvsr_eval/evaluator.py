from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import socket
import sys
import time
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import EXECUTION_REVISION, EVALUATOR_VERSION, PROTOCOL_VERSION, REGIONS, SCHEMA_VERSION
from .atomic import atomic_json, atomic_jsonl_gz, atomic_parquet
from .geometry import make_adapter, validate_temporal_mapping
from .io import probe_video, read_video_frames, sha256_file
from .metrics import (
    DISTSNonFiniteError,
    PerceptualMetrics,
    RegionAccumulator,
    masked_map_mean,
    masked_psnr,
    ssim_scalar_and_map,
    ssim_support,
)
from .regions import glyph_mask, load_bboxes, make_regions, rasterize_bboxes
from .record_compatibility import runtime_record_is_compatible
from .sampling import build_sampling_plan
from .schema import Sample, VideoProbe
from .signature import evaluation_signature_for_model
from .temporal import make_flow_estimator, temporal_change_map, temporal_maps


LOG = logging.getLogger("mfvsr_eval")
ALLOWED_METRICS = {"psnr", "ssim", "lpips", "dists", "twe", "tc"}


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def record_paths(output_root: str | Path, sample: Sample) -> dict[str, Path]:
    base = Path(output_root) / "records" / sample.model / sample.split / sample.video_id
    return {
        "base": base,
        "result": base / "result.json",
        "frames_parquet": base / "frames.parquet",
        "frames_jsonl": base / "frames.jsonl.gz",
        "lock": base / ".evaluation.lock",
    }


def _file_fingerprint(path: Path, with_hash: bool) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path) if with_hash else None,
    }


def _fingerprints(sample: Sample, with_hash: bool) -> dict[str, dict]:
    return {
        "prediction": _file_fingerprint(sample.prediction_path, with_hash),
        "ground_truth": _file_fingerprint(sample.gt_path, with_hash),
        "glyph_mask": _file_fingerprint(sample.glyph_mask_path, with_hash),
        "bbox_json": _file_fingerprint(sample.bbox_json_path, with_hash),
        "inference_info": _file_fingerprint(sample.info_path, with_hash),
    }


def evaluation_signature(sample: Sample, config: dict, enabled: set[str]) -> str:
    return evaluation_signature_for_model(sample.model, config, enabled)


def _fps_fraction(probe: VideoProbe) -> Fraction:
    return Fraction(probe.fps_numerator, probe.fps_denominator)


def _validate_fps_pair(
    candidate_name: str,
    candidate: VideoProbe,
    target: VideoProbe,
    tolerance: float,
) -> dict:
    tolerance_fraction = Fraction(str(tolerance))
    if tolerance_fraction < 0:
        raise ValueError(f"FPS tolerance must be non-negative: {tolerance}")
    candidate_fraction = _fps_fraction(candidate)
    target_fraction = _fps_fraction(target)
    delta = abs(candidate_fraction - target_fraction)
    if delta > tolerance_fraction:
        raise ValueError(
            f"FPS mismatch: {candidate_name}={float(candidate_fraction):.9f} "
            f"({candidate_fraction}), target={float(target_fraction):.9f} "
            f"({target_fraction}), delta={float(delta):.9f}, tolerance={tolerance}"
        )
    return {
        "candidate": candidate_name,
        "candidate_fraction": str(candidate_fraction),
        "target_fraction": str(target_fraction),
        "candidate_fps": float(candidate_fraction),
        "target_fps": float(target_fraction),
        "delta_fraction": str(delta),
        "delta_fps": float(delta),
        "tolerance_fps": float(tolerance_fraction),
        "matched": True,
    }


def is_current_record(
    path: Path,
    signature: str,
    fingerprints: dict,
    config: dict,
    enabled: set[str],
) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        record.get("status") == "success"
        and record.get("evaluation_signature") == signature
        and record.get("inputs") == fingerprints
        and runtime_record_is_compatible(record, config, enabled)
    )


@contextmanager
def sample_lock(path: Path, stale_seconds: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "host": socket.gethostname(), "created_unix": time.time()}
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            break
        except FileExistsError:
            stale = False
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                age = time.time() - float(existing.get("created_unix", 0))
                same_host = existing.get("host") == socket.gethostname()
                pid = int(existing.get("pid", -1))
                pid_alive = False
                if same_host and pid > 0:
                    try:
                        os.kill(pid, 0)
                        pid_alive = True
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        pid_alive = True
                stale = (same_host and not pid_alive) or age > stale_seconds
            except Exception:
                stale = True
            if stale:
                path.unlink(missing_ok=True)
                continue
            raise RuntimeError(f"sample is locked by another evaluator: {path}")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _required_input_check(sample: Sample) -> dict:
    if not sample.info_path.is_file():
        raise FileNotFoundError(f"missing inference info.json: {sample.info_path}")
    try:
        info = json.loads(sample.info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed inference info.json: {sample.info_path}") from exc
    if info.get("status") != "success":
        raise RuntimeError(f"inference status is {info.get('status')!r}: {info.get('error', '')}")
    for label, path in (
        ("prediction", sample.prediction_path),
        ("ground truth", sample.gt_path),
        ("glyph mask", sample.glyph_mask_path),
        ("BBox JSON", sample.bbox_json_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    return info


def _next_frame(iterator, label: str, frame_index: int):
    try:
        return next(iterator)
    except StopIteration as exc:
        raise RuntimeError(f"{label} decode ended at frame {frame_index}") from exc


def _mean_map(score_map: np.ndarray, support: np.ndarray) -> tuple[float | None, int]:
    count = int(support.sum())
    return (float(score_map[support].mean()), count) if count else (None, 0)


def _metric_regions(config: dict, metric: str) -> set[str]:
    return set((config["metrics"].get("region_matrix") or {}).get(metric, []))


def _result_efficiency(info: dict) -> dict:
    performance = info.get("performance") or {}
    environment = info.get("environment") or {}
    output = info.get("output_raw") or {}
    elapsed = performance.get("total_inference_time_seconds")
    frame_count = output.get("num_frames")
    return {
        "reported_total_seconds": elapsed,
        "reported_seconds_per_output_frame": (
            float(elapsed) / int(frame_count) if elapsed is not None and frame_count else None
        ),
        "reported_peak_gpu_memory_mb": performance.get("peak_gpu_memory_MB"),
        "reported_gpu": environment.get("gpu"),
        "reported_software": {
            "python": environment.get("python_version"),
            "pytorch": environment.get("pytorch_version"),
            "hip": environment.get("hip_version"),
        },
        "comparable_across_models": False,
        "reason": "native-run logs may use different hardware/software; use the standardized efficiency subset",
    }


def _coverage_classification(spatial_coverage: float, temporal_coverage: float) -> tuple[str, list[str]]:
    reasons = []
    if spatial_coverage < 1.0:
        reasons.append("partial_spatial_coverage")
    if temporal_coverage < 1.0:
        reasons.append("partial_temporal_coverage")
    return ("partial", reasons) if reasons else ("complete", [])


def _software_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for distribution in (
        "numpy", "opencv-python-headless", "scikit-image", "pandas", "pyarrow",
        "torch", "torchvision", "lpips", "DISTS-pytorch",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def evaluate_sample(
    sample: Sample,
    config: dict,
    output_root: str | Path,
    enabled: set[str] | None = None,
    device: str = "auto",
    overwrite: bool = False,
    hash_inputs: bool = False,
    backends: dict | None = None,
) -> tuple[str, dict]:
    started = time.monotonic()
    if config.get("runtime", {}).get("torch_home"):
        os.environ.setdefault("TORCH_HOME", str(config["runtime"]["torch_home"]))
    enabled = set(enabled or config["metrics"]["enabled"])
    unknown = enabled - ALLOWED_METRICS
    if unknown:
        raise ValueError(f"unknown metrics: {sorted(unknown)}")
    paths = record_paths(output_root, sample)
    info = _required_input_check(sample)
    fingerprints = _fingerprints(sample, hash_inputs)
    signature = evaluation_signature(sample, config, enabled)
    if not overwrite and is_current_record(
        paths["result"], signature, fingerprints, config, enabled
    ):
        return "skipped", json.loads(paths["result"].read_text(encoding="utf-8"))
    stale_seconds = float(config.get("runtime", {}).get("lock_stale_hours", 2.0)) * 3600.0
    with sample_lock(paths["lock"], stale_seconds):
        if not overwrite and is_current_record(
            paths["result"], signature, fingerprints, config, enabled
        ):
            return "skipped", json.loads(paths["result"].read_text(encoding="utf-8"))
        resolved_device = choose_device(device)
        if backends is None:
            backends = {
                "perceptual": PerceptualMetrics(
                    enabled, config["metrics"], resolved_device, config.get("runtime")
                ),
                "flow": (
                    make_flow_estimator(
                        config["metrics"]["temporal"],
                        resolved_device,
                        config.get("runtime"),
                    )
                    if "twe" in enabled else None
                ),
                "device": resolved_device,
            }
        else:
            if backends.get("device") != resolved_device:
                raise ValueError(
                    f"backend/device mismatch: {backends.get('device')} != {resolved_device}"
                )
            if "perceptual" not in backends:
                backends["perceptual"] = PerceptualMetrics(
                    enabled, config["metrics"], resolved_device, config.get("runtime")
                )
                backends["flow"] = (
                    make_flow_estimator(
                        config["metrics"]["temporal"],
                        resolved_device,
                        config.get("runtime"),
                    )
                    if "twe" in enabled else None
                )
        result = _evaluate_locked(
            sample, config, output_root, info, fingerprints, signature, enabled,
            resolved_device, started, backends
        )
        return "evaluated", result


def _evaluate_locked(
    sample: Sample,
    config: dict,
    output_root: str | Path,
    info: dict,
    fingerprints: dict,
    signature: str,
    enabled: set[str],
    device: str,
    started: float,
    backends: dict,
) -> dict:
    ffprobe = config.get("runtime", {}).get("ffprobe", "ffprobe")
    prediction_probe = probe_video(sample.prediction_path, ffprobe)
    target_probe = probe_video(sample.gt_path, ffprobe)
    mask_probe = probe_video(sample.glyph_mask_path, ffprobe)
    fps_tolerance = float(config.get("runtime", {}).get("fps_tolerance", 0.01))
    fps_validation = {
        "method": "exact_avg_frame_rate_fraction_with_absolute_tolerance",
        "prediction_vs_target": _validate_fps_pair(
            "prediction", prediction_probe, target_probe, fps_tolerance
        ),
        "mask_vs_target": _validate_fps_pair(
            "glyph_mask", mask_probe, target_probe, fps_tolerance
        ),
    }
    if mask_probe.frame_count < target_probe.frame_count:
        raise ValueError(
            f"glyph mask is shorter than target: {mask_probe.frame_count} < {target_probe.frame_count}"
        )
    temporal_mapping = validate_temporal_mapping(
        sample.model,
        config,
        info,
        prediction_probe.frame_count,
        target_probe.frame_count,
    )
    adapter = make_adapter(
        sample.model,
        config,
        info,
        (prediction_probe.width, prediction_probe.height),
        (target_probe.width, target_probe.height),
    )
    perceptual = backends["perceptual"]
    flow_estimator = backends["flow"]
    bboxes = load_bboxes(sample.bbox_json_path)
    sampling = build_sampling_plan(
        target_probe.frame_count, bboxes, config.get("sampling")
    )
    spatial_frame_indices = set(sampling["spatial_frame_indices"])
    temporal_pair_end_indices = set(sampling["temporal_pair_end_indices"])
    ffmpeg = config.get("runtime", {}).get("ffmpeg", "ffmpeg")
    prediction_frames = iter(read_video_frames(
        sample.prediction_path, (prediction_probe.width, prediction_probe.height), ffmpeg
    ))
    target_frames = iter(read_video_frames(
        sample.gt_path, (target_probe.width, target_probe.height), ffmpeg
    ))
    mask_frames = iter(read_video_frames(
        sample.glyph_mask_path, (mask_probe.width, mask_probe.height), ffmpeg
    ))
    accumulator = RegionAccumulator()
    frame_rows: list[dict] = []
    previous = None
    previous_regions = None
    geometry_metadata = None
    ssim_cfg = config["metrics"].get("ssim", {})
    ssim_interior = ssim_support((target_probe.height, target_probe.width), ssim_cfg)
    region_matrix = {name: _metric_regions(config, name) for name in ALLOWED_METRICS}
    dists_conditioned = set(config["metrics"].get("dists", {}).get("conditioned_regions", []))
    bbox_tolerance = int(config["metrics"].get("max_glyph_outside_bbox_pixels", 0))

    for frame_index in range(target_probe.frame_count):
        target = _next_frame(target_frames, "target", frame_index)
        raw_mask = _next_frame(mask_frames, "glyph mask", frame_index)
        if target.shape[:2] != (target_probe.height, target_probe.width):
            raise RuntimeError(f"target resolution changed at frame {frame_index}")
        glyph = glyph_mask(raw_mask, (target_probe.width, target_probe.height), int(config["metrics"].get("mask_threshold", 127)))
        bbox = rasterize_bboxes(bboxes.get(frame_index), target_probe.width, target_probe.height)
        regions = make_regions(
            target_probe.width,
            target_probe.height,
            bbox,
            glyph,
            max_glyph_outside_bbox_pixels=bbox_tolerance,
        )
        if frame_index >= temporal_mapping["evaluated_frame_count"]:
            for region, region_mask in regions.items():
                row = {
                    "model": sample.model,
                    "video_id": sample.video_id,
                    "split": sample.split,
                    "frame_index": frame_index,
                    "region": region,
                    "status": "missing_prediction",
                    "effective_pixels": 0,
                    "target_pixels": int(region_mask.sum()),
                    "coverage": 0.0 if region_mask.any() else None,
                }
                frame_rows.append(row)
                accumulator.add(region, row)
            continue

        raw_prediction = _next_frame(prediction_frames, "prediction", frame_index)
        mapped = adapter.map(raw_prediction)
        geometry_metadata = mapped.metadata
        prediction = mapped.prediction
        valid = mapped.valid_mask
        if prediction.shape != target.shape:
            raise RuntimeError(f"mapped prediction shape mismatch at frame {frame_index}")

        evaluate_spatial = frame_index in spatial_frame_indices
        evaluate_temporal = frame_index in temporal_pair_end_indices
        ssim_scalar = ssim_map = None
        if evaluate_spatial and "ssim" in enabled:
            ssim_scalar, ssim_map = ssim_scalar_and_map(prediction, target, ssim_cfg)
        lpips_scalar = lpips_spatial = None
        if evaluate_spatial and "lpips" in enabled:
            lpips_scalar, lpips_spatial = perceptual.lpips_outputs(prediction, target)
        temporal_output = None
        tc_output = None
        if evaluate_temporal and previous is not None and flow_estimator is not None:
            temporal_output = temporal_maps(
                previous[0], prediction, previous[1], target, previous[2], valid,
                previous_regions, flow_estimator,
                float(config["metrics"]["temporal"].get("occlusion_alpha", 0.01)),
                float(config["metrics"]["temporal"].get("occlusion_beta", 0.5)),
            )
        elif evaluate_temporal and previous is not None and "tc" in enabled:
            tc_output = temporal_change_map(
                previous[0], prediction, previous[1], target, previous[2], valid
            )

        for region, region_mask in regions.items():
            effective = region_mask & valid
            target_pixels = int(region_mask.sum())
            effective_pixels = int(effective.sum())
            row = {
                "model": sample.model,
                "video_id": sample.video_id,
                "split": sample.split,
                "frame_index": frame_index,
                "region": region,
                "status": "evaluated",
                "effective_pixels": effective_pixels,
                "target_pixels": target_pixels,
                "coverage": effective_pixels / target_pixels if target_pixels else None,
            }

            if evaluate_spatial and "psnr" in enabled and region in region_matrix["psnr"]:
                psnr = masked_psnr(prediction, target, effective)
                row.update({
                    "psnr": psnr["value"],
                    "psnr_exact_match": psnr["exact_match"],
                    "sse": psnr["sse"],
                    "psnr_value_count": psnr["value_count"],
                })
            if evaluate_spatial and "ssim" in enabled and region in region_matrix["ssim"]:
                if region == "overall" and valid.all():
                    value, count = ssim_scalar, 1
                else:
                    value, count = masked_map_mean(ssim_map, effective & ssim_interior)
                row.update({"ssim": value, "ssim_sum": (value or 0.0) * count, "ssim_count": count})
            if evaluate_spatial and "lpips" in enabled and region in region_matrix["lpips"]:
                if region == "overall" and valid.all():
                    value, count = lpips_scalar, 1
                    mass = None
                else:
                    value, mass = perceptual.masked_lpips(lpips_spatial, effective)
                    count = 1 if value is not None else 0
                row.update({
                    "lpips": value,
                    "lpips_sum": (value or 0.0) * count,
                    "lpips_count": count,
                    "lpips_lowres_mask_mass": mass,
                })
            if evaluate_spatial and "dists" in enabled and region in region_matrix["dists"]:
                if region == "overall" and not valid.all():
                    dists_result = {
                        "value": None,
                        "backend": None,
                        "input_size": None,
                        "input_resized": False,
                        "fallback_count": 0,
                        "gpu_retry_count": 0,
                        "nonfinite_count": 0,
                        "null_reason": "incomplete_spatial_coverage",
                    }
                else:
                    context = None if region == "overall" else bbox
                    try:
                        dists_result = perceptual.dists_with_diagnostics(
                            prediction,
                            target,
                            effective,
                            context,
                            conditioned=region in dists_conditioned,
                        )
                    except DISTSNonFiniteError as exc:
                        raise DISTSNonFiniteError(
                            f"frame {frame_index}, region {region}: {exc}"
                        ) from exc
                value = dists_result["value"]
                row.update({
                    "dists": value,
                    "dists_sum": value or 0.0,
                    "dists_count": 1 if value is not None else 0,
                    "dists_backend": dists_result["backend"],
                    "dists_input_width": (
                        dists_result["input_size"][0]
                        if dists_result["input_size"] is not None else None
                    ),
                    "dists_input_height": (
                        dists_result["input_size"][1]
                        if dists_result["input_size"] is not None else None
                    ),
                    "dists_input_resized": dists_result["input_resized"],
                    "dists_fallback_count": dists_result["fallback_count"],
                    "dists_gpu_retry_count": dists_result["gpu_retry_count"],
                    "dists_nonfinite_count": dists_result["nonfinite_count"],
                    "dists_null_reason": dists_result["null_reason"],
                })
            if temporal_output is not None:
                if "twe" in enabled and region in region_matrix["twe"]:
                    pair_mask = (
                        effective
                        & temporal_output.warp_valid
                        & temporal_output.warped_previous_regions[region]
                    )
                    pair_count = int(pair_mask.sum())
                    if pair_count:
                        twe = float(temporal_output.prediction_warp_error[pair_mask].mean())
                        target_twe = float(temporal_output.target_warp_error[pair_mask].mean())
                    else:
                        twe = target_twe = None
                    row.update({
                        "twe": twe,
                        "twe_target": target_twe,
                        "twe_excess": twe - target_twe if twe is not None else None,
                        "twe_sum": (twe or 0.0) * pair_count,
                        "twe_count": pair_count,
                        "twe_target_sum": (target_twe or 0.0) * pair_count,
                        "twe_target_count": pair_count,
                    })
                if "tc" in enabled and region in region_matrix["tc"]:
                    change_mask = (
                        effective
                        & previous_regions[region]
                        & temporal_output.change_valid
                    )
                    change_count = int(change_mask.sum())
                    tc_value = (
                        float(temporal_output.change_error[change_mask].mean()) if change_count else None
                    )
                    row.update({
                        "tc": tc_value,
                        "tc_sum": (tc_value or 0.0) * change_count,
                        "tc_count": change_count,
                    })
            elif tc_output is not None and "tc" in enabled and region in region_matrix["tc"]:
                change_error, change_valid = tc_output
                change_mask = effective & previous_regions[region] & change_valid
                change_count = int(change_mask.sum())
                tc_value = float(change_error[change_mask].mean()) if change_count else None
                row.update({
                    "tc": tc_value,
                    "tc_sum": (tc_value or 0.0) * change_count,
                    "tc_count": change_count,
                })
            frame_rows.append(row)
            accumulator.add(region, row)
        previous = (prediction, target, valid)
        previous_regions = regions

    if geometry_metadata is None:
        raise RuntimeError("no prediction frame was evaluated")
    paths = record_paths(output_root, sample)
    record_cfg = config.get("records", {})
    if record_cfg.get("write_frames_parquet", True):
        atomic_parquet(paths["frames_parquet"], frame_rows)
        frame_record_path = paths["frames_parquet"]
        frame_record_format = "parquet"
    else:
        atomic_jsonl_gz(paths["frames_jsonl"], frame_rows)
        frame_record_path = paths["frames_jsonl"]
        frame_record_format = "jsonl.gz"
    region_summaries = {region: accumulator.summary(region) for region in REGIONS}
    spatial_coverage = geometry_metadata["valid_gt_pixels"] / (target_probe.width * target_probe.height)
    completeness, partial_reasons = _coverage_classification(
        spatial_coverage,
        temporal_mapping["temporal_coverage"],
    )
    temporal_estimator = None
    if flow_estimator is not None:
        input_height, input_width = (
            flow_estimator.input_size(target_probe.height, target_probe.width)
            if hasattr(flow_estimator, "input_size")
            else (target_probe.height, target_probe.width)
        )
        temporal_estimator = {
            "backend": config["metrics"]["temporal"].get("backend"),
            "target_size": [target_probe.width, target_probe.height],
            "flow_input_size": [input_width, input_height],
            "resized": (input_height, input_width)
            != (target_probe.height, target_probe.width),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "execution_revision": EXECUTION_REVISION,
        "evaluation_signature": signature,
        "status": "success",
        "completeness": completeness,
        "partial_reasons": partial_reasons,
        "model": sample.model,
        "video_id": sample.video_id,
        "split": sample.split,
        "requested_metrics": sorted(enabled),
        "evaluation_profile": config.get("evaluation_profile", "legacy_full"),
        "sampling": sampling,
        "evaluation_software": _software_versions(),
        "inputs": fingerprints,
        "prediction_probe": prediction_probe.__dict__,
        "target_probe": target_probe.__dict__,
        "mask_probe": mask_probe.__dict__,
        "fps_validation": fps_validation,
        "temporal_mapping": temporal_mapping,
        "temporal_estimator": temporal_estimator,
        "dists_execution": {
            "overall_max_input_long_side": int(
                config.get("runtime", {}).get(
                    "dists_overall_max_input_long_side", 0
                ) or 0
            ),
            "overall_downsampling": "pytorch_area",
            "conditioned_context_resized": False,
        } if "dists" in enabled else None,
        "geometry": geometry_metadata,
        "spatial_coverage": spatial_coverage,
        "spatiotemporal_coverage": region_summaries["overall"]["coverage"],
        "regions": region_summaries,
        "efficiency": _result_efficiency(info),
        "frame_record_path": str(frame_record_path),
        "frame_record_format": frame_record_format,
        "evaluation_seconds": round(time.monotonic() - started, 3),
    }
    atomic_json(paths["result"], result)
    return result


def write_failure(sample: Sample, output_root: str | Path, exc: Exception) -> dict:
    message = str(exc)
    if isinstance(exc, FileNotFoundError):
        category = "missing_artifact"
    elif isinstance(exc, DISTSNonFiniteError):
        category = "metric_backend_failure"
    elif "inference status" in message:
        category = "inference_failure"
    elif "decode" in message:
        category = "decode_failure"
    elif "glyph/BBox" in message:
        category = "annotation_failure"
    else:
        category = "evaluation_failure"
    record = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "execution_revision": EXECUTION_REVISION,
        "status": "failure",
        "model": sample.model,
        "video_id": sample.video_id,
        "split": sample.split,
        "failure_category": category,
        "failure_type": type(exc).__name__,
        "error": message,
        "paths": {
            "info": str(sample.info_path),
            "prediction": str(sample.prediction_path),
            "ground_truth": str(sample.gt_path),
        },
    }
    atomic_json(record_paths(output_root, sample)["result"], record)
    return record
