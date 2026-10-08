from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import REGIONS


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON by default; YAML is supported only when PyYAML is installed."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML; use JSON or install PyYAML") from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"config must be JSON or YAML: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"config root must be an object: {path}")
    return data


def require_keys(value: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{context} is missing keys: {', '.join(missing)}")


def validate_protocol_config(config: dict[str, Any]) -> None:
    require_keys(config, ("manifest", "input_root", "output_root", "models", "metrics"), "config")
    if not config["models"]:
        raise ValueError("config.models must not be empty")
    sampling = config.get("sampling") or {"mode": "all"}
    if sampling.get("mode", "all") not in {"all", "stratified_fixed_budget"}:
        raise ValueError("sampling.mode must be all or stratified_fixed_budget")
    for name in (
        "subtitle_active_frames", "global_frames",
        "subtitle_active_pairs", "global_pairs",
    ):
        if name in sampling and int(sampling[name]) < 0:
            raise ValueError(f"sampling.{name} must be non-negative")
    fps_tolerance = float((config.get("runtime") or {}).get("fps_tolerance", 0.01))
    if fps_tolerance < 0:
        raise ValueError("runtime.fps_tolerance must be non-negative")
    dists_action = str(
        (config.get("runtime") or {}).get("dists_nonfinite_action", "legacy_policy")
    )
    if dists_action not in {"legacy_policy", "gpu_retry"}:
        raise ValueError(
            "runtime.dists_nonfinite_action must be legacy_policy or gpu_retry"
        )
    dists_channel_chunk = int(
        (config.get("runtime") or {}).get("dists_l2pool_channel_chunk", 0)
    )
    if dists_channel_chunk < 0:
        raise ValueError("runtime.dists_l2pool_channel_chunk must be non-negative")
    native_conditioned = (config.get("runtime") or {}).get(
        "dists_native_gpu_for_conditioned", False
    )
    if not isinstance(native_conditioned, bool):
        raise ValueError("runtime.dists_native_gpu_for_conditioned must be boolean")
    dists_overall_cap = int(
        (config.get("runtime") or {}).get("dists_overall_max_input_long_side", 0) or 0
    )
    if dists_overall_cap < 0:
        raise ValueError(
            "runtime.dists_overall_max_input_long_side must be non-negative"
        )
    raft_max_input_long_side = int(
        (config.get("runtime") or {}).get("raft_max_input_long_side", 0) or 0
    )
    if raft_max_input_long_side < 0:
        raise ValueError("runtime.raft_max_input_long_side must be non-negative")
    raft_resize_trigger = int(
        (config.get("runtime") or {}).get("raft_resize_trigger_long_side", 0) or 0
    )
    if raft_resize_trigger < 0:
        raise ValueError("runtime.raft_resize_trigger_long_side must be non-negative")
    if (
        raft_max_input_long_side > 0
        and raft_resize_trigger > 0
        and raft_max_input_long_side > raft_resize_trigger
    ):
        raise ValueError(
            "runtime.raft_max_input_long_side must not exceed its resize trigger"
        )
    for model, rule in config["models"].items():
        require_keys(rule, ("result_root", "geometry", "temporal"), f"models.{model}")
        if rule["geometry"] not in {"full_frame_resize_to_gt", "clear_center_crop_inverse"}:
            raise ValueError(f"models.{model}.geometry is unsupported: {rule['geometry']}")
        if rule["temporal"] not in {"exact", "verified_prefix"}:
            raise ValueError(f"models.{model}.temporal is unsupported: {rule['temporal']}")
    metrics = config["metrics"]
    require_keys(metrics, ("enabled", "region_matrix"), "metrics")
    allowed_metrics = {"psnr", "ssim", "lpips", "dists", "twe", "tc"}
    unknown_metrics = set(metrics["enabled"]) - allowed_metrics
    if unknown_metrics:
        raise ValueError(f"metrics.enabled contains unsupported values: {sorted(unknown_metrics)}")
    for metric in metrics["enabled"]:
        if metric not in metrics["region_matrix"]:
            raise ValueError(f"metrics.region_matrix lacks enabled metric {metric}")
        regions = set(metrics["region_matrix"][metric])
        unknown_regions = regions - set(REGIONS)
        if unknown_regions:
            raise ValueError(f"metrics.region_matrix.{metric} contains unknown regions: {sorted(unknown_regions)}")
        if not regions:
            raise ValueError(f"metrics.region_matrix.{metric} must not be empty")
    conditioned = set((metrics.get("dists") or {}).get("conditioned_regions", []))
    dists_regions = set(metrics["region_matrix"].get("dists", []))
    if "dists" in metrics["enabled"] and not conditioned <= dists_regions:
        raise ValueError("DISTS conditioned_regions must be included in the DISTS region matrix")
    dists_nonfinite_policy = str(
        (metrics.get("dists") or {}).get("nonfinite_policy", "null")
    )
    if dists_nonfinite_policy not in {"null", "cpu_fallback_same_weights"}:
        raise ValueError(
            "metrics.dists.nonfinite_policy must be null or cpu_fallback_same_weights"
        )
    analysis = config.get("analysis") or {}
    for name in ("require_terminal_records", "require_current_records"):
        if name in analysis and not isinstance(analysis[name], bool):
            raise ValueError(f"analysis.{name} must be boolean")
