from __future__ import annotations

import math
from typing import Iterable


SCORE_VERSION = "sr_audit300_decoupled_v2"


SCORE_COMPONENTS = {
    "restoration": (
        ("subtitle", "psnr", "higher"),
        ("subtitle", "lpips", "lower"),
        ("subtitle", "dists", "lower"),
    ),
    "global_preservation": (
        ("outside", "psnr", "higher"),
        ("outside", "lpips", "lower"),
    ),
    "local_temporal": (
        ("bbox", "twe_excess", "lower_nonnegative"),
        ("bbox", "tc", "lower"),
    ),
    "global_temporal": (
        ("outside", "twe_excess", "lower_nonnegative"),
        ("outside", "tc", "lower"),
    ),
}


def metric_key(region: str, metric: str) -> str:
    return f"{region}.{metric}"


def metric_value(record: dict, region: str, metric: str) -> float | None:
    metrics = (((record.get("regions") or {}).get(region) or {}).get("metrics") or {})
    value = metrics.get(metric)
    if value is None and metric == "psnr" and metrics.get("infinite_psnr") is True:
        return float("inf")
    if value is None:
        return None
    value = float(value)
    if metric == "twe_excess":
        value = max(0.0, value)
    return value


def normalized_quality(value: float, anchor: dict, direction: str) -> float:
    low = float(anchor["p05"])
    high = float(anchor["p95"])
    if not high > low:
        raise ValueError(f"invalid anchors: p05={low}, p95={high}")
    clipped = min(high, max(low, value))
    if direction == "higher":
        result = (clipped - low) / (high - low)
    elif direction in {"lower", "lower_nonnegative"}:
        result = (high - clipped) / (high - low)
    else:
        raise ValueError(f"unknown metric direction: {direction}")
    return min(1.0, max(0.0, result))


def weighted_geometric(values: Iterable[float], weights: Iterable[float] | None = None) -> float:
    values = list(values)
    if not values:
        raise ValueError("geometric mean needs at least one value")
    weights = list(weights) if weights is not None else [1.0] * len(values)
    if len(weights) != len(values) or sum(weights) <= 0:
        raise ValueError("invalid geometric-mean weights")
    if any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(weight * math.log(value) for value, weight in zip(values, weights)) / sum(weights))


def _component(record: dict, name: str, anchors: dict) -> tuple[float | None, dict]:
    qualities = []
    detail = {}
    for region, metric, direction in SCORE_COMPONENTS[name]:
        key = metric_key(region, metric)
        raw = metric_value(record, region, metric)
        if raw is None or key not in anchors:
            return None, {"missing": key}
        quality = normalized_quality(raw, anchors[key], direction)
        qualities.append(quality)
        detail[key] = {"raw": raw, "quality": quality}
    return sum(qualities) / len(qualities), detail


def score_record(record: dict, anchor_file: dict) -> dict:
    score_version = anchor_file.get("score_version")
    if score_version != SCORE_VERSION:
        raise ValueError(
            f"incompatible SR anchor score_version: {score_version!r}; "
            f"expected {SCORE_VERSION!r}"
        )
    anchors = anchor_file["anchors"]
    values = {}
    details = {}
    for name in SCORE_COMPONENTS:
        values[name], details[name] = _component(record, name, anchors)
    missing = [name for name, value in values.items() if value is None]
    if missing:
        return {"sr_q": None, "effective_sr": None, "missing_components": missing, "details": details}
    # BBox-minus-glyph is intentionally excluded from SR-Score. That support
    # may contain both untouched local background and non-core subtitle effects
    # (e.g. shadows, outlines, glow, or boxes), so it is a mixed diagnostic
    # rather than an identifiable preservation component.
    preservation = values["global_preservation"]
    temporal = weighted_geometric([
        values["local_temporal"], values["global_temporal"]
    ])
    top_weights = anchor_file.get("dimension_weights", {
        "restoration": 1.0, "preservation": 1.0, "temporal": 1.0
    })
    sr_q = 100.0 * weighted_geometric(
        [values["restoration"], preservation, temporal],
        [top_weights["restoration"], top_weights["preservation"], top_weights["temporal"]],
    )
    coverage = float(record.get("spatiotemporal_coverage") or 0.0)
    return {
        "score_version": SCORE_VERSION,
        "restoration": 100.0 * values["restoration"],
        "global_preservation": 100.0 * values["global_preservation"],
        "preservation": 100.0 * preservation,
        "local_temporal": 100.0 * values["local_temporal"],
        "global_temporal": 100.0 * values["global_temporal"],
        "temporal": 100.0 * temporal,
        "sr_q": sr_q,
        # Kept as a null compatibility field.  The final protocol reports
        # failures and coverage separately instead of imputing failed samples
        # as zero-quality observations or extrapolating dense audit metrics.
        "effective_sr": None,
        "coverage": coverage,
        "missing_components": [],
        "details": details,
    }
