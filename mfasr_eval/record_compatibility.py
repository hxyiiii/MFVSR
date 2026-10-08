from __future__ import annotations


def runtime_record_is_compatible(record: dict, config: dict, enabled: set[str]) -> bool:
    """Selectively invalidate records affected by runtime-scoped metric fixes."""
    if "dists" not in enabled:
        return True
    cap = int(
        (config.get("runtime") or {}).get("dists_overall_max_input_long_side", 0) or 0
    )
    if cap <= 0:
        return True
    probe = record.get("target_probe") or {}
    try:
        target_long_side = max(int(probe["width"]), int(probe["height"]))
    except (KeyError, TypeError, ValueError):
        return False
    if target_long_side <= cap:
        return True
    policy = record.get("dists_execution") or {}
    return (
        policy.get("overall_max_input_long_side") == cap
        and policy.get("overall_downsampling") == "pytorch_area"
        and policy.get("conditioned_context_resized") is False
    )
