from __future__ import annotations


SAMPLING_VERSION = "stratified_annotation_support_v1"


def evenly_spaced(values: list[int], budget: int) -> list[int]:
    """Select deterministic, endpoint-covering indices without duplicates."""
    values = sorted(dict.fromkeys(int(value) for value in values))
    if budget <= 0 or not values:
        return []
    if len(values) <= budget:
        return values
    if budget == 1:
        return [values[len(values) // 2]]
    positions = [round(index * (len(values) - 1) / (budget - 1)) for index in range(budget)]
    return [values[position] for position in positions]


def _union_with_fill(
    primary: list[int],
    secondary: list[int],
    primary_budget: int,
    secondary_budget: int,
) -> list[int]:
    total_budget = max(0, primary_budget) + max(0, secondary_budget)
    selected = set(evenly_spaced(primary, primary_budget))
    selected.update(evenly_spaced(secondary, secondary_budget))
    if len(selected) < total_budget:
        remaining = [value for value in secondary if value not in selected]
        selected.update(evenly_spaced(remaining, total_budget - len(selected)))
    return sorted(selected)[:total_budget]


def build_sampling_plan(frame_count: int, bboxes: dict[int, object], config: dict | None) -> dict:
    """Freeze model-independent spatial frames and adjacent temporal pairs.

    A temporal index ``t`` denotes the adjacent pair ``(t-1, t)``.  Subtitle
    activity is derived only from benchmark BBoxes, never from model output.
    """
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    cfg = config or {"mode": "all"}
    mode = str(cfg.get("mode", "all"))
    all_frames = list(range(frame_count))
    all_pairs = list(range(1, frame_count))
    active_frames = [index for index in all_frames if bboxes.get(index)]
    active_set = set(active_frames)
    active_pairs = [
        index for index in all_pairs
        if index in active_set or (index - 1) in active_set
    ]
    if mode == "all":
        spatial = all_frames
        temporal = all_pairs
    elif mode == "stratified_fixed_budget":
        spatial = _union_with_fill(
            active_frames,
            all_frames,
            int(cfg.get("subtitle_active_frames", 32)),
            int(cfg.get("global_frames", 16)),
        )
        temporal = _union_with_fill(
            active_pairs,
            all_pairs,
            int(cfg.get("subtitle_active_pairs", 24)),
            int(cfg.get("global_pairs", 8)),
        )
    else:
        raise ValueError(f"unsupported sampling mode: {mode}")
    return {
        "version": SAMPLING_VERSION,
        "mode": mode,
        "frame_count": frame_count,
        "subtitle_active_frame_count": len(active_frames),
        "spatial_frame_indices": spatial,
        "temporal_pair_end_indices": temporal,
        "spatial_frame_count": len(spatial),
        "temporal_pair_count": len(temporal),
        "configured_budgets": {
            "subtitle_active_frames": cfg.get("subtitle_active_frames"),
            "global_frames": cfg.get("global_frames"),
            "subtitle_active_pairs": cfg.get("subtitle_active_pairs"),
            "global_pairs": cfg.get("global_pairs"),
        },
    }

