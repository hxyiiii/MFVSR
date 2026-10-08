from __future__ import annotations

from copy import deepcopy


def apply_evaluation_profile(config: dict, profile_name: str | None) -> dict:
    """Return an isolated config with one declared evaluation profile applied.

    Profiles only narrow/override execution fields.  The source config object is
    never mutated, which keeps launcher tests and repeated in-process runs safe.
    """
    result = deepcopy(config)
    if profile_name is None:
        result["evaluation_profile"] = "legacy_full"
        return result
    profiles = result.get("evaluation_profiles") or {}
    if profile_name not in profiles:
        raise ValueError(
            f"unknown evaluation profile {profile_name!r}; configured: {sorted(profiles)}"
        )
    profile = deepcopy(profiles[profile_name])
    if profile.get("output_root"):
        result["output_root"] = profile["output_root"]
    if "metrics" in profile:
        metrics = result.setdefault("metrics", {})
        for key, value in profile["metrics"].items():
            metrics[key] = deepcopy(value)
    if "sampling" in profile:
        result["sampling"] = deepcopy(profile["sampling"])
    result["evaluation_profile"] = profile_name
    return result

