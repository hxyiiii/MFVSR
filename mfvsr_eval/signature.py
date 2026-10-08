from __future__ import annotations

import hashlib
import json

from . import METRIC_DEFINITION_VERSION, PROTOCOL_VERSION


def evaluation_signature_for_model(model: str, config: dict, enabled: set[str]) -> str:
    payload = {
        "protocol": PROTOCOL_VERSION,
        # Execution-only bug fixes must not invalidate numerically compatible
        # records. This version changes only when a metric definition changes.
        "evaluator": METRIC_DEFINITION_VERSION,
        "model": model,
        "model_rule": config["models"][model],
        "metrics": config["metrics"],
        "enabled": sorted(enabled),
    }
    profile = config.get("evaluation_profile", "legacy_full")
    sampling = config.get("sampling") or {"mode": "all"}
    # Preserve byte-for-byte legacy signatures so ongoing/full-frame records
    # remain resumable after installing the lean-profile implementation.
    if profile != "legacy_full" or sampling.get("mode", "all") != "all":
        payload["evaluation_profile"] = profile
        payload["sampling"] = sampling
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
