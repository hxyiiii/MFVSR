from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

from .cli import parse_models
from .config import load_config, validate_protocol_config
from .io import load_manifest_rows, sample_from_row
from .profiles import apply_evaluation_profile


def check(config: dict, models: list[str], manifest_path: str | Path, deep: bool, device: str) -> dict:
    report = {"ok": True, "checks": [], "errors": []}

    def add(name: str, ok: bool, detail: str):
        report["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            report["ok"] = False
            report["errors"].append(f"{name}: {detail}")

    try:
        validate_protocol_config(config)
        add("config_schema", True, "protocol structure and metric-region matrix are valid")
    except Exception as exc:
        add("config_schema", False, str(exc))
        return report
    for command in (config.get("runtime", {}).get("ffprobe", "ffprobe"),):
        add(f"command:{command}", shutil.which(command) is not None, shutil.which(command) or "not found")
    required_modules = ["numpy", "cv2", "skimage", "pandas", "pyarrow"]
    enabled = set(config["metrics"].get("enabled", []))
    if enabled & {"lpips", "dists", "twe", "tc"}:
        required_modules.append("torch")
    if "lpips" in enabled:
        required_modules.append("lpips")
    if "dists" in enabled:
        required_modules.append("DISTS_pytorch")
    if "twe" in enabled and str(config["metrics"].get("temporal", {}).get("backend", "")).startswith("raft"):
        required_modules.append("torchvision")
    for module in required_modules:
        try:
            imported = importlib.import_module(module)
            add(f"module:{module}", True, str(getattr(imported, "__version__", "available")))
        except Exception as exc:
            add(f"module:{module}", False, f"{type(exc).__name__}: {exc}")
    try:
        rows = load_manifest_rows(manifest_path)
        add("manifest", len(rows) > 0, f"{len(rows)} unique rows")
    except Exception as exc:
        add("manifest", False, str(exc))
        return report
    for model in models:
        add(f"model_root:{model}", Path(config["models"][model]["result_root"]).is_dir(), config["models"][model]["result_root"])
        representative = None
        for row in rows:
            candidate = sample_from_row(config, row, model)
            if candidate.info_path.is_file():
                representative = candidate
                break
        add(
            f"sample_info:{model}", representative is not None,
            str(representative.info_path) if representative else "no info.json found under configured result root",
        )
    if deep and report["ok"]:
        try:
            from .metrics import PerceptualMetrics
            from .temporal import make_flow_estimator

            if config.get("runtime", {}).get("torch_home"):
                os.environ.setdefault("TORCH_HOME", str(config["runtime"]["torch_home"]))
            PerceptualMetrics(enabled, config["metrics"], device)
            if "twe" in enabled:
                make_flow_estimator(config["metrics"]["temporal"], device)
            add("deep_model_initialization", True, f"metric weights loaded on {device}")
        except Exception as exc:
            add("deep_model_initialization", False, f"{type(exc).__name__}: {exc}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-fast dependency and data audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--profile")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    config = apply_evaluation_profile(load_config(args.config), args.profile)
    models = parse_models(args.model or ["all"], config["models"])
    report = check(config, models, args.manifest or config["manifest"], args.deep, args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
