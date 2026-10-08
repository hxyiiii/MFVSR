from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .atomic import atomic_json
from .cli import parse_gpus, parse_models
from .config import load_config, validate_protocol_config
from .evaluator import record_paths
from .io import iter_samples, load_manifest_rows
from .profiles import apply_evaluation_profile
from .selection_io import load_video_ids
from .worker import shard_for


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume-safe single/multi-GPU MFASR evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--model", action="append", required=True, help="name, comma list, or all")
    parser.add_argument("--gpus", default="0", help="e.g. 0,1,2,3 or cpu")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--output-root")
    parser.add_argument("--profile", help="declared evaluation profile")
    parser.add_argument("--ids-file", help="one video_id per line, or CSV with video_id")
    parser.add_argument("--video-id")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--metrics")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-inputs", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _filtered_rows(rows: list[dict], args) -> list[dict]:
    allowed_ids = load_video_ids(args.ids_file) if args.ids_file else None
    output = [
        row for row in rows
        if (args.video_id is None or str(row["video_id"]) == args.video_id)
        and (allowed_ids is None or str(row["video_id"]) in allowed_ids)
        and (
            args.split is None
            or str(row["split"]) == args.split
            or str(row["split"]).startswith(args.split.rstrip("/") + "/")
        )
    ]
    output.sort(key=lambda row: (str(row["split"]), str(row["video_id"])))
    return output[: args.limit] if args.limit is not None else output


def _scan_status(config: dict, output_root: Path, rows: list[dict], models: list[str]) -> dict:
    counts = Counter()
    per_model: dict[str, Counter] = {model: Counter() for model in models}
    for sample in iter_samples(config, rows, models):
        path = record_paths(output_root, sample)["result"]
        if not path.is_file():
            status = "missing_record"
        else:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                status = record.get("status", "unknown")
                if status == "success":
                    status = f"success_{record.get('completeness', 'unknown')}"
            except (OSError, json.JSONDecodeError):
                status = "malformed_record"
        counts[status] += 1
        per_model[sample.model][status] += 1
    return {
        "counts": dict(counts),
        "per_model": {model: dict(values) for model, values in per_model.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be at least 1")
    config = apply_evaluation_profile(load_config(args.config), args.profile)
    validate_protocol_config(config)
    models = parse_models(args.model, config["models"])
    gpus = parse_gpus(args.gpus)
    if gpus == ["cpu"] and args.workers_per_gpu != 1:
        raise SystemExit("CPU mode supports exactly one worker")
    manifest = args.manifest or config["manifest"]
    output_root = Path(args.output_root or config["output_root"])
    rows = _filtered_rows(load_manifest_rows(manifest), args)
    world_size = len(gpus) * args.workers_per_gpu
    plan = {
        "models": models,
        "sample_count": len(rows),
        "task_count": len(rows) * len(models),
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "world_size": world_size,
        "evaluation_profile": config.get("evaluation_profile"),
        "ids_file": args.ids_file,
        "tasks_per_worker": {
            str(rank): sum(shard_for(str(row["video_id"]), world_size) == rank for row in rows) * len(models)
            for rank in range(world_size)
        },
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    atomic_json(output_root / "summaries" / "config_snapshot.json", {
        "config_path": str(Path(args.config).resolve()),
        "manifest_path": str(Path(manifest).resolve()),
        "selected_models": models,
        "selected_metrics": args.metrics or config["metrics"]["enabled"],
        "config": config,
    })

    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    processes = []
    started = time.time()
    rank = 0
    for gpu in gpus:
        for _ in range(args.workers_per_gpu):
            command = [
                sys.executable,
                "-m", "mfasr_eval.worker",
                "--config", args.config,
                "--manifest", manifest,
                "--output-root", str(output_root),
                "--rank", str(rank),
                "--world-size", str(world_size),
                "--device", ("cpu" if gpu == "cpu" else args.device),
                "--log-file", str(log_root / f"worker_{rank}.log"),
            ]
            for model in models:
                command.extend(["--model", model])
            for name, value in (
                ("--profile", args.profile),
                ("--ids-file", args.ids_file),
                ("--video-id", args.video_id),
                ("--split", args.split),
                ("--limit", args.limit),
                ("--metrics", args.metrics),
            ):
                if value is not None:
                    command.extend([name, str(value)])
            for flag, enabled in (
                ("--overwrite", args.overwrite),
                ("--hash-inputs", args.hash_inputs),
                ("--strict", args.strict),
            ):
                if enabled:
                    command.append(flag)
            environment = os.environ.copy()
            if gpu != "cpu":
                # ROCm applies device filters cumulatively. Setting the same
                # physical id in ROCR/HIP/CUDA visibility variables can first
                # remap it to logical device 0 and then filter that device out.
                # Keep one authoritative ROCm filter; the worker consequently
                # addresses its assigned physical GPU as cuda:0 through PyTorch.
                environment.pop("HIP_VISIBLE_DEVICES", None)
                environment.pop("CUDA_VISIBLE_DEVICES", None)
                environment["ROCR_VISIBLE_DEVICES"] = gpu
            processes.append((rank, subprocess.Popen(command, env=environment)))
            rank += 1
    return_codes = {str(rank): process.wait() for rank, process in processes}
    status = {
        "schema_version": "1.0",
        "plan": plan,
        "return_codes": return_codes,
        "elapsed_seconds": round(time.time() - started, 3),
        **_scan_status(config, output_root, rows, models),
    }
    atomic_json(output_root / "summaries" / "latest_run_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 1 if any(code != 0 for code in return_codes.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
