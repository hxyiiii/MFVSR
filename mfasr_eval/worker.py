from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from .cli import parse_metrics, parse_models
from .config import load_config, validate_protocol_config
from .evaluator import choose_device, evaluate_sample, write_failure
from .io import iter_samples, load_manifest_rows
from .profiles import apply_evaluation_profile
from .selection_io import load_video_ids


def shard_for(video_id: str, world_size: int) -> int:
    digest = hashlib.sha256(video_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % world_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal MFASR evaluation worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--profile")
    parser.add_argument("--ids-file")
    parser.add_argument("--video-id")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--metrics")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-inputs", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--log-file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise SystemExit("invalid rank/world-size")
    config = apply_evaluation_profile(load_config(args.config), args.profile)
    validate_protocol_config(config)
    models = parse_models(args.model, config["models"])
    enabled = parse_metrics(args.metrics, config["metrics"]["enabled"])
    device = choose_device(args.device)
    # Heavy LPIPS/DISTS/RAFT weights are initialized lazily after the first
    # non-current sample, then reused for the lifetime of this GPU worker.
    backends = {"device": device}
    output_root = Path(args.output_root or config["output_root"])
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    rows = load_manifest_rows(args.manifest or config["manifest"])
    allowed_ids = load_video_ids(args.ids_file) if args.ids_file else None
    filtered_rows = [
        row for row in rows
        if (args.video_id is None or str(row["video_id"]) == args.video_id)
        and (allowed_ids is None or str(row["video_id"]) in allowed_ids)
        and (
            args.split is None
            or str(row["split"]) == args.split
            or str(row["split"]).startswith(args.split.rstrip("/") + "/")
        )
    ]
    filtered_rows.sort(key=lambda row: (str(row["split"]), str(row["video_id"])))
    if args.limit is not None:
        filtered_rows = filtered_rows[: args.limit]
    filtered_rows = [
        row for row in filtered_rows
        if shard_for(str(row["video_id"]), args.world_size) == args.rank
    ]
    counts = Counter()
    for sample in iter_samples(config, filtered_rows, models):
        try:
            action, record = evaluate_sample(
                sample,
                config,
                output_root,
                enabled=enabled,
                device=device,
                overwrite=args.overwrite,
                hash_inputs=args.hash_inputs,
                backends=backends,
            )
            counts[action] += 1
        except Exception as exc:
            if "sample is locked by another evaluator" in str(exc):
                action = "locked"
                record = {"status": "locked", "error": str(exc)}
            else:
                logging.exception("evaluation failed: model=%s video=%s", sample.model, sample.video_id)
                record = write_failure(sample, output_root, exc)
                action = "failure"
            counts[action] += 1
            if args.strict and action == "failure":
                print(json.dumps({
                    "model": sample.model,
                    "video_id": sample.video_id,
                    "action": action,
                    "status": record.get("status"),
                    "error": record.get("error"),
                }, ensure_ascii=False), flush=True)
                break
        print(json.dumps({
            "model": sample.model,
            "video_id": sample.video_id,
            "action": action,
            "status": record.get("status"),
            "completeness": record.get("completeness"),
            "error": record.get("error"),
        }, ensure_ascii=False), flush=True)
    print(json.dumps({"rank": args.rank, "counts": dict(counts)}, ensure_ascii=False), flush=True)
    return 1 if args.strict and counts["failure"] else 0


if __name__ == "__main__":
    sys.exit(main())
