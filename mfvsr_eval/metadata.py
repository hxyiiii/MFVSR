from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .atomic import atomic_json
from .config import load_config
from .io import load_manifest_rows, probe_video
from .inventory import is_box_style


FACTOR_PATTERNS = {
    "spatial_layout": re.compile(r"_(Fixed|Dynamic)-"),
    "temporal_behavior": re.compile(r"_(Static)(?:_|$)|_(Animated)-"),
    "visual_style": re.compile(r"_(Plain|Decorated|Stylized)-"),
    "language_mode": re.compile(r"_(Monolingual|Bilingual)-"),
}

METADATA_SCHEMA_VERSION = "1.3"


def metadata_signature(metadata_cfg: dict) -> str:
    """Fingerprint semantic grouping rules, excluding operational settings."""
    semantic_config = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "resolution_short_side_threshold": int(
            metadata_cfg.get("resolution_short_side_threshold", 900)
        ),
        "short_duration_max_seconds": float(
            metadata_cfg.get("short_duration_max_seconds", 10.0)
        ),
        "medium_duration_max_seconds": float(
            metadata_cfg.get("medium_duration_max_seconds", 30.0)
        ),
        "difficulty_weights": metadata_cfg["difficulty_weights"],
    }
    encoded = json.dumps(
        semantic_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_factors(video_id: str) -> dict[str, str]:
    output = {}
    for name, pattern in FACTOR_PATTERNS.items():
        match = pattern.search(video_id)
        if match is None:
            raise ValueError(f"cannot parse {name} from video_id: {video_id}")
        output[name] = next(group for group in match.groups() if group is not None)
    return output


def source_class(split: str) -> str:
    if split == "AI_gen" or split.startswith("AI_gen/"):
        return "AI-Generated"
    if split == "origin_clean/1080P_above" or split.startswith("origin_clean/1080P_above/"):
        return "Real-HQ"
    if split == "origin_clean/512P_to_1080P" or split.startswith("origin_clean/512P_to_1080P/"):
        return "Real-LQ"
    raise ValueError(f"unrecognized source split: {split}")


def duration_class(seconds: float, short_max: float = 10.0, medium_max: float = 30.0) -> str:
    if seconds <= short_max:
        return "short_le_10s"
    if seconds <= medium_max:
        return "medium_10_to_30s"
    return "long_gt_30s"


def resolution_class(width: int, height: int, threshold: int = 900) -> str:
    return "720p-class" if min(width, height) < threshold else "1080p-class"


def difficulty(factors: dict[str, str], weights: dict) -> tuple[int, str]:
    values = {
        "spatial_layout": weights["spatial_layout"][factors["spatial_layout"]],
        "temporal_behavior": weights["temporal_behavior"][factors["temporal_behavior"]],
        "visual_style": weights["visual_style"][factors["visual_style"]],
        "language_mode": weights["language_mode"][factors["language_mode"]],
    }
    score = int(sum(values.values()))
    cutoffs = weights.get("cutoffs", {"easy_max": 1, "medium_max": 3})
    if score <= int(cutoffs["easy_max"]):
        tier = "Easy"
    elif score <= int(cutoffs["medium_max"]):
        tier = "Medium"
    else:
        tier = "Hard"
    return score, tier


def validate_dimension_coverage(samples: list[dict], metadata_cfg: dict) -> dict[str, dict[str, int]]:
    expected_count = metadata_cfg.get("expected_sample_count")
    if expected_count is not None and len(samples) != int(expected_count):
        raise ValueError(f"metadata sample count mismatch: {len(samples)} != {expected_count}")
    summary = {}
    for name, expected_values in (metadata_cfg.get("expected_categories") or {}).items():
        counts = Counter(str(sample[name]) for sample in samples)
        observed = set(counts)
        expected = {str(value) for value in expected_values}
        if observed != expected:
            raise ValueError(
                f"metadata categories mismatch for {name}: "
                f"observed={sorted(observed)}, expected={sorted(expected)}"
            )
        summary[name] = dict(sorted(counts.items()))
    return summary


def _build_sample_metadata(row: dict, config: dict, fingerprint: dict) -> dict:
    metadata_cfg = config["analysis"]["metadata"]
    ffprobe = config.get("runtime", {}).get("ffprobe", "ffprobe")
    probe = probe_video(Path(row["gt_video_path"]), ffprobe)
    factors = parse_factors(str(row["video_id"]))
    score, tier = difficulty(factors, metadata_cfg["difficulty_weights"])
    return {
        "video_id": str(row["video_id"]),
        "split": str(row["split"]),
        "source": source_class(str(row["split"])),
        "width": probe.width,
        "height": probe.height,
        "resolution": f"{probe.width}x{probe.height}",
        "resolution_class": resolution_class(
            probe.width, probe.height,
            int(metadata_cfg.get("resolution_short_side_threshold", 900)),
        ),
        "fps": probe.fps,
        "frame_count": probe.frame_count,
        "duration_seconds": probe.duration_seconds,
        "duration_class": duration_class(
            probe.duration_seconds,
            float(metadata_cfg.get("short_duration_max_seconds", 10.0)),
            float(metadata_cfg.get("medium_duration_max_seconds", 30.0)),
        ),
        **factors,
        "difficulty_score": score,
        "difficulty_tier": tier,
        "box_style": is_box_style(str(row["video_id"])),
        "gt_fingerprint": fingerprint,
    }


def build_metadata(config: dict, manifest_path: str | Path, output: str | Path, resume: bool = True) -> dict:
    rows = load_manifest_rows(manifest_path)
    output = Path(output)
    metadata_cfg = config["analysis"]["metadata"]
    signature = metadata_signature(metadata_cfg)
    cached = {}
    if resume and output.is_file():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior.get("metadata_signature") == signature:
            cached = {str(row["video_id"]): row for row in prior.get("samples", [])}
    probe_workers = int(metadata_cfg.get("probe_workers", 8))
    if probe_workers < 1:
        raise ValueError("metadata probe_workers must be at least 1")
    samples = []
    pending: list[dict | Future] = []
    with ThreadPoolExecutor(max_workers=probe_workers) as executor:
        for row in rows:
            gt_path = Path(row["gt_video_path"])
            stat = gt_path.stat()
            fingerprint = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            prior = cached.get(str(row["video_id"]))
            if prior and prior.get("gt_fingerprint") == fingerprint:
                pending.append(prior)
            else:
                pending.append(executor.submit(_build_sample_metadata, row, config, fingerprint))
        for index, item in enumerate(pending, start=1):
            samples.append(item.result() if isinstance(item, Future) else item)
            if index % int(metadata_cfg.get("checkpoint_every", 50)) == 0:
                atomic_json(output, {
                    "schema_version": METADATA_SCHEMA_VERSION,
                    "metadata_signature": signature,
                    "samples": samples,
                })
    dimension_counts = validate_dimension_coverage(samples, metadata_cfg)
    result = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "metadata_signature": signature,
        "sample_count": len(samples),
        "dimension_counts": dimension_counts,
        "samples": samples,
    }
    atomic_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build frozen MFVSR analysis metadata")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output = args.output or config["analysis"]["metadata_path"]
    result = build_metadata(config, args.manifest or config["manifest"], output, not args.no_resume)
    print(json.dumps({"output": str(output), "sample_count": len(result["samples"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
