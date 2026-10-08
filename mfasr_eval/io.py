from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

from .schema import Sample, VideoProbe


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest_rows(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("samples")
    if not isinstance(rows, list):
        raise ValueError("manifest must be a list or an object containing a 'samples' list")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("split")), str(row.get("video_id")))
        if None in key or "None" in key:
            raise ValueError(f"manifest row lacks split/video_id: {row}")
        if key in seen:
            raise ValueError(f"duplicate manifest row: {key}")
        seen.add(key)
        for name in ("input_video_path", "gt_video_path", "glyph_mask_path", "bbox_json_path"):
            if name not in row:
                raise ValueError(f"manifest row {key} lacks {name}")
    return rows


def probe_video(path: str | Path, ffprobe: str = "ffprobe") -> VideoProbe:
    command = [
        ffprobe,
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,nb_frames,codec_name,pix_fmt,duration:format=duration",
        "-of", "json",
        str(path),
    ]
    data = json.loads(subprocess.check_output(command, text=True))
    if not data.get("streams"):
        raise RuntimeError(f"no video stream: {path}")
    stream = data["streams"][0]
    raw_fps = stream["avg_frame_rate"]
    numerator_text, denominator_text = raw_fps.split("/")
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if numerator <= 0 or denominator <= 0:
        raise RuntimeError(f"invalid average frame rate {raw_fps!r}: {path}")
    fps = numerator / denominator
    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if raw_count in (None, "N/A"):
        raise RuntimeError(f"ffprobe did not report a frame count: {path}")
    raw_duration = stream.get("duration") or (data.get("format") or {}).get("duration")
    duration = float(raw_duration) if raw_duration not in (None, "N/A") else int(raw_count) / fps
    return VideoProbe(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        fps_numerator=numerator,
        fps_denominator=denominator,
        frame_count=int(raw_count),
        duration_seconds=duration,
        codec=stream.get("codec_name"),
        pix_fmt=stream.get("pix_fmt"),
    )


def _info_candidates(config: dict, row: dict, model: str) -> list[Path]:
    model_root = Path(config["models"][model]["result_root"])
    input_root = Path(config["input_root"])
    input_path = Path(row["input_video_path"])
    try:
        relative = input_path.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(f"input path is outside input_root: {input_path}") from exc
    video_id = str(row["video_id"])
    mirrored = model_root / relative.parent
    return [
        mirrored / video_id / "info.json",
        mirrored / "info.json",
        model_root / str(row["split"]) / video_id / video_id / "info.json",
        model_root / str(row["split"]) / video_id / "info.json",
    ]


def sample_from_row(config: dict, row: dict, model: str) -> Sample:
    candidates = list(dict.fromkeys(_info_candidates(config, row, model)))
    info_path = next((path for path in candidates if path.is_file()), candidates[0])
    prediction = info_path.parent / f"{row['video_id']}_cleaned.mp4"
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            reported = (info.get("output_raw") or {}).get("video_path")
            if reported:
                prediction = Path(reported)
        except (OSError, json.JSONDecodeError):
            pass
    return Sample(
        model=model,
        video_id=str(row["video_id"]),
        split=str(row["split"]),
        input_path=Path(row["input_video_path"]),
        gt_path=Path(row["gt_video_path"]),
        glyph_mask_path=Path(row["glyph_mask_path"]),
        bbox_json_path=Path(row["bbox_json_path"]),
        info_path=info_path,
        prediction_path=prediction,
    )


def iter_samples(
    config: dict,
    rows: list[dict],
    models: list[str],
    video_id: str | None = None,
    split: str | None = None,
) -> Iterator[Sample]:
    for model in models:
        if model not in config["models"]:
            raise ValueError(f"unknown model {model!r}; configured: {sorted(config['models'])}")
        for row in sorted(rows, key=lambda item: (str(item["split"]), str(item["video_id"]))):
            if video_id is not None and str(row["video_id"]) != video_id:
                continue
            row_split = str(row["split"])
            if split is not None and row_split != split and not row_split.startswith(split.rstrip("/") + "/"):
                continue
            yield sample_from_row(config, row, model)


def read_video_frames(
    path: str | Path,
    size: tuple[int, int],
    ffmpeg: str = "ffmpeg",
):
    """Yield RGB uint8 frames from an FFmpeg rawvideo pipe."""
    width, height = size
    frame_bytes = width * height * 3
    command = [
        ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
        # -vsync 0 is supported by both the benchmark server's older FFmpeg
        # and current FFmpeg; it prevents timestamp-based frame duplication.
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    try:
        while True:
            payload = process.stdout.read(frame_bytes)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise RuntimeError(f"partial decoded frame from {path}: {len(payload)}/{frame_bytes} bytes")
            yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg decode failed for {path}: {stderr.strip()}")
