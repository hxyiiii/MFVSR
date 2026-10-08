from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    model: str
    video_id: str
    split: str
    input_path: Path
    gt_path: Path
    glyph_mask_path: Path
    bbox_json_path: Path
    info_path: Path
    prediction_path: Path


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    fps_numerator: int
    fps_denominator: int
    frame_count: int
    duration_seconds: float
    codec: str | None = None
    pix_fmt: str | None = None
