from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import REGIONS


def load_bboxes(path: str | Path) -> dict[int, list[tuple[int, int, int, int]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"BBox annotation must be a list: {path}")
    output: dict[int, list[tuple[int, int, int, int]]] = {}
    empty_frames: set[int] = set()
    for item in data:
        frame_index = int(item["frame"])
        x0, y0 = int(item["x"]), int(item["y"])
        width, height = int(item["width"]), int(item["height"])
        if x0 == 0 and y0 == 0 and width == 0 and height == 0:
            if output.get(frame_index):
                raise ValueError(
                    f"empty and non-empty BBoxes coexist at frame {frame_index}: {item}"
                )
            empty_frames.add(frame_index)
            output.setdefault(frame_index, [])
            continue
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid non-empty BBox at frame {frame_index}: {item}")
        if frame_index in empty_frames:
            raise ValueError(
                f"empty and non-empty BBoxes coexist at frame {frame_index}: {item}"
            )
        output.setdefault(frame_index, []).append((x0, y0, x0 + width, y0 + height))
    return output


def rasterize_bboxes(
    boxes: list[tuple[int, int, int, int]] | None,
    width: int,
    height: int,
) -> np.ndarray:
    result = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes or []:
        left, right = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
        top, bottom = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
        if right > left and bottom > top:
            result[top:bottom, left:right] = True
    return result


def glyph_mask(frame: np.ndarray, target_size: tuple[int, int], threshold: int = 127) -> np.ndarray:
    if (frame.shape[1], frame.shape[0]) != target_size:
        import cv2

        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
    if frame.ndim == 3:
        # Rendered masks are achromatic; max avoids unnecessary color conversion
        # and preserves any positive mask channel exactly.
        frame = frame.max(axis=2)
    return frame > threshold


def make_regions(
    width: int,
    height: int,
    bbox: np.ndarray,
    glyph: np.ndarray,
    max_glyph_outside_bbox_pixels: int = 0,
) -> dict[str, np.ndarray]:
    expected_shape = (height, width)
    if bbox.shape != expected_shape or glyph.shape != expected_shape:
        raise ValueError(f"mask shape mismatch: bbox={bbox.shape}, glyph={glyph.shape}, expected={expected_shape}")
    outside_pixels = int((glyph & ~bbox).sum())
    if outside_pixels > max_glyph_outside_bbox_pixels:
        raise ValueError(
            f"glyph/BBox inconsistency: {outside_pixels} glyph pixels lie outside BBox "
            f"(allowed {max_glyph_outside_bbox_pixels})"
        )
    # A nonzero tolerance is diagnostic only. Evaluation still clips those few
    # pixels so that the three atomic regions remain disjoint and exhaustive.
    subtitle = glyph & bbox
    # ``bbox_bg`` is retained as a stable serialization key. Semantically it
    # is BBox-minus-glyph (a mixed local non-glyph support), not guaranteed
    # untouched background: the glyph mask may omit rendered effects.
    bbox_bg = bbox & ~subtitle
    outside = ~bbox
    overall = np.ones(expected_shape, dtype=bool)
    result = {
        "overall": overall,
        "bbox": bbox,
        "subtitle": subtitle,
        "bbox_bg": bbox_bg,
        "outside": outside,
    }
    if set(result) != set(REGIONS):
        raise AssertionError("region names do not match protocol")
    if (subtitle & bbox_bg).any() or (subtitle & outside).any() or (bbox_bg & outside).any():
        raise AssertionError("atomic regions overlap")
    if not np.array_equal(subtitle | bbox_bg | outside, overall):
        raise AssertionError("atomic regions are not exhaustive")
    return result


def support_box(mask: np.ndarray, margin: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + 1 + margin),
        min(height, int(ys.max()) + 1 + margin),
    )


def interior_mask(shape: tuple[int, int], radius: int) -> np.ndarray:
    height, width = shape
    result = np.ones(shape, dtype=bool)
    if radius <= 0:
        return result
    if height <= 2 * radius or width <= 2 * radius:
        return np.zeros(shape, dtype=bool)
    result[:radius] = False
    result[-radius:] = False
    result[:, :radius] = False
    result[:, -radius:] = False
    return result
