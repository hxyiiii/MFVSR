from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryResult:
    prediction: np.ndarray
    valid_mask: np.ndarray
    metadata: dict


class FullFrameAdapter:
    policy = "full_frame_resize_to_gt"

    def __init__(self, raw_size: tuple[int, int], gt_size: tuple[int, int]):
        self.raw_size = raw_size
        self.gt_size = gt_size

    def map(self, frame: np.ndarray) -> GeometryResult:
        target_width, target_height = self.gt_size
        if (frame.shape[1], frame.shape[0]) == self.gt_size:
            prediction = frame
            interpolation = "none"
        else:
            import cv2

            prediction = cv2.resize(frame, self.gt_size, interpolation=cv2.INTER_CUBIC)
            interpolation = "INTER_CUBIC"
        valid = np.ones((target_height, target_width), dtype=bool)
        return GeometryResult(prediction, valid, {
            "policy": self.policy,
            "interpolation": interpolation,
            "raw_resolution": list(self.raw_size),
            "gt_resolution": list(self.gt_size),
            "valid_gt_pixels": int(valid.sum()),
        })


class ClearCenterCropInverseAdapter(FullFrameAdapter):
    policy = "clear_center_crop_inverse_to_gt"

    def __init__(self, info: dict, raw_size: tuple[int, int], gt_size: tuple[int, int]):
        super().__init__(raw_size, gt_size)
        raw_width, raw_height = raw_size
        gt_width, gt_height = gt_size
        processing = (info.get("model_processing") or {}).get("processing_resolution")
        if processing:
            processing_size = tuple(map(int, processing.lower().split("x")))
            if processing_size != raw_size:
                raise ValueError(f"CLEAR processing/raw mismatch: {processing_size} != {raw_size}")
        self.scale = max(raw_width / gt_width, raw_height / gt_height)
        self.scaled_width = round(gt_width * self.scale)
        self.scaled_height = round(gt_height * self.scale)
        self.left = int(round((self.scaled_width - raw_width) / 2.0))
        self.top = int(round((self.scaled_height - raw_height) / 2.0))
        yy, xx = np.indices((gt_height, gt_width), dtype=np.float32)
        self.map_x = (xx + 0.5) * self.scale - 0.5 - self.left
        self.map_y = (yy + 0.5) * self.scale - 0.5 - self.top
        self.valid = (
            (self.map_x >= 0)
            & (self.map_x <= raw_width - 1)
            & (self.map_y >= 0)
            & (self.map_y <= raw_height - 1)
        )

    def map(self, frame: np.ndarray) -> GeometryResult:
        import cv2

        prediction = cv2.remap(
            frame,
            self.map_x,
            self.map_y,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return GeometryResult(prediction, self.valid.copy(), {
            "policy": self.policy,
            "interpolation": "INTER_CUBIC",
            "raw_resolution": list(self.raw_size),
            "gt_resolution": list(self.gt_size),
            "scale_to_cover": self.scale,
            "scaled_resolution": [self.scaled_width, self.scaled_height],
            "center_crop_offset": [self.left, self.top],
            "valid_gt_pixels": int(self.valid.sum()),
        })


def make_adapter(model: str, config: dict, info: dict, raw_size: tuple[int, int], gt_size: tuple[int, int]):
    policy = config["models"][model]["geometry"]
    if policy == "full_frame_resize_to_gt":
        return FullFrameAdapter(raw_size, gt_size)
    if policy == "clear_center_crop_inverse":
        return ClearCenterCropInverseAdapter(info, raw_size, gt_size)
    raise ValueError(f"unsupported geometry policy: {policy}")


def validate_temporal_mapping(model: str, config: dict, info: dict, raw_count: int, gt_count: int) -> dict:
    policy = config["models"][model]["temporal"]
    if model == "effecterase":
        params = ((info.get("model_processing") or {}).get("model_specific_params") or {})
        if int(params.get("frame_interval", 1)) != 1:
            raise ValueError("EffectErase frame_interval != 1 has no one-to-one GT mapping")
    if policy == "exact" and raw_count != gt_count:
        raise ValueError(f"{model} requires exact frame count: raw={raw_count}, gt={gt_count}")
    if policy == "verified_prefix" and raw_count > gt_count:
        raise ValueError(f"verified-prefix output is longer than GT: raw={raw_count}, gt={gt_count}")
    evaluated = min(raw_count, gt_count)
    if evaluated <= 0:
        raise ValueError("no evaluable prediction frames")
    return {
        "policy": policy,
        "raw_decoded_frames": raw_count,
        "gt_decoded_frames": gt_count,
        "evaluated_frame_count": evaluated,
        "evaluated_frame_range": [0, evaluated - 1],
        "missing_prediction_frame_count": gt_count - evaluated,
        "temporal_coverage": evaluated / gt_count,
    }
