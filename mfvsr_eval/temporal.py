from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalMaps:
    prediction_warp_error: np.ndarray
    target_warp_error: np.ndarray
    change_error: np.ndarray
    warp_valid: np.ndarray
    change_valid: np.ndarray
    warped_previous_regions: dict[str, np.ndarray]


def temporal_change_map(
    previous_prediction: np.ndarray,
    prediction: np.ndarray,
    previous_target: np.ndarray,
    target: np.ndarray,
    previous_valid: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Official flow-free temporal change residual used by TC.

    This is exactly the ``change_error`` branch historically returned by
    :func:`temporal_maps`; separating it prevents TC-only profiles from paying
    for RAFT without changing the metric definition.
    """
    pred_float = prediction.astype(np.float32) / 255.0
    target_float = target.astype(np.float32) / 255.0
    previous_pred_float = previous_prediction.astype(np.float32) / 255.0
    previous_target_float = previous_target.astype(np.float32) / 255.0
    change_error = np.abs(
        (pred_float - previous_pred_float) - (target_float - previous_target_float)
    ).mean(axis=2)
    return change_error, valid & previous_valid


class TorchvisionRAFT:
    """Torchvision RAFT-Large flow estimator using released default weights."""

    def __init__(
        self,
        device: str,
        variant: str = "large",
        max_input_long_side: int = 0,
        resize_trigger_long_side: int = 0,
    ):
        import torch
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        if variant == "large":
            self.weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=self.weights, progress=False).to(device).eval()
        elif variant == "small":
            self.weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=self.weights, progress=False).to(device).eval()
        else:
            raise ValueError(f"unknown RAFT variant: {variant}")
        self.device = device
        self.max_input_long_side = max_input_long_side
        self.resize_trigger_long_side = resize_trigger_long_side
        self.torch = torch
        self.transforms = self.weights.transforms()

    def input_size(self, height: int, width: int) -> tuple[int, int]:
        limit = int(getattr(self, "max_input_long_side", 0) or 0)
        trigger = int(getattr(self, "resize_trigger_long_side", 0) or 0)
        if limit <= 0 or (trigger > 0 and max(height, width) <= trigger):
            return height, width
        if max(height, width) <= limit:
            return height, width
        scale = limit / max(height, width)
        return max(1, round(height * scale)), max(1, round(width * scale))

    @staticmethod
    def _pad(array, multiple: int = 8):
        import torch.nn.functional as functional

        height, width = array.shape[-2:]
        pad_height = (-height) % multiple
        pad_width = (-width) % multiple
        return functional.pad(array, (0, pad_width, 0, pad_height), mode="replicate"), height, width

    def _tensor(self, image: np.ndarray):
        # Keep uint8 here. Torchvision's released RAFT weight transform performs
        # the official dtype conversion to [0, 1] before normalization to [-1, 1].
        tensor = self.torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None]
        return tensor.to(self.device)

    def flow(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        if first.shape != second.shape:
            raise ValueError(f"RAFT frame shape mismatch: {first.shape} != {second.shape}")
        original_height, original_width = first.shape[:2]
        height, width = self.input_size(original_height, original_width)
        first_tensor = self._tensor(first)
        second_tensor = self._tensor(second)
        if (height, width) != (original_height, original_width):
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.functional import resize

            first_tensor = resize(
                first_tensor, [height, width], InterpolationMode.BILINEAR, antialias=False
            )
            second_tensor = resize(
                second_tensor, [height, width], InterpolationMode.BILINEAR, antialias=False
            )
        first_tensor, _, _ = self._pad(first_tensor)
        second_tensor, _, _ = self._pad(second_tensor)
        first_tensor, second_tensor = self.transforms(first_tensor, second_tensor)
        predictions = None
        final_flow = None
        try:
            with self.torch.inference_mode():
                predictions = self.model(first_tensor, second_tensor)
                final_flow = predictions[-1][0, :, :height, :width]
                if (height, width) != (original_height, original_width):
                    import torch.nn.functional as functional

                    final_flow = functional.interpolate(
                        final_flow[None],
                        size=(original_height, original_width),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    final_flow[0] *= original_width / width
                    final_flow[1] *= original_height / height
                # The CPU copy synchronizes the selected final flow before the
                # model-owned correlation pyramid is released below.
                result = final_flow.permute(1, 2, 0).float().cpu().numpy()
        finally:
            # Torchvision CorrBlock retains its full all-pairs pyramid on the
            # module after forward() returns. Releasing it here prevents the
            # forward volume from remaining live while backward flow allocates
            # another volume, and prevents it leaking into the next video.
            corr_block = getattr(self.model, "corr_block", None)
            if corr_block is not None:
                corr_block.corr_pyramid = [self.torch.tensor(0)]
            del final_flow
            del predictions
            del first_tensor
            del second_tensor
        return result


class FarnebackFlow:
    """Deterministic CPU fallback; not used by the paper's primary protocol."""

    def flow(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        import cv2

        first_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
        second_gray = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
        return cv2.calcOpticalFlowFarneback(first_gray, second_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def make_flow_estimator(config: dict, device: str, runtime_config: dict | None = None):
    backend = config.get("backend", "raft_large")
    max_input_long_side = int(
        (runtime_config or {}).get("raft_max_input_long_side", 0) or 0
    )
    resize_trigger_long_side = int(
        (runtime_config or {}).get("raft_resize_trigger_long_side", 0) or 0
    )
    if backend == "raft_large":
        return TorchvisionRAFT(
            device, "large", max_input_long_side, resize_trigger_long_side
        )
    if backend == "raft_small":
        return TorchvisionRAFT(
            device, "small", max_input_long_side, resize_trigger_long_side
        )
    if backend == "farneback":
        return FarnebackFlow()
    raise ValueError(f"unsupported optical-flow backend: {backend}")


def _remap(array: np.ndarray, backward_flow: np.ndarray, interpolation: int, border_value=0):
    import cv2

    height, width = backward_flow.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    map_x = xx + backward_flow[..., 0]
    map_y = yy + backward_flow[..., 1]
    warped = cv2.remap(
        array,
        map_x,
        map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    in_bounds = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    return warped, in_bounds, map_x, map_y


def temporal_maps(
    previous_prediction: np.ndarray,
    prediction: np.ndarray,
    previous_target: np.ndarray,
    target: np.ndarray,
    previous_valid: np.ndarray,
    valid: np.ndarray,
    previous_regions: dict[str, np.ndarray],
    flow_estimator,
    occlusion_alpha: float = 0.01,
    occlusion_beta: float = 0.5,
) -> TemporalMaps:
    """Compute target-flow warping MSE, its target baseline, and change error."""
    import cv2

    forward = flow_estimator.flow(previous_target, target)
    backward = flow_estimator.flow(target, previous_target)
    warped_forward, in_bounds, map_x, map_y = _remap(forward, backward, cv2.INTER_LINEAR)
    consistency = np.square(backward + warped_forward).sum(axis=2)
    magnitude = np.square(backward).sum(axis=2) + np.square(warped_forward).sum(axis=2)
    non_occluded = consistency <= occlusion_alpha * magnitude + occlusion_beta

    warped_prediction, _, _, _ = _remap(previous_prediction, backward, cv2.INTER_LINEAR)
    warped_target, _, _, _ = _remap(previous_target, backward, cv2.INTER_LINEAR)
    warped_valid, _, _, _ = _remap(previous_valid.astype(np.uint8), backward, cv2.INTER_NEAREST)
    warp_valid = valid & warped_valid.astype(bool) & in_bounds & non_occluded
    change_error, change_valid = temporal_change_map(
        previous_prediction, prediction, previous_target, target,
        previous_valid, valid,
    )

    pred_float = prediction.astype(np.float32) / 255.0
    target_float = target.astype(np.float32) / 255.0
    warped_pred_float = warped_prediction.astype(np.float32) / 255.0
    warped_target_float = warped_target.astype(np.float32) / 255.0
    pred_error = np.square(pred_float - warped_pred_float).mean(axis=2)
    target_error = np.square(target_float - warped_target_float).mean(axis=2)
    warped_regions = {
        name: _remap(mask.astype(np.uint8), backward, cv2.INTER_NEAREST)[0].astype(bool)
        for name, mask in previous_regions.items()
    }
    return TemporalMaps(
        prediction_warp_error=pred_error,
        target_warp_error=target_error,
        change_error=change_error,
        warp_valid=warp_valid,
        change_valid=change_valid,
        warped_previous_regions=warped_regions,
    )
