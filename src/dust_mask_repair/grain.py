from __future__ import annotations

from typing import Any

import numpy as np

from .io import as_float32, restore_dtype
from .mask import dilate_mask


def reinject_grain_roi(
    original_roi: np.ndarray,
    candidate_roi: np.ndarray,
    component_mask: np.ndarray,
    repair_mask: np.ndarray,
    *,
    label: int = 0,
    strength: float = 0.25,
    context_radius: int = 8,
    blur_radius: int = 1,
    min_context_pixels: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_arr = np.asarray(candidate_roi)
    input_dtype = candidate_arr.dtype
    original = as_float32(np.asarray(original_roi))
    candidate = as_float32(candidate_arr)
    output = candidate.copy()
    component = np.asarray(component_mask, dtype=bool)
    full_repair = np.asarray(repair_mask, dtype=bool)
    stats: dict[str, Any] = {
        "applied": False,
        "pixel_count": int(np.count_nonzero(component)),
        "skipped_no_context": False,
    }
    if strength <= 0.0 or not component.any():
        return _restore_grain_dtype(output, input_dtype), stats
    if original.ndim != 3 or candidate.ndim != 3 or original.shape[2] < 3 or candidate.shape[2] < 3:
        stats["skipped_no_context"] = True
        return _restore_grain_dtype(output, input_dtype), stats

    context = dilate_mask(component, context_radius) & ~full_repair
    context_count = int(np.count_nonzero(context))
    if context_count < min_context_pixels:
        stats["skipped_no_context"] = True
        return _restore_grain_dtype(output, input_dtype), stats

    original_rgb = original[:, :, :3]
    low = _box_blur_image(original_rgb, blur_radius)
    residual = original_rgb - low
    residual_samples = residual[context]
    if residual_samples.size == 0:
        stats["skipped_no_context"] = True
        return _restore_grain_dtype(output, input_dtype), stats

    context_values = original_rgb[context]
    median = np.median(context_values, axis=0)
    mad = np.median(np.abs(context_values - median), axis=0)
    lower = median - np.maximum(4.0 * mad, 0.02)
    upper = median + np.maximum(4.0 * mad, 0.02)

    ys, xs = np.nonzero(component)
    for y, x in zip(ys, xs):
        index = _grain_index(int(y), int(x), int(label), len(residual_samples))
        output[y, x, :3] = output[y, x, :3] + residual_samples[index] * float(strength)
    output[component, :3] = np.clip(output[component, :3], lower, upper)
    output[:, :, :3] = np.clip(output[:, :, :3], 0.0, 1.0)
    if output.shape[2] == 4:
        output[:, :, 3] = candidate[:, :, 3]
    stats["applied"] = True
    return _restore_grain_dtype(output, input_dtype), stats


def _grain_index(y: int, x: int, label: int, count: int) -> int:
    if count <= 0:
        return 0
    return int(((y * 73856093) ^ (x * 19349663) ^ (label * 83492791)) % count)


def _box_blur_image(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.astype(np.float32, copy=False)
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    result = np.zeros_like(image, dtype=np.float32)
    count = 0
    size = radius * 2 + 1
    for dy in range(size):
        for dx in range(size):
            result += padded[dy : dy + image.shape[0], dx : dx + image.shape[1], :]
            count += 1
    return result / float(count)


def _restore_grain_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(np.dtype(dtype), np.floating):
        return values.astype(dtype, copy=False)
    return restore_dtype(values, dtype)


__all__ = ["reinject_grain_roi"]
