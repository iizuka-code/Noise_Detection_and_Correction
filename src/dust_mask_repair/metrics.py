from __future__ import annotations

import numpy as np

from .io import as_float32


def compute_repair_metrics(
    original: np.ndarray,
    repaired: np.ndarray,
    soft_mask: np.ndarray,
    changed_bboxes: list[tuple[int, int, int, int]],
    processing_time_ms: float,
) -> dict[str, float | int]:
    original_float = as_float32(original)
    repaired_float = as_float32(repaired)
    diff = np.abs(repaired_float - original_float)
    if diff.ndim == 3:
        per_pixel_diff = np.max(diff, axis=2)
    else:
        per_pixel_diff = diff

    inside = np.asarray(soft_mask, dtype=np.float32) > 0.0
    outside = ~inside
    changed = per_pixel_diff > 0.0

    return {
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_bbox_count": int(len(changed_bboxes)),
        "max_abs_diff_outside_mask": _max_or_zero(per_pixel_diff[outside]),
        "mean_abs_diff_inside_mask": _mean_or_zero(diff[inside]),
        "mean_abs_diff_outside_mask": _mean_or_zero(diff[outside]),
        "processing_time_ms": float(processing_time_ms),
    }


def _mean_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _max_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))
