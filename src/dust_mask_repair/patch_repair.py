from __future__ import annotations

from typing import Any

import numpy as np

from .io import as_float32, restore_dtype


def repair_patch_match_roi(
    roi: np.ndarray,
    component_mask: np.ndarray,
    repair_mask: np.ndarray,
    *,
    search_radius: int = 64,
    patch_margin: int = 5,
    max_component_area: int = 1600,
    max_candidates: int = 2000,
    stride: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(roi)
    input_dtype = arr.dtype
    roi_float = as_float32(arr)
    output = roi_float.copy()
    component = np.asarray(component_mask, dtype=bool)
    full_repair = np.asarray(repair_mask, dtype=bool)
    area = int(np.count_nonzero(component))
    stats: dict[str, Any] = {
        "candidate_count": 0,
        "fallback": False,
        "best_score": None,
        "stride_used": int(max(1, stride)),
        "cap_exceeded": False,
    }
    if roi_float.ndim != 3 or roi_float.shape[2] not in (3, 4):
        stats["fallback"] = True
        return _restore_patch_dtype(output, input_dtype), stats
    if area == 0:
        return _restore_patch_dtype(output, input_dtype), stats
    if area > max_component_area:
        stats["fallback"] = True
        stats["cap_exceeded"] = True
        return _restore_patch_dtype(output, input_dtype), stats

    bbox = _bbox_from_mask(component)
    if bbox is None:
        return _restore_patch_dtype(output, input_dtype), stats
    x0, y0, x1, y1 = _expand_bbox(bbox, patch_margin, roi_float.shape[1], roi_float.shape[0])
    target_rgb = roi_float[y0:y1, x0:x1, :3]
    target_known = ~full_repair[y0:y1, x0:x1]
    target_component = component[y0:y1, x0:x1]
    if not target_known.any() or not target_component.any():
        stats["fallback"] = True
        return _restore_patch_dtype(output, input_dtype), stats

    win_h, win_w = target_known.shape
    search_x0, search_y0, search_x1, search_y1 = _search_bounds(
        x0,
        y0,
        x1,
        y1,
        search_radius,
        roi_float.shape[1],
        roi_float.shape[0],
    )
    best_score = float("inf")
    best_xy: tuple[int, int] | None = None
    candidate_count = 0
    stride_used = int(max(1, stride))

    for cy in range(search_y0, search_y1 - win_h + 1, stride_used):
        for cx in range(search_x0, search_x1 - win_w + 1, stride_used):
            if cx == x0 and cy == y0:
                continue
            candidate_repair = full_repair[cy : cy + win_h, cx : cx + win_w]
            if candidate_repair.any():
                continue
            candidate_rgb = roi_float[cy : cy + win_h, cx : cx + win_w, :3]
            diff = target_rgb[target_known] - candidate_rgb[target_known]
            score = float(np.mean(diff * diff)) if diff.size else float("inf")
            candidate_count += 1
            if score < best_score:
                best_score = score
                best_xy = (cx, cy)
            if candidate_count >= max_candidates:
                stats["cap_exceeded"] = True
                break
        if candidate_count >= max_candidates:
            break

    stats["candidate_count"] = candidate_count
    if best_xy is None:
        stats["fallback"] = True
        return _restore_patch_dtype(output, input_dtype), stats

    best_x, best_y = best_xy
    best_window = roi_float[best_y : best_y + win_h, best_x : best_x + win_w, :3]
    target_out = output[y0:y1, x0:x1, :3]
    target_out[target_component] = best_window[target_component]
    if roi_float.shape[2] == 4:
        output[:, :, 3] = roi_float[:, :, 3]
    stats["best_score"] = best_score
    return _restore_patch_dtype(output, input_dtype), stats


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    margin: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(width, x1 + margin),
        min(height, y1 + margin),
    )


def _search_bounds(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, x0 - radius),
        max(0, y0 - radius),
        min(width, x1 + radius),
        min(height, y1 + radius),
    )


def _restore_patch_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(np.dtype(dtype), np.floating):
        return values.astype(dtype, copy=False)
    return restore_dtype(values, dtype)


__all__ = ["repair_patch_match_roi"]
