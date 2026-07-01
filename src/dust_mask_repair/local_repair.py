from __future__ import annotations

from typing import Any

import numpy as np

from .io import as_float32, restore_dtype
from .mask import dilate_mask


def repair_small_local_roi(
    roi: np.ndarray,
    component_mask: np.ndarray,
    repair_mask: np.ndarray,
    *,
    strategy: str,
    context_radius: int = 4,
    use_local_plane: bool = True,
    edge_guided_enabled: bool = True,
    edge_guided_max_component_area: int = 64,
    edge_guided_max_roi_area: int = 4096,
    edge_guided_search_radius: int = 8,
    edge_guided_min_coherence: float = 0.35,
    edge_guided_min_gradient_energy: float = 2.5e-4,
    edge_guided_max_total_search: int = 4096,
    tone_guided_enabled: bool = True,
    tone_guided_max_component_area: int = 64,
    tone_guided_max_roi_area: int = 4096,
    tone_guided_context_radius: int = 6,
    tone_guided_search_radius: int = 10,
    tone_guided_patch_radius: int = 1,
    tone_guided_candidate_cap: int = 256,
    tone_guided_top_k: int = 5,
    tone_guided_tone_weight: float = 2.0,
    tone_guided_spatial_weight: float = 0.35,
    tone_guided_texture_weight: float = 0.25,
    tone_guided_gradient_weight: float = 0.15,
    tone_guided_min_context_pixels: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(roi)
    input_dtype = arr.dtype
    roi_float = as_float32(arr)
    output = roi_float.copy()
    component = np.asarray(component_mask, dtype=bool)
    unknown = component
    full_repair = np.asarray(repair_mask, dtype=bool)
    stats: dict[str, Any] = {
        "strategy": strategy,
        "method": "fallback",
        "component_pixels": int(np.count_nonzero(component)),
        "context_pixels": 0,
        "fallback": False,
        "edge_guided_enabled": bool(edge_guided_enabled),
        "edge_guided_used": False,
        "edge_guided_pixel_count": 0,
        "edge_guided_fallback": False,
        "edge_guided_low_confidence": False,
        "edge_guided_coherence": 0.0,
        "edge_guided_gradient_energy": 0.0,
        "edge_guided_sample_count": 0,
        "edge_guided_fallback_reason": "not_attempted",
        "tone_guided_component_count": 0,
        "tone_guided_pixel_count": 0,
        "tone_guided_fallback_count": 0,
        "tone_guided_no_context_count": 0,
        "tone_guided_candidate_count_total": 0,
        "tone_guided_top_k_total": 0,
        "tone_guided_top_k_count": 0,
        "tone_guided_score_total": 0.0,
        "tone_guided_score_count": 0,
        "tone_guided_context_rgb_distance_total": 0.0,
        "tone_guided_context_rgb_distance_count": 0,
    }
    if roi_float.ndim != 3 or roi_float.shape[2] not in (3, 4):
        stats["fallback"] = True
        return _restore_local_dtype(output, input_dtype), stats
    if not unknown.any():
        stats["method"] = "none"
        return _restore_local_dtype(output, input_dtype), stats

    context = _context_ring(component, full_repair, context_radius)
    context_count = int(np.count_nonzero(context))
    stats["context_pixels"] = context_count
    if context_count < 3:
        stats["fallback"] = True
        return _restore_local_dtype(output, input_dtype), stats

    rgb = roi_float[:, :, :3]
    candidate = None
    if use_local_plane:
        candidate = _fit_small_local_plane(rgb, unknown, context)
    if candidate is not None:
        fallback_candidate = candidate
        fallback_method = "plane"
    else:
        fallback_candidate = _median_fill(rgb, unknown, context)
        fallback_method = "median"
    stats["fallback_method"] = fallback_method

    edge_candidate, edge_stats = _edge_guided_local_candidate(
        rgb,
        unknown,
        full_repair,
        context,
        fallback_candidate,
        enabled=edge_guided_enabled,
        max_component_area=edge_guided_max_component_area,
        max_roi_area=edge_guided_max_roi_area,
        search_radius=edge_guided_search_radius,
        min_coherence=edge_guided_min_coherence,
        min_gradient_energy=edge_guided_min_gradient_energy,
        max_total_search=edge_guided_max_total_search,
    )
    stats.update(edge_stats)
    candidate = edge_candidate if edge_candidate is not None else fallback_candidate
    method_used = "edge_guided" if edge_candidate is not None else fallback_method
    tone_candidate, tone_stats = _tone_guided_local_candidate(
        rgb,
        unknown,
        full_repair,
        context,
        candidate,
        fallback_candidate,
        enabled=tone_guided_enabled,
        max_component_area=tone_guided_max_component_area,
        max_roi_area=tone_guided_max_roi_area,
        context_radius=tone_guided_context_radius,
        search_radius=tone_guided_search_radius,
        patch_radius=tone_guided_patch_radius,
        candidate_cap=tone_guided_candidate_cap,
        top_k=tone_guided_top_k,
        tone_weight=tone_guided_tone_weight,
        spatial_weight=tone_guided_spatial_weight,
        texture_weight=tone_guided_texture_weight,
        gradient_weight=tone_guided_gradient_weight,
        min_context_pixels=tone_guided_min_context_pixels,
        edge_guided_used=edge_candidate is not None,
    )
    stats.update(tone_stats)
    if tone_candidate is not None:
        output[:, :, :3] = tone_candidate
        stats["method"] = "tone_guided" if method_used != "edge_guided" else "edge_guided"
    else:
        output[:, :, :3] = candidate
        stats["method"] = method_used

    output[:, :, :3] = _clamp_unknown_to_context(output[:, :, :3], unknown, context)
    if roi_float.shape[2] == 4:
        output[:, :, 3] = roi_float[:, :, 3]
    return _restore_local_dtype(output, input_dtype), stats


def _edge_guided_local_candidate(
    rgb: np.ndarray,
    unknown_mask: np.ndarray,
    repair_mask: np.ndarray,
    context_mask: np.ndarray,
    fallback_candidate: np.ndarray,
    *,
    enabled: bool,
    max_component_area: int,
    max_roi_area: int,
    search_radius: int,
    min_coherence: float,
    min_gradient_energy: float,
    max_total_search: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    stats: dict[str, Any] = {
        "edge_guided_used": False,
        "edge_guided_pixel_count": 0,
        "edge_guided_fallback": False,
        "edge_guided_low_confidence": False,
        "edge_guided_coherence": 0.0,
        "edge_guided_gradient_energy": 0.0,
        "edge_guided_sample_count": 0,
        "edge_guided_fallback_reason": "",
    }
    component_area = int(np.count_nonzero(unknown_mask))
    if not enabled:
        stats["edge_guided_fallback_reason"] = "disabled"
        return None, stats
    if component_area <= 0:
        stats["edge_guided_fallback_reason"] = "empty_component"
        return None, stats
    if component_area > max_component_area:
        stats["edge_guided_fallback"] = True
        stats["edge_guided_fallback_reason"] = "component_area_cap"
        return None, stats
    if int(unknown_mask.size) > max_roi_area:
        stats["edge_guided_fallback"] = True
        stats["edge_guided_fallback_reason"] = "roi_area_cap"
        return None, stats
    if search_radius <= 0 or max_total_search <= 0:
        stats["edge_guided_fallback"] = True
        stats["edge_guided_fallback_reason"] = "search_cap"
        return None, stats

    known_mask = ~np.asarray(repair_mask, dtype=bool)
    tensor = _rgb_structure_tensor(rgb, known_mask, context_mask)
    stats["edge_guided_sample_count"] = int(tensor["sample_count"])
    stats["edge_guided_coherence"] = float(tensor["coherence"])
    stats["edge_guided_gradient_energy"] = float(tensor["gradient_energy"])
    if tensor["sample_count"] < 3:
        stats["edge_guided_fallback"] = True
        stats["edge_guided_low_confidence"] = True
        stats["edge_guided_fallback_reason"] = "insufficient_gradient_samples"
        return None, stats
    if tensor["gradient_energy"] < min_gradient_energy:
        stats["edge_guided_fallback"] = True
        stats["edge_guided_low_confidence"] = True
        stats["edge_guided_fallback_reason"] = "low_gradient_energy"
        return None, stats
    if tensor["coherence"] < min_coherence:
        stats["edge_guided_fallback"] = True
        stats["edge_guided_low_confidence"] = True
        stats["edge_guided_fallback_reason"] = "low_coherence"
        return None, stats

    edge_values, filled_mask, budget_exhausted = _fill_along_isophote(
        rgb,
        unknown_mask,
        known_mask,
        tangent=(float(tensor["tangent_x"]), float(tensor["tangent_y"])),
        search_radius=int(search_radius),
        max_total_search=int(max_total_search),
    )
    filled_count = int(np.count_nonzero(filled_mask & unknown_mask))
    stats["edge_guided_pixel_count"] = filled_count
    if filled_count == 0:
        stats["edge_guided_fallback"] = True
        stats["edge_guided_fallback_reason"] = "search_budget_exhausted" if budget_exhausted else "no_bilateral_candidates"
        return None, stats

    candidate = fallback_candidate.copy()
    candidate[filled_mask] = edge_values[filled_mask]
    stats["edge_guided_used"] = True
    stats["edge_guided_fallback"] = filled_count < component_area
    if filled_count < component_area:
        stats["edge_guided_fallback_reason"] = (
            "search_budget_exhausted" if budget_exhausted else "partial_bilateral_candidates"
        )
    else:
        stats["edge_guided_fallback_reason"] = ""
    return candidate.astype(np.float32, copy=False), stats


def _rgb_structure_tensor(
    rgb: np.ndarray,
    known_mask: np.ndarray,
    context_mask: np.ndarray,
) -> dict[str, float | int]:
    height, width = known_mask.shape
    if height < 3 or width < 3:
        return _empty_tensor_stats()

    valid = np.asarray(context_mask, dtype=bool) & np.asarray(known_mask, dtype=bool)
    valid[0, :] = False
    valid[-1, :] = False
    valid[:, 0] = False
    valid[:, -1] = False
    valid[1:-1, 1:-1] &= (
        known_mask[1:-1, :-2]
        & known_mask[1:-1, 2:]
        & known_mask[:-2, 1:-1]
        & known_mask[2:, 1:-1]
    )
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return _empty_tensor_stats()

    gx = (rgb[ys, xs + 1, :3] - rgb[ys, xs - 1, :3]) * 0.5
    gy = (rgb[ys + 1, xs, :3] - rgb[ys - 1, xs, :3]) * 0.5
    gxx = np.sum(gx * gx, axis=1)
    gyy = np.sum(gy * gy, axis=1)
    gxy = np.sum(gx * gy, axis=1)
    jxx = float(np.sum(gxx))
    jyy = float(np.sum(gyy))
    jxy = float(np.sum(gxy))
    sample_count = int(len(xs))
    trace = jxx + jyy
    if trace <= 1.0e-12:
        return {
            "sample_count": sample_count,
            "coherence": 0.0,
            "gradient_energy": 0.0,
            "tangent_x": 1.0,
            "tangent_y": 0.0,
        }
    delta = float(np.sqrt(max((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy, 0.0)))
    lambda_max = max(0.0, (trace + delta) * 0.5)
    lambda_min = max(0.0, (trace - delta) * 0.5)
    coherence = (lambda_max - lambda_min) / (lambda_max + lambda_min + 1.0e-12)
    if abs(jxy) > 1.0e-12 or abs(lambda_max - jxx) > 1.0e-12:
        gradient_x = float(lambda_max - jyy)
        gradient_y = float(jxy)
    elif jxx >= jyy:
        gradient_x, gradient_y = 1.0, 0.0
    else:
        gradient_x, gradient_y = 0.0, 1.0
    norm = float(np.hypot(gradient_x, gradient_y))
    if norm <= 1.0e-12:
        gradient_x, gradient_y = 1.0, 0.0
        norm = 1.0
    gradient_x /= norm
    gradient_y /= norm
    tangent_x = -gradient_y
    tangent_y = gradient_x
    tangent_norm = float(np.hypot(tangent_x, tangent_y))
    if tangent_norm <= 1.0e-12:
        tangent_x, tangent_y = 1.0, 0.0
    else:
        tangent_x /= tangent_norm
        tangent_y /= tangent_norm
    return {
        "sample_count": sample_count,
        "coherence": float(coherence),
        "gradient_energy": float(trace / max(sample_count, 1)),
        "tangent_x": float(tangent_x),
        "tangent_y": float(tangent_y),
    }


def _empty_tensor_stats() -> dict[str, float | int]:
    return {
        "sample_count": 0,
        "coherence": 0.0,
        "gradient_energy": 0.0,
        "tangent_x": 1.0,
        "tangent_y": 0.0,
    }


def _fill_along_isophote(
    rgb: np.ndarray,
    unknown_mask: np.ndarray,
    known_mask: np.ndarray,
    *,
    tangent: tuple[float, float],
    search_radius: int,
    max_total_search: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    out = np.zeros_like(rgb, dtype=np.float32)
    filled = np.zeros(unknown_mask.shape, dtype=bool)
    total_evaluated = 0
    budget_exhausted = False
    dy = float(tangent[1])
    dx = float(tangent[0])
    ys, xs = np.nonzero(unknown_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if total_evaluated >= max_total_search:
            budget_exhausted = True
            break
        seen: set[tuple[int, int]] = set()
        positive, positive_distance, evaluated = _find_known_along_direction(
            rgb,
            known_mask,
            y,
            x,
            dy=dy,
            dx=dx,
            sign=1.0,
            search_radius=search_radius,
            seen=seen,
            remaining=max_total_search - total_evaluated,
        )
        total_evaluated += evaluated
        if total_evaluated >= max_total_search:
            budget_exhausted = True
            continue
        negative, negative_distance, evaluated = _find_known_along_direction(
            rgb,
            known_mask,
            y,
            x,
            dy=dy,
            dx=dx,
            sign=-1.0,
            search_radius=search_radius,
            seen=seen,
            remaining=max_total_search - total_evaluated,
        )
        total_evaluated += evaluated
        if positive is None or negative is None:
            continue
        denominator = positive_distance + negative_distance
        if denominator <= 1.0e-6:
            continue
        out[y, x] = ((positive * negative_distance) + (negative * positive_distance)) / denominator
        filled[y, x] = True
    return out, filled, budget_exhausted


def _find_known_along_direction(
    rgb: np.ndarray,
    known_mask: np.ndarray,
    y: int,
    x: int,
    *,
    dy: float,
    dx: float,
    sign: float,
    search_radius: int,
    seen: set[tuple[int, int]],
    remaining: int,
) -> tuple[np.ndarray | None, float, int]:
    height, width = known_mask.shape
    evaluated = 0
    for step in range(1, int(search_radius) + 1):
        if evaluated >= remaining:
            break
        yy = int(np.rint(y + (dy * sign * step)))
        xx = int(np.rint(x + (dx * sign * step)))
        coord = (yy, xx)
        if coord in seen:
            continue
        seen.add(coord)
        if yy < 0 or yy >= height or xx < 0 or xx >= width:
            evaluated += 1
            continue
        evaluated += 1
        if known_mask[yy, xx]:
            return rgb[yy, xx, :3].astype(np.float32, copy=True), float(step), evaluated
    return None, 0.0, evaluated

def repair_fast_inpaint_roi(
    roi: np.ndarray,
    repair_mask: np.ndarray,
    *,
    max_iterations: int = 80,
    min_known_weight: float = 1.0e-6,
    preserve_alpha: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(roi)
    input_dtype = arr.dtype
    roi_float = as_float32(arr)
    output = roi_float.copy()
    unknown = np.asarray(repair_mask, dtype=bool)
    known = ~unknown
    stats: dict[str, Any] = {
        "iterations": 0,
        "pixel_count": int(np.count_nonzero(unknown)),
        "fallback": False,
        "init_method": "none",
    }
    if roi_float.ndim != 3 or roi_float.shape[2] not in (3, 4):
        stats["fallback"] = True
        return _restore_local_dtype(output, input_dtype), stats
    if not unknown.any():
        return _restore_local_dtype(output, input_dtype), stats
    if not known.any():
        stats["fallback"] = True
        return _restore_local_dtype(output, input_dtype), stats

    rgb = roi_float[:, :, :3]
    current = rgb.copy()
    context = _context_ring(unknown, unknown, radius=6)
    sample_mask = context if np.count_nonzero(context) >= 3 else known
    plane = _fit_small_local_plane(rgb, unknown, sample_mask)
    if plane is not None:
        current[unknown] = plane[unknown]
        stats["init_method"] = "plane"
    else:
        median = np.median(rgb[sample_mask], axis=0)
        current[unknown] = median
        stats["init_method"] = "median"

    iteration_cap = int(min(max_iterations, max(8, 2 * max(unknown.shape))))
    for iteration in range(iteration_cap):
        neighbor_sum = np.zeros_like(current, dtype=np.float32)
        neighbor_count = np.zeros((*unknown.shape, 1), dtype=np.float32)
        padded_current = np.pad(current, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_weight = np.pad(np.ones(unknown.shape, dtype=np.float32), 1, mode="constant", constant_values=0.0)
        for oy in range(3):
            for ox in range(3):
                if oy == 1 and ox == 1:
                    continue
                weight = padded_weight[oy : oy + unknown.shape[0], ox : ox + unknown.shape[1]]
                neighbor_sum += padded_current[oy : oy + unknown.shape[0], ox : ox + unknown.shape[1]] * weight[:, :, None]
                neighbor_count += weight[:, :, None]
        next_values = neighbor_sum / np.maximum(neighbor_count, float(min_known_weight))
        current[unknown] = next_values[unknown]
        current[known] = rgb[known]
        stats["iterations"] = iteration + 1

    output[:, :, :3] = current
    if preserve_alpha and roi_float.shape[2] == 4:
        output[:, :, 3] = roi_float[:, :, 3]
    return _restore_local_dtype(output, input_dtype), stats


def _context_ring(component_mask: np.ndarray, repair_mask: np.ndarray, radius: int) -> np.ndarray:
    expanded = dilate_mask(component_mask, max(1, int(radius)))
    return expanded & ~repair_mask


def _fit_small_local_plane(
    rgb: np.ndarray,
    unknown_mask: np.ndarray,
    context_mask: np.ndarray,
) -> np.ndarray | None:
    ys, xs = np.nonzero(context_mask)
    if len(xs) < 6:
        return None

    height, width = rgb.shape[:2]
    x_center = (width - 1) / 2.0
    y_center = (height - 1) / 2.0
    x_scale = max(width - 1, 1)
    y_scale = max(height - 1, 1)
    design = np.column_stack(
        (
            (xs.astype(np.float32) - x_center) / x_scale,
            (ys.astype(np.float32) - y_center) / y_scale,
            np.ones(len(xs), dtype=np.float32),
        )
    )
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    full_design = np.stack(
        (
            (grid_x - x_center) / x_scale,
            (grid_y - y_center) / y_scale,
            np.ones((height, width), dtype=np.float32),
        ),
        axis=2,
    )

    out = rgb.copy()
    for channel in range(3):
        values = rgb[ys, xs, channel]
        inliers = _robust_inliers(values)
        if np.count_nonzero(inliers) < 3:
            inliers = np.ones(values.shape, dtype=bool)
        try:
            coeffs, *_ = np.linalg.lstsq(design[inliers], values[inliers], rcond=None)
        except np.linalg.LinAlgError:
            return None
        out[:, :, channel] = np.sum(full_design * coeffs.reshape(1, 1, 3), axis=2)
    result = rgb.copy()
    result[unknown_mask] = out[unknown_mask]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _median_fill(rgb: np.ndarray, unknown_mask: np.ndarray, context_mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    median = np.median(rgb[context_mask], axis=0)
    out[unknown_mask] = median
    return out


def _clamp_unknown_to_context(rgb: np.ndarray, unknown_mask: np.ndarray, context_mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    sample = rgb[context_mask]
    if sample.size == 0:
        return out
    median = np.median(sample, axis=0)
    mad = np.median(np.abs(sample - median), axis=0)
    sample_min = np.min(sample, axis=0)
    sample_max = np.max(sample, axis=0)
    lower = np.where(mad > 1.0e-6, median - (4.0 * mad), sample_min)
    upper = np.where(mad > 1.0e-6, median + (4.0 * mad), sample_max)
    lower = np.minimum(lower, sample_min)
    upper = np.maximum(upper, sample_max)
    out[unknown_mask] = np.clip(out[unknown_mask], lower, upper)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _robust_inliers(values: np.ndarray) -> np.ndarray:
    if values.size < 8:
        return np.ones(values.shape, dtype=bool)
    low = np.percentile(values, 5.0)
    high = np.percentile(values, 95.0)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    margin = max(0.02, float(mad) * 4.0)
    return (values >= low) & (values <= high) & (np.abs(values - median) <= margin)


def _tone_guided_local_candidate(
    rgb: np.ndarray,
    unknown_mask: np.ndarray,
    repair_mask: np.ndarray,
    context_mask: np.ndarray,
    current_candidate: np.ndarray,
    expected_candidate: np.ndarray,
    *,
    enabled: bool,
    max_component_area: int,
    max_roi_area: int,
    context_radius: int,
    search_radius: int,
    patch_radius: int,
    candidate_cap: int,
    top_k: int,
    tone_weight: float,
    spatial_weight: float,
    texture_weight: float,
    gradient_weight: float,
    min_context_pixels: int,
    edge_guided_used: bool,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    stats: dict[str, Any] = {
        "tone_guided_component_count": 0,
        "tone_guided_pixel_count": 0,
        "tone_guided_fallback_count": 0,
        "tone_guided_no_context_count": 0,
        "tone_guided_candidate_count_total": 0,
        "tone_guided_top_k_total": 0,
        "tone_guided_top_k_count": 0,
        "tone_guided_score_total": 0.0,
        "tone_guided_score_count": 0,
        "tone_guided_context_rgb_distance_total": 0.0,
        "tone_guided_context_rgb_distance_count": 0,
    }
    unknown = np.asarray(unknown_mask, dtype=bool)
    repair = np.asarray(repair_mask, dtype=bool)
    context = np.asarray(context_mask, dtype=bool) & ~repair
    component_area = int(np.count_nonzero(unknown))
    if not enabled or edge_guided_used or component_area <= 0:
        return None, stats
    if component_area > int(max_component_area) or int(unknown.size) > int(max_roi_area):
        stats["tone_guided_fallback_count"] = 1
        return None, stats
    if int(search_radius) <= 0 or int(candidate_cap) <= 0 or int(top_k) <= 0:
        stats["tone_guided_fallback_count"] = 1
        return None, stats

    expanded_context = _context_ring(unknown, repair, max(context_radius, 1))
    context = (context | expanded_context) & ~repair
    if int(np.count_nonzero(context)) < int(min_context_pixels):
        stats["tone_guided_no_context_count"] = 1
        stats["tone_guided_fallback_count"] = 1
        return None, stats


    if _context_plane_residual(rgb, context) < 0.006:
        stats["tone_guided_fallback_count"] = 1
        return None, stats

    donor_mask = _valid_donor_mask(context, repair, int(patch_radius))
    donor_ys, donor_xs = np.nonzero(donor_mask)
    if donor_ys.size < int(min_context_pixels):
        stats["tone_guided_no_context_count"] = 1
        stats["tone_guided_fallback_count"] = 1
        return None, stats

    component_ys, component_xs = np.nonzero(unknown)
    cy = float(np.mean(component_ys))
    cx = float(np.mean(component_xs))
    center_distance = (donor_ys.astype(np.float32) - cy) ** 2 + (donor_xs.astype(np.float32) - cx) ** 2
    order = np.argsort(center_distance, kind="mergesort")
    if order.size > int(candidate_cap):
        order = order[: int(candidate_cap)]
    donor_ys = donor_ys[order]
    donor_xs = donor_xs[order]
    if donor_ys.size == 0:
        stats["tone_guided_no_context_count"] = 1
        stats["tone_guided_fallback_count"] = 1
        return None, stats

    patch_means, patch_stds, donor_residuals = _donor_patch_descriptors(rgb, donor_ys, donor_xs, int(patch_radius))
    context_std = np.std(rgb[context], axis=0).astype(np.float32)
    expected = np.asarray(expected_candidate, dtype=np.float32)
    current = np.asarray(current_candidate, dtype=np.float32)
    expected_gradient = _gradient_vectors(expected)
    donor_gradient = _gradient_vectors(rgb)
    output = current.copy()

    used_pixels = 0
    total_candidates = 0
    total_top_k = 0
    score_total = 0.0
    tone_distance_total = 0.0
    search_radius_f = float(max(1, int(search_radius)))
    search_radius_sq = search_radius_f * search_radius_f

    for y, x in zip(component_ys.tolist(), component_xs.tolist()):
        dy = donor_ys.astype(np.float32) - float(y)
        dx = donor_xs.astype(np.float32) - float(x)
        spatial_sq = dy * dy + dx * dx
        local = np.flatnonzero(spatial_sq <= search_radius_sq)
        if local.size == 0:
            local = np.arange(donor_ys.size, dtype=np.int64)
        if local.size > int(candidate_cap):
            local_order = np.argsort(spatial_sq[local], kind="mergesort")[: int(candidate_cap)]
            local = local[local_order]

        expected_rgb = expected[y, x, :3]
        tone_distance = np.mean(np.abs(patch_means[local] - expected_rgb.reshape(1, 3)), axis=1)
        spatial_distance = np.sqrt(spatial_sq[local]) / search_radius_f
        texture_distance = np.mean(np.abs(patch_stds[local] - context_std.reshape(1, 3)), axis=1)
        gradient_distance = _gradient_direction_distance(
            expected_gradient[y, x],
            donor_gradient[donor_ys[local], donor_xs[local]],
        )
        scores = (
            float(spatial_weight) * spatial_distance
            + float(tone_weight) * tone_distance
            + float(texture_weight) * texture_distance
            + float(gradient_weight) * gradient_distance
        )
        rank = np.argsort(scores, kind="mergesort")
        chosen = local[rank[: max(1, min(int(top_k), rank.size))]]
        chosen_scores = scores[rank[: chosen.size]]
        weights = 1.0 / np.maximum(chosen_scores, 1.0e-4)
        weights = weights / np.sum(weights)
        low_frequency = np.sum(patch_means[chosen] * weights[:, None], axis=0)
        residual = np.sum(donor_residuals[chosen] * weights[:, None], axis=0) * 0.25
        output[y, x, :3] = np.clip(low_frequency + residual, 0.0, 1.0)

        used_pixels += 1
        total_candidates += int(local.size)
        total_top_k += int(chosen.size)
        score_total += float(np.mean(chosen_scores))
        tone_distance_total += float(np.mean(tone_distance[rank[: chosen.size]]))

    if used_pixels == 0:
        stats["tone_guided_fallback_count"] = 1
        return None, stats

    stats["tone_guided_component_count"] = 1
    stats["tone_guided_pixel_count"] = int(used_pixels)
    stats["tone_guided_candidate_count_total"] = int(total_candidates)
    stats["tone_guided_top_k_total"] = int(total_top_k)
    stats["tone_guided_top_k_count"] = int(used_pixels)
    stats["tone_guided_score_total"] = float(score_total)
    stats["tone_guided_score_count"] = int(used_pixels)
    stats["tone_guided_context_rgb_distance_total"] = float(tone_distance_total)
    stats["tone_guided_context_rgb_distance_count"] = int(used_pixels)
    return output.astype(np.float32, copy=False), stats



def _context_plane_residual(rgb: np.ndarray, context_mask: np.ndarray) -> float:
    ys, xs = np.nonzero(context_mask)
    if len(xs) < 6:
        return 0.0
    height, width = rgb.shape[:2]
    x_center = (width - 1) / 2.0
    y_center = (height - 1) / 2.0
    x_scale = max(width - 1, 1)
    y_scale = max(height - 1, 1)
    design = np.column_stack(
        (
            (xs.astype(np.float32) - x_center) / x_scale,
            (ys.astype(np.float32) - y_center) / y_scale,
            np.ones(len(xs), dtype=np.float32),
        )
    )
    residuals: list[float] = []
    for channel in range(3):
        values = rgb[ys, xs, channel]
        inliers = _robust_inliers(values)
        if np.count_nonzero(inliers) < 3:
            inliers = np.ones(values.shape, dtype=bool)
        try:
            coeffs, *_ = np.linalg.lstsq(design[inliers], values[inliers], rcond=None)
        except np.linalg.LinAlgError:
            return 0.0
        predicted = design @ coeffs
        residuals.append(float(np.median(np.abs(values - predicted))))
    return float(np.mean(residuals))

def _valid_donor_mask(context_mask: np.ndarray, repair_mask: np.ndarray, patch_radius: int) -> np.ndarray:
    context = np.asarray(context_mask, dtype=bool)
    repair = np.asarray(repair_mask, dtype=bool)
    valid = context & ~repair
    if patch_radius <= 0:
        return valid
    height, width = valid.shape
    out = np.zeros_like(valid, dtype=bool)
    known = ~repair
    for y in range(patch_radius, height - patch_radius):
        for x in range(patch_radius, width - patch_radius):
            if not valid[y, x]:
                continue
            patch_known = known[y - patch_radius : y + patch_radius + 1, x - patch_radius : x + patch_radius + 1]
            if bool(np.all(patch_known)):
                out[y, x] = True
    return out


def _donor_patch_descriptors(
    rgb: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    patch_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.zeros((ys.size, 3), dtype=np.float32)
    stds = np.zeros((ys.size, 3), dtype=np.float32)
    residuals = np.zeros((ys.size, 3), dtype=np.float32)
    for idx, (y, x) in enumerate(zip(ys.tolist(), xs.tolist())):
        if patch_radius > 0:
            patch = rgb[y - patch_radius : y + patch_radius + 1, x - patch_radius : x + patch_radius + 1, :3]
        else:
            patch = rgb[y : y + 1, x : x + 1, :3]
        flat = patch.reshape(-1, 3)
        mean = np.mean(flat, axis=0).astype(np.float32)
        means[idx] = mean
        stds[idx] = np.std(flat, axis=0).astype(np.float32)
        residuals[idx] = rgb[y, x, :3] - mean
    return means, stds, residuals


def _gradient_vectors(rgb: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(rgb[:, :, :3], dtype=np.float32)
    gy = np.zeros_like(rgb[:, :, :3], dtype=np.float32)
    gx[:, 1:-1] = (rgb[:, 2:, :3] - rgb[:, :-2, :3]) * 0.5
    gy[1:-1, :] = (rgb[2:, :, :3] - rgb[:-2, :, :3]) * 0.5
    return np.concatenate([gx, gy], axis=2).astype(np.float32)


def _gradient_direction_distance(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float32).reshape(1, -1)
    cand = np.asarray(candidates, dtype=np.float32).reshape(candidates.shape[0], -1)
    ref_norm = np.linalg.norm(ref, axis=1)
    cand_norm = np.linalg.norm(cand, axis=1)
    valid = (ref_norm[0] > 1.0e-6) & (cand_norm > 1.0e-6)
    out = np.zeros(cand.shape[0], dtype=np.float32)
    if np.any(valid):
        dot = np.sum(cand[valid] * ref, axis=1) / np.maximum(cand_norm[valid] * ref_norm[0], 1.0e-6)
        out[valid] = 1.0 - np.abs(np.clip(dot, -1.0, 1.0))
    return out
def _restore_local_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(np.dtype(dtype), np.floating):
        return values.astype(dtype, copy=False)
    return restore_dtype(values, dtype)


__all__ = ["repair_fast_inpaint_roi", "repair_small_local_roi"]

