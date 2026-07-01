from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from .config import RepairConfig
from .io import as_float32, restore_dtype
from .local_repair import repair_small_local_roi
from .mask import Component, bbox_with_padding, connected_components, dilate_mask

FREQUENCY_REPAIR_VERSION = 1
FREQUENCY_PATTERN_VALUES = {
    "smooth_gradient": 72,
    "directional": 144,
    "textured": 216,
    "ambiguous": 32,
}


@dataclass(frozen=True)
class FrequencySelection:
    enabled: bool
    scope_mask: np.ndarray | None
    selected_core_mask: np.ndarray
    selected_region_count: int = 0
    selected_component_count: int = 0
    selected_core_pixel_count: int = 0
    cap_exceeded_count: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class FrequencyRepairResult:
    image: np.ndarray
    fallback: bool
    stats: dict[str, Any]
    debug: dict[str, Any]


def normalize_frequency_scope_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported frequency scope mask shape: {arr.shape}")
    if arr.shape[:2] != tuple(shape):
        raise ValueError(
            "image and frequency scope mask dimensions differ: "
            f"image={shape[1]}x{shape[0]}, "
            f"frequency_scope_mask={arr.shape[1]}x{arr.shape[0]}"
        )
    if arr.ndim == 3:
        values = np.max(arr, axis=2)
    else:
        values = arr
    return np.asarray(values > 0, dtype=bool)


def prepare_frequency_selection(
    core_mask: np.ndarray,
    frequency_scope_mask: np.ndarray | None,
    config: RepairConfig,
) -> FrequencySelection:
    core = np.asarray(core_mask, dtype=bool)
    selected = np.zeros(core.shape, dtype=bool)
    if not bool(config.frequency_guided_enabled) or frequency_scope_mask is None:
        return FrequencySelection(False, None, selected)

    scope = normalize_frequency_scope_mask(frequency_scope_mask, core.shape)
    if not scope.any():
        return FrequencySelection(True, scope, selected)

    region_labels, regions = connected_components(scope)
    max_regions = int(config.frequency_max_selected_regions)
    records: list[dict[str, Any]] = []
    cap_exceeded = 0
    visited_core = np.zeros(core.shape, dtype=bool)
    component_count = 0
    selected_pixels = 0

    for region_index, region in enumerate(regions):
        if region_index >= max_regions:
            cap_exceeded += len(regions) - max_regions
            break
        x0, y0, x1, y1 = region.bbox
        roi_w = int(x1 - x0)
        roi_h = int(y1 - y0)
        roi_pixels = roi_w * roi_h
        if roi_w > int(config.frequency_max_roi_side) or roi_h > int(config.frequency_max_roi_side) or roi_pixels > int(config.frequency_max_roi_pixels):
            cap_exceeded += 1
            records.append(
                {
                    "selection_region_label": int(region.label),
                    "bbox": [int(y0), int(x0), int(y1), int(x1)],
                    "selected": False,
                    "fallback_reason": "selection_roi_cap",
                }
            )
            continue

        region_mask = region_labels[y0:y1, x0:x1] == region.label
        seeds = np.argwhere(core[y0:y1, x0:x1] & region_mask)
        for sy, sx in seeds.tolist():
            gy = int(y0 + sy)
            gx = int(x0 + sx)
            if visited_core[gy, gx] or not core[gy, gx]:
                continue
            coords, exceeded = _flood_core_component(core, visited_core, gy, gx, int(config.frequency_max_component_area))
            if not coords:
                continue
            ys = np.asarray([item[0] for item in coords], dtype=np.int32)
            xs = np.asarray([item[1] for item in coords], dtype=np.int32)
            bbox = [int(ys.min()), int(xs.min()), int(ys.max() + 1), int(xs.max() + 1)]
            if exceeded:
                cap_exceeded += 1
                records.append(
                    {
                        "selection_region_label": int(region.label),
                        "bbox": bbox,
                        "core_area": int(len(coords)),
                        "selected": False,
                        "fallback_reason": "component_area_cap",
                    }
                )
                continue
            selected[ys, xs] = True
            component_count += 1
            selected_pixels += int(len(coords))
            records.append(
                {
                    "selection_region_label": int(region.label),
                    "bbox": bbox,
                    "core_area": int(len(coords)),
                    "selected": True,
                    "fallback_reason": "",
                }
            )

    return FrequencySelection(
        True,
        scope,
        selected,
        selected_region_count=min(len(regions), max_regions),
        selected_component_count=component_count,
        selected_core_pixel_count=selected_pixels,
        cap_exceeded_count=cap_exceeded,
        records=records,
    )


def empty_frequency_stats(selection: FrequencySelection | None = None) -> dict[str, Any]:
    enabled = bool(selection.enabled) if selection is not None else False
    scope = selection.scope_mask if selection is not None else None
    return {
        "frequency_guided_enabled": enabled,
        "frequency_scope_mask_pixel_count": int(np.count_nonzero(scope)) if scope is not None else 0,
        "frequency_selected_region_count": int(selection.selected_region_count) if selection is not None else 0,
        "frequency_selected_component_count": int(selection.selected_component_count) if selection is not None else 0,
        "frequency_selected_core_pixel_count": int(selection.selected_core_pixel_count) if selection is not None else 0,
        "frequency_analyzed_component_count": 0,
        "frequency_roi_pixel_count_total": 0,
        "frequency_roi_pixel_count_max": 0,
        "frequency_candidate_count_total": 0,
        "frequency_descriptor_time_ms": 0.0,
        "frequency_repair_time_ms": 0.0,
        "frequency_cap_exceeded_count": int(selection.cap_exceeded_count) if selection is not None else 0,
        "frequency_pattern_counts": {"smooth_gradient": 0, "directional": 0, "textured": 0, "ambiguous": 0},
        "frequency_low_confidence_count": 0,
        "frequency_no_context_count": 0,
        "frequency_fallback_count": 0,
        "frequency_fallback_reason_counts": {},
        "frequency_context_low_energy_total": 0.0,
        "frequency_context_low_energy_count": 0,
        "frequency_context_mid_energy_total": 0.0,
        "frequency_context_mid_energy_count": 0,
        "frequency_context_fine_energy_total": 0.0,
        "frequency_context_fine_energy_count": 0,
        "frequency_anisotropy_total": 0.0,
        "frequency_anisotropy_count": 0,
        "frequency_signature_distance_before_total": 0.0,
        "frequency_signature_distance_before_count": 0,
        "frequency_signature_distance_after_total": 0.0,
        "frequency_signature_distance_after_count": 0,
        "frequency_midband_transfer_pixel_count": 0,
        "frequency_fast_mode_override_count": 0,
        "frequency_fast_mode_selected_pixel_count": 0,
    }


def add_frequency_means(metrics: dict[str, Any]) -> None:
    _mean_from_total(metrics, "frequency_context_low_energy")
    _mean_from_total(metrics, "frequency_context_mid_energy")
    _mean_from_total(metrics, "frequency_context_fine_energy")
    _mean_from_total(metrics, "frequency_anisotropy")
    _mean_from_total(metrics, "frequency_signature_distance_before")
    _mean_from_total(metrics, "frequency_signature_distance_after")


def repair_frequency_guided_roi(
    roi: np.ndarray,
    core_mask: np.ndarray,
    repair_mask: np.ndarray,
    config: RepairConfig,
    *,
    defect_label: int,
) -> FrequencyRepairResult:
    started = perf_counter()
    arr = np.asarray(roi)
    input_dtype = arr.dtype
    roi_float = as_float32(arr)
    core = np.asarray(core_mask, dtype=bool)
    repair = np.asarray(repair_mask, dtype=bool)
    stats = _component_stats_template()
    debug: dict[str, Any] = {
        "frequency_guided_selected": True,
        "frequency_pattern_class": "ambiguous",
        "frequency_fallback_reason": "",
    }

    roi_pixels = int(core.size)
    stats["frequency_roi_pixel_count_total"] = roi_pixels
    stats["frequency_roi_pixel_count_max"] = roi_pixels
    if roi_pixels > int(config.frequency_max_roi_pixels) or max(core.shape) > int(config.frequency_max_roi_side):
        stats["frequency_cap_exceeded_count"] = 1
        return _frequency_fallback(roi_float, input_dtype, stats, debug, "roi_cap")
    if int(np.count_nonzero(core)) > int(config.frequency_max_component_area):
        stats["frequency_cap_exceeded_count"] = 1
        return _frequency_fallback(roi_float, input_dtype, stats, debug, "component_area_cap")

    descriptor_started = perf_counter()
    descriptor = describe_frequency_context(roi_float, core, repair, config)
    stats["frequency_descriptor_time_ms"] = _elapsed_ms(descriptor_started)
    debug.update(_descriptor_debug(descriptor))
    pattern = str(descriptor["pattern"])
    stats["frequency_pattern_counts"] = {"smooth_gradient": 0, "directional": 0, "textured": 0, "ambiguous": 0}
    stats["frequency_pattern_counts"][pattern] = 1
    stats["frequency_context_low_energy_total"] = float(descriptor["low_energy"])
    stats["frequency_context_low_energy_count"] = 1
    stats["frequency_context_mid_energy_total"] = float(descriptor["mid_energy"])
    stats["frequency_context_mid_energy_count"] = 1
    stats["frequency_context_fine_energy_total"] = float(descriptor["fine_energy"])
    stats["frequency_context_fine_energy_count"] = 1
    stats["frequency_anisotropy_total"] = float(descriptor["anisotropy"])
    stats["frequency_anisotropy_count"] = 1
    stats["frequency_analyzed_component_count"] = 1 if descriptor["valid"] else 0
    if not bool(descriptor["valid"]):
        if str(descriptor["fallback_reason"]) == "no_context":
            stats["frequency_no_context_count"] = 1
        return _frequency_fallback(roi_float, input_dtype, stats, debug, str(descriptor["fallback_reason"]))

    repair_started = perf_counter()
    candidate = None
    helper_stats: dict[str, Any] = {}
    if pattern == "smooth_gradient":
        candidate, helper_stats = _small_local_candidate(roi_float, core, repair, config, edge_guided=False)
    elif pattern == "directional":
        candidate, helper_stats = _small_local_candidate(roi_float, core, repair, config, edge_guided=True)
    elif pattern == "textured":
        candidate, helper_stats = _frequency_patch_candidate(roi_float, core, repair, descriptor, config)
        if candidate is None:
            candidate, helper_stats = _small_local_candidate(roi_float, core, repair, config, edge_guided=True)
    else:
        stats["frequency_low_confidence_count"] = 1
        return _frequency_fallback(roi_float, input_dtype, stats, debug, "ambiguous")

    stats["frequency_repair_time_ms"] = _elapsed_ms(repair_started)
    stats["frequency_candidate_count_total"] = int(helper_stats.get("candidate_count", 0))
    debug["frequency_helper_method"] = str(helper_stats.get("method", helper_stats.get("local_method", "")))
    debug["frequency_candidate_count"] = int(helper_stats.get("candidate_count", 0))
    if candidate is None:
        return _frequency_fallback(roi_float, input_dtype, stats, debug, str(helper_stats.get("fallback_reason", "candidate_fallback")))

    candidate_float = as_float32(candidate)
    if pattern in {"directional", "textured"} and float(config.frequency_midband_strength) > 0.0:
        candidate_float, transferred = _apply_midband_transfer(
            roi_float[:, :, :3],
            candidate_float[:, :, :3],
            core,
            repair,
            descriptor,
            float(config.frequency_midband_strength),
        )
        stats["frequency_midband_transfer_pixel_count"] = int(transferred)

    before_distance, after_distance = _signature_distances(roi_float[:, :, :3], candidate_float[:, :, :3], core, descriptor)
    stats["frequency_signature_distance_before_total"] = before_distance
    stats["frequency_signature_distance_before_count"] = 1
    stats["frequency_signature_distance_after_total"] = after_distance
    stats["frequency_signature_distance_after_count"] = 1
    debug["frequency_signature_distance_before"] = before_distance
    debug["frequency_signature_distance_after"] = after_distance

    out = roi_float.copy()
    out[:, :, :3] = candidate_float[:, :, :3]
    if out.shape[2] == 4:
        out[:, :, 3] = roi_float[:, :, 3]
    return FrequencyRepairResult(_restore_frequency_dtype(out, input_dtype), False, stats, debug)


def describe_frequency_context(
    roi: np.ndarray,
    core_mask: np.ndarray,
    repair_mask: np.ndarray,
    config: RepairConfig,
) -> dict[str, Any]:
    rgb = as_float32(np.asarray(roi))[:, :, :3]
    core = np.asarray(core_mask, dtype=bool)
    repair = np.asarray(repair_mask, dtype=bool)
    known = ~repair
    context = dilate_mask(core, int(config.frequency_context_radius)) & known
    context_pixels = int(np.count_nonzero(context))
    context_band = dilate_mask(core, int(config.frequency_context_radius)) & ~core
    available = context_pixels / float(max(1, int(np.count_nonzero(context_band))))
    if context_pixels < int(config.frequency_min_context_pixels) or available < float(config.frequency_min_known_fraction):
        return _invalid_descriptor("no_context", context, available)

    scales = tuple(int(max(1, s)) for s in config.frequency_scales)
    small_radius = min(scales)
    large_radius = max(scales)
    small = _normalized_box_blur_rgb(rgb, known, small_radius)
    large = _normalized_box_blur_rgb(rgb, known, large_radius)
    mid = small - large
    fine = rgb - small
    low_energy = _mean_energy(_gradient_energy(large, known, context, 1))
    mid_energy = _mean_rgb_energy(mid, context)
    fine_energy = _mean_rgb_energy(fine, context)
    direction = _directional_energies(rgb, known, context, scales)
    dir_values = np.asarray([direction[key] for key in ("horizontal", "vertical", "diag_down", "diag_up")], dtype=np.float32)
    max_dir = float(np.max(dir_values)) if dir_values.size else 0.0
    min_dir = float(np.min(dir_values)) if dir_values.size else 0.0
    anisotropy = (max_dir - min_dir) / max(max_dir + min_dir, 1.0e-8)
    dominant = max(direction, key=lambda key: direction[key]) if direction else "none"
    centroid = float((mid_energy + 2.0 * fine_energy) / max(low_energy + mid_energy + fine_energy, 1.0e-8))
    structure_energy = mid_energy + fine_energy
    if anisotropy >= float(config.frequency_anisotropy_threshold) and structure_energy >= float(config.frequency_smooth_threshold) * 0.35:
        pattern = "directional"
    elif mid_energy >= float(config.frequency_smooth_threshold) or fine_energy >= float(config.frequency_smooth_threshold) * 2.0:
        pattern = "textured"
    else:
        pattern = "smooth_gradient"
    low = large.copy()
    return {
        "valid": True,
        "fallback_reason": "",
        "pattern": pattern,
        "low_energy": float(low_energy),
        "mid_energy": float(mid_energy),
        "fine_energy": float(fine_energy),
        "directional_energies": {key: float(value) for key, value in direction.items()},
        "dominant_orientation": dominant,
        "anisotropy": float(anisotropy),
        "frequency_centroid": centroid,
        "context_known_fraction": float(available),
        "context_pixels": context_pixels,
        "context_mask": context,
        "low_frequency": low,
        "mid_band": mid,
    }


def _small_local_candidate(
    roi: np.ndarray,
    core: np.ndarray,
    repair: np.ndarray,
    config: RepairConfig,
    *,
    edge_guided: bool,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    candidate, stats = repair_small_local_roi(
        roi,
        core,
        repair,
        strategy="frequency_guided",
        context_radius=int(config.frequency_context_radius),
        use_local_plane=True,
        edge_guided_enabled=bool(edge_guided),
        edge_guided_max_component_area=int(config.frequency_max_component_area),
        edge_guided_max_roi_area=int(config.frequency_max_roi_pixels),
        edge_guided_search_radius=int(config.frequency_search_radius),
        edge_guided_min_coherence=float(config.edge_guided_min_coherence),
        edge_guided_min_gradient_energy=float(config.edge_guided_min_gradient_energy),
        edge_guided_max_total_search=int(config.frequency_candidate_cap * max(1, np.count_nonzero(core))),
        tone_guided_enabled=False,
    )
    if stats.get("fallback"):
        stats["fallback_reason"] = "small_local_fallback"
        return None, stats
    stats["candidate_count"] = int(stats.get("edge_guided_sample_count", 0))
    return candidate, stats


def _frequency_patch_candidate(
    roi: np.ndarray,
    core: np.ndarray,
    repair: np.ndarray,
    descriptor: dict[str, Any],
    config: RepairConfig,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    rgb = as_float32(roi)[:, :, :3]
    bbox = _bbox_from_mask(core)
    if bbox is None:
        return None, {"fallback_reason": "empty_component", "candidate_count": 0}
    height, width = core.shape
    x0, y0, x1, y1 = bbox_with_padding(bbox, int(config.frequency_patch_radius), width, height)
    win_h = int(y1 - y0)
    win_w = int(x1 - x0)
    if win_h <= 0 or win_w <= 0:
        return None, {"fallback_reason": "empty_window", "candidate_count": 0}
    target_known = ~repair[y0:y1, x0:x1]
    target_core = core[y0:y1, x0:x1]
    if not target_core.any() or not target_known.any():
        return None, {"fallback_reason": "no_target_context", "candidate_count": 0}

    sx0, sy0, sx1, sy1 = bbox_with_padding((x0, y0, x1, y1), int(config.frequency_search_radius), width, height)
    target_rgb = rgb[y0:y1, x0:x1]
    target_var = _patch_variance(target_rgb[target_known])
    target_dir = str(descriptor.get("dominant_orientation", "none"))
    target_mid = float(descriptor.get("mid_energy", 0.0))
    candidates: list[tuple[float, int, int]] = []
    candidate_count = 0
    for cy in range(sy0, sy1 - win_h + 1):
        for cx in range(sx0, sx1 - win_w + 1):
            if cy == y0 and cx == x0:
                continue
            if np.any(repair[cy : cy + win_h, cx : cx + win_w]):
                continue
            candidate_rgb = rgb[cy : cy + win_h, cx : cx + win_w]
            border_error = float(np.mean((target_rgb[target_known] - candidate_rgb[target_known]) ** 2))
            texture_distance = abs(_patch_variance(candidate_rgb.reshape(-1, 3)) - target_var)
            mid_distance = abs(_patch_mid_energy(candidate_rgb) - target_mid)
            orientation_distance = 0.0 if _patch_orientation(candidate_rgb) == target_dir else 1.0
            spatial = float(np.hypot(cy - y0, cx - x0) / max(1, int(config.frequency_search_radius)))
            score = (
                float(config.frequency_border_weight) * border_error
                + float(config.frequency_band_weight) * mid_distance
                + float(config.frequency_texture_weight) * texture_distance
                + float(config.frequency_orientation_weight) * orientation_distance
                + float(config.frequency_spatial_weight) * spatial
            )
            candidates.append((score, cy, cx))
            candidate_count += 1
            if candidate_count >= int(config.frequency_candidate_cap):
                break
        if candidate_count >= int(config.frequency_candidate_cap):
            break
    if not candidates:
        return None, {"fallback_reason": "no_donor", "candidate_count": 0}
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    top = candidates[: max(1, min(int(config.frequency_top_k), len(candidates)))]
    weights = np.asarray([1.0 / max(item[0], 1.0e-6) for item in top], dtype=np.float32)
    weights /= float(np.sum(weights))
    blended = np.zeros((win_h, win_w, 3), dtype=np.float32)
    for weight, (_score, cy, cx) in zip(weights, top):
        blended += rgb[cy : cy + win_h, cx : cx + win_w] * float(weight)
    output = as_float32(roi).copy()
    output[y0:y1, x0:x1, :3][target_core] = blended[target_core]
    if output.shape[2] == 4:
        output[:, :, 3] = as_float32(roi)[:, :, 3]
    return _restore_frequency_dtype(output, np.asarray(roi).dtype), {
        "method": "frequency_patch",
        "candidate_count": candidate_count,
        "best_score": float(top[0][0]),
    }


def _apply_midband_transfer(
    original_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    core: np.ndarray,
    repair: np.ndarray,
    descriptor: dict[str, Any],
    strength: float,
) -> tuple[np.ndarray, int]:
    context = np.asarray(descriptor.get("context_mask"), dtype=bool)
    mid = np.asarray(descriptor.get("mid_band"), dtype=np.float32)
    if strength <= 0.0 or not core.any() or not context.any() or mid.shape[:2] != core.shape:
        return candidate_rgb, 0
    samples = mid[context]
    if samples.size == 0:
        return candidate_rgb, 0
    donor_rms = float(np.sqrt(np.mean(np.sum(samples * samples, axis=1))))
    scale = float(np.clip(donor_rms / max(donor_rms, 1.0e-6), 0.5, 2.0))
    output = candidate_rgb.copy()
    ys, xs = np.nonzero(core)
    for y, x in zip(ys.tolist(), xs.tolist()):
        idx = int(((y * 73856093) ^ (x * 19349663)) % samples.shape[0])
        output[y, x, :3] = np.clip(output[y, x, :3] + samples[idx] * float(strength) * scale, 0.0, 1.0)
    return output.astype(np.float32, copy=False), int(len(xs))


def _signature_distances(original_rgb: np.ndarray, candidate_rgb: np.ndarray, core: np.ndarray, descriptor: dict[str, Any]) -> tuple[float, float]:
    low = np.asarray(descriptor.get("low_frequency"), dtype=np.float32)
    if low.shape[:2] != core.shape or not core.any():
        return 0.0, 0.0
    before = float(np.mean(np.abs(original_rgb[core, :3] - low[core, :3])))
    after = float(np.mean(np.abs(candidate_rgb[core, :3] - low[core, :3])))
    return before, after


def _directional_energies(rgb: np.ndarray, known: np.ndarray, context: np.ndarray, scales: tuple[int, ...]) -> dict[str, float]:
    directions = {
        "horizontal": (0, 1),
        "vertical": (1, 0),
        "diag_down": (1, 1),
        "diag_up": (-1, 1),
    }
    out: dict[str, float] = {}
    for name, (dy, dx) in directions.items():
        values = [_finite_difference_energy(rgb, known, context, dy * scale, dx * scale) for scale in scales]
        out[name] = float(np.mean(values)) if values else 0.0
    return out


def _finite_difference_energy(rgb: np.ndarray, known: np.ndarray, context: np.ndarray, dy: int, dx: int) -> float:
    if dy == 0 and dx == 0:
        return 0.0
    h, w = known.shape
    y0a = max(0, -dy)
    y1a = min(h, h - dy)
    x0a = max(0, -dx)
    x1a = min(w, w - dx)
    y0b = y0a + dy
    y1b = y1a + dy
    x0b = x0a + dx
    x1b = x1a + dx
    if y0a >= y1a or x0a >= x1a:
        return 0.0
    valid = known[y0a:y1a, x0a:x1a] & known[y0b:y1b, x0b:x1b]
    valid &= context[y0a:y1a, x0a:x1a] | context[y0b:y1b, x0b:x1b]
    if not np.any(valid):
        return 0.0
    diff = rgb[y0b:y1b, x0b:x1b, :3] - rgb[y0a:y1a, x0a:x1a, :3]
    return float(np.mean(np.sum(diff[valid] * diff[valid], axis=1)))


def _gradient_energy(rgb: np.ndarray, known: np.ndarray, context: np.ndarray, scale: int) -> np.ndarray:
    return np.asarray(
        [
            _finite_difference_energy(rgb, known, context, 0, scale),
            _finite_difference_energy(rgb, known, context, scale, 0),
        ],
        dtype=np.float32,
    )


def _normalized_box_blur_rgb(rgb: np.ndarray, known: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return rgb.astype(np.float32, copy=False)
    known_f = known.astype(np.float32)
    denom = np.maximum(_box_sum_2d(known_f, radius), 1.0e-6)
    out = np.zeros_like(rgb[:, :, :3], dtype=np.float32)
    for ch in range(3):
        out[:, :, ch] = _box_sum_2d(rgb[:, :, ch] * known_f, radius) / denom
    return out


def _box_sum_2d(values: np.ndarray, radius: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    r = int(max(0, radius))
    padded = np.pad(arr, ((r, r), (r, r)), mode="constant", constant_values=0.0)
    integ = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    size = 2 * r + 1
    return (
        integ[size:, size:]
        - integ[:-size, size:]
        - integ[size:, :-size]
        + integ[:-size, :-size]
    ).astype(np.float32)


def _patch_variance(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.var(arr, axis=0)))


def _patch_mid_energy(patch: np.ndarray) -> float:
    arr = np.asarray(patch, dtype=np.float32)
    if min(arr.shape[:2]) < 3:
        return 0.0
    known = np.ones(arr.shape[:2], dtype=bool)
    small = _normalized_box_blur_rgb(arr, known, 1)
    large = _normalized_box_blur_rgb(arr, known, 2)
    return _mean_rgb_energy(small - large, np.ones(arr.shape[:2], dtype=bool))


def _patch_orientation(patch: np.ndarray) -> str:
    arr = np.asarray(patch, dtype=np.float32)
    known = np.ones(arr.shape[:2], dtype=bool)
    context = known.copy()
    energies = _directional_energies(arr, known, context, (1,))
    return max(energies, key=lambda key: energies[key]) if energies else "none"


def _mean_rgb_energy(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    sample = values[mask, :3]
    return float(np.mean(np.sum(sample * sample, axis=1))) if sample.size else 0.0


def _mean_energy(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(np.mean(arr)) if arr.size else 0.0


def _invalid_descriptor(reason: str, context: np.ndarray | None = None, available: float = 0.0) -> dict[str, Any]:
    return {
        "valid": False,
        "fallback_reason": reason,
        "pattern": "ambiguous",
        "low_energy": 0.0,
        "mid_energy": 0.0,
        "fine_energy": 0.0,
        "directional_energies": {"horizontal": 0.0, "vertical": 0.0, "diag_down": 0.0, "diag_up": 0.0},
        "dominant_orientation": "none",
        "anisotropy": 0.0,
        "frequency_centroid": 0.0,
        "context_known_fraction": float(available),
        "context_pixels": int(np.count_nonzero(context)) if context is not None else 0,
        "context_mask": context if context is not None else np.zeros((0, 0), dtype=bool),
    }


def _component_stats_template() -> dict[str, Any]:
    stats = empty_frequency_stats(None)
    stats["frequency_guided_enabled"] = True
    return stats


def _frequency_fallback(
    roi_float: np.ndarray,
    input_dtype: np.dtype,
    stats: dict[str, Any],
    debug: dict[str, Any],
    reason: str,
) -> FrequencyRepairResult:
    stats["frequency_fallback_count"] = 1
    reasons = stats.setdefault("frequency_fallback_reason_counts", {})
    reasons[str(reason)] = int(reasons.get(str(reason), 0)) + 1
    if reason in {"ambiguous", "low_confidence"}:
        stats["frequency_low_confidence_count"] = 1
    if reason == "no_context":
        stats["frequency_no_context_count"] = 1
    debug["frequency_fallback_reason"] = str(reason)
    return FrequencyRepairResult(_restore_frequency_dtype(roi_float, input_dtype), True, stats, debug)


def _descriptor_debug(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        "frequency_pattern_class": str(descriptor.get("pattern", "ambiguous")),
        "frequency_low_energy": float(descriptor.get("low_energy", 0.0)),
        "frequency_mid_energy": float(descriptor.get("mid_energy", 0.0)),
        "frequency_fine_energy": float(descriptor.get("fine_energy", 0.0)),
        "frequency_dominant_orientation": str(descriptor.get("dominant_orientation", "none")),
        "frequency_anisotropy": float(descriptor.get("anisotropy", 0.0)),
        "frequency_centroid": float(descriptor.get("frequency_centroid", 0.0)),
        "frequency_context_known_fraction": float(descriptor.get("context_known_fraction", 0.0)),
        "frequency_context_pixels": int(descriptor.get("context_pixels", 0)),
    }


def _flood_core_component(
    core: np.ndarray,
    visited: np.ndarray,
    start_y: int,
    start_x: int,
    max_area: int,
) -> tuple[list[tuple[int, int]], bool]:
    h, w = core.shape
    stack = [(int(start_y), int(start_x))]
    visited[start_y, start_x] = True
    coords: list[tuple[int, int]] = []
    exceeded = False
    while stack:
        y, x = stack.pop()
        coords.append((y, x))
        if len(coords) > max_area:
            exceeded = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or not core[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
    return coords, exceeded


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _restore_frequency_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(np.dtype(dtype), np.floating):
        return values.astype(dtype, copy=False)
    return restore_dtype(values, dtype)


def _mean_from_total(metrics: dict[str, Any], prefix: str) -> None:
    count = int(metrics.get(f"{prefix}_count", 0))
    total = float(metrics.get(f"{prefix}_total", 0.0))
    metrics[f"{prefix}_mean"] = total / float(count) if count > 0 else 0.0


def _elapsed_ms(started: float) -> float:
    return float((perf_counter() - started) * 1000.0)


__all__ = [
    "FREQUENCY_PATTERN_VALUES",
    "FREQUENCY_REPAIR_VERSION",
    "FrequencyRepairResult",
    "FrequencySelection",
    "add_frequency_means",
    "describe_frequency_context",
    "empty_frequency_stats",
    "normalize_frequency_scope_mask",
    "prepare_frequency_selection",
    "repair_frequency_guided_roi",
]
