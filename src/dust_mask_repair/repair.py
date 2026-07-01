from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import RepairConfig
from .defects import DefectFeatures, classify_defects, defect_debug_payload, summarize_defect_features
from .grain import reinject_grain_roi
from .frequency_repair import (
    FREQUENCY_PATTERN_VALUES,
    add_frequency_means,
    empty_frequency_stats,
    prepare_frequency_selection,
    repair_frequency_guided_roi,
)
from .io import as_float32, restore_dtype, write_image
from .local_repair import repair_fast_inpaint_roi, repair_small_local_roi
from .mask import (
    bbox_with_padding,
    bboxes_from_mask,
    connected_components,
    dilate_mask,
    filter_components,
    normalize_mask,
    threshold_mask,
)
from .metrics import compute_repair_metrics
from .patch_repair import repair_patch_match_roi

_KL_BINS_PER_CHANNEL = 8
_KL_BIN_COUNT = _KL_BINS_PER_CHANNEL**3
_KL_EPSILON = 1.0e-9
_DEFECT_AWARE_VERSION = 1
_DEFECT_AWARE_FALLBACK_METHOD = "adaptive"
_DIRECTIONAL_MAX_PIXELS = 512


@dataclass
class RepairResult:
    repaired_image: np.ndarray
    binary_mask: np.ndarray
    soft_mask: np.ndarray
    changed_bbox_list: list[tuple[int, int, int, int]]
    metrics: dict[str, Any]
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    debug_paths: dict[str, str] = field(default_factory=dict)
    core_mask: np.ndarray | None = None
    repair_mask: np.ndarray | None = None
    blend_alpha: np.ndarray | None = None
    guard_rejected_mask: np.ndarray | None = None
    defect_component_repairs: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class _RoiRepairResult:
    image: np.ndarray
    candidate_before_guard: np.ndarray
    rejected_mask: np.ndarray
    alpha_scale: float
    low_confidence: bool
    defect_stats: dict[str, Any] = field(default_factory=dict)
    defect_debug: dict[str, Any] = field(default_factory=dict)


def repair_image(
    image: np.ndarray,
    mask: np.ndarray,
    config: RepairConfig | None = None,
    frequency_scope_mask: np.ndarray | None = None,
) -> RepairResult:
    cfg = config or RepairConfig()
    cfg.validate()
    start = time.perf_counter()

    original = np.asarray(image)
    if original.ndim != 3 or original.shape[2] not in (3, 4):
        raise ValueError("image must be an RGB or RGBA array")

    normalized = normalize_mask(mask, cfg.mask_channel)
    normalized_mask = normalized.values
    if normalized_mask.shape != original.shape[:2]:
        raise ValueError(
            "image and mask dimensions differ: "
            f"image={original.shape[1]}x{original.shape[0]}, "
            f"mask={normalized_mask.shape[1]}x{normalized_mask.shape[0]}"
        )

    thresholded = threshold_mask(normalized_mask, cfg.threshold)
    filtered = filter_components(
        thresholded,
        min_area=cfg.min_component_area,
        max_area=cfg.max_component_area,
    )
    core_mask = filtered.mask
    repair_mask = dilate_mask(core_mask, cfg.dilate_radius)
    blend_alpha = _make_blend_alpha(core_mask, repair_mask, cfg.feather_radius)
    changed_bbox_list = bboxes_from_mask(blend_alpha > 0.0)
    collect_debug_images = bool(cfg.collect_debug_images or cfg.debug_dir is not None)
    frequency_selection = (
        prepare_frequency_selection(core_mask, frequency_scope_mask, cfg)
        if cfg.method == "defect_aware"
        else None
    )
    frequency_pattern_map = np.zeros(core_mask.shape, dtype=np.uint8)
    defect_labels = None
    defect_components = None
    defect_features: list[DefectFeatures] | None = None
    if cfg.method == "defect_aware":
        defect_labels, defect_components = connected_components(core_mask)
        defect_features = classify_defects(
            original,
            core_mask,
            labels=defect_labels,
            components=defect_components,
            repair_mask=repair_mask,
        )

    image_float = as_float32(original)
    candidate_before_guard = image_float.copy()
    candidate_after_guard = image_float.copy()
    guard_rejected_mask = np.zeros(core_mask.shape, dtype=bool)
    component_alpha_map = np.ones(core_mask.shape, dtype=np.float32)
    component_alpha_values: list[float] = []
    low_confidence_component_count = 0
    defect_repair_stats = _empty_defect_repair_stats() if cfg.method == "defect_aware" else {}
    if defect_repair_stats:
        defect_repair_stats.update(empty_frequency_stats(frequency_selection))
        defect_repair_stats["grain_reinject_enabled"] = cfg.grain_reinject_strength > 0.0
        defect_repair_stats["grain_reinject_strength"] = float(cfg.grain_reinject_strength)

    if cfg.strength == 0.0 or not repair_mask.any():
        repaired = original.copy()
        elapsed = (time.perf_counter() - start) * 1000.0
        metrics = _repair_metrics(
            original=original,
            repaired=repaired,
            blend_alpha=blend_alpha,
            core_mask=core_mask,
            repair_mask=repair_mask,
            guard_rejected_mask=guard_rejected_mask,
            changed_bbox_list=changed_bbox_list,
            processing_time_ms=elapsed,
            channel_used=normalized.channel_used,
            filtered=filtered,
            component_alpha_values=component_alpha_values,
            low_confidence_component_count=low_confidence_component_count,
        )
        _add_defect_aware_metrics(
            metrics,
            cfg.method,
            skipped=True,
            defect_features=defect_features,
            defect_repair_stats=defect_repair_stats,
        )
        _add_defect_aware_blend_metrics(
            metrics,
            cfg.method,
            final_alpha=np.clip(blend_alpha * cfg.strength, 0.0, 1.0),
            core_mask=core_mask,
            repair_mask=repair_mask,
        )
        result = RepairResult(
            repaired_image=repaired,
            binary_mask=repair_mask,
            soft_mask=blend_alpha,
            changed_bbox_list=changed_bbox_list,
            metrics=metrics,
            debug_images=_debug_images(
                original=original,
                repaired=repaired,
                normalized_mask=normalized_mask,
                core_mask=core_mask,
                repair_mask=repair_mask,
                blend_alpha=blend_alpha,
                candidate_before_guard=candidate_before_guard,
                candidate_after_guard=candidate_after_guard,
                guard_rejected_mask=guard_rejected_mask,
                output_dtype=original.dtype,
            )
            if collect_debug_images
            else {},
            core_mask=core_mask,
            repair_mask=repair_mask,
            blend_alpha=blend_alpha,
            guard_rejected_mask=guard_rejected_mask,
        )
        if collect_debug_images and frequency_selection is not None:
            result.debug_images.update(_frequency_debug_images(original, frequency_selection, frequency_pattern_map))
        _write_debug_outputs(cfg.debug_dir, result)
        _write_defect_debug_outputs(cfg.debug_dir, result, defect_features)
        _write_frequency_debug_outputs(cfg.debug_dir, result)
        return result

    repair_candidate = image_float.copy()
    if defect_labels is not None and defect_components is not None:
        core_labels, core_components = defect_labels, defect_components
    else:
        core_labels, core_components = connected_components(core_mask)
    height, width = core_mask.shape
    defect_feature_by_label = {feature.label: feature for feature in defect_features or []}
    defect_component_repairs: dict[int, dict[str, Any]] = {}

    for component in core_components:
        component_core = core_labels == component.label
        component_repair = dilate_mask(component_core, cfg.dilate_radius)
        component_bbox = _bbox_from_mask(component_repair)
        if component_bbox is None:
            continue
        x0, y0, x1, y1 = bbox_with_padding(component_bbox, cfg.padding, width, height)
        roi_slice = (slice(y0, y1), slice(x0, x1))
        original_roi = image_float[y0:y1, x0:x1, :3]
        core_roi = component_core[roi_slice]
        repair_roi_mask = component_repair[roi_slice]
        defect_strategy = None
        defect_feature = defect_feature_by_label.get(component.label)
        if defect_feature is not None:
            defect_strategy = defect_feature.recommended_strategy
        frequency_selected = bool(
            frequency_selection is not None
            and frequency_selection.enabled
            and np.any(component_core & frequency_selection.selected_core_mask)
        )

        roi_result = _repair_roi(
            original_roi,
            repair_roi_mask,
            cfg.method,
            component.area,
            core_roi,
            defect_strategy,
            component.label,
            cfg.grain_reinject_strength,
            cfg.grain_context_radius,
            cfg.grain_blur_radius,
            cfg.grain_min_context_pixels,
            cfg.color_match_strength,
            cfg.color_match_radius,
            cfg.color_match_min_context_pixels,
            cfg.edge_guided_enabled,
            cfg.edge_guided_max_component_area,
            cfg.edge_guided_context_radius,
            cfg.edge_guided_search_radius,
            cfg.edge_guided_min_coherence,
            cfg.edge_guided_min_gradient_energy,
            cfg.edge_guided_max_roi_area,
            cfg.edge_guided_max_total_search,
            cfg.tone_guided_enabled,
            cfg.tone_guided_max_component_area,
            cfg.tone_guided_max_roi_area,
            cfg.tone_guided_context_radius,
            cfg.tone_guided_search_radius,
            cfg.tone_guided_patch_radius,
            cfg.tone_guided_candidate_cap,
            cfg.tone_guided_top_k,
            cfg.tone_guided_tone_weight,
            cfg.tone_guided_spatial_weight,
            cfg.tone_guided_texture_weight,
            cfg.tone_guided_gradient_weight,
            cfg.tone_guided_min_context_pixels,
            cfg,
            frequency_selected,
        )
        _merge_defect_repair_stats(defect_repair_stats, roi_result.defect_stats)
        if roi_result.defect_debug:
            defect_component_repairs[int(component.label)] = roi_result.defect_debug
            pattern = roi_result.defect_debug.get("frequency_pattern_class")
            if pattern in FREQUENCY_PATTERN_VALUES:
                frequency_pattern_map[component_core] = FREQUENCY_PATTERN_VALUES[str(pattern)]

        rgb_candidate_roi = repair_candidate[y0:y1, x0:x1, :3]
        before_roi = candidate_before_guard[y0:y1, x0:x1, :3]
        after_roi = candidate_after_guard[y0:y1, x0:x1, :3]
        rgb_candidate_roi[repair_roi_mask] = roi_result.image[repair_roi_mask]
        before_roi[repair_roi_mask] = roi_result.candidate_before_guard[repair_roi_mask]
        after_roi[repair_roi_mask] = roi_result.image[repair_roi_mask]

        rejected_roi = guard_rejected_mask[roi_slice]
        rejected_roi |= roi_result.rejected_mask
        alpha_roi = component_alpha_map[roi_slice]
        alpha_roi[repair_roi_mask] = np.minimum(alpha_roi[repair_roi_mask], roi_result.alpha_scale)
        component_alpha_values.append(float(roi_result.alpha_scale))
        if roi_result.low_confidence:
            low_confidence_component_count += 1

    # Candidate generation may use a dilated repair mask, but the shell receives
    # low alpha so normal pixels near a defect are not fully replaced.
    final_alpha = np.clip(blend_alpha * component_alpha_map * cfg.strength, 0.0, 1.0).astype(np.float32)
    if cfg.method == "defect_aware" and cfg.defect_core_full_replace and cfg.strength > 0.0:
        final_alpha[core_mask] = float(cfg.strength)
    output_float = image_float.copy()
    output_float[:, :, :3] = (
        image_float[:, :, :3] * (1.0 - final_alpha[:, :, None])
        + repair_candidate[:, :, :3] * final_alpha[:, :, None]
    )
    outside = final_alpha <= 0.0
    output_float[outside, :] = image_float[outside, :]
    repaired = restore_dtype(output_float, original.dtype)
    repaired[outside, :] = original[outside, :]

    elapsed = (time.perf_counter() - start) * 1000.0
    metrics = _repair_metrics(
        original=original,
        repaired=repaired,
        blend_alpha=blend_alpha,
        core_mask=core_mask,
        repair_mask=repair_mask,
        guard_rejected_mask=guard_rejected_mask,
        changed_bbox_list=changed_bbox_list,
        processing_time_ms=elapsed,
        channel_used=normalized.channel_used,
        filtered=filtered,
        component_alpha_values=component_alpha_values,
        low_confidence_component_count=low_confidence_component_count,
    )
    _add_defect_aware_metrics(
        metrics,
        cfg.method,
        skipped=False,
        defect_features=defect_features,
        defect_repair_stats=defect_repair_stats,
    )
    _add_defect_aware_blend_metrics(
        metrics,
        cfg.method,
        final_alpha=final_alpha,
        core_mask=core_mask,
        repair_mask=repair_mask,
    )

    result = RepairResult(
        repaired_image=repaired,
        binary_mask=repair_mask,
        soft_mask=blend_alpha,
        changed_bbox_list=changed_bbox_list,
        metrics=metrics,
        debug_images=_debug_images(
            original=original,
            repaired=repaired,
            normalized_mask=normalized_mask,
            core_mask=core_mask,
            repair_mask=repair_mask,
            blend_alpha=blend_alpha,
            candidate_before_guard=candidate_before_guard,
            candidate_after_guard=candidate_after_guard,
            guard_rejected_mask=guard_rejected_mask,
            output_dtype=original.dtype,
        )
        if collect_debug_images
        else {},
        core_mask=core_mask,
        repair_mask=repair_mask,
        blend_alpha=blend_alpha,
        guard_rejected_mask=guard_rejected_mask,
        defect_component_repairs=defect_component_repairs,
    )
    if collect_debug_images and frequency_selection is not None:
        result.debug_images.update(_frequency_debug_images(original, frequency_selection, frequency_pattern_map))
    _write_debug_outputs(cfg.debug_dir, result)
    _write_defect_debug_outputs(cfg.debug_dir, result, defect_features)
    _write_frequency_debug_outputs(cfg.debug_dir, result)
    return result


def _repair_roi(
    roi: np.ndarray,
    repair_mask: np.ndarray,
    method: str,
    area: int,
    core_mask: np.ndarray | None = None,
    defect_strategy: str | None = None,
    defect_label: int = 0,
    grain_reinject_strength: float = 0.25,
    grain_context_radius: int = 8,
    grain_blur_radius: int = 1,
    grain_min_context_pixels: int = 16,
    color_match_strength: float = 0.0,
    color_match_radius: int = 8,
    color_match_min_context_pixels: int = 12,
    edge_guided_enabled: bool = True,
    edge_guided_max_component_area: int = 64,
    edge_guided_context_radius: int = 4,
    edge_guided_search_radius: int = 8,
    edge_guided_min_coherence: float = 0.35,
    edge_guided_min_gradient_energy: float = 2.5e-4,
    edge_guided_max_roi_area: int = 4096,
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
    frequency_config: RepairConfig | None = None,
    frequency_guided_selected: bool = False,
) -> _RoiRepairResult:
    if core_mask is None:
        core_mask = repair_mask
    alpha_scale = 1.0
    low_confidence = False
    defect_stats: dict[str, Any] = {}
    defect_debug: dict[str, Any] = {}
    if method == "defect_aware":
        repaired, alpha_scale, low_confidence, defect_stats, defect_debug = _defect_aware_repair(
            roi,
            repair_mask,
            core_mask,
            area,
            defect_strategy,
            defect_label,
            grain_reinject_strength,
            grain_context_radius,
            grain_blur_radius,
            grain_min_context_pixels,
            edge_guided_enabled,
            edge_guided_max_component_area,
            edge_guided_context_radius,
            edge_guided_search_radius,
            edge_guided_min_coherence,
            edge_guided_min_gradient_energy,
            edge_guided_max_roi_area,
            edge_guided_max_total_search,
            tone_guided_enabled,
            tone_guided_max_component_area,
            tone_guided_max_roi_area,
            tone_guided_context_radius,
            tone_guided_search_radius,
            tone_guided_patch_radius,
            tone_guided_candidate_cap,
            tone_guided_top_k,
            tone_guided_tone_weight,
            tone_guided_spatial_weight,
            tone_guided_texture_weight,
            tone_guided_gradient_weight,
            tone_guided_min_context_pixels,
            frequency_config,
            frequency_guided_selected,
        )
    else:
        effective_method = _effective_repair_method(method)
        method_mask = repair_mask

        if effective_method == "linear":
            repaired = _linear_complement_repair(roi, method_mask)
        elif effective_method == "kl":
            repaired = _kl_complement_repair(roi, method_mask)
        elif effective_method == "median":
            repaired = _median_repair(roi, repair_mask)
        elif effective_method == "inpaint":
            repaired = _diffusion_inpaint(roi, repair_mask)
        elif effective_method == "denoise":
            repaired = _masked_denoise(roi, repair_mask)
        elif effective_method == "hybrid":
            repaired, alpha_scale, low_confidence = _adaptive_repair(roi, repair_mask, core_mask, area)
        elif effective_method == "adaptive":
            repaired, alpha_scale, low_confidence = _adaptive_repair(roi, repair_mask, core_mask, area)
        elif effective_method == "aggressive":
            repaired = _aggressive_repair(roi, repair_mask)
        elif effective_method == "wide_scratch":
            repaired = _wide_scratch_repair(roi, repair_mask)
        else:
            raise ValueError(f"Unsupported repair method: {method}")

    if color_match_strength > 0.0:
        repaired, color_stats = _match_repair_to_context(
            roi,
            repaired,
            repair_mask,
            strength=color_match_strength,
            radius=color_match_radius,
            min_context_pixels=color_match_min_context_pixels,
        )
        defect_stats.update(color_stats)

    before_guard = repaired.copy()
    guarded, rejected, guard_stats = _guard_repair_candidate(roi, repaired, repair_mask, core_mask)
    defect_stats.update(guard_stats)
    seam_score = _component_seam_score(roi, guarded, repair_mask)
    seam_alpha = _alpha_scale_from_seam(seam_score)
    alpha_scale = min(alpha_scale, seam_alpha)
    if alpha_scale < 0.65:
        low_confidence = True
    return _RoiRepairResult(
        image=guarded,
        candidate_before_guard=before_guard,
        rejected_mask=rejected,
        alpha_scale=float(np.clip(alpha_scale, 0.0, 1.0)),
        low_confidence=low_confidence,
        defect_stats=defect_stats,
        defect_debug=defect_debug,
    )


def _defect_aware_repair(
    roi: np.ndarray,
    repair_mask: np.ndarray,
    core_mask: np.ndarray,
    area: int,
    defect_strategy: str | None,
    defect_label: int,
    grain_reinject_strength: float,
    grain_context_radius: int,
    grain_blur_radius: int,
    grain_min_context_pixels: int,
    edge_guided_enabled: bool,
    edge_guided_max_component_area: int,
    edge_guided_context_radius: int,
    edge_guided_search_radius: int,
    edge_guided_min_coherence: float,
    edge_guided_min_gradient_energy: float,
    edge_guided_max_roi_area: int,
    edge_guided_max_total_search: int,
    tone_guided_enabled: bool,
    tone_guided_max_component_area: int,
    tone_guided_max_roi_area: int,
    tone_guided_context_radius: int,
    tone_guided_search_radius: int,
    tone_guided_patch_radius: int,
    tone_guided_candidate_cap: int,
    tone_guided_top_k: int,
    tone_guided_tone_weight: float,
    tone_guided_spatial_weight: float,
    tone_guided_texture_weight: float,
    tone_guided_gradient_weight: float,
    tone_guided_min_context_pixels: int,
    frequency_config: RepairConfig | None,
    frequency_guided_selected: bool,
) -> tuple[np.ndarray, float, bool, dict[str, Any], dict[str, Any]]:
    if frequency_guided_selected and frequency_config is not None and frequency_config.frequency_guided_enabled:
        frequency_result = repair_frequency_guided_roi(
            roi,
            core_mask,
            repair_mask,
            frequency_config,
            defect_label=defect_label,
        )
        if not frequency_result.fallback:
            stats = dict(frequency_result.stats)
            debug = {
                "repair_strategy": "frequency_guided",
                **frequency_result.debug,
            }
            grain_strength = grain_reinject_strength
            if int(stats.get("frequency_midband_transfer_pixel_count", 0)) > 0:
                grain_strength *= 0.5
            repaired = _maybe_reinject_grain(
                roi,
                frequency_result.image.astype(np.float32, copy=False),
                core_mask,
                repair_mask,
                defect_label,
                grain_strength,
                grain_context_radius,
                grain_blur_radius,
                grain_min_context_pixels,
                stats,
            )
            return repaired.astype(np.float32, copy=False), 1.0, False, stats, debug
        stats = dict(frequency_result.stats)
        stats["defect_fallback_count"] = int(stats.get("defect_fallback_count", 0)) + 1
        debug = {
            "repair_strategy": defect_strategy or "frequency_guided_fallback",
            **frequency_result.debug,
        }
        repaired, alpha_scale, low_confidence = _adaptive_repair(roi, repair_mask, core_mask, area)
        return repaired, alpha_scale, low_confidence, stats, debug

    if defect_strategy in {"tiny_local", "small_local"}:
        repaired, small_stats = repair_small_local_roi(
            roi,
            repair_mask,
            repair_mask,
            strategy=defect_strategy,
            context_radius=edge_guided_context_radius,
            use_local_plane=True,
            edge_guided_enabled=edge_guided_enabled,
            edge_guided_max_component_area=edge_guided_max_component_area,
            edge_guided_max_roi_area=edge_guided_max_roi_area,
            edge_guided_search_radius=edge_guided_search_radius,
            edge_guided_min_coherence=edge_guided_min_coherence,
            edge_guided_min_gradient_energy=edge_guided_min_gradient_energy,
            edge_guided_max_total_search=edge_guided_max_total_search,
            tone_guided_enabled=tone_guided_enabled,
            tone_guided_max_component_area=tone_guided_max_component_area,
            tone_guided_max_roi_area=tone_guided_max_roi_area,
            tone_guided_context_radius=tone_guided_context_radius,
            tone_guided_search_radius=tone_guided_search_radius,
            tone_guided_patch_radius=tone_guided_patch_radius,
            tone_guided_candidate_cap=tone_guided_candidate_cap,
            tone_guided_top_k=tone_guided_top_k,
            tone_guided_tone_weight=tone_guided_tone_weight,
            tone_guided_spatial_weight=tone_guided_spatial_weight,
            tone_guided_texture_weight=tone_guided_texture_weight,
            tone_guided_gradient_weight=tone_guided_gradient_weight,
            tone_guided_min_context_pixels=tone_guided_min_context_pixels,
        )
        fallback_method = str(small_stats.get("fallback_method", small_stats.get("method", "fallback")))
        edge_sample_count = int(small_stats.get("edge_guided_sample_count", 0))
        stats = {
            "small_local_component_count": 1,
            "small_local_pixel_count": int(np.count_nonzero(repair_mask)),
            "small_local_plane_count": 1 if fallback_method == "plane" or small_stats.get("method") == "plane" else 0,
            "small_local_median_count": 1 if fallback_method == "median" or small_stats.get("method") == "median" else 0,
            "small_local_fallback_count": 1 if small_stats.get("fallback") else 0,
            "small_local_edge_guided_component_count": 1 if small_stats.get("edge_guided_used") else 0,
            "small_local_edge_guided_pixel_count": int(small_stats.get("edge_guided_pixel_count", 0)),
            "small_local_edge_guided_fallback_count": 1 if small_stats.get("edge_guided_fallback") else 0,
            "small_local_edge_guided_low_confidence_count": 1 if small_stats.get("edge_guided_low_confidence") else 0,
            "small_local_edge_guided_coherence_total": float(small_stats.get("edge_guided_coherence", 0.0)) if edge_sample_count > 0 else 0.0,
            "small_local_edge_guided_coherence_count": 1 if edge_sample_count > 0 else 0,
        }
        for key in (
            "tone_guided_component_count",
            "tone_guided_pixel_count",
            "tone_guided_fallback_count",
            "tone_guided_no_context_count",
            "tone_guided_candidate_count_total",
            "tone_guided_top_k_total",
            "tone_guided_top_k_count",
            "tone_guided_score_total",
            "tone_guided_score_count",
            "tone_guided_context_rgb_distance_total",
            "tone_guided_context_rgb_distance_count",
        ):
            stats[key] = small_stats.get(key, 0)
        debug = {
            "repair_strategy": defect_strategy or "unknown",
            "local_method": str(small_stats.get("method", "fallback")),
            "local_fallback_method": fallback_method,
            "edge_guided_used": bool(small_stats.get("edge_guided_used", False)),
            "edge_guided_fallback_reason": str(small_stats.get("edge_guided_fallback_reason", "")),
            "edge_guided_coherence": float(small_stats.get("edge_guided_coherence", 0.0)),
            "edge_guided_gradient_energy": float(small_stats.get("edge_guided_gradient_energy", 0.0)),
            "edge_guided_sample_count": edge_sample_count,
        }
        if not small_stats.get("fallback"):
            repaired = _maybe_reinject_grain(
                roi,
                repaired.astype(np.float32, copy=False),
                core_mask,
                repair_mask,
                defect_label,
                grain_reinject_strength,
                grain_context_radius,
                grain_blur_radius,
                grain_min_context_pixels,
                stats,
            )
            return repaired.astype(np.float32, copy=False), 1.0, False, stats, debug
        stats["defect_fallback_count"] = 1
        repaired, alpha_scale, low_confidence = _adaptive_repair(roi, repair_mask, core_mask, area)
        return repaired, alpha_scale, low_confidence, stats, debug
    if defect_strategy == "fast_inpaint":
        repaired, fast_stats = repair_fast_inpaint_roi(
            roi,
            repair_mask,
            max_iterations=80,
            preserve_alpha=True,
        )
        stats = {
            "fast_inpaint_component_count": 1,
            "fast_inpaint_pixel_count": int(np.count_nonzero(repair_mask)),
            "fast_inpaint_iterations_total": int(fast_stats.get("iterations", 0)),
            "fast_inpaint_fallback_count": 1 if fast_stats.get("fallback") else 0,
        }
        if not fast_stats.get("fallback"):
            repaired = _maybe_reinject_grain(
                roi,
                repaired.astype(np.float32, copy=False),
                core_mask,
                repair_mask,
                defect_label,
                grain_reinject_strength,
                grain_context_radius,
                grain_blur_radius,
                grain_min_context_pixels,
                stats,
            )
            return repaired.astype(np.float32, copy=False), 1.0, False, stats, {}
        stats["defect_fallback_count"] = 1
        repaired, alpha_scale, low_confidence = _adaptive_repair(roi, repair_mask, core_mask, area)
        return repaired, alpha_scale, low_confidence, stats, {}
    if defect_strategy == "directional":
        pixel_count = int(np.count_nonzero(repair_mask))
        stats = {
            "directional_component_count": 1,
            "directional_pixel_count": pixel_count,
            "directional_fallback_count": 0,
            "directional_cap_exceeded_count": 0,
        }
        if pixel_count > _DIRECTIONAL_MAX_PIXELS:
            stats["directional_fallback_count"] = 1
            stats["directional_cap_exceeded_count"] = 1
            stats["defect_fallback_count"] = 1
            repaired, alpha_scale, low_confidence = _adaptive_repair(roi, repair_mask, core_mask, area)
            return repaired, alpha_scale, low_confidence, stats, {}

        directional, directional_filled = _pca_directional_fill(roi, repair_mask, core_mask)
        if directional_filled.any():
            fallback, fast_stats = repair_fast_inpaint_roi(roi, repair_mask, max_iterations=48, preserve_alpha=True)
            repaired = fallback.astype(np.float32, copy=False)
            repaired[directional_filled] = directional[directional_filled]
            if int(np.count_nonzero(directional_filled & repair_mask)) < pixel_count:
                stats["directional_fallback_count"] = 1
                stats["fast_inpaint_iterations_total"] = int(fast_stats.get("iterations", 0))
            repaired = _maybe_reinject_grain(
                roi,
                repaired,
                core_mask,
                repair_mask,
                defect_label,
                grain_reinject_strength,
                grain_context_radius,
                grain_blur_radius,
                grain_min_context_pixels,
                stats,
            )
            return repaired, 1.0, False, stats, {}

        stats["directional_fallback_count"] = 1
        stats["defect_fallback_count"] = 1
        repaired, alpha_scale, low_confidence = _adaptive_repair(roi, repair_mask, core_mask, area)
        return repaired, alpha_scale, low_confidence, stats, {}
    if defect_strategy == "patch":
        repaired, patch_stats = repair_patch_match_roi(
            roi,
            core_mask,
            repair_mask,
            search_radius=64,
            patch_margin=3,
            max_component_area=1600,
            max_candidates=2000,
            stride=1,
        )
        best_score = patch_stats.get("best_score")
        stats = {
            "patch_component_count": 1,
            "patch_pixel_count": int(np.count_nonzero(core_mask)),
            "patch_candidate_count_total": int(patch_stats.get("candidate_count", 0)),
            "patch_fallback_count": 1 if patch_stats.get("fallback") else 0,
            "patch_best_score_total": float(best_score) if best_score is not None else 0.0,
            "patch_best_score_count": 1 if best_score is not None else 0,
            "patch_stride_used_counts": {str(int(patch_stats.get("stride_used", 1))): 1},
        }
        if not patch_stats.get("fallback"):
            repaired = _maybe_reinject_grain(
                roi,
                repaired.astype(np.float32, copy=False),
                core_mask,
                repair_mask,
                defect_label,
                grain_reinject_strength,
                grain_context_radius,
                grain_blur_radius,
                grain_min_context_pixels,
                stats,
            )
            return repaired.astype(np.float32, copy=False), 1.0, False, stats, {}
        stats["defect_fallback_count"] = 1
        repaired, alpha_scale, low_confidence = _adaptive_repair(roi, repair_mask, core_mask, area)
        return repaired, alpha_scale, low_confidence, stats, {}

    repaired, alpha_scale, low_confidence = _adaptive_repair(roi, repair_mask, core_mask, area)
    return repaired, alpha_scale, low_confidence, {"defect_fallback_count": 1}, {}

def _maybe_reinject_grain(
    roi: np.ndarray,
    candidate: np.ndarray,
    core_mask: np.ndarray,
    repair_mask: np.ndarray,
    defect_label: int,
    strength: float,
    context_radius: int,
    blur_radius: int,
    min_context_pixels: int,
    stats: dict[str, Any],
) -> np.ndarray:
    if strength <= 0.0:
        return candidate
    grain_mask = repair_mask if np.asarray(repair_mask, dtype=bool).any() else core_mask
    grained, grain_stats = reinject_grain_roi(
        roi,
        candidate,
        grain_mask,
        repair_mask,
        label=defect_label,
        strength=strength,
        context_radius=context_radius,
        blur_radius=blur_radius,
        min_context_pixels=min_context_pixels,
    )
    if grain_stats.get("applied"):
        stats["grain_reinject_component_count"] = int(stats.get("grain_reinject_component_count", 0)) + 1
        stats["grain_reinject_pixel_count"] = int(stats.get("grain_reinject_pixel_count", 0)) + int(
            grain_stats.get("pixel_count", 0)
        )
    elif grain_stats.get("skipped_no_context"):
        stats["grain_reinject_skipped_no_context_count"] = int(
            stats.get("grain_reinject_skipped_no_context_count", 0)
        ) + 1
    return grained.astype(np.float32, copy=False)


def _effective_repair_method(method: str) -> str:
    if method == "defect_aware":
        return _DEFECT_AWARE_FALLBACK_METHOD
    return method


def _adaptive_repair(
    roi: np.ndarray,
    repair_mask: np.ndarray,
    core_mask: np.ndarray,
    area: int,
) -> tuple[np.ndarray, float, bool]:
    if not repair_mask.any():
        return roi.copy(), 1.0, False
    context = ~repair_mask
    if np.count_nonzero(context) < 3:
        return roi.copy(), 0.2, True

    features = _component_features(core_mask if core_mask.any() else repair_mask)
    alpha_scale = 1.0
    low_confidence = False

    if features["is_slender"]:
        directional, directional_filled = _pca_directional_fill(roi, repair_mask, core_mask)
        fallback = _normalized_convolution_fill(roi, repair_mask)
        repaired = fallback
        repaired[directional_filled] = directional[directional_filled]
        repaired = _boost_core_defect_replacement(roi, repaired, repair_mask, core_mask)
        return repaired, alpha_scale, low_confidence

    if area <= 256:
        cv2_result = _cv2_telea_inpaint(roi, repair_mask)
        repaired = cv2_result if cv2_result is not None else _normalized_convolution_fill(roi, repair_mask)
        repaired = _boost_core_defect_replacement(roi, repaired, repair_mask, core_mask)
        return repaired, alpha_scale, low_confidence

    if area <= 5000:
        repaired = _blend_convolution_and_plane(roi, repair_mask)
        repaired = _boost_core_defect_replacement(roi, repaired, repair_mask, core_mask)
        return repaired, alpha_scale, low_confidence

    # Large uncertain regions are intentionally blended weakly rather than
    # flattened with a single median color.
    repaired = _blend_convolution_and_plane(roi, repair_mask)
    repaired = _boost_core_defect_replacement(roi, repaired, repair_mask, core_mask)
    return repaired, 0.4, True


def _median_repair(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = roi.copy()
    context = ~mask
    if not context.any() or not mask.any():
        return out
    median = np.median(roi[context], axis=0)
    out[mask] = median
    return out


def _diffusion_inpaint(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = roi.copy()
    unknown = mask.copy()
    filled = ~unknown
    if not unknown.any() or not filled.any():
        return out

    max_iterations = max(4, roi.shape[0] + roi.shape[1])
    for _ in range(max_iterations):
        if not unknown.any():
            break
        padded_filled = np.pad(filled, 1, mode="constant", constant_values=False)
        padded_out = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        sums = np.zeros_like(out, dtype=np.float32)
        counts = np.zeros(unknown.shape, dtype=np.float32)

        for oy in range(3):
            for ox in range(3):
                if oy == 1 and ox == 1:
                    continue
                neighbor_filled = padded_filled[oy : oy + roi.shape[0], ox : ox + roi.shape[1]]
                neighbor_values = padded_out[oy : oy + roi.shape[0], ox : ox + roi.shape[1]]
                sums += neighbor_values * neighbor_filled[:, :, None]
                counts += neighbor_filled.astype(np.float32)

        candidates = unknown & (counts > 0)
        if not candidates.any():
            break
        out[candidates] = sums[candidates] / counts[candidates][:, None]
        filled[candidates] = True
        unknown[candidates] = False

    if unknown.any() and filled.any():
        out[unknown] = np.median(out[filled], axis=0)
    return out


def _aggressive_repair(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return roi.copy()
    context = ~mask
    if not context.any():
        return roi.copy()

    repaired = _diffusion_inpaint(roi, mask)
    ring = _context_ring(mask, radius=8)
    sample = roi[ring] if ring.any() else roi[context]
    ring_median = np.median(sample, axis=0)

    out = repaired.copy()
    out[mask] = (repaired[mask] * 0.55) + (ring_median * 0.45)

    for _ in range(3):
        smoothed = _box_blur_image(out, radius=1)
        out[mask] = smoothed[mask]
        out[context] = roi[context]

    return out


def _wide_scratch_repair(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return roi.copy()
    context = ~mask
    if not context.any():
        return roi.copy()

    ys, xs = np.nonzero(mask)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    primary_axis = 1 if height >= width else 0
    secondary_axis = 0 if primary_axis == 1 else 1

    primary, primary_filled = _directional_span_fill(roi, mask, primary_axis)
    secondary, secondary_filled = _directional_span_fill(roi, mask, secondary_axis)

    unfilled = mask & ~primary_filled & ~secondary_filled
    out = _diffusion_inpaint(roi, mask) if unfilled.any() else roi.copy()
    secondary_only = secondary_filled & ~primary_filled
    out[secondary_only] = secondary[secondary_only]
    out[primary_filled] = primary[primary_filled]

    smoothed = _box_blur_image(out, radius=1)
    out[mask] = (out[mask] * 0.8) + (smoothed[mask] * 0.2)
    out[context] = roi[context]
    return out


def _linear_complement_repair(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    unknown = np.asarray(mask, dtype=bool)
    if not unknown.any():
        return roi.copy()
    known = ~unknown
    if not known.any():
        return roi.copy()

    out = roi.copy()
    context_median = np.median(roi[known], axis=0)
    out[unknown] = context_median

    unknown_count = int(np.count_nonzero(unknown))
    iterations = int(np.clip(np.ceil(np.sqrt(unknown_count)) * 2, 20, 300))
    current = out.copy()

    for _ in range(iterations):
        neighbor_sum = np.zeros_like(current, dtype=np.float32)
        neighbor_count = np.zeros((*unknown.shape, 1), dtype=np.float32)

        neighbor_sum[:, 1:] += current[:, :-1]
        neighbor_count[:, 1:] += 1.0
        neighbor_sum[:, :-1] += current[:, 1:]
        neighbor_count[:, :-1] += 1.0
        neighbor_sum[1:, :] += current[:-1, :]
        neighbor_count[1:, :] += 1.0
        neighbor_sum[:-1, :] += current[1:, :]
        neighbor_count[:-1, :] += 1.0

        next_values = current.copy()
        averages = neighbor_sum / np.maximum(neighbor_count, 1.0)
        next_values[unknown] = averages[unknown]
        next_values[known] = roi[known]
        current = next_values

    result = roi.copy()
    result[unknown] = current[unknown]
    return result.astype(np.float32, copy=False)


def _kl_complement_repair(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    unknown = np.asarray(mask, dtype=bool)
    if not unknown.any():
        return roi.copy()
    context = ~unknown
    if not context.any():
        return roi.copy()

    linear = _linear_complement_repair(roi, unknown)
    ring = _context_ring(unknown, radius=3)
    sample_mask = ring if ring.any() else context
    histogram = _RgbHistogram.from_pixels(roi[sample_mask])
    representatives = histogram.representative_colors()
    quotas = histogram.quotas_for(int(np.count_nonzero(unknown)))
    if int(np.sum(quotas)) == 0:
        return linear

    result = linear.copy()
    ys, xs = np.nonzero(unknown)
    for y, x in zip(ys, xs):
        bin_index = _nearest_available_kl_bin(linear[y, x], representatives, quotas)
        if bin_index is None:
            break
        result[y, x] = representatives[bin_index]
        quotas[bin_index] -= 1
    result[context] = roi[context]
    return result.astype(np.float32, copy=False)


class _RgbHistogram:
    def __init__(self) -> None:
        self.counts = np.zeros(_KL_BIN_COUNT, dtype=np.int64)
        self.sums = np.zeros((_KL_BIN_COUNT, 3), dtype=np.float64)
        self.total = 0

    @classmethod
    def from_pixels(cls, pixels: np.ndarray) -> "_RgbHistogram":
        histogram = cls()
        flat = np.asarray(pixels, dtype=np.float32).reshape(-1, 3)
        for pixel in flat:
            bin_index = _kl_color_bin(pixel)
            histogram.counts[bin_index] += 1
            histogram.sums[bin_index] += pixel.astype(np.float64)
            histogram.total += 1
        return histogram

    def representative_colors(self) -> np.ndarray:
        colors = np.zeros((_KL_BIN_COUNT, 3), dtype=np.float32)
        for bin_index in range(_KL_BIN_COUNT):
            count = int(self.counts[bin_index])
            if count > 0:
                colors[bin_index] = (self.sums[bin_index] / float(count)).astype(np.float32)
            else:
                colors[bin_index] = _kl_bin_center_color(bin_index)
        return colors

    def quotas_for(self, target_count: int) -> np.ndarray:
        quotas = np.zeros(_KL_BIN_COUNT, dtype=np.int64)
        if self.total <= 0 or target_count <= 0:
            return quotas

        non_empty = np.flatnonzero(self.counts > 0)
        if non_empty.size == 0:
            return quotas
        if non_empty.size >= target_count:
            ordered = sorted(non_empty.tolist(), key=lambda idx: int(self.counts[idx]), reverse=True)
            quotas[ordered[:target_count]] = 1
            return quotas

        quotas[non_empty] = 1
        remaining = target_count - int(non_empty.size)
        desired = self.counts[non_empty].astype(np.float64) * float(remaining) / float(self.total)
        extra = np.floor(desired).astype(np.int64)
        quotas[non_empty] += extra
        assigned = int(np.sum(extra))
        leftover = remaining - assigned
        if leftover > 0:
            remainders = desired - extra.astype(np.float64)
            order = np.argsort(-remainders)
            quotas[non_empty[order[:leftover]]] += 1
        return quotas


def _nearest_available_kl_bin(
    reference: np.ndarray,
    representatives: np.ndarray,
    quotas: np.ndarray,
) -> int | None:
    available = np.flatnonzero(quotas > 0)
    if available.size == 0:
        return None
    diffs = representatives[available] - reference.astype(np.float32)
    distances = np.sum(diffs * diffs, axis=1)
    return int(available[int(np.argmin(distances))])


def _kl_color_bin(pixel: np.ndarray) -> int:
    values = np.clip(np.asarray(pixel, dtype=np.float32), 0.0, 1.0 - _KL_EPSILON)
    bins = np.floor(values * _KL_BINS_PER_CHANNEL).astype(np.int64)
    bins = np.clip(bins, 0, _KL_BINS_PER_CHANNEL - 1)
    return int((bins[0] * _KL_BINS_PER_CHANNEL + bins[1]) * _KL_BINS_PER_CHANNEL + bins[2])


def _kl_bin_center_color(bin_index: int) -> np.ndarray:
    blue = bin_index % _KL_BINS_PER_CHANNEL
    green = (bin_index // _KL_BINS_PER_CHANNEL) % _KL_BINS_PER_CHANNEL
    red = bin_index // (_KL_BINS_PER_CHANNEL * _KL_BINS_PER_CHANNEL)
    step = 1.0 / float(_KL_BINS_PER_CHANNEL)
    return np.asarray(
        [red * step + step * 0.5, green * step + step * 0.5, blue * step + step * 0.5],
        dtype=np.float32,
    )


def _normalized_convolution_fill(
    roi: np.ndarray,
    unknown_mask: np.ndarray,
    sigma_list: tuple[float, ...] = (1.2, 2.5, 5.0),
    iterations: int = 2,
) -> np.ndarray:
    out = roi.copy()
    unknown = np.asarray(unknown_mask, dtype=bool).copy()
    known = ~unknown
    if not unknown.any() or not known.any():
        return out

    eps = 1.0e-6
    for _ in range(max(1, int(iterations))):
        for sigma in sigma_list:
            if not unknown.any():
                break
            radius = max(1, int(round(float(sigma) * 2.0)))
            known_weight = known.astype(np.float32)
            den = _box_blur_scalar(known_weight, radius)
            num = _box_blur_image(out * known_weight[:, :, None], radius)
            fill = num / np.maximum(den[:, :, None], eps)
            candidates = unknown & (den > eps)
            out[candidates] = fill[candidates]
            known[candidates] = True
            unknown[candidates] = False

    if unknown.any():
        fallback = _trimmed_context_mean(out, known)
        out[unknown] = fallback
    return out


def _blend_convolution_and_plane(roi: np.ndarray, unknown_mask: np.ndarray) -> np.ndarray:
    conv = _normalized_convolution_fill(roi, unknown_mask)
    context = _context_ring(unknown_mask, radius=8)
    if not context.any():
        context = ~unknown_mask
    plane = _fit_local_plane_fill(roi, unknown_mask, context)
    if plane is None:
        return conv

    plane_residual = _plane_context_residual(roi, plane, context)
    plane_weight = 0.55 if plane_residual < 0.035 else 0.25
    out = roi.copy()
    out[unknown_mask] = (conv[unknown_mask] * (1.0 - plane_weight)) + (plane[unknown_mask] * plane_weight)
    return out


def _boost_core_defect_replacement(
    roi: np.ndarray,
    candidate: np.ndarray,
    repair_mask: np.ndarray,
    core_mask: np.ndarray,
) -> np.ndarray:
    if not core_mask.any():
        return candidate
    context = _context_ring(repair_mask, radius=6)
    if not context.any():
        context = ~repair_mask
    if np.count_nonzero(context) < 3:
        return candidate

    sample = roi[context]
    context_median = np.median(sample, axis=0)
    defect_like = core_mask


    plane = _fit_local_plane_fill(roi, repair_mask, context)
    local_fill = candidate if plane is None else plane
    robust_fill = np.broadcast_to(context_median.reshape(1, 1, 3), roi.shape).astype(np.float32)
    boosted = candidate.copy()
    # For confirmed core defects, use a stronger local replacement. This fixes
    # the real-world case where a thin proxy mask leaves bright dust fringes in
    # the known set and normalized convolution alone becomes too conservative.
    boosted[defect_like] = (local_fill[defect_like] * 0.65) + (robust_fill[defect_like] * 0.35)
    return boosted


def _fit_local_plane_fill(
    roi: np.ndarray,
    unknown_mask: np.ndarray,
    context_mask: np.ndarray,
) -> np.ndarray | None:
    ys, xs = np.nonzero(context_mask)
    if len(xs) < 6:
        return None

    height, width = roi.shape[:2]
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

    plane = roi.copy()
    for channel in range(roi.shape[2]):
        values = roi[ys, xs, channel]
        inliers = _trimmed_inliers(values)
        if np.count_nonzero(inliers) < 3:
            inliers = np.ones(values.shape, dtype=bool)
        try:
            coeffs, *_ = np.linalg.lstsq(design[inliers], values[inliers], rcond=None)
        except np.linalg.LinAlgError:
            return None
        plane[:, :, channel] = np.sum(full_design * coeffs.reshape(1, 1, 3), axis=2)
    plane = np.clip(plane, 0.0, 1.0).astype(np.float32)
    out = roi.copy()
    out[unknown_mask] = plane[unknown_mask]
    return out


def _pca_directional_fill(
    roi: np.ndarray,
    repair_mask: np.ndarray,
    core_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out = roi.copy()
    filled = np.zeros(repair_mask.shape, dtype=bool)
    direction = _principal_direction(core_mask if core_mask.any() else repair_mask)
    if direction is None:
        return out, filled

    dx, dy = direction
    nx, ny = -dy, dx
    max_search = int(max(8, min(max(roi.shape[0], roi.shape[1]), 64)))
    ys, xs = np.nonzero(repair_mask)
    for y, x in zip(ys, xs):
        left = _sample_along_ray(roi, repair_mask, float(y), float(x), ny, nx, max_search)
        right = _sample_along_ray(roi, repair_mask, float(y), float(x), -ny, -nx, max_search)
        if left is None and right is None:
            continue
        if left is not None and right is not None:
            out[y, x] = (left + right) * 0.5
        elif left is not None:
            out[y, x] = left
        else:
            out[y, x] = right
        filled[y, x] = True
    return out, filled


def _sample_along_ray(
    roi: np.ndarray,
    mask: np.ndarray,
    y: float,
    x: float,
    dy: float,
    dx: float,
    max_search: int,
) -> np.ndarray | None:
    height, width = mask.shape
    for step in range(1, max_search + 1):
        yy = int(round(y + dy * step))
        xx = int(round(x + dx * step))
        if yy < 0 or yy >= height or xx < 0 or xx >= width:
            return None
        if not mask[yy, xx]:
            return roi[yy, xx].copy()
    return None


def _cv2_telea_inpaint(roi: np.ndarray, unknown_mask: np.ndarray) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        return None
    except Exception:
        return None

    if roi.shape[2] != 3 or not unknown_mask.any():
        return None
    try:
        src = np.rint(np.clip(roi, 0.0, 1.0) * 255.0).astype(np.uint8)
        mask_u8 = (unknown_mask.astype(np.uint8) * 255)
        repaired = cv2.inpaint(src, mask_u8, 3.0, cv2.INPAINT_TELEA)
    except Exception:
        return None
    return repaired.astype(np.float32) / 255.0


def _directional_span_fill(roi: np.ndarray, mask: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    out = roi.copy()
    filled = np.zeros(mask.shape, dtype=bool)
    if axis == 1:
        for y in range(mask.shape[0]):
            _fill_line_between_context(
                values=roi[y, :, :],
                line_mask=mask[y, :],
                write_values=out[y, :, :],
                write_filled=filled[y, :],
            )
    else:
        for x in range(mask.shape[1]):
            _fill_line_between_context(
                values=roi[:, x, :],
                line_mask=mask[:, x],
                write_values=out[:, x, :],
                write_filled=filled[:, x],
            )
    return out, filled


def _fill_line_between_context(
    values: np.ndarray,
    line_mask: np.ndarray,
    write_values: np.ndarray,
    write_filled: np.ndarray,
) -> None:
    size = line_mask.shape[0]
    index = 0
    while index < size:
        if not line_mask[index]:
            index += 1
            continue

        start = index
        while index < size and line_mask[index]:
            index += 1
        end = index

        left = _nearest_unmasked_index(line_mask, start - 1, -1)
        right = _nearest_unmasked_index(line_mask, end, 1)
        if left is None and right is None:
            continue

        if left is not None and right is not None:
            span = np.arange(start, end, dtype=np.float32)
            denom = float(right - left)
            weights = ((span - float(left)) / denom).reshape(-1, 1)
            write_values[start:end] = values[left] * (1.0 - weights) + values[right] * weights
        elif left is not None:
            write_values[start:end] = values[left]
        else:
            write_values[start:end] = values[right]
        write_filled[start:end] = True


def _nearest_unmasked_index(mask: np.ndarray, start: int, step: int) -> int | None:
    index = start
    while 0 <= index < mask.shape[0]:
        if not mask[index]:
            return int(index)
        index += step
    return None



def _match_repair_to_context(
    roi: np.ndarray,
    candidate: np.ndarray,
    repair_mask: np.ndarray,
    *,
    strength: float,
    radius: int,
    min_context_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    repair = np.asarray(repair_mask, dtype=bool)
    stats: dict[str, Any] = {
        "color_match_component_count": 0,
        "color_match_pixel_count": 0,
        "color_match_context_pixel_count": 0,
        "tone_guided_component_count": 0,
        "tone_guided_pixel_count": 0,
        "tone_guided_fallback_count": 0,
        "tone_guided_no_context_count": 0,
        "tone_guided_candidate_count_total": 0,
        "tone_guided_top_k_total": 0,
        "tone_guided_top_k_count": 0,
        "tone_guided_top_k_mean": 0.0,
        "tone_guided_score_total": 0.0,
        "tone_guided_score_count": 0,
        "tone_guided_score_mean": 0.0,
        "tone_guided_context_rgb_distance_total": 0.0,
        "tone_guided_context_rgb_distance_count": 0,
        "tone_guided_context_rgb_distance_mean": 0.0,
        "guard_rejected_core_pixel_count": 0,
        "guard_rejected_shell_pixel_count": 0,
        "guard_core_fallback_success_count": 0,
        "guard_core_unrepaired_pixel_count": 0,
    }
    if strength <= 0.0 or not repair.any():
        return candidate, stats

    context = _context_ring(repair, radius) if radius > 0 else ~repair
    context_count = int(np.count_nonzero(context))
    stats["color_match_context_pixel_count"] = context_count
    if context_count < min_context_pixels:
        return candidate, stats

    roi_float = as_float32(roi)
    candidate_float = as_float32(candidate)
    if roi_float.ndim != 3 or candidate_float.ndim != 3 or roi_float.shape[2] < 3 or candidate_float.shape[2] < 3:
        return candidate, stats

    context_rgb = roi_float[context, :3]
    repair_rgb = candidate_float[repair, :3]
    if context_rgb.size == 0 or repair_rgb.size == 0:
        return candidate, stats

    context_mean = np.mean(context_rgb, axis=0)
    repair_mean = np.mean(repair_rgb, axis=0)
    context_std = np.std(context_rgb, axis=0)
    repair_std = np.std(repair_rgb, axis=0)
    scale = np.where(repair_std > 1.0e-5, context_std / np.maximum(repair_std, 1.0e-5), 1.0)
    scale = np.clip(scale, 0.25, 4.0)

    matched = (repair_rgb - repair_mean) * scale + context_mean
    q_low = np.percentile(context_rgb, 2.0, axis=0)
    q_high = np.percentile(context_rgb, 98.0, axis=0)
    context_mad = np.median(np.abs(context_rgb - np.median(context_rgb, axis=0)), axis=0)
    margin = np.maximum(context_mad * 3.0, 0.025)
    matched = np.clip(matched, q_low - margin, q_high + margin)

    amount = float(np.clip(strength, 0.0, 1.0))
    output = candidate_float.copy()
    output[repair, :3] = repair_rgb * (1.0 - amount) + matched * amount
    output[:, :, :3] = np.clip(output[:, :, :3], 0.0, 1.0)
    if output.shape[2] == 4:
        output[:, :, 3] = candidate_float[:, :, 3]

    stats["color_match_component_count"] = 1
    stats["color_match_pixel_count"] = int(np.count_nonzero(repair))
    if np.issubdtype(np.asarray(candidate).dtype, np.floating):
        return output.astype(np.asarray(candidate).dtype, copy=False), stats
    return restore_dtype(output, np.asarray(candidate).dtype), stats

def _guard_repair_candidate(
    roi: np.ndarray,
    candidate: np.ndarray,
    repair_mask: np.ndarray,
    core_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    stats = {
        "guard_rejected_core_pixel_count": 0,
        "guard_rejected_shell_pixel_count": 0,
        "guard_core_fallback_success_count": 0,
        "guard_core_unrepaired_pixel_count": 0,
    }
    repair = np.asarray(repair_mask, dtype=bool)
    if not repair.any():
        return candidate, np.zeros(repair.shape, dtype=bool), stats

    context = ~repair
    if not context.any():
        # No trusted pixels are available. Keep the generated candidate instead of
        # restoring the masked source pixels, because masked pixels are assumed defective.
        return candidate, np.zeros(repair.shape, dtype=bool), stats

    ring = _context_ring(repair, radius=10)
    sample = roi[ring] if ring.any() else roi[context]
    if sample.size == 0:
        return candidate, np.zeros(repair.shape, dtype=bool), stats

    q_low = np.percentile(sample, 3.0, axis=0)
    q_high = np.percentile(sample, 97.0, axis=0)
    context_median = np.median(sample, axis=0)
    context_mad = np.median(np.abs(sample - context_median), axis=0)
    channel_margin = np.maximum(context_mad * 3.0, 0.03)
    lower = q_low - channel_margin
    upper = q_high + channel_margin

    guarded = candidate.copy()
    clipped_values = np.clip(guarded[repair], lower, upper)
    out_of_range = np.any(np.abs(clipped_values - guarded[repair]) > 1.0e-6, axis=1)
    guarded[repair] = clipped_values
    clipped_mask = np.zeros(repair.shape, dtype=bool)
    clipped_mask[repair] = out_of_range

    rejected = clipped_mask
    if core_mask is not None:
        core = np.asarray(core_mask, dtype=bool)
        core_reject = rejected & core
        shell_reject = rejected & ~core
    else:
        core = repair
        core_reject = rejected
        shell_reject = np.zeros(repair.shape, dtype=bool)

    if np.any(core_reject):
        fallback, fallback_valid = _guard_core_fallback_candidate(roi, guarded, repair, core)
        success = core_reject & fallback_valid
        guarded[success] = fallback[success]
        stats["guard_core_fallback_success_count"] = int(np.count_nonzero(success))
        # Do not restore failed pixels from roi; keep the context-clamped candidate.
        stats["guard_core_unrepaired_pixel_count"] = 0

    stats["guard_rejected_core_pixel_count"] = int(np.count_nonzero(core_reject))
    stats["guard_rejected_shell_pixel_count"] = int(np.count_nonzero(shell_reject))
    return guarded, rejected, stats


def _guard_core_fallback_candidate(
    roi: np.ndarray,
    candidate: np.ndarray,
    repair_mask: np.ndarray,
    core_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    context = _context_ring(repair_mask, radius=8)
    if not context.any():
        context = ~repair_mask
    if int(np.count_nonzero(context)) < 3:
        return candidate, np.zeros(repair_mask.shape, dtype=bool)

    fallback = _blend_convolution_and_plane(roi, repair_mask)
    plane = _fit_local_plane_fill(roi, repair_mask, context)
    if plane is not None:
        fallback[core_mask] = plane[core_mask]
    else:
        median = np.median(roi[context], axis=0)
        fallback[core_mask] = median
    fallback = _clamp_candidate_to_context(fallback, core_mask, roi[context])
    valid = np.zeros(repair_mask.shape, dtype=bool)
    valid[core_mask] = True
    return fallback.astype(np.float32, copy=False), valid


def _clamp_candidate_to_context(candidate: np.ndarray, mask: np.ndarray, context_values: np.ndarray) -> np.ndarray:
    if context_values.size == 0 or not np.any(mask):
        return candidate
    median = np.median(context_values, axis=0)
    mad = np.median(np.abs(context_values - median), axis=0)
    q_low = np.percentile(context_values, 2.0, axis=0)
    q_high = np.percentile(context_values, 98.0, axis=0)
    margin = np.maximum(mad * 4.0, 0.03)
    out = candidate.copy()
    out[mask] = np.clip(out[mask], q_low - margin, q_high + margin)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _masked_denoise(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return roi.copy()
    blurred = _box_blur_image(roi, radius=1)
    out = roi.copy()
    out[mask] = blurred[mask]
    return out


def _make_blend_alpha(core_mask: np.ndarray, repair_mask: np.ndarray, feather_radius: int) -> np.ndarray:
    core = np.asarray(core_mask, dtype=bool)
    repair = np.asarray(repair_mask, dtype=bool)
    alpha = np.zeros(core.shape, dtype=np.float32)
    if not repair.any():
        return alpha
    shell = repair & ~core
    alpha[core] = 1.0
    if shell.any():
        if feather_radius > 0:
            blurred_core = _box_blur_scalar(core.astype(np.float32), feather_radius)
            shell_alpha = np.clip(blurred_core, 0.15, 0.45)
            alpha[shell] = shell_alpha[shell]
        else:
            alpha[shell] = 0.25
    return np.clip(alpha, 0.0, 1.0)


def _context_ring(mask: np.ndarray, radius: int) -> np.ndarray:
    expanded = dilate_mask(mask, radius)
    return expanded & ~mask


def _component_seam_score(roi: np.ndarray, candidate: np.ndarray, repair_mask: np.ndarray) -> float:
    outer = _context_ring(repair_mask, radius=1)
    if not outer.any():
        return 0.0
    inner = repair_mask & dilate_mask(outer, 1)
    if not inner.any():
        inner = repair_mask
    outer_mean = np.mean(roi[outer], axis=0)
    diff = np.mean(np.abs(candidate[inner] - outer_mean))
    return float(diff)


def _alpha_scale_from_seam(seam_score: float) -> float:
    if seam_score > 0.16:
        return 0.45
    if seam_score > 0.10:
        return 0.65
    if seam_score > 0.07:
        return 0.82
    return 1.0


def _component_features(mask: np.ndarray) -> dict[str, float | bool]:
    ys, xs = np.nonzero(mask)
    area = int(len(xs))
    if area == 0:
        return {"area": 0.0, "aspect_ratio": 1.0, "pca_ratio": 1.0, "fill_ratio": 0.0, "is_slender": False}
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    aspect = max(width, height) / float(max(1, min(width, height)))
    fill_ratio = area / float(max(1, width * height))
    pca_ratio = _pca_ratio(mask)
    is_slender = bool((aspect >= 3.0 or pca_ratio >= 3.2) and area >= 3 and fill_ratio <= 0.65)
    return {
        "area": float(area),
        "aspect_ratio": float(aspect),
        "pca_ratio": float(pca_ratio),
        "fill_ratio": float(fill_ratio),
        "is_slender": is_slender,
    }


def _principal_direction(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    coords -= np.mean(coords, axis=0)
    cov = np.cov(coords, rowvar=False)
    try:
        values, vectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    vector = vectors[:, int(np.argmax(values))]
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-6:
        return None
    vector = vector / norm
    return float(vector[0]), float(vector[1])


def _pca_ratio(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return 1.0
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    coords -= np.mean(coords, axis=0)
    cov = np.cov(coords, rowvar=False)
    try:
        values = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 1.0
    values = np.maximum(values, 1.0e-6)
    return float(np.sqrt(np.max(values) / np.min(values)))


def _plane_context_residual(roi: np.ndarray, plane: np.ndarray, context: np.ndarray) -> float:
    if not context.any():
        return 1.0
    return float(np.median(np.abs(roi[context] - plane[context])))


def _trimmed_inliers(values: np.ndarray) -> np.ndarray:
    if values.size < 8:
        return np.ones(values.shape, dtype=bool)
    low = np.percentile(values, 5.0)
    high = np.percentile(values, 95.0)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    mad_margin = max(0.03, float(mad) * 3.5)
    return (values >= low) & (values <= high) & (np.abs(values - median) <= mad_margin)


def _trimmed_context_mean(values: np.ndarray, context_mask: np.ndarray) -> np.ndarray:
    sample = values[context_mask]
    if sample.size == 0:
        return np.zeros(values.shape[2], dtype=np.float32)
    out = np.zeros(values.shape[2], dtype=np.float32)
    for channel in range(values.shape[2]):
        channel_values = sample[:, channel]
        if channel_values.size >= 8:
            low = np.percentile(channel_values, 10.0)
            high = np.percentile(channel_values, 90.0)
            trimmed = channel_values[(channel_values >= low) & (channel_values <= high)]
            out[channel] = float(np.mean(trimmed)) if trimmed.size else float(np.median(channel_values))
        else:
            out[channel] = float(np.median(channel_values))
    return out


def _luminance(image: np.ndarray) -> np.ndarray:
    return (
        image[:, :, 0] * 0.2126
        + image[:, :, 1] * 0.7152
        + image[:, :, 2] * 0.0722
    ).astype(np.float32)


def _box_blur_image(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.astype(np.float32)
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    result = np.zeros_like(image, dtype=np.float32)
    count = 0
    size = radius * 2 + 1
    for dy in range(size):
        for dx in range(size):
            result += padded[dy : dy + image.shape[0], dx : dx + image.shape[1], :]
            count += 1
    return result / float(count)


def _box_blur_scalar(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float32)
    padded = np.pad(values, radius, mode="edge")
    result = np.zeros_like(values, dtype=np.float32)
    count = 0
    size = radius * 2 + 1
    for dy in range(size):
        for dx in range(size):
            result += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
            count += 1
    return result / float(count)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _repair_metrics(
    *,
    original: np.ndarray,
    repaired: np.ndarray,
    blend_alpha: np.ndarray,
    core_mask: np.ndarray,
    repair_mask: np.ndarray,
    guard_rejected_mask: np.ndarray,
    changed_bbox_list: list[tuple[int, int, int, int]],
    processing_time_ms: float,
    channel_used: str,
    filtered: Any,
    component_alpha_values: list[float],
    low_confidence_component_count: int,
) -> dict[str, Any]:
    metrics = compute_repair_metrics(
        original,
        repaired,
        blend_alpha,
        changed_bbox_list,
        processing_time_ms,
    )
    metrics.update(_mask_metrics(channel_used, filtered))
    metrics.update(
        _extended_repair_metrics(
            original=original,
            repaired=repaired,
            core_mask=core_mask,
            repair_mask=repair_mask,
            guard_rejected_mask=guard_rejected_mask,
            component_alpha_values=component_alpha_values,
            low_confidence_component_count=low_confidence_component_count,
        )
    )
    return metrics


def _add_defect_aware_metrics(
    metrics: dict[str, Any],
    method: str,
    *,
    skipped: bool,
    defect_features: list[DefectFeatures] | None,
    defect_repair_stats: dict[str, Any],
) -> None:
    if method != "defect_aware":
        return
    component_count = int(metrics.get("kept_component_count", 0))
    if defect_features is not None:
        defect_summary = summarize_defect_features(defect_features)
        strategy_counts = defect_summary["defect_strategy_counts"]
        metrics.update(defect_summary)
    elif skipped:
        strategy_counts = {"skipped": component_count}
    else:
        strategy_counts = {f"{_DEFECT_AWARE_FALLBACK_METHOD}_fallback": component_count}
    metrics.update(
        {
            "defect_aware": True,
            "defect_aware_version": _DEFECT_AWARE_VERSION,
            "defect_strategy_counts": strategy_counts,
            "defect_aware_fallback_method": _DEFECT_AWARE_FALLBACK_METHOD,
            "defect_processing_skipped": bool(skipped),
        }
    )
    metrics.update(defect_repair_stats)
    patch_score_count = int(metrics.get("patch_best_score_count", 0))
    if patch_score_count > 0:
        metrics["patch_best_score_mean"] = float(metrics.get("patch_best_score_total", 0.0)) / float(patch_score_count)
    else:
        metrics["patch_best_score_mean"] = 0.0
    edge_coherence_count = int(metrics.get("small_local_edge_guided_coherence_count", 0))
    if edge_coherence_count > 0:
        metrics["small_local_edge_guided_coherence_mean"] = float(
            metrics.get("small_local_edge_guided_coherence_total", 0.0)
        ) / float(edge_coherence_count)
    else:
        metrics["small_local_edge_guided_coherence_mean"] = 0.0
    tone_top_k_count = int(metrics.get("tone_guided_top_k_count", 0))
    metrics["tone_guided_top_k_mean"] = (
        float(metrics.get("tone_guided_top_k_total", 0.0)) / float(tone_top_k_count)
        if tone_top_k_count > 0
        else 0.0
    )
    tone_score_count = int(metrics.get("tone_guided_score_count", 0))
    metrics["tone_guided_score_mean"] = (
        float(metrics.get("tone_guided_score_total", 0.0)) / float(tone_score_count)
        if tone_score_count > 0
        else 0.0
    )
    tone_distance_count = int(metrics.get("tone_guided_context_rgb_distance_count", 0))
    metrics["tone_guided_context_rgb_distance_mean"] = (
        float(metrics.get("tone_guided_context_rgb_distance_total", 0.0)) / float(tone_distance_count)
        if tone_distance_count > 0
        else 0.0
    )
    add_frequency_means(metrics)


def _empty_defect_repair_stats() -> dict[str, Any]:
    return {
        "small_local_component_count": 0,
        "small_local_pixel_count": 0,
        "small_local_plane_count": 0,
        "small_local_median_count": 0,
        "small_local_fallback_count": 0,
        "small_local_edge_guided_component_count": 0,
        "small_local_edge_guided_pixel_count": 0,
        "small_local_edge_guided_fallback_count": 0,
        "small_local_edge_guided_low_confidence_count": 0,
        "small_local_edge_guided_coherence_total": 0.0,
        "small_local_edge_guided_coherence_count": 0,
        "small_local_edge_guided_coherence_mean": 0.0,
        "fast_inpaint_component_count": 0,
        "fast_inpaint_pixel_count": 0,
        "fast_inpaint_iterations_total": 0,
        "fast_inpaint_fallback_count": 0,
        "directional_component_count": 0,
        "directional_pixel_count": 0,
        "directional_fallback_count": 0,
        "directional_cap_exceeded_count": 0,
        "patch_component_count": 0,
        "patch_pixel_count": 0,
        "patch_candidate_count_total": 0,
        "patch_fallback_count": 0,
        "patch_best_score_total": 0.0,
        "patch_best_score_count": 0,
        "patch_best_score_mean": 0.0,
        "patch_stride_used_counts": {},
        "grain_reinject_enabled": False,
        "grain_reinject_strength": 0.0,
        "grain_reinject_component_count": 0,
        "grain_reinject_pixel_count": 0,
        "grain_reinject_skipped_no_context_count": 0,
        "color_match_component_count": 0,
        "color_match_pixel_count": 0,
        "color_match_context_pixel_count": 0,
        "defect_fallback_count": 0,
    }


def _merge_defect_repair_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, bool) or isinstance(target.get(key), bool):
            target[key] = bool(target.get(key, False) or value)
            continue
        if isinstance(value, dict):
            current = target.setdefault(key, {})
            if not isinstance(current, dict):
                current = {}
                target[key] = current
            for nested_key, nested_value in value.items():
                current[str(nested_key)] = int(current.get(str(nested_key), 0)) + int(nested_value)
            continue
        if isinstance(value, float) or isinstance(target.get(key), float):
            target[key] = float(target.get(key, 0.0)) + float(value)
        else:
            target[key] = int(target.get(key, 0)) + int(value)


def _add_defect_aware_blend_metrics(
    metrics: dict[str, Any],
    method: str,
    *,
    final_alpha: np.ndarray,
    core_mask: np.ndarray,
    repair_mask: np.ndarray,
) -> None:
    if method != "defect_aware":
        return
    alpha = np.asarray(final_alpha, dtype=np.float32)
    alpha_nonzero = alpha > 0.0
    core = np.asarray(core_mask, dtype=bool)
    shell_mask = np.asarray(repair_mask, dtype=bool) & ~core
    core_alpha = alpha[core]
    shell_alpha = alpha[shell_mask]
    metrics.update(
        {
            "defect_aware_blend_shell_pixel_count": int(np.count_nonzero(alpha_nonzero & shell_mask)),
            "defect_aware_alpha_nonzero_pixel_count": int(np.count_nonzero(alpha_nonzero)),
            "defect_core_alpha_min": _min_or_zero(core_alpha),
            "defect_core_alpha_mean": _mean_or_zero(core_alpha),
            "defect_core_alpha_max": _max_or_zero(core_alpha),
            "defect_core_alpha_below_full_count": int(np.count_nonzero(core_alpha < 0.999999)),
            "defect_shell_alpha_min": _min_or_zero(shell_alpha),
            "defect_shell_alpha_mean": _mean_or_zero(shell_alpha),
            "defect_shell_alpha_max": _max_or_zero(shell_alpha),
        }
    )


def _extended_repair_metrics(
    *,
    original: np.ndarray,
    repaired: np.ndarray,
    core_mask: np.ndarray,
    repair_mask: np.ndarray,
    guard_rejected_mask: np.ndarray,
    component_alpha_values: list[float],
    low_confidence_component_count: int,
) -> dict[str, float | int]:
    original_float = as_float32(original)
    repaired_float = as_float32(repaired)
    diff = np.abs(repaired_float[:, :, :3] - original_float[:, :, :3])
    per_pixel_diff = np.max(diff, axis=2)
    changed = per_pixel_diff > 0.0
    shell_mask = repair_mask & ~core_mask
    outside_original = ~core_mask
    outside_repair = ~repair_mask
    return {
        "core_mask_pixel_count": int(np.count_nonzero(core_mask)),
        "repair_mask_pixel_count": int(np.count_nonzero(repair_mask)),
        "changed_pixel_count_core": int(np.count_nonzero(changed & core_mask)),
        "changed_pixel_count_shell": int(np.count_nonzero(changed & shell_mask)),
        "mean_abs_diff_core": _mean_or_zero(diff[core_mask]),
        "mean_abs_diff_shell": _mean_or_zero(diff[shell_mask]),
        "max_abs_diff_outside_original_mask": _max_or_zero(per_pixel_diff[outside_original]),
        "max_abs_diff_outside_repair_mask": _max_or_zero(per_pixel_diff[outside_repair]),
        "guard_rejected_pixel_count": int(np.count_nonzero(guard_rejected_mask)),
        "average_component_alpha": float(np.mean(component_alpha_values)) if component_alpha_values else 0.0,
        "low_confidence_component_count": int(low_confidence_component_count),
    }


def _min_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.min(values))


def _mean_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _max_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))


def _mask_metrics(channel_used: str, filtered: Any) -> dict[str, int | str]:
    return {
        "mask_channel_used": channel_used,
        "kept_component_count": len(filtered.kept_components),
        "removed_small_component_count": len(filtered.removed_small),
        "removed_large_component_count": len(filtered.removed_large),
    }


def _debug_images(
    *,
    original: np.ndarray,
    repaired: np.ndarray,
    normalized_mask: np.ndarray,
    core_mask: np.ndarray,
    repair_mask: np.ndarray,
    blend_alpha: np.ndarray,
    candidate_before_guard: np.ndarray,
    candidate_after_guard: np.ndarray,
    guard_rejected_mask: np.ndarray,
    output_dtype: np.dtype,
) -> dict[str, np.ndarray]:
    original_float = as_float32(original)
    repaired_float = as_float32(repaired)
    diff = np.max(np.abs(repaired_float[:, :, :3] - original_float[:, :, :3]), axis=2)
    diff_boosted = np.clip(diff * 8.0, 0.0, 1.0)
    diff_visualization = np.zeros((*diff.shape, 3), dtype=np.uint8)
    diff_visualization[:, :, 0] = np.rint(diff_boosted * 255.0).astype(np.uint8)
    diff_visualization[:, :, 1] = np.rint(np.clip(blend_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    shell_mask = repair_mask & ~core_mask

    return {
        "normalized_mask": np.rint(np.clip(normalized_mask, 0.0, 1.0) * 255.0).astype(np.uint8),
        "binary_mask": (repair_mask.astype(np.uint8) * 255),
        "soft_mask": np.rint(np.clip(blend_alpha, 0.0, 1.0) * 255.0).astype(np.uint8),
        "core_mask": (core_mask.astype(np.uint8) * 255),
        "repair_mask": (repair_mask.astype(np.uint8) * 255),
        "blend_alpha": np.rint(np.clip(blend_alpha, 0.0, 1.0) * 255.0).astype(np.uint8),
        "candidate_before_guard": restore_dtype(candidate_before_guard, output_dtype),
        "candidate_after_guard": restore_dtype(candidate_after_guard, output_dtype),
        "rejected_by_guard": (guard_rejected_mask.astype(np.uint8) * 255),
        "shell_mask": (shell_mask.astype(np.uint8) * 255),
        "repaired_preview": repaired,
        "diff_visualization": diff_visualization,
    }


def _write_debug_outputs(debug_dir: str | Path | None, result: RepairResult) -> None:
    if debug_dir is None:
        return
    output_dir = Path(debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "normalized_mask": "normalized_mask.png",
        "binary_mask": "binary_mask.png",
        "soft_mask": "soft_mask.png",
        "core_mask": "core_mask.png",
        "repair_mask": "repair_mask.png",
        "blend_alpha": "blend_alpha.png",
        "candidate_before_guard": "candidate_before_guard.png",
        "candidate_after_guard": "candidate_after_guard.png",
        "rejected_by_guard": "rejected_by_guard.png",
        "shell_mask": "shell_mask.png",
        "repaired_preview": "repaired_preview.png",
        "diff_visualization": "diff_visualization.png",
    }
    for key, filename in names.items():
        path = output_dir / filename
        write_image(path, result.debug_images[key])
        result.debug_paths[key] = str(path)
    optional_names = {
        "frequency_scope_mask": "frequency_scope_mask.png",
        "frequency_selected_core_mask": "frequency_selected_core_mask.png",
        "frequency_selected_overlay": "frequency_selected_overlay.png",
        "frequency_pattern_map": "frequency_pattern_map.png",
    }
    for key, filename in optional_names.items():
        if key not in result.debug_images:
            continue
        path = output_dir / filename
        write_image(path, result.debug_images[key])
        result.debug_paths[key] = str(path)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2, sort_keys=True), encoding="utf-8")
    result.debug_paths["metrics"] = str(metrics_path)


def _write_defect_debug_outputs(
    debug_dir: str | Path | None,
    result: RepairResult,
    defect_features: list[DefectFeatures] | None,
) -> None:
    if debug_dir is None or defect_features is None:
        return
    output_dir = Path(debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "defect_components.json"
    payload = defect_debug_payload(defect_features)
    if result.defect_component_repairs:
        for component in payload.get("components", []):
            label = int(component.get("label", 0))
            repair_debug = result.defect_component_repairs.get(label)
            if repair_debug:
                component.update(repair_debug)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    result.debug_paths["defect_components"] = str(path)

    summary_path = output_dir / "defect_strategy_summary.json"
    summary_path.write_text(json.dumps(payload["summary"], indent=2, sort_keys=True), encoding="utf-8")
    result.debug_paths["defect_strategy_summary"] = str(summary_path)


def _frequency_debug_images(
    original: np.ndarray,
    selection: Any,
    pattern_map: np.ndarray,
) -> dict[str, np.ndarray]:
    if selection.scope_mask is None:
        return {}
    scope = (selection.scope_mask.astype(np.uint8) * 255)
    selected = (selection.selected_core_mask.astype(np.uint8) * 255)
    preview = restore_dtype(as_float32(original), np.uint8)
    overlay = preview[:, :, :3].copy()
    active = selection.selected_core_mask
    if np.any(active):
        color = np.asarray([255.0, 255.0, 255.0], dtype=np.float32)
        overlay[active] = np.rint(overlay[active].astype(np.float32) * 0.35 + color * 0.65).astype(np.uint8)
    return {
        "frequency_scope_mask": scope,
        "frequency_selected_core_mask": selected,
        "frequency_selected_overlay": overlay,
        "frequency_pattern_map": np.asarray(pattern_map, dtype=np.uint8),
    }


def _write_frequency_debug_outputs(debug_dir: str | Path | None, result: RepairResult) -> None:
    if debug_dir is None:
        return
    components = []
    for label, payload in sorted(result.defect_component_repairs.items()):
        if not payload.get("frequency_guided_selected"):
            continue
        components.append({"label": int(label), **payload})
    if not components:
        return
    output_dir = Path(debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "frequency_components.json"
    payload = {
        "version": 1,
        "component_count": len(components),
        "components": components,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    result.debug_paths["frequency_components"] = str(path)
