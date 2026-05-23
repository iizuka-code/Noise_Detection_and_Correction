from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .io import as_float32, read_image, write_image
from .mask import connected_components


SUPPORTED_RED_HIGHLIGHT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
COMPONENT_SAMPLE_LIMIT = 50


@dataclass(frozen=True)
class RedHighlightConfig:
    detection_long_edge: int = 1920
    local_radius: int = 0
    mask_edge_mode: str = "normal"
    full_resolution_refine: bool = True
    red_min: float | None = None
    red_excess_min: float | None = None
    red_ratio_min: float | None = None
    contrast_red_min: float | None = None
    contrast_excess_min: float | None = None
    glow_signal_min: float | None = None
    contrast_glow_min: float | None = None
    hot_core_value_min: float | None = None
    hot_core_contrast_min: float | None = None
    threshold_sensitivity: float = 1.0
    min_area: int = 1
    max_area: int = 1400
    max_dim: int = 95
    max_aspect: float = 11.5
    include_long_scratches: bool = False
    min_scratch_aspect: float = 5.0
    max_scratch_area: int = 9000
    max_scratch_dim: int = 720
    max_scratch_width: int = 48
    hot_core_max_ratio_relax: float = 0.55
    suppress_border_glow: bool = True
    border_x_fraction: float = 0.045
    border_y_fraction: float = 0.02
    debug_artifacts: bool = False

    def validate(self) -> None:
        if self.detection_long_edge <= 0:
            raise ValueError("detection_long_edge must be > 0")
        if self.local_radius < 0:
            raise ValueError("local_radius must be >= 0")
        if _normalize_mask_edge_mode(self.mask_edge_mode) not in {"tight", "normal", "wide", "legacy"}:
            raise ValueError(f"Unsupported mask_edge_mode: {self.mask_edge_mode}")
        if self.threshold_sensitivity <= 0.0:
            raise ValueError("threshold_sensitivity must be > 0")
        if self.min_area < 0:
            raise ValueError("min_area must be >= 0")
        if self.max_area < self.min_area:
            raise ValueError("max_area must be >= min_area")
        if self.max_dim <= 0:
            raise ValueError("max_dim must be > 0")
        if self.max_aspect <= 0.0:
            raise ValueError("max_aspect must be > 0")
        if self.min_scratch_aspect <= 0.0:
            raise ValueError("min_scratch_aspect must be > 0")
        if self.max_scratch_area < self.min_area:
            raise ValueError("max_scratch_area must be >= min_area")
        if self.max_scratch_dim <= 0:
            raise ValueError("max_scratch_dim must be > 0")
        if self.max_scratch_width <= 0:
            raise ValueError("max_scratch_width must be > 0")
        if self.border_x_fraction < 0.0 or self.border_y_fraction < 0.0:
            raise ValueError("border fractions must be >= 0")


@dataclass(frozen=True)
class RedHighlightResult:
    mask: np.ndarray
    preview_mask: np.ndarray
    overlay_preview: np.ndarray
    score_map: np.ndarray
    components: list[dict[str, Any]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class RedHighlightSourceResult:
    mask: np.ndarray
    preview_mask: np.ndarray
    overlay_preview: np.ndarray
    overlay: np.ndarray
    score_map: np.ndarray
    components: list[dict[str, Any]]
    manifest: dict[str, Any]


def detect_red_highlight_source_image(
    image: np.ndarray,
    config: RedHighlightConfig | None = None,
) -> RedHighlightSourceResult:
    cfg = config or RedHighlightConfig()
    cfg.validate()
    started = perf_counter()

    source_rgb = _ensure_rgb_u8(image)
    detection_rgb = _resize_long_edge_rgb(source_rgb, cfg.detection_long_edge)
    detection = detect_red_highlight_mask(detection_rgb, cfg)

    refine_started = perf_counter()
    final_mask, final_refine = _final_mask_for_source(source_rgb, detection.preview_mask, cfg)
    final_refine_ms = _elapsed_ms(refine_started)
    overlay = _overlay_mask(source_rgb, final_mask > 0)

    manifest = dict(detection.manifest)
    full_components = list(manifest.get("components", []))
    manifest["component_sample"] = full_components[:COMPONENT_SAMPLE_LIMIT]
    manifest["component_features_in_manifest"] = len(full_components) <= COMPONENT_SAMPLE_LIMIT
    if len(full_components) > COMPONENT_SAMPLE_LIMIT:
        manifest["component_sample_truncated"] = len(full_components) - COMPONENT_SAMPLE_LIMIT
    manifest.pop("components", None)
    manifest.update(
        {
            "detector_version": "red_highlight_v1",
            "source_shape": [int(source_rgb.shape[0]), int(source_rgb.shape[1])],
            "detection_shape": [int(detection_rgb.shape[0]), int(detection_rgb.shape[1])],
            "final_mask_pixels": int(np.count_nonzero(final_mask)),
            "final_refine": final_refine,
            "timings_ms": {
                "final_refine": final_refine_ms,
                "total_ms": _elapsed_ms(started),
            },
        }
    )
    return RedHighlightSourceResult(
        mask=final_mask,
        preview_mask=detection.preview_mask,
        overlay_preview=detection.overlay_preview,
        overlay=overlay,
        score_map=detection.score_map,
        components=detection.components,
        manifest=manifest,
    )


def run_red_highlight_detector(
    source: str | Path,
    output_dir: str | Path,
    config: RedHighlightConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RedHighlightConfig()
    cfg.validate()
    source_path = Path(source)
    if source_path.suffix.lower() not in SUPPORTED_RED_HIGHLIGHT_EXTENSIONS:
        raise ValueError(f"Unsupported red-highlight source extension: {source_path.suffix.lower()}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    load_started = perf_counter()
    image = read_image(source_path)
    source_rgb = _ensure_rgb_u8(image.pixels)
    load_ms = _elapsed_ms(load_started)

    result = detect_red_highlight_source_image(source_rgb, cfg)

    output_started = perf_counter()
    write_image(output_path / "input_preview.png", _resize_long_edge_rgb(source_rgb, cfg.detection_long_edge))
    write_image(output_path / "mask.png", result.mask)
    write_image(output_path / f"{_safe_stem(source_path)}_red_highlight_mask.png", result.mask)
    write_image(output_path / "preview_mask.png", result.preview_mask)
    write_image(output_path / "overlay_preview.png", result.overlay_preview)
    write_image(output_path / "overlay.png", result.overlay)
    if cfg.debug_artifacts:
        write_image(output_path / "local_red_contrast_score.png", result.score_map)
    component_json = output_path / "component_features.json"
    component_json.write_text(
        json.dumps({"components": result.components}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_ms = _elapsed_ms(output_started)

    manifest = dict(result.manifest)
    manifest["source"] = str(source_path)
    manifest["source_mode"] = image.color_mode
    manifest["output_dir"] = str(output_path)
    manifest["artifacts"] = {
        "mask": str(output_path / "mask.png"),
        "named_mask": str(output_path / f"{_safe_stem(source_path)}_red_highlight_mask.png"),
        "input_preview": str(output_path / "input_preview.png"),
        "preview_mask": str(output_path / "preview_mask.png"),
        "overlay_preview": str(output_path / "overlay_preview.png"),
        "overlay": str(output_path / "overlay.png"),
        "component_features_json": str(component_json),
    }
    if cfg.debug_artifacts:
        manifest["artifacts"]["local_red_contrast_score"] = str(output_path / "local_red_contrast_score.png")
    timings = dict(manifest.get("timings_ms", {}))
    timings.update({"load_decode": load_ms, "output_save": output_ms})
    manifest["timings_ms"] = timings
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def detect_red_highlight_mask(
    image: np.ndarray,
    config: RedHighlightConfig | None = None,
) -> RedHighlightResult:
    cfg = config or RedHighlightConfig()
    cfg.validate()
    rgb = _ensure_rgb_u8(image)
    height, width = rgb.shape[:2]

    work = rgb.astype(np.float32)
    red = work[:, :, 0]
    green = work[:, :, 1]
    blue = work[:, :, 2]
    max_green_blue = np.maximum(green, blue)
    value = np.maximum.reduce([red, green, blue])
    red_excess = np.maximum(red - max_green_blue, 0.0)
    red_ratio = red_excess / np.maximum(red, 1.0)
    glow_signal = np.maximum.reduce(
        [
            red_excess,
            red - (green * 0.62 + blue * 0.28),
            red * 0.70 - np.maximum(green, blue) * 0.34,
        ]
    )
    glow_signal = np.maximum(glow_signal, 0.0).astype(np.float32)

    local_radius = int(cfg.local_radius) if cfg.local_radius > 0 else max(5, int(round(max(height, width) / 275.0)))
    local_red = _box_blur(red, radius=local_radius)
    local_excess = _box_blur(red_excess, radius=local_radius)
    local_value = _box_blur(value, radius=local_radius)
    local_glow = _box_blur(glow_signal, radius=local_radius)
    contrast_red = np.maximum(red - local_red, 0.0)
    contrast_excess = np.maximum(red_excess - local_excess, 0.0)
    contrast_value = np.maximum(value - local_value, 0.0)
    contrast_glow = np.maximum(glow_signal - local_glow, 0.0)
    thresholds = _resolve_thresholds(
        config=cfg,
        red=red,
        red_excess=red_excess,
        red_ratio=red_ratio,
        glow_signal=glow_signal,
        contrast_red=contrast_red,
        contrast_excess=contrast_excess,
        contrast_glow=contrast_glow,
        contrast_value=contrast_value,
    )

    red_dominant = (
        (red >= thresholds["red_min"])
        & (red_ratio >= thresholds["red_ratio_min"])
        & ((red_excess >= thresholds["red_excess_min"]) | (glow_signal >= thresholds["glow_signal_min"]))
    )
    local_peak = (
        (contrast_red >= thresholds["contrast_red_min"])
        | (contrast_excess >= thresholds["contrast_excess_min"])
        | (contrast_glow >= thresholds["contrast_glow_min"])
    )
    normal_seed = red_dominant & local_peak
    strong_seed = (
        (red >= thresholds["strong_red_min"])
        & (red_ratio >= thresholds["strong_ratio_min"])
        & ((red_excess >= thresholds["strong_excess_min"]) | (glow_signal >= thresholds["strong_glow_min"]))
        & (
            (contrast_red >= thresholds["strong_contrast_min"])
            | (contrast_excess >= thresholds["strong_contrast_min"] * 0.8)
            | (contrast_glow >= thresholds["strong_contrast_min"] * 0.8)
            | (contrast_value >= thresholds["strong_contrast_min"] * 1.6)
        )
    )
    red_anchor = normal_seed | strong_seed
    near_red_anchor = _max_filter_bool(red_anchor, max(7, int(round(local_radius * 1.4)) | 1))
    hot_core_seed = (
        (value >= thresholds["hot_core_value_min"])
        & (red >= thresholds["hot_core_red_min"])
        & (contrast_value >= thresholds["hot_core_contrast_min"])
        & (
            near_red_anchor
            | (
                (red_ratio >= thresholds["red_ratio_min"] * float(cfg.hot_core_max_ratio_relax))
                & (contrast_glow >= thresholds["contrast_glow_min"] * 0.55)
            )
        )
    )
    candidate_raw = red_anchor | hot_core_seed

    if cfg.suppress_border_glow:
        yy, xx = np.indices((height, width))
        border = (
            (xx < int(round(width * float(cfg.border_x_fraction))))
            | (xx > width - int(round(width * float(cfg.border_x_fraction))))
            | (yy < int(round(height * float(cfg.border_y_fraction))))
            | (yy > height - int(round(height * float(cfg.border_y_fraction))))
        )
        broad_red_glow = border & (local_excess > 20.0) & (contrast_excess < 18.0)
        candidate_raw = candidate_raw & ~broad_red_glow

    candidate_prefilter = _min_filter_bool(_max_filter_bool(candidate_raw, 3), 3)
    labels, components = connected_components(candidate_prefilter)

    preview_mask = np.zeros((height, width), dtype=bool)
    kept_components: list[dict[str, Any]] = []
    reject_counts = _empty_reject_counts()
    for component in components:
        component_mask = labels == component.label
        accepted, record, grown_mask = _review_component(
            component_mask=component_mask,
            bbox=component.bbox,
            area=component.area,
            red=red,
            red_excess=red_excess,
            red_ratio=red_ratio,
            local_red=local_red,
            local_excess=local_excess,
            glow_signal=glow_signal,
            local_glow=local_glow,
            value=value,
            contrast_value=contrast_value,
            config=cfg,
            thresholds=thresholds,
        )
        if not accepted:
            reject_reason = str(record["reject_reason"])
            reject_counts[reject_reason] = reject_counts.get(reject_reason, 0) + 1
            continue
        y0, x0, y1, x1 = record["review_patch"]
        preview_mask[y0:y1, x0:x1] |= grown_mask
        kept_components.append({key: value for key, value in record.items() if key != "review_patch"})

    preview_mask = _apply_edge_mode(preview_mask, cfg.mask_edge_mode)
    overlay = _overlay_mask(rgb, preview_mask)
    score_map = _to_score_image(np.maximum.reduce([contrast_excess, contrast_glow, contrast_value * 0.55]))
    mask = preview_mask.astype(np.uint8) * 255
    manifest = {
        "detector_version": "red_highlight_v1",
        "component_count": len(kept_components),
        "candidate_pixels_raw": int(np.count_nonzero(candidate_raw)),
        "candidate_pixels_after_prefilter": int(np.count_nonzero(candidate_prefilter)),
        "preview_mask_pixels": int(np.count_nonzero(preview_mask)),
        "reject_counts": reject_counts,
        "parameters": asdict(cfg),
        "resolved_thresholds": thresholds,
        "local_radius": local_radius,
        "components": kept_components,
    }
    return RedHighlightResult(
        mask=mask,
        preview_mask=mask,
        overlay_preview=overlay,
        score_map=score_map,
        components=kept_components,
        manifest=manifest,
    )


def _review_component(
    component_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    area: int,
    red: np.ndarray,
    red_excess: np.ndarray,
    red_ratio: np.ndarray,
    local_red: np.ndarray,
    local_excess: np.ndarray,
    glow_signal: np.ndarray,
    local_glow: np.ndarray,
    value: np.ndarray,
    contrast_value: np.ndarray,
    config: RedHighlightConfig,
    thresholds: dict[str, float],
) -> tuple[bool, dict[str, Any], np.ndarray]:
    height, width = red.shape
    x0, y0, x1, y1 = bbox
    bbox_height = max(1, y1 - y0)
    bbox_width = max(1, x1 - x0)
    max_dim = max(bbox_height, bbox_width)
    aspect = max_dim / float(max(1, min(bbox_height, bbox_width)))
    record: dict[str, Any] = {
        "bbox": [y0, x0, y1, x1],
        "area_seed": int(area),
        "aspect_ratio": round(aspect, 4),
        "reject_reason": "",
    }
    empty = np.zeros((1, 1), dtype=bool)
    elongated = _is_elongated_scratch_candidate(
        area=area,
        bbox_height=bbox_height,
        bbox_width=bbox_width,
        max_dim=max_dim,
        aspect=aspect,
        config=config,
    )
    record["elongated_scratch"] = elongated
    if area < int(config.min_area):
        record["reject_reason"] = "area_min"
        return False, record, empty
    if not elongated and (area > int(config.max_area) or max_dim > int(config.max_dim)):
        record["reject_reason"] = "area_or_dim_max"
        return False, record, empty
    if not elongated and aspect > float(config.max_aspect) and area > 10:
        record["reject_reason"] = "aspect"
        return False, record, empty
    if y0 <= 0 or x0 <= 0 or y1 >= height or x1 >= width:
        record["reject_reason"] = "border_touch"
        return False, record, empty

    pad = 8
    py0 = max(0, y0 - pad)
    py1 = min(height, y1 + pad)
    px0 = max(0, x0 - pad)
    px1 = min(width, x1 + pad)
    patch_mask = component_mask[py0:py1, px0:px1]
    inner = _max_filter_bool(patch_mask, 3)
    ring = _max_filter_bool(patch_mask, 9) & ~inner
    record["review_patch"] = [py0, px0, py1, px1]
    if not np.any(ring):
        record["reject_reason"] = "ring_empty"
        return False, record, empty

    patch_red = red[py0:py1, px0:px1]
    patch_excess = red_excess[py0:py1, px0:px1]
    patch_ratio = red_ratio[py0:py1, px0:px1]
    patch_glow = glow_signal[py0:py1, px0:px1]
    patch_value = value[py0:py1, px0:px1]
    patch_contrast_value = contrast_value[py0:py1, px0:px1]
    inner_excess = float(np.mean(patch_excess[patch_mask]))
    ring_excess = float(np.mean(patch_excess[ring]))
    inner_red = float(np.mean(patch_red[patch_mask]))
    ring_red = float(np.mean(patch_red[ring]))
    inner_glow = float(np.mean(patch_glow[patch_mask]))
    ring_glow = float(np.mean(patch_glow[ring]))
    peak_red = float(np.max(patch_red[patch_mask]))
    peak_value = float(np.max(patch_value[patch_mask]))
    peak_value_contrast = float(np.max(patch_contrast_value[patch_mask]))
    mean_ratio = float(np.mean(patch_ratio[patch_mask]))
    ring_excess_contrast = inner_excess - ring_excess
    ring_red_contrast = inner_red - ring_red
    ring_glow_contrast = inner_glow - ring_glow
    hot_core_like = peak_value >= thresholds["hot_core_value_min"] and peak_value_contrast >= thresholds["hot_core_contrast_min"]
    record.update(
        {
            "peak_red": round(peak_red, 4),
            "peak_value": round(peak_value, 4),
            "mean_red_ratio": round(mean_ratio, 6),
            "ring_excess_contrast": round(ring_excess_contrast, 4),
            "ring_red_contrast": round(ring_red_contrast, 4),
            "ring_glow_contrast": round(ring_glow_contrast, 4),
        }
    )
    if peak_red < thresholds["peak_red_min"] and not hot_core_like:
        record["reject_reason"] = "weak_peak"
        return False, record, empty
    if mean_ratio < thresholds["mean_red_ratio_min"] and not hot_core_like:
        record["reject_reason"] = "weak_red_ratio"
        return False, record, empty
    if (
        ring_excess_contrast < thresholds["ring_excess_contrast_min"]
        and ring_red_contrast < thresholds["ring_red_contrast_min"]
        and ring_glow_contrast < thresholds["ring_glow_contrast_min"]
        and peak_red < thresholds["strong_red_min"]
        and not hot_core_like
    ):
        record["reject_reason"] = "weak_ring_contrast"
        return False, record, empty

    grow = _max_filter_bool(patch_mask, 5)
    red_growth = (
        grow
        & (patch_red >= thresholds["grow_red_min"])
        & ((patch_excess >= thresholds["grow_excess_min"]) | (patch_glow >= thresholds["grow_glow_min"]))
        & (patch_ratio >= thresholds["grow_ratio_min"])
        & (
            (patch_red - local_red[py0:py1, px0:px1] >= thresholds["grow_contrast_min"])
            | (patch_excess - local_excess[py0:py1, px0:px1] >= thresholds["grow_contrast_min"])
            | (patch_glow - local_glow[py0:py1, px0:px1] >= thresholds["grow_contrast_min"])
            | (patch_red >= thresholds["strong_red_min"])
        )
    )
    seed_or_growth = red_growth | patch_mask
    near_red_growth = _max_filter_bool(seed_or_growth, 5)
    hot_growth = (
        grow
        & near_red_growth
        & (patch_value >= thresholds["hot_core_value_min"] * 0.82)
        & (patch_red >= thresholds["hot_core_red_min"] * 0.65)
        & (
            (patch_contrast_value >= thresholds["hot_core_contrast_min"] * 0.42)
            | (patch_red >= thresholds["strong_red_min"])
        )
    )
    grown = red_growth | hot_growth
    grown = _refine_patch_mask_by_edge_mode(grown, patch_mask, config.mask_edge_mode)
    area_grown = int(np.count_nonzero(grown))
    if area_grown == 0:
        record["reject_reason"] = "empty_after_growth"
        return False, record, empty
    record["area_grown"] = area_grown
    return True, record, grown


def _is_elongated_scratch_candidate(
    area: int,
    bbox_height: int,
    bbox_width: int,
    max_dim: int,
    aspect: float,
    config: RedHighlightConfig,
) -> bool:
    if not bool(config.include_long_scratches):
        return False
    min_dim = min(bbox_height, bbox_width)
    return (
        area <= int(config.max_scratch_area)
        and max_dim <= int(config.max_scratch_dim)
        and min_dim <= int(config.max_scratch_width)
        and aspect >= float(config.min_scratch_aspect)
    )


def _final_mask_for_source(
    source_rgb: np.ndarray,
    preview_mask: np.ndarray,
    config: RedHighlightConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_shape = tuple(int(value) for value in source_rgb.shape[:2])
    if not bool(config.full_resolution_refine):
        mask = _resize_mask_nearest(preview_mask, source_shape)
        return mask, {"mode": "resized_preview", "reason": "full_resolution_refine_disabled"}
    seed = _resize_mask_nearest(preview_mask, source_shape) > 0
    refined_bool, summary = refine_red_highlight_mask_to_image(source_rgb, seed, config=config)
    return refined_bool.astype(np.uint8) * 255, summary


def refine_red_highlight_mask_to_image(
    image: np.ndarray,
    seed_mask: np.ndarray,
    config: RedHighlightConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = config or RedHighlightConfig()
    edge_mode = _normalize_mask_edge_mode(cfg.mask_edge_mode)
    rgb = _ensure_rgb_u8(image)
    height, width = rgb.shape[:2]
    seed = np.asarray(seed_mask).astype(bool)
    if seed.shape != (height, width):
        seed = _resize_mask_nearest(seed.astype(np.uint8) * 255, (height, width)) > 0
    if edge_mode == "legacy":
        return seed, {"mode": "legacy", "component_count": len(connected_components(seed)[1])}
    if not np.any(seed):
        return np.zeros((height, width), dtype=bool), {"mode": edge_mode, "component_count": 0, "refined_component_count": 0}

    work = rgb.astype(np.float32)
    red = work[:, :, 0]
    green = work[:, :, 1]
    blue = work[:, :, 2]
    max_green_blue = np.maximum(green, blue)
    value = np.maximum.reduce([red, green, blue])
    red_excess = np.maximum(red - max_green_blue, 0.0)
    red_ratio = red_excess / np.maximum(red, 1.0)
    glow_signal = np.maximum.reduce(
        [
            red_excess,
            red - (green * 0.62 + blue * 0.28),
            red * 0.70 - np.maximum(green, blue) * 0.34,
        ]
    )
    glow_signal = np.maximum(glow_signal, 0.0).astype(np.float32)

    _labels, components = connected_components(seed)
    refined = np.zeros((height, width), dtype=bool)
    refined_count = 0
    for component in components:
        x0, y0, x1, y1 = component.bbox
        max_dim = max(y1 - y0, x1 - x0)
        pad = max(8, min(72, int(round(max_dim * 0.75)) + 6))
        py0 = max(0, y0 - pad)
        py1 = min(height, y1 + pad)
        px0 = max(0, x0 - pad)
        px1 = min(width, x1 + pad)
        patch_seed = seed[py0:py1, px0:px1]
        if not np.any(patch_seed):
            continue
        inner = _max_filter_bool(patch_seed, 3)
        ring_size = _odd_filter_size(max(9, min(65, int(round(max_dim * 0.85)) + 5)))
        ring = _max_filter_bool(patch_seed, ring_size) & ~inner
        if not np.any(ring):
            ring = ~inner
        candidate = _max_filter_bool(patch_seed, _mask_edge_params(edge_mode)["candidate_filter"])
        refined_patch = _refine_patch_mask_by_local_ring(
            patch_red=red[py0:py1, px0:px1],
            patch_excess=red_excess[py0:py1, px0:px1],
            patch_ratio=red_ratio[py0:py1, px0:px1],
            patch_glow=glow_signal[py0:py1, px0:px1],
            patch_value=value[py0:py1, px0:px1],
            seed_mask=patch_seed,
            candidate_mask=candidate,
            ring=ring,
            mode=edge_mode,
        )
        if np.any(refined_patch):
            refined[py0:py1, px0:px1] |= refined_patch
            refined_count += 1
    return refined, {
        "mode": "source_rgb_roi_refine",
        "edge_mode": edge_mode,
        "component_count": len(components),
        "refined_component_count": refined_count,
        "mask_pixels_before_refine": int(np.count_nonzero(seed)),
        "mask_pixels_after_refine": int(np.count_nonzero(refined)),
    }


def _refine_patch_mask_by_local_ring(
    patch_red: np.ndarray,
    patch_excess: np.ndarray,
    patch_ratio: np.ndarray,
    patch_glow: np.ndarray,
    patch_value: np.ndarray,
    seed_mask: np.ndarray,
    candidate_mask: np.ndarray,
    ring: np.ndarray,
    mode: str,
) -> np.ndarray:
    params = _mask_edge_params(mode)
    if not np.any(ring):
        return seed_mask.copy()
    ring_red = float(np.median(patch_red[ring]))
    ring_excess = float(np.median(patch_excess[ring]))
    ring_glow = float(np.median(patch_glow[ring]))
    ring_value = float(np.median(patch_value[ring]))
    red_cut = ring_red + params["red_delta"]
    excess_cut = ring_excess + params["excess_delta"]
    glow_cut = ring_glow + params["glow_delta"]
    value_cut = ring_value + params["value_delta"]
    refined = (
        candidate_mask
        & (
            ((patch_red >= red_cut) & (patch_ratio >= params["ratio_min"]))
            | (patch_excess >= excess_cut)
            | (patch_glow >= glow_cut)
            | ((patch_value >= value_cut) & _max_filter_bool(seed_mask, 5))
        )
    )
    if mode == "tight":
        refined &= _max_filter_bool(seed_mask, 5)
    elif mode == "wide":
        refined = _max_filter_bool(refined | seed_mask, 3)
    else:
        refined = _min_filter_bool(_max_filter_bool(refined | seed_mask, 3), 3)
    return refined


def _resolve_thresholds(
    config: RedHighlightConfig,
    red: np.ndarray,
    red_excess: np.ndarray,
    red_ratio: np.ndarray,
    glow_signal: np.ndarray,
    contrast_red: np.ndarray,
    contrast_excess: np.ndarray,
    contrast_glow: np.ndarray,
    contrast_value: np.ndarray,
) -> dict[str, float]:
    sensitivity = max(0.35, float(config.threshold_sensitivity))
    red60 = float(np.quantile(red, 0.60))
    positive_ratio = red_ratio[red > max(16.0, red60)]
    ratio_reference = float(np.quantile(positive_ratio, 0.90)) if positive_ratio.size else 0.0
    red_min = _configured_or_auto(config.red_min, max(32.0, float(np.quantile(red, 0.92)) * 0.38))
    red_excess_min = _configured_or_auto(
        config.red_excess_min,
        _adaptive_floor(red_excess, 14.0, 0.992, 0.18, sensitivity),
    )
    red_ratio_min = _configured_or_auto(
        config.red_ratio_min,
        min(0.46, max(0.20, ratio_reference * 0.48)),
    )
    contrast_red_min = _configured_or_auto(
        config.contrast_red_min,
        _adaptive_floor(contrast_red, 5.0, 0.994, 0.14, sensitivity),
    )
    contrast_excess_min = _configured_or_auto(
        config.contrast_excess_min,
        _adaptive_floor(contrast_excess, 4.0, 0.994, 0.14, sensitivity),
    )
    glow_signal_min = _configured_or_auto(
        config.glow_signal_min,
        _adaptive_floor(glow_signal, 12.0, 0.992, 0.18, sensitivity),
    )
    contrast_glow_min = _configured_or_auto(
        config.contrast_glow_min,
        _adaptive_floor(contrast_glow, 4.0, 0.994, 0.14, sensitivity),
    )
    hot_core_value_min = _configured_or_auto(
        config.hot_core_value_min,
        max(80.0, float(np.quantile(red, 0.985)) * 0.72, float(np.quantile(contrast_value, 0.995)) * 2.2),
    )
    hot_core_contrast_min = _configured_or_auto(
        config.hot_core_contrast_min,
        _adaptive_floor(contrast_value, 7.0, 0.996, 0.20, sensitivity),
    )
    strong_red_min = red_min * max(1.6, float(np.quantile(red, 0.985)) * 0.72 / max(red_min, 1.0))
    strong_excess_min = max(red_excess_min * 1.7, _adaptive_floor(red_excess, 18.0, 0.997, 0.28, sensitivity))
    strong_glow_min = max(glow_signal_min * 1.65, _adaptive_floor(glow_signal, 18.0, 0.997, 0.28, sensitivity))
    strong_ratio_min = max(0.18, red_ratio_min * 0.86)
    strong_contrast_min = max(contrast_red_min, contrast_excess_min, contrast_glow_min, 5.0)
    return {
        "red_min": round(red_min, 6),
        "red_excess_min": round(red_excess_min, 6),
        "red_ratio_min": round(red_ratio_min, 6),
        "contrast_red_min": round(contrast_red_min, 6),
        "contrast_excess_min": round(contrast_excess_min, 6),
        "glow_signal_min": round(glow_signal_min, 6),
        "contrast_glow_min": round(contrast_glow_min, 6),
        "hot_core_value_min": round(hot_core_value_min, 6),
        "hot_core_red_min": round(max(28.0, red_min * 0.55), 6),
        "hot_core_contrast_min": round(hot_core_contrast_min, 6),
        "strong_red_min": round(strong_red_min, 6),
        "strong_excess_min": round(strong_excess_min, 6),
        "strong_glow_min": round(strong_glow_min, 6),
        "strong_ratio_min": round(strong_ratio_min, 6),
        "strong_contrast_min": round(strong_contrast_min, 6),
        "peak_red_min": round(max(red_min, 50.0), 6),
        "mean_red_ratio_min": round(max(0.18, red_ratio_min * 0.82), 6),
        "ring_excess_contrast_min": round(max(3.5, contrast_excess_min * 0.70), 6),
        "ring_red_contrast_min": round(max(5.0, contrast_red_min * 0.85), 6),
        "ring_glow_contrast_min": round(max(3.5, contrast_glow_min * 0.70), 6),
        "grow_red_min": round(max(28.0, red_min * 0.62), 6),
        "grow_excess_min": round(max(8.0, red_excess_min * 0.46), 6),
        "grow_glow_min": round(max(8.0, glow_signal_min * 0.46), 6),
        "grow_ratio_min": round(max(0.12, red_ratio_min * 0.62), 6),
        "grow_contrast_min": round(max(2.5, min(contrast_red_min, contrast_excess_min, contrast_glow_min) * 0.38), 6),
    }


def _configured_or_auto(configured: float | None, auto: float) -> float:
    return float(auto) if configured is None else float(configured)


def _adaptive_floor(
    values: np.ndarray,
    absolute_floor: float,
    quantile: float,
    multiplier: float,
    sensitivity: float,
) -> float:
    work = np.maximum(values.astype(np.float32, copy=False), 0.0)
    positive = work[work > 0.0]
    if positive.size == 0:
        return float(absolute_floor)
    q_value = float(np.quantile(positive, quantile))
    median = float(np.median(positive))
    mad = float(np.median(np.abs(positive - median))) + 1e-6
    robust = median + mad * 1.4826 * 3.0 * sensitivity
    tail = q_value * multiplier * sensitivity
    return float(max(absolute_floor, min(max(robust, tail), q_value * 0.82)))


def _apply_edge_mode(mask: np.ndarray, mode: str) -> np.ndarray:
    edge_mode = _normalize_mask_edge_mode(mode)
    if edge_mode == "legacy":
        return _max_filter_bool(mask, 3)
    if edge_mode == "wide":
        return _max_filter_bool(mask, 3)
    if edge_mode == "tight":
        return mask.astype(bool)
    return _min_filter_bool(_max_filter_bool(mask, 3), 3)


def _refine_patch_mask_by_edge_mode(mask: np.ndarray, seed: np.ndarray, mode: str) -> np.ndarray:
    edge_mode = _normalize_mask_edge_mode(mode)
    if edge_mode == "legacy":
        return _max_filter_bool(mask | seed, 3)
    if edge_mode == "wide":
        return _max_filter_bool(mask | seed, 5)
    if edge_mode == "tight":
        return mask & _max_filter_bool(seed, 5)
    return _min_filter_bool(_max_filter_bool(mask | seed, 3), 3)


def _mask_edge_params(mode: str) -> dict[str, float | int]:
    edge_mode = _normalize_mask_edge_mode(mode)
    if edge_mode == "tight":
        return {"candidate_filter": 3, "red_delta": 8.0, "excess_delta": 5.0, "glow_delta": 5.0, "value_delta": 10.0, "ratio_min": 0.18}
    if edge_mode == "wide":
        return {"candidate_filter": 7, "red_delta": 2.5, "excess_delta": 1.5, "glow_delta": 1.5, "value_delta": 3.5, "ratio_min": 0.10}
    return {"candidate_filter": 5, "red_delta": 4.0, "excess_delta": 2.5, "glow_delta": 2.5, "value_delta": 5.0, "ratio_min": 0.13}


def _normalize_mask_edge_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"tight", "normal", "wide", "legacy"}:
        return normalized
    raise ValueError(f"Unsupported mask_edge_mode: {mode}")


def _ensure_rgb_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Unsupported image shape for red-highlight detection: {arr.shape}")
    arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(np.rint(as_float32(arr) * 255.0).astype(np.uint8))


def _resize_long_edge_rgb(image: np.ndarray, long_edge: int) -> np.ndarray:
    rgb = _ensure_rgb_u8(image)
    height, width = rgb.shape[:2]
    target = max(1, int(long_edge))
    scale = target / float(max(height, width))
    if scale >= 1.0:
        return rgb
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.asarray(Image.fromarray(rgb).resize(new_size, Image.Resampling.BILINEAR), dtype=np.uint8)


def _resize_mask_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_shape
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize(
            (int(target_width), int(target_height)),
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )


def _max_filter_bool(mask: np.ndarray, size: int) -> np.ndarray:
    size = _odd_filter_size(size)
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(size)),
        dtype=np.uint8,
    ) > 0


def _min_filter_bool(mask: np.ndarray, size: int) -> np.ndarray:
    size = _odd_filter_size(size)
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(size)),
        dtype=np.uint8,
    ) > 0


def _odd_filter_size(size: int) -> int:
    value = max(3, int(size))
    return value if value % 2 == 1 else value + 1


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float32)
    work = values.astype(np.float32, copy=False)
    padded = np.pad(work, radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    size = radius * 2 + 1
    area = float(size * size)
    y2 = np.arange(size, size + work.shape[0])
    x2 = np.arange(size, size + work.shape[1])
    y1 = y2 - size
    x1 = x2 - size
    result = (
        integral[np.ix_(y2, x2)]
        - integral[np.ix_(y1, x2)]
        - integral[np.ix_(y2, x1)]
        + integral[np.ix_(y1, x1)]
    ) / area
    return result.astype(np.float32)


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = _ensure_rgb_u8(rgb).copy()
    selected = mask.astype(bool)
    if np.any(selected):
        overlay[selected] = np.clip(
            overlay[selected].astype(np.float32) * 0.35
            + np.asarray([0.0, 255.0, 40.0], dtype=np.float32) * 0.65,
            0,
            255,
        ).astype(np.uint8)
    return overlay


def _to_score_image(score: np.ndarray) -> np.ndarray:
    work = np.maximum(score.astype(np.float32, copy=False), 0.0)
    scale = float(np.quantile(work, 0.997)) if work.size else 0.0
    if scale <= 1e-6:
        return np.zeros(work.shape[:2], dtype=np.uint8)
    return np.clip(work / scale * 255.0, 0, 255).astype(np.uint8)


def _empty_reject_counts() -> dict[str, int]:
    return {
        "area_min": 0,
        "area_or_dim_max": 0,
        "aspect": 0,
        "border_touch": 0,
        "ring_empty": 0,
        "weak_peak": 0,
        "weak_red_ratio": 0,
        "weak_ring_contrast": 0,
        "empty_after_growth": 0,
    }


def _safe_stem(path: Path) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in path.stem)
    return safe[:80] or "image"


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)
