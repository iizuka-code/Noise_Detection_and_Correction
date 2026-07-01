from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copyfile
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .io import RAW_EXTENSIONS, as_float32, read_image, write_image
from .mask import connected_components
from .xmp import MASK_OUTPUT_MODE_LEGACY_PLUS_XMP, write_mask_xmp


SUPPORTED_WHITE_DUST_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", *RAW_EXTENSIONS}


@dataclass(frozen=True)
class WhiteDustConfig:
    detection_long_edge: int = 1024
    local_radius: int = 0
    background_mode: str = "dark"
    mask_edge_mode: str = "normal"
    threshold_sensitivity: float = 1.0
    value_min: float | None = None
    luma_min: float | None = None
    white_floor_min: float | None = None
    value_contrast_min: float | None = None
    bright_contrast_min: float | None = None
    white_contrast_min: float | None = None
    whiteness_min: float = 0.60
    min_area: int = 2
    max_area: int = 12000
    max_dim: int = 900
    max_thickness: float = 54.0
    require_brown_background: bool = True
    brown_blue_deficit_min: float = 8.0
    brown_red_blue_ratio_min: float = 1.10
    brown_luma_max: float | None = 170.0
    dark_luma_max: float = 72.0
    dark_value_max: float = 96.0
    focus_margin_x: float = 0.0
    focus_margin_y: float = 0.0
    visual_artifacts: bool = True

    def validate(self) -> None:
        if self.detection_long_edge <= 0:
            raise ValueError("detection_long_edge must be > 0")
        if self.local_radius < 0:
            raise ValueError("local_radius must be >= 0")
        if _normalize_background_mode(self.background_mode) not in {"dark", "brown", "any"}:
            raise ValueError(f"Unsupported background_mode: {self.background_mode}")
        if _normalize_mask_edge_mode(self.mask_edge_mode) not in {"tight", "normal", "wide"}:
            raise ValueError(f"Unsupported mask_edge_mode: {self.mask_edge_mode}")
        if self.threshold_sensitivity <= 0.0:
            raise ValueError("threshold_sensitivity must be > 0")
        if self.whiteness_min < 0.0:
            raise ValueError("whiteness_min must be >= 0")
        if self.min_area < 0:
            raise ValueError("min_area must be >= 0")
        if self.max_area < self.min_area:
            raise ValueError("max_area must be >= min_area")
        if self.max_dim <= 0:
            raise ValueError("max_dim must be > 0")
        if self.max_thickness <= 0.0:
            raise ValueError("max_thickness must be > 0")
        if self.brown_blue_deficit_min < 0.0:
            raise ValueError("brown_blue_deficit_min must be >= 0")
        if self.brown_red_blue_ratio_min <= 0.0:
            raise ValueError("brown_red_blue_ratio_min must be > 0")
        if self.brown_luma_max is not None and self.brown_luma_max <= 0.0:
            raise ValueError("brown_luma_max must be > 0 when set")
        if self.dark_luma_max <= 0.0:
            raise ValueError("dark_luma_max must be > 0")
        if self.dark_value_max <= 0.0:
            raise ValueError("dark_value_max must be > 0")
        if not 0.0 <= self.focus_margin_x < 0.49:
            raise ValueError("focus_margin_x must be >= 0 and < 0.49")
        if not 0.0 <= self.focus_margin_y < 0.49:
            raise ValueError("focus_margin_y must be >= 0 and < 0.49")


@dataclass(frozen=True)
class WhiteDustResult:
    mask: np.ndarray
    preview_mask: np.ndarray
    overlay_preview: np.ndarray
    score_map: np.ndarray
    components: list[dict[str, Any]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class WhiteDustSourceResult:
    mask: np.ndarray
    preview_mask: np.ndarray
    overlay: np.ndarray
    overlay_preview: np.ndarray
    score_map: np.ndarray
    components: list[dict[str, Any]]
    manifest: dict[str, Any]


def detect_white_dust_source_image(
    image: np.ndarray,
    config: WhiteDustConfig | None = None,
    *,
    source_overlay: bool = True,
) -> WhiteDustSourceResult:
    cfg = config or WhiteDustConfig()
    cfg.validate()
    started = perf_counter()

    source_rgb = _ensure_rgb_u8(image)
    detection_rgb = _resize_long_edge_rgb(source_rgb, cfg.detection_long_edge)
    detection = detect_white_dust_mask(detection_rgb, cfg)
    source_shape = tuple(int(value) for value in source_rgb.shape[:2])
    final_mask = _resize_mask_nearest(detection.preview_mask, source_shape)
    overlay = _empty_rgb_artifact()
    if cfg.visual_artifacts and source_overlay:
        overlay = _overlay_mask(source_rgb, final_mask > 0)

    manifest = dict(detection.manifest)
    manifest.update(
        {
            "detector_version": "dust_on_dark_or_brown_v2",
            "source_shape": [int(source_rgb.shape[0]), int(source_rgb.shape[1])],
            "detection_shape": [int(detection_rgb.shape[0]), int(detection_rgb.shape[1])],
            "final_mask_pixels": int(np.count_nonzero(final_mask)),
            "timings_ms": {
                "total_ms": _elapsed_ms(started),
            },
        }
    )
    return WhiteDustSourceResult(
        mask=final_mask,
        preview_mask=detection.preview_mask,
        overlay=overlay,
        overlay_preview=detection.overlay_preview,
        score_map=detection.score_map,
        components=detection.components,
        manifest=manifest,
    )


def detect_white_dust_proxy_image(
    image: np.ndarray,
    config: WhiteDustConfig | None = None,
) -> WhiteDustSourceResult:
    cfg = config or WhiteDustConfig()
    cfg.validate()
    started = perf_counter()

    source_rgb = _ensure_rgb_u8(image)
    detection_rgb = _resize_long_edge_rgb(source_rgb, cfg.detection_long_edge)
    detection = detect_white_dust_mask(detection_rgb, cfg)

    manifest = dict(detection.manifest)
    manifest.update(
        {
            "detector_version": "dust_on_dark_or_brown_v2",
            "source_shape": [int(source_rgb.shape[0]), int(source_rgb.shape[1])],
            "detection_shape": [int(detection_rgb.shape[0]), int(detection_rgb.shape[1])],
            "mask_shape": [int(detection.mask.shape[0]), int(detection.mask.shape[1])],
            "final_mask_pixels": int(np.count_nonzero(detection.mask)),
            "proxy_mask": True,
            "timings_ms": {
                "total_ms": _elapsed_ms(started),
            },
        }
    )
    return WhiteDustSourceResult(
        mask=detection.mask,
        preview_mask=detection.preview_mask,
        overlay=detection.overlay_preview,
        overlay_preview=detection.overlay_preview,
        score_map=detection.score_map,
        components=detection.components,
        manifest=manifest,
    )


def run_white_dust_detector(
    source: str | Path,
    output_dir: str | Path,
    config: WhiteDustConfig | None = None,
) -> dict[str, Any]:
    cfg = config or WhiteDustConfig()
    cfg.validate()
    source_path = Path(source)
    if source_path.suffix.lower() not in SUPPORTED_WHITE_DUST_EXTENSIONS:
        raise ValueError(f"Unsupported white-dust source extension: {source_path.suffix.lower()}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_source = source_path.suffix.lower() in RAW_EXTENSIONS
    load_started = perf_counter()
    image = read_image(source_path, raw_half_size=raw_source, raw_output_bps=8 if raw_source else 16)
    source_rgb = _ensure_rgb_u8(image.pixels)
    load_ms = _elapsed_ms(load_started)

    if raw_source:
        result = detect_white_dust_proxy_image(source_rgb, cfg)
    else:
        result = detect_white_dust_source_image(source_rgb, cfg)

    output_started = perf_counter()
    mask_path = output_path / "mask.png"
    named_mask_path = output_path / f"{_safe_stem(source_path)}_white_dust_mask.png"
    xmp_path = output_path / "mask.xmp"
    named_xmp_path = output_path / f"{_safe_stem(source_path)}_white_dust_mask.xmp"
    overlay_path = output_path / "overlay.png"
    overlay_preview_path = output_path / "overlay_preview.png"
    write_image(output_path / "input_preview.png", _resize_long_edge_rgb(source_rgb, cfg.detection_long_edge))
    write_image(mask_path, result.mask)
    copyfile(mask_path, named_mask_path)
    write_image(output_path / "preview_mask.png", result.preview_mask)
    write_image(overlay_preview_path, result.overlay_preview)
    if raw_source:
        copyfile(overlay_preview_path, overlay_path)
    else:
        write_image(overlay_path, result.overlay)
    write_image(output_path / "white_dust_score.png", result.score_map)
    component_json = output_path / "component_features.json"
    component_json.write_text(
        json.dumps({"components": result.components}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest = dict(result.manifest)
    manifest["source"] = str(source_path)
    manifest["source_mode"] = image.color_mode
    manifest["source_metadata"] = image.metadata
    manifest["raw_fast_proxy"] = raw_source
    manifest["output_dir"] = str(output_path)
    manifest["mask_output_mode"] = MASK_OUTPUT_MODE_LEGACY_PLUS_XMP
    manifest["artifacts"] = {
        "mask": str(mask_path),
        "named_mask": str(named_mask_path),
        "input_preview": str(output_path / "input_preview.png"),
        "preview_mask": str(output_path / "preview_mask.png"),
        "overlay_preview": str(overlay_preview_path),
        "overlay": str(overlay_path),
        "score": str(output_path / "white_dust_score.png"),
        "component_features_json": str(component_json),
        "xmp": str(xmp_path),
        "named_xmp": str(named_xmp_path),
    }
    xmp_summary = write_mask_xmp(
        xmp_path,
        mask=result.mask,
        manifest=manifest,
        source_path=source_path,
        role="white_dust_detection",
    )
    copyfile(xmp_path, named_xmp_path)
    manifest["xmp"] = {**xmp_summary, "named_path": str(named_xmp_path)}
    output_ms = _elapsed_ms(output_started)
    timings = dict(manifest.get("timings_ms", {}))
    timings.update({"load_decode": load_ms, "output_save": output_ms})
    manifest["timings_ms"] = timings
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def make_white_dust_input_preview(image: np.ndarray, long_edge: int) -> np.ndarray:
    return _resize_long_edge_rgb(_ensure_rgb_u8(image), long_edge)


def detect_white_dust_mask(
    image: np.ndarray,
    config: WhiteDustConfig | None = None,
) -> WhiteDustResult:
    cfg = config or WhiteDustConfig()
    cfg.validate()
    rgb = _ensure_rgb_u8(image)
    height, width = rgb.shape[:2]

    work = rgb.astype(np.float32)
    red = work[:, :, 0]
    green = work[:, :, 1]
    blue = work[:, :, 2]
    value = np.maximum.reduce([red, green, blue])
    channel_min = np.minimum.reduce([red, green, blue])
    channel_spread = value - channel_min
    luma = (red * 0.2126 + green * 0.7152 + blue * 0.0722).astype(np.float32)
    whiteness = channel_min / np.maximum(value, 1.0)

    local_radius = int(cfg.local_radius) if cfg.local_radius > 0 else max(5, int(round(max(height, width) / 230.0)))
    local_red = _box_blur(red, radius=local_radius)
    local_green = _box_blur(green, radius=local_radius)
    local_blue = _box_blur(blue, radius=local_radius)
    local_luma = _box_blur(luma, radius=local_radius)
    local_value = _box_blur(value, radius=local_radius)
    local_floor = _box_blur(channel_min, radius=local_radius)
    value_contrast = np.maximum(value - local_value, 0.0)
    bright_contrast = np.maximum(luma - local_luma, 0.0)
    white_contrast = np.maximum(channel_min - local_floor, 0.0)
    score = np.maximum.reduce([value_contrast, bright_contrast, white_contrast])

    thresholds = _resolve_thresholds(
        config=cfg,
        value=value,
        luma=luma,
        channel_min=channel_min,
        value_contrast=value_contrast,
        bright_contrast=bright_contrast,
        white_contrast=white_contrast,
    )
    background_mode = _normalize_background_mode(cfg.background_mode)
    warm_background = _warm_brown_background(local_red, local_green, local_blue, local_luma, cfg)
    dark_background = _dark_background(local_luma, local_value, cfg)
    if background_mode == "brown":
        background_mask = warm_background
    elif background_mode == "dark":
        background_mask = dark_background
    else:
        background_mask = np.ones((height, width), dtype=bool)
    white_like = (
        (whiteness >= thresholds["whiteness_min"])
        | ((channel_spread <= thresholds["max_channel_spread"]) & (channel_min >= thresholds["white_floor_min"]))
    )
    white_candidate = (
        (luma >= thresholds["luma_min"])
        & (channel_min >= thresholds["white_floor_min"])
        & white_like
        & (
            (bright_contrast >= thresholds["bright_contrast_min"])
            | (white_contrast >= thresholds["white_contrast_min"])
        )
    )
    colored_candidate = (
        (value >= thresholds["value_min"])
        & (value_contrast >= thresholds["value_contrast_min"])
        & (score >= thresholds["value_contrast_min"])
    )
    candidate_raw = white_candidate | (colored_candidate if background_mode in {"dark", "any"} else False)
    if cfg.require_brown_background and background_mode == "brown":
        candidate_raw &= background_mask
    focus_region = _center_focus_mask(height, width, cfg.focus_margin_x, cfg.focus_margin_y)
    candidate_raw &= focus_region

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
            value=value,
            luma=luma,
            channel_min=channel_min,
            score=score,
            background_mask=background_mask,
            background_mode=background_mode,
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
    preview_mask &= focus_region
    mask = preview_mask.astype(np.uint8) * 255
    overlay = _overlay_mask(rgb, preview_mask) if cfg.visual_artifacts else _empty_rgb_artifact()
    focused_score = np.where(focus_region, score, 0.0)
    score_map = _to_score_image(focused_score) if cfg.visual_artifacts else _empty_gray_artifact()
    manifest = {
        "detector_version": "dust_on_dark_or_brown_v2",
        "component_count": len(kept_components),
        "candidate_pixels_raw": int(np.count_nonzero(candidate_raw)),
        "candidate_pixels_after_prefilter": int(np.count_nonzero(candidate_prefilter)),
        "preview_mask_pixels": int(np.count_nonzero(preview_mask)),
        "reject_counts": reject_counts,
        "parameters": asdict(cfg),
        "resolved_thresholds": thresholds,
        "local_radius": local_radius,
        "focus_bounds": _center_focus_bounds(height, width, cfg.focus_margin_x, cfg.focus_margin_y),
        "components": kept_components,
    }
    return WhiteDustResult(
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
    value: np.ndarray,
    luma: np.ndarray,
    channel_min: np.ndarray,
    score: np.ndarray,
    background_mask: np.ndarray,
    background_mode: str,
    config: WhiteDustConfig,
    thresholds: dict[str, float],
) -> tuple[bool, dict[str, Any], np.ndarray]:
    height, width = luma.shape
    x0, y0, x1, y1 = bbox
    bbox_height = max(1, y1 - y0)
    bbox_width = max(1, x1 - x0)
    max_dim = max(bbox_height, bbox_width)
    thickness = float(area) / float(max_dim)
    fill_ratio = float(area) / float(max(1, bbox_height * bbox_width))
    record: dict[str, Any] = {
        "bbox": [y0, x0, y1, x1],
        "area_seed": int(area),
        "thickness_estimate": round(thickness, 4),
        "fill_ratio": round(fill_ratio, 4),
        "reject_reason": "",
    }
    empty = np.zeros((1, 1), dtype=bool)
    if area < int(config.min_area):
        record["reject_reason"] = "area_min"
        return False, record, empty
    if area > int(config.max_area) or max_dim > int(config.max_dim):
        record["reject_reason"] = "area_or_dim_max"
        return False, record, empty
    if thickness > float(config.max_thickness):
        record["reject_reason"] = "thickness"
        return False, record, empty

    pad = max(8, min(64, int(round(max_dim * 0.35)) + 6))
    py0 = max(0, y0 - pad)
    py1 = min(height, y1 + pad)
    px0 = max(0, x0 - pad)
    px1 = min(width, x1 + pad)
    patch_mask = component_mask[py0:py1, px0:px1]
    inner = _max_filter_bool(patch_mask, 3)
    ring_size = max(9, min(51, int(round(max_dim * 0.5)) | 1))
    ring = _max_filter_bool(patch_mask, ring_size) & ~inner
    record["review_patch"] = [py0, px0, py1, px1]
    if not np.any(ring):
        record["reject_reason"] = "ring_empty"
        return False, record, empty

    patch_luma = luma[py0:py1, px0:px1]
    patch_value = value[py0:py1, px0:px1]
    patch_floor = channel_min[py0:py1, px0:px1]
    patch_score = score[py0:py1, px0:px1]
    patch_background = background_mask[py0:py1, px0:px1]
    inner_luma = float(np.mean(patch_luma[patch_mask]))
    ring_luma = float(np.median(patch_luma[ring]))
    inner_value = float(np.mean(patch_value[patch_mask]))
    ring_value = float(np.median(patch_value[ring]))
    inner_floor = float(np.mean(patch_floor[patch_mask]))
    ring_floor = float(np.median(patch_floor[ring]))
    luma_contrast = inner_luma - ring_luma
    value_contrast = inner_value - ring_value
    floor_contrast = inner_floor - ring_floor
    peak_score = float(np.max(patch_score[patch_mask]))
    background_ring_fraction = float(np.mean(patch_background[ring]))
    record.update(
        {
            "luma_contrast": round(luma_contrast, 4),
            "value_contrast": round(value_contrast, 4),
            "white_floor_contrast": round(floor_contrast, 4),
            "peak_score": round(peak_score, 4),
            "background_mode": background_mode,
            "background_ring_fraction": round(background_ring_fraction, 4),
        }
    )
    if config.require_brown_background and background_ring_fraction < 0.35:
        record["reject_reason"] = "weak_background"
        return False, record, empty
    if (
        value_contrast < thresholds["ring_value_contrast_min"]
        and luma_contrast < thresholds["ring_luma_contrast_min"]
        and floor_contrast < thresholds["ring_white_contrast_min"]
    ):
        record["reject_reason"] = "weak_ring_contrast"
        return False, record, empty

    grow = _max_filter_bool(patch_mask, _mask_edge_params(config.mask_edge_mode)["candidate_filter"])
    grown = (
        grow
        & (
            (patch_value >= ring_value + thresholds["grow_value_delta"])
            | (patch_luma >= ring_luma + thresholds["grow_luma_delta"])
            | (patch_floor >= ring_floor + thresholds["grow_white_delta"])
        )
        & (patch_score >= thresholds["grow_score_min"])
    )
    grown |= patch_mask
    grown = _refine_patch_mask_by_edge_mode(grown, patch_mask, config.mask_edge_mode)
    area_grown = int(np.count_nonzero(grown))
    if area_grown == 0:
        record["reject_reason"] = "empty_after_growth"
        return False, record, empty
    record["area_grown"] = area_grown
    return True, record, grown


def _resolve_thresholds(
    config: WhiteDustConfig,
    value: np.ndarray,
    luma: np.ndarray,
    channel_min: np.ndarray,
    value_contrast: np.ndarray,
    bright_contrast: np.ndarray,
    white_contrast: np.ndarray,
) -> dict[str, float]:
    sensitivity = max(0.35, float(config.threshold_sensitivity))
    value_min = _configured_or_auto(config.value_min, max(36.0, float(np.quantile(value, 0.88)) + 8.0))
    luma_min = _configured_or_auto(config.luma_min, max(82.0, float(np.quantile(luma, 0.70)) + 8.0))
    white_floor_min = _configured_or_auto(
        config.white_floor_min,
        max(70.0, float(np.quantile(channel_min, 0.70)) + 8.0),
    )
    value_min_contrast = _configured_or_auto(
        config.value_contrast_min,
        max(5.0, _adaptive_floor(value_contrast, 6.0, 0.995, 0.20, sensitivity) * 0.50),
    )
    bright_min = _configured_or_auto(
        config.bright_contrast_min,
        _adaptive_floor(bright_contrast, 7.0, 0.995, 0.20, sensitivity),
    )
    white_min = _configured_or_auto(
        config.white_contrast_min,
        _adaptive_floor(white_contrast, 7.0, 0.995, 0.20, sensitivity),
    )
    return {
        "value_min": round(value_min, 6),
        "luma_min": round(luma_min, 6),
        "white_floor_min": round(white_floor_min, 6),
        "value_contrast_min": round(value_min_contrast, 6),
        "bright_contrast_min": round(bright_min, 6),
        "white_contrast_min": round(white_min, 6),
        "whiteness_min": round(float(config.whiteness_min), 6),
        "max_channel_spread": round(max(34.0, white_floor_min * 0.30), 6),
        "ring_value_contrast_min": round(max(3.0, value_min_contrast * 0.40), 6),
        "ring_luma_contrast_min": round(max(4.0, bright_min * 0.46), 6),
        "ring_white_contrast_min": round(max(4.0, white_min * 0.46), 6),
        "grow_value_delta": round(max(2.0, value_min_contrast * 0.22), 6),
        "grow_luma_delta": round(max(2.5, bright_min * 0.25), 6),
        "grow_white_delta": round(max(2.5, white_min * 0.25), 6),
        "grow_score_min": round(max(2.5, min(value_min_contrast, bright_min, white_min) * 0.22), 6),
    }


def _warm_brown_background(
    local_red: np.ndarray,
    local_green: np.ndarray,
    local_blue: np.ndarray,
    local_luma: np.ndarray,
    config: WhiteDustConfig,
) -> np.ndarray:
    blue = np.maximum(local_blue, 1.0)
    warm = (
        ((local_red + local_green) * 0.5 - local_blue >= float(config.brown_blue_deficit_min))
        & (local_red / blue >= float(config.brown_red_blue_ratio_min))
        & (local_green >= local_blue * 0.82)
    )
    if config.brown_luma_max is not None:
        warm &= local_luma <= float(config.brown_luma_max)
    return warm


def _dark_background(local_luma: np.ndarray, local_value: np.ndarray, config: WhiteDustConfig) -> np.ndarray:
    return (local_luma <= float(config.dark_luma_max)) & (local_value <= float(config.dark_value_max))


def _configured_or_auto(configured: float | None, auto: float) -> float:
    return float(auto) if configured is None else float(configured)


def _center_focus_mask(height: int, width: int, margin_x: float, margin_y: float) -> np.ndarray:
    y0, x0, y1, x1 = _center_focus_bounds(height, width, margin_x, margin_y)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _center_focus_bounds(height: int, width: int, margin_x: float, margin_y: float) -> list[int]:
    x0 = int(round(width * float(margin_x)))
    x1 = int(round(width * (1.0 - float(margin_x))))
    y0 = int(round(height * float(margin_y)))
    y1 = int(round(height * (1.0 - float(margin_y))))
    return [y0, x0, y1, x1]


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
    if edge_mode == "wide":
        return _max_filter_bool(mask, 3)
    if edge_mode == "tight":
        return mask.astype(bool)
    return _min_filter_bool(_max_filter_bool(mask, 3), 3)


def _refine_patch_mask_by_edge_mode(mask: np.ndarray, seed: np.ndarray, mode: str) -> np.ndarray:
    edge_mode = _normalize_mask_edge_mode(mode)
    if edge_mode == "wide":
        return _max_filter_bool(mask | seed, 5)
    if edge_mode == "tight":
        return mask & _max_filter_bool(seed, 5)
    return _min_filter_bool(_max_filter_bool(mask | seed, 3), 3)


def _mask_edge_params(mode: str) -> dict[str, int]:
    edge_mode = _normalize_mask_edge_mode(mode)
    if edge_mode == "tight":
        return {"candidate_filter": 3}
    if edge_mode == "wide":
        return {"candidate_filter": 7}
    return {"candidate_filter": 5}


def _normalize_mask_edge_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"tight", "normal", "wide"}:
        return normalized
    raise ValueError(f"Unsupported mask_edge_mode: {mode}")


def _normalize_background_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"dark", "brown", "any"}:
        return normalized
    raise ValueError(f"Unsupported background_mode: {mode}")


def _ensure_rgb_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Unsupported image shape for white-dust detection: {arr.shape}")
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
            overlay[selected].astype(np.float32) * 0.32
            + np.asarray([0.0, 210.0, 255.0], dtype=np.float32) * 0.68,
            0,
            255,
        ).astype(np.uint8)
    return overlay


def _empty_rgb_artifact() -> np.ndarray:
    return np.zeros((0, 0, 3), dtype=np.uint8)


def _empty_gray_artifact() -> np.ndarray:
    return np.zeros((0, 0), dtype=np.uint8)


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
        "thickness": 0,
        "ring_empty": 0,
        "weak_brown_background": 0,
        "weak_background": 0,
        "weak_ring_contrast": 0,
        "empty_after_growth": 0,
    }


def _safe_stem(path: Path) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in path.stem)
    return safe[:80] or "image"


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)
