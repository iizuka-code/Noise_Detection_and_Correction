from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .io import as_float32
from .mask import Component, connected_components, dilate_mask

DEFECT_CLASSIFIER_VERSION = 1


@dataclass(frozen=True)
class DefectFeatures:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    width: int
    height: int
    aspect_ratio: float
    long_axis: float
    short_axis: float
    elongation: float
    thickness: float
    density: float
    touches_border: bool
    context_pixel_count: int
    context_available_ratio: float
    luminance_std: float
    gradient_mean: float
    gradient_anisotropy: float
    texture_score: float
    recommended_strategy: str


@dataclass(frozen=True)
class DefectClassificationConfig:
    tiny_area_threshold: int = 9
    small_area_threshold: int = 64
    max_patch_area: int = 1600
    line_elongation_threshold: float = 4.0
    structure_texture_threshold: float = 0.035
    structure_gradient_threshold: float = 0.030
    context_radius: int = 6


def classify_defects(
    image: np.ndarray,
    core_mask: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    components: list[Component] | None = None,
    repair_mask: np.ndarray | None = None,
    config: DefectClassificationConfig | None = None,
) -> list[DefectFeatures]:
    cfg = config or DefectClassificationConfig()
    mask = np.asarray(core_mask, dtype=bool)
    if labels is None or components is None:
        labels, components = connected_components(mask)
    global_repair_mask = np.asarray(repair_mask, dtype=bool) if repair_mask is not None else mask
    rgb = as_float32(np.asarray(image))[:, :, :3]
    luminance = _luminance(rgb)
    gradient_x, gradient_y, gradient_mag = _image_gradients(luminance)

    features: list[DefectFeatures] = []
    for component in components:
        component_mask = labels == component.label
        if not component_mask.any():
            continue
        features.append(
            _classify_component(
                component,
                component_mask,
                global_repair_mask,
                luminance,
                gradient_x,
                gradient_y,
                gradient_mag,
                cfg,
            )
        )
    return features


def summarize_defect_features(features: list[DefectFeatures]) -> dict[str, Any]:
    strategy_counts: dict[str, int] = {}
    areas: list[int] = []
    textures: list[float] = []
    gradients: list[float] = []
    for feature in features:
        strategy_counts[feature.recommended_strategy] = strategy_counts.get(feature.recommended_strategy, 0) + 1
        areas.append(feature.area)
        textures.append(feature.texture_score)
        gradients.append(feature.gradient_mean)
    return {
        "defect_classification_enabled": True,
        "defect_component_count": len(features),
        "defect_strategy_counts": strategy_counts,
        "defect_area_histogram": _area_histogram(areas),
        "defect_texture_summary": {
            "mean": _mean_or_zero(textures),
            "max": _max_or_zero(textures),
            "gradient_mean": _mean_or_zero(gradients),
        },
        "defect_classifier_version": DEFECT_CLASSIFIER_VERSION,
    }


def defect_debug_payload(features: list[DefectFeatures], *, max_components: int = 1000) -> dict[str, Any]:
    summary = summarize_defect_features(features)
    truncated = len(features) > max_components
    return {
        "version": DEFECT_CLASSIFIER_VERSION,
        "summary": summary,
        "component_count": len(features),
        "components_truncated": truncated,
        "components": [asdict(feature) for feature in features[:max_components]],
    }


def _classify_component(
    component: Component,
    component_mask: np.ndarray,
    repair_mask: np.ndarray,
    luminance: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    gradient_mag: np.ndarray,
    cfg: DefectClassificationConfig,
) -> DefectFeatures:
    height, width = component_mask.shape
    x0, y0, x1, y1 = component.bbox
    component_width = int(x1 - x0)
    component_height = int(y1 - y0)
    area = int(component.area)
    aspect_ratio = max(component_width, component_height) / float(max(1, min(component_width, component_height)))
    long_axis, short_axis, elongation = _pca_axes(component_mask, component_width, component_height)
    thickness = area / max(long_axis, 1.0e-6)
    density = area / float(max(1, component_width * component_height))
    touches_border = bool(x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height)
    context = _component_context(component_mask, repair_mask, cfg.context_radius)
    context_band = dilate_mask(component_mask, cfg.context_radius) & ~component_mask
    context_count = int(np.count_nonzero(context))
    context_available_ratio = context_count / float(max(1, int(np.count_nonzero(context_band))))
    luminance_values = luminance[context]
    luminance_std = float(np.std(luminance_values)) if luminance_values.size else 0.0
    gradient_context = context & ~dilate_mask(repair_mask, 1)
    if not gradient_context.any():
        gradient_context = context
    gx_values = np.abs(gradient_x[gradient_context])
    gy_values = np.abs(gradient_y[gradient_context])
    gradient_values = gradient_mag[gradient_context]
    gradient_mean = float(np.mean(gradient_values)) if gradient_values.size else 0.0
    gx_mean = float(np.mean(gx_values)) if gx_values.size else 0.0
    gy_mean = float(np.mean(gy_values)) if gy_values.size else 0.0
    gradient_anisotropy = abs(gx_mean - gy_mean) / max(gx_mean + gy_mean, 1.0e-6)
    texture_score = float(0.6 * luminance_std + 0.4 * gradient_mean)
    recommended_strategy = _recommend_strategy(
        area=area,
        elongation=elongation,
        thickness=thickness,
        texture_score=texture_score,
        gradient_mean=gradient_mean,
        context_pixel_count=context_count,
        context_available_ratio=context_available_ratio,
        touches_border=touches_border,
        cfg=cfg,
    )
    return DefectFeatures(
        label=int(component.label),
        area=area,
        bbox=(int(y0), int(x0), int(y1), int(x1)),
        width=component_width,
        height=component_height,
        aspect_ratio=float(aspect_ratio),
        long_axis=float(long_axis),
        short_axis=float(short_axis),
        elongation=float(elongation),
        thickness=float(thickness),
        density=float(density),
        touches_border=touches_border,
        context_pixel_count=context_count,
        context_available_ratio=float(context_available_ratio),
        luminance_std=luminance_std,
        gradient_mean=gradient_mean,
        gradient_anisotropy=float(gradient_anisotropy),
        texture_score=texture_score,
        recommended_strategy=recommended_strategy,
    )


def _recommend_strategy(
    *,
    area: int,
    elongation: float,
    thickness: float,
    texture_score: float,
    gradient_mean: float,
    context_pixel_count: int,
    context_available_ratio: float,
    touches_border: bool,
    cfg: DefectClassificationConfig,
) -> str:
    if context_pixel_count < 3 or (touches_border and context_available_ratio < 0.15):
        return "skip"
    if area <= cfg.tiny_area_threshold:
        return "tiny_local"
    if elongation >= cfg.line_elongation_threshold and thickness <= 4.0:
        return "directional"
    if area <= cfg.small_area_threshold and texture_score < cfg.structure_texture_threshold:
        return "small_local"
    if area <= cfg.max_patch_area and (
        texture_score >= cfg.structure_texture_threshold or gradient_mean >= cfg.structure_gradient_threshold
    ):
        return "patch"
    if context_pixel_count >= max(8, min(area, 16)):
        return "fast_inpaint"
    return "skip"


def _component_context(component_mask: np.ndarray, repair_mask: np.ndarray, radius: int) -> np.ndarray:
    expanded = dilate_mask(component_mask, radius)
    return expanded & ~repair_mask


def _pca_axes(mask: np.ndarray, bbox_width: int, bbox_height: int) -> tuple[float, float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        long_axis = float(max(bbox_width, bbox_height, 1))
        short_axis = float(max(1, min(bbox_width, bbox_height)))
        return long_axis, short_axis, long_axis / max(short_axis, 1.0e-6)
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    coords -= np.mean(coords, axis=0)
    try:
        values = np.linalg.eigvalsh(np.cov(coords, rowvar=False))
    except np.linalg.LinAlgError:
        values = np.array([0.0, 0.0], dtype=np.float32)
    values = np.maximum(np.sort(values), 0.0)
    short_axis = max(float(np.sqrt(values[0]) * 4.0), 1.0)
    long_axis = max(float(np.sqrt(values[-1]) * 4.0), float(max(bbox_width, bbox_height, 1)))
    return long_axis, short_axis, long_axis / max(short_axis, 1.0e-6)


def _image_gradients(luminance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = np.zeros_like(luminance, dtype=np.float32)
    gy = np.zeros_like(luminance, dtype=np.float32)
    gx[:, 1:-1] = (luminance[:, 2:] - luminance[:, :-2]) * 0.5
    gx[:, 0] = luminance[:, 1] - luminance[:, 0] if luminance.shape[1] > 1 else 0.0
    gx[:, -1] = luminance[:, -1] - luminance[:, -2] if luminance.shape[1] > 1 else 0.0
    gy[1:-1, :] = (luminance[2:, :] - luminance[:-2, :]) * 0.5
    gy[0, :] = luminance[1, :] - luminance[0, :] if luminance.shape[0] > 1 else 0.0
    gy[-1, :] = luminance[-1, :] - luminance[-2, :] if luminance.shape[0] > 1 else 0.0
    mag = np.sqrt((gx * gx) + (gy * gy)).astype(np.float32)
    return gx.astype(np.float32), gy.astype(np.float32), mag


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    ).astype(np.float32)


def _area_histogram(areas: list[int]) -> dict[str, int]:
    histogram = {"1-9": 0, "10-64": 0, "65-256": 0, "257-1600": 0, "1601+": 0}
    for area in areas:
        if area <= 9:
            histogram["1-9"] += 1
        elif area <= 64:
            histogram["10-64"] += 1
        elif area <= 256:
            histogram["65-256"] += 1
        elif area <= 1600:
            histogram["257-1600"] += 1
        else:
            histogram["1601+"] += 1
    return histogram


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _max_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(np.asarray(values, dtype=np.float32)))


__all__ = [
    "DEFECT_CLASSIFIER_VERSION",
    "DefectClassificationConfig",
    "DefectFeatures",
    "classify_defects",
    "defect_debug_payload",
    "summarize_defect_features",
]
