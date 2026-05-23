from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import RepairConfig
from .io import as_float32, restore_dtype, write_image
from .mask import (
    bbox_with_padding,
    bboxes_from_mask,
    connected_components,
    dilate_mask,
    feather_mask,
    filter_components,
    normalize_mask,
    threshold_mask,
)
from .metrics import compute_repair_metrics


@dataclass
class RepairResult:
    repaired_image: np.ndarray
    binary_mask: np.ndarray
    soft_mask: np.ndarray
    changed_bbox_list: list[tuple[int, int, int, int]]
    metrics: dict[str, Any]
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    debug_paths: dict[str, str] = field(default_factory=dict)


def repair_image(image: np.ndarray, mask: np.ndarray, config: RepairConfig | None = None) -> RepairResult:
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
    binary_mask = dilate_mask(filtered.mask, cfg.dilate_radius)
    soft_mask = feather_mask(binary_mask, cfg.feather_radius)
    changed_bbox_list = bboxes_from_mask(soft_mask > 0.0)
    collect_debug_images = bool(cfg.collect_debug_images or cfg.debug_dir is not None)

    if cfg.strength == 0.0 or not binary_mask.any():
        elapsed = (time.perf_counter() - start) * 1000.0
        repaired = original.copy()
        metrics = compute_repair_metrics(original, repaired, soft_mask, changed_bbox_list, elapsed)
        metrics.update(_mask_metrics(normalized.channel_used, filtered))
        result = RepairResult(
            repaired_image=repaired,
            binary_mask=binary_mask,
            soft_mask=soft_mask,
            changed_bbox_list=changed_bbox_list,
            metrics=metrics,
            debug_images=_debug_images(original, repaired, normalized_mask, binary_mask, soft_mask)
            if collect_debug_images
            else {},
        )
        _write_debug_outputs(cfg.debug_dir, result)
        return result

    image_float = as_float32(original)
    repair_candidate = image_float.copy()
    height, width = binary_mask.shape
    _labels, repair_components = connected_components(binary_mask)

    for component in repair_components:
        x0, y0, x1, y1 = bbox_with_padding(component.bbox, cfg.padding, width, height)
        roi = repair_candidate[y0:y1, x0:x1, :3]
        original_roi = image_float[y0:y1, x0:x1, :3]
        roi_mask = binary_mask[y0:y1, x0:x1]
        repaired_roi = _repair_roi(original_roi, roi_mask, cfg.method, component.area)
        roi[:, :, :] = repaired_roi

    alpha = np.clip(soft_mask * cfg.strength, 0.0, 1.0).astype(np.float32)
    output_float = image_float.copy()
    output_float[:, :, :3] = (
        image_float[:, :, :3] * (1.0 - alpha[:, :, None])
        + repair_candidate[:, :, :3] * alpha[:, :, None]
    )
    outside = alpha <= 0.0
    output_float[outside, :] = image_float[outside, :]
    repaired = restore_dtype(output_float, original.dtype)
    repaired[outside, :] = original[outside, :]

    elapsed = (time.perf_counter() - start) * 1000.0
    metrics = compute_repair_metrics(original, repaired, soft_mask, changed_bbox_list, elapsed)
    metrics.update(_mask_metrics(normalized.channel_used, filtered))

    result = RepairResult(
        repaired_image=repaired,
        binary_mask=binary_mask,
        soft_mask=soft_mask,
        changed_bbox_list=changed_bbox_list,
        metrics=metrics,
        debug_images=_debug_images(original, repaired, normalized_mask, binary_mask, soft_mask)
        if collect_debug_images
        else {},
    )
    _write_debug_outputs(cfg.debug_dir, result)
    return result


def _repair_roi(roi: np.ndarray, mask: np.ndarray, method: str, area: int) -> np.ndarray:
    if method == "median":
        repaired = _median_repair(roi, mask)
    elif method == "inpaint":
        repaired = _diffusion_inpaint(roi, mask)
    elif method == "denoise":
        repaired = _masked_denoise(roi, mask)
    elif method == "hybrid":
        if area <= 256:
            repaired = _diffusion_inpaint(roi, mask)
        else:
            repaired = _median_repair(roi, mask)
        smoothed = _box_blur_image(repaired, radius=1)
        out = repaired.copy()
        out[mask] = (repaired[mask] * 0.85) + (smoothed[mask] * 0.15)
        repaired = out
    elif method == "aggressive":
        repaired = _aggressive_repair(roi, mask)
    else:
        raise ValueError(f"Unsupported repair method: {method}")
    return _guard_repair_candidate(roi, repaired, mask)


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


def _guard_repair_candidate(roi: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return candidate
    context = ~mask
    if not context.any():
        return roi.copy()

    ring = _context_ring(mask, radius=10)
    sample = roi[ring] if ring.any() else roi[context]
    context_median = np.median(sample, axis=0)
    context_mad = np.median(np.abs(sample - context_median), axis=0)
    tolerance = max(0.025, float(np.max(context_mad)) * 3.0)

    original_distance = np.max(np.abs(roi - context_median), axis=2)
    candidate_distance = np.max(np.abs(candidate - context_median), axis=2)
    worse_than_original = mask & (candidate_distance > original_distance + tolerance)

    original_luma = _luminance(roi)
    candidate_luma = _luminance(candidate)
    local_luma = _luminance(_box_blur_image(roi, radius=2))
    darkens_clean_bright_area = (
        mask
        & (original_luma > 0.78)
        & (local_luma > 0.68)
        & (candidate_luma < original_luma - 0.08)
    )

    guarded = candidate.copy()
    reject = worse_than_original | darkens_clean_bright_area
    guarded[reject] = roi[reject]
    return guarded


def _masked_denoise(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return roi.copy()
    blurred = _box_blur_image(roi, radius=1)
    out = roi.copy()
    out[mask] = blurred[mask]
    return out


def _context_ring(mask: np.ndarray, radius: int) -> np.ndarray:
    from .mask import dilate_mask

    expanded = dilate_mask(mask, radius)
    return expanded & ~mask


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


def _mask_metrics(channel_used: str, filtered: Any) -> dict[str, int | str]:
    return {
        "mask_channel_used": channel_used,
        "kept_component_count": len(filtered.kept_components),
        "removed_small_component_count": len(filtered.removed_small),
        "removed_large_component_count": len(filtered.removed_large),
    }


def _debug_images(
    original: np.ndarray,
    repaired: np.ndarray,
    normalized_mask: np.ndarray,
    binary_mask: np.ndarray,
    soft_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    original_float = as_float32(original)
    repaired_float = as_float32(repaired)
    diff = np.max(np.abs(repaired_float[:, :, :3] - original_float[:, :, :3]), axis=2)
    diff_boosted = np.clip(diff * 8.0, 0.0, 1.0)
    diff_visualization = np.zeros((*diff.shape, 3), dtype=np.uint8)
    diff_visualization[:, :, 0] = np.rint(diff_boosted * 255.0).astype(np.uint8)
    diff_visualization[:, :, 1] = np.rint(np.clip(soft_mask, 0.0, 1.0) * 255.0).astype(np.uint8)

    return {
        "normalized_mask": np.rint(np.clip(normalized_mask, 0.0, 1.0) * 255.0).astype(np.uint8),
        "binary_mask": (binary_mask.astype(np.uint8) * 255),
        "soft_mask": np.rint(np.clip(soft_mask, 0.0, 1.0) * 255.0).astype(np.uint8),
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
        "repaired_preview": "repaired_preview.png",
        "diff_visualization": "diff_visualization.png",
    }
    for key, filename in names.items():
        path = output_dir / filename
        write_image(path, result.debug_images[key])
        result.debug_paths[key] = str(path)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2, sort_keys=True), encoding="utf-8")
    result.debug_paths["metrics"] = str(metrics_path)
