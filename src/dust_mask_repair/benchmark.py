from __future__ import annotations

import argparse
import json
import statistics
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .config import REPAIR_METHODS, RepairConfig
from .mask import dilate_mask
from .red_highlight import RedHighlightConfig, detect_red_highlight_source_image
from .repair import repair_image


BENCHMARK_VERSION = "red_highlight_repair_bench_v1"
DEFECT_AWARE_QUALITY_VERSION = "defect_aware_quality_v1"
FREQUENCY_GUIDED_QUALITY_VERSION = "frequency_guided_quality_v1"
FREQUENCY_GUIDED_CASES = (
    "selected_sky_gradient",
    "selected_vertical_hair",
    "selected_grid",
    "many_defects_one_selected",
    "no_selection_regression",
)

DEFECT_AWARE_QUALITY_CASES = (
    "flat_dots",
    "gradient_dust",
    "grain_dust",
    "stripe_texture",
    "diagonal_edge",
    "thin_scratch",
    "diagonal_edge_micro_dust",
    "chroma_edge_micro_dust",
    "thin_line_micro_dust",
    "gradient_micro_dust",
    "mottled_background_dark_dust",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    width: int = 1280
    height: int = 853
    iterations: int = 3
    warmup: int = 1
    seed: int = 20260523
    detection_long_edge: int = 1280
    include_long_scratches: bool = False
    method: str = "hybrid"
    dilate_radius: int = 1
    feather_radius: int = 1
    padding: int = 16

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be > 0")
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.detection_long_edge <= 0:
            raise ValueError("detection_long_edge must be > 0")
        if self.method not in REPAIR_METHODS:
            raise ValueError(f"Unsupported repair method: {self.method}")
        if self.dilate_radius < 0:
            raise ValueError("dilate_radius must be >= 0")
        if self.feather_radius < 0:
            raise ValueError("feather_radius must be >= 0")
        if self.padding < 0:
            raise ValueError("padding must be >= 0")


def run_benchmark(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    cfg = config or BenchmarkConfig()
    cfg.validate()

    normal_image, red_image = make_benchmark_fixture(cfg.width, cfg.height, cfg.seed, cfg.include_long_scratches)
    red_config = RedHighlightConfig(
        detection_long_edge=cfg.detection_long_edge,
        mask_edge_mode="normal",
        include_long_scratches=cfg.include_long_scratches,
        max_area=max(1400, int(cfg.width * cfg.height * 0.006)),
        max_dim=max(95, int(max(cfg.width, cfg.height) * 0.08)),
        max_scratch_area=max(9000, int(cfg.width * cfg.height * 0.035)),
        max_scratch_dim=max(720, int(max(cfg.width, cfg.height) * 0.75)),
        max_scratch_width=max(48, int(min(cfg.width, cfg.height) * 0.035)),
        visual_artifacts=False,
    )
    repair_config = RepairConfig(
        method=cfg.method,
        mask_channel="grayscale",
        threshold=0.5,
        dilate_radius=cfg.dilate_radius,
        feather_radius=cfg.feather_radius,
        strength=1.0,
        min_component_area=1,
        max_component_area=max(5000, int(cfg.width * cfg.height * 0.04)),
        padding=cfg.padding,
    )

    runs: list[dict[str, Any]] = []
    total_runs = cfg.warmup + cfg.iterations
    for index in range(total_runs):
        run = _run_once(normal_image, red_image, red_config, repair_config)
        run["phase"] = "warmup" if index < cfg.warmup else "measured"
        run["run_index"] = index
        if index >= cfg.warmup:
            runs.append(run)

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "image_shape": [cfg.height, cfg.width],
        "config": asdict(cfg),
        "red_config": asdict(red_config),
        "repair_config": _repair_config_summary(repair_config),
        "input_bytes": {
            "normal_image": int(normal_image.nbytes),
            "red_image": int(red_image.nbytes),
        },
        "memory_notes": "peak_traced_memory_bytes is measured with tracemalloc and may exclude some native allocations.",
        "runs": runs,
        "summary": _summary(runs),
    }


def make_benchmark_fixture(
    width: int,
    height: int,
    seed: int = 20260523,
    include_long_scratch: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    grain = rng.normal(0.0, 2.0, size=(height, width, 1)).astype(np.float32)
    normal = np.stack(
        [
            80.0 + 42.0 * xx + 14.0 * yy,
            94.0 + 22.0 * xx + 24.0 * yy,
            116.0 + 16.0 * xx + 30.0 * yy,
        ],
        axis=2,
    )
    normal += 4.0 * np.sin(xx[:, :, None] * 24.0) + 3.0 * np.cos(yy[:, :, None] * 17.0)
    normal += grain

    red = np.zeros((height, width, 3), dtype=np.float32)
    red[:, :, 0] = 6.0 + 4.0 * yy + 2.0 * np.sin(xx * 10.0)
    red[:, :, 1] = 3.0 + 1.5 * yy
    red[:, :, 2] = 4.0 + 1.5 * xx
    border_width = max(2, int(round(width * 0.035)))
    red[:, width - border_width :, 0] += 82.0
    red[:, width - border_width :, 1] += 8.0
    red[:, width - border_width :, 2] += 4.0

    for cy, cx, radius in _scaled_spots(width, height):
        _paint_disk(normal, cy, cx, radius, (16.0, 15.0, 14.0))
        _paint_disk(red, cy, cx, radius + 5, (44.0, 4.0, 5.0))
        _paint_disk(red, cy, cx, radius + 2, (122.0, 9.0, 11.0))
        _paint_disk(red, cy, cx, radius, (238.0, 18.0, 24.0))

    if include_long_scratch:
        _paint_scratch(normal, red, width, height)

    return np.clip(normal, 0, 255).astype(np.uint8), np.clip(red, 0, 255).astype(np.uint8)


def make_defect_aware_quality_case(
    case: str,
    *,
    width: int = 72,
    height: int = 56,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if case not in DEFECT_AWARE_QUALITY_CASES:
        raise ValueError(f"Unsupported defect-aware quality case: {case}")
    yy, xx = np.indices((height, width), dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.float32)
    if case == "flat_dots":
        clean[:, :] = [0.34, 0.43, 0.52]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 3, width // 3] = 255
        mask[height // 2, width // 2] = 255
    elif case == "gradient_dust":
        clean[:, :, 0] = 0.12 + xx * 0.009
        clean[:, :, 1] = 0.20 + yy * 0.010
        clean[:, :, 2] = 0.72 - xx * 0.004 + yy * 0.002
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 2 - 5 : height // 2 + 5, width // 2 - 5 : width // 2 + 5] = 255
    elif case == "grain_dust":
        texture = (((xx * 17 + yy * 29) % 11) - 5.0) / 255.0
        clean[:, :, 0] = 0.34 + texture
        clean[:, :, 1] = 0.42 - texture * 0.8
        clean[:, :, 2] = 0.50 + texture * 0.5
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 2 - 5 : height // 2 + 5, width // 2 - 6 : width // 2 + 6] = 255
    elif case == "stripe_texture":
        stripe = ((np.arange(width) // 4) % 2).astype(bool)
        clean[:, :, 0] = np.where(stripe[None, :], 0.75, 0.18)
        clean[:, :, 1] = np.where(stripe[None, :], 0.55, 0.30)
        clean[:, :, 2] = np.where(stripe[None, :], 0.25, 0.62)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 2 - 5 : height // 2 + 5, width // 2 - 5 : width // 2 + 5] = 255
    elif case == "diagonal_edge":
        edge = yy > (0.55 * xx + 8.0)
        clean[:, :] = [0.25, 0.32, 0.42]
        clean[edge] = [0.70, 0.66, 0.52]
        mask = np.zeros((height, width), dtype=np.uint8)
        for index in range(18, min(width - 10, height - 8)):
            mask[index, index] = 255
    elif case == "diagonal_edge_micro_dust":
        cy = height // 2
        cx = width // 2
        edge = (yy - float(cy)) > (xx - float(cx))
        clean[:, :] = [0.16, 0.31, 0.71]
        clean[edge] = [0.82, 0.63, 0.24]
        mask = np.zeros((height, width), dtype=np.uint8)
        for y, x in ((cy, cx + 1), (cy + 1, cx), (cy + 2, cx + 1), (cy + 3, cx + 2)):
            if 0 <= y < height and 0 <= x < width:
                mask[y, x] = 255
    elif case == "chroma_edge_micro_dust":
        edge = xx >= width // 2
        clean[:, :] = [0.25, 0.55, 0.48]
        clean[edge] = [0.72, 0.31, 0.47]
        mask = np.zeros((height, width), dtype=np.uint8)
        cy = height // 2
        cx = width // 2 + 1
        mask[cy - 1 : cy + 2, cx] = 255
    elif case == "thin_line_micro_dust":
        clean[:, :, 0] = 0.34 + xx * 0.002
        clean[:, :, 1] = 0.42 + yy * 0.001
        clean[:, :, 2] = 0.50 - xx * 0.001
        line_y = height // 2
        clean[line_y, 10 : width - 10] = [0.08, 0.10, 0.12]
        mask = np.zeros((height, width), dtype=np.uint8)
        cx = width // 2
        mask[line_y, cx - 1 : cx + 2] = 255
    elif case == "gradient_micro_dust":
        clean[:, :, 0] = 0.16 + xx * 0.006
        clean[:, :, 1] = 0.24 + yy * 0.005
        clean[:, :, 2] = 0.70 - xx * 0.003 + yy * 0.001
        mask = np.zeros((height, width), dtype=np.uint8)
        cy = height // 2
        cx = width // 2
        mask[cy - 1 : cy + 2, cx - 1 : cx + 2] = 255
    elif case == "mottled_background_dark_dust":
        rng = np.random.default_rng(8021 + width * 17 + height * 31)
        mottle = 0.034 * np.sin(xx * 0.33 + yy * 0.19) + 0.026 * np.cos(xx * 0.12 - yy * 0.29)
        grain = rng.normal(0.0, 0.0075, size=(height, width)).astype(np.float32)
        warm_texture = mottle + grain
        clean[:, :, 0] = 0.50 + xx * 0.0016 + warm_texture * 1.10
        clean[:, :, 1] = 0.42 + yy * 0.0014 + warm_texture * 0.78 + 0.012 * np.sin(yy * 0.42)
        clean[:, :, 2] = 0.57 - xx * 0.0011 + warm_texture * 0.95 + 0.010 * np.cos(xx * 0.27)
        cy = height // 2
        cx = width // 2
        mask = np.zeros((height, width), dtype=np.uint8)
        yy_i, xx_i = np.ogrid[:height, :width]
        for dy, dx, radius in ((-6, -8, 1), (4, 7, 2), (9, -2, 1)):
            disk = (yy_i - (cy + dy)) ** 2 + (xx_i - (cx + dx)) ** 2 <= radius * radius
            mask[disk] = 255
        for step in range(-8, 9):
            y = cy - 1 + int(round(step * 0.35))
            x = cx + step
            if 0 <= y < height and 0 <= x < width:
                mask[y, x] = 255
                if step % 3 == 0 and y + 1 < height:
                    mask[y + 1, x] = 255
    else:
        clean[:, :] = [0.36, 0.44, 0.53]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 2, 12 : width - 12] = 255

    clean = np.clip(clean, 0.0, 1.0)
    damaged = clean.copy()
    defect_rgb = [0.02, 0.018, 0.020] if case == "mottled_background_dark_dust" else [1.0, 1.0, 1.0]
    damaged[mask > 0] = defect_rgb
    return (
        np.rint(clean * 255.0).astype(np.uint8),
        np.rint(damaged * 255.0).astype(np.uint8),
        mask,
    )


def evaluate_defect_aware_quality_case(
    case: str,
    *,
    method: str = "defect_aware",
    width: int = 72,
    height: int = 56,
) -> dict[str, Any]:
    clean, damaged, mask = make_defect_aware_quality_case(case, width=width, height=height)
    repair_config = RepairConfig(
        method=method,
        mask_channel="grayscale",
        dilate_radius=0,
        feather_radius=0,
        padding=18,
        max_component_area=5000,
    )
    result = repair_image(damaged, mask, repair_config)

    edge_disabled_error = None
    tone_enabled_isolated_error = None
    tone_disabled_error = None
    if method == "defect_aware":
        edge_disabled_config = RepairConfig(
            method=method,
            mask_channel="grayscale",
            dilate_radius=0,
            feather_radius=0,
            padding=18,
            max_component_area=5000,
            edge_guided_enabled=False,
        )
        edge_disabled = repair_image(damaged, mask, edge_disabled_config)
        edge_disabled_error = _mean_abs_error(edge_disabled.repaired_image, clean, mask > 0)

        tone_enabled_isolated_config = RepairConfig(
            method=method,
            mask_channel="grayscale",
            dilate_radius=0,
            feather_radius=0,
            padding=18,
            max_component_area=5000,
            edge_guided_enabled=False,
        )
        tone_enabled_isolated = repair_image(damaged, mask, tone_enabled_isolated_config)
        tone_enabled_isolated_error = _mean_abs_error(tone_enabled_isolated.repaired_image, clean, mask > 0)

        tone_disabled_config = RepairConfig(
            method=method,
            mask_channel="grayscale",
            dilate_radius=0,
            feather_radius=0,
            padding=18,
            max_component_area=5000,
            edge_guided_enabled=False,
            tone_guided_enabled=False,
        )
        tone_disabled = repair_image(damaged, mask, tone_disabled_config)
        tone_disabled_error = _mean_abs_error(tone_disabled.repaired_image, clean, mask > 0)

    inside = mask > 0
    repaired_error = _mean_abs_error(result.repaired_image, clean, inside)
    corrupted_error = _mean_abs_error(damaged, clean, inside)
    clean_float = clean.astype(np.float32)
    damaged_float = damaged.astype(np.float32)
    repaired_float = result.repaired_image.astype(np.float32)
    luma_clean = _luminance_255(clean_float)
    luma_damaged = _luminance_255(damaged_float)
    luma_repaired = _luminance_255(repaired_float)
    context = dilate_mask(inside, 6) & ~inside
    clean_var = float(np.mean(np.var(clean_float[context], axis=0))) if np.any(context) else 0.0
    repaired_var = float(np.mean(np.var(repaired_float[inside], axis=0))) if np.any(inside) else 0.0
    variance_retention = float(min(repaired_var, clean_var) / max(clean_var, 1.0e-6)) if clean_var > 0.0 else 0.0
    context_luma = float(np.mean(luma_clean[context])) if np.any(context) else 0.0
    repaired_luma = float(np.mean(luma_repaired[inside])) if np.any(inside) else 0.0
    residual_dark_contrast = max(0.0, context_luma - repaired_luma)

    return {
        "benchmark_version": DEFECT_AWARE_QUALITY_VERSION,
        "case": case,
        "method": method,
        "processing_time_ms": float(result.metrics["processing_time_ms"]),
        "mask_pixel_count": int(np.count_nonzero(mask)),
        "component_count": int(result.metrics.get("kept_component_count", 0)),
        "mean_abs_error_inside_mask": repaired_error,
        "corrupted_mean_abs_error_inside_mask": corrupted_error,
        "clean_core_mae": repaired_error,
        "core_rgb_mae": repaired_error,
        "core_luminance_mae": float(np.mean(np.abs(luma_repaired[inside] - luma_clean[inside]))) if np.any(inside) else 0.0,
        "corrupted_core_luminance_mae": float(np.mean(np.abs(luma_damaged[inside] - luma_clean[inside]))) if np.any(inside) else 0.0,
        "corrupted_improvement_ratio": float((corrupted_error - repaired_error) / max(corrupted_error, 1.0e-6)),
        "core_outside_max_diff": float(result.metrics.get("max_abs_diff_outside_original_mask", result.metrics["max_abs_diff_outside_mask"])),
        "shell_outside_max_diff": float(result.metrics.get("max_abs_diff_outside_repair_mask", result.metrics["max_abs_diff_outside_mask"])),
        "max_abs_diff_outside_mask": float(result.metrics["max_abs_diff_outside_mask"]),
        "local_variance_retention": variance_retention,
        "residual_dark_contrast": float(residual_dark_contrast),
        "edge_guided_disabled_mean_abs_error_inside_mask": edge_disabled_error,
        "tone_guided_enabled_isolated_mean_abs_error_inside_mask": tone_enabled_isolated_error,
        "tone_guided_disabled_mean_abs_error_inside_mask": tone_disabled_error,
        "strategy_counts": result.metrics.get("defect_strategy_counts", {}),
        "small_local_edge_guided_component_count": int(result.metrics.get("small_local_edge_guided_component_count", 0)),
        "small_local_edge_guided_pixel_count": int(result.metrics.get("small_local_edge_guided_pixel_count", 0)),
        "tone_guided_component_count": int(result.metrics.get("tone_guided_component_count", 0)),
        "tone_guided_pixel_count": int(result.metrics.get("tone_guided_pixel_count", 0)),
        "tone_guided_score_mean": float(result.metrics.get("tone_guided_score_mean", 0.0)),
        "tone_guided_context_rgb_distance_mean": float(result.metrics.get("tone_guided_context_rgb_distance_mean", 0.0)),
        "defect_core_alpha_below_full_count": int(result.metrics.get("defect_core_alpha_below_full_count", 0)),
    }



def make_frequency_guided_quality_case(
    case: str,
    *,
    width: int = 96,
    height: int = 72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if case not in FREQUENCY_GUIDED_CASES:
        raise ValueError(f"Unsupported frequency-guided quality case: {case}")
    yy, xx = np.indices((height, width), dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    scope = np.zeros((height, width), dtype=np.uint8)
    if case in {"selected_sky_gradient", "no_selection_regression"}:
        clean[:, :, 0] = 0.18 + xx * 0.003
        clean[:, :, 1] = 0.30 + yy * 0.0025
        clean[:, :, 2] = 0.72 - xx * 0.0015
        points = [(height // 2, width // 3), (height // 2 + 8, width - width // 3)]
        for y, x in points:
            mask[y - 1 : y + 2, x - 1 : x + 2] = 255
        scope[points[0][0] - 4 : points[0][0] + 5, points[0][1] - 4 : points[0][1] + 5] = 255
    elif case == "selected_vertical_hair":
        clean[:, :] = [0.44, 0.50, 0.58]
        x = width // 2
        clean[8 : height - 8, x : x + 2] = [0.08, 0.08, 0.09]
        mask[height // 2 - 2 : height // 2 + 3, x : x + 2] = 255
        scope[height // 2 - 6 : height // 2 + 7, x - 3 : x + 5] = 255
    elif case == "selected_grid":
        clean[:, :] = [0.38, 0.44, 0.52]
        clean[::8, :, :] = [0.08, 0.10, 0.13]
        clean[:, ::8, :] = [0.78, 0.72, 0.62]
        y = (height // 2 // 8) * 8
        x = (width // 2 // 8) * 8
        mask[y - 2 : y + 3, x - 2 : x + 3] = 255
        scope[y - 5 : y + 6, x - 5 : x + 6] = 255
    else:  # many_defects_one_selected
        clean[:, :, 0] = 0.25 + xx * 0.002
        clean[:, :, 1] = 0.34 + yy * 0.002
        clean[:, :, 2] = 0.55
        sy, sx = height // 2, width // 2
        for y in range(6, height - 6, 12):
            for x in range(6, width - 6, 12):
                if abs(y - sy) <= 6 and abs(x - sx) <= 6:
                    continue
                mask[y, x] = 255
        mask[sy, sx] = 255
        scope[sy - 2 : sy + 3, sx - 2 : sx + 3] = 255
    damaged = clean.copy()
    damaged[mask > 0] = [1.0, 1.0, 1.0]
    if case == "no_selection_regression":
        scope[:, :] = 0
    return (
        np.rint(np.clip(clean, 0.0, 1.0) * 255.0).astype(np.uint8),
        np.rint(np.clip(damaged, 0.0, 1.0) * 255.0).astype(np.uint8),
        mask,
        scope,
    )


def evaluate_frequency_guided_quality_case(
    case: str,
    *,
    width: int = 96,
    height: int = 72,
) -> dict[str, Any]:
    clean, damaged, mask, scope = make_frequency_guided_quality_case(case, width=width, height=height)
    cfg = RepairConfig(
        method="defect_aware",
        mask_channel="grayscale",
        dilate_radius=0,
        feather_radius=0,
        padding=18,
        max_component_area=max(5000, int(width * height)),
        grain_reinject_strength=0.0,
        frequency_guided_enabled=case != "no_selection_regression",
    )
    started = perf_counter()
    result = repair_image(
        damaged,
        mask,
        cfg,
        frequency_scope_mask=None if case == "no_selection_regression" else scope,
    )
    elapsed = _elapsed_ms(started)
    selected = scope > 0
    selected_core = (mask > 0) & selected
    if not selected_core.any():
        selected_core = mask > 0
    repaired = result.repaired_image
    luma_clean = _luminance_255(clean.astype(np.float32))
    luma_repaired = _luminance_255(repaired.astype(np.float32))
    luma_damaged = _luminance_255(damaged.astype(np.float32))
    selected_error = _mean_abs_error(repaired, clean, selected_core)
    corrupted_error = _mean_abs_error(damaged, clean, selected_core)
    unselected = (mask > 0) & ~selected_core
    return {
        "benchmark_version": FREQUENCY_GUIDED_QUALITY_VERSION,
        "case": case,
        "processing_time_ms": float(elapsed),
        "selected_core_rgb_mae": selected_error,
        "selected_core_luminance_mae": float(np.mean(np.abs(luma_repaired[selected_core] - luma_clean[selected_core]))) if selected_core.any() else 0.0,
        "corrupted_selected_core_rgb_mae": corrupted_error,
        "corrupted_selected_core_luminance_mae": float(np.mean(np.abs(luma_damaged[selected_core] - luma_clean[selected_core]))) if selected_core.any() else 0.0,
        "corrupted_improvement_ratio": float((corrupted_error - selected_error) / max(corrupted_error, 1.0e-6)),
        "unselected_component_output_diff": _mean_abs_error(repaired, damaged, unselected) if unselected.any() else 0.0,
        "max_abs_diff_outside_mask": float(result.metrics["max_abs_diff_outside_mask"]),
        "frequency_analyzed_component_count": int(result.metrics.get("frequency_analyzed_component_count", 0)),
        "frequency_selected_component_count": int(result.metrics.get("frequency_selected_component_count", 0)),
        "frequency_candidate_count_total": int(result.metrics.get("frequency_candidate_count_total", 0)),
        "frequency_pattern_counts": result.metrics.get("frequency_pattern_counts", {}),
        "frequency_context_low_energy_mean": float(result.metrics.get("frequency_context_low_energy_mean", 0.0)),
        "frequency_context_mid_energy_mean": float(result.metrics.get("frequency_context_mid_energy_mean", 0.0)),
        "frequency_anisotropy_mean": float(result.metrics.get("frequency_anisotropy_mean", 0.0)),
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark red-highlight detection followed by masked repair.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=853)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--detection-long-edge", type=int, default=1280)
    parser.add_argument("--include-long-scratches", action="store_true")
    parser.add_argument("--method", choices=REPAIR_METHODS, default="hybrid")
    parser.add_argument("--dilate-radius", type=int, default=1)
    parser.add_argument("--feather-radius", type=int, default=1)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--output-json", default=None, help="Optional path to write the benchmark JSON.")
    args = parser.parse_args(argv)

    result = run_benchmark(
        BenchmarkConfig(
            width=args.width,
            height=args.height,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
            detection_long_edge=args.detection_long_edge,
            include_long_scratches=bool(args.include_long_scratches),
            method=args.method,
            dilate_radius=args.dilate_radius,
            feather_radius=args.feather_radius,
            padding=args.padding,
        )
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


def _run_once(
    normal_image: np.ndarray,
    red_image: np.ndarray,
    red_config: RedHighlightConfig,
    repair_config: RepairConfig,
) -> dict[str, Any]:
    tracemalloc.start()
    started = perf_counter()
    red_started = perf_counter()
    red_result = detect_red_highlight_source_image(red_image, red_config)
    detect_ms = _elapsed_ms(red_started)
    repair_started = perf_counter()
    repair_result = repair_image(normal_image, red_result.mask, repair_config)
    repair_ms = _elapsed_ms(repair_started)
    total_ms = _elapsed_ms(started)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "detect_ms": detect_ms,
        "repair_ms": repair_ms,
        "total_ms": total_ms,
        "peak_traced_memory_bytes": int(peak),
        "component_count": int(red_result.manifest["component_count"]),
        "final_mask_pixels": int(red_result.manifest["final_mask_pixels"]),
        "changed_pixel_count": int(repair_result.metrics["changed_pixel_count"]),
        "max_abs_diff_outside_mask": float(repair_result.metrics["max_abs_diff_outside_mask"]),
    }


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "detect_ms": _stats([float(run["detect_ms"]) for run in runs]),
        "repair_ms": _stats([float(run["repair_ms"]) for run in runs]),
        "total_ms": _stats([float(run["total_ms"]) for run in runs]),
        "peak_traced_memory_bytes": _stats([float(run["peak_traced_memory_bytes"]) for run in runs]),
        "final_mask_pixels": _stats([float(run["final_mask_pixels"]) for run in runs]),
        "changed_pixel_count": _stats([float(run["changed_pixel_count"]) for run in runs]),
        "max_abs_diff_outside_mask_max": max(float(run["max_abs_diff_outside_mask"]) for run in runs),
    }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 3),
        "median": round(float(statistics.median(values)), 3),
        "mean": round(float(statistics.fmean(values)), 3),
        "max": round(max(values), 3),
    }


def _repair_config_summary(config: RepairConfig) -> dict[str, Any]:
    data = asdict(config)
    data["debug_dir"] = None if config.debug_dir is None else str(config.debug_dir)
    return data


def _scaled_spots(width: int, height: int) -> list[tuple[int, int, int]]:
    scale = max(1.0, min(width, height) / 640.0)
    points = [
        (0.18, 0.20, 5),
        (0.34, 0.54, 7),
        (0.58, 0.40, 6),
        (0.72, 0.76, 8),
        (0.84, 0.32, 5),
    ]
    return [
        (
            int(round(height * y)),
            int(round(width * x)),
            max(2, int(round(radius * scale))),
        )
        for y, x, radius in points
    ]


def _paint_disk(image: np.ndarray, cy: int, cx: int, radius: int, color: tuple[float, float, float]) -> None:
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    image[disk] = np.asarray(color, dtype=np.float32)


def _paint_scratch(normal: np.ndarray, red: np.ndarray, width: int, height: int) -> None:
    grid_y, grid_x = np.indices((height, width))
    x0 = int(round(width * 0.18))
    x1 = int(round(width * 0.82))
    center_y = height * 0.62 + (grid_x - x0) * 0.08
    scratch = (grid_x >= x0) & (grid_x <= x1) & (np.abs(grid_y - center_y) <= max(1.5, height * 0.003))
    halo = (grid_x >= x0) & (grid_x <= x1) & (np.abs(grid_y - center_y) <= max(4.0, height * 0.006))
    normal[scratch] = np.asarray([18.0, 17.0, 16.0], dtype=np.float32)
    red[halo] = np.asarray([54.0, 5.0, 6.0], dtype=np.float32)
    red[scratch] = np.asarray([235.0, 18.0, 24.0], dtype=np.float32)


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)


def _luminance_255(rgb_255: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb_255, dtype=np.float32)
    return values[:, :, 0] * 0.2126 + values[:, :, 1] * 0.7152 + values[:, :, 2] * 0.0722


def _mean_abs_error(image: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(image[mask].astype(np.int64) - reference[mask].astype(np.int64))))


if __name__ == "__main__":
    raise SystemExit(main())
