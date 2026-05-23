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
from .red_highlight import RedHighlightConfig, detect_red_highlight_source_image
from .repair import repair_image


BENCHMARK_VERSION = "red_highlight_repair_bench_v1"


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


if __name__ == "__main__":
    raise SystemExit(main())
