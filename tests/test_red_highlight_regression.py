from __future__ import annotations

from pathlib import Path

import numpy as np

from dust_mask_repair.config import RepairConfig
from dust_mask_repair.io import write_image
from dust_mask_repair.red_highlight import RedHighlightConfig, detect_red_highlight_source_image
from dust_mask_repair.repair import repair_image


ROOT = Path(__file__).resolve().parents[1]


def _artifact_path(name: str) -> Path:
    directory = ROOT / "test_outputs" / "red_highlight_regression"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _disk_mask(shape: tuple[int, int], cy: int, cx: int, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius


def _short_scratch_mask(shape: tuple[int, int], y0: int, x0: int, length: int, half_width: int) -> np.ndarray:
    yy, xx = np.indices(shape)
    # Slight diagonal scratch with bounded length. This is deliberately shorter than
    # the dedicated long-scratch case that will be handled in a later slice.
    center_y = y0 + (xx - x0) * 0.22
    in_length = (xx >= x0) & (xx <= x0 + length)
    return in_length & (np.abs(yy - center_y) <= half_width)


def _realistic_small_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 180, 260
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    rng = np.random.default_rng(20260523)
    grain = rng.normal(0.0, 2.2, size=(height, width, 1)).astype(np.float32)

    clean = np.stack(
        [
            82.0 + 40.0 * xx + 18.0 * yy,
            96.0 + 22.0 * xx + 28.0 * yy,
            118.0 + 16.0 * xx + 34.0 * yy,
        ],
        axis=2,
    )
    clean += 5.0 * np.sin(xx[:, :, None] * 18.0) + 3.0 * np.cos(yy[:, :, None] * 15.0)
    clean += grain
    clean[:, 126:131, :] += np.asarray([22.0, 18.0, 10.0], dtype=np.float32)
    clean = np.clip(clean, 0, 255).astype(np.uint8)

    target = np.zeros((height, width), dtype=bool)
    for cy, cx, radius in [(42, 54, 5), (88, 142, 7), (132, 204, 6)]:
        target |= _disk_mask((height, width), cy, cx, radius)
    target |= _short_scratch_mask((height, width), 124, 72, length=28, half_width=2)

    dusty = clean.copy()
    dusty[target] = np.asarray([14, 13, 12], dtype=np.uint8)

    red = np.zeros((height, width, 3), dtype=np.float32)
    red[:, :, 0] = 6.0 + 4.0 * yy + 2.0 * np.sin(xx * 10.0)
    red[:, :, 1] = 3.0 + 1.5 * yy
    red[:, :, 2] = 4.0 + 1.5 * xx
    red[:, width - 18 :, 0] += 88.0
    red[:, width - 18 :, 1] += 8.0
    red[:, width - 18 :, 2] += 4.0

    halo = np.zeros((height, width), dtype=bool)
    mid = np.zeros((height, width), dtype=bool)
    core = np.zeros((height, width), dtype=bool)
    for cy, cx, radius in [(42, 54, 5), (88, 142, 7), (132, 204, 6)]:
        halo |= _disk_mask((height, width), cy, cx, radius + 5)
        mid |= _disk_mask((height, width), cy, cx, radius + 2)
        core |= _disk_mask((height, width), cy, cx, radius)
    scratch = _short_scratch_mask((height, width), 124, 72, length=28, half_width=2)
    halo |= _short_scratch_mask((height, width), 124, 72, length=28, half_width=5)
    mid |= _short_scratch_mask((height, width), 124, 72, length=28, half_width=3)
    core |= scratch
    red[halo] = np.asarray([44.0, 4.0, 5.0], dtype=np.float32)
    red[mid] = np.asarray([122.0, 9.0, 11.0], dtype=np.float32)
    red[core] = np.asarray([238.0, 18.0, 24.0], dtype=np.float32)

    return dusty, np.clip(red, 0, 255).astype(np.uint8), target


def test_red_highlight_regression_repairs_realistic_small_scan_fixture() -> None:
    image, red_image, target_mask = _realistic_small_fixture()

    red_result = detect_red_highlight_source_image(
        red_image,
        RedHighlightConfig(
            detection_long_edge=260,
            local_radius=4,
            mask_edge_mode="normal",
            max_area=520,
            max_dim=44,
            max_aspect=12.0,
        ),
    )
    generated_mask = red_result.mask > 0

    assert red_result.manifest["component_count"] >= 4
    assert red_result.manifest["final_mask_pixels"] < 1700
    assert int(np.count_nonzero(generated_mask[:, -16:])) < 15
    assert _coverage(target_mask, generated_mask) > 0.55

    repair_result = repair_image(
        image,
        red_result.mask,
        RepairConfig(
            method="hybrid",
            mask_channel="grayscale",
            threshold=0.5,
            dilate_radius=1,
            feather_radius=1,
            strength=1.0,
            min_component_area=1,
            max_component_area=2000,
            padding=16,
        ),
    )

    covered_target = target_mask & (repair_result.soft_mask > 0.0)
    assert int(np.count_nonzero(covered_target)) > 120
    assert np.array_equal(repair_result.repaired_image[repair_result.soft_mask <= 0.0], image[repair_result.soft_mask <= 0.0])
    assert repair_result.metrics["max_abs_diff_outside_mask"] == 0.0
    assert _mean_luma(repair_result.repaired_image[covered_target]) > _mean_luma(image[covered_target]) + 25.0

    write_image(_artifact_path("normal_with_defects.png"), image)
    write_image(_artifact_path("red_inspection.png"), red_image)
    write_image(_artifact_path("generated_mask.png"), red_result.mask)
    write_image(_artifact_path("repaired.png"), repair_result.repaired_image)


def _coverage(target: np.ndarray, detected: np.ndarray) -> float:
    return float(np.count_nonzero(target & detected)) / float(max(1, np.count_nonzero(target)))


def _mean_luma(pixels: np.ndarray) -> float:
    work = pixels.astype(np.float32)
    return float(np.mean(work[:, 0] * 0.2126 + work[:, 1] * 0.7152 + work[:, 2] * 0.0722))
