import json
from pathlib import Path

import numpy as np

from dust_mask_repair import RepairConfig, repair_image
from dust_mask_repair.benchmark import evaluate_defect_aware_quality_case
from dust_mask_repair.local_repair import repair_small_local_roi


def _centered_diagonal_case(shape=(44, 56)):
    height, width = shape
    yy, xx = np.indices((height, width))
    cy = height // 2
    cx = width // 2
    clean = np.zeros((height, width, 3), dtype=np.uint8)
    edge = (yy - cy) > (xx - cx)
    clean[:] = [41, 79, 181]
    clean[edge] = [209, 161, 61]
    mask = np.zeros((height, width), dtype=np.uint8)
    for y, x in ((cy, cx + 1), (cy + 1, cx), (cy + 2, cx + 1), (cy + 3, cx + 2)):
        mask[y, x] = 255
    damaged = clean.copy()
    damaged[mask > 0] = [255, 255, 255]
    return clean, damaged, mask


def _mae(image, reference, mask):
    return float(np.mean(np.abs(image[mask].astype(np.int64) - reference[mask].astype(np.int64))))


def test_edge_guided_is_selected_for_diagonal_micro_defect_roi():
    clean, damaged, mask_u8 = _centered_diagonal_case((25, 25))
    mask = mask_u8 > 0

    repaired, stats = repair_small_local_roi(damaged, mask, mask, strategy="tiny_local", context_radius=5)

    assert stats["method"] == "edge_guided"
    assert stats["edge_guided_used"] is True
    assert stats["edge_guided_pixel_count"] == int(np.count_nonzero(mask))
    assert stats["edge_guided_coherence"] > 0.8
    assert _mae(repaired, clean, mask) < _mae(damaged, clean, mask)


def test_edge_guided_falls_back_on_low_gradient_energy():
    height, width = 24, 24
    yy, xx = np.indices((height, width), dtype=np.float32)
    clean = np.dstack((0.20 + xx * 0.002, 0.30 + yy * 0.002, 0.50 + xx * 0.001)).astype(np.float32)
    damaged = clean.copy()
    mask = np.zeros((height, width), dtype=bool)
    mask[12, 12] = True
    damaged[mask] = [1.0, 1.0, 1.0]

    repaired, stats = repair_small_local_roi(damaged, mask, mask, strategy="tiny_local", context_radius=5)

    assert stats["method"] == "plane"
    assert stats["edge_guided_used"] is False
    assert stats["edge_guided_low_confidence"] is True
    assert stats["edge_guided_fallback_reason"] == "low_gradient_energy"
    assert _mae(repaired, clean, mask) < _mae(damaged, clean, mask)


def test_edge_guided_does_not_use_masked_pixels_as_source():
    clean = np.zeros((21, 21, 3), dtype=np.uint8)
    yy, xx = np.indices(clean.shape[:2])
    clean[:] = [40, 80, 180]
    clean[yy > xx] = [210, 160, 60]
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=bool)
    mask[10, 9] = True
    damaged[mask] = [255, 255, 255]

    repaired, stats = repair_small_local_roi(damaged, mask, mask, strategy="tiny_local", context_radius=4)

    assert stats["edge_guided_used"] is True
    assert np.array_equal(repaired[10, 9], clean[10, 9])
    assert not np.array_equal(repaired[10, 9], damaged[10, 9])


def test_edge_guided_preserves_16bit_values_without_8bit_quantization():
    clean = np.zeros((21, 21, 3), dtype=np.uint16)
    yy, xx = np.indices(clean.shape[:2])
    clean[:] = [1234, 23456, 50123]
    clean[yy > xx] = [45678, 34567, 2345]
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=bool)
    mask[10, 9] = True
    damaged[mask] = [65535, 65535, 65535]

    repaired, stats = repair_small_local_roi(damaged, mask, mask, strategy="tiny_local", context_radius=4)

    assert repaired.dtype == np.uint16
    assert stats["edge_guided_used"] is True
    assert np.array_equal(repaired[10, 9], clean[10, 9])
    assert int(repaired[10, 9, 0]) % 257 != 0


def test_defect_aware_edge_guided_metrics_and_outside_invariance():
    clean, damaged, mask = _centered_diagonal_case()
    common = dict(
        method="defect_aware",
        mask_channel="grayscale",
        dilate_radius=0,
        feather_radius=0,
        padding=18,
        grain_reinject_strength=0.0,
    )

    debug_dir = Path("test_outputs") / "edge_guided_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    result = repair_image(damaged, mask, RepairConfig(**common, debug_dir=debug_dir))
    disabled = repair_image(damaged, mask, RepairConfig(**common, edge_guided_enabled=False))
    inside = mask > 0

    assert result.metrics["small_local_edge_guided_component_count"] == 1
    assert result.metrics["small_local_edge_guided_pixel_count"] == int(np.count_nonzero(inside))
    assert result.metrics["small_local_edge_guided_fallback_count"] == 0
    assert result.metrics["small_local_edge_guided_coherence_mean"] > 0.8
    assert _mae(result.repaired_image, clean, inside) < _mae(disabled.repaired_image, clean, inside)
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])

    payload = json.loads((debug_dir / "defect_components.json").read_text(encoding="utf-8"))
    assert payload["components"][0]["edge_guided_used"] is True
    assert payload["components"][0]["edge_guided_coherence"] > 0.8
    assert payload["components"][0]["edge_guided_gradient_energy"] > 0.0


def test_edge_guided_safe_on_border_and_large_component_cap():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:] = [80, 110, 140]
    damaged = image.copy()
    border_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    border_mask[0, 0] = 255
    damaged[0, 0] = [255, 255, 255]

    border_result = repair_image(
        damaged,
        border_mask,
        RepairConfig(method="defect_aware", mask_channel="grayscale", dilate_radius=0, feather_radius=0),
    )

    assert border_result.repaired_image.shape == image.shape
    assert border_result.metrics["max_abs_diff_outside_mask"] == 0.0

    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[8:10, 8:10] = True
    repaired, stats = repair_small_local_roi(
        damaged,
        mask,
        mask,
        strategy="small_local",
        edge_guided_max_component_area=3,
    )

    assert stats["edge_guided_used"] is False
    assert stats["edge_guided_fallback_reason"] == "component_area_cap"
    assert repaired.shape == damaged.shape


def test_edge_guided_quality_benchmark_cases():
    diagonal = evaluate_defect_aware_quality_case("diagonal_edge_micro_dust", width=56, height=44)
    chroma = evaluate_defect_aware_quality_case("chroma_edge_micro_dust", width=56, height=44)
    gradient = evaluate_defect_aware_quality_case("gradient_micro_dust", width=56, height=44)

    assert diagonal["mean_abs_error_inside_mask"] < diagonal["edge_guided_disabled_mean_abs_error_inside_mask"]
    assert diagonal["mean_abs_error_inside_mask"] < diagonal["corrupted_mean_abs_error_inside_mask"]
    assert diagonal["small_local_edge_guided_component_count"] == 1
    assert chroma["mean_abs_error_inside_mask"] <= chroma["edge_guided_disabled_mean_abs_error_inside_mask"]
    assert chroma["small_local_edge_guided_component_count"] == 1
    assert gradient["small_local_edge_guided_component_count"] == 0
    assert gradient["mean_abs_error_inside_mask"] < gradient["corrupted_mean_abs_error_inside_mask"]