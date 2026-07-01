import numpy as np

from dust_mask_repair import RepairConfig, repair_image
from dust_mask_repair.patch_repair import repair_patch_match_roi


def _config(**overrides):
    values = {
        "method": "defect_aware",
        "mask_channel": "grayscale",
        "threshold": 0.5,
        "dilate_radius": 0,
        "feather_radius": 0,
        "strength": 1.0,
        "padding": 18,
    }
    values.update(overrides)
    return RepairConfig(**values)


def _stripe_image(shape=(64, 80), dtype=np.uint8):
    height, width = shape
    max_value = np.iinfo(dtype).max
    image = np.zeros((height, width, 3), dtype=dtype)
    stripe = ((np.arange(width) // 4) % 2).astype(bool)
    image[:, :, 0] = np.where(stripe[None, :], int(max_value * 0.75), int(max_value * 0.18)).astype(dtype)
    image[:, :, 1] = np.where(stripe[None, :], int(max_value * 0.55), int(max_value * 0.30)).astype(dtype)
    image[:, :, 2] = np.where(stripe[None, :], int(max_value * 0.25), int(max_value * 0.62)).astype(dtype)
    return image


def test_defect_aware_patch_repairs_stripe_texture_and_records_metrics():
    clean = _stripe_image()
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[24:34, 31:41] = 255
    damaged[mask > 0] = [255, 255, 255]

    result = repair_image(damaged, mask, _config())

    inside = mask > 0
    repaired_error = np.mean(np.abs(result.repaired_image[inside].astype(int) - clean[inside].astype(int)))
    damaged_error = np.mean(np.abs(damaged[inside].astype(int) - clean[inside].astype(int)))
    assert result.metrics["defect_strategy_counts"]["patch"] == 1
    assert result.metrics["patch_component_count"] == 1
    assert result.metrics["patch_candidate_count_total"] > 0
    assert result.metrics["patch_fallback_count"] == 0
    assert result.metrics["patch_best_score_mean"] >= 0.0
    assert repaired_error < damaged_error
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])


def test_patch_candidate_overlapping_repair_mask_is_excluded():
    roi = _stripe_image(shape=(40, 50)).astype(np.float32) / 255.0
    component = np.zeros(roi.shape[:2], dtype=bool)
    component[15:21, 20:26] = True
    repair_mask = component.copy()
    repair_mask[15:21, 8:14] = True
    damaged = roi.copy()
    damaged[component] = [1.0, 1.0, 1.0]

    repaired, stats = repair_patch_match_roi(damaged, component, repair_mask, search_radius=24, patch_margin=3)

    assert stats["fallback"] is False
    assert stats["candidate_count"] > 0
    assert np.array_equal(repaired[repair_mask & ~component], damaged[repair_mask & ~component])


def test_patch_no_candidate_falls_back_without_exception():
    roi = np.zeros((16, 16, 3), dtype=np.float32)
    component = np.zeros(roi.shape[:2], dtype=bool)
    component[4:12, 4:12] = True
    repair_mask = np.ones(roi.shape[:2], dtype=bool)

    repaired, stats = repair_patch_match_roi(roi, component, repair_mask, search_radius=8)

    assert stats["fallback"] is True
    assert np.array_equal(repaired, roi)


def test_patch_candidate_cap_is_deterministic():
    roi = _stripe_image(shape=(50, 60)).astype(np.float32) / 255.0
    component = np.zeros(roi.shape[:2], dtype=bool)
    component[20:26, 25:31] = True
    damaged = roi.copy()
    damaged[component] = [1.0, 1.0, 1.0]

    first, first_stats = repair_patch_match_roi(damaged, component, component, max_candidates=3, search_radius=30)
    second, second_stats = repair_patch_match_roi(damaged, component, component, max_candidates=3, search_radius=30)

    assert first_stats["candidate_count"] == 3
    assert first_stats["cap_exceeded"] is True
    assert first_stats == second_stats
    assert np.array_equal(first, second)


def test_defect_aware_patch_preserves_uint16_and_outside_mask():
    clean = _stripe_image(dtype=np.uint16)
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[22:32, 28:40] = 255
    damaged[mask > 0] = 65535

    result = repair_image(damaged, mask, _config())

    assert result.repaired_image.dtype == np.uint16
    assert result.metrics["patch_component_count"] == 1
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])
