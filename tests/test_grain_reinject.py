import numpy as np
import pytest

from dust_mask_repair import RepairConfig, repair_image
from dust_mask_repair.grain import reinject_grain_roi


def _grainy_roi(shape=(40, 48), channels=3):
    yy, xx = np.indices(shape, dtype=np.float32)
    base = np.zeros((*shape, channels), dtype=np.float32)
    texture = (((xx * 17 + yy * 29) % 11) - 5.0) / 255.0
    base[:, :, 0] = 0.35 + texture
    base[:, :, 1] = 0.42 - texture * 0.8
    base[:, :, 2] = 0.50 + texture * 0.5
    if channels == 4:
        base[:, :, 3] = 0.75
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def test_grain_strength_zero_keeps_candidate_exact():
    original = _grainy_roi()
    candidate = original.copy()
    mask = np.zeros(original.shape[:2], dtype=bool)
    mask[14:24, 16:26] = True
    candidate[mask, :3] = [0.4, 0.4, 0.4]

    grained, stats = reinject_grain_roi(original, candidate, mask, mask, strength=0.0)

    assert stats["applied"] is False
    assert np.array_equal(grained, candidate)


def test_grain_increases_repaired_region_std_and_is_deterministic():
    original = _grainy_roi()
    candidate = original.copy()
    mask = np.zeros(original.shape[:2], dtype=bool)
    mask[14:25, 17:29] = True
    candidate[mask, :3] = [0.42, 0.42, 0.42]

    first, first_stats = reinject_grain_roi(original, candidate, mask, mask, label=7, strength=0.6)
    second, second_stats = reinject_grain_roi(original, candidate, mask, mask, label=7, strength=0.6)

    assert first_stats["applied"] is True
    assert first_stats == second_stats
    assert np.array_equal(first, second)
    assert float(np.std(first[mask, :3])) > float(np.std(candidate[mask, :3]))
    assert np.array_equal(first[~mask], candidate[~mask])


def test_grain_preserves_rgba_alpha():
    original = _grainy_roi(channels=4)
    candidate = original.copy()
    candidate[:, :, 3] = np.linspace(0.2, 0.9, candidate.shape[1], dtype=np.float32)[None, :]
    mask = np.zeros(original.shape[:2], dtype=bool)
    mask[10:20, 12:23] = True
    candidate[mask, :3] = [0.45, 0.45, 0.45]

    grained, stats = reinject_grain_roi(original, candidate, mask, mask, strength=0.5)

    assert stats["applied"] is True
    assert np.array_equal(grained[:, :, 3], candidate[:, :, 3])


def test_grain_context_shortage_skips_without_exception():
    original = _grainy_roi(shape=(10, 10))
    candidate = original.copy()
    mask = np.ones(original.shape[:2], dtype=bool)

    grained, stats = reinject_grain_roi(original, candidate, mask, mask, strength=0.5)

    assert stats["applied"] is False
    assert stats["skipped_no_context"] is True
    assert np.array_equal(grained, candidate)


@pytest.mark.parametrize("dtype,max_value", [(np.uint8, 255), (np.uint16, 65535)])
def test_defect_aware_grain_metrics_and_outside_invariance(dtype, max_value):
    clean_float = _grainy_roi(shape=(44, 52))
    clean = np.rint(clean_float * max_value).astype(dtype)
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[16:26, 20:30] = 255
    damaged[mask > 0] = max_value

    result = repair_image(
        damaged,
        mask,
        RepairConfig(
            method="defect_aware",
            mask_channel="grayscale",
            dilate_radius=0,
            feather_radius=0,
            padding=12,
            grain_reinject_strength=0.4,
        ),
    )

    assert result.repaired_image.dtype == dtype
    assert result.metrics["grain_reinject_enabled"] is True
    assert result.metrics["grain_reinject_strength"] == 0.4
    assert result.metrics["grain_reinject_component_count"] >= 1
    assert result.metrics["grain_reinject_pixel_count"] > 0
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])
