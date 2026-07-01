import numpy as np
import pytest

from dust_mask_repair import RepairConfig, repair_image
from dust_mask_repair.local_repair import repair_fast_inpaint_roi


def _config(**overrides):
    values = {
        "method": "defect_aware",
        "mask_channel": "grayscale",
        "threshold": 0.5,
        "dilate_radius": 0,
        "feather_radius": 0,
        "strength": 1.0,
        "padding": 10,
    }
    values.update(overrides)
    return RepairConfig(**values)


def test_fast_inpaint_repairs_constant_medium_mask_and_records_metrics():
    clean = np.zeros((44, 52, 3), dtype=np.uint8)
    clean[:, :] = [80, 110, 140]
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[16:26, 20:30] = 255
    damaged[mask > 0] = [255, 255, 255]

    result = repair_image(damaged, mask, _config())

    assert result.metrics["defect_strategy_counts"]["fast_inpaint"] == 1
    assert result.metrics["fast_inpaint_component_count"] == 1
    assert result.metrics["fast_inpaint_pixel_count"] == 100
    assert result.metrics["fast_inpaint_iterations_total"] > 0
    assert result.metrics["fast_inpaint_fallback_count"] == 0
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])
    assert np.mean(np.abs(result.repaired_image[mask > 0].astype(int) - clean[mask > 0].astype(int))) < 3.0


def test_fast_inpaint_reduces_gradient_error():
    height, width = 48, 64
    yy, xx = np.indices((height, width), dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.float32)
    clean[:, :, 0] = 0.10 + xx * 0.010
    clean[:, :, 1] = 0.18 + yy * 0.011
    clean[:, :, 2] = 0.78 - xx * 0.005 + yy * 0.002
    clean = np.clip(clean, 0.0, 1.0)
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=bool)
    mask[18:30, 25:38] = True
    damaged[mask] = [1.0, 1.0, 1.0]

    repaired, stats = repair_fast_inpaint_roi(damaged, mask, max_iterations=40)

    repaired_error = np.mean(np.abs(repaired[mask] - clean[mask]))
    damaged_error = np.mean(np.abs(damaged[mask] - clean[mask]))
    assert stats["iterations"] > 0
    assert repaired_error < damaged_error


@pytest.mark.parametrize("dtype,max_value", [(np.uint8, 255), (np.uint16, 65535)])
def test_defect_aware_fast_inpaint_preserves_rgb_dtype_and_outside(dtype, max_value):
    clean = np.zeros((42, 50, 3), dtype=dtype)
    clean[:, :] = np.array([max_value // 5, max_value // 3, max_value // 2], dtype=dtype)
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[15:25, 18:29] = 255
    damaged[mask > 0] = max_value

    result = repair_image(damaged, mask, _config())

    assert result.repaired_image.dtype == dtype
    assert result.metrics["fast_inpaint_component_count"] == 1
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0


def test_fast_inpaint_roi_preserves_rgba_alpha():
    roi = np.zeros((30, 30, 4), dtype=np.uint8)
    roi[:, :, :3] = [70, 95, 130]
    roi[:, :, 3] = np.arange(30, dtype=np.uint8)[None, :]
    damaged = roi.copy()
    mask = np.zeros(roi.shape[:2], dtype=bool)
    mask[10:20, 11:21] = True
    damaged[mask, :3] = [255, 255, 255]

    repaired, stats = repair_fast_inpaint_roi(damaged, mask, max_iterations=16)

    assert stats["fallback"] is False
    assert np.array_equal(repaired[:, :, 3], damaged[:, :, 3])
    assert np.array_equal(repaired[~mask], damaged[~mask])


def test_fast_inpaint_iteration_cap_and_no_context_fallback():
    roi = np.zeros((12, 12, 3), dtype=np.float32)
    mask = np.ones(roi.shape[:2], dtype=bool)

    repaired, stats = repair_fast_inpaint_roi(roi, mask, max_iterations=5)

    assert stats["fallback"] is True
    assert stats["iterations"] == 0
    assert np.array_equal(repaired, roi)

    mask = np.zeros(roi.shape[:2], dtype=bool)
    mask[3:9, 3:9] = True
    repaired, stats = repair_fast_inpaint_roi(roi, mask, max_iterations=5)

    assert stats["fallback"] is False
    assert stats["iterations"] == 5
    assert np.array_equal(repaired[~mask], roi[~mask])
