import numpy as np

from dust_mask_repair import RepairConfig, repair_image
from dust_mask_repair.local_repair import repair_small_local_roi


def _config(**overrides):
    values = {
        "method": "defect_aware",
        "mask_channel": "grayscale",
        "threshold": 0.5,
        "dilate_radius": 0,
        "feather_radius": 0,
        "strength": 1.0,
        "padding": 8,
    }
    values.update(overrides)
    return RepairConfig(**values)


def _gradient_uint8(shape=(48, 64)):
    height, width = shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.uint8)
    clean[:, :, 0] = np.rint(35 + xx * 2.2).astype(np.uint8)
    clean[:, :, 1] = np.rint(55 + yy * 2.0).astype(np.uint8)
    clean[:, :, 2] = np.rint(180 - xx * 0.9 + yy * 0.3).astype(np.uint8)
    return clean


def test_defect_aware_tiny_local_repairs_constant_background():
    clean = np.zeros((32, 32, 3), dtype=np.uint8)
    clean[:, :] = [90, 120, 150]
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[16, 17] = 255
    damaged[16, 17] = [255, 255, 255]

    result = repair_image(damaged, mask, _config())

    assert result.metrics["small_local_component_count"] == 1
    assert result.metrics["small_local_pixel_count"] == 1
    assert result.metrics["small_local_plane_count"] + result.metrics["small_local_median_count"] == 1
    assert result.metrics["small_local_fallback_count"] == 0
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])
    assert np.mean(np.abs(result.repaired_image[mask > 0].astype(int) - clean[mask > 0].astype(int))) < 3.0


def test_small_local_plane_beats_median_on_smooth_gradient():
    height, width = 40, 54
    yy, xx = np.indices((height, width), dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.float32)
    clean[:, :, 0] = 0.15 + xx * 0.010
    clean[:, :, 1] = 0.20 + yy * 0.012
    clean[:, :, 2] = 0.75 - xx * 0.006 + yy * 0.003
    clean = np.clip(clean, 0.0, 1.0)
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=bool)
    mask[17:22, 23:28] = True
    damaged[mask] = [1.0, 1.0, 1.0]

    repaired, stats = repair_small_local_roi(damaged, mask, mask, strategy="small_local", context_radius=6)
    context = np.logical_and(~mask, np.ones(mask.shape, dtype=bool))
    median = np.median(damaged[context], axis=0)
    median_candidate = damaged.copy()
    median_candidate[mask] = median

    plane_error = np.mean(np.abs(repaired[mask] - clean[mask]))
    median_error = np.mean(np.abs(median_candidate[mask] - clean[mask]))
    damaged_error = np.mean(np.abs(damaged[mask] - clean[mask]))

    assert stats["method"] == "plane"
    assert plane_error < median_error
    assert plane_error < damaged_error


def test_defect_aware_small_local_preserves_uint16_and_outside_mask():
    clean8 = _gradient_uint8()
    clean = (clean8.astype(np.uint16) * 257).astype(np.uint16)
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[18:22, 25:29] = 255
    damaged[mask > 0] = 65535

    result = repair_image(damaged, mask, _config())

    assert result.repaired_image.dtype == np.uint16
    assert result.metrics["small_local_component_count"] == 1
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0


def test_small_local_roi_preserves_rgba_alpha():
    roi = np.zeros((24, 24, 4), dtype=np.uint8)
    roi[:, :, :3] = [80, 100, 120]
    roi[:, :, 3] = np.arange(24, dtype=np.uint8)[None, :]
    damaged = roi.copy()
    mask = np.zeros(roi.shape[:2], dtype=bool)
    mask[10:13, 10:13] = True
    damaged[mask, :3] = [255, 255, 255]

    repaired, stats = repair_small_local_roi(damaged, mask, mask, strategy="small_local")

    assert stats["fallback"] is False
    assert np.array_equal(repaired[:, :, 3], damaged[:, :, 3])
    assert np.array_equal(repaired[~mask], damaged[~mask])


def test_small_local_context_shortage_falls_back_without_exception():
    roi = np.zeros((8, 8, 3), dtype=np.float32)
    roi[:, :] = [0.2, 0.3, 0.4]
    mask = np.ones(roi.shape[:2], dtype=bool)

    repaired, stats = repair_small_local_roi(roi, mask, mask, strategy="small_local")

    assert stats["fallback"] is True
    assert np.array_equal(repaired, roi)
