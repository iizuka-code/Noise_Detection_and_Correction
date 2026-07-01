import numpy as np

from dust_mask_repair import RepairConfig, repair_image


def _config(**overrides):
    values = {
        "method": "defect_aware",
        "mask_channel": "grayscale",
        "threshold": 0.5,
        "dilate_radius": 0,
        "feather_radius": 0,
        "strength": 1.0,
        "padding": 16,
        "max_component_area": 5000,
    }
    values.update(overrides)
    return RepairConfig(**values)


def test_directional_repairs_vertical_line_across_horizontal_gradient():
    height, width = 48, 72
    xx = np.linspace(30, 220, width, dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.uint8)
    clean[:, :, 0] = np.rint(xx).astype(np.uint8)
    clean[:, :, 1] = np.rint(xx * 0.6 + 40).astype(np.uint8)
    clean[:, :, 2] = np.rint(230 - xx * 0.5).astype(np.uint8)
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[10:38, 34] = 255
    damaged[mask > 0] = [255, 255, 255]

    result = repair_image(damaged, mask, _config())

    inside = mask > 0
    mean_error = np.mean(np.abs(result.repaired_image[inside].astype(int) - clean[inside].astype(int)))
    assert result.metrics["defect_strategy_counts"]["directional"] == 1
    assert result.metrics["directional_component_count"] == 1
    assert result.metrics["directional_cap_exceeded_count"] == 0
    assert mean_error < 4.0
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])


def test_directional_repairs_horizontal_line_across_vertical_gradient():
    height, width = 52, 70
    yy = np.linspace(35, 210, height, dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.uint16)
    clean[:, :, 0] = np.rint(yy * 257).astype(np.uint16)[:, None]
    clean[:, :, 1] = np.rint((yy * 0.7 + 40) * 257).astype(np.uint16)[:, None]
    clean[:, :, 2] = np.rint((230 - yy * 0.5) * 257).astype(np.uint16)[:, None]
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[26, 14:58] = 255
    damaged[mask > 0] = 65535

    result = repair_image(damaged, mask, _config())

    inside = mask > 0
    mean_error = np.mean(np.abs(result.repaired_image[inside].astype(np.int64) - clean[inside].astype(np.int64)))
    assert result.repaired_image.dtype == np.uint16
    assert result.metrics["directional_component_count"] == 1
    assert mean_error < 900.0
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0


def test_directional_handles_diagonal_line_without_exception():
    clean = np.zeros((44, 44, 3), dtype=np.uint8)
    clean[:, :] = [90, 115, 140]
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    for offset in range(8, 34):
        mask[offset, offset] = 255
    damaged[mask > 0] = [255, 255, 255]

    result = repair_image(damaged, mask, _config())

    assert result.metrics["directional_component_count"] == 1
    assert result.repaired_image.dtype == np.uint8
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])


def test_directional_cap_exceeded_falls_back():
    clean = np.zeros((24, 640, 3), dtype=np.uint8)
    clean[:, :] = [80, 100, 120]
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[12, 20:560] = 255
    damaged[mask > 0] = [255, 255, 255]

    result = repair_image(damaged, mask, _config(padding=20, max_component_area=2000))

    assert result.metrics["defect_strategy_counts"]["directional"] == 1
    assert result.metrics["directional_component_count"] == 1
    assert result.metrics["directional_cap_exceeded_count"] == 1
    assert result.metrics["directional_fallback_count"] == 1
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])
