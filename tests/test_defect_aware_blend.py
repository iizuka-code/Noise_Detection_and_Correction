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
        "padding": 10,
        "grain_reinject_strength": 0.0,
    }
    values.update(overrides)
    return RepairConfig(**values)


def test_defect_aware_blend_feather_zero_preserves_outside_exactly():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :] = [80, 110, 140]
    damaged = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[12:16, 13:17] = 255
    damaged[mask > 0] = [255, 255, 255]

    result = repair_image(damaged, mask, _config())

    assert result.metrics["defect_aware_alpha_nonzero_pixel_count"] == int(np.count_nonzero(mask))
    assert result.metrics["defect_aware_blend_shell_pixel_count"] == 0
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])


def test_defect_aware_blend_shell_is_weaker_than_core():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, :] = [80, 110, 140]
    damaged = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[16:22, 17:23] = 255
    damaged[mask > 0] = [255, 255, 255]

    result = repair_image(damaged, mask, _config(dilate_radius=3, feather_radius=3))

    core = result.core_mask
    shell = result.repair_mask & ~result.core_mask
    diff = np.max(np.abs(result.repaired_image.astype(int) - damaged.astype(int)), axis=2)
    assert result.metrics["defect_aware_blend_shell_pixel_count"] == int(np.count_nonzero(shell & (result.blend_alpha > 0)))
    assert np.array_equal(result.repaired_image[result.blend_alpha <= 0.0], damaged[result.blend_alpha <= 0.0])
    assert float(np.mean(diff[shell])) <= float(np.mean(diff[core]))


def test_defect_aware_strength_zero_returns_exact_and_zero_alpha_metric():
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    image[:, :] = [90, 120, 150]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[10:15, 10:15] = 255

    result = repair_image(image, mask, _config(strength=0.0))

    assert np.array_equal(result.repaired_image, image)
    assert result.metrics["defect_aware_alpha_nonzero_pixel_count"] == 0


def test_defect_aware_strength_half_is_not_full_replacement():
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    image[:, :] = [90, 120, 150]
    damaged = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[12:15, 12:15] = 255
    damaged[mask > 0] = [255, 255, 255]

    half = repair_image(damaged, mask, _config(strength=0.5))
    full = repair_image(damaged, mask, _config(strength=1.0))

    inside = mask > 0
    half_diff = np.mean(np.abs(half.repaired_image[inside].astype(int) - damaged[inside].astype(int)))
    full_diff = np.mean(np.abs(full.repaired_image[inside].astype(int) - damaged[inside].astype(int)))
    assert 0.0 < half_diff < full_diff


def test_defect_aware_blend_preserves_uint16_dtype():
    image = np.zeros((32, 32, 3), dtype=np.uint16)
    image[:, :] = [12000, 24000, 36000]
    damaged = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[12:18, 12:18] = 255
    damaged[mask > 0] = 65535

    result = repair_image(damaged, mask, _config(dilate_radius=2, feather_radius=2))

    assert result.repaired_image.dtype == np.uint16
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0
    assert np.array_equal(result.repaired_image[result.blend_alpha <= 0.0], damaged[result.blend_alpha <= 0.0])
