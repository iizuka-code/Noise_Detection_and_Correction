import numpy as np

from dust_mask_repair import RepairConfig, repair_image


def _clean_image(value=(80, 110, 140), shape=(24, 24)):
    image = np.zeros((*shape, 3), dtype=np.uint8)
    image[:, :] = np.array(value, dtype=np.uint8)
    return image


def _strict_config(**overrides):
    values = {
        "method": "hybrid",
        "mask_channel": "grayscale",
        "threshold": 0.5,
        "dilate_radius": 0,
        "feather_radius": 0,
        "strength": 1.0,
        "padding": 4,
    }
    values.update(overrides)
    return RepairConfig(**values)


def test_empty_mask_returns_exact_input():
    image = _clean_image()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    result = repair_image(image, mask, _strict_config())

    assert np.array_equal(result.repaired_image, image)
    assert result.metrics["changed_pixel_count"] == 0


def test_strength_zero_returns_exact_input():
    image = _clean_image()
    image[10, 10] = [255, 255, 255]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[10, 10] = 255

    result = repair_image(image, mask, _strict_config(strength=0.0))

    assert np.array_equal(result.repaired_image, image)
    assert result.metrics["changed_pixel_count"] == 0


def test_wide_scratch_empty_mask_returns_exact_input():
    image = _clean_image(shape=(32, 32))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    result = repair_image(image, mask, _strict_config(method="wide_scratch"))

    assert np.array_equal(result.repaired_image, image)
    assert result.metrics["changed_pixel_count"] == 0


def test_wide_scratch_strength_zero_returns_exact_input():
    image = _clean_image(shape=(32, 32))
    image[12:20, 14:18] = [0, 0, 0]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[12:20, 14:18] = 255

    result = repair_image(image, mask, _strict_config(method="wide_scratch", strength=0.0))

    assert np.array_equal(result.repaired_image, image)
    assert result.metrics["changed_pixel_count"] == 0


def test_white_dust_is_repaired_only_in_mask():
    clean = _clean_image()
    dusty = clean.copy()
    dusty[10:12, 10:12] = [255, 255, 255]
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[10:12, 10:12] = 255

    result = repair_image(dusty, mask, _strict_config())

    outside = result.soft_mask <= 0.0
    assert np.array_equal(result.repaired_image[outside], dusty[outside])
    assert np.mean(np.abs(result.repaired_image[mask > 0].astype(int) - clean[mask > 0].astype(int))) < 3
    assert not np.array_equal(result.repaired_image[mask > 0], dusty[mask > 0])


def test_black_dust_is_repaired_only_in_mask():
    clean = _clean_image(value=(170, 120, 90))
    dusty = clean.copy()
    dusty[8:10, 13:15] = [0, 0, 0]
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[8:10, 13:15] = 255

    result = repair_image(dusty, mask, _strict_config(method="inpaint"))

    outside = result.soft_mask <= 0.0
    assert np.array_equal(result.repaired_image[outside], dusty[outside])
    assert np.mean(np.abs(result.repaired_image[mask > 0].astype(int) - clean[mask > 0].astype(int))) < 3
    assert not np.array_equal(result.repaired_image[mask > 0], dusty[mask > 0])


def test_aggressive_method_preserves_pixels_outside_mask():
    image = _clean_image(value=(235, 235, 235), shape=(32, 32))
    image[12:18, 12:18] = [0, 0, 0]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[12:18, 12:18] = 255

    result = repair_image(image, mask, _strict_config(method="aggressive", padding=8))

    outside = result.soft_mask <= 0.0
    assert np.array_equal(result.repaired_image[outside], image[outside])
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0
    assert not np.array_equal(result.repaired_image[mask > 0], image[mask > 0])


def test_aggressive_guard_does_not_create_dark_stain_on_clean_bright_area():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[10:30, 10:30] = [245, 245, 245]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[15:25, 15:25] = 255

    result = repair_image(image, mask, _strict_config(method="aggressive", padding=8))

    center = result.repaired_image[18:22, 18:22]
    assert np.min(center[:, :, 0]) >= 240
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], image[result.soft_mask <= 0.0])


def test_wide_scratch_repairs_broad_vertical_defect_with_gradient_context():
    height, width = 48, 72
    x = np.linspace(40, 220, width, dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.uint8)
    clean[:, :, 0] = np.rint(x).astype(np.uint8)
    clean[:, :, 1] = np.rint(x * 0.6 + 30).astype(np.uint8)
    clean[:, :, 2] = np.rint(210 - x * 0.45).astype(np.uint8)
    scratched = clean.copy()
    scratched[8:40, 31:40] = [0, 0, 0]
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[8:40, 31:40] = 255

    result = repair_image(scratched, mask, _strict_config(method="wide_scratch", padding=14))

    outside = result.soft_mask <= 0.0
    inside = mask > 0
    mean_error = np.mean(np.abs(result.repaired_image[inside].astype(int) - clean[inside].astype(int)))
    assert np.array_equal(result.repaired_image[outside], scratched[outside])
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0
    assert mean_error < 4.0


def test_wide_scratch_preserves_uint16_for_horizontal_defect():
    height, width = 40, 64
    y = np.linspace(12000, 52000, height, dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.uint16)
    clean[:, :, 0] = np.rint(y).astype(np.uint16)[:, None]
    clean[:, :, 1] = np.rint(y * 0.8).astype(np.uint16)[:, None]
    clean[:, :, 2] = np.rint(60000 - y * 0.55).astype(np.uint16)[:, None]
    scratched = clean.copy()
    scratched[17:24, 10:54] = 65535
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[17:24, 10:54] = 255

    result = repair_image(scratched, mask, _strict_config(method="wide_scratch", padding=12))

    inside = mask > 0
    mean_error = np.mean(np.abs(result.repaired_image[inside].astype(np.int64) - clean[inside].astype(np.int64)))
    assert result.repaired_image.dtype == np.uint16
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], scratched[result.soft_mask <= 0.0])
    assert mean_error < 900.0


def test_debug_images_are_opt_in_without_debug_dir():
    image = _clean_image(shape=(24, 24))
    image[10:12, 10:12] = [0, 0, 0]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[10:12, 10:12] = 255

    fast_result = repair_image(image, mask, _strict_config())
    debug_result = repair_image(image, mask, _strict_config(collect_debug_images=True))

    assert fast_result.debug_images == {}
    assert {"normalized_mask", "binary_mask", "soft_mask", "repaired_preview", "diff_visualization"}.issubset(
        debug_result.debug_images
    )
