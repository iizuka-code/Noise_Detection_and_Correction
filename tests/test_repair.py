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


def test_linear_method_repairs_gradient_mask_without_touching_outside():
    height, width = 48, 72
    yy, xx = np.indices((height, width), dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.uint8)
    clean[:, :, 0] = np.rint(40 + xx * 1.8).astype(np.uint8)
    clean[:, :, 1] = np.rint(60 + yy * 2.2).astype(np.uint8)
    clean[:, :, 2] = np.rint(180 - xx * 0.7 + yy * 0.25).astype(np.uint8)
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[18:30, 28:44] = 255
    damaged[mask > 0] = [255, 255, 255]

    result = repair_image(damaged, mask, _strict_config(method="linear", padding=18))

    inside = mask > 0
    mean_error = np.mean(np.abs(result.repaired_image[inside].astype(int) - clean[inside].astype(int)))
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0
    assert mean_error < 10.0


def test_kl_method_uses_context_distribution_and_preserves_outside():
    image = np.zeros((42, 42, 3), dtype=np.uint8)
    image[:, :] = [210, 35, 35]
    image[12:31, 13] = [35, 35, 220]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[14:28, 14:28] = 255
    damaged = image.copy()
    damaged[mask > 0] = [255, 255, 255]

    linear = repair_image(damaged, mask, _strict_config(method="linear", padding=18))
    kl = repair_image(damaged, mask, _strict_config(method="kl", padding=18))

    inside = mask > 0
    changed_pixels = np.count_nonzero(np.any(linear.repaired_image[inside] != kl.repaired_image[inside], axis=1))
    assert changed_pixels > 0
    assert np.array_equal(kl.repaired_image[kl.soft_mask <= 0.0], damaged[kl.soft_mask <= 0.0])
    assert kl.metrics["max_abs_diff_outside_mask"] == 0.0


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


def test_adaptive_repairs_gradient_dust_better_than_median_fill():
    height, width = 56, 80
    yy, xx = np.indices((height, width), dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.uint8)
    clean[:, :, 0] = np.rint(45 + xx * 1.9 + yy * 0.35).astype(np.uint8)
    clean[:, :, 1] = np.rint(70 + xx * 0.8 + yy * 1.1).astype(np.uint8)
    clean[:, :, 2] = np.rint(160 - xx * 0.55 + yy * 0.25).astype(np.uint8)
    dusty = clean.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[20:39, 30:49] = 255
    dusty[mask > 0] = [245, 245, 245]

    adaptive = repair_image(dusty, mask, _strict_config(method="adaptive", padding=14))
    median = repair_image(dusty, mask, _strict_config(method="median", padding=14))

    inside = mask > 0
    adaptive_error = np.mean(np.abs(adaptive.repaired_image[inside].astype(int) - clean[inside].astype(int)))
    median_error = np.mean(np.abs(median.repaired_image[inside].astype(int) - clean[inside].astype(int)))
    assert adaptive_error < median_error * 0.7
    assert adaptive.metrics["max_abs_diff_outside_mask"] == 0.0


def test_adaptive_strongly_replaces_bright_dust_core_on_dark_surface():
    height, width = 48, 64
    yy, xx = np.indices((height, width), dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.uint8)
    clean[:, :, 0] = np.rint(20 + xx * 0.26 + yy * 0.10).astype(np.uint8)
    clean[:, :, 1] = np.rint(24 + xx * 0.22 + yy * 0.08).astype(np.uint8)
    clean[:, :, 2] = np.rint(32 + xx * 0.20 + yy * 0.06).astype(np.uint8)
    dusty = clean.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[21:27, 29:35] = 255
    dusty[mask > 0] = [232, 226, 238]

    result = repair_image(
        dusty,
        mask,
        _strict_config(method="adaptive", dilate_radius=2, feather_radius=1, padding=12),
    )

    inside = mask > 0
    before_error = np.mean(np.abs(dusty[inside].astype(int) - clean[inside].astype(int)))
    after_error = np.mean(np.abs(result.repaired_image[inside].astype(int) - clean[inside].astype(int)))
    mean_core_change = np.mean(np.abs(result.repaired_image[inside].astype(int) - dusty[inside].astype(int)))
    assert after_error < before_error * 0.25
    assert mean_core_change > 120.0
    assert result.metrics["max_abs_diff_outside_repair_mask"] == 0.0


def test_dilated_shell_is_not_fully_replaced_for_overwide_mask():
    image = _clean_image(value=(92, 110, 128), shape=(42, 42))
    image[20, 20] = [255, 255, 255]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[20, 20] = 255

    result = repair_image(
        image,
        mask,
        _strict_config(method="adaptive", dilate_radius=4, feather_radius=4, padding=10),
    )

    assert result.core_mask is not None
    assert result.repair_mask is not None
    assert result.blend_alpha is not None
    shell = result.repair_mask & ~result.core_mask
    assert shell.any()
    assert float(np.max(result.blend_alpha[shell])) <= 0.45
    assert np.array_equal(result.repaired_image[result.blend_alpha <= 0.0], image[result.blend_alpha <= 0.0])
    assert result.metrics["max_abs_diff_outside_repair_mask"] == 0.0


def test_adaptive_repairs_diagonal_thin_scratch_with_pca_directional_fill():
    height, width = 64, 64
    yy, xx = np.indices((height, width), dtype=np.float32)
    clean = np.zeros((height, width, 3), dtype=np.uint8)
    clean[:, :, 0] = np.rint(55 + xx * 1.6).astype(np.uint8)
    clean[:, :, 1] = np.rint(80 + yy * 1.4).astype(np.uint8)
    clean[:, :, 2] = np.rint(180 - xx * 0.7 + yy * 0.25).astype(np.uint8)
    scratched = clean.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    for offset in range(-1, 2):
        for x in range(10, 54):
            y = x + offset
            mask[y, x] = 255
    scratched[mask > 0] = [250, 250, 250]

    result = repair_image(scratched, mask, _strict_config(method="adaptive", padding=16))

    inside = mask > 0
    mean_error = np.mean(np.abs(result.repaired_image[inside].astype(int) - clean[inside].astype(int)))
    assert mean_error < 12.0
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0


def test_adaptive_preserves_rgba_alpha_channel_exactly():
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[:, :, :3] = [86, 108, 130]
    image[:, :, 3] = np.arange(32, dtype=np.uint8)[None, :]
    image[14:18, 14:18, :3] = [255, 255, 255]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[14:18, 14:18] = 255

    result = repair_image(image, mask, _strict_config(method="adaptive", padding=8))

    assert result.repaired_image.dtype == image.dtype
    assert np.array_equal(result.repaired_image[:, :, 3], image[:, :, 3])
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0



def test_defect_aware_expanded_mask_repairs_spillover_and_records_color_match():
    clean = _clean_image(shape=(32, 32))
    damaged = clean.copy()
    damaged[16, 16] = [255, 255, 255]
    damaged[16, 17] = [255, 255, 255]
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[16, 16] = 255

    result = repair_image(
        damaged,
        mask,
        _strict_config(
            method="defect_aware",
            dilate_radius=1,
            feather_radius=0,
            padding=8,
            color_match_strength=0.8,
            grain_reinject_strength=0.0,
        ),
    )

    assert result.metrics["repair_mask_pixel_count"] > result.metrics["core_mask_pixel_count"]
    assert result.metrics["changed_pixel_count_shell"] >= 1
    assert result.metrics["color_match_component_count"] >= 1
    assert result.metrics["color_match_pixel_count"] >= result.metrics["repair_mask_pixel_count"]
    assert not np.array_equal(result.repaired_image[16, 17], damaged[16, 17])
    assert result.metrics["max_abs_diff_outside_repair_mask"] == 0.0
def test_debug_outputs_include_separated_masks_and_guard_artifacts():
    image = _clean_image(shape=(24, 24))
    image[10:12, 10:12] = [255, 255, 255]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[10:12, 10:12] = 255

    result = repair_image(image, mask, _strict_config(method="adaptive", collect_debug_images=True))

    assert {
        "core_mask",
        "repair_mask",
        "blend_alpha",
        "candidate_before_guard",
        "candidate_after_guard",
        "rejected_by_guard",
        "shell_mask",
    }.issubset(result.debug_images)
    assert {
        "changed_pixel_count_core",
        "changed_pixel_count_shell",
        "mean_abs_diff_core",
        "mean_abs_diff_shell",
        "max_abs_diff_outside_original_mask",
        "max_abs_diff_outside_repair_mask",
        "guard_rejected_pixel_count",
        "average_component_alpha",
        "low_confidence_component_count",
    }.issubset(result.metrics)
