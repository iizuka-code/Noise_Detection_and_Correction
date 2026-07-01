import numpy as np

from dust_mask_repair import RepairConfig, repair_image
from dust_mask_repair.benchmark import (
    DEFECT_AWARE_QUALITY_CASES,
    DEFECT_AWARE_QUALITY_VERSION,
    evaluate_defect_aware_quality_case,
    make_defect_aware_quality_case,
)


def test_defect_aware_quality_case_helper_shapes():
    clean, damaged, mask = make_defect_aware_quality_case("gradient_dust", width=48, height=36)

    assert clean.shape == (36, 48, 3)
    assert damaged.shape == clean.shape
    assert mask.shape == clean.shape[:2]
    assert clean.dtype == np.uint8
    assert damaged.dtype == np.uint8
    assert int(np.count_nonzero(mask)) > 0


def test_defect_aware_quality_evaluation_improves_gradient_case():
    result = evaluate_defect_aware_quality_case("gradient_dust", width=56, height=44)

    assert result["benchmark_version"] == DEFECT_AWARE_QUALITY_VERSION
    assert result["method"] == "defect_aware"
    assert result["processing_time_ms"] >= 0.0
    assert result["mask_pixel_count"] > 0
    assert result["component_count"] > 0
    assert result["max_abs_diff_outside_mask"] == 0.0
    assert result["mean_abs_error_inside_mask"] < result["corrupted_mean_abs_error_inside_mask"]
    assert sum(result["strategy_counts"].values()) == result["component_count"]


def test_defect_aware_repair_is_deterministic_on_quality_case():
    clean, damaged, mask = make_defect_aware_quality_case("grain_dust", width=56, height=44)
    config = RepairConfig(
        method="defect_aware",
        mask_channel="grayscale",
        dilate_radius=0,
        feather_radius=0,
        padding=18,
    )

    first = repair_image(damaged, mask, config)
    second = repair_image(damaged, mask, config)

    assert np.array_equal(first.repaired_image, second.repaired_image)
    assert np.array_equal(first.repaired_image[first.soft_mask <= 0.0], damaged[first.soft_mask <= 0.0])
    assert first.metrics["max_abs_diff_outside_mask"] == 0.0
    assert second.metrics["max_abs_diff_outside_mask"] == 0.0


def test_all_defect_aware_quality_cases_smoke():
    for case in DEFECT_AWARE_QUALITY_CASES:
        clean, damaged, mask = make_defect_aware_quality_case(case, width=48, height=40)
        result = repair_image(
            damaged,
            mask,
            RepairConfig(
                method="defect_aware",
                mask_channel="grayscale",
                dilate_radius=0,
                feather_radius=0,
                padding=18,
            ),
        )

        assert result.repaired_image.shape == clean.shape
        assert result.repaired_image.dtype == np.uint8
        assert result.metrics["max_abs_diff_outside_mask"] == 0.0
        assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])
