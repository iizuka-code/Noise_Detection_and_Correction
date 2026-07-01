import numpy as np
import pytest

from dust_mask_repair import RepairConfig, repair_image
from dust_mask_repair.cli import build_parser
from dust_mask_repair.config import REPAIR_METHODS


def _config(**overrides):
    values = {
        "method": "defect_aware",
        "mask_channel": "grayscale",
        "threshold": 0.5,
        "dilate_radius": 0,
        "feather_radius": 0,
        "strength": 1.0,
        "padding": 6,
    }
    values.update(overrides)
    return RepairConfig(**values)


def _gradient_image(dtype, shape=(28, 32)):
    max_value = np.iinfo(dtype).max
    height, width = shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    image = np.zeros((height, width, 3), dtype=dtype)
    image[:, :, 0] = np.rint(max_value * (0.15 + 0.55 * xx / max(width - 1, 1))).astype(dtype)
    image[:, :, 1] = np.rint(max_value * (0.20 + 0.45 * yy / max(height - 1, 1))).astype(dtype)
    image[:, :, 2] = np.rint(max_value * (0.75 - 0.30 * xx / max(width - 1, 1))).astype(dtype)
    return image


def test_defect_aware_method_is_accepted_and_records_baseline_metrics():
    image = _gradient_image(np.uint8)
    damaged = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[11:14, 13:16] = 255
    damaged[mask > 0] = [255, 255, 255]

    result = repair_image(damaged, mask, _config())

    assert result.metrics["defect_aware"] is True
    assert result.metrics["defect_aware_version"] == 1
    assert result.metrics["defect_aware_fallback_method"] == "adaptive"
    assert result.metrics["defect_classification_enabled"] is True
    assert sum(result.metrics["defect_strategy_counts"].values()) == 1
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0


def test_defect_aware_empty_mask_returns_exact_input():
    image = _gradient_image(np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    result = repair_image(image, mask, _config())

    assert np.array_equal(result.repaired_image, image)
    assert result.metrics["defect_component_count"] == 0
    assert result.metrics["defect_strategy_counts"] == {}
    assert result.metrics["changed_pixel_count"] == 0


def test_defect_aware_strength_zero_returns_exact_input():
    image = _gradient_image(np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[10:13, 12:15] = 255

    result = repair_image(image, mask, _config(strength=0.0))

    assert np.array_equal(result.repaired_image, image)
    assert result.metrics["defect_component_count"] == 1
    assert sum(result.metrics["defect_strategy_counts"].values()) == 1
    assert result.metrics["defect_processing_skipped"] is True
    assert result.metrics["changed_pixel_count"] == 0


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_defect_aware_preserves_dtype_and_pixels_outside_mask(dtype):
    image = _gradient_image(dtype)
    damaged = image.copy()
    max_value = np.iinfo(dtype).max
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[9:16, 10:18] = 255
    damaged[mask > 0] = max_value

    result = repair_image(damaged, mask, _config(padding=10))

    outside = result.soft_mask <= 0.0
    assert result.repaired_image.dtype == dtype
    assert np.array_equal(result.repaired_image[outside], damaged[outside])
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0


def test_cli_method_choices_include_defect_aware():
    assert "defect_aware" in REPAIR_METHODS

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--mask",
            "mask.png",
            "--output",
            "output.png",
            "--method",
            "defect_aware",
        ]
    )

    assert args.method == "defect_aware"
