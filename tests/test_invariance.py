import numpy as np
import pytest
from pathlib import Path

from dust_mask_repair import RepairConfig, repair_image
from dust_mask_repair.io import read_image, write_image


def _config(**overrides):
    values = {
        "method": "hybrid",
        "mask_channel": "grayscale",
        "threshold": 0.5,
        "dilate_radius": 0,
        "feather_radius": 0,
        "strength": 1.0,
        "padding": 3,
    }
    values.update(overrides)
    return RepairConfig(**values)


def _artifact_path(name):
    directory = Path(__file__).resolve().parents[1] / "test_outputs" / "invariance"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def test_pixels_outside_soft_mask_are_unchanged():
    rng = np.random.default_rng(123)
    image = rng.integers(0, 255, size=(30, 30, 3), dtype=np.uint8)
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[14:16, 14:16] = 255

    result = repair_image(image, mask, _config())

    outside = result.soft_mask <= 0.0
    assert np.array_equal(result.repaired_image[outside], image[outside])
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0


def test_unmasked_edge_content_is_unchanged():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 5] = [255, 255, 255]
    image[:, 20] = [100, 120, 140]
    image[16, 20] = [255, 255, 255]
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[16, 20] = 255

    result = repair_image(image, mask, _config(method="median"))

    assert np.array_equal(result.repaired_image[:, 5], image[:, 5])
    outside = result.soft_mask <= 0.0
    assert np.array_equal(result.repaired_image[outside], image[outside])


def test_size_mismatch_raises_clear_error():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((9, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="dimensions differ"):
        repair_image(image, mask, _config())


def test_uint16_png_roundtrip_and_repair_preserve_uint16():
    image = np.zeros((16, 16, 3), dtype=np.uint16)
    image[:, :] = [40000, 30000, 20000]
    image[7, 7] = [65535, 65535, 65535]
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[7, 7] = 255

    input_path = _artifact_path("input_uint16.png")
    write_image(input_path, image)
    loaded = read_image(input_path).pixels
    result = repair_image(loaded, mask, _config(method="inpaint"))

    assert loaded.dtype == np.uint16
    assert loaded.max() > 255
    assert result.repaired_image.dtype == np.uint16
    assert result.repaired_image[0, 0, 0] == 40000
    assert result.repaired_image[7, 7, 0] < 65535


def test_jpeg_input_can_be_read_for_cli_workflows():
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:, :] = [30, 120, 210]
    jpeg_path = _artifact_path("input_rgb.jpg")
    write_image(jpeg_path, image)

    loaded = read_image(jpeg_path)

    assert loaded.pixels.dtype == np.uint8
    assert loaded.bit_depth == 8
    assert loaded.color_mode == "rgb"
    assert loaded.pixels.shape == image.shape
