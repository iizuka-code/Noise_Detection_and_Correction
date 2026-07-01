import numpy as np
from pathlib import Path

from dust_mask_repair.io import read_image, write_image
from dust_mask_repair.mask import normalize_mask


def _artifact_path(name):
    directory = Path(__file__).resolve().parents[1] / "test_outputs" / "mask_loading"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def test_grayscale_mask_loads_and_normalizes():
    mask = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    path = _artifact_path("mask.png")
    write_image(path, mask)

    loaded = read_image(path).pixels
    normalized = normalize_mask(loaded, "grayscale")

    assert normalized.channel_used == "grayscale"
    assert normalized.values.dtype == np.float32
    assert np.isclose(normalized.values[0, 0], 0.0)
    assert np.isclose(normalized.values[1, 0], 1.0)


def test_alpha_mask_loads_and_normalizes():
    mask = np.zeros((3, 3, 4), dtype=np.uint8)
    mask[:, :, :3] = 10
    mask[1, 1, 3] = 255
    path = _artifact_path("alpha_mask.png")
    write_image(path, mask)

    loaded = read_image(path).pixels
    normalized = normalize_mask(loaded, "alpha")
    auto = normalize_mask(loaded, "auto")

    assert normalized.channel_used == "alpha"
    assert auto.channel_used == "alpha"
    assert normalized.values[1, 1] == 1.0
    assert np.count_nonzero(normalized.values) == 1


def test_red_channel_mask_loads_and_auto_detects():
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 2, 0] = 255
    path = _artifact_path("red_mask.png")
    write_image(path, mask)

    loaded = read_image(path).pixels
    normalized = normalize_mask(loaded, "red")
    auto = normalize_mask(loaded, "auto")

    assert normalized.channel_used == "red"
    assert auto.channel_used == "red"
    assert normalized.values[1, 2] == 1.0
    assert np.count_nonzero(normalized.values) == 1


def test_rgb_grayscale_channel_normalizes_before_luma():
    mask = np.zeros((2, 2, 3), dtype=np.uint8)
    mask[0, 1] = [128, 128, 128]

    normalized = normalize_mask(mask, "grayscale")

    assert np.isclose(normalized.values[0, 1], 128 / 255)
    assert normalized.values[1, 1] == 0.0
