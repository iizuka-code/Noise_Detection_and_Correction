from __future__ import annotations

import numpy as np

from dust_mask_repair import WhiteDustConfig, detect_white_dust_mask
from dust_mask_repair.white_dust import detect_white_dust_source_image


def _dark_floating_dust_fixture(height: int = 220, width: int = 340) -> tuple[np.ndarray, np.ndarray]:
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    texture = 2.5 * np.sin(xx * 23.0) + 2.0 * np.cos(yy * 17.0)
    image = np.stack(
        [
            15.0 + 8.0 * xx + texture,
            15.0 + 7.0 * yy + texture * 0.8,
            18.0 + 9.0 * xx + texture * 0.7,
        ],
        axis=2,
    )
    target = np.zeros((height, width), dtype=bool)
    _paint_disk(image, target, int(height * 0.34), int(width * 0.30), 3, (230.0, 226.0, 220.0))
    _paint_disk(image, target, int(height * 0.58), int(width * 0.45), 3, (210.0, 128.0, 180.0))
    _paint_disk(image, target, int(height * 0.73), int(width * 0.70), 2, (126.0, 170.0, 238.0))
    for x in range(int(width * 0.56), int(width * 0.76)):
        y = int(height * 0.22 + np.sin((x - width * 0.56) / 5.5) * 7.0)
        _paint_disk(image, target, y, x, 1, (214.0, 205.0, 230.0))

    # A low-contrast smudge should stay below the detector threshold.
    _paint_disk(image, None, int(height * 0.55), int(width * 0.78), 10, (38.0, 39.0, 45.0))
    return np.clip(image, 0, 255).astype(np.uint8), target


def _brown_white_dust_fixture(height: int = 240, width: int = 360) -> tuple[np.ndarray, np.ndarray]:
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    texture = 5.0 * np.sin(xx * 18.0) + 3.5 * np.cos(yy * 15.0)
    image = np.stack(
        [
            116.0 + 18.0 * xx + 8.0 * yy + texture,
            77.0 + 10.0 * xx + 7.0 * yy + texture * 0.55,
            42.0 + 6.0 * xx + 5.0 * yy + texture * 0.28,
        ],
        axis=2,
    )
    target = np.zeros((height, width), dtype=bool)
    xs = np.arange(int(width * 0.28), int(width * 0.74))
    center = height * 0.52 + np.sin((xs - xs.min()) / 10.0) * 14.0 + np.sin((xs - xs.min()) / 4.8) * 3.2
    for index, x in enumerate(xs):
        y = int(round(center[index]))
        _paint_disk(image, target, y, int(x), 2, (236.0, 232.0, 219.0))
        if index % 13 == 0:
            _paint_disk(image, target, y + 3, int(x), 1, (225.0, 220.0, 208.0))

    # A dull warm spot should not be treated as white dust.
    _paint_disk(image, None, int(height * 0.24), int(width * 0.22), 9, (150.0, 106.0, 64.0))
    return np.clip(image, 0, 255).astype(np.uint8), target


def _paint_disk(
    image: np.ndarray,
    target: np.ndarray | None,
    cy: int,
    cx: int,
    radius: int,
    color: tuple[float, float, float],
) -> None:
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    image[disk] = np.asarray(color, dtype=np.float32)
    if target is not None:
        target[disk] = True


def test_white_dust_detector_marks_white_wavy_dust_on_brown_base() -> None:
    image, target = _brown_white_dust_fixture()
    result = detect_white_dust_mask(
        image,
        WhiteDustConfig(
            detection_long_edge=360,
            local_radius=5,
            background_mode="brown",
            max_area=6000,
            max_dim=260,
            max_thickness=18,
        ),
    )
    mask = result.preview_mask > 0
    coverage = int(np.count_nonzero(mask & target)) / int(np.count_nonzero(target))

    assert result.manifest["detector_version"] == "dust_on_dark_or_brown_v2"
    assert result.manifest["component_count"] >= 1
    assert coverage > 0.70
    assert int(np.count_nonzero(mask & ~target)) < int(np.count_nonzero(target)) * 3
    assert result.overlay_preview.shape == image.shape
    assert result.score_map.shape == image.shape[:2]


def test_white_dust_detector_can_skip_visual_artifacts_without_changing_mask() -> None:
    image, _target = _brown_white_dust_fixture()
    config = WhiteDustConfig(
        detection_long_edge=360,
        local_radius=5,
        background_mode="brown",
        max_area=6000,
        max_dim=260,
        max_thickness=18,
    )

    with_artifacts = detect_white_dust_mask(image, config)
    without_artifacts = detect_white_dust_mask(
        image,
        WhiteDustConfig(
            detection_long_edge=360,
            local_radius=5,
            background_mode="brown",
            max_area=6000,
            max_dim=260,
            max_thickness=18,
            visual_artifacts=False,
        ),
    )

    assert np.array_equal(without_artifacts.preview_mask, with_artifacts.preview_mask)
    assert without_artifacts.overlay_preview.shape == (0, 0, 3)
    assert without_artifacts.score_map.shape == (0, 0)


def test_white_dust_detector_suppresses_bright_warm_frame() -> None:
    image = np.zeros((180, 260, 3), dtype=np.uint8)
    image[:, :] = [118, 80, 44]
    image[:, 8:18] = [242, 214, 168]
    target = np.zeros(image.shape[:2], dtype=bool)
    _paint_disk(image, target, 92, 132, 5, (238.0, 233.0, 221.0))

    result = detect_white_dust_mask(
        image,
        WhiteDustConfig(
            detection_long_edge=260,
            local_radius=5,
            background_mode="brown",
            max_area=2000,
            max_dim=80,
            max_thickness=18,
            brown_luma_max=170.0,
        ),
    )
    mask = result.preview_mask > 0

    assert int(np.count_nonzero(mask & target)) > 0
    assert int(np.count_nonzero(mask[:, 8:18])) == 0


def test_dark_background_detector_marks_white_and_colored_floating_dust() -> None:
    image, target = _dark_floating_dust_fixture()
    result = detect_white_dust_mask(
        image,
        WhiteDustConfig(
            detection_long_edge=340,
            local_radius=5,
            background_mode="dark",
            max_area=5000,
            max_dim=220,
            max_thickness=16,
        ),
    )
    mask = result.preview_mask > 0
    coverage = int(np.count_nonzero(mask & target)) / int(np.count_nonzero(target))

    assert result.manifest["detector_version"] == "dust_on_dark_or_brown_v2"
    assert result.manifest["parameters"]["background_mode"] == "dark"
    assert result.manifest["component_count"] >= 3
    assert coverage > 0.62
    assert int(np.count_nonzero(mask & ~target)) < int(np.count_nonzero(target)) * 4


def test_white_dust_source_detection_returns_source_sized_mask() -> None:
    image, target = _brown_white_dust_fixture(height=360, width=540)
    result = detect_white_dust_source_image(
        image,
        WhiteDustConfig(
            detection_long_edge=270,
            local_radius=4,
            background_mode="brown",
            max_area=6000,
            max_dim=260,
            max_thickness=18,
            mask_edge_mode="wide",
        ),
    )

    assert result.mask.shape == image.shape[:2]
    assert result.manifest["source_shape"] == [360, 540]
    assert result.manifest["detection_shape"] == [180, 270]
    assert int(np.count_nonzero((result.mask > 0) & target)) > 0
