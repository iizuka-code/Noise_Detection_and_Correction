from __future__ import annotations

import numpy as np

from dust_mask_repair import (
    RedHighlightConfig,
    RepairConfig,
    WhiteDustConfig,
    repair_image_from_red_highlight,
    repair_image_from_white_dust,
)


def _add_disk(image: np.ndarray, cy: int, cx: int, radius: int, color: tuple[int, int, int]) -> None:
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    image[disk] = np.asarray(color, dtype=np.uint8)


def test_repair_image_from_red_highlight_accepts_decoded_rgb_arrays() -> None:
    normal = np.zeros((80, 96, 3), dtype=np.uint8)
    normal[:, :] = [100, 120, 140]
    normal[35:43, 42:50] = [15, 15, 15]
    red = np.zeros((80, 96, 3), dtype=np.uint8)
    red[:, :] = [5, 3, 4]
    _add_disk(red, 39, 46, 8, (56, 4, 5))
    _add_disk(red, 39, 46, 5, (232, 18, 24))

    result = repair_image_from_red_highlight(
        normal,
        red,
        red_config=RedHighlightConfig(detection_long_edge=96, max_area=260, max_dim=28),
        repair_config=RepairConfig(
            method="hybrid",
            mask_channel="grayscale",
            threshold=0.5,
            dilate_radius=0,
            feather_radius=0,
            padding=8,
        ),
    )

    assert result.generated_mask.shape == normal.shape[:2]
    assert result.repaired_image.shape == normal.shape
    assert result.red_highlight.manifest["detector_version"] == "red_highlight_v1"
    assert result.repair.metrics["max_abs_diff_outside_mask"] == 0.0
    assert int(np.count_nonzero(result.generated_mask)) > 0


def test_repair_image_from_red_highlight_defaults_to_nonvisual_artifacts() -> None:
    normal = np.zeros((80, 96, 3), dtype=np.uint8)
    normal[:, :] = [100, 120, 140]
    normal[35:43, 42:50] = [15, 15, 15]
    red = np.zeros((80, 96, 3), dtype=np.uint8)
    red[:, :] = [5, 3, 4]
    _add_disk(red, 39, 46, 8, (56, 4, 5))
    _add_disk(red, 39, 46, 5, (232, 18, 24))

    result = repair_image_from_red_highlight(
        normal,
        red,
        repair_config=RepairConfig(
            method="hybrid",
            mask_channel="grayscale",
            threshold=0.5,
            dilate_radius=0,
            feather_radius=0,
            padding=8,
        ),
    )

    assert int(np.count_nonzero(result.generated_mask)) > 0
    assert result.red_highlight.overlay.shape == (0, 0, 3)
    assert result.red_highlight.overlay_preview.shape == (0, 0, 3)
    assert result.red_highlight.score_map.shape == (0, 0)
    assert result.red_highlight.manifest["parameters"]["visual_artifacts"] is False


def test_repair_image_from_red_highlight_rejects_mismatched_dimensions() -> None:
    normal = np.zeros((32, 32, 3), dtype=np.uint8)
    red = np.zeros((30, 32, 3), dtype=np.uint8)

    try:
        repair_image_from_red_highlight(normal, red)
    except ValueError as exc:
        assert "dimensions differ" in str(exc)
    else:
        raise AssertionError("expected dimension mismatch to fail")


def test_repair_image_from_white_dust_generates_mask_and_repairs_with_kl_default() -> None:
    normal = np.zeros((96, 128, 3), dtype=np.uint8)
    normal[:, :] = [72, 92, 118]
    normal[38:46, 58:66] = [245, 245, 245]

    inspection = np.zeros_like(normal)
    inspection[:, :] = [10, 10, 12]
    _add_disk(inspection, 42, 62, 5, (238, 235, 230))

    result = repair_image_from_white_dust(
        normal,
        inspection,
        white_config=WhiteDustConfig(
            detection_long_edge=128,
            local_radius=3,
            background_mode="dark",
            min_area=2,
            max_area=200,
            max_dim=24,
            max_thickness=12,
            visual_artifacts=False,
        ),
    )

    assert result.generated_mask.shape == normal.shape[:2]
    assert result.repaired_image.shape == normal.shape
    assert result.white_dust.manifest["detector_version"] == "dust_on_dark_or_brown_v2"
    assert result.white_dust.manifest["parameters"]["visual_artifacts"] is False
    assert int(np.count_nonzero(result.generated_mask)) > 0
    assert result.repair.metrics["max_abs_diff_outside_mask"] == 0.0


def test_repair_image_from_white_dust_rejects_mismatched_dimensions() -> None:
    normal = np.zeros((32, 32, 3), dtype=np.uint8)
    inspection = np.zeros((30, 32, 3), dtype=np.uint8)

    try:
        repair_image_from_white_dust(normal, inspection)
    except ValueError as exc:
        assert "dimensions differ" in str(exc)
    else:
        raise AssertionError("expected dimension mismatch to fail")
