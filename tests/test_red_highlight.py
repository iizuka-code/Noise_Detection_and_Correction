from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from dust_mask_repair.io import read_image, write_image
from dust_mask_repair.red_highlight import (
    RedHighlightConfig,
    detect_red_highlight_mask,
    detect_red_highlight_source_image,
    run_red_highlight_detector,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _artifact_path(name: str) -> Path:
    directory = ROOT / "test_outputs" / "red_highlight"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _add_disk(image: np.ndarray, cy: int, cx: int, radius: int, color: tuple[int, int, int]) -> None:
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    image[disk] = np.asarray(color, dtype=np.uint8)


def _red_highlight_fixture(height: int = 360, width: int = 540) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    red = 8.0 + 8.0 * yy + 4.0 * np.sin(xx * 8.0)
    green = 3.0 + 2.0 * yy + 0.0 * xx
    blue = 4.0 + 2.0 * xx + 0.0 * yy
    image = np.stack([red, green, blue], axis=2)
    image[:, width - 32 :, 0] += 95.0
    image[:, width - 32 :, 1] += 9.0
    image[:, width - 32 :, 2] += 5.0
    image = np.clip(image, 0, 255).astype(np.uint8)

    for cy, cx, radius in [(92, 120, 5), (164, 244, 7), (250, 355, 6), (290, 180, 4)]:
        _add_disk(image, cy, cx, radius + 2, (80, 8, 8))
        _add_disk(image, cy, cx, radius, (225, 15, 18))
        _add_disk(image, cy, cx, max(1, radius // 2), (255, 38, 42))
    return image


def _black_red_glow_fixture(height: int = 480, width: int = 720) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    image = np.stack(
        [
            5.0 + 5.0 * yy + 2.0 * np.sin(xx * 11.0),
            3.0 + 1.5 * yy + 0.0 * xx,
            4.0 + 1.5 * xx + 0.0 * yy,
        ],
        axis=2,
    )
    image = np.clip(image, 0, 255).astype(np.uint8)
    for cy, cx, radius in [(72, 120, 5), (130, 455, 9), (245, 305, 7), (356, 580, 10), (410, 180, 6)]:
        _add_disk(image, cy, cx, radius + 6, (42, 3, 5))
        _add_disk(image, cy, cx, radius + 3, (112, 8, 12))
        _add_disk(image, cy, cx, radius, (235, 18, 28))
        _add_disk(image, cy, cx, max(2, radius // 2), (255, 235, 230))
    return image


def _precise_red_boundary_fixture(height: int = 360, width: int = 540) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = 5
    image[:, :, 1] = 3
    image[:, :, 2] = 4
    target = np.zeros((height, width), dtype=bool)
    yy, xx = np.ogrid[:height, :width]
    for cy, cx, radius in [(96, 130, 8), (190, 275, 10), (270, 400, 7)]:
        halo = (yy - cy) ** 2 + (xx - cx) ** 2 <= (radius + 8) ** 2
        mid = (yy - cy) ** 2 + (xx - cx) ** 2 <= (radius + 3) ** 2
        core = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
        image[halo] = np.asarray((30, 4, 5), dtype=np.uint8)
        image[mid] = np.asarray((118, 8, 10), dtype=np.uint8)
        image[core] = np.asarray((230, 20, 24), dtype=np.uint8)
        target |= mid
    return image, target


def test_red_highlight_detector_marks_local_red_dust_and_suppresses_border_glow() -> None:
    image = _red_highlight_fixture()
    result = detect_red_highlight_mask(
        image,
        RedHighlightConfig(
            detection_long_edge=540,
            local_radius=4,
            max_area=360,
            max_dim=42,
        ),
    )

    assert result.manifest["component_count"] >= 4
    assert int(np.count_nonzero(result.preview_mask[70:115, 95:145])) > 20
    assert int(np.count_nonzero(result.preview_mask[140:190, 220:270])) > 30
    assert int(np.count_nonzero(result.preview_mask[:, -28:])) < 20


def test_red_highlight_detector_marks_red_glow_points_on_black_background() -> None:
    image = _black_red_glow_fixture()
    result = detect_red_highlight_mask(
        image,
        RedHighlightConfig(
            detection_long_edge=720,
            local_radius=5,
            max_area=520,
            max_dim=52,
        ),
    )

    assert result.manifest["component_count"] >= 5
    assert result.manifest["component_count"] < 24
    for y0, y1, x0, x1 in [
        (55, 92, 103, 140),
        (107, 154, 432, 478),
        (222, 270, 282, 330),
        (330, 382, 555, 608),
        (390, 430, 160, 205),
    ]:
        assert int(np.count_nonzero(result.preview_mask[y0:y1, x0:x1])) > 30


def test_tight_mask_edge_stays_closer_to_red_boundary_than_wide() -> None:
    image, target = _precise_red_boundary_fixture()
    tight = detect_red_highlight_mask(
        image,
        RedHighlightConfig(
            detection_long_edge=540,
            local_radius=5,
            max_area=520,
            max_dim=56,
            mask_edge_mode="tight",
        ),
    )
    wide = detect_red_highlight_mask(
        image,
        RedHighlightConfig(
            detection_long_edge=540,
            local_radius=5,
            max_area=520,
            max_dim=56,
            mask_edge_mode="wide",
        ),
    )

    tight_mask = tight.preview_mask > 0
    wide_mask = wide.preview_mask > 0
    assert tight.manifest["component_count"] >= 3
    assert int(np.count_nonzero(tight_mask & target)) > 180
    assert int(np.count_nonzero(tight_mask & ~target)) < 110
    assert int(np.count_nonzero(wide_mask)) > int(np.count_nonzero(tight_mask)) + 80


def test_red_highlight_source_detection_returns_source_sized_mask() -> None:
    image, _target = _precise_red_boundary_fixture(height=540, width=810)
    result = detect_red_highlight_source_image(
        image,
        RedHighlightConfig(
            detection_long_edge=360,
            local_radius=5,
            max_area=520,
            max_dim=56,
            mask_edge_mode="tight",
        ),
    )

    assert result.mask.shape == image.shape[:2]
    assert result.manifest["source_shape"] == [540, 810]
    assert result.manifest["detection_shape"] == [240, 360]
    assert result.manifest["final_refine"]["mode"] == "source_rgb_roi_refine"


def test_run_red_highlight_detector_writes_mask_manifest_and_artifacts() -> None:
    source = _artifact_path("red_fixture.png")
    output = _artifact_path("red_out")
    write_image(source, _red_highlight_fixture())

    manifest = run_red_highlight_detector(
        source,
        output,
        RedHighlightConfig(
            detection_long_edge=540,
            max_area=360,
            max_dim=42,
        ),
    )

    mask = read_image(output / "mask.png").pixels
    assert manifest["detector_version"] == "red_highlight_v1"
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert (output / "red_fixture_red_highlight_mask.png").exists()
    assert (output / "overlay_preview.png").exists()
    assert (output / "manifest.json").exists()


def test_red_highlight_cli_writes_mask_manifest_and_combined_repair() -> None:
    normal = _artifact_path("normal.png")
    red = _artifact_path("red_cli.png")
    mask_dir = _artifact_path("cli_detect")
    repaired = _artifact_path("repaired.png")
    normal_image = np.zeros((64, 64, 3), dtype=np.uint8)
    normal_image[:, :] = [100, 120, 140]
    normal_image[25:30, 25:30] = [255, 255, 255]
    red_image = np.zeros((64, 64, 3), dtype=np.uint8)
    red_image[:, :] = [5, 3, 4]
    _add_disk(red_image, 27, 27, 5, (230, 18, 24))
    write_image(normal, normal_image)
    write_image(red, red_image)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    detect = subprocess.run(
        [
            sys.executable,
            "-m",
            "dust_mask_repair.red_highlight_cli",
            "--source",
            str(red),
            "--output-dir",
            str(mask_dir),
            "--detection-long-edge",
            "64",
            "--max-red-area",
            "200",
            "--max-red-dim",
            "24",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert detect.returncode == 0, detect.stderr
    detect_payload = json.loads(detect.stdout)
    assert detect_payload["detector_version"] == "red_highlight_v1"
    assert (mask_dir / "mask.png").exists()

    repair = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dust_mask_repair.red_highlight_cli import repair_main; raise SystemExit(repair_main())",
            "--image",
            str(normal),
            "--red-image",
            str(red),
            "--output",
            str(repaired),
            "--detection-long-edge",
            "64",
            "--max-red-area",
            "200",
            "--max-red-dim",
            "24",
            "--dilate-radius",
            "0",
            "--feather-radius",
            "0",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert repair.returncode == 0, repair.stderr
    repair_payload = json.loads(repair.stdout)
    assert repaired.exists()
    assert Path(repair_payload["generated_mask"]).exists()
    assert repair_payload["repair"]["max_abs_diff_outside_mask"] == 0.0
