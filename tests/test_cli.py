import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from dust_mask_repair.cli import build_parser
from dust_mask_repair.io import read_image, write_image
from dust_mask_repair.red_highlight_cli import build_repair_parser as build_red_repair_parser
from dust_mask_repair.white_dust_cli import build_repair_parser as build_white_repair_parser
from dust_mask_repair.xmp import write_mask_xmp


def test_repair_clis_accept_defect_aware_method():
    assert (
        build_parser()
        .parse_args(["--image", "in.png", "--mask", "mask.png", "--output", "out.png", "--method", "defect_aware"])
        .method
        == "defect_aware"
    )
    assert (
        build_white_repair_parser()
        .parse_args(
            [
                "--image",
                "in.png",
                "--source",
                "inspection.png",
                "--output",
                "out.png",
                "--method",
                "defect_aware",
            ]
        )
        .method
        == "defect_aware"
    )
    assert (
        build_red_repair_parser()
        .parse_args(
            [
                "--image",
                "in.png",
                "--red-image",
                "red.png",
                "--output",
                "out.png",
                "--method",
                "defect_aware",
            ]
        )
        .method
        == "defect_aware"
    )


def test_repair_clis_accept_frequency_scope_args():
    parsed = build_parser().parse_args(
        [
            "--image",
            "in.png",
            "--mask",
            "mask.png",
            "--output",
            "out.png",
            "--method",
            "defect_aware",
            "--frequency-scope-mask",
            "scope.png",
            "--frequency-guided",
        ]
    )
    assert parsed.frequency_scope_mask == "scope.png"
    assert parsed.frequency_guided is True

    white = build_white_repair_parser().parse_args(
        [
            "--image",
            "in.png",
            "--source",
            "inspection.png",
            "--output",
            "out.png",
            "--method",
            "defect_aware",
            "--frequency-scope-mask",
            "scope.png",
        ]
    )
    assert white.frequency_scope_mask == "scope.png"


def test_cli_writes_output_and_debug_dir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_root = Path(repo_root) / "test_outputs" / "cli"
    output_root.mkdir(parents=True, exist_ok=True)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :] = [90, 100, 110]
    image[8, 8] = [255, 255, 255]
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8, 8] = 255

    image_path = output_root / "input.png"
    mask_path = output_root / "mask.png"
    output_path = output_root / "repaired.png"
    debug_dir = output_root / "debug"
    write_image(image_path, image)
    write_image(mask_path, mask)

    env = os.environ.copy()
    src_path = os.path.join(repo_root, "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dust_mask_repair.cli",
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--output",
            str(output_path),
            "--method",
            "hybrid",
            "--mask-channel",
            "auto",
            "--threshold",
            "0.5",
            "--dilate-radius",
            "0",
            "--feather-radius",
            "0",
            "--strength",
            "1.0",
            "--max-component-area",
            "5000",
            "--debug-dir",
            str(debug_dir),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    repaired = read_image(output_path).pixels
    assert repaired.dtype == np.uint8
    assert np.array_equal(repaired[0, 0], image[0, 0])
    for filename in [
        "normalized_mask.png",
        "binary_mask.png",
        "soft_mask.png",
        "repaired_preview.png",
        "diff_visualization.png",
        "metrics.json",
    ]:
        assert (debug_dir / filename).exists()


def test_cli_accepts_xmp_mask():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_root = Path(repo_root) / "test_outputs" / "cli_xmp"
    output_root.mkdir(parents=True, exist_ok=True)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :] = [90, 100, 110]
    image[8, 8] = [255, 255, 255]
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8, 8] = 255

    image_path = output_root / "input.png"
    mask_path = output_root / "mask.xmp"
    output_path = output_root / "repaired.png"
    write_image(image_path, image)
    write_mask_xmp(
        mask_path,
        mask=mask,
        manifest={"final_mask_pixels": 1},
        source_path="inspection.arw",
        role="white_dust_detection",
    )

    env = os.environ.copy()
    src_path = os.path.join(repo_root, "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dust_mask_repair.cli",
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--output",
            str(output_path),
            "--method",
            "hybrid",
            "--mask-channel",
            "auto",
            "--threshold",
            "0.5",
            "--dilate-radius",
            "0",
            "--feather-radius",
            "0",
            "--strength",
            "1.0",
            "--max-component-area",
            "5000",
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    repaired = read_image(output_path).pixels
    assert repaired.dtype == np.uint8
    assert np.array_equal(repaired[0, 0], image[0, 0])
    assert not np.array_equal(repaired[8, 8], image[8, 8])


def test_white_dust_repair_cli_generates_mask_and_repairs_with_kl():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_root = Path(repo_root) / "test_outputs" / "white_cli"
    output_root.mkdir(parents=True, exist_ok=True)

    image = np.zeros((80, 96, 3), dtype=np.uint8)
    image[:, :] = [70, 92, 118]
    image[35:43, 42:50] = [245, 245, 245]
    source = np.zeros_like(image)
    source[:, :] = [9, 9, 11]
    yy, xx = np.ogrid[: source.shape[0], : source.shape[1]]
    disk = (yy - 39) ** 2 + (xx - 46) ** 2 <= 5 * 5
    source[disk] = [238, 236, 230]

    image_path = output_root / "target.png"
    source_path = output_root / "inspection.png"
    output_path = output_root / "repaired.png"
    mask_path = output_root / "mask.png"
    write_image(image_path, image)
    write_image(source_path, source)

    env = os.environ.copy()
    src_path = os.path.join(repo_root, "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dust_mask_repair.white_dust_cli import repair_main; raise SystemExit(repair_main())",
            "--image",
            str(image_path),
            "--source",
            str(source_path),
            "--output",
            str(output_path),
            "--mask-output",
            str(mask_path),
            "--method",
            "kl",
            "--detection-long-edge",
            "96",
            "--local-radius",
            "3",
            "--background-mode",
            "dark",
            "--min-area",
            "2",
            "--max-area",
            "200",
            "--max-dim",
            "24",
            "--max-thickness",
            "12",
            "--dilate-radius",
            "0",
            "--feather-radius",
            "0",
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    assert mask_path.exists()
    repaired = read_image(output_path).pixels
    mask = read_image(mask_path).pixels
    assert int(np.count_nonzero(mask)) > 0
    assert repaired.dtype == np.uint8
    assert np.array_equal(repaired[0, 0], image[0, 0])
