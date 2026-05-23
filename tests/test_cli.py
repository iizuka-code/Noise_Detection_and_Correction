import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from dust_mask_repair.io import read_image, write_image


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
