from __future__ import annotations

import json
from pathlib import Path

from dust_mask_repair.benchmark import BENCHMARK_VERSION, BenchmarkConfig, main, run_benchmark


ROOT = Path(__file__).resolve().parents[1]


def _artifact_path(name: str) -> Path:
    directory = ROOT / "test_outputs" / "benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def test_run_benchmark_returns_red_detection_and_repair_summary() -> None:
    result = run_benchmark(
        BenchmarkConfig(
            width=96,
            height=72,
            iterations=1,
            warmup=0,
            detection_long_edge=96,
        )
    )

    assert result["benchmark_version"] == BENCHMARK_VERSION
    assert result["image_shape"] == [72, 96]
    assert len(result["runs"]) == 1
    run = result["runs"][0]
    assert run["detect_ms"] >= 0.0
    assert run["repair_ms"] >= 0.0
    assert run["final_mask_pixels"] > 0
    assert run["max_abs_diff_outside_mask"] == 0.0
    assert result["summary"]["total_ms"]["max"] >= result["summary"]["total_ms"]["min"]


def test_benchmark_cli_writes_json_file() -> None:
    output = _artifact_path("benchmark.json")
    exit_code = main(
        [
            "--width",
            "96",
            "--height",
            "72",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--detection-long-edge",
            "96",
            "--include-long-scratches",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark_version"] == BENCHMARK_VERSION
    assert payload["config"]["include_long_scratches"] is True
    assert payload["summary"]["max_abs_diff_outside_mask_max"] == 0.0
