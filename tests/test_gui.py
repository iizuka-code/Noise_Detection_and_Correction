import json

import numpy as np
import pytest

from dust_mask_repair import gui as gui_module
from dust_mask_repair.gui import (
    GUI_METHODS,
    _read_gui_image,
    run_edge_guided_gui_test,
    run_white_dust_gui_job,
)
from dust_mask_repair.io import ImageData, read_image, write_image


def test_gui_methods_expose_defect_aware_without_changing_default():
    assert GUI_METHODS[0] == "kl"
    assert "defect_aware" in GUI_METHODS



def test_gui_edge_guided_test_job_writes_visual_outputs(tmp_path):
    result = run_edge_guided_gui_test(output_dir=tmp_path / "out")

    assert result.repaired_path.exists()
    assert result.target_preview_path.exists()
    assert result.mask_path.exists()
    assert result.overlay_path.exists()
    assert result.score_path.exists()
    assert result.manifest_path.exists()
    assert result.metrics_path.exists()
    assert (result.run_dir / "edge_guided_clean_answer.png").exists()
    assert (result.run_dir / "edge_guided_disabled_comparison.png").exists()
    assert (result.run_dir / "debug" / "defect_components.json").exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["補正方法"] == "defect_aware"
    assert manifest["テストケース"] == "diagonal_edge_micro_dust"
    assert manifest["合否"] is True
    assert manifest["正答率"]["edge_guided_mae_0_255"] < manifest["正答率"]["edge_guided_disabled_mae_0_255"]
    assert manifest["正答率"]["edge_guided_mae_0_255"] < manifest["正答率"]["corrupted_mae_0_255"]

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["gui_edge_guided_test"] is True
    assert metrics["small_local_edge_guided_component_count"] == 1
    assert metrics["small_local_edge_guided_pixel_count"] > 0
    assert metrics["max_abs_diff_outside_mask"] == 0.0

def test_gui_job_generates_repaired_image_mask_and_manifest(tmp_path):
    target = np.zeros((80, 96, 3), dtype=np.uint8)
    target[:, :] = [70, 92, 118]
    target[35:43, 42:50] = [245, 245, 245]

    inspection = np.zeros_like(target)
    inspection[:, :] = [9, 9, 11]
    yy, xx = np.ogrid[: inspection.shape[0], : inspection.shape[1]]
    disk = (yy - 39) ** 2 + (xx - 46) ** 2 <= 5 * 5
    inspection[disk] = [238, 236, 230]

    target_path = tmp_path / "target.png"
    inspection_path = tmp_path / "inspection.png"
    write_image(target_path, target)
    write_image(inspection_path, inspection)

    result = run_white_dust_gui_job(
        target_path=target_path,
        inspection_path=inspection_path,
        output_dir=tmp_path / "out",
        method="kl",
        detection_long_edge=96,
    )

    assert result.repaired_path.exists()
    assert result.mask_path.exists()
    assert result.overlay_path.exists()
    assert result.score_path.exists()
    assert result.manifest_path.exists()
    assert result.metrics_path.exists()
    status_path = result.run_dir / "processing_status.json"
    assert status_path.exists()

    repaired = read_image(result.repaired_path).pixels
    mask = read_image(result.mask_path).pixels
    assert int(np.count_nonzero(mask)) > 0
    assert repaired.dtype == np.uint8
    assert np.array_equal(repaired[0, 0], target[0, 0])

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"].startswith("KLComplementary 2.0")
    assert manifest["対象写真"] == str(target_path)
    assert manifest["マスク作成用写真"] == str(inspection_path)
    assert manifest["補正方法"] == "kl"
    assert manifest["mask_pixels"] > 0
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["phase"] == "complete"
    assert status["repaired_path"] == str(result.repaired_path)



def test_gui_job_records_repair_tuning_options_and_expanded_mask(tmp_path):
    target = np.zeros((80, 96, 3), dtype=np.uint8)
    target[:, :] = [70, 92, 118]
    target[36:42, 43:49] = [245, 245, 245]

    inspection = np.zeros_like(target)
    inspection[:, :] = [9, 9, 11]
    yy, xx = np.ogrid[: inspection.shape[0], : inspection.shape[1]]
    disk = (yy - 39) ** 2 + (xx - 46) ** 2 <= 4 * 4
    inspection[disk] = [238, 236, 230]

    target_path = tmp_path / "target.png"
    inspection_path = tmp_path / "inspection.png"
    write_image(target_path, target)
    write_image(inspection_path, inspection)

    result = run_white_dust_gui_job(
        target_path=target_path,
        inspection_path=inspection_path,
        output_dir=tmp_path / "out",
        method="defect_aware",
        detection_long_edge=96,
        repair_expand_radius=2,
        feather_radius=2,
        color_match_strength=0.6,
        grain_strength=0.4,
    )

    repair_mask_path = result.run_dir / "repair_mask_expanded.png"
    assert repair_mask_path.exists()
    detected_mask = read_image(result.mask_path).pixels > 0
    repair_mask = read_image(repair_mask_path).pixels > 0
    assert int(np.count_nonzero(repair_mask)) > int(np.count_nonzero(detected_mask))

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["補正対象マスク"] == str(repair_mask_path)
    assert manifest["補正設定"] == {
        "mask_expand_radius": 2,
        "feather_radius": 2,
        "color_match_strength": 0.6,
        "grain_strength": 0.4,
    }
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["repair_mask_pixel_count"] > metrics["core_mask_pixel_count"]
def test_gui_job_rejects_mismatched_dimensions(tmp_path):
    target_path = tmp_path / "target.png"
    inspection_path = tmp_path / "inspection.png"
    write_image(target_path, np.zeros((12, 10, 3), dtype=np.uint8))
    write_image(inspection_path, np.zeros((10, 12, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="画像サイズ"):
        run_white_dust_gui_job(
            target_path=target_path,
            inspection_path=inspection_path,
            output_dir=tmp_path / "out",
        )


def test_gui_job_accepts_arw_target_and_inspection(monkeypatch, tmp_path):
    target = np.zeros((80, 96, 3), dtype=np.uint16)
    target[:, :] = [18000, 24000, 30000]
    target[35:43, 42:50] = [62000, 62000, 61000]

    inspection = np.zeros_like(target)
    inspection[:, :] = [1500, 1500, 1800]
    yy, xx = np.ogrid[: inspection.shape[0], : inspection.shape[1]]
    disk = (yy - 39) ** 2 + (xx - 46) ** 2 <= 5 * 5
    inspection[disk] = [61000, 60000, 58000]

    target_path = tmp_path / "target.arw"
    inspection_path = tmp_path / "inspection.arw"
    target_path.write_bytes(b"fake target arw")
    inspection_path.write_bytes(b"fake inspection arw")
    calls = []

    def fake_read_image(path, *, raw_half_size=False, raw_output_bps=16):
        calls.append((path.name, raw_half_size, raw_output_bps))
        suffix = path.suffix.lower()
        if suffix == ".arw" and path.name.startswith("target"):
            return ImageData(target, 16, "rgb", path, {"format": "RAW", "raw_suffix": ".arw"})
        if suffix == ".arw" and path.name.startswith("inspection"):
            return ImageData(np.rint(inspection / 257.0).astype(np.uint8), 8, "rgb", path, {"format": "RAW", "raw_suffix": ".arw"})
        return read_image(path)

    monkeypatch.setattr(gui_module, "read_image", fake_read_image)

    result = run_white_dust_gui_job(
        target_path=target_path,
        inspection_path=inspection_path,
        output_dir=tmp_path / "out",
        method="linear",
        detection_long_edge=96,
    )

    assert result.repaired_path.suffix == ".png"
    assert result.repaired_path.exists()
    assert result.target_preview_path.exists()
    assert result.mask_path.exists()
    manifest = result.manifest
    assert manifest["対象写真"] == str(target_path)
    assert manifest["マスク作成用写真"] == str(inspection_path)
    assert manifest["補正方法"] == "linear"
    assert ("target.arw", False, 16) in calls
    assert ("inspection.arw", False, 8) in calls


def test_gui_job_fast_mode_completes_for_large_masks(monkeypatch, tmp_path):
    target = np.zeros((40, 44, 3), dtype=np.uint8)
    target[:, :] = [70, 92, 118]
    target[12:30, 14:32] = [245, 245, 245]

    inspection = np.zeros_like(target)
    inspection[:, :] = [9, 9, 11]
    inspection[12:30, 14:32] = [238, 236, 230]

    target_path = tmp_path / "target.png"
    inspection_path = tmp_path / "inspection.png"
    write_image(target_path, target)
    write_image(inspection_path, inspection)
    monkeypatch.setattr(gui_module, "GUI_FAST_MASK_PIXEL_THRESHOLD", 1)

    result = run_white_dust_gui_job(
        target_path=target_path,
        inspection_path=inspection_path,
        output_dir=tmp_path / "out",
        method="linear",
        detection_long_edge=44,
    )

    assert result.repaired_path.exists()
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["gui_fast_mode"] is True
    repaired = read_image(result.repaired_path).pixels
    mask = read_image(result.mask_path).pixels > 0
    assert np.array_equal(repaired[~mask], target[~mask])


def test_gui_defect_aware_fast_mode_records_linear_fallback(monkeypatch, tmp_path):
    target = np.zeros((40, 44, 3), dtype=np.uint8)
    target[:, :] = [70, 92, 118]
    target[12:30, 14:32] = [245, 245, 245]

    inspection = np.zeros_like(target)
    inspection[:, :] = [9, 9, 11]
    inspection[12:30, 14:32] = [238, 236, 230]

    target_path = tmp_path / "target.png"
    inspection_path = tmp_path / "inspection.png"
    write_image(target_path, target)
    write_image(inspection_path, inspection)
    monkeypatch.setattr(gui_module, "GUI_FAST_MASK_PIXEL_THRESHOLD", 1)

    result = run_white_dust_gui_job(
        target_path=target_path,
        inspection_path=inspection_path,
        output_dir=tmp_path / "out",
        method="defect_aware",
        detection_long_edge=44,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["gui_fast_mode"] is True
    assert metrics["defect_aware"] is True
    assert metrics["gui_fast_fallback_method"] == "linear"
    assert metrics["defect_strategy_counts"] == {"gui_fast_linear_fallback": 1}
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["補正方法"] == "defect_aware"


def test_web_method_selects_include_defect_aware():
    repo_root = gui_module.Path(__file__).resolve().parents[1]
    index_html = (repo_root / "web" / "index.html").read_text(encoding="utf-8")
    white_html = (repo_root / "web" / "white_dust.html").read_text(encoding="utf-8")

    assert '<option value="defect_aware">defect_aware</option>' in index_html
    assert '<option value="defect_aware">defect_aware（自動・高品質）</option>' in white_html


def test_gui_raw_missing_dependency_error_is_actionable(monkeypatch, tmp_path):
    raw_path = tmp_path / "target.arw"
    raw_path.write_bytes(b"fake arw")

    def fake_read_image(_path, *, raw_half_size=False, raw_output_bps=16):
        raise ValueError("RAW input requires optional dependency: rawpy")

    monkeypatch.setattr(gui_module, "read_image", fake_read_image)

    with pytest.raises(ValueError, match="rawpy"):
        _read_gui_image(raw_path, role="補正対象写真", raw_output_bps=16)
