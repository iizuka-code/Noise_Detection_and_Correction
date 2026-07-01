import json

import numpy as np
import pytest

from dust_mask_repair import RepairConfig, repair_image
from dust_mask_repair.frequency_repair import describe_frequency_context
from dust_mask_repair.gui import run_white_dust_gui_job
from dust_mask_repair.io import read_image, write_image


def _gradient_case(width=54, height=42):
    yy, xx = np.indices((height, width), dtype=np.float32)
    clean = np.dstack((0.20 + xx * 0.004, 0.32 + yy * 0.003, 0.64 - xx * 0.002))
    damaged = np.rint(np.clip(clean, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 2, width // 3] = 255
    mask[height // 2, width - width // 3] = 255
    damaged[mask > 0] = [255, 255, 255]
    scope = np.zeros((height, width), dtype=np.uint8)
    scope[height // 2 - 3 : height // 2 + 4, width // 3 - 3 : width // 3 + 4] = 255
    return damaged, mask, scope


def _freq_config(**overrides):
    values = {
        "method": "defect_aware",
        "mask_channel": "grayscale",
        "dilate_radius": 0,
        "feather_radius": 0,
        "padding": 14,
        "grain_reinject_strength": 0.0,
        "frequency_guided_enabled": True,
    }
    values.update(overrides)
    return RepairConfig(**values)


def test_no_frequency_scope_is_no_selection_regression():
    damaged, mask, _scope = _gradient_case()
    base = repair_image(damaged, mask, _freq_config(frequency_guided_enabled=False))
    no_selection = repair_image(damaged, mask, _freq_config())

    assert np.array_equal(no_selection.repaired_image, base.repaired_image)
    assert no_selection.metrics["frequency_guided_enabled"] is False
    assert no_selection.metrics["frequency_analyzed_component_count"] == 0
    assert no_selection.metrics["max_abs_diff_outside_mask"] == 0.0


def test_frequency_scope_mask_size_mismatch_is_clear_error():
    damaged, mask, _scope = _gradient_case()
    bad_scope = np.zeros((damaged.shape[0] - 1, damaged.shape[1]), dtype=np.uint8)

    with pytest.raises(ValueError, match="frequency scope mask dimensions differ"):
        repair_image(damaged, mask, _freq_config(), frequency_scope_mask=bad_scope)


def test_frequency_guided_selects_only_overlapping_component_and_debug_outputs(tmp_path):
    damaged, mask, scope = _gradient_case()
    result = repair_image(damaged, mask, _freq_config(debug_dir=tmp_path), frequency_scope_mask=scope)

    assert result.metrics["frequency_guided_enabled"] is True
    assert result.metrics["frequency_scope_mask_pixel_count"] == int(np.count_nonzero(scope))
    assert result.metrics["frequency_selected_component_count"] == 1
    assert result.metrics["frequency_analyzed_component_count"] == 1
    assert result.metrics["frequency_selected_core_pixel_count"] == 1
    assert result.metrics["frequency_pattern_counts"]["smooth_gradient"] == 1
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0
    assert (tmp_path / "frequency_scope_mask.png").exists()
    assert (tmp_path / "frequency_selected_core_mask.png").exists()
    assert (tmp_path / "frequency_selected_overlay.png").exists()
    assert (tmp_path / "frequency_pattern_map.png").exists()
    payload = json.loads((tmp_path / "frequency_components.json").read_text(encoding="utf-8"))
    assert payload["component_count"] == 1
    assert payload["components"][0]["frequency_guided_selected"] is True


def test_frequency_descriptor_uses_rgb_directional_energy_and_known_pairs_only():
    height, width = 36, 42
    yy, xx = np.indices((height, width), dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[:, :, 0] = np.where(xx < width // 2, 0.15, 0.85)
    rgb[:, :, 1] = np.where(xx < width // 2, 0.75, 0.20)
    rgb[:, :, 2] = 0.45
    core = np.zeros((height, width), dtype=bool)
    core[height // 2 - 1 : height // 2 + 2, width // 2] = True
    repair = core.copy()
    corrupted = rgb.copy()
    corrupted[core] = [1.0, 1.0, 1.0]

    descriptor = describe_frequency_context(corrupted, core, repair, _freq_config())

    assert descriptor["valid"] is True
    assert descriptor["pattern"] == "directional"
    assert descriptor["directional_energies"]["horizontal"] > descriptor["directional_energies"]["vertical"]
    assert descriptor["anisotropy"] > 0.35


def test_frequency_guided_preserves_rgba_alpha_and_mask_outside():
    damaged, mask, scope = _gradient_case(width=40, height=34)
    alpha = np.full(damaged.shape[:2], 173, dtype=np.uint8)
    rgba = np.dstack([damaged, alpha])

    result = repair_image(rgba, mask, _freq_config(), frequency_scope_mask=scope)

    assert result.repaired_image.shape[2] == 4
    assert np.array_equal(result.repaired_image[:, :, 3], alpha)
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0


def test_frequency_guided_gui_fast_mode_overrides_only_selected_component(monkeypatch, tmp_path):
    from dust_mask_repair import gui as gui_module

    target = np.zeros((48, 60, 3), dtype=np.uint8)
    target[:, :] = [70, 90, 120]
    inspection = np.zeros_like(target)
    inspection[:, :] = [8, 8, 10]
    defect = np.zeros(target.shape[:2], dtype=np.uint8)
    defect[20:23, 20:23] = 255
    defect[22:25, 42:45] = 255
    target[defect > 0] = [245, 245, 245]
    inspection[defect > 0] = [238, 236, 230]
    scope = np.zeros(target.shape[:2], dtype=np.uint8)
    scope[18:26, 18:26] = 255
    target_path = tmp_path / "target.png"
    inspection_path = tmp_path / "inspection.png"
    scope_path = tmp_path / "scope.png"
    write_image(target_path, target)
    write_image(inspection_path, inspection)
    write_image(scope_path, scope)
    monkeypatch.setattr(gui_module, "GUI_FAST_MASK_PIXEL_THRESHOLD", 1)

    result = run_white_dust_gui_job(
        target_path=target_path,
        inspection_path=inspection_path,
        output_dir=tmp_path / "out",
        method="defect_aware",
        detection_long_edge=60,
        frequency_scope_mask_path=scope_path,
        grain_strength=0.0,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["gui_fast_mode"] is True
    assert metrics["frequency_guided_enabled"] is True
    assert metrics["frequency_selected_component_count"] == 1
    assert metrics["frequency_analyzed_component_count"] == 1
    assert metrics["frequency_fast_mode_override_count"] == 1
    assert (result.run_dir / "frequency_scope_mask.png").exists()
    assert (result.run_dir / "frequency_selected_core_mask.png").exists()
    repaired = read_image(result.repaired_path).pixels
    assert np.array_equal(repaired[0, 0], target[0, 0])
