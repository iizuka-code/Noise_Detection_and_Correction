import json

import numpy as np

from dust_mask_repair import RepairConfig, repair_image


def test_defect_aware_router_handles_mixed_strategies_and_debug_outputs(tmp_path):
    height, width = 86, 116
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = [82, 110, 138]
    stripe = ((np.arange(width) // 4) % 2).astype(bool)
    image[:, 76:, 0] = np.where(stripe[76:][None, :], 210, 55)
    image[:, 76:, 1] = np.where(stripe[76:][None, :], 170, 80)
    image[:, 76:, 2] = np.where(stripe[76:][None, :], 80, 150)

    damaged = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[12, 12] = 255
    mask[24:34, 20:30] = 255
    mask[48, 15:50] = 255
    mask[28:39, 88:99] = 255
    damaged[mask > 0] = [255, 255, 255]

    result = repair_image(
        damaged,
        mask,
        RepairConfig(
            method="defect_aware",
            mask_channel="grayscale",
            dilate_radius=0,
            feather_radius=0,
            padding=18,
            debug_dir=tmp_path,
            grain_reinject_strength=0.2,
        ),
    )

    strategy_counts = result.metrics["defect_strategy_counts"]
    assert result.metrics["defect_component_count"] == 4
    assert strategy_counts["tiny_local"] == 1
    assert strategy_counts["fast_inpaint"] == 1
    assert strategy_counts["directional"] == 1
    assert strategy_counts["patch"] == 1
    assert result.metrics["small_local_component_count"] == 1
    assert result.metrics["fast_inpaint_component_count"] == 1
    assert result.metrics["directional_component_count"] == 1
    assert result.metrics["patch_component_count"] == 1
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0
    assert np.array_equal(result.repaired_image[result.soft_mask <= 0.0], damaged[result.soft_mask <= 0.0])

    components_path = tmp_path / "defect_components.json"
    summary_path = tmp_path / "defect_strategy_summary.json"
    assert components_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["defect_component_count"] == 4
    assert summary["defect_strategy_counts"] == strategy_counts
    assert result.debug_paths["defect_components"] == str(components_path)
    assert result.debug_paths["defect_strategy_summary"] == str(summary_path)
