import numpy as np

from dust_mask_repair import RepairConfig, repair_image
from dust_mask_repair.benchmark import evaluate_defect_aware_quality_case, make_defect_aware_quality_case
from dust_mask_repair.repair import _guard_repair_candidate


def _defect_config(**overrides):
    values = {
        "method": "defect_aware",
        "mask_channel": "grayscale",
        "threshold": 0.5,
        "dilate_radius": 1,
        "feather_radius": 1,
        "strength": 1.0,
        "padding": 12,
        "grain_reinject_strength": 0.0,
        "color_match_strength": 0.0,
        "collect_debug_images": True,
    }
    values.update(overrides)
    return RepairConfig(**values)


def test_defect_aware_core_uses_full_candidate_alpha():
    image = np.zeros((34, 34, 3), dtype=np.uint8)
    image[:, :] = [122, 96, 145]
    damaged = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[14:19, 15:20] = 255
    damaged[mask > 0] = [8, 8, 8]

    result = repair_image(damaged, mask, _defect_config())
    core = result.core_mask

    assert result.metrics["defect_core_alpha_min"] == 1.0
    assert result.metrics["defect_core_alpha_mean"] == 1.0
    assert result.metrics["defect_core_alpha_max"] == 1.0
    assert result.metrics["defect_core_alpha_below_full_count"] == 0
    assert np.array_equal(result.repaired_image[core], result.debug_images["candidate_after_guard"][core])
    assert not np.array_equal(result.repaired_image[core], damaged[core])
    assert result.metrics["max_abs_diff_outside_mask"] == 0.0


def test_defect_aware_guard_core_reject_uses_fallback_not_original_defect():
    roi = np.full((11, 11, 3), 0.62, dtype=np.float32)
    roi[5, 5] = [0.02, 0.02, 0.02]
    repair_mask = np.zeros(roi.shape[:2], dtype=bool)
    repair_mask[5, 5] = True
    candidate = roi.copy()
    candidate[5, 5] = [0.0, 0.0, 1.0]

    guarded, rejected, stats = _guard_repair_candidate(roi, candidate, repair_mask, repair_mask)

    assert rejected[5, 5]
    assert stats["guard_rejected_core_pixel_count"] == 1
    assert stats["guard_core_fallback_success_count"] == 1
    assert stats["guard_core_unrepaired_pixel_count"] == 0
    assert np.mean(np.abs(guarded[5, 5] - np.array([0.62, 0.62, 0.62], dtype=np.float32))) < 0.08
    assert np.mean(np.abs(guarded[5, 5] - roi[5, 5])) > 0.35


def test_tone_guided_repair_improves_mottled_dark_dust_case():
    clean, damaged, mask = make_defect_aware_quality_case("mottled_background_dark_dust", width=72, height=56)
    common = {
        "method": "defect_aware",
        "mask_channel": "grayscale",
        "dilate_radius": 0,
        "feather_radius": 0,
        "padding": 18,
        "max_component_area": 5000,
        "edge_guided_enabled": False,
    }
    enabled = repair_image(damaged, mask, RepairConfig(**common))
    disabled = repair_image(damaged, mask, RepairConfig(**common, tone_guided_enabled=False))
    inside = mask > 0
    enabled_mae = float(np.mean(np.abs(enabled.repaired_image[inside].astype(int) - clean[inside].astype(int))))
    disabled_mae = float(np.mean(np.abs(disabled.repaired_image[inside].astype(int) - clean[inside].astype(int))))
    corrupted_mae = float(np.mean(np.abs(damaged[inside].astype(int) - clean[inside].astype(int))))
    benchmark = evaluate_defect_aware_quality_case("mottled_background_dark_dust", width=72, height=56)

    assert enabled.metrics["max_abs_diff_outside_mask"] == 0.0
    assert enabled.metrics["defect_core_alpha_below_full_count"] == 0
    assert enabled.metrics["tone_guided_component_count"] > 0
    assert enabled_mae < disabled_mae
    assert enabled_mae < corrupted_mae
    assert benchmark["tone_guided_enabled_isolated_mean_abs_error_inside_mask"] < benchmark["tone_guided_disabled_mean_abs_error_inside_mask"]
    assert benchmark["corrupted_improvement_ratio"] > 0.35


def test_tone_guided_dark_dust_case_is_deterministic():
    clean, damaged, mask = make_defect_aware_quality_case("mottled_background_dark_dust", width=64, height=48)
    cfg = _defect_config(dilate_radius=0, feather_radius=0, collect_debug_images=False)

    first = repair_image(damaged, mask, cfg)
    second = repair_image(damaged, mask, cfg)

    assert np.array_equal(first.repaired_image, second.repaired_image)
    assert np.array_equal(first.repaired_image[first.soft_mask <= 0.0], damaged[first.soft_mask <= 0.0])
    assert first.metrics["tone_guided_component_count"] == second.metrics["tone_guided_component_count"]
    assert first.metrics["max_abs_diff_outside_mask"] == 0.0
    assert clean.shape == first.repaired_image.shape


def test_guard_shell_does_not_restore_masked_source_pixel():
    roi = np.full((9, 9, 3), 0.62, dtype=np.float32)
    roi[4, 4] = [0.60, 0.61, 0.60]
    repair_mask = np.zeros(roi.shape[:2], dtype=bool)
    repair_mask[4, 4] = True
    core_mask = np.zeros(roi.shape[:2], dtype=bool)
    candidate = roi.copy()
    candidate[4, 4] = [0.0, 0.0, 1.0]

    guarded, rejected, stats = _guard_repair_candidate(roi, candidate, repair_mask, core_mask)

    assert rejected[4, 4]
    assert stats["guard_rejected_shell_pixel_count"] == 1
    assert not np.array_equal(guarded[4, 4], roi[4, 4])
    assert np.mean(np.abs(guarded[4, 4] - np.array([0.62, 0.62, 0.62], dtype=np.float32))) < 0.05


def test_defect_aware_repair_does_not_trust_masked_source_color_when_close_to_context():
    clean = np.full((34, 34, 3), 158, dtype=np.uint8)
    damaged = clean.copy()
    damaged[15:19, 15:19] = [145, 160, 150]
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    mask[15:19, 15:19] = 255

    result = repair_image(damaged, mask, _defect_config(dilate_radius=0, feather_radius=0))
    inside = mask > 0

    assert not np.array_equal(result.repaired_image[inside], damaged[inside])
    assert np.mean(np.abs(result.repaired_image[inside].astype(int) - clean[inside].astype(int))) < 4.0
