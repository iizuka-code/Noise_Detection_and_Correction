from dust_mask_repair.benchmark import evaluate_frequency_guided_quality_case


def test_frequency_guided_quality_benchmark_selected_gradient_and_no_selection():
    selected = evaluate_frequency_guided_quality_case("selected_sky_gradient", width=72, height=56)
    no_selection = evaluate_frequency_guided_quality_case("no_selection_regression", width=72, height=56)

    assert selected["frequency_selected_component_count"] == 1
    assert selected["frequency_analyzed_component_count"] == 1
    assert selected["selected_core_rgb_mae"] < selected["corrupted_selected_core_rgb_mae"]
    assert selected["max_abs_diff_outside_mask"] == 0.0
    assert no_selection["frequency_analyzed_component_count"] == 0
    assert no_selection["max_abs_diff_outside_mask"] == 0.0


def test_frequency_guided_quality_benchmark_many_defects_one_selected():
    result = evaluate_frequency_guided_quality_case("many_defects_one_selected", width=72, height=56)

    assert result["frequency_selected_component_count"] == 1
    assert result["frequency_analyzed_component_count"] == 1
    assert result["selected_core_rgb_mae"] < result["corrupted_selected_core_rgb_mae"]
    assert result["max_abs_diff_outside_mask"] == 0.0
