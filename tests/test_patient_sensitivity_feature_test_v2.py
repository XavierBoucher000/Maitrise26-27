import numpy as np
import pytest

import patient_sensitivity_feature_test_v2 as feature_test


def test_loo_activity_error_is_zero_for_exact_log_linear_relation():
    feature = np.arange(5, dtype=float)
    target = 2.0 * np.exp(0.2 * feature)
    result = feature_test.leave_one_out_log_linear(feature, target)

    np.testing.assert_allclose(result["predictions"], target, rtol=1e-12)
    assert result["activity_mare_percent"] == pytest.approx(0.0, abs=1e-10)
    assert result["slope_sign_consistent"]


def test_nested_selection_uses_only_outer_training_values():
    target = 3.0 * np.exp(0.1 * np.arange(5, dtype=float))
    features = {
        "good": np.arange(5, dtype=float),
        "bad": np.array([3.0, 1.0, 4.0, 0.0, 2.0]),
    }
    result = feature_test.nested_univariate_feature_selection(
        features, target, candidate_names=("good", "bad")
    )

    assert result["selected_features"] == ["good"] * 5
    assert result["activity_mare_percent"] == pytest.approx(0.0, abs=1e-10)


def test_prediction_metrics_reports_inverse_effect_on_activity():
    metrics = feature_test._prediction_metrics(
        np.array([2.0]), np.array([2.2])
    )
    assert metrics["sensitivity_bias_percent"] == pytest.approx(10.0)
    assert metrics["activity_bias_percent"] == pytest.approx(-100.0 / 11.0)
