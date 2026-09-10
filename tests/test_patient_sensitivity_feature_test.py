import numpy as np
import pytest

import patient_sensitivity_feature_test as feature_test


def test_log_linear_loo_beats_constant_for_exact_exponential_relation():
    feature = np.arange(5, dtype=float)
    target = 2.0 * np.exp(0.2 * feature)
    result = feature_test.leave_one_out_log_linear(feature, target)

    np.testing.assert_allclose(result["predictions"], target, rtol=1e-12)
    assert result["mare_percent"] == pytest.approx(0.0, abs=1e-10)
    assert result["mare_percent"] < result["baseline_mare_percent"]


def test_log_linear_loo_rejects_nonpositive_target():
    with pytest.raises(ValueError, match="positive"):
        feature_test.leave_one_out_log_linear([0, 1, 2], [1, 0, 2])
