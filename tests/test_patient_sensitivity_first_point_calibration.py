import numpy as np
import pytest

import patient_sensitivity_first_point_calibration as first_point


def test_calibrate_first_point_locks_one_sensitivity():
    sensitivity, prediction = first_point.calibrate_first_point(
        [100.0, 50.0, 25.0], 20.0
    )

    assert sensitivity == pytest.approx(5.0)
    assert prediction == pytest.approx([20.0, 10.0, 5.0])


def test_prediction_metrics_can_exclude_calibration_point():
    metrics = first_point.prediction_metrics(
        reference_mbq=[20.0, 8.0, 4.0],
        predicted_mbq=[20.0, 10.0, 5.0],
        start_index=1,
    )

    assert metrics["n_evaluated"] == 2
    assert metrics["bias_percent"] == pytest.approx(25.0)
    assert metrics["mare_percent"] == pytest.approx(25.0)
    assert metrics["rmse_percent"] == pytest.approx(25.0)
    assert metrics["maximum_absolute_error_percent"] == pytest.approx(25.0)


def test_calibration_rejects_invalid_values():
    with pytest.raises(ValueError):
        first_point.calibrate_first_point([100.0, np.nan], 20.0)
    with pytest.raises(ValueError):
        first_point.calibrate_first_point([100.0, 50.0], 0.0)
