import numpy as np
import pandas as pd
import pytest

from dead_time import (
    PARALYZABLE_DOMAIN_LIMIT,
    correct_dead_time,
    correct_dead_time_dataframe,
)


def test_zero_wide_rate_gives_dtcf_one():
    result = correct_dead_time(
        rpo_cps=100.0,
        rwo_cps=0.0,
        tau_us=0.632,
        calibration_factor_cps_per_mbq=9.36,
    )
    assert result.dtcf == pytest.approx(1.0)
    assert result.rpt_corrected_cps == pytest.approx(100.0)


def test_low_dead_time_gives_dtcf_slightly_above_one():
    result = correct_dead_time(
        rpo_cps=1000.0,
        rwo_cps=1000.0,
        tau_us=0.632,
        calibration_factor_cps_per_mbq=9.36,
    )
    assert result.dtcf > 1.0
    assert result.dtcf < 1.01


def test_solution_satisfies_frezza_equation():
    rwo_cps = 95000.0
    tau_us = 0.632
    result = correct_dead_time(
        rpo_cps=3301.7,
        rwo_cps=rwo_cps,
        tau_us=tau_us,
        calibration_factor_cps_per_mbq=9.36,
    )
    a = rwo_cps * tau_us * 1e-6
    assert result.dtcf * np.exp(-result.dtcf * a) == pytest.approx(1.0)
    assert result.rpt_corrected_cps == pytest.approx(result.dtcf * 3301.7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rpo_cps": -1.0, "rwo_cps": 0.0, "tau_us": 0.632, "calibration_factor_cps_per_mbq": 9.36},
        {"rpo_cps": 1.0, "rwo_cps": -1.0, "tau_us": 0.632, "calibration_factor_cps_per_mbq": 9.36},
        {"rpo_cps": 1.0, "rwo_cps": 0.0, "tau_us": 0.0, "calibration_factor_cps_per_mbq": 9.36},
        {"rpo_cps": 1.0, "rwo_cps": 0.0, "tau_us": 0.632, "calibration_factor_cps_per_mbq": 0.0},
        {"rpo_cps": np.nan, "rwo_cps": 0.0, "tau_us": 0.632, "calibration_factor_cps_per_mbq": 9.36},
    ],
)
def test_invalid_scalar_inputs(kwargs):
    with pytest.raises(ValueError):
        correct_dead_time(**kwargs)


def test_exceeding_one_over_e_raises():
    tau_us = 0.632
    tau_s = tau_us * 1e-6
    rwo_cps = PARALYZABLE_DOMAIN_LIMIT / tau_s
    with pytest.raises(ValueError, match="1/e"):
        correct_dead_time(
            rpo_cps=1.0,
            rwo_cps=rwo_cps,
            tau_us=tau_us,
            calibration_factor_cps_per_mbq=9.36,
        )


def test_max_calibrated_rate_raises():
    with pytest.raises(ValueError, match="max_calibrated"):
        correct_dead_time(
            rpo_cps=1.0,
            rwo_cps=5000.0,
            tau_us=0.632,
            calibration_factor_cps_per_mbq=9.36,
            max_calibrated_rwo_cps=1000.0,
        )


def test_image_correction():
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = correct_dead_time(
        rpo_cps=1000.0,
        rwo_cps=100000.0,
        tau_us=0.632,
        calibration_factor_cps_per_mbq=9.36,
        image_primary=image,
    )
    assert result.corrected_image is not None
    np.testing.assert_allclose(result.corrected_image, image * result.dtcf)


def test_negative_image_values_raise():
    with pytest.raises(ValueError, match="negative"):
        correct_dead_time(
            rpo_cps=1000.0,
            rwo_cps=100000.0,
            tau_us=0.632,
            calibration_factor_cps_per_mbq=9.36,
            image_primary=np.array([1.0, -1.0]),
        )


def test_dataframe_correction_multiple_timepoints():
    data = pd.DataFrame(
        {
            "timepoint": ["D0", "D1"],
            "rpo_cps": [3301.7, 819.3],
            "rwo_cps": [95000.0, 24000.0],
            "tau_us": [0.632, 0.632],
            "CF_cps_per_MBq": [9.36, 9.36],
        }
    )
    corrected = correct_dead_time_dataframe(data)
    assert list(corrected["timepoint"]) == ["D0", "D1"]
    assert np.all(corrected["dtcf"] > 1.0)
    assert np.all(corrected["recovered_activity_mbq"] > data["rpo_cps"] / data["CF_cps_per_MBq"])


def test_dataframe_missing_columns_raise():
    with pytest.raises(ValueError, match="Missing required columns"):
        correct_dead_time_dataframe(pd.DataFrame({"timepoint": ["D0"]}))
