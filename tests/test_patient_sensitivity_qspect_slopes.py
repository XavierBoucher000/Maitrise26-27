import numpy as np

from models.patient_sensitivity_qspect_slopes import linear_sensitivity_fit


def test_free_intercept_recovers_exact_line() -> None:
    activity = np.asarray([7000.0, 5000.0, 3000.0])
    rate = 2.0 * activity + 500.0

    fit = linear_sensitivity_fit(activity, rate, "free_intercept")

    assert np.isclose(fit["slope_cps_per_mbq"], 2.0)
    assert np.isclose(fit["intercept_cps"], 500.0)
    assert np.isclose(fit["mare_activity_percent"], 0.0)


def test_j0_anchor_passes_through_j0_without_forcing_origin() -> None:
    activity = np.asarray([7000.0, 5000.0, 3000.0])
    rate = np.asarray([14500.0, 10300.0, 6600.0])

    fit = linear_sensitivity_fit(activity, rate, "anchored_j0")
    slope = float(fit["slope_cps_per_mbq"])
    intercept = float(fit["intercept_cps"])

    assert np.isclose(slope * activity[0] + intercept, rate[0])
    assert not np.isclose(intercept, 0.0)


def test_origin_fit_keeps_zero_intercept() -> None:
    activity = np.asarray([7000.0, 5000.0, 3000.0])
    rate = np.asarray([14500.0, 10300.0, 6600.0])

    fit = linear_sensitivity_fit(activity, rate, "origin")

    assert fit["intercept_cps"] == 0.0
    assert fit["fit_method"] == "origin"
