import numpy as np
import pytest

import planar_excretion_fit_balance as pefb
import sang_urine


def _fit_inputs():
    data = sang_urine.load_sang_urine_csv()
    balance = sang_urine.calculate_urine_mass_balance(data)
    imaging = sang_urine.load_imaging_comparison(data["injection_datetime"])
    assert imaging is not None
    fits = pefb.fit_planar_and_excretion(data, balance, imaging)
    return data, balance, imaging, fits


def test_cumulative_excretion_fit_is_zero_at_injection_and_monotonic():
    times_h = np.linspace(0.0, 150.0, 500)
    values_mbq = pefb.cumulative_excretion_biexponential(
        times_h, 6500.0, 0.8, 0.1, 0.005
    )

    assert values_mbq[0] == pytest.approx(0.0)
    assert np.all(np.diff(values_mbq) >= 0.0)


def test_independent_fits_use_injection_equivalent_activities():
    data, balance, imaging, fits = _fit_inputs()

    expected_planar = imaging["planar_ctac_mbq"] / sang_urine.decay_factor(
        imaging["planar_times_h"], data["half_life_h"]
    )
    np.testing.assert_allclose(fits["planar_equivalent_mbq"], expected_planar)
    np.testing.assert_allclose(
        fits["excretion_equivalent_mbq"],
        balance["cumulative_injection_equivalent_mbq"],
    )
    assert fits["planar_rmse_mbq"] < 100.0
    assert fits["excretion_rmse_mbq"] < 250.0


def test_body_fit_is_forced_to_injected_activity_at_time_zero():
    data, _, _, fits = _fit_inputs()

    fitted_at_zero = pefb.planar_biexponential(0.0, *fits["planar_parameters"])

    assert fitted_at_zero == pytest.approx(data["injected_activity_mbq"])
    assert fits["planar_parameters"][0] + fits["planar_parameters"][1] == (
        pytest.approx(data["injected_activity_mbq"])
    )


def test_cross_time_balances_add_measured_and_independently_fitted_components():
    data, _, _, fits = _fit_inputs()
    result = pefb.calculate_cross_time_balances(fits)

    np.testing.assert_allclose(
        result["sum_at_planar_mbq"],
        fits["planar_equivalent_mbq"]
        + result["excretion_fit_at_planar_mbq"],
    )
    np.testing.assert_allclose(
        result["sum_at_excretion_mbq"],
        fits["excretion_equivalent_mbq"]
        + result["planar_fit_at_excretion_mbq"],
    )
    assert len(result["planar_rows"]) == 5
    assert len(result["excretion_rows"]) == 6
    assert result["planar_rows"][0]["closure_ratio"] == pytest.approx(
        result["sum_at_planar_mbq"][0] / data["injected_activity_mbq"]
    )
    assert not result["planar_rows"][1][
        "fit_is_extrapolated_after_last_collection"
    ]
    assert result["planar_rows"][2][
        "fit_is_extrapolated_after_last_collection"
    ]
