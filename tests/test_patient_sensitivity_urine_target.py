import numpy as np

import patient_sensitivity_multicycle_v3 as v3
import patient_sensitivity_urine_target as urine


def _observation(rate_scale: float = 1.0):
    activity = np.array([100.0, 80.0, 60.0])
    x = np.array([-0.4, 0.0, 0.4])
    sensitivity = np.exp(0.5 - 0.2 * x)
    rate = rate_scale * sensitivity * activity
    features = {
        v3.LOCKED_FEATURE: x,
        v3.RAW_PHOTO_GENERAL_FEATURE: -x,
        v3.ADJACENT_FEATURE: x + 0.1,
        v3.PHOTO_ALL_FRACTION_FEATURE: np.array([0.2, 0.25, 0.3]),
        v3.TEW_FRACTION_FEATURE: np.array([0.6, 0.65, 0.7]),
        v3.LOWER_FEATURE: x + 0.2,
        v3.UPPER_FEATURE: x - 0.2,
        v3.ASYMMETRY_FEATURE: np.array([0.03, 0.05, 0.07]),
        v3.ALL_WINDOW_ASYMMETRY_FEATURE: np.array([0.04, 0.06, 0.08]),
    }
    return {
        "records": [
            {
                "dataset_key": "synthetic",
                "patient_id": "p",
                "cycle_id": "c",
                "dataset_label": "synthetic",
                "timepoint": f"J{index}",
            }
            for index in range(3)
        ],
        "features": features,
        "measurement_rate": rate,
        "reference_activity": activity,
        "target_sensitivity": rate / activity,
    }


def test_raw_mode_excludes_tew_fraction_and_two_predictor_models_are_not_estimable():
    result = urine.fit_and_test_mode(
        v3.MODE_RAW_FULL_GM,
        _observation(),
        _observation(1.05),
    )
    assert v3.MODEL_TEW_FRACTION not in result["fits"]
    assert result["fits"][v3.MODEL_ASYMMETRY]["status"] == "not_estimable"
    assert result["fits"][v3.MODEL_RAW_ASYMMETRY]["status"] == "not_estimable"
    assert len(
        [
            row
            for row in result["predictions"]
            if row["model"] == v3.MODEL_CONSTANT
        ]
    ) == 3


def test_simple_zero_origin_transfer_recovers_exact_activity():
    activity_train = np.array([100.0, 80.0, 60.0])
    activity_test = np.array([95.0, 75.0, 55.0])
    train = {
        "reference_activity": activity_train,
        "simple_rates": {
            name: 2.5 * activity_train
            for name in urine.SIMPLE_METHOD_ORDER
        },
    }
    test = {
        "reference_activity": activity_test,
        "simple_rates": {
            name: 2.5 * activity_test
            for name in urine.SIMPLE_METHOD_ORDER
        },
    }
    result = urine.simple_zero_origin_transfer(train, test)
    assert result[urine.SIMPLE_PHOTO_AP_PA]["sensitivity_cps_per_mbq"] == 2.5
    assert result[urine.SIMPLE_PHOTO_AP_PA]["activity_mare_percent"] < 1e-12


def test_j0_calibration_uses_only_first_point_and_scores_followups():
    activity = np.array([100.0, 80.0, 60.0])
    dataset = {
        "reference_activity": activity,
        "simple_rates": {
            name: 3.0 * activity for name in urine.SIMPLE_METHOD_ORDER
        },
    }
    result = urine.j0_calibration_results(dataset)
    method = result[urine.SIMPLE_PHOTO_AP_PA]
    assert method["sensitivity_j0_cps_per_mbq"] == 3.0
    assert method["followup_metrics"]["activity_mare_percent"] < 1e-12


def test_urine_timing_scenarios_bracket_linear_interpolation():
    data = {"injected_activity_mbq": 100.0, "half_life_h": 10.0}
    balance = {
        "times_h": np.array([2.0, 4.0]),
        "interval_injection_equivalent_mbq": np.array([20.0, 20.0]),
    }
    scenarios = urine.urine_target_scenarios(
        np.array([1.0, 3.0, 4.0]), data, balance
    )
    completed = scenarios["completed_only_activity_mbq"]
    linear = scenarios["linear_interval_activity_mbq"]
    interval_complete = scenarios["current_interval_complete_activity_mbq"]
    assert np.all(completed >= linear)
    assert np.all(linear >= interval_complete)
    assert completed[-1] == interval_complete[-1]


def test_fitted_excretion_target_marks_only_times_after_last_collection():
    collection_times = np.array([2.0, 4.0, 12.0, 24.0, 36.0, 48.0])
    parameters = np.array([850.0, 0.7, 0.12, 0.008])
    cumulative = urine.cumulative_excretion_biexponential(
        collection_times, *parameters
    )
    fit = urine.fit_cumulative_excretion_target(
        np.array([1.0, 24.0, 72.0]),
        {"injected_activity_mbq": 1000.0, "half_life_h": 160.0},
        {
            "times_h": collection_times,
            "cumulative_injection_equivalent_mbq": cumulative,
        },
        bootstrap_samples=0,
    )
    assert fit["rmse_mbq"] < 1e-4
    assert fit["is_extrapolated"].tolist() == [False, False, True]
    assert np.all(fit["reference_activity_mbq"] > 0.0)
    assert fit["hours_after_last_collection"].tolist() == [0.0, 0.0, 24.0]


def test_apply_fitted_target_and_subset_update_every_model_target():
    collection_times = np.array([2.0, 4.0, 12.0, 24.0, 36.0, 48.0])
    cumulative = urine.cumulative_excretion_biexponential(
        collection_times, 850.0, 0.7, 0.12, 0.008
    )
    records = [
        {
            "timepoint": label,
            "hours_after_injection": time_h,
            "urine_remaining_activity_mbq": activity,
        }
        for label, time_h, activity in (
            ("J0", 1.0, 990.0),
            ("J1", 24.0, 500.0),
            ("J3", 72.0, 250.0),
        )
    ]
    modes = {mode: _observation() for mode in urine.MODE_ORDER}
    dataset = {
        "spec": urine.TRAINING_DATASET,
        "records": records,
        "reference_activity": np.array([990.0, 500.0, 250.0]),
        "mode_observations": modes,
        "simple_rates": {
            name: np.array([100.0, 80.0, 60.0])
            for name in urine.SIMPLE_METHOD_ORDER
        },
        "urine_data": {
            "injected_activity_mbq": 1000.0,
            "half_life_h": 160.0,
        },
        "urine_balance": {
            "times_h": collection_times,
            "cumulative_injection_equivalent_mbq": cumulative,
        },
        "timing_scenarios": {"example": np.array([1.0, 2.0, 3.0])},
    }
    fitted = urine.apply_fitted_excretion_target(
        dataset, bootstrap_samples=0
    )
    assert fitted["records"][-1]["excretion_target_status"] == "extrapolated"
    for mode in urine.MODE_ORDER:
        observation = fitted["mode_observations"][mode]
        assert np.allclose(
            observation["target_sensitivity"],
            observation["measurement_rate"] / fitted["reference_activity"],
        )
    subset = urine.subset_urine_dataset(fitted, 2)
    assert len(subset["records"]) == 2
    assert subset["excretion_fit"]["is_extrapolated"].tolist() == [False, False]


def test_sensitivity_stability_is_reported_separately_by_dataset_and_mode():
    training_modes = {mode: _observation() for mode in urine.MODE_ORDER}
    test_modes = {mode: _observation(1.1) for mode in urine.MODE_ORDER}
    training = {"spec": urine.TRAINING_DATASET, "mode_observations": training_modes}
    test = {"spec": urine.TEST_DATASET, "mode_observations": test_modes}
    rows = urine.sensitivity_stability_rows(training, test)
    assert len(rows) == 2 * len(urine.MODE_ORDER)
    assert {row["dataset_role"] for row in rows} == {"training", "test"}
    assert all(row["cv_percent"] >= 0.0 for row in rows)


def test_methodology_states_urine_target_and_no_qspect_input():
    simple = {
        name: {
            "sensitivity_cps_per_mbq": 2.0,
            "activity_mare_percent": 5.0,
            "activity_bias_percent": 1.0,
            "maximum_activity_error_percent": 8.0,
        }
        for name in urine.SIMPLE_METHOD_ORDER
    }
    metrics = [
        {
            "rate_mode": mode,
            "fit_status": "fitted",
            "model": v3.MODEL_CONSTANT,
            "activity_mare_percent": 5.0,
        }
        for mode in urine.MODE_ORDER
    ]
    report = urine.build_methodology_report(simple, metrics)
    assert "A_corps,urine(t)" in report
    assert "Aucun CT, ACCT, Q/SPECT" in report
    assert "P8 juin" in report
    assert "non estimables" in report
    assert "calibration J0" in report
