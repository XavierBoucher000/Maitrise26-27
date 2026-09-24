import numpy as np
import pytest

import patient_sensitivity_multicycle_v3 as v3


def _synthetic_observations():
    observations = {}
    for dataset_index, key in enumerate(v3.DATASETS):
        x1 = np.array([-0.4, 0.0, 0.4], dtype=float) + 0.05 * dataset_index
        x4 = np.array([0.05, 0.10, 0.15], dtype=float)
        target = np.exp(0.6 - 0.8 * x1 + 0.4 * x4)
        rate = target * np.array([1000.0, 700.0, 400.0])
        records = [
            {
                "dataset_key": key,
                "patient_id": v3.DATASETS[key].patient_id,
                "cycle_id": v3.DATASETS[key].cycle_id,
                "dataset_label": v3.DATASETS[key].label,
                "timepoint": f"Day{index}",
                "planar_day": float(index),
                "planar_datetime": f"planar-{index}",
                "qspect_datetime": f"qspect-{index}",
                "crop_top": 10,
                "crop_bottom": 20,
                "profile_match_accepted": True,
            }
            for index in range(3)
        ]
        observations[key] = {
            "records": records,
            "features": {
                v3.LOCKED_FEATURE: x1,
                v3.ASYMMETRY_FEATURE: x4,
                v3.LOWER_FEATURE: x1 + 0.1,
                v3.UPPER_FEATURE: x1 - 0.1,
                v3.TEW_FRACTION_FEATURE: np.full(3, 0.6),
            },
            "target_sensitivity": target,
            "tew_rate": rate,
            "reference_activity": rate / target,
        }
    return observations


def test_default_splits_include_six_pairwise_and_primary_patient_holdout():
    splits = v3.default_split_definitions()
    pairwise = [split for split in splits if split.split_type == "pairwise_cycle_transfer"]
    assert len(pairwise) == 6
    assert len(splits) == 10
    primary = next(split for split in splits if split.split_id == "loco_holdout_p8_june")
    assert primary.training_datasets == ("p11_may", "p11_july")
    assert primary.test_datasets == ("p8_june",)


def test_validate_split_rejects_training_test_overlap():
    with pytest.raises(ValueError, match="must be disjoint"):
        v3.validate_split(("p11_may", "p8_june"), ("p8_june",))


def test_locked_model_recovers_exact_log_linear_relation_on_external_cycle():
    observations = _synthetic_observations()
    result = v3.fit_and_evaluate(
        ("p11_may", "p11_july"),
        ("p8_june",),
        observations,
        split_id="synthetic",
    )
    predictions = [
        row
        for row in result["predictions"]
        if row["model"] == v3.MODEL_LOCKED
    ]
    assert len(predictions) == 3
    # The synthetic target also contains an AP/PA term, but x4 is identical
    # across cycles and linearly related to x1 within each cycle.  The pooled
    # training data therefore gives a deterministic near-exact transfer.
    assert max(abs(row["activity_error_percent"]) for row in predictions) < 1.0


def test_two_predictor_model_requires_residual_degrees_of_freedom():
    observations = _synthetic_observations()
    training = v3._combine_observations(observations, ("p8_june",))
    fit = v3.fit_model(v3.MODEL_ASYMMETRY, training)
    assert fit["status"] == "not_estimable"
    assert "must exceed 3" in fit["reason"]


def test_two_predictor_model_is_fitted_when_two_cycles_are_pooled():
    observations = _synthetic_observations()
    training = v3._combine_observations(
        observations, ("p11_may", "p11_july")
    )
    fit = v3.fit_model(v3.MODEL_ASYMMETRY, training)
    assert fit["status"] == "fitted"
    assert fit["rank"] == 3
