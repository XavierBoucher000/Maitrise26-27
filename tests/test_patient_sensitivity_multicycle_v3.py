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
                v3.RAW_PHOTO_GENERAL_FEATURE: -x1,
                v3.ADJACENT_FEATURE: x1 + 0.2,
                v3.PHOTO_ALL_FRACTION_FEATURE: np.array([0.20, 0.22, 0.24]),
                v3.ALL_WINDOW_ASYMMETRY_FEATURE: 1.2 * x4,
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


def test_raw_full_image_features_match_simple_ap_pa_sums():
    values = {
        "Lower Scatter": (10.0, 20.0),
        "Photopeak": (100.0, 80.0),
        "Upper Scatter": (5.0, 5.0),
        "Low Energy Scatter": (400.0, 320.0),
    }
    images = []
    for energy, (ap, pa) in values.items():
        images.extend(
            [
                {"energy_window": energy, "view": "AP", "image": np.array([[ap]])},
                {"energy_window": energy, "view": "PA", "image": np.array([[pa]])},
            ]
        )
    for image in images:
        image["actual_frame_duration_ms"] = 1000
        image["energy_window_lower_limit"] = 0.0
        image["energy_window_upper_limit"] = 1.0
    features = v3.raw_full_image_features({"images": images})
    assert features[v3.RAW_PHOTO_GENERAL_FEATURE] == pytest.approx(
        np.log(180.0 / 720.0)
    )
    assert features[v3.ADJACENT_FEATURE] == pytest.approx(
        np.log(40.0 / 180.0)
    )
    assert features[v3.PHOTO_ALL_FRACTION_FEATURE] == pytest.approx(
        180.0 / 940.0
    )
    assert features[v3.ALL_WINDOW_ASYMMETRY_FEATURE] == pytest.approx(
        abs(0.5 * np.log(515.0 / 425.0))
    )
    assert features[v3.RAW_FULL_PHOTOPEAK_RATE] == pytest.approx(180.0)
    assert features[v3.RAW_FULL_ALL_WINDOWS_RATE] == pytest.approx(940.0)
    assert features[v3.RAW_FULL_GM_RATE] == pytest.approx(np.sqrt(100.0 * 80.0))
    assert features[v3.TEW_FULL_GM_RATE] == pytest.approx(
        np.sqrt(92.5 * 67.5)
    )
    assert features[v3.TEW_FULL_AP_PA_RATE] == pytest.approx(160.0)


def test_mode_matched_features_follow_crop_and_ap_pa_geometry():
    values = {
        "Lower Scatter": (
            np.array([[20.0], [20.0]]),
            np.array([[20.0], [20.0]]),
        ),
        "Photopeak": (
            np.array([[100.0], [100.0]]),
            np.array([[100.0], [400.0]]),
        ),
        "Upper Scatter": (
            np.array([[10.0], [10.0]]),
            np.array([[10.0], [10.0]]),
        ),
        "Low Energy Scatter": (
            np.array([[50.0], [20.0]]),
            np.array([[50.0], [20.0]]),
        ),
    }
    images = []
    for energy, (ap, pa) in values.items():
        for view, image in (("AP", ap), ("PA", pa)):
            images.append(
                {
                    "energy_window": energy,
                    "view": view,
                    "image": image,
                    "actual_frame_duration_ms": 1000,
                    "energy_window_lower_limit": 0.0,
                    "energy_window_upper_limit": 1.0,
                }
            )

    matched = v3.mode_matched_measurements(
        {"images": images}, crop_bounds=(0, 1)
    )
    crop = matched["features_by_geometry"][v3.GEOMETRY_CROP_GM]
    full_gm = matched["features_by_geometry"][v3.GEOMETRY_FULL_GM]
    full_ap_pa = matched["features_by_geometry"][v3.GEOMETRY_FULL_AP_PA]

    assert crop[v3.LOWER_FEATURE] == pytest.approx(np.log(40.0 / 200.0))
    assert full_gm[v3.LOWER_FEATURE] == pytest.approx(np.log(40.0 / 300.0))
    assert full_ap_pa[v3.LOWER_FEATURE] == pytest.approx(np.log(80.0 / 700.0))
    assert len(
        {
            round(crop[v3.LOWER_FEATURE], 8),
            round(full_gm[v3.LOWER_FEATURE], 8),
            round(full_ap_pa[v3.LOWER_FEATURE], 8),
        }
    ) == 3


def test_model_methodology_document_covers_every_model_and_common_target():
    document = v3.build_model_methodology_document()
    for index in range(10):
        assert f"M{index} —" in document
    assert "S_eff(m) = R_planaire(m) / A_QSPECT" in document
    assert "A_planaire prédite = R_planaire(m) / S_eff(m) prédite" in document
    for rate_mode in v3.RATE_MODES:
        assert rate_mode in document
    assert "Aucun de ces modèles" in document


def test_rate_mode_selection_changes_target_but_preserves_reference_activity():
    reference = np.array([100.0, 50.0])
    base = {
        "d": {
            "records": [{}, {}],
            "features": {},
            "tew_rate": np.array([200.0, 100.0]),
            "target_sensitivity": np.array([2.0, 2.0]),
            "reference_activity": reference,
            "rate_modes": {
                v3.MODE_TEW_CROP_GM: {
                    "measurement_rate": np.array([200.0, 100.0]),
                    "target_sensitivity": np.array([2.0, 2.0]),
                },
                v3.MODE_RAW_CROP_GM: {
                    "measurement_rate": np.array([300.0, 150.0]),
                    "target_sensitivity": np.array([3.0, 3.0]),
                },
                v3.MODE_TEW_FULL_GM: {
                    "measurement_rate": np.array([325.0, 162.5]),
                    "target_sensitivity": np.array([3.25, 3.25]),
                },
                v3.MODE_RAW_FULL_GM: {
                    "measurement_rate": np.array([350.0, 175.0]),
                    "target_sensitivity": np.array([3.5, 3.5]),
                },
                v3.MODE_TEW_FULL_AP_PA: {
                    "measurement_rate": np.array([375.0, 187.5]),
                    "target_sensitivity": np.array([3.75, 3.75]),
                },
                v3.MODE_RAW_FULL_AP_PA: {
                    "measurement_rate": np.array([400.0, 200.0]),
                    "target_sensitivity": np.array([4.0, 4.0]),
                    "features": {
                        v3.LOCKED_FEATURE: np.array([7.0, 8.0]),
                    },
                },
                v3.MODE_ALL_FOUR_FULL_AP_PA: {
                    "measurement_rate": np.array([500.0, 250.0]),
                    "target_sensitivity": np.array([5.0, 5.0]),
                },
            },
        }
    }
    selected = v3.select_rate_mode_observations(
        base, v3.MODE_RAW_FULL_AP_PA
    )["d"]
    assert np.allclose(selected["measurement_rate"], [400.0, 200.0])
    assert np.allclose(selected["target_sensitivity"], [4.0, 4.0])
    assert np.allclose(selected["reference_activity"], reference)
    assert np.allclose(selected["features"][v3.LOCKED_FEATURE], [7.0, 8.0])


def test_tew_fraction_is_available_only_in_tew_modes():
    for rate_mode in (
        v3.MODE_TEW_CROP_GM,
        v3.MODE_TEW_FULL_GM,
        v3.MODE_TEW_FULL_AP_PA,
    ):
        assert v3.MODEL_TEW_FRACTION in v3.RATE_MODES[rate_mode].model_names
    for rate_mode in (
        v3.MODE_RAW_CROP_GM,
        v3.MODE_RAW_FULL_GM,
        v3.MODE_RAW_FULL_AP_PA,
        v3.MODE_ALL_FOUR_FULL_AP_PA,
    ):
        assert v3.MODEL_TEW_FRACTION not in v3.RATE_MODES[rate_mode].model_names
