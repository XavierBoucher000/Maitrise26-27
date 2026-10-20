import numpy as np
import pytest

import patient_sensitivity_urine_generalized as generalized
import patient_sensitivity_urine_target as urine


def test_zero_origin_slope_recovers_known_sensitivity():
    slope = generalized.zero_origin_slope(
        [100.0, 80.0, 60.0], [250.0, 200.0, 150.0]
    )
    assert slope == pytest.approx(2.5)


def test_pair_directions_cover_every_directed_patient_transfer():
    pairs = generalized._pair_directions(("p8", "p11", "p17"))
    assert len(pairs) == 6
    assert ("p11", "p17") in pairs
    assert all(source != target for source, target in pairs)


def test_configured_urine_datasets_use_canonical_patient_folders():
    assert {spec.patient_id for spec in urine.URINE_DATASETS} == {
        "p8",
        "p11",
        "p17",
    }
    assert all(" copy" not in str(spec.planar_dir.parent.name).lower() for spec in urine.URINE_DATASETS)
    assert urine.P17_DATASET.urine_csv.name == "P-017.xlsx"


def test_activity_metrics_can_exclude_j0():
    metrics = generalized._activity_metrics(
        np.array([100.0, 80.0, 60.0]),
        np.array([100.0, 88.0, 54.0]),
        start=1,
    )
    assert metrics["n_evaluated"] == 2
    assert metrics["mare_percent"] == pytest.approx(10.0)


def test_raw_count_slope_series_uses_same_counts_as_retention_figure():
    results = {
        ("p17", "GM"): {
            "comparison_timepoints": 3,
            "rows": [
                {"photopeak_counts": 30.0},
                {"photopeak_counts": 20.0},
                {"photopeak_counts": 10.0},
                {"photopeak_counts": 5.0},
            ],
        }
    }
    values = generalized._raw_count_series(
        results, "p17", urine.SIMPLE_PHOTO_GM_PIXEL
    )
    assert values.tolist() == [30.0, 20.0, 10.0]
