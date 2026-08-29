import numpy as np
import pytest

import sang_urine


def test_source_csv_parser_finds_expected_measurements():
    data = sang_urine.load_sang_urine_csv()

    assert data["injected_activity_mbq"] == pytest.approx(7200.0)
    assert data["half_life_h"] == pytest.approx(159.5)
    assert len(data["blood"]) == 9
    assert len(data["urine"]) == 6
    assert data["blood"][0]["concentration_kbq_ml"] == pytest.approx(601.4683845)
    assert data["urine"][0]["diaper_activity_mbq"] == pytest.approx(169.68)
    assert data["urine"][3]["diaper_activity_mbq"] == pytest.approx(123.04)
    assert data["urine"][3]["component_sum_difference_mbq"] == pytest.approx(0.01)


def test_decay_corrected_mass_balance_closes_numerically():
    data = sang_urine.load_sang_urine_csv()
    balance = sang_urine.calculate_urine_mass_balance(data)

    np.testing.assert_allclose(balance["closure_error_mbq"], 0.0, atol=1e-10)
    assert balance["predicted_body_mbq"][-1] == pytest.approx(862.66, abs=0.02)
    assert balance["interval_excreted_mbq"].sum() == pytest.approx(5785.65)


def test_blood_concentration_is_not_added_to_excreted_activity():
    data = sang_urine.load_sang_urine_csv()
    balance = sang_urine.calculate_urine_mass_balance(data)

    expected = sum(item["total_excreted_activity_mbq"] for item in data["urine"])
    assert balance["interval_excreted_mbq"].sum() == pytest.approx(expected)


def test_cumulative_excreted_activity_matches_balance_collection_values():
    data = sang_urine.load_sang_urine_csv()
    balance = sang_urine.calculate_urine_mass_balance(data)

    evaluated = sang_urine.cumulative_excreted_activity(
        balance["times_h"], data, balance
    )

    np.testing.assert_allclose(evaluated, balance["cumulative_excreted_at_time_mbq"])


def test_imaging_excretion_closure_ratio_uses_same_time_reference():
    data = sang_urine.load_sang_urine_csv()
    balance = sang_urine.calculate_urine_mass_balance(data)
    times = np.asarray([2.28, 22.95])
    predicted = sang_urine.predicted_body_activity(times, data, balance)
    imaging = {
        "labels": np.asarray(["A", "B"]),
        "planar_times_h": times,
        "qspect_times_h": times,
        "planar_ctac_mbq": predicted,
        "qspect_mbq": predicted,
    }

    rows = sang_urine.calculate_imaging_excretion_comparison(data, balance, imaging)

    assert all(row["mass_balance_closure_ratio"] == pytest.approx(1.0) for row in rows)
