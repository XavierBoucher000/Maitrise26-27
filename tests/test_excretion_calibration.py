import numpy as np
import pytest

import excretion_calibration_test
import sang_urine


def _inputs():
    data = sang_urine.load_sang_urine_csv()
    balance = sang_urine.calculate_urine_mass_balance(data)
    imaging = sang_urine.load_imaging_comparison(data["injection_datetime"])
    assert imaging is not None
    return data, balance, imaging


def test_day1_calibration_matches_excretion_reference_exactly():
    data, balance, imaging = _inputs()

    result = excretion_calibration_test.calculate_urine_calibration(
        data, balance, imaging, calibration_label="Day1"
    )
    row = result["rows"][result["calibration_index"]]

    assert result["calibration_factor"] == pytest.approx(0.83542, abs=1e-5)
    assert row["new_urine_calibrated_planar_mbq"] == pytest.approx(
        row["excretion_reference_at_planar_time_mbq"]
    )
    assert row["new_planar_over_excretion_reference"] == pytest.approx(1.0)


def test_qspect_values_do_not_change_urine_calibration_factor():
    data, balance, imaging = _inputs()
    baseline = excretion_calibration_test.calculate_urine_calibration(
        data, balance, imaging
    )
    changed_qspect = dict(imaging)
    changed_qspect["qspect_mbq"] = np.asarray(imaging["qspect_mbq"]) * 10.0

    changed = excretion_calibration_test.calculate_urine_calibration(
        data, balance, changed_qspect
    )

    assert changed["calibration_factor"] == pytest.approx(
        baseline["calibration_factor"]
    )


def test_calibration_rejects_day0_before_first_completed_collection():
    data, balance, imaging = _inputs()

    with pytest.raises(ValueError, match="completed urine collection"):
        excretion_calibration_test.calculate_urine_calibration(
            data, balance, imaging, calibration_label="Day0"
        )
