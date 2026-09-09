import numpy as np
import pytest

import imaging_excretion_fit_balance as iefb
import planar_excretion_fit_balance as pefb
import sang_urine


def _inputs():
    data = sang_urine.load_sang_urine_csv()
    balance = sang_urine.calculate_urine_mass_balance(data)
    imaging = sang_urine.load_imaging_comparison(data["injection_datetime"])
    assert imaging is not None
    return data, balance, imaging


def test_qspect_modality_mapping_uses_qspect_times_and_activities():
    _, _, imaging = _inputs()

    mapped = iefb.build_modality_imaging(imaging, "qspect")

    np.testing.assert_allclose(mapped["planar_times_h"], imaging["qspect_times_h"])
    np.testing.assert_allclose(mapped["planar_ctac_mbq"], imaging["qspect_mbq"])
    np.testing.assert_array_equal(mapped["labels"], imaging["labels"])


def test_unknown_modality_is_rejected():
    _, _, imaging = _inputs()

    with pytest.raises(ValueError, match="Unknown imaging modality"):
        iefb.build_modality_imaging(imaging, "spect")


def test_planar_and_qspect_use_same_excretion_fit_but_exact_imaging_times():
    data, balance, imaging = _inputs()
    planar = iefb.calculate_modality_balance(data, balance, imaging, "planar")
    qspect = iefb.calculate_modality_balance(data, balance, imaging, "qspect")
    rows = iefb.comparison_rows(planar, qspect)

    np.testing.assert_allclose(
        planar["fits"]["excretion_parameters"],
        qspect["fits"]["excretion_parameters"],
    )
    assert pefb.planar_biexponential(
        0.0, *planar["fits"]["planar_parameters"]
    ) == pytest.approx(data["injected_activity_mbq"])
    assert pefb.planar_biexponential(
        0.0, *qspect["fits"]["planar_parameters"]
    ) == pytest.approx(data["injected_activity_mbq"])
    assert len(rows) == 10
    day0_planar = next(
        row for row in rows if row["modality"] == "Planaire" and row["label"] == "Day0"
    )
    day0_qspect = next(
        row for row in rows if row["modality"] == "Q/SPECT" and row["label"] == "Day0"
    )
    assert day0_planar["elapsed_h"] == pytest.approx(
        imaging["planar_times_h"][0]
    )
    assert day0_qspect["elapsed_h"] == pytest.approx(
        imaging["qspect_times_h"][0]
    )
    assert day0_planar["closure_error_percent"] == pytest.approx(5.0184, abs=1e-3)
    assert day0_qspect["closure_error_percent"] == pytest.approx(8.0468, abs=1e-3)
