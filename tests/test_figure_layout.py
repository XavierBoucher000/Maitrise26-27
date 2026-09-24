from pathlib import Path

import pytest

import figure_layout


def test_patient_cycle_dir_uses_script_patient_cycle_hierarchy():
    assert figure_layout.patient_cycle_dir(
        Path("analysis.py"), "p8", "2026-06__Studies"
    ) == figure_layout.FIG_ROOT / "analysis" / "p8" / "2026-06__Studies"


def test_shared_dir_is_not_assigned_to_a_patient():
    assert figure_layout.shared_dir("hu_calibration.py") == (
        figure_layout.FIG_ROOT / "hu_calibration" / "shared"
    )


@pytest.mark.parametrize("value", ["patient", "patient_x", "subject11"])
def test_invalid_patient_identifier_is_rejected(value):
    with pytest.raises(ValueError):
        figure_layout.patient_cycle_dir("analysis.py", value, "2026-05__Studies")


@pytest.mark.parametrize("value", ["cycle1", "2026-05", "Studies"])
def test_invalid_cycle_identifier_is_rejected(value):
    with pytest.raises(ValueError):
        figure_layout.patient_cycle_dir("analysis.py", "p11", value)
