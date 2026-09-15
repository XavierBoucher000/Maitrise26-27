from pathlib import Path

import pytest

import figure_layout


def test_patient_cycle_dir_uses_script_patient_cycle_hierarchy():
    assert figure_layout.patient_cycle_dir(
        Path("analysis.py"), "Patient 2", "2026-06__Studies"
    ) == figure_layout.FIG_ROOT / "analysis" / "patient 2" / "2026-06__Studies"


def test_shared_dir_is_not_assigned_to_a_patient():
    assert figure_layout.shared_dir("hu_calibration.py") == (
        figure_layout.FIG_ROOT / "hu_calibration" / "shared"
    )


@pytest.mark.parametrize("value", ["p1", "patient", "patient_x"])
def test_invalid_patient_identifier_is_rejected(value):
    with pytest.raises(ValueError):
        figure_layout.patient_cycle_dir("analysis.py", value, "2026-05__Studies")


@pytest.mark.parametrize("value", ["cycle1", "2026-05", "Studies"])
def test_invalid_cycle_identifier_is_rejected(value):
    with pytest.raises(ValueError):
        figure_layout.patient_cycle_dir("analysis.py", "patient 1", value)
