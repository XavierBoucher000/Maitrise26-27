import pytest
from pathlib import Path

import run_all_patient_figures as runner
import study_layout


def test_selected_analyses_preserves_registry_order():
    selected = runner._selected_analyses(
        ["compare_crop_methods", "acquisition_timeline"]
    )
    assert [item.name for item in selected] == [
        "acquisition_timeline",
        "compare_crop_methods",
    ]


def test_selected_analyses_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown analyses"):
        runner._selected_analyses(["not_an_analysis"])


def test_dry_run_recognizes_an_existing_successful_run(monkeypatch, tmp_path):
    dataset = study_layout.CycleDataset(
        "patient 1", "2026-05__Studies", tmp_path / "q", tmp_path / "p"
    )
    monkeypatch.setattr(
        runner.study_layout,
        "cycle_inventory",
        lambda _dataset: {
            "planar_timepoints": 5,
            "qspect_complete_timepoints": 5,
            "qspect_incomplete_timepoints": 0,
            "complete_for_full_pipeline": True,
        },
    )
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    (output_dir / "run.log").write_text("return_code=0\n", encoding="utf-8")
    monkeypatch.setattr(
        runner.figure_layout, "patient_cycle_dir", lambda *_args: output_dir
    )

    result = runner._run_analysis(runner.ANALYSES[0], dataset, dry_run=True)

    assert result["status"] == "completed_existing"
    assert result["return_code"] == 0
