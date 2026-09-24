from pathlib import Path

import study_layout


def test_discover_cycles_pairs_qspect_and_wbp_folders(tmp_path):
    patient = tmp_path / "p8"
    (patient / "2026-06__Studies").mkdir(parents=True)
    (patient / "2026-06__Studies_WBP").mkdir()
    (patient / "notes").mkdir()

    cycles = study_layout.discover_cycles(tmp_path)

    assert cycles == [
        study_layout.CycleDataset(
            "p8",
            "2026-06__Studies",
            patient / "2026-06__Studies",
            patient / "2026-06__Studies_WBP",
        )
    ]
