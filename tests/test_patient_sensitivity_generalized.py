from pathlib import Path

import patient_sensitivity_generalized as generalized
import patient_sensitivity_multicycle_v3 as v3


def _spec(key, patient, cycle):
    return v3.DatasetSpec(
        key=key,
        patient_id=patient,
        cycle_id=cycle,
        label=key,
        short_label=key,
        planar_dir=Path("planar") / key,
        qspect_dir=Path("qspect") / key,
    )


def test_dataset_key_is_stable_and_command_line_safe():
    assert generalized.dataset_key("p17", "2026-09__Studies") == "p17_2026_09_studies"


def test_grouped_splits_include_every_cycle_and_patient_holdout():
    datasets = {
        "p11_a": _spec("p11_a", "p11", "cycle_a"),
        "p11_b": _spec("p11_b", "p11", "cycle_b"),
        "p8_a": _spec("p8_a", "p8", "cycle_a"),
        "p17_a": _spec("p17_a", "p17", "cycle_a"),
    }
    splits = generalized.grouped_splits(datasets)
    cycle_splits = [s for s in splits if s.split_type == "leave_one_cycle_out"]
    patient_splits = [s for s in splits if s.split_type == "leave_one_patient_out"]
    assert len(cycle_splits) == 4
    assert len(patient_splits) == 3

    p11 = next(s for s in patient_splits if s.split_id == "patient_holdout_p11")
    assert p11.test_datasets == ("p11_a", "p11_b")
    assert p11.training_datasets == ("p8_a", "p17_a")


def test_pairwise_splits_are_optional_to_limit_quadratic_growth():
    datasets = {
        "a": _spec("a", "p1", "cycle_a"),
        "b": _spec("b", "p2", "cycle_b"),
        "c": _spec("c", "p3", "cycle_c"),
    }
    without = generalized.grouped_splits(datasets, include_pairwise=False)
    with_pairwise = generalized.grouped_splits(datasets, include_pairwise=True)
    assert not any(s.split_type == "pairwise_cycle_transfer" for s in without)
    assert len([s for s in with_pairwise if s.split_type == "pairwise_cycle_transfer"]) == 6
