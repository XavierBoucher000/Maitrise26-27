from pathlib import Path

import qspect_processing


def test_primary_qspect_description_excludes_derived_and_alternate_series():
    assert qspect_processing.is_primary_qspect_description("WB QSPECT Day2")
    assert not qspect_processing.is_primary_qspect_description("WB QSPECT Day2 -T6")
    assert not qspect_processing.is_primary_qspect_description("MIP AC QSPECT")
    assert not qspect_processing.is_primary_qspect_description("MFSC FUSION TRANS QSPECT")


def test_expected_series_file_count_from_export_folder_name():
    path = Path("example_WB.QSPECT.Day0_n232__00000")
    assert qspect_processing.expected_series_file_count(path) == 232


def test_expected_series_file_count_is_optional():
    assert qspect_processing.expected_series_file_count(Path("legacy_export")) is None
