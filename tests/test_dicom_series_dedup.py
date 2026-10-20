from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import dicom_series_dedup


def _write_series(
    path: Path, values, acquisition_time="101010", study_date="20260101"
) -> None:
    path.mkdir(parents=True)
    for index, value in enumerate(values, start=1):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(path / f"{index}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = generate_uid()
        ds.SOPInstanceUID = generate_uid()
        ds.Modality = "PT"
        ds.SeriesDescription = "WB QSPECT Day0"
        ds.StudyDate = study_date
        ds.AcquisitionTime = acquisition_time
        ds.InstanceNumber = index
        ds.Rows = 2
        ds.Columns = 2
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = np.full((2, 2), value, dtype=np.uint16).tobytes()
        ds.save_as(path / f"{index}.dcm", enforce_file_format=True)


def test_exact_copy_is_excluded_but_distinct_pixels_are_kept(tmp_path):
    original = tmp_path / "a"
    copy = tmp_path / "b"
    different = tmp_path / "c"
    _write_series(original, [1, 2])
    _write_series(copy, [1, 2])
    _write_series(different, [1, 3])

    kept, duplicates = dicom_series_dedup.deduplicate_series_dirs(
        [different, copy, original]
    )

    assert kept == [original, different]
    assert len(duplicates) == 1
    assert duplicates[0].kept == original
    assert duplicates[0].excluded == copy


def test_same_pixels_with_changed_export_time_are_still_excluded(tmp_path):
    first = tmp_path / "a"
    later = tmp_path / "b"
    _write_series(first, [1, 2], acquisition_time="101010")
    _write_series(later, [1, 2], acquisition_time="121010")

    kept, duplicates = dicom_series_dedup.deduplicate_series_dirs([first, later])

    assert kept == [first]
    assert len(duplicates) == 1


def test_same_pixels_on_different_study_dates_are_not_excluded(tmp_path):
    first = tmp_path / "a"
    later = tmp_path / "b"
    _write_series(first, [1, 2], study_date="20260101")
    _write_series(later, [1, 2], study_date="20260102")

    kept, duplicates = dicom_series_dedup.deduplicate_series_dirs([first, later])

    assert kept == [first, later]
    assert duplicates == []
