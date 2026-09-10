from datetime import datetime

import acquisition_timeline as timeline


def test_parse_dicom_date_time_preserves_fractional_seconds():
    value = timeline.parse_dicom_date_time("20260519", "105026.281000")
    assert value == datetime(2026, 5, 19, 10, 50, 26, 281000)


def test_acquisition_datetime_falls_back_to_study_date():
    class Dataset:
        AcquisitionDateTime = None
        AcquisitionDate = None
        SeriesDate = None
        StudyDate = "20260519"
        AcquisitionTime = "105026.281000"

    assert timeline.acquisition_datetime(Dataset()) == datetime(
        2026, 5, 19, 10, 50, 26, 281000
    )
