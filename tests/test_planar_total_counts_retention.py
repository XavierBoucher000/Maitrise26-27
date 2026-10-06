from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import planar_total_counts_retention as analysis


def _image(window, view, counts, pixels=None):
    item = {
        "energy_window": window,
        "view": view,
        "pixel_sum": float(counts),
    }
    if pixels is not None:
        item["image"] = np.asarray(pixels, dtype=float)
    return item


def _complete_scan():
    images = []
    value = 1.0
    for window in analysis.EXPECTED_WINDOWS:
        images.append(_image(window, "AP", value))
        images.append(_image(window, "PA", value + 1.0))
        value += 2.0
    return {"images": images}


def test_total_ap_pa_counts_by_window_sums_exactly_two_views():
    counts = analysis.total_ap_pa_counts_by_window(_complete_scan())
    assert counts == {
        "Lower Scatter": 3.0,
        "Photopeak": 7.0,
        "Upper Scatter": 11.0,
        "Low Energy Scatter": 15.0,
    }


def test_total_ap_pa_counts_by_window_rejects_incomplete_scan():
    scan = _complete_scan()
    scan["images"].pop()
    with pytest.raises(ValueError, match="Low Energy Scatter PA"):
        analysis.total_ap_pa_counts_by_window(scan)


def test_total_gm_counts_by_window_is_pixelwise_and_aligns_pa():
    images = []
    for window in analysis.EXPECTED_WINDOWS:
        images.append(_image(window, "AP", 5.0, [[1.0, 4.0]]))
        images.append(_image(window, "PA", 25.0, [[9.0, 16.0]]))
    scan = {"images": images}

    aligned = analysis.total_gm_counts_by_window(
        scan, align_pa_to_ap=True
    )
    unaligned = analysis.total_gm_counts_by_window(
        scan, align_pa_to_ap=False
    )

    assert aligned == {window: 10.0 for window in analysis.EXPECTED_WINDOWS}
    assert unaligned == {window: 11.0 for window in analysis.EXPECTED_WINDOWS}
    assert aligned["Photopeak"] != pytest.approx(
        np.sqrt(5.0 * 25.0)
    )


def test_total_counts_by_window_dispatches_modes_and_rejects_unknown():
    scan = _complete_scan()
    assert analysis.total_counts_by_window(scan, "AP_PA") == (
        analysis.total_ap_pa_counts_by_window(scan)
    )
    with pytest.raises(ValueError, match="Unknown combination"):
        analysis.total_counts_by_window(scan, "invalid")


def test_add_urine_reference_preserves_decay_consistent_closure():
    rows = [
        {"hours_after_injection": 0.0},
        {"hours_after_injection": 10.0},
    ]
    data = {"injected_activity_mbq": 100.0, "half_life_h": 10.0}
    balance = {
        "times_h": np.asarray([5.0]),
        "interval_injection_equivalent_mbq": np.asarray([20.0]),
    }
    output = analysis.add_urine_retention_reference(rows, data, balance)
    assert output[0]["physical_available_activity_mbq"] == pytest.approx(100.0)
    assert output[0]["measured_cumulative_excreted_activity_mbq"] == pytest.approx(0.0)
    assert output[0]["urine_derived_remaining_activity_mbq"] == pytest.approx(100.0)
    assert output[1]["physical_available_activity_mbq"] == pytest.approx(50.0)
    assert output[1]["measured_cumulative_excreted_activity_mbq"] == pytest.approx(10.0)
    assert output[1]["urine_derived_remaining_activity_mbq"] == pytest.approx(40.0)


def test_through_origin_summary_recovers_known_slope():
    result = analysis.through_origin_summary([1.0, 2.0, 3.0], [5.0, 10.0, 15.0])
    assert result["slope_counts_per_mbq"] == pytest.approx(5.0)
    assert result["r_squared_zero_origin"] == pytest.approx(1.0)
    assert result["cv_pointwise_counts_per_mbq_percent"] == pytest.approx(0.0)


def test_timepoint_labels_follow_elapsed_days():
    values = [
        datetime(2026, 5, 19, 10),
        datetime(2026, 5, 20, 9),
        datetime(2026, 5, 21, 8),
        datetime(2026, 5, 25, 9),
    ]
    assert analysis._timepoint_labels(values) == ["J0", "J1", "J2", "J6"]


@pytest.mark.parametrize(
    ("relative_path", "injected_mbq", "first_total_mbq"),
    [
        ("Data/p11/P-011csv.csv", 7200.0, 1048.87),
        ("Data/p8/P-008.csv", 2030.0, 140.10),
    ],
)
def test_generic_urine_loader_supports_both_patient_layouts(
    relative_path, injected_mbq, first_total_mbq
):
    data = analysis.load_urine_retention_csv(
        Path(analysis.PROJECT_DIR) / relative_path
    )
    assert data["injected_activity_mbq"] == pytest.approx(injected_mbq)
    assert len(data["urine"]) == 6
    assert data["urine"][0]["total_excreted_activity_mbq"] == pytest.approx(
        first_total_mbq
    )


def test_every_configured_cycle_has_a_planar_directory():
    assert set(analysis.DATASETS) == {
        ("p11", "2026-05__Studies"),
        ("p11", "2026-07__Studies"),
        ("p8", "2026-06__Studies"),
    }
    for configured in analysis.DATASETS.values():
        assert Path(configured["planar_dir"]).is_dir()
