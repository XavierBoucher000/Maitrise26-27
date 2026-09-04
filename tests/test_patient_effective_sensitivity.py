from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import patient_effective_sensitivity as pes


def test_effective_sensitivity_units():
    assert pes.effective_sensitivity_cps_per_mbq(
        counts=3870.0,
        local_dwell_time_s=38.7,
        reference_activity_mbq=20.0,
    ) == pytest.approx(5.0)


def test_activity_time_matching_uses_physical_decay_direction():
    reference_time = datetime(2026, 5, 19, 12, 0)
    earlier_time = reference_time - timedelta(hours=2.0)
    later_time = reference_time + timedelta(hours=2.0)

    earlier = pes.activity_at_target_time_physical_decay(
        1000.0, reference_time, earlier_time
    )
    later = pes.activity_at_target_time_physical_decay(
        1000.0, reference_time, later_time
    )

    assert earlier > 1000.0
    assert later < 1000.0
    assert earlier == pytest.approx(1000.0 * 2.0 ** (2.0 / 159.5))
    assert later == pytest.approx(1000.0 * 2.0 ** (-2.0 / 159.5))


def test_constant_series_has_zero_cv_and_slope():
    metrics = pes.stability_metrics([5.0, 5.0, 5.0], [0.0, 1.0, 2.0])

    assert metrics["mean_cps_per_mbq"] == pytest.approx(5.0)
    assert metrics["cv_percent"] == pytest.approx(0.0)
    assert metrics["relative_range_percent"] == pytest.approx(0.0)
    assert metrics["last_over_first"] == pytest.approx(1.0)
    assert metrics["slope_cps_per_mbq_per_day"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "counts,dwell,activity",
    [(-1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)],
)
def test_effective_sensitivity_rejects_invalid_inputs(counts, dwell, activity):
    with pytest.raises(ValueError):
        pes.effective_sensitivity_cps_per_mbq(counts, dwell, activity)


def test_stability_metrics_uses_sample_sd():
    metrics = pes.stability_metrics([2.0, 4.0], [0.0, 1.0])

    assert metrics["mean_cps_per_mbq"] == pytest.approx(3.0)
    assert metrics["sd_cps_per_mbq"] == pytest.approx(np.sqrt(2.0))
    assert metrics["slope_cps_per_mbq_per_day"] == pytest.approx(2.0)
