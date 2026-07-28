from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import planar_processing


class PlanarTimingTests(unittest.TestCase):
    def test_planar_timing_from_dicom_values(self):
        images = [
            {
                "actual_frame_duration_ms": 239188,
                "scan_velocity": 10.0,
                "scan_length": 2000.0,
            }
        ]

        timing = planar_processing.planar_timing_from_dicom(images)

        self.assertAlmostEqual(timing.actual_frame_duration_s, 239.188)
        self.assertAlmostEqual(timing.scan_time_s, 200.0)
        self.assertAlmostEqual(timing.local_dwell_time_s, 38.7)
        self.assertLess(timing.local_dwell_time_s, timing.scan_time_s)
        self.assertLess(timing.scan_time_s, timing.actual_frame_duration_s)

    def test_planar_timing_rejects_missing_velocity(self):
        images = [{"actual_frame_duration_ms": 239188, "scan_length": 2000.0}]
        with self.assertRaisesRegex(ValueError, "ScanVelocity"):
            planar_processing.planar_timing_from_dicom(images)

    def test_planar_timing_rejects_zero_velocity(self):
        images = [
            {
                "actual_frame_duration_ms": 239188,
                "scan_velocity": 0.0,
                "scan_length": 2000.0,
            }
        ]
        with self.assertRaisesRegex(ValueError, "ScanVelocity"):
            planar_processing.planar_timing_from_dicom(images)

    def test_counts_to_activity_for_frame_scan_and_local_times(self):
        counts = 809347.9
        sensitivity = 9.36

        frame = planar_processing.counts_to_activity_mbq(counts, 239.188, sensitivity)
        scan = planar_processing.counts_to_activity_mbq(counts, 200.0, sensitivity)
        local = planar_processing.counts_to_activity_mbq(counts, 38.7, sensitivity)

        self.assertAlmostEqual(frame, 361.5, places=1)
        self.assertAlmostEqual(scan, 432.3, places=1)
        self.assertAlmostEqual(local, 2234.3, places=1)


if __name__ == "__main__":
    unittest.main()
