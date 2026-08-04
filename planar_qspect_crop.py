"""
Planar/QSPECT crop helper.

Goal
----
Select the planar whole-body image region that corresponds approximately to
the Q/SPECT axial volume coverage. This is needed because the planar image is
longer than the Q/SPECT/CT field of view, so direct total-count comparison
would mix different anatomical coverage.

Main idea
---------
The dynamic crop uses the existing visual matching logic:
1. measure the Q/SPECT head-to-toe extent in physical units;
2. measure the planar signal profile along the patient long axis;
3. center the crop around the planar signal center of mass;
4. add the Q/SPECT head-side gap so the crop matches the test-match figures.

The same function can also apply fixed Day0 crop bounds to every time point.
That option is useful when later planar images are noisy and the dynamic crop
becomes unstable.

Main functions
--------------
- compute_planar_crop_for_qspect(...):
    returns crop bounds, cropped planar image, dynamic bounds, crop strategy,
    physical crop height, and QC measurements.
- planar_crop_matching_qspect(...):
    backward-compatible alias for compute_planar_crop_for_qspect.

Inputs
------
- planar_image: 2D planar image, usually TEW-corrected AP/PA geometric mean.
- planar_record: planar metadata/images used to read pixel size.
- qspect: loaded Q/SPECT series dictionary with a 3D volume.
- fixed_crop_bounds: optional (top, bottom) pixel coordinates, typically from
  Day0, reused for all days.

Limitations
-----------
This is not rigid registration. It is a practical 1D crop along the planar
long axis. It should be treated as a quality-control/analysis helper, not as a
validated image-registration method.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from plots import view_patient_images


PLANAR_QSPECT_CROP_THRESHOLD = 0.1


def compute_planar_crop_for_qspect(
    planar_image: np.ndarray,
    planar_record: Dict[str, Any],
    qspect: Dict[str, Any],
    threshold_fraction: float = PLANAR_QSPECT_CROP_THRESHOLD,
    fixed_crop_bounds: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Crop a planar image to the approximate Q/SPECT head-to-toe coverage.

    The dynamic crop is centered on the planar mean-profile center of mass,
    uses the Q/SPECT thresholded coronal signal length, then adds the Q/SPECT
    head-side gap to match the current imaging test-match logic.
    """
    line = view_patient_images.qspect_length_line_on_planar(planar_image, planar_record, qspect)
    planar_measurement = view_patient_images.measure_signal_length_y(
        planar_image,
        line["pixel_spacing_mm"],
        threshold_fraction=threshold_fraction,
    )
    qspect_volume = qspect["volume"]
    coronal_index = qspect_volume.shape[1] // 2
    qspect_coronal = view_patient_images.orient_qspect_display(qspect_volume[:, coronal_index, :])
    qspect_spacing_y_mm = line["qspect_length_mm"] / qspect_coronal.shape[0]
    qspect_measurement = view_patient_images.measure_signal_length_y(
        qspect_coronal,
        qspect_spacing_y_mm,
        threshold_fraction=threshold_fraction,
    )

    matched_length_px = qspect_measurement["length_cm"] * 10.0 / line["pixel_spacing_mm"]
    matched_center_y = planar_measurement["center_of_mass_y"]
    matched_top = matched_center_y - matched_length_px / 2.0
    matched_bottom = matched_center_y + matched_length_px / 2.0
    qspect_gap_px = float(qspect_coronal.shape[0]) - qspect_measurement["length_pixels"]
    qspect_gap_cm = qspect_gap_px * qspect_spacing_y_mm / 10.0
    qspect_gap_planar_px = qspect_gap_cm * 10.0 / line["pixel_spacing_mm"]
    adjusted_top = max(0.0, matched_top - qspect_gap_planar_px)
    adjusted_bottom = min(float(planar_image.shape[0] - 1), matched_bottom)
    dynamic_crop_top = int(np.floor(adjusted_top))
    dynamic_crop_bottom = int(np.ceil(adjusted_bottom)) + 1

    if fixed_crop_bounds is None:
        crop_top = dynamic_crop_top
        crop_bottom = dynamic_crop_bottom
        crop_strategy = "individual"
    else:
        crop_top, crop_bottom = fixed_crop_bounds
        crop_top = max(0, int(crop_top))
        crop_bottom = min(int(crop_bottom), planar_image.shape[0])
        crop_strategy = "fixed_day0"
        if crop_bottom <= crop_top:
            raise ValueError(f"Invalid fixed crop bounds: {fixed_crop_bounds}")

    crop = planar_image[crop_top:crop_bottom, :]
    return {
        "crop": crop,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "dynamic_crop_top": dynamic_crop_top,
        "dynamic_crop_bottom": dynamic_crop_bottom,
        "crop_strategy": crop_strategy,
        "crop_height_cm": crop.shape[0] * line["pixel_spacing_mm"] / 10.0,
        "line": line,
        "planar_measurement": planar_measurement,
        "qspect_measurement": qspect_measurement,
        "qspect_coronal": qspect_coronal,
    }


def planar_crop_matching_qspect(
    planar_image: np.ndarray,
    planar_record: Dict[str, Any],
    qspect: Dict[str, Any],
    threshold_fraction: float = PLANAR_QSPECT_CROP_THRESHOLD,
    fixed_crop_bounds: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Backward-compatible alias for the crop function."""
    return compute_planar_crop_for_qspect(
        planar_image,
        planar_record,
        qspect,
        threshold_fraction=threshold_fraction,
        fixed_crop_bounds=fixed_crop_bounds,
    )
