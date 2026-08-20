"""
CT-based attenuation-correction helper for planar Lu-177 images.

Goal
----
Build a CT-derived attenuation correction map and apply it to a planar image
crop that was already selected elsewhere. This file intentionally does not
decide where the planar crop should be; crop selection lives in
planar_qspect_crop.py.

Physical idea
-------------
For AP/PA geometric-mean planar imaging, source-depth dependence is partly
cancelled:

    GM = sqrt(AP * PA)

For a simple slab, the remaining attenuation factor is approximately:

    F_CT(x,y) = exp(0.5 * integral(mu_208(x,y,z) dz))

This file estimates mu_208 from CT HU using a simple water-scaled
approximation:

    mu_208 ~= mu_water_208 * (1 + HU / 1000)

Then it integrates through the assumed AP/PA CT axis, resizes the resulting
2D factor map to the planar crop shape, and multiplies the planar crop
pixel-by-pixel.

Main functions
--------------
- hu_to_mu_208_cm_inv(...):
    converts CT HU values to approximate 208 keV linear attenuation
    coefficients in cm^-1.
- ct_attenuation_factor_map(...):
    computes the 2D attenuation factor map F_CT from a CT volume.
- resize_map_to_image(...):
    resizes a 2D CT/projection map to match a planar crop.
- apply_ct_attenuation_correction_to_crop(...):
    applies F_CT to one planar crop and reports before/after counts plus the
    effective correction factor.

Limitations
-----------
The HU-to-mu conversion is a first-pass approximation, not a scanner-specific
bilinear calibration. The AP/PA integration axis is assumed from the
transformed CT/QSPECT grid orientation. The map is resized to the planar crop;
this is not full 2D/3D registration.
"""

from typing import Any, Dict, Tuple

import numpy as np
from scipy.ndimage import zoom

from plots import view_patient_images


MU_WATER_208_CM_INV = 0.135
CT_FACTOR_CLIP = (1.0, 20.0)


def hu_to_mu_208_cm_inv(
    ct_hu: np.ndarray,
    mu_water_208_cm_inv: float = MU_WATER_208_CM_INV,
) -> np.ndarray:
    """Approximate CT HU to linear attenuation coefficient at 208 keV."""
    mu = mu_water_208_cm_inv * (1.0 + np.asarray(ct_hu, dtype=np.float64) / 1000.0)
    return np.clip(mu, 0.0, None)


def ct_attenuation_factor_map(
    ct: Dict[str, Any],
    mu_water_208_cm_inv: float = MU_WATER_208_CM_INV,
    clip_range: Tuple[float, float] = CT_FACTOR_CLIP,
) -> Dict[str, Any]:
    """Compute F_CT = exp(0.5 * integral mu_208 dz) from a CT volume."""
    volume_hu = np.asarray(ct["volume"], dtype=np.float64)
    pixel_spacing = ct.get("pixel_spacing") or []
    if len(pixel_spacing) < 1:
        raise ValueError("CT PixelSpacing is required for attenuation-map integration")

    ap_spacing_cm = float(pixel_spacing[0]) / 10.0
    mu_208 = hu_to_mu_208_cm_inv(volume_hu, mu_water_208_cm_inv)
    mu_integral = np.sum(mu_208, axis=1) * ap_spacing_cm
    factor = np.exp(0.5 * mu_integral)
    factor = np.clip(factor, clip_range[0], clip_range[1])
    return {
        "factor_map": view_patient_images.orient_qspect_display(factor),
        "mu_integral": view_patient_images.orient_qspect_display(mu_integral),
        "ap_spacing_cm": ap_spacing_cm,
        "mu_water_208_cm_inv": mu_water_208_cm_inv,
        "clip_range": clip_range,
    }


def ct_planar_equivalent_images(ct: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Return CT views comparable to a planar AP/PA projection.

    The mean projection is the CT analogue of a planar projection: it collapses
    the CT volume through the assumed AP/PA axis. It is for QC/visualization,
    not for activity quantification.
    """
    volume_hu = np.asarray(ct["volume"], dtype=np.float64)
    center_index = volume_hu.shape[1] // 2
    return {
        "coronal_center_hu": view_patient_images.orient_qspect_display(volume_hu[:, center_index, :]),
        "mean_projection_hu": view_patient_images.orient_qspect_display(np.mean(volume_hu, axis=1)),
    }


def factor_map_statistics(
    factor_map: np.ndarray,
    clip_range: Tuple[float, float] = CT_FACTOR_CLIP,
) -> Dict[str, float]:
    """Summarize a CT attenuation-factor map for QC."""
    finite = np.asarray(factor_map, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "min": np.nan,
            "p05": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "clip_low_fraction": np.nan,
            "clip_high_fraction": np.nan,
        }
    return {
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "clip_low_fraction": float(np.mean(finite <= clip_range[0])),
        "clip_high_fraction": float(np.mean(finite >= clip_range[1])),
    }


def resize_map_to_image(image: np.ndarray, reference_shape: Tuple[int, int], order: int = 1) -> np.ndarray:
    """Resize a 2D CT/projection map to a target 2D image shape."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("Only 2D maps can be resized to the planar crop")
    factors = (reference_shape[0] / image.shape[0], reference_shape[1] / image.shape[1])
    resized = zoom(image, factors, order=order)
    return resized[: reference_shape[0], : reference_shape[1]]


def apply_ct_attenuation_correction_to_crop(
    planar_crop: np.ndarray,
    ct: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the CT attenuation-factor map to one already-cropped planar image."""
    ct_map = ct_attenuation_factor_map(ct)
    factor_resized = resize_map_to_image(ct_map["factor_map"], planar_crop.shape)
    corrected_crop = np.asarray(planar_crop, dtype=np.float64) * factor_resized
    before_counts = float(np.sum(planar_crop))
    after_counts = float(np.sum(corrected_crop))
    effective_factor = after_counts / before_counts if before_counts > 0 else np.nan
    return {
        "ct_map": ct_map,
        "ct_factor_stats": factor_map_statistics(ct_map["factor_map"], tuple(ct_map["clip_range"])),
        "ct_factor_resized": factor_resized,
        "ctac_crop": corrected_crop,
        "planar_crop_counts": before_counts,
        "ctac_crop_counts": after_counts,
        "effective_ct_factor": effective_factor,
    }
