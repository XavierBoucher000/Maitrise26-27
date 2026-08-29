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

    mu_integral = np.sum(mu_208, axis=1) * ap_spacing_cm

The original method estimates mu_208 from CT HU using a simple water-scaled
approximation (rearrange the standard HU definition):

    mu_208 ~= mu_water_208 * (1 + HU / 1000)

Then it integrates through the assumed AP/PA CT axis, resizes the resulting
2D factor map to the planar crop shape, and multiplies the planar crop
pixel-by-pixel. A parallel exploratory method also converts HU to RayStation
mass density, assigns a broad material family, and uses that material's
mass-attenuation coefficient at 208 keV.

Main functions
--------------
- hu_to_mu_208_cm_inv(...):
    converts CT HU values to approximate 208 keV linear attenuation
    coefficients in cm^-1.
- hu_to_mu_208_raystation_materials(...):
    converts HU through RayStation mass density and material-specific mu/rho.
- ct_attenuation_factor_map(...):
    computes the 2D attenuation factor map F_CT from a CT volume.
- ct_attenuation_factor_map_raystation_materials(...):
    computes the same map using the new density/material conversion.
- resize_map_to_image(...):
    resizes a 2D CT/projection map to match a planar crop.
- apply_ct_attenuation_correction_to_crop(...):
    applies F_CT to one planar crop and reports before/after counts plus the
    effective correction factor.

Limitations
-----------
Both HU-to-mu conversions remain development methods. The broad HU material
classes are not tissue segmentation, and the RayStation density curve must
match the patient's CT imaging system. The AP/PA integration axis is assumed
from the transformed CT/QSPECT grid orientation. The map is resized to the
planar crop; this is not full 2D/3D registration.
"""

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

from plots import view_patient_images


MU_WATER_208_CM_INV = 0.135
CT_FACTOR_CLIP = (1.0, 20.0)

# Commissioned RayStation HU-to-mass-density control points supplied for this
# project. Values outside the calibrated HU range are clamped to the endpoint
# densities, matching the behavior displayed in RayStation.
RAYSTATION_HU_NODES = np.asarray(
    [-1000.0, -803.0, -505.0, -67.0, -31.0, 9.0, 50.0, 55.0, 196.0, 844.0, 4000.0],
    dtype=np.float64,
)
RAYSTATION_MASS_DENSITY_G_CM3 = np.asarray(
    [0.00121, 0.217, 0.508, 0.967, 0.990, 1.018, 1.061, 1.071, 1.159, 1.575, 3.308],
    dtype=np.float64,
)

# Exploratory HU classes. They identify broad material families, not organs.
# Boundaries remain configurable because HU depends on the CT scanner/protocol
# and can be altered by contrast, artifacts, and partial-volume effects.
DEFAULT_MATERIAL_NAMES = ("air", "lung", "adipose", "soft_tissue", "bone")
DEFAULT_MATERIAL_THRESHOLDS_HU = (-950.0, -300.0, -50.0, 300.0)

# NIST XCOM total mass attenuation coefficients bracketing 208 keV.
# Each pair is (photon energy in keV, mu/rho in cm^2/g). These are replaceable
# by scanner/project-specific material tables passed to the conversion function.
DEFAULT_MASS_ATTENUATION_TABLES = {
    "air": ((200.0, 0.1233), (300.0, 0.1067)),
    "lung": ((200.0, 0.1359), (300.0, 0.1177)),
    "adipose": ((200.0, 0.1368), (300.0, 0.1187)),
    "soft_tissue": ((200.0, 0.1358), (300.0, 0.1175)),
    "bone": ((200.0, 0.1309), (300.0, 0.1113)),
}


def raystation_hu_to_mass_density_g_cm3(
    ct_hu: np.ndarray,
    hu_nodes: np.ndarray = RAYSTATION_HU_NODES,
    density_nodes_g_cm3: np.ndarray = RAYSTATION_MASS_DENSITY_G_CM3,
) -> np.ndarray:
    """Interpolate the commissioned RayStation HU-to-density table."""
    hu_nodes = np.asarray(hu_nodes, dtype=np.float64)
    density_nodes = np.asarray(density_nodes_g_cm3, dtype=np.float64)
    if hu_nodes.ndim != 1 or density_nodes.ndim != 1 or hu_nodes.size != density_nodes.size:
        raise ValueError("HU and density nodes must be one-dimensional arrays of equal length")
    if hu_nodes.size < 2 or np.any(np.diff(hu_nodes) <= 0.0):
        raise ValueError("HU nodes must contain at least two strictly increasing values")
    if np.any(density_nodes <= 0.0) or not np.all(np.isfinite(density_nodes)):
        raise ValueError("Mass-density nodes must be finite and strictly positive")

    hu = np.asarray(ct_hu, dtype=np.float64)
    return np.interp(hu, hu_nodes, density_nodes, left=density_nodes[0], right=density_nodes[-1])


def classify_material_from_hu(
    ct_hu: np.ndarray,
    thresholds_hu: Sequence[float] = DEFAULT_MATERIAL_THRESHOLDS_HU,
    material_names: Sequence[str] = DEFAULT_MATERIAL_NAMES,
) -> Dict[str, Any]:
    """Assign broad material classes from configurable HU thresholds.

    The returned uint8 map contains indices into ``material_names``. This is an
    exploratory material approximation, not an organ segmentation.
    """
    thresholds = np.asarray(thresholds_hu, dtype=np.float64)
    names = tuple(str(name) for name in material_names)
    if thresholds.ndim != 1 or np.any(np.diff(thresholds) <= 0.0):
        raise ValueError("Material HU thresholds must be strictly increasing")
    if len(names) != thresholds.size + 1:
        raise ValueError("The number of material names must equal number of thresholds plus one")
    if len(names) > np.iinfo(np.uint8).max + 1:
        raise ValueError("At most 256 material classes are supported")

    hu = np.asarray(ct_hu, dtype=np.float64)
    material_codes = np.digitize(hu, thresholds, right=False).astype(np.uint8)
    return {
        "material_codes": material_codes,
        "material_names": names,
        "thresholds_hu": thresholds,
    }


def interpolate_mass_attenuation_cm2_g(
    energy_mu_pairs: Sequence[Tuple[float, float]],
    target_energy_kev: float = 208.0,
) -> float:
    """Log-log interpolate mu/rho at one monoenergetic photon energy.

    Input energies must be photon energies in keV, not x-ray tube kVp values.
    Extrapolation is rejected so that 208 keV must be bracketed by the table.
    """
    table = np.asarray(energy_mu_pairs, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < 2:
        raise ValueError("A material table must contain at least two (energy_keV, mu_over_rho) pairs")
    order = np.argsort(table[:, 0])
    energies = table[order, 0]
    coefficients = table[order, 1]
    if np.any(np.diff(energies) <= 0.0):
        raise ValueError("Material-table energies must be unique")
    if np.any(energies <= 0.0) or np.any(coefficients <= 0.0):
        raise ValueError("Material-table energies and coefficients must be strictly positive")
    if not np.all(np.isfinite(table)) or not np.isfinite(target_energy_kev) or target_energy_kev <= 0.0:
        raise ValueError("Material-table values and target energy must be finite and positive")
    if target_energy_kev < energies[0] or target_energy_kev > energies[-1]:
        raise ValueError(
            f"Target energy {target_energy_kev:g} keV is outside the material table "
            f"[{energies[0]:g}, {energies[-1]:g}] keV"
        )

    return float(
        np.exp(
            np.interp(
                np.log(target_energy_kev),
                np.log(energies),
                np.log(coefficients),
            )
        )
    )


def hu_to_mu_208_raystation_materials(
    ct_hu: np.ndarray,
    material_tables: Mapping[str, Sequence[Tuple[float, float]]] = DEFAULT_MASS_ATTENUATION_TABLES,
    target_energy_kev: float = 208.0,
    thresholds_hu: Sequence[float] = DEFAULT_MATERIAL_THRESHOLDS_HU,
    material_names: Sequence[str] = DEFAULT_MATERIAL_NAMES,
) -> Dict[str, Any]:
    """Convert CT HU to linear attenuation at 208 keV using density and material.

    Pipeline:
        HU -> RayStation mass density -> broad material class
           -> material mu/rho at 208 keV -> linear mu at 208 keV.

    The default material tables use NIST XCOM values at 200 and 300 keV.
    Callers can replace them with project tables using the same mapping format.
    """
    hu = np.asarray(ct_hu, dtype=np.float64)
    density = raystation_hu_to_mass_density_g_cm3(hu)
    classification = classify_material_from_hu(hu, thresholds_hu, material_names)
    names = classification["material_names"]
    missing_materials = [name for name in names if name not in material_tables]
    if missing_materials:
        raise ValueError(f"Missing attenuation tables for materials: {', '.join(missing_materials)}")

    material_mu_over_rho = {
        name: interpolate_mass_attenuation_cm2_g(material_tables[name], target_energy_kev)
        for name in names
    }
    coefficient_by_code = np.asarray([material_mu_over_rho[name] for name in names], dtype=np.float64)
    mu_over_rho_map = coefficient_by_code[classification["material_codes"]]
    mu_208 = density * mu_over_rho_map
    outside_density_calibration = (hu < RAYSTATION_HU_NODES[0]) | (hu > RAYSTATION_HU_NODES[-1])

    return {
        "mu_208_cm_inv": mu_208,
        "mass_density_g_cm3": density,
        "mu_over_rho_cm2_g": mu_over_rho_map,
        "material_codes": classification["material_codes"],
        "material_names": names,
        "material_thresholds_hu": classification["thresholds_hu"],
        "material_mu_over_rho_cm2_g": material_mu_over_rho,
        "target_energy_kev": float(target_energy_kev),
        "outside_density_calibration": outside_density_calibration,
    }


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


def ct_attenuation_factor_map_raystation_materials(
    ct: Dict[str, Any],
    material_tables: Mapping[str, Sequence[Tuple[float, float]]] = DEFAULT_MASS_ATTENUATION_TABLES,
    target_energy_kev: float = 208.0,
    thresholds_hu: Sequence[float] = DEFAULT_MATERIAL_THRESHOLDS_HU,
    material_names: Sequence[str] = DEFAULT_MATERIAL_NAMES,
    clip_range: Tuple[float, float] = CT_FACTOR_CLIP,
) -> Dict[str, Any]:
    """Compute the CT factor map using RayStation density and material mu/rho."""
    volume_hu = np.asarray(ct["volume"], dtype=np.float64)
    pixel_spacing = ct.get("pixel_spacing") or []
    if len(pixel_spacing) < 1:
        raise ValueError("CT PixelSpacing is required for attenuation-map integration")

    ap_spacing_cm = float(pixel_spacing[0]) / 10.0
    conversion = hu_to_mu_208_raystation_materials(
        volume_hu,
        material_tables=material_tables,
        target_energy_kev=target_energy_kev,
        thresholds_hu=thresholds_hu,
        material_names=material_names,
    )
    mu_integral = np.sum(conversion["mu_208_cm_inv"], axis=1) * ap_spacing_cm
    factor_unclipped = np.exp(0.5 * mu_integral)
    factor = np.clip(factor_unclipped, clip_range[0], clip_range[1])
    return {
        "factor_map": view_patient_images.orient_qspect_display(factor),
        "factor_map_unclipped": view_patient_images.orient_qspect_display(factor_unclipped),
        "mu_integral": view_patient_images.orient_qspect_display(mu_integral),
        "ap_spacing_cm": ap_spacing_cm,
        "clip_range": clip_range,
        "conversion": conversion,
        "conversion_method": "raystation_density_plus_material_mu_over_rho",
    }


def ct_planar_equivalent_images(ct: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Return CT views comparable to an AP/PA planar projection for QC."""
    volume_hu = np.asarray(ct["volume"], dtype=np.float64)
    center_index = volume_hu.shape[1] // 2
    return {
        "coronal_center_hu": view_patient_images.orient_qspect_display(
            volume_hu[:, center_index, :]
        ),
        "mean_projection_hu": view_patient_images.orient_qspect_display(
            np.mean(volume_hu, axis=1)
        ),
    }


def factor_map_statistics(
    factor_map: np.ndarray,
    clip_range: Tuple[float, float] = CT_FACTOR_CLIP,
) -> Dict[str, float]:
    """Summarize a CT attenuation-factor map for quality control."""
    finite = np.asarray(factor_map, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            key: np.nan
            for key in (
                "min", "p05", "median", "mean", "p95", "max",
                "clip_low_fraction", "clip_high_fraction",
            )
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


def _centered_origin_xy_mm(
    shape_yx: Tuple[int, int],
    spacing_yx_mm: Tuple[float, float],
) -> Tuple[float, float]:
    """Return an origin that places the center pixel at physical (0, 0)."""
    height, width = (int(shape_yx[0]), int(shape_yx[1]))
    spacing_y, spacing_x = (float(spacing_yx_mm[0]), float(spacing_yx_mm[1]))
    if height <= 0 or width <= 0 or spacing_y <= 0.0 or spacing_x <= 0.0:
        raise ValueError("Image shape and physical spacing must be positive")
    return (
        -0.5 * (width - 1) * spacing_x,
        -0.5 * (height - 1) * spacing_y,
    )


def resample_optical_depth_to_planar_simpleitk(
    mu_integral: np.ndarray,
    source_spacing_yx_mm: Tuple[float, float],
    planar_shape_yx: Tuple[int, int],
    planar_spacing_yx_mm: Tuple[float, float],
    blur_fwhm_mm: float = 0.0,
    output_to_source_offset_xy_mm: Tuple[float, float] = (0.0, 0.0),
    clip_range: Tuple[float, float] = CT_FACTOR_CLIP,
) -> Dict[str, Any]:
    """Resample a projected CT optical-depth map onto a planar physical grid.

    The source and output grids are centered at the same physical point because
    the whole-body planar DICOM currently lacks a validated common origin with
    the transformed CT.  Unlike ``resize_map_to_image``, this function preserves
    the physical pixel spacings and crops/pads non-overlapping physical extent.

    ``blur_fwhm_mm`` optionally degrades the CT projection in physical units
    before resampling. ``output_to_source_offset_xy_mm`` is the SimpleITK
    output-point to source-point sampling offset; it is zero for the geometry-
    only comparison and must not be tuned against Q/SPECT activity.
    """
    optical_depth = np.nan_to_num(
        np.asarray(mu_integral, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if optical_depth.ndim != 2:
        raise ValueError("Projected optical depth must be a 2D array")
    source_spacing = tuple(float(value) for value in source_spacing_yx_mm)
    planar_spacing = tuple(float(value) for value in planar_spacing_yx_mm)
    if len(source_spacing) != 2 or len(planar_spacing) != 2:
        raise ValueError("Source and planar spacing must each contain two values")
    if blur_fwhm_mm < 0.0:
        raise ValueError("blur_fwhm_mm must be non-negative")
    if len(output_to_source_offset_xy_mm) != 2:
        raise ValueError("output_to_source_offset_xy_mm must contain x and y")

    source_image = sitk.GetImageFromArray(optical_depth, isVector=False)
    source_image.SetSpacing((source_spacing[1], source_spacing[0]))
    source_image.SetOrigin(_centered_origin_xy_mm(optical_depth.shape, source_spacing))
    source_image.SetDirection((1.0, 0.0, 0.0, 1.0))

    if blur_fwhm_mm > 0.0:
        sigma_mm = float(blur_fwhm_mm) / 2.354820045
        source_image = sitk.SmoothingRecursiveGaussian(source_image, sigma_mm)
    else:
        sigma_mm = 0.0

    planar_height, planar_width = (int(planar_shape_yx[0]), int(planar_shape_yx[1]))
    reference = sitk.Image([planar_width, planar_height], sitk.sitkFloat64)
    reference.SetSpacing((planar_spacing[1], planar_spacing[0]))
    reference.SetOrigin(_centered_origin_xy_mm(planar_shape_yx, planar_spacing))
    reference.SetDirection((1.0, 0.0, 0.0, 1.0))

    transform = sitk.TranslationTransform(2)
    transform.SetOffset(tuple(float(value) for value in output_to_source_offset_xy_mm))
    resampled_image = sitk.Resample(
        source_image,
        reference,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat64,
    )
    resampled_optical_depth = sitk.GetArrayFromImage(resampled_image)
    factor_map = np.exp(0.5 * resampled_optical_depth)
    factor_map = np.clip(factor_map, clip_range[0], clip_range[1])

    source_extent_yx_mm = (
        (optical_depth.shape[0] - 1) * source_spacing[0],
        (optical_depth.shape[1] - 1) * source_spacing[1],
    )
    planar_extent_yx_mm = (
        (planar_height - 1) * planar_spacing[0],
        (planar_width - 1) * planar_spacing[1],
    )
    return {
        "mu_integral_resampled": resampled_optical_depth,
        "factor_map": factor_map,
        "source_spacing_yx_mm": source_spacing,
        "planar_spacing_yx_mm": planar_spacing,
        "source_extent_yx_mm": source_extent_yx_mm,
        "planar_extent_yx_mm": planar_extent_yx_mm,
        "blur_fwhm_mm": float(blur_fwhm_mm),
        "blur_sigma_mm": sigma_mm,
        "output_to_source_offset_xy_mm": tuple(
            float(value) for value in output_to_source_offset_xy_mm
        ),
        "alignment_assumption": "physical-grid center alignment",
    }


def apply_ct_attenuation_correction_to_crop_simpleitk(
    planar_crop: np.ndarray,
    planar_spacing_yx_mm: Tuple[float, float],
    ct: Dict[str, Any],
    conversion_method: str = "water_scaled",
    blur_fwhm_mm: float = 0.0,
    output_to_source_offset_xy_mm: Tuple[float, float] = (0.0, 0.0),
) -> Dict[str, Any]:
    """Apply a center-aligned, physical-grid SimpleITK CT correction to a crop."""
    planar = np.asarray(planar_crop, dtype=np.float64)
    if planar.ndim != 2:
        raise ValueError("Planar crop must be a 2D image")
    if conversion_method == "water_scaled":
        ct_map = ct_attenuation_factor_map(ct)
    elif conversion_method == "raystation_materials":
        ct_map = ct_attenuation_factor_map_raystation_materials(ct)
    else:
        raise ValueError(f"Unknown CT conversion method: {conversion_method}")

    pixel_spacing = ct.get("pixel_spacing") or []
    slice_spacing_mm = ct.get("slice_spacing") or ct.get("slice_thickness")
    if len(pixel_spacing) < 2 or slice_spacing_mm is None:
        raise ValueError("CT in-plane spacing and slice spacing are required")
    source_spacing_yx_mm = (float(slice_spacing_mm), float(pixel_spacing[1]))
    physical = resample_optical_depth_to_planar_simpleitk(
        ct_map["mu_integral"],
        source_spacing_yx_mm=source_spacing_yx_mm,
        planar_shape_yx=planar.shape,
        planar_spacing_yx_mm=planar_spacing_yx_mm,
        blur_fwhm_mm=blur_fwhm_mm,
        output_to_source_offset_xy_mm=output_to_source_offset_xy_mm,
        clip_range=ct_map["clip_range"],
    )
    factor_map = physical["factor_map"]
    corrected_crop = planar * factor_map
    before_counts = float(np.sum(planar))
    after_counts = float(np.sum(corrected_crop))
    effective_factor = after_counts / before_counts if before_counts > 0.0 else np.nan
    return {
        "ct_map": ct_map,
        "ct_factor_resampled": factor_map,
        "ctac_crop": corrected_crop,
        "planar_crop_counts": before_counts,
        "ctac_crop_counts": after_counts,
        "effective_ct_factor": effective_factor,
        "ct_factor_stats": factor_map_statistics(factor_map, ct_map["clip_range"]),
        "conversion_method": conversion_method,
        "resampling_method": "SimpleITK physical-grid center alignment",
        **physical,
    }


def apply_ct_attenuation_correction_to_crop(
    planar_crop: np.ndarray,
    ct: Dict[str, Any],
    conversion_method: str = "water_scaled",
) -> Dict[str, Any]:
    """Apply the CT attenuation-factor map to one already-cropped planar image."""
    if conversion_method == "water_scaled":
        ct_map = ct_attenuation_factor_map(ct)
    elif conversion_method == "raystation_materials":
        ct_map = ct_attenuation_factor_map_raystation_materials(ct)
    else:
        raise ValueError(f"Unknown CT conversion method: {conversion_method}")
    factor_resized = resize_map_to_image(ct_map["factor_map"], planar_crop.shape)
    corrected_crop = np.asarray(planar_crop, dtype=np.float64) * factor_resized
    before_counts = float(np.sum(planar_crop))
    after_counts = float(np.sum(corrected_crop))
    effective_factor = after_counts / before_counts if before_counts > 0 else np.nan
    factor_stats = factor_map_statistics(factor_resized, ct_map["clip_range"])
    return {
        "ct_map": ct_map,
        "ct_factor_resized": factor_resized,
        "ctac_crop": corrected_crop,
        "planar_crop_counts": before_counts,
        "ctac_crop_counts": after_counts,
        "effective_ct_factor": effective_factor,
        "ct_factor_stats": factor_stats,
        "conversion_method": conversion_method,
    }
