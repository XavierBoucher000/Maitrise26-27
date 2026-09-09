"""Global one-patient Lu-177 attenuation model, version 2.

This file intentionally keeps the original patch model unchanged.  The v2
model targets the single effective attenuation factor needed for whole-body
planar activity estimation:

    C_eff = sum(GM_TEW * F_CT) / sum(GM_TEW)
    M_eff = 2 * log(C_eff)

CT is used only to create the development target.  Deployment inputs are
emission-only global spectral ratios over the profile-matched planar crop. The primary
model excludes elapsed time, absolute photopeak intensity, Q/SPECT activity,
and the broad 55.45--166.35 keV window.

Primary predictors
------------------
* log(lower-scatter density / photopeak density)
* log(upper-scatter density / photopeak density)
* absolute AP/PA photopeak log-asymmetry

The broad low-energy window is evaluated only as a named ablation because it
contains the Lu-177 113-keV photopeak, downscatter, and lead-characteristic
radiation; it is not interpreted as a pure attenuation feature.

Validation
----------
Every reported prediction is leave-one-day-out.  Ridge alpha is chosen using
an inner leave-one-day-out loop containing only the four outer-training days.
This is still a five-time-point, one-patient feasibility experiment and is not
evidence of inter-patient generalization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction as ac
import attenuation_window_model as patch_model
import ct_attenuation_correction as ctac
import planar_processing
import qspect_processing


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "attenuation_global_model_v2"
REPORT_PATH = OUTPUT_DIR / "one_patient_global_v2_report.txt"
VALIDATION_FIGURE_PATH = OUTPUT_DIR / "one_patient_global_v2_validation.png"
MODEL_PATH = OUTPUT_DIR / "one_patient_global_v2_model.npz"

PRIMARY_FEATURE_NAMES = (
    "log lower/photopeak",
    "log upper/photopeak",
    "abs AP/PA photopeak asymmetry",
)
BROAD_FEATURE_NAME = "log broad-low-energy/photopeak"
ALL_FEATURE_NAMES = PRIMARY_FEATURE_NAMES + (BROAD_FEATURE_NAME,)
DEFAULT_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
PRIMARY_WINDOWS = ("Lower Scatter", "Photopeak", "Upper Scatter")
BROAD_WINDOW = "Low Energy Scatter"


def _positive(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return np.clip(
        np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        None,
    )


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def line(values: Sequence[str]) -> str:
        return " | ".join(
            str(value).rjust(widths[index])
            for index, value in enumerate(values)
        )

    return [line(headers), line(["-" * width for width in widths])] + [
        line(row) for row in rows
    ]


def global_spectral_features(
    scan: Dict[str, Any],
    crop_bounds: Tuple[int, int],
    require_broad_low_energy: bool = False,
) -> Dict[str, Any]:
    """Return global emission-only predictors and count-rate QC values."""
    crop_top, crop_bottom = (int(crop_bounds[0]), int(crop_bounds[1]))
    groups = ac.group_by_energy_and_view(scan["images"])
    widths = planar_processing.window_widths(scan["images"])
    timing = planar_processing.planar_timing_from_dicom(scan["images"])
    duration_seconds = float(timing.actual_frame_duration_s)
    if duration_seconds <= 0.0:
        raise ValueError("A positive planar frame duration is required")

    missing_primary = [
        energy
        for energy in PRIMARY_WINDOWS
        if energy not in groups
        or "AP" not in groups[energy]
        or "PA" not in groups[energy]
        or energy not in widths
    ]
    if missing_primary:
        raise ValueError(
            "Missing primary AP/PA image or window width for: "
            + ", ".join(missing_primary)
        )
    broad_is_available = (
        BROAD_WINDOW in groups
        and "AP" in groups[BROAD_WINDOW]
        and "PA" in groups[BROAD_WINDOW]
        and BROAD_WINDOW in widths
    )
    if require_broad_low_energy and not broad_is_available:
        raise ValueError(
            "The broad low-energy window is required for the requested ablation"
        )

    density_maps: Dict[str, Dict[str, np.ndarray]] = {}
    available_windows = list(PRIMARY_WINDOWS)
    if broad_is_available:
        available_windows.append(BROAD_WINDOW)
    for energy in available_windows:
        width = float(widths[energy])
        if width <= 0.0:
            raise ValueError(f"Invalid energy-window width for {energy}: {width}")
        ap = _positive(groups[energy]["AP"]["image"])
        pa = _positive(groups[energy]["PA"]["image"])
        density_maps[energy] = {
            "ap": ap / (width * duration_seconds),
            "pa": pa / (width * duration_seconds),
            "gm": ac.geometric_mean(ap, pa) / (width * duration_seconds),
        }

    reference_shape = density_maps["Photopeak"]["gm"].shape
    if crop_top < 0 or crop_bottom <= crop_top or crop_bottom > reference_shape[0]:
        raise ValueError(
            f"Crop {crop_bounds} is invalid for planar shape {reference_shape}"
        )

    crop_slice = (slice(crop_top, crop_bottom), slice(None))

    def density_total(energy: str, view: str = "gm") -> float:
        return float(np.sum(density_maps[energy][view][crop_slice]))

    photopeak = density_total("Photopeak")
    epsilon = max(1e-12, 1e-6 * photopeak)
    feature_values = {
        "log lower/photopeak": float(
            np.log(
                (density_total("Lower Scatter") + epsilon)
                / (photopeak + epsilon)
            )
        ),
        "log upper/photopeak": float(
            np.log(
                (density_total("Upper Scatter") + epsilon)
                / (photopeak + epsilon)
            )
        ),
        "abs AP/PA photopeak asymmetry": float(
            abs(
                0.5
                * np.log(
                    (density_total("Photopeak", "ap") + epsilon)
                    / (density_total("Photopeak", "pa") + epsilon)
                )
            )
        ),
        BROAD_FEATURE_NAME: (
            float(
                np.log(
                    (density_total(BROAD_WINDOW) + epsilon)
                    / (photopeak + epsilon)
                )
            )
            if broad_is_available
            else np.nan
        ),
    }

    def combined_head_rate(energy: str) -> float:
        counts = 0.0
        for view in ("AP", "PA"):
            counts += float(np.sum(_positive(groups[energy][view]["image"])))
        return counts / duration_seconds

    rates = {
        energy: combined_head_rate(energy)
        for energy in available_windows
    }
    primary_windows_rate = float(sum(rates[energy] for energy in PRIMARY_WINDOWS))
    broad_rate = float(rates[BROAD_WINDOW]) if broad_is_available else np.nan
    return {
        "features": feature_values,
        "duration_seconds": duration_seconds,
        "photopeak_rate_cps": rates["Photopeak"],
        "adjacent_scatter_rate_cps": (
            rates["Lower Scatter"] + rates["Upper Scatter"]
        ),
        "primary_windows_rate_cps": primary_windows_rate,
        "broad_low_energy_rate_cps": broad_rate,
        "all_windows_rate_cps": (
            primary_windows_rate + broad_rate
            if broad_is_available
            else primary_windows_rate
        ),
        "broad_low_energy_available": broad_is_available,
    }


def build_global_dataset(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    ct_root: Path = qspect_processing.default_qspect_dir(),
    crop_bounds: Tuple[int, int] = (141, 609),
    crop_strategy: str = "profile_qspect",
    conversion_method: str = ctac.DEFAULT_CT_CONVERSION_METHOD,
) -> Dict[str, Any]:
    """Create one global observation per acquisition day."""
    scans = ac.sorted_planar_scans(planar_dir)
    ct_rows = patch_model.build_ct_reference_rows(
        scans,
        ct_root,
        crop_bounds=crop_bounds,
        crop_strategy=crop_strategy,
        conversion_method=conversion_method,
    )
    if len(scans) != len(ct_rows) or len(scans) < 4:
        raise ValueError("At least four paired planar/raw-ACCT days are required")

    spectral_rows = [
        global_spectral_features(
            scan,
            (int(ct_row["crop_top"]), int(ct_row["crop_bottom"])),
            require_broad_low_energy=True,
        )
        for scan, ct_row in zip(scans, ct_rows)
    ]
    primary_features = np.asarray(
        [
            [spectral["features"][name] for name in PRIMARY_FEATURE_NAMES]
            for spectral in spectral_rows
        ],
        dtype=np.float64,
    )
    broad_features = np.asarray(
        [
            [spectral["features"][name] for name in ALL_FEATURE_NAMES]
            for spectral in spectral_rows
        ],
        dtype=np.float64,
    )
    target_factor = np.asarray(
        [row["effective_ct_factor"] for row in ct_rows],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(target_factor)) or np.any(target_factor <= 0.0):
        raise ValueError("Every CT-derived effective factor must be finite and positive")
    target_optical_depth = 2.0 * np.log(target_factor)

    return {
        "scans": scans,
        "ct_rows": ct_rows,
        "spectral_rows": spectral_rows,
        "primary_features": primary_features,
        "broad_features": broad_features,
        "target_factor": target_factor,
        "target_optical_depth": target_optical_depth,
        "feature_names": PRIMARY_FEATURE_NAMES,
        "broad_feature_names": ALL_FEATURE_NAMES,
        # The saved inference fallback uses the Day0 profile crop. Training and
        # validation use the per-day bounds below.
        "crop_bounds": (
            int(ct_rows[0]["crop_top"]), int(ct_rows[0]["crop_bottom"])
        ),
        "crop_strategy": crop_strategy,
        "crop_bounds_by_day": [
            (int(row["crop_top"]), int(row["crop_bottom"])) for row in ct_rows
        ],
        "planar_dir": Path(planar_dir),
        "ct_root": Path(ct_root),
    }


def _factor_from_optical_depth(optical_depth: np.ndarray | float) -> np.ndarray:
    maximum = 2.0 * np.log(float(ac.CT_FACTOR_CLIP[1]))
    return np.exp(
        0.5 * np.clip(np.asarray(optical_depth, dtype=np.float64), 0.0, maximum)
    )


def _factor_mare(
    true_optical_depth: np.ndarray,
    predicted_optical_depth: np.ndarray,
) -> float:
    true_factor = _factor_from_optical_depth(true_optical_depth)
    predicted_factor = _factor_from_optical_depth(predicted_optical_depth)
    return float(
        np.mean(np.abs(predicted_factor - true_factor) / np.maximum(true_factor, 1e-12))
    )


def _loo_scores_for_alpha(
    features: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    alpha_grid: Sequence[float],
) -> Dict[float, float]:
    """Score alpha using only the supplied training indices."""
    if indices.size < 3:
        raise ValueError("At least three indices are required for inner LOO")
    scores: Dict[float, float] = {}
    for alpha in alpha_grid:
        inner_prediction = []
        inner_target = []
        for validation_index in indices:
            train_indices = indices[indices != validation_index]
            model = patch_model.fit_weighted_ridge(
                features[train_indices],
                target[train_indices],
                np.ones(train_indices.size, dtype=np.float64),
                alpha=float(alpha),
            )
            prediction = patch_model.predict_optical_depth(
                model,
                features[[validation_index]],
            )[0]
            inner_prediction.append(float(prediction))
            inner_target.append(float(target[validation_index]))
        scores[float(alpha)] = _factor_mare(
            np.asarray(inner_target),
            np.asarray(inner_prediction),
        )
    return scores


def _select_alpha(scores: Dict[float, float]) -> float:
    # Prefer the stronger penalty if two scores are numerically tied.
    return min(scores, key=lambda alpha: (scores[alpha], -alpha))


def nested_leave_one_day_out(
    features: np.ndarray,
    target: np.ndarray,
    day_labels: Sequence[str],
    day_offsets: Sequence[float],
    feature_names: Sequence[str],
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> Dict[str, Any]:
    """Outer day LOO with ridge alpha selected inside each training fold."""
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != target.size:
        raise ValueError("Invalid global feature/target shapes")
    if target.size < 4:
        raise ValueError("Nested day-level LOO requires at least four days")

    predictions = np.full(target.shape, np.nan, dtype=np.float64)
    constant_predictions = np.full(target.shape, np.nan, dtype=np.float64)
    folds: List[Dict[str, Any]] = []
    all_indices = np.arange(target.size)

    for held_out in all_indices:
        training = all_indices[all_indices != held_out]
        inner_scores = _loo_scores_for_alpha(
            features,
            target,
            training,
            alpha_grid,
        )
        selected_alpha = _select_alpha(inner_scores)
        model = patch_model.fit_weighted_ridge(
            features[training],
            target[training],
            np.ones(training.size, dtype=np.float64),
            alpha=selected_alpha,
        )
        prediction = float(
            patch_model.predict_optical_depth(model, features[[held_out]])[0]
        )
        constant_prediction = float(np.mean(target[training]))
        predictions[held_out] = prediction
        constant_predictions[held_out] = constant_prediction

        below = features[held_out] < np.min(features[training], axis=0)
        above = features[held_out] > np.max(features[training], axis=0)
        outside_names = [
            str(name)
            for name, outside in zip(feature_names, below | above)
            if bool(outside)
        ]
        true_factor = float(_factor_from_optical_depth(target[held_out]))
        predicted_factor = float(_factor_from_optical_depth(prediction))
        constant_factor = float(_factor_from_optical_depth(constant_prediction))
        folds.append(
            {
                "held_out_index": int(held_out),
                "label": str(day_labels[held_out]),
                "day_offset": float(day_offsets[held_out]),
                "selected_alpha": float(selected_alpha),
                "inner_alpha_scores": inner_scores,
                "true_factor": true_factor,
                "predicted_factor": predicted_factor,
                "constant_factor": constant_factor,
                "relative_error_percent": (
                    100.0 * (predicted_factor - true_factor) / true_factor
                ),
                "constant_relative_error_percent": (
                    100.0 * (constant_factor - true_factor) / true_factor
                ),
                "outside_training_features": outside_names,
            }
        )

    residual = predictions - target
    target_variation = float(np.sum((target - float(np.mean(target))) ** 2))
    return {
        "folds": folds,
        "predictions": predictions,
        "constant_predictions": constant_predictions,
        "factor_mare_percent": 100.0 * _factor_mare(target, predictions),
        "constant_factor_mare_percent": (
            100.0 * _factor_mare(target, constant_predictions)
        ),
        "optical_depth_rmse": float(np.sqrt(np.mean(residual**2))),
        "optical_depth_r2": (
            float(1.0 - np.sum(residual**2) / target_variation)
            if target_variation > 0.0
            else np.nan
        ),
    }


def select_final_alpha(
    features: np.ndarray,
    target: np.ndarray,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> Tuple[float, Dict[float, float]]:
    indices = np.arange(np.asarray(target).size)
    scores = _loo_scores_for_alpha(features, target, indices, alpha_grid)
    return _select_alpha(scores), scores


def save_global_model(
    model: Dict[str, np.ndarray | float],
    dataset: Dict[str, Any],
    alpha_scores: Dict[float, float],
    output_path: Path = MODEL_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_alphas = np.asarray(sorted(alpha_scores), dtype=np.float64)
    np.savez_compressed(
        output_path,
        model_kind=np.asarray("global_adjacent_scatter_ap_pa_v2"),
        intercept=np.asarray(model["intercept"], dtype=np.float64),
        coefficients=np.asarray(model["coefficients"], dtype=np.float64),
        feature_mean=np.asarray(model["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(model["feature_scale"], dtype=np.float64),
        feature_min=np.min(dataset["primary_features"], axis=0),
        feature_max=np.max(dataset["primary_features"], axis=0),
        alpha=np.asarray(model["alpha"], dtype=np.float64),
        feature_names=np.asarray(PRIMARY_FEATURE_NAMES),
        crop_bounds=np.asarray(dataset["crop_bounds"], dtype=np.int64),
        alpha_grid=ordered_alphas,
        alpha_loo_factor_mare=np.asarray(
            [alpha_scores[float(alpha)] for alpha in ordered_alphas],
            dtype=np.float64,
        ),
        primary_windows_rate_range_cps=np.asarray(
            [
                min(
                    spectral["primary_windows_rate_cps"]
                    for spectral in dataset["spectral_rows"]
                ),
                max(
                    spectral["primary_windows_rate_cps"]
                    for spectral in dataset["spectral_rows"]
                ),
            ],
            dtype=np.float64,
        ),
        uses_time=np.asarray(False),
        uses_absolute_photopeak=np.asarray(False),
        uses_broad_low_energy=np.asarray(False),
        uses_ct_at_inference=np.asarray(False),
        uses_qspect=np.asarray(False),
    )
    return output_path


def load_global_model(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    with np.load(model_path, allow_pickle=False) as bundle:
        return {
            "model_kind": str(bundle["model_kind"]),
            "intercept": float(bundle["intercept"]),
            "coefficients": np.asarray(bundle["coefficients"], dtype=np.float64),
            "feature_mean": np.asarray(bundle["feature_mean"], dtype=np.float64),
            "feature_scale": np.asarray(bundle["feature_scale"], dtype=np.float64),
            "feature_min": np.asarray(bundle["feature_min"], dtype=np.float64),
            "feature_max": np.asarray(bundle["feature_max"], dtype=np.float64),
            "alpha": float(bundle["alpha"]),
            "feature_names": tuple(str(name) for name in bundle["feature_names"]),
            "crop_bounds": tuple(int(value) for value in bundle["crop_bounds"]),
            "primary_windows_rate_range_cps": tuple(
                float(value)
                for value in bundle["primary_windows_rate_range_cps"]
            ),
            "uses_time": bool(bundle["uses_time"]),
            "uses_absolute_photopeak": bool(bundle["uses_absolute_photopeak"]),
            "uses_broad_low_energy": bool(bundle["uses_broad_low_energy"]),
            "uses_ct_at_inference": bool(bundle["uses_ct_at_inference"]),
            "uses_qspect": bool(bundle["uses_qspect"]),
        }


def predict_scan_emission_only(
    scan: Dict[str, Any],
    model_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the saved global v2 model without CT, Q/SPECT, or elapsed time."""
    spectral = global_spectral_features(scan, tuple(model_bundle["crop_bounds"]))
    features = np.asarray(
        [[spectral["features"][name] for name in model_bundle["feature_names"]]],
        dtype=np.float64,
    )
    predicted_optical_depth = float(
        patch_model.predict_optical_depth(model_bundle, features)[0]
    )
    factor = float(_factor_from_optical_depth(predicted_optical_depth))
    outside_features = [
        name
        for name, value, minimum, maximum in zip(
            model_bundle["feature_names"],
            features[0],
            model_bundle["feature_min"],
            model_bundle["feature_max"],
        )
        if value < minimum or value > maximum
    ]
    rate_min, rate_max = model_bundle["primary_windows_rate_range_cps"]
    count_rate_outside_range = not (
        rate_min <= spectral["primary_windows_rate_cps"] <= rate_max
    )

    planar_result = ac.tew_geometric_mean_for_scan(scan)
    crop_top, crop_bottom = model_bundle["crop_bounds"]
    planar_crop = _positive(planar_result["image"])[crop_top:crop_bottom, :]
    corrected_crop = planar_crop * factor
    before_counts = float(np.sum(planar_crop))
    after_counts = float(np.sum(corrected_crop))
    corrected_activity = planar_processing.counts_to_activity_mbq(
        after_counts,
        float(planar_result["timing"].local_dwell_time_s),
        planar_processing.CAMERA_SENSITIVITY_CPS_PER_MBQ,
    )
    return {
        "features": spectral["features"],
        "predicted_optical_depth": predicted_optical_depth,
        "effective_factor": factor,
        "planar_crop": planar_crop,
        "corrected_crop": corrected_crop,
        "planar_crop_counts": before_counts,
        "corrected_crop_counts": after_counts,
        "corrected_activity_mbq": float(corrected_activity),
        "count_rate_qc": spectral,
        "outside_training_features": outside_features,
        "is_feature_extrapolation": bool(outside_features),
        "count_rate_outside_training_range": count_rate_outside_range,
        "training_primary_windows_rate_range_cps": (rate_min, rate_max),
        "uses_time": False,
        "uses_ct": False,
        "uses_qspect": False,
    }


def write_report(
    dataset: Dict[str, Any],
    primary_validation: Dict[str, Any],
    broad_validation: Dict[str, Any],
    full_model: Dict[str, np.ndarray | float],
    final_alpha_scores: Dict[float, float],
    output_path: Path = REPORT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    broad_by_index = {
        fold["held_out_index"]: fold for fold in broad_validation["folds"]
    }
    validation_rows = []
    for fold in primary_validation["folds"]:
        broad_fold = broad_by_index[fold["held_out_index"]]
        outside = ", ".join(fold["outside_training_features"]) or "none"
        validation_rows.append(
            [
                fold["label"],
                f"{fold['true_factor']:.3f}",
                f"{fold['predicted_factor']:.3f}",
                f"{fold['relative_error_percent']:.1f}",
                f"{broad_fold['predicted_factor']:.3f}",
                f"{fold['constant_factor']:.3f}",
                f"{fold['selected_alpha']:.1f}",
                outside,
            ]
        )

    feature_rows = []
    for ct_row, spectral in zip(dataset["ct_rows"], dataset["spectral_rows"]):
        feature_rows.append(
            [
                ct_row["day_label"],
                f"{spectral['features'][PRIMARY_FEATURE_NAMES[0]]:.4f}",
                f"{spectral['features'][PRIMARY_FEATURE_NAMES[1]]:.4f}",
                f"{spectral['features'][PRIMARY_FEATURE_NAMES[2]]:.4f}",
                f"{spectral['features'][BROAD_FEATURE_NAME]:.4f}",
            ]
        )

    count_rate_rows = []
    for ct_row, spectral in zip(dataset["ct_rows"], dataset["spectral_rows"]):
        count_rate_rows.append(
            [
                ct_row["day_label"],
                f"{spectral['photopeak_rate_cps']:.0f}",
                f"{spectral['adjacent_scatter_rate_cps']:.0f}",
                f"{spectral['broad_low_energy_rate_cps']:.0f}",
                f"{spectral['all_windows_rate_cps']:.0f}",
            ]
        )

    coefficient_rows = [
        [name, f"{coefficient:.6f}"]
        for name, coefficient in zip(
            PRIMARY_FEATURE_NAMES,
            np.asarray(full_model["coefficients"]),
        )
    ]
    alpha_rows = [
        [f"{alpha:.1f}", f"{100.0 * final_alpha_scores[alpha]:.3f}"]
        for alpha in sorted(final_alpha_scores)
    ]
    last_fold = max(
        primary_validation["folds"],
        key=lambda fold: fold["day_offset"],
    )
    first_fold = min(
        primary_validation["folds"],
        key=lambda fold: fold["day_offset"],
    )
    chronological_rates = [
        spectral["all_windows_rate_cps"]
        for spectral in dataset["spectral_rows"]
    ]
    first_to_second_rate_ratio = (
        chronological_rates[0] / chronological_rates[1]
        if len(chronological_rates) > 1 and chronological_rates[1] > 0.0
        else np.nan
    )
    first_to_last_rate_ratio = (
        chronological_rates[0] / chronological_rates[-1]
        if chronological_rates[-1] > 0.0
        else np.nan
    )
    absolute_errors = np.asarray(
        [abs(fold["relative_error_percent"]) for fold in primary_validation["folds"]]
    )

    lines = [
        "One-patient Lu-177 global attenuation model v2",
        "================================================",
        "",
        "Primary model:",
        "  Predict one CT-derived effective attenuation factor per acquisition.",
        "  Inputs = adjacent 208-keV scatter/photopeak ratios + AP/PA asymmetry.",
        "  No elapsed time, absolute photopeak intensity, broad-low-energy input, or Q/SPECT.",
        "  Raw ACCT is used only to create development targets.",
        "",
        "Target:",
        "  C_eff = sum(GM_TEW * F_CT) / sum(GM_TEW)",
        "  M_eff = 2*log(C_eff)",
        "",
        "Validation:",
        "  Outer leave-one-day-out; alpha selected by inner LOO on outer-training days only.",
        "  Each day contributes one observation; patches are not treated as independent samples.",
        f"  Crop strategy = {dataset['crop_strategy']}.",
        "  Crop bounds by day = " + ", ".join(
            f"{row['day_label']}[{row['crop_top']}:{row['crop_bottom']}]"
            for row in dataset["ct_rows"]
        ),
        "",
    ]
    lines.extend(
        _format_table(
            [
                "held out",
                "CT ref",
                "v2 primary",
                "error %",
                "+ broad ablation",
                "constant",
                "alpha",
                "outside training range",
            ],
            validation_rows,
        )
    )
    lines.extend(
        [
            "",
            "Overall day-level results:",
            f"  primary nested-LOO factor MARE = {primary_validation['factor_mare_percent']:.3f}%",
            f"  primary median absolute factor error = {np.median(absolute_errors):.3f}%",
            f"  primary maximum absolute factor error = {np.max(absolute_errors):.3f}%",
            f"  broad-window ablation factor MARE = {broad_validation['factor_mare_percent']:.3f}%",
            f"  constant baseline factor MARE = {primary_validation['constant_factor_mare_percent']:.3f}%",
            f"  primary optical-depth RMSE = {primary_validation['optical_depth_rmse']:.4f}",
            f"  primary optical-depth R2 = {primary_validation['optical_depth_r2']:.4f}",
            "",
            "Deployment-like final-day check (train earlier days, hold out latest day):",
            f"  {last_fold['label']} CT reference factor = {last_fold['true_factor']:.4f}",
            f"  {last_fold['label']} predicted factor = {last_fold['predicted_factor']:.4f}",
            f"  signed relative error = {last_fold['relative_error_percent']:.2f}%",
            f"  selected alpha = {last_fold['selected_alpha']:.1f}",
            "",
            "Global emission features by day:",
        ]
    )
    lines.extend(
        _format_table(
            ["day", "log L/P", "log U/P", "abs AP/PA", "log broad/P"],
            feature_rows,
        )
    )
    lines.extend(["", "Combined-head count-rate QC (counts/s):"])
    lines.extend(
        _format_table(
            ["day", "photopeak", "adjacent scatter", "broad low", "all windows"],
            count_rate_rows,
        )
    )
    lines.extend(
        [
            "",
            "Count-rate warning:",
            f"  Day0 all-window rate is {first_to_second_rate_ratio:.2f}x Day1 and {first_to_last_rate_ratio:.2f}x the last day.",
            f"  Day0 is also the worst held-out primary prediction ({first_fold['relative_error_percent']:+.1f}%).",
            "  This coincidence cannot distinguish attenuation from count-rate, dead-time, pile-up, or biodistribution effects.",
        ]
    )
    lines.extend(["", "Final-model alpha selection on all five development days:"])
    lines.extend(_format_table(["alpha", "LOO factor MARE %"], alpha_rows))
    lines.extend(
        [
            f"  selected alpha = {float(full_model['alpha']):.1f}",
            "",
            "Final standardized ridge coefficients:",
        ]
    )
    lines.extend(_format_table(["feature", "coefficient"], coefficient_rows))
    lines.extend(
        [
            "",
            "Interpretation limits:",
            "  - Five days from one patient are a feasibility test, not independent validation.",
            "  - Model variants were examined on these same five days; reported errors remain exploratory.",
            "  - The broad 55.45-166.35 keV channel is an ablation, not pure scatter.",
            "  - Dead-time/pile-up correction is not applied; count-rate QC is reported instead.",
            "  - CT-to-planar mapping remains approximate resizing inside each profile-matched crop.",
            "  - Q/SPECT supplies crop geometry during development; an emission-only crop localizer",
            "    is still required for deployment without Q/SPECT.",
            f"  - CT target conversion = {dataset['ct_rows'][0]['ct_conversion_method']}.",
            "  - The default Catphan T2 curve is provisional until the QC reconstruction matches the patient ACCT protocol.",
            "  - A new patient requires external validation before this factor can be used quantitatively.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def plot_validation(
    dataset: Dict[str, Any],
    primary_validation: Dict[str, Any],
    broad_validation: Dict[str, Any],
    output_path: Path = VALIDATION_FIGURE_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    primary_folds = sorted(
        primary_validation["folds"],
        key=lambda fold: fold["day_offset"],
    )
    broad_by_index = {
        fold["held_out_index"]: fold for fold in broad_validation["folds"]
    }
    days = np.asarray([fold["day_offset"] for fold in primary_folds])
    true_factor = np.asarray([fold["true_factor"] for fold in primary_folds])
    primary_factor = np.asarray(
        [fold["predicted_factor"] for fold in primary_folds]
    )
    broad_factor = np.asarray(
        [broad_by_index[fold["held_out_index"]]["predicted_factor"] for fold in primary_folds]
    )
    constant_factor = np.asarray(
        [fold["constant_factor"] for fold in primary_folds]
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor="white")
    axes[0].plot(days, true_factor, marker="o", linewidth=2, label="CT-derived reference")
    axes[0].plot(days, primary_factor, marker="s", linewidth=2, label="Global v2 primary")
    axes[0].plot(
        days,
        constant_factor,
        marker="^",
        linestyle="--",
        color="0.45",
        label="Constant baseline",
    )
    axes[0].set_xlabel("Time after first acquisition (days)")
    axes[0].set_ylabel("Effective attenuation factor")
    axes[0].set_title("Nested leave-one-day-out")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(0.0, color="0.25", linewidth=1)
    axes[1].plot(
        days,
        100.0 * (primary_factor - true_factor) / true_factor,
        marker="s",
        linewidth=2,
        label="Primary",
    )
    axes[1].plot(
        days,
        100.0 * (broad_factor - true_factor) / true_factor,
        marker="D",
        linewidth=1.5,
        label="+ broad ablation",
    )
    axes[1].plot(
        days,
        100.0 * (constant_factor - true_factor) / true_factor,
        marker="^",
        linestyle="--",
        color="0.45",
        label="Constant",
    )
    axes[1].set_xlabel("Time after first acquisition (days)")
    axes[1].set_ylabel("Signed factor error (%)")
    axes[1].set_title("Held-out error by day")
    axes[1].legend(frameon=False, fontsize=8)

    descriptive_features = np.asarray(dataset["broad_features"], dtype=np.float64)
    feature_mean = np.mean(descriptive_features, axis=0)
    feature_scale = np.std(descriptive_features, axis=0)
    standardized = (descriptive_features - feature_mean) / np.maximum(feature_scale, 1e-12)
    for index, name in enumerate(ALL_FEATURE_NAMES):
        axes[2].plot(days, standardized[:, index], marker="o", linewidth=1.5, label=name)
    axes[2].set_xlabel("Time after first acquisition (days)")
    axes[2].set_ylabel("Descriptive standardized value")
    axes[2].set_title("Emission features over time")
    axes[2].legend(frameon=False, fontsize=7)

    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def run_model(
    planar_dir: Path | None = None,
    ct_root: Path | None = None,
    crop_bounds: Tuple[int, int] = (141, 609),
    crop_strategy: str = "profile_qspect",
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    conversion_method: str = ctac.DEFAULT_CT_CONVERSION_METHOD,
) -> Dict[str, Any]:
    alpha_grid = tuple(float(alpha) for alpha in alpha_grid)
    if not alpha_grid or any(alpha < 0.0 for alpha in alpha_grid):
        raise ValueError("Alpha grid must contain non-negative values")
    dataset = build_global_dataset(
        planar_dir=(planar_dir or planar_processing.default_planar_study_dir()),
        ct_root=(ct_root or qspect_processing.default_qspect_dir()),
        crop_bounds=crop_bounds,
        crop_strategy=crop_strategy,
        conversion_method=conversion_method,
    )
    day_labels = [row["day_label"] for row in dataset["ct_rows"]]
    day_offsets = [row["day_offset"] for row in dataset["ct_rows"]]
    target = dataset["target_optical_depth"]

    primary_validation = nested_leave_one_day_out(
        dataset["primary_features"],
        target,
        day_labels,
        day_offsets,
        PRIMARY_FEATURE_NAMES,
        alpha_grid,
    )
    broad_validation = nested_leave_one_day_out(
        dataset["broad_features"],
        target,
        day_labels,
        day_offsets,
        ALL_FEATURE_NAMES,
        alpha_grid,
    )
    final_alpha, final_alpha_scores = select_final_alpha(
        dataset["primary_features"],
        target,
        alpha_grid,
    )
    full_model = patch_model.fit_weighted_ridge(
        dataset["primary_features"],
        target,
        np.ones(target.size, dtype=np.float64),
        alpha=final_alpha,
    )
    report_path = write_report(
        dataset,
        primary_validation,
        broad_validation,
        full_model,
        final_alpha_scores,
    )
    figure_path = plot_validation(
        dataset,
        primary_validation,
        broad_validation,
    )
    model_path = save_global_model(
        full_model,
        dataset,
        final_alpha_scores,
    )

    last_fold = max(
        primary_validation["folds"],
        key=lambda fold: fold["day_offset"],
    )
    print(f"Saved global-v2 report: {report_path}")
    print(f"Saved global-v2 validation figure: {figure_path}")
    print(f"Saved global-v2 emission-only model: {model_path}")
    print(
        "Nested LOO: "
        f"primary MARE={primary_validation['factor_mare_percent']:.2f}%, "
        f"broad ablation={broad_validation['factor_mare_percent']:.2f}%, "
        f"constant={primary_validation['constant_factor_mare_percent']:.2f}%"
    )
    print(
        f"Latest-day holdout ({last_fold['label']}): "
        f"predicted={last_fold['predicted_factor']:.3f}, "
        f"CT reference={last_fold['true_factor']:.3f}, "
        f"error={last_fold['relative_error_percent']:.1f}%"
    )
    return {
        "dataset": dataset,
        "primary_validation": primary_validation,
        "broad_validation": broad_validation,
        "full_model": full_model,
        "model_path": model_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Global CT-guided, emission-only-at-inference Lu-177 attenuation model v2."
    )
    parser.add_argument("--crop-top", type=int, default=141)
    parser.add_argument("--crop-bottom", type=int, default=609)
    parser.add_argument(
        "--crop-strategy",
        choices=("profile_qspect", "fixed"),
        default="profile_qspect",
    )
    parser.add_argument("--planar-dir", type=Path)
    parser.add_argument("--ct-root", type=Path)
    parser.add_argument(
        "--ct-conversion-method",
        choices=("water_scaled", "raystation_materials", "catphan_t2_110kvp"),
        default=ctac.DEFAULT_CT_CONVERSION_METHOD,
    )
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHA_GRID),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_model(
        planar_dir=arguments.planar_dir,
        ct_root=arguments.ct_root,
        crop_bounds=(arguments.crop_top, arguments.crop_bottom),
        crop_strategy=arguments.crop_strategy,
        alpha_grid=arguments.alpha_grid,
        conversion_method=arguments.ct_conversion_method,
    )
