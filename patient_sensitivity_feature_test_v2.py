"""Stronger one-patient sensitivity and univariate planar-feature audit.

The Q/SPECT reconstructed-volume activity is the quantitative target.  No CT
image or voxel value is used.  The analysis evaluates whether one
emission-derived planar feature can predict the effective sensitivity required
to recover Q/SPECT:

    S_eff(t) = R_planar(t) / A_QSPECT(t)
    A_hat(t) = R_planar(t) / S_eff_hat(t)

This remains a feasibility analysis, not an independent validation.  The
broad-low-energy/photopeak feature was discovered on patient 1 and must be
locked before applying the workflow to another patient.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import acquisition_timeline
import attenuation_correction as ac
import attenuation_global_model_v2 as spectral_model
import patient_effective_sensitivity as sensitivity
import planar_processing
import qspect_processing


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "fig" / "patient_sensitivity_features_v2"
LOCKED_FEATURE = spectral_model.BROAD_FEATURE_NAME
ELIGIBLE_FEATURES = (
    *spectral_model.ALL_FEATURE_NAMES,
    "TEW retained fraction",
)
CROP_OFFSETS_PIXELS = (-10, -5, 0, 5, 10)


def _fit_log_linear_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: float,
) -> Tuple[float, float]:
    """Fit log(y)=b0+b1*x and return prediction and slope."""
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if x_train.ndim != 1 or y_train.shape != x_train.shape or x_train.size < 2:
        raise ValueError("At least two paired training values are required")
    if np.any(~np.isfinite(x_train)) or np.any(~np.isfinite(y_train)) or np.any(y_train <= 0):
        raise ValueError("Training feature must be finite and target must be positive")
    if float(np.ptp(x_train)) <= np.finfo(float).eps:
        return float(np.exp(np.mean(np.log(y_train)))), 0.0
    design = np.column_stack([np.ones(x_train.size), x_train])
    intercept, slope = np.linalg.lstsq(design, np.log(y_train), rcond=None)[0]
    return float(np.exp(intercept + slope * float(x_test))), float(slope)


def _prediction_metrics(
    target_sensitivity: np.ndarray,
    predicted_sensitivity: np.ndarray,
) -> Dict[str, float]:
    """Return errors for sensitivity and the resulting activity estimate."""
    target = np.asarray(target_sensitivity, dtype=np.float64)
    predicted = np.asarray(predicted_sensitivity, dtype=np.float64)
    sensitivity_error = 100.0 * (predicted - target) / target
    # R / predicted_S divided by R / true_S; count rate cancels exactly.
    activity_error = 100.0 * (target / predicted - 1.0)
    return {
        "sensitivity_bias_percent": float(np.mean(sensitivity_error)),
        "sensitivity_mare_percent": float(np.mean(np.abs(sensitivity_error))),
        "activity_bias_percent": float(np.mean(activity_error)),
        "activity_mare_percent": float(np.mean(np.abs(activity_error))),
        "activity_rmse_percent": float(np.sqrt(np.mean(activity_error**2))),
        "maximum_activity_error_percent": float(np.max(np.abs(activity_error))),
    }


def leave_one_out_log_linear(
    feature: Sequence[float],
    target_sensitivity: Sequence[float],
) -> Dict[str, Any]:
    """Evaluate one pre-specified feature with leave-one-timepoint-out."""
    x = np.asarray(feature, dtype=np.float64)
    y = np.asarray(target_sensitivity, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 3:
        raise ValueError("feature and target must be same-length 1D arrays with >=3 points")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(y <= 0.0):
        raise ValueError("feature must be finite and target must be finite and positive")

    predictions = np.empty_like(y)
    constant_predictions = np.empty_like(y)
    slopes = np.empty_like(y)
    extrapolated = np.zeros(y.size, dtype=bool)
    indices = np.arange(y.size)
    for held_out in indices:
        training = indices != held_out
        predictions[held_out], slopes[held_out] = _fit_log_linear_predict(
            x[training], y[training], x[held_out]
        )
        constant_predictions[held_out] = float(np.mean(y[training]))
        extrapolated[held_out] = bool(
            x[held_out] < np.min(x[training]) or x[held_out] > np.max(x[training])
        )

    correlation = (
        np.nan
        if float(np.ptp(x)) <= np.finfo(float).eps
        else float(np.corrcoef(x, np.log(y))[0, 1])
    )
    metrics = _prediction_metrics(y, predictions)
    baseline_metrics = _prediction_metrics(y, constant_predictions)
    return {
        "predictions": predictions,
        "constant_predictions": constant_predictions,
        "slopes": slopes,
        "extrapolated": extrapolated,
        "extrapolated_folds": int(np.count_nonzero(extrapolated)),
        "correlation_with_log_target": correlation,
        "slope_min": float(np.min(slopes)),
        "slope_max": float(np.max(slopes)),
        "slope_sign_consistent": bool(np.all(slopes >= 0) or np.all(slopes <= 0)),
        **metrics,
        **{f"constant_{key}": value for key, value in baseline_metrics.items()},
    }


def nested_univariate_feature_selection(
    feature_values: Mapping[str, Sequence[float]],
    target_sensitivity: Sequence[float],
    candidate_names: Sequence[str] = ELIGIBLE_FEATURES,
) -> Dict[str, Any]:
    """Select the feature inside each outer LOO fold, then predict that fold."""
    y = np.asarray(target_sensitivity, dtype=np.float64)
    if y.ndim != 1 or y.size < 5 or np.any(y <= 0) or np.any(~np.isfinite(y)):
        raise ValueError("Nested feature selection requires at least five positive targets")
    features = {
        name: np.asarray(feature_values[name], dtype=np.float64)
        for name in candidate_names
    }
    if any(values.shape != y.shape or np.any(~np.isfinite(values)) for values in features.values()):
        raise ValueError("Every candidate feature must contain one finite value per target")

    indices = np.arange(y.size)
    predictions = np.empty_like(y)
    selected_features: List[str] = []
    inner_scores_by_fold: List[Dict[str, float]] = []
    extrapolated = np.zeros(y.size, dtype=bool)
    for held_out in indices:
        outer_training = indices[indices != held_out]
        scores: Dict[str, float] = {}
        for name in candidate_names:
            x = features[name]
            inner_predictions = []
            inner_targets = []
            for inner_validation in outer_training:
                inner_training = outer_training[outer_training != inner_validation]
                prediction, _slope = _fit_log_linear_predict(
                    x[inner_training], y[inner_training], x[inner_validation]
                )
                inner_predictions.append(prediction)
                inner_targets.append(y[inner_validation])
            scores[name] = _prediction_metrics(
                np.asarray(inner_targets), np.asarray(inner_predictions)
            )["activity_mare_percent"]
        selected = min(candidate_names, key=lambda name: (scores[name], name))
        x = features[selected]
        predictions[held_out], _slope = _fit_log_linear_predict(
            x[outer_training], y[outer_training], x[held_out]
        )
        extrapolated[held_out] = bool(
            x[held_out] < np.min(x[outer_training])
            or x[held_out] > np.max(x[outer_training])
        )
        selected_features.append(selected)
        inner_scores_by_fold.append(scores)

    return {
        "predictions": predictions,
        "selected_features": selected_features,
        "inner_scores_by_fold": inner_scores_by_fold,
        "extrapolated": extrapolated,
        "extrapolated_folds": int(np.count_nonzero(extrapolated)),
        **_prediction_metrics(y, predictions),
    }


def _tew_and_raw_rates(
    scan: Dict[str, Any],
    crop_bounds: Tuple[int, int],
) -> Tuple[float, float]:
    images = sensitivity._emission_images(  # shared processing used by the sensitivity audit
        scan, align_pa_to_ap=planar_processing.ALIGN_PA_TO_AP
    )
    top, bottom = crop_bounds
    dwell = float(images["timing"].local_dwell_time_s)
    raw_rate = float(np.sum(images["raw_gm"][top:bottom, :])) / dwell
    tew_rate = float(np.sum(images["tew_gm"][top:bottom, :])) / dwell
    return raw_rate, tew_rate


def planar_tew_poisson_relative_uncertainty(
    scan: Dict[str, Any],
    crop_bounds: Tuple[int, int],
) -> float:
    """First-order planar Poisson uncertainty after per-head TEW and GM.

    This propagates independent Poisson variance from the three TEW windows of
    each detector.  It does not include Q/SPECT calibration, reconstruction,
    registration, or systematic uncertainty, and clipping makes it an
    approximation in very low-count pixels.
    """
    groups = ac.group_by_energy_and_view(scan["images"])
    widths = planar_processing.window_widths(scan["images"])
    w_photo = float(widths["Photopeak"])
    coefficients = {
        "Lower Scatter": w_photo / (2.0 * float(widths["Lower Scatter"])),
        "Upper Scatter": w_photo / (2.0 * float(widths["Upper Scatter"])),
    }

    corrected: Dict[str, np.ndarray] = {}
    variances: Dict[str, np.ndarray] = {}
    for view in ("AP", "PA"):
        photo = np.asarray(groups["Photopeak"][view]["image"], dtype=np.float64)
        lower = np.asarray(groups["Lower Scatter"][view]["image"], dtype=np.float64)
        upper = np.asarray(groups["Upper Scatter"][view]["image"], dtype=np.float64)
        a_lower = coefficients["Lower Scatter"]
        a_upper = coefficients["Upper Scatter"]
        corrected[view] = np.clip(photo - a_lower * lower - a_upper * upper, 0.0, None)
        variances[view] = np.clip(photo, 0.0, None) + a_lower**2 * np.clip(lower, 0.0, None) + a_upper**2 * np.clip(upper, 0.0, None)

    ap = corrected["AP"]
    var_ap = variances["AP"]
    pa = corrected["PA"]
    var_pa = variances["PA"]
    if planar_processing.ALIGN_PA_TO_AP:
        pa = np.fliplr(pa)
        var_pa = np.fliplr(var_pa)

    valid = (ap > 0.0) & (pa > 0.0)
    gm = np.zeros_like(ap)
    gm[valid] = np.sqrt(ap[valid] * pa[valid])
    variance_gm = np.zeros_like(ap)
    variance_gm[valid] = 0.25 * (
        (pa[valid] / ap[valid]) * var_ap[valid]
        + (ap[valid] / pa[valid]) * var_pa[valid]
    )
    top, bottom = crop_bounds
    total = float(np.sum(gm[top:bottom, :]))
    variance_total = float(np.sum(variance_gm[top:bottom, :]))
    if total <= 0.0:
        raise ValueError("The TEW geometric-mean crop contains no positive counts")
    return 100.0 * np.sqrt(max(variance_total, 0.0)) / total


def _crop_robustness(
    scans: Sequence[Dict[str, Any]],
    sensitivity_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for offset in CROP_OFFSETS_PIXELS:
        feature = []
        target = []
        for scan, row in zip(scans, sensitivity_rows):
            height = np.asarray(scan["images"][0]["image"]).shape[0]
            top = int(row["crop_top"]) + offset
            bottom = int(row["crop_bottom"]) + offset
            if top < 0:
                bottom -= top
                top = 0
            if bottom > height:
                top -= bottom - height
                bottom = height
            bounds = (max(0, top), min(height, bottom))
            spectral = spectral_model.global_spectral_features(
                scan, bounds, require_broad_low_energy=True
            )
            _raw_rate, tew_rate = _tew_and_raw_rates(scan, bounds)
            feature.append(spectral["features"][LOCKED_FEATURE])
            target.append(tew_rate / float(row["qspect_activity_at_planar_time_mbq"]))
        metrics = leave_one_out_log_linear(feature, target)
        output.append(
            {
                "crop_offset_pixels": offset,
                "crop_offset_cm": offset * 0.23975999355316,
                "activity_mare_percent": metrics["activity_mare_percent"],
                "maximum_activity_error_percent": metrics["maximum_activity_error_percent"],
            }
        )
    return output


def build_analysis(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
) -> Dict[str, Any]:
    """Build and evaluate the CT-free one-patient development dataset."""
    sensitivity_rows = sensitivity.calculate_patient_effective_sensitivity(
        planar_dir=Path(planar_dir),
        qspect_dir=Path(qspect_dir),
        crop_strategy="profile_qspect",
    )
    scans = ac.sorted_planar_scans(Path(planar_dir))
    if len(scans) != len(sensitivity_rows):
        raise ValueError("Planar scans and sensitivity rows have different lengths")

    spectral_rows = []
    for scan, row in zip(scans, sensitivity_rows):
        bounds = (int(row["crop_top"]), int(row["crop_bottom"]))
        spectral_rows.append(
            spectral_model.global_spectral_features(
                scan, bounds, require_broad_low_energy=True
            )
        )

    days = np.asarray([row["planar_day"] for row in sensitivity_rows], dtype=np.float64)
    raw_rates = np.asarray([row["raw_photopeak_gm_cps"] for row in sensitivity_rows])
    tew_rates = np.asarray([row["tew_gm_cps"] for row in sensitivity_rows])
    features: Dict[str, np.ndarray] = {
        name: np.asarray([row["features"][name] for row in spectral_rows], dtype=np.float64)
        for name in spectral_model.ALL_FEATURE_NAMES
    }
    features["TEW retained fraction"] = np.asarray(
        [row["tew_retained_fraction"] for row in sensitivity_rows], dtype=np.float64
    )
    features["elapsed time"] = days
    features["log raw photopeak rate"] = np.log(raw_rates)
    features["log TEW count rate"] = np.log(tew_rates)

    targets = {
        "raw": np.asarray(
            [row["raw_effective_sensitivity_cps_per_mbq"] for row in sensitivity_rows],
            dtype=np.float64,
        ),
        "tew": np.asarray(
            [row["tew_effective_sensitivity_cps_per_mbq"] for row in sensitivity_rows],
            dtype=np.float64,
        ),
    }
    rates = {"raw": raw_rates, "tew": tew_rates}
    results: Dict[str, Dict[str, Any]] = {}
    for target_name, target in targets.items():
        target_results: Dict[str, Any] = {}
        for name in ELIGIBLE_FEATURES:
            target_results[name] = leave_one_out_log_linear(features[name], target)
        target_results["elapsed time"] = leave_one_out_log_linear(features["elapsed time"], target)
        rate_feature = "log raw photopeak rate" if target_name == "raw" else "log TEW count rate"
        target_results[rate_feature] = leave_one_out_log_linear(features[rate_feature], target)
        results[target_name] = target_results

    nested = (
        nested_univariate_feature_selection(features, targets["tew"])
        if targets["tew"].size >= 5
        else None
    )
    crop_robustness = _crop_robustness(scans, sensitivity_rows)
    poisson_relative_percent = np.asarray(
        [
            planar_tew_poisson_relative_uncertainty(
                scan, (int(row["crop_top"]), int(row["crop_bottom"]))
            )
            for scan, row in zip(scans, sensitivity_rows)
        ],
        dtype=np.float64,
    )

    # Timing sensitivity: retain the PT DICOM reference as the main result, but
    # also test the alternative assumption that START means the first raw WB bed.
    timeline = acquisition_timeline.collect_acquisition_timeline(planar_dir, qspect_dir)
    wb_start_by_label = {row["label"]: row["qspect_wb_start"] for row in timeline}
    alternative_tew_target = []
    alternative_target_shift = []
    for row in sensitivity_rows:
        wb_start = wb_start_by_label.get(row["label"])
        if wb_start is None:
            alternative_tew_target.append(row["tew_effective_sensitivity_cps_per_mbq"])
            alternative_target_shift.append(0.0)
            continue
        alternative_activity = sensitivity.activity_at_target_time_physical_decay(
            row["qspect_activity_measured_mbq"],
            reference_time=wb_start,
            target_time=row["planar_datetime"],
        )
        alternative_sensitivity = row["tew_gm_cps"] / alternative_activity
        alternative_tew_target.append(alternative_sensitivity)
        alternative_target_shift.append(
            100.0
            * (alternative_sensitivity / row["tew_effective_sensitivity_cps_per_mbq"] - 1.0)
        )
    alternative_tew_target_array = np.asarray(alternative_tew_target)
    timing_locked_result = leave_one_out_log_linear(
        features[LOCKED_FEATURE], alternative_tew_target_array
    )

    return {
        "labels": [row["label"] for row in sensitivity_rows],
        "days": days,
        "sensitivity_rows": sensitivity_rows,
        "scans": scans,
        "spectral_rows": spectral_rows,
        "features": features,
        "targets": targets,
        "rates": rates,
        "results": results,
        "nested": nested,
        "crop_robustness": crop_robustness,
        "poisson_relative_percent": poisson_relative_percent,
        "alternative_tew_target": alternative_tew_target_array,
        "alternative_target_shift_percent": np.asarray(alternative_target_shift),
        "timing_locked_result": timing_locked_result,
    }


def _method_rows(analysis: Dict[str, Any], target_name: str) -> List[Dict[str, Any]]:
    rows = []
    for name, result in analysis["results"][target_name].items():
        category = (
            "eligible"
            if name in ELIGIBLE_FEATURES
            else "time diagnostic"
            if name == "elapsed time"
            else "absolute-rate diagnostic"
        )
        rows.append({"method": name, "category": category, **result})
    return rows


def write_outputs(
    analysis: Dict[str, Any],
    output_dir: Path,
    patient_label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "v2_univariate_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "target", "method", "category", "correlation_with_log_target",
            "activity_bias_percent", "activity_mare_percent", "activity_rmse_percent",
            "maximum_activity_error_percent", "constant_activity_mare_percent",
            "extrapolated_folds", "slope_min", "slope_max", "slope_sign_consistent",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for target_name in ("raw", "tew"):
            for row in _method_rows(analysis, target_name):
                writer.writerow({key: target_name if key == "target" else row[key] for key in fieldnames})

    locked = analysis["results"]["tew"][LOCKED_FEATURE]
    time_only = analysis["results"]["tew"]["elapsed time"]
    target = analysis["targets"]["tew"]
    rate = analysis["rates"]["tew"]
    nested = analysis["nested"]
    timepoint_path = output_dir / "v2_timepoint_predictions.csv"
    with timepoint_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "label", "day", "crop_top", "crop_bottom", "qspect_activity_at_planar_mbq",
                "tew_rate_cps", "tew_sensitivity", LOCKED_FEATURE,
                "locked_predicted_sensitivity", "locked_predicted_activity_mbq",
                "locked_activity_error_percent", "time_predicted_activity_mbq",
                "nested_selected_feature", "nested_predicted_activity_mbq",
                "planar_poisson_relative_uncertainty_percent",
            ]
        )
        for index, row in enumerate(analysis["sensitivity_rows"]):
            locked_activity = rate[index] / locked["predictions"][index]
            time_activity = rate[index] / time_only["predictions"][index]
            nested_activity = "" if nested is None else rate[index] / nested["predictions"][index]
            writer.writerow(
                [
                    row["label"], row["planar_day"], row["crop_top"], row["crop_bottom"],
                    rate[index] / target[index], rate[index], target[index],
                    analysis["features"][LOCKED_FEATURE][index], locked["predictions"][index],
                    locked_activity, 100.0 * (locked_activity / (rate[index] / target[index]) - 1.0),
                    time_activity, "" if nested is None else nested["selected_features"][index],
                    nested_activity, analysis["poisson_relative_percent"][index],
                ]
            )

    robustness_path = output_dir / "v2_robustness.csv"
    with robustness_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(analysis["crop_robustness"][0].keys()))
        writer.writeheader()
        writer.writerows(analysis["crop_robustness"])

    constant_mare = locked["constant_activity_mare_percent"]
    eligible_ranked = sorted(
        ((name, analysis["results"]["tew"][name]) for name in ELIGIBLE_FEATURES),
        key=lambda item: item[1]["activity_mare_percent"],
    )
    lines = [
        f"Improved planar sensitivity/univariate audit — {patient_label}",
        "=" * (47 + len(patient_label)),
        "",
        "Primary question:",
        "  Can one planar emission feature predict the effective sensitivity needed",
        "  to recover the Q/SPECT reconstructed-volume activity?",
        "",
        "Methodological safeguards:",
        "  - no CT image or voxel value is used; Q/SPECT is the quantitative development target",
        "  - TEW is performed separately for AP and PA before their geometric mean",
        "  - one feature at a time with leave-one-timepoint-out prediction",
        "  - final error is reported on recovered activity, not only sensitivity",
        "  - constant patient sensitivity and elapsed-time-only comparators",
        "  - absolute count rate is diagnostic only and cannot win the main selection",
        "  - nested feature selection: feature selected using training days only",
        "  - crop-shift, timing-reference and first-order planar Poisson robustness audits",
        "",
        "Important limitation:",
        f"  {LOCKED_FEATURE} was discovered on this patient. Its result is hypothesis-",
        "  generating and must be tested unchanged on patient 2.",
        "",
        "TEW target — pre-specified univariate results:",
    ]
    for name, result in eligible_ranked:
        lines.append(
            f"  {name}: activity MARE={result['activity_mare_percent']:.2f}%, "
            f"bias={result['activity_bias_percent']:+.2f}%, max={result['maximum_activity_error_percent']:.2f}%, "
            f"extrapolated folds={result['extrapolated_folds']}/{len(target)}"
        )
    lines.extend(
        [
            "",
            "Comparators:",
            f"  constant sensitivity: activity MARE={constant_mare:.2f}%",
            f"  elapsed time only: activity MARE={time_only['activity_mare_percent']:.2f}%",
            f"  absolute TEW count rate (diagnostic): activity MARE="
            f"{analysis['results']['tew']['log TEW count rate']['activity_mare_percent']:.2f}%",
        ]
    )
    if nested is not None:
        selection_counts = {
            name: nested["selected_features"].count(name) for name in ELIGIBLE_FEATURES
        }
        lines.extend(
            [
                f"  nested feature selection: activity MARE={nested['activity_mare_percent']:.2f}%",
                "  selected features by held-out fold: "
                + ", ".join(f"{name}={count}" for name, count in selection_counts.items() if count),
            ]
        )
    lines.extend(
        [
            "",
            "Timing robustness:",
            "  Main result uses the PT DICOM time declared with DecayCorrection=START.",
            "  Alternative audit treats the first raw TOMO bed as the WB reference.",
            f"  Maximum target change={np.max(np.abs(analysis['alternative_target_shift_percent'])):.3f}%.",
            f"  Locked-feature activity MARE under alternative timing="
            f"{analysis['timing_locked_result']['activity_mare_percent']:.2f}%.",
            "",
            "Crop-shift robustness for locked feature:",
        ]
    )
    for row in analysis["crop_robustness"]:
        lines.append(
            f"  shift {row['crop_offset_pixels']:+d} px ({row['crop_offset_cm']:+.2f} cm): "
            f"activity MARE={row['activity_mare_percent']:.2f}%"
        )
    lines.extend(
        [
            "",
            "Planar Poisson uncertainty only:",
            f"  range={np.min(analysis['poisson_relative_percent']):.3f}% to "
            f"{np.max(analysis['poisson_relative_percent']):.3f}%.",
            "  This excludes Q/SPECT, calibration, scatter-model, registration and other",
            "  systematic uncertainties; it is not the total measurement uncertainty.",
            "",
            "Decision rule for patient 2:",
            f"  Freeze the processing and the locked feature ({LOCKED_FEATURE}).",
            "  Do not reselect the best feature on patient 2 before reporting its error.",
        ]
    )
    (output_dir / "v2_univariate_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(analysis: Dict[str, Any], output_dir: Path) -> None:
    methods = ["constant sensitivity", "elapsed time", *ELIGIBLE_FEATURES]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), layout="constrained")
    for axis, target_name, title in zip(
        axes,
        ("raw", "tew"),
        ("Sans correction de diffusé", "Avec correction TEW"),
    ):
        first = analysis["results"][target_name][LOCKED_FEATURE]
        values = [
            first["constant_activity_mare_percent"],
            analysis["results"][target_name]["elapsed time"]["activity_mare_percent"],
            *[
                analysis["results"][target_name][name]["activity_mare_percent"]
                for name in ELIGIBLE_FEATURES
            ],
        ]
        colors = ["#7f7f7f", "#9467bd"] + [
            "#2ca02c" if name == LOCKED_FEATURE else "#1f77b4" for name in ELIGIBLE_FEATURES
        ]
        axis.bar(np.arange(len(methods)), values, color=colors, alpha=0.88)
        axis.set_xticks(np.arange(len(methods)), methods, rotation=18, ha="right")
        axis.set_ylabel("Erreur relative moyenne sur l’activité (%)")
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Sensibilité effective — comparateurs et modèles univariés LOO")
    fig.savefig(output_dir / "v2_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    days = analysis["days"]
    target = analysis["targets"]["tew"]
    rate = analysis["rates"]["tew"]
    reference_activity = rate / target
    locked = analysis["results"]["tew"][LOCKED_FEATURE]
    time_only = analysis["results"]["tew"]["elapsed time"]
    nested = analysis["nested"]
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    ax.plot(days, reference_activity, "ko-", linewidth=2.3, label="Activité Q/SPECT de référence")
    ax.plot(days, rate / locked["constant_predictions"], "o--", label="Sensibilité constante")
    ax.plot(days, rate / time_only["predictions"], "s--", label="Temps seulement")
    ax.plot(days, rate / locked["predictions"], "D-", linewidth=2, label=f"Variable verrouillée: {LOCKED_FEATURE}")
    if nested is not None:
        ax.plot(days, rate / nested["predictions"], "^:", linewidth=2, label="Sélection univariée imbriquée")
    ax.set_xlabel("Temps après la première acquisition (jours)")
    ax.set_ylabel("Activité reconstruite (MBq)")
    ax.set_title("Activité planaire prédite — validation leave-one-timepoint-out")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "v2_tew_activity_predictions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    crop = analysis["crop_robustness"]
    axes[0].plot(
        [row["crop_offset_cm"] for row in crop],
        [row["activity_mare_percent"] for row in crop],
        "o-",
        linewidth=2,
    )
    axes[0].axvline(0.0, color="0.5", linestyle="--")
    axes[0].set_xlabel("Translation appliquée au crop (cm)")
    axes[0].set_ylabel("MARE sur l’activité (%)")
    axes[0].set_title("Robustesse à une erreur de position du crop")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[1].plot(days, analysis["poisson_relative_percent"], "o-", linewidth=2, color="#d62728")
    axes[1].set_xlabel("Temps après la première acquisition (jours)")
    axes[1].set_ylabel("Incertitude relative approximative (%)")
    axes[1].set_title("Composante de Poisson planaire après TEW et GM")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "v2_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    planar_dir: Path = planar_processing.default_planar_study_dir(),
    qspect_dir: Path = qspect_processing.default_qspect_dir(),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    patient_label: str = "Patient 1",
) -> Dict[str, Any]:
    analysis = build_analysis(planar_dir, qspect_dir)
    write_outputs(analysis, Path(output_dir), patient_label)
    plot_outputs(analysis, Path(output_dir))
    locked = analysis["results"]["tew"][LOCKED_FEATURE]
    time_only = analysis["results"]["tew"]["elapsed time"]
    print(
        f"Locked feature activity MARE={locked['activity_mare_percent']:.2f}% | "
        f"constant={locked['constant_activity_mare_percent']:.2f}% | "
        f"time only={time_only['activity_mare_percent']:.2f}%"
    )
    if analysis["nested"] is not None:
        print(
            "Nested feature-selection activity MARE="
            f"{analysis['nested']['activity_mare_percent']:.2f}%"
        )
    print(f"Saved outputs to {Path(output_dir)}")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planar-dir", type=Path, default=planar_processing.default_planar_study_dir())
    parser.add_argument("--qspect-dir", type=Path, default=qspect_processing.default_qspect_dir())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patient-label", default="Patient 1")
    args = parser.parse_args()
    run_analysis(args.planar_dir, args.qspect_dir, args.output_dir, args.patient_label)


if __name__ == "__main__":
    main()
