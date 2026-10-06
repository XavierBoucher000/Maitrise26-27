"""Planar sensitivity transfer using urine-derived body activity as target.

This analysis deliberately precedes the Q/SPECT-target pathway.  It trains on
the first three P11 May planar acquisitions and tests, without refitting, on
the three P8 June acquisitions.  At every planar acquisition time, the target
is the decay-consistent body activity inferred from injected activity and
measured urine/diaper excretion.

No CT, ACCT, Q/SPECT activity or Q/SPECT-guided crop is loaded.  All image
measurements use the complete planar image.  M0 is the primary simple test;
the spectral models reused from v3 are exploratory.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction as ac
import figure_layout
import patient_sensitivity_multicycle_v3 as v3
import planar_processing
import planar_total_counts_retention as retention


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = figure_layout.shared_dir(__file__)
DEFAULT_TIMEPOINT_COUNT = 3


# Primary analysis requested by the supervisor: uncorrected whole-image counts.
# ``GM_PIXEL`` is deliberately named because it is the sum of a pixelwise GM
# image, not the geometric mean of the already-integrated AP and PA totals.
SIMPLE_PHOTO_AP_PA = "photopeak_ap_pa"
SIMPLE_PHOTO_GM_PIXEL = "photopeak_gm_pixel"
SIMPLE_GENERAL_AP_PA = "general_scatter_ap_pa"
SIMPLE_GENERAL_GM_PIXEL = "general_scatter_gm_pixel"
SIMPLE_ALL_AP_PA = "all_four_windows_ap_pa"
SIMPLE_ALL_GM_PIXEL = "all_four_windows_gm_pixel"
SIMPLE_METHOD_ORDER = (
    SIMPLE_PHOTO_AP_PA,
    SIMPLE_PHOTO_GM_PIXEL,
    SIMPLE_GENERAL_AP_PA,
    SIMPLE_GENERAL_GM_PIXEL,
    SIMPLE_ALL_AP_PA,
    SIMPLE_ALL_GM_PIXEL,
)
SIMPLE_METHOD_LABELS = {
    SIMPLE_PHOTO_AP_PA: "Photopeak — AP+PA",
    SIMPLE_PHOTO_GM_PIXEL: "Photopeak — GM pixel",
    SIMPLE_GENERAL_AP_PA: "General scatter — AP+PA",
    SIMPLE_GENERAL_GM_PIXEL: "General scatter — GM pixel",
    SIMPLE_ALL_AP_PA: "4 fenêtres — AP+PA",
    SIMPLE_ALL_GM_PIXEL: "4 fenêtres — GM pixel",
}
SIMPLE_METHOD_COLORS = {
    SIMPLE_PHOTO_AP_PA: "#1F77B4",
    SIMPLE_PHOTO_GM_PIXEL: "#6BAED6",
    SIMPLE_GENERAL_AP_PA: "#D62728",
    SIMPLE_GENERAL_GM_PIXEL: "#FF9896",
    SIMPLE_ALL_AP_PA: "#2CA02C",
    SIMPLE_ALL_GM_PIXEL: "#98DF8A",
}
SIMPLE_METHOD_MARKERS = {
    SIMPLE_PHOTO_AP_PA: "o",
    SIMPLE_PHOTO_GM_PIXEL: "D",
    SIMPLE_GENERAL_AP_PA: "s",
    SIMPLE_GENERAL_GM_PIXEL: "P",
    SIMPLE_ALL_AP_PA: "^",
    SIMPLE_ALL_GM_PIXEL: "v",
}


@dataclass(frozen=True)
class UrineDatasetSpec:
    key: str
    patient_id: str
    cycle_id: str
    label: str
    planar_dir: Path
    urine_csv: Path


TRAINING_DATASET = UrineDatasetSpec(
    key="p11_may_urine",
    patient_id="p11",
    cycle_id="2026-05__Studies",
    label="P11 — mai 2026",
    planar_dir=PROJECT_DIR / "Data/p11/2026-05__Studies_WBP",
    urine_csv=PROJECT_DIR / "Data/p11/P-011csv.csv",
)
TEST_DATASET = UrineDatasetSpec(
    key="p8_june_urine",
    patient_id="p8",
    cycle_id="2026-06__Studies",
    label="P8 — juin 2026",
    planar_dir=PROJECT_DIR / "Data/p8/2026-06__Studies_WBP",
    urine_csv=PROJECT_DIR / "Data/p8/P-008.csv",
)


# Only full-image modes are retained.  Crop modes from v3 depend on Q/SPECT
# profile matching and would contradict the purpose of this analysis.
MODE_ORDER = (
    v3.MODE_TEW_FULL_GM,
    v3.MODE_RAW_FULL_GM,
    v3.MODE_TEW_FULL_AP_PA,
    v3.MODE_RAW_FULL_AP_PA,
    v3.MODE_ALL_FOUR_FULL_AP_PA,
)
MODE_SHORT_LABELS = {
    v3.MODE_TEW_FULL_GM: "TEW\ncomplet GM",
    v3.MODE_RAW_FULL_GM: "Photopeak brut\ncomplet GM",
    v3.MODE_TEW_FULL_AP_PA: "TEW\ncomplet AP+PA",
    v3.MODE_RAW_FULL_AP_PA: "Photopeak brut\ncomplet AP+PA",
    v3.MODE_ALL_FOUR_FULL_AP_PA: "4 fenêtres\ncomplet AP+PA",
}
COMMON_PLOT_MODELS = (
    v3.MODEL_CONSTANT,
    v3.MODEL_LOCKED,
    v3.MODEL_ADJACENT,
    v3.MODEL_PHOTO_ALL,
    v3.MODEL_LOWER,
    v3.MODEL_UPPER,
)


def _mode_model_names(mode: str) -> tuple[str, ...]:
    """Return v3 candidates allowed by one urine-target full-image mode."""
    if mode not in MODE_ORDER:
        raise ValueError(f"Unsupported urine-target mode: {mode}")
    if mode in (v3.MODE_TEW_FULL_GM, v3.MODE_TEW_FULL_AP_PA):
        return v3.MODEL_ORDER
    return tuple(
        model for model in v3.MODEL_ORDER if model != v3.MODEL_TEW_FRACTION
    )


def _mode_rate_and_geometry(
    matched: Mapping[str, Any], mode: str
) -> tuple[float, str]:
    if mode == v3.MODE_TEW_FULL_GM:
        return (
            float(matched["tew_rates"][v3.GEOMETRY_FULL_GM]),
            v3.GEOMETRY_FULL_GM,
        )
    if mode == v3.MODE_RAW_FULL_GM:
        return (
            float(matched["raw_rates"][v3.GEOMETRY_FULL_GM]),
            v3.GEOMETRY_FULL_GM,
        )
    if mode == v3.MODE_TEW_FULL_AP_PA:
        return (
            float(matched["tew_rates"][v3.GEOMETRY_FULL_AP_PA]),
            v3.GEOMETRY_FULL_AP_PA,
        )
    if mode == v3.MODE_RAW_FULL_AP_PA:
        return (
            float(matched["raw_rates"][v3.GEOMETRY_FULL_AP_PA]),
            v3.GEOMETRY_FULL_AP_PA,
        )
    if mode == v3.MODE_ALL_FOUR_FULL_AP_PA:
        return float(matched["all_four_full_ap_pa_rate"]), v3.GEOMETRY_FULL_AP_PA
    raise ValueError(f"Unsupported urine-target mode: {mode}")


def urine_target_scenarios(
    evaluation_times_h: np.ndarray,
    urine_data: Mapping[str, Any],
    urine_balance: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Bracket the timing ambiguity of interval urine collections.

    ``completed_only`` is the primary target used throughout the analysis: an
    interval contributes only after its recorded collection time. ``linear``
    distributes each interval's injection-equivalent activity uniformly
    between two collection times. ``current_interval_complete`` treats the
    current interval as fully excreted and is an upper-removal bound.
    """
    times = np.asarray(evaluation_times_h, dtype=np.float64)
    collection_times = np.asarray(urine_balance["times_h"], dtype=np.float64)
    intervals = np.asarray(
        urine_balance["interval_injection_equivalent_mbq"], dtype=np.float64
    )
    cumulative = np.cumsum(intervals)
    decay = sang_decay_factor(times, float(urine_data["half_life_h"]))
    injected = float(urine_data["injected_activity_mbq"])

    completed_removed = np.asarray(
        [float(intervals[collection_times <= time].sum()) for time in times]
    )
    interpolation_times = np.concatenate(([0.0], collection_times))
    interpolation_values = np.concatenate(([0.0], cumulative))
    linear_removed = np.interp(
        times,
        interpolation_times,
        interpolation_values,
        left=0.0,
        right=float(cumulative[-1]),
    )
    current_interval_removed = np.empty_like(times)
    for index, time in enumerate(times):
        next_index = int(np.searchsorted(collection_times, time, side="left"))
        next_index = min(next_index, cumulative.size - 1)
        current_interval_removed[index] = cumulative[next_index]

    def body_activity(removed_equivalent: np.ndarray) -> np.ndarray:
        return (injected - removed_equivalent) * decay

    return {
        "physical_available_activity_mbq": injected * decay,
        "completed_only_activity_mbq": body_activity(completed_removed),
        "linear_interval_activity_mbq": body_activity(linear_removed),
        "current_interval_complete_activity_mbq": body_activity(
            current_interval_removed
        ),
        "completed_removed_injection_equivalent_mbq": completed_removed,
        "linear_removed_injection_equivalent_mbq": linear_removed,
        "current_interval_removed_injection_equivalent_mbq": (
            current_interval_removed
        ),
    }


def sang_decay_factor(times_h: np.ndarray, half_life_h: float) -> np.ndarray:
    """Local wrapper kept explicit for timing-scenario unit tests."""
    return np.exp(-math.log(2.0) * np.asarray(times_h) / half_life_h)


def build_urine_dataset(
    spec: UrineDatasetSpec,
    timepoint_count: int = DEFAULT_TIMEPOINT_COUNT,
) -> Dict[str, Any]:
    """Build full-image planar observations with a urine-derived target."""
    if timepoint_count < 2:
        raise ValueError("At least two urine-referenced timepoints are required")
    result = retention.analyze_total_counts_retention(
        planar_dir=spec.planar_dir,
        urine_csv=spec.urine_csv,
        comparison_timepoints=timepoint_count,
    )
    rows = list(result["rows"][:timepoint_count])
    scans = ac.sorted_planar_scans(spec.planar_dir)[:timepoint_count]
    if len(rows) != timepoint_count or len(scans) != timepoint_count:
        raise ValueError(
            f"{spec.label} requires {timepoint_count} planar/urine observations"
        )

    matched_rows: list[Dict[str, Any]] = []
    records: list[Dict[str, Any]] = []
    simple_rates = {name: [] for name in SIMPLE_METHOD_ORDER}
    for row, scan in zip(rows, scans):
        if str(row["scan_name"]) != str(scan["scan_name"]):
            raise ValueError(
                f"Planar ordering mismatch for {spec.label}: "
                f"{row['scan_name']} versus {scan['scan_name']}"
            )
        height = int(np.asarray(scan["images"][0]["image"]).shape[0])
        matched = v3.mode_matched_measurements(scan, (0, height))
        matched_rows.append(matched)
        duration_s = float(matched["duration_seconds"])
        counts_ap_pa = retention.total_ap_pa_counts_by_window(scan)
        counts_gm_pixel = retention.total_gm_counts_by_window(scan)
        simple_rates[SIMPLE_PHOTO_AP_PA].append(
            float(counts_ap_pa[retention.PHOTOPEAK]) / duration_s
        )
        simple_rates[SIMPLE_PHOTO_GM_PIXEL].append(
            float(counts_gm_pixel[retention.PHOTOPEAK]) / duration_s
        )
        simple_rates[SIMPLE_GENERAL_AP_PA].append(
            float(counts_ap_pa[retention.GENERAL_SCATTER]) / duration_s
        )
        simple_rates[SIMPLE_GENERAL_GM_PIXEL].append(
            float(counts_gm_pixel[retention.GENERAL_SCATTER]) / duration_s
        )
        simple_rates[SIMPLE_ALL_AP_PA].append(
            float(
                sum(
                    counts_ap_pa[name] for name in retention.EXPECTED_WINDOWS
                )
            )
            / duration_s
        )
        simple_rates[SIMPLE_ALL_GM_PIXEL].append(
            float(
                sum(
                    counts_gm_pixel[name] for name in retention.EXPECTED_WINDOWS
                )
            )
            / duration_s
        )
        records.append(
            {
                "dataset_key": spec.key,
                "patient_id": spec.patient_id,
                "cycle_id": spec.cycle_id,
                "dataset_label": spec.label,
                "timepoint": str(row["timepoint"]),
                "planar_datetime": row["acquisition_datetime"],
                "hours_after_injection": float(row["hours_after_injection"]),
                "days_after_injection": float(row["days_after_injection"]),
                "injection_datetime": result["urine_data"]["injection_datetime"],
                "injected_activity_mbq": float(
                    result["urine_data"]["injected_activity_mbq"]
                ),
                "physical_half_life_h": float(
                    result["urine_data"]["half_life_h"]
                ),
                "physical_available_activity_mbq": float(
                    row["physical_available_activity_mbq"]
                ),
                "cumulative_excreted_activity_mbq": float(
                    row["measured_cumulative_excreted_activity_mbq"]
                ),
                "urine_remaining_activity_mbq": float(
                    row["urine_derived_remaining_activity_mbq"]
                ),
            }
        )

    reference_activity = np.asarray(
        [record["urine_remaining_activity_mbq"] for record in records],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(reference_activity)) or np.any(reference_activity <= 0):
        raise ValueError(f"Invalid urine-derived activity for {spec.label}")

    timing_scenarios = urine_target_scenarios(
        np.asarray(
            [record["hours_after_injection"] for record in records], dtype=float
        ),
        result["urine_data"],
        result["urine_balance"],
    )
    if not np.allclose(
        reference_activity,
        timing_scenarios["completed_only_activity_mbq"],
    ):
        raise ValueError("Completed-collection urine target is inconsistent")
    for index, record in enumerate(records):
        for name, values in timing_scenarios.items():
            record[name] = float(values[index])

    mode_observations: Dict[str, Dict[str, Any]] = {}
    for mode in MODE_ORDER:
        rates: list[float] = []
        feature_rows: list[Mapping[str, float]] = []
        for matched in matched_rows:
            rate, geometry = _mode_rate_and_geometry(matched, mode)
            rates.append(rate)
            feature_rows.append(matched["features_by_geometry"][geometry])
        rate_array = np.asarray(rates, dtype=np.float64)
        mode_observations[mode] = {
            "records": records,
            "features": {
                name: np.asarray([item[name] for item in feature_rows], dtype=float)
                for name in v3.FEATURE_ORDER
            },
            "measurement_rate": rate_array,
            "reference_activity": reference_activity,
            "target_sensitivity": rate_array / reference_activity,
        }

    return {
        "spec": spec,
        "records": records,
        "reference_activity": reference_activity,
        "mode_observations": mode_observations,
        "simple_rates": {
            name: np.asarray(values, dtype=np.float64)
            for name, values in simple_rates.items()
        },
        "urine_data": result["urine_data"],
        "urine_balance": result["urine_balance"],
        "timing_scenarios": timing_scenarios,
    }


def _activity_metrics(reference: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    errors = 100.0 * (predicted / reference - 1.0)
    return {
        "activity_bias_percent": float(np.mean(errors)),
        "activity_mare_percent": float(np.mean(np.abs(errors))),
        "activity_rmse_percent": float(np.sqrt(np.mean(errors**2))),
        "maximum_activity_error_percent": float(np.max(np.abs(errors))),
    }


def _extrapolation_flags(
    fit: Mapping[str, Any], test: Mapping[str, Any]
) -> np.ndarray:
    count = int(np.asarray(test["reference_activity"]).size)
    flags = np.zeros(count, dtype=bool)
    for name, bounds in fit.get("feature_ranges", {}).items():
        values = np.asarray(test["features"][name], dtype=float)
        flags |= (values < float(bounds[0])) | (values > float(bounds[1]))
    return flags


def fit_and_test_mode(
    mode: str,
    training: Mapping[str, Any],
    test: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fit on P11 urine target and evaluate unchanged on P8 urine target."""
    metrics_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    fits: Dict[str, Dict[str, Any]] = {}
    for model in _mode_model_names(mode):
        fit = v3.fit_model(model, training)
        fits[model] = fit
        metric_base = {
            "rate_mode": mode,
            "rate_mode_label": v3.RATE_MODES[mode].label,
            "model": model,
            "model_label": v3.MODEL_LABELS[model],
            "training_dataset": TRAINING_DATASET.key,
            "test_dataset": TEST_DATASET.key,
            "n_train": len(training["records"]),
            "n_test": len(test["records"]),
            "fit_status": fit["status"],
            "fit_reason": fit["reason"],
            "fit_rank": int(fit["rank"]),
            "fit_condition_number": float(fit["condition_number"]),
        }
        if fit["status"] != "fitted":
            metrics_rows.append(
                {
                    **metric_base,
                    "activity_bias_percent": math.nan,
                    "activity_mare_percent": math.nan,
                    "activity_rmse_percent": math.nan,
                    "maximum_activity_error_percent": math.nan,
                    "extrapolated_predictions": 0,
                }
            )
            continue

        predicted_sensitivity = v3.predict_model(fit, test)
        rate = np.asarray(test["measurement_rate"], dtype=float)
        reference = np.asarray(test["reference_activity"], dtype=float)
        predicted_activity = rate / predicted_sensitivity
        flags = _extrapolation_flags(fit, test)
        metrics_rows.append(
            {
                **metric_base,
                **_activity_metrics(reference, predicted_activity),
                "extrapolated_predictions": int(np.count_nonzero(flags)),
            }
        )

        coefficients = np.asarray(fit["coefficients"], dtype=float)
        for index, record in enumerate(test["records"]):
            row = {
                "rate_mode": mode,
                "rate_mode_label": v3.RATE_MODES[mode].label,
                "model": model,
                "model_label": v3.MODEL_LABELS[model],
                **record,
                "measurement_rate_cps": float(rate[index]),
                "urine_reference_activity_mbq": float(reference[index]),
                "observed_sensitivity_cps_per_mbq": float(
                    test["target_sensitivity"][index]
                ),
                "predicted_sensitivity_cps_per_mbq": float(
                    predicted_sensitivity[index]
                ),
                "predicted_activity_mbq": float(predicted_activity[index]),
                "predicted_over_urine_reference": float(
                    predicted_activity[index] / reference[index]
                ),
                "activity_error_percent": float(
                    100.0 * (predicted_activity[index] / reference[index] - 1.0)
                ),
                "feature_extrapolation": bool(flags[index]),
                "intercept_or_constant": float(coefficients[0]),
            }
            for coefficient_index, feature_name in enumerate(
                fit["feature_names"], start=1
            ):
                row[f"coefficient_{feature_name}"] = float(
                    coefficients[coefficient_index]
                )
            for feature_name in v3.FEATURE_ORDER:
                row[feature_name] = float(test["features"][feature_name][index])
            prediction_rows.append(row)
    return {"fits": fits, "metrics": metrics_rows, "predictions": prediction_rows}


def simple_zero_origin_transfer(
    training_dataset: Mapping[str, Any],
    test_dataset: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Fit R=kA on P11 and apply k unchanged to P8 for six raw modes."""
    output: Dict[str, Dict[str, Any]] = {}
    train_activity = np.asarray(training_dataset["reference_activity"], dtype=float)
    test_activity = np.asarray(test_dataset["reference_activity"], dtype=float)
    for name in SIMPLE_METHOD_ORDER:
        train_rate = np.asarray(training_dataset["simple_rates"][name], dtype=float)
        test_rate = np.asarray(test_dataset["simple_rates"][name], dtype=float)
        denominator = float(np.dot(train_activity, train_activity))
        if denominator <= 0.0:
            raise ValueError("Cannot fit a zero-origin sensitivity")
        sensitivity = float(np.dot(train_activity, train_rate) / denominator)
        prediction = test_rate / sensitivity
        output[name] = {
            "sensitivity_cps_per_mbq": sensitivity,
            "training_rate_cps": train_rate,
            "test_rate_cps": test_rate,
            "predicted_test_activity_mbq": prediction,
            **_activity_metrics(test_activity, prediction),
        }
    return output


def j0_calibration_results(
    dataset: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Calibrate one sensitivity at J0 and test it on later same-patient points."""
    activity = np.asarray(dataset["reference_activity"], dtype=np.float64)
    if activity.size < 2:
        raise ValueError("J0 calibration requires at least one follow-up")
    output: Dict[str, Dict[str, Any]] = {}
    for method in SIMPLE_METHOD_ORDER:
        rate = np.asarray(dataset["simple_rates"][method], dtype=np.float64)
        sensitivity_j0 = float(rate[0] / activity[0])
        prediction = rate / sensitivity_j0
        output[method] = {
            "sensitivity_j0_cps_per_mbq": sensitivity_j0,
            "rate_cps": rate,
            "reference_activity_mbq": activity,
            "predicted_activity_mbq": prediction,
            "all_timepoints_metrics": _activity_metrics(activity, prediction),
            "followup_metrics": _activity_metrics(activity[1:], prediction[1:]),
        }
    return output


def j0_calibration_rows(
    dataset: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Flatten J0 predictions and follow-up summaries for CSV output."""
    prediction_rows: list[Dict[str, Any]] = []
    summary_rows: list[Dict[str, Any]] = []
    records = list(dataset["records"])
    for method in SIMPLE_METHOD_ORDER:
        result = results[method]
        reference = np.asarray(result["reference_activity_mbq"], dtype=float)
        prediction = np.asarray(result["predicted_activity_mbq"], dtype=float)
        rate = np.asarray(result["rate_cps"], dtype=float)
        for index, record in enumerate(records):
            prediction_rows.append(
                {
                    "dataset_key": dataset["spec"].key,
                    "patient_id": dataset["spec"].patient_id,
                    "cycle_id": dataset["spec"].cycle_id,
                    "method": method,
                    "method_label": SIMPLE_METHOD_LABELS[method],
                    "timepoint": record["timepoint"],
                    "hours_after_injection": record["hours_after_injection"],
                    "used_for_calibration": index == 0,
                    "measurement_rate_cps": float(rate[index]),
                    "urine_reference_activity_mbq": float(reference[index]),
                    "predicted_activity_mbq": float(prediction[index]),
                    "predicted_over_reference": float(
                        prediction[index] / reference[index]
                    ),
                    "activity_error_percent": float(
                        100.0 * (prediction[index] / reference[index] - 1.0)
                    ),
                }
            )
        metrics = result["followup_metrics"]
        summary_rows.append(
            {
                "dataset_key": dataset["spec"].key,
                "patient_id": dataset["spec"].patient_id,
                "cycle_id": dataset["spec"].cycle_id,
                "method": method,
                "method_label": SIMPLE_METHOD_LABELS[method],
                "sensitivity_j0_cps_per_mbq": result[
                    "sensitivity_j0_cps_per_mbq"
                ],
                "followup_n": int(reference.size - 1),
                "followup_bias_percent": metrics["activity_bias_percent"],
                "followup_mare_percent": metrics["activity_mare_percent"],
                "followup_rmse_percent": metrics["activity_rmse_percent"],
                "followup_max_error_percent": metrics[
                    "maximum_activity_error_percent"
                ],
            }
        )
    return prediction_rows, summary_rows


def sensitivity_stability_rows(
    training_dataset: Mapping[str, Any],
    test_dataset: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Summarize within-patient urine-target sensitivity for every mode."""
    output: list[Dict[str, Any]] = []
    for dataset, role in (
        (training_dataset, "training"),
        (test_dataset, "test"),
    ):
        for mode in MODE_ORDER:
            values = np.asarray(
                dataset["mode_observations"][mode]["target_sensitivity"],
                dtype=float,
            )
            mean = float(np.mean(values))
            sample_sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            output.append(
                {
                    "dataset_role": role,
                    "dataset_key": dataset["spec"].key,
                    "patient_id": dataset["spec"].patient_id,
                    "cycle_id": dataset["spec"].cycle_id,
                    "rate_mode": mode,
                    "rate_mode_label": v3.RATE_MODES[mode].label,
                    "n": int(values.size),
                    "mean_sensitivity_cps_per_mbq": mean,
                    "sample_sd_cps_per_mbq": sample_sd,
                    "cv_percent": 100.0 * sample_sd / mean,
                    "last_over_first": float(values[-1] / values[0]),
                    "first_to_last_change_percent": float(
                        100.0 * (values[-1] / values[0] - 1.0)
                    ),
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        value.isoformat(sep=" ")
                        if hasattr(value, "isoformat")
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return path


def plot_simple_transfer(
    training_dataset: Mapping[str, Any],
    test_dataset: Mapping[str, Any],
    simple_results: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> Path:
    """Show P11 zero-origin lines and held-out P8 for six raw modes."""
    train_activity = np.asarray(training_dataset["reference_activity"], dtype=float)
    test_activity = np.asarray(test_dataset["reference_activity"], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), layout="constrained")
    for axis, name in zip(axes.flat, SIMPLE_METHOD_ORDER):
        result = simple_results[name]
        train_rate = np.asarray(result["training_rate_cps"], dtype=float)
        test_rate = np.asarray(result["test_rate_cps"], dtype=float)
        axis.scatter(
            train_activity / 1000.0,
            train_rate / 1000.0,
            s=72,
            color="#1F77B4",
            edgecolor="white",
            label="P11 entraînement",
            zorder=3,
        )
        axis.scatter(
            test_activity / 1000.0,
            test_rate / 1000.0,
            s=74,
            marker="D",
            color="#E67E22",
            edgecolor="white",
            label="P8 test verrouillé",
            zorder=3,
        )
        maximum_activity = 1.06 * max(
            float(np.max(train_activity)), float(np.max(test_activity))
        )
        line_activity = np.linspace(0.0, maximum_activity, 100)
        axis.plot(
            line_activity / 1000.0,
            result["sensitivity_cps_per_mbq"] * line_activity / 1000.0,
            color="#1F77B4",
            linewidth=2.0,
            label="Droite ajustée sur P11",
        )
        for index, (x_value, y_value) in enumerate(
            zip(test_activity / 1000.0, test_rate / 1000.0)
        ):
            axis.annotate(
                f"J{index}",
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.5,
            )
        axis.set_title(SIMPLE_METHOD_LABELS[name])
        axis.set_xlabel("Activité corporelle selon urine (GBq)")
        axis.set_ylabel("Taux planaire complet (kcps)")
        axis.grid(True, linestyle="--", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.04,
            0.95,
            (
                f"S(P11) = {result['sensitivity_cps_per_mbq']:.3f} cps/MBq\n"
                f"MARE P8 = {result['activity_mare_percent']:.1f} %"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
        )
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=3,
        frameon=False,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_simple_prediction_parity(
    test_dataset: Mapping[str, Any],
    simple_results: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> Path:
    """Plot held-out P8 predictions directly against the urine reference."""
    reference = np.asarray(test_dataset["reference_activity"], dtype=float)
    fig, axis = plt.subplots(figsize=(7.8, 6.6), layout="constrained")
    maximum_gbq = float(np.max(reference)) * 1.08 / 1000.0
    axis.plot(
        [0.0, maximum_gbq],
        [0.0, maximum_gbq],
        color="black",
        linewidth=1.5,
    )
    axis.fill_between(
        [0.0, maximum_gbq],
        [0.0, 0.9 * maximum_gbq],
        [0.0, 1.1 * maximum_gbq],
        color="#D9EAD3",
        alpha=0.65,
        label="±10 %",
    )
    for method in SIMPLE_METHOD_ORDER:
        predicted = np.asarray(
            simple_results[method]["predicted_test_activity_mbq"], dtype=float
        )
        axis.scatter(
            reference / 1000.0,
            predicted / 1000.0,
            s=62,
            marker=SIMPLE_METHOD_MARKERS[method],
            color=SIMPLE_METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.7,
            label=(
                f"{SIMPLE_METHOD_LABELS[method]} "
                f"({simple_results[method]['activity_mare_percent']:.1f} %)"
            ),
            zorder=3,
        )
    axis.set_xlim(0.0, maximum_gbq)
    axis.set_ylim(0.0, maximum_gbq)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Activité P8 selon le bilan urinaire (GBq)")
    axis.set_ylabel("Activité P8 prédite avec la calibration P11 (GBq)")
    axis.set_title("Transfert interpatient verrouillé P11 → P8")
    axis.grid(True, linestyle="--", alpha=0.24)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_simple_relative_error(
    test_dataset: Mapping[str, Any],
    simple_results: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> Path:
    """Plot held-out P8 percentage error for each raw-count definition."""
    records = list(test_dataset["records"])
    reference = np.asarray(test_dataset["reference_activity"], dtype=float)
    x = np.arange(reference.size)
    fig, axis = plt.subplots(figsize=(9.2, 5.8), layout="constrained")
    axis.axhspan(-10.0, 10.0, color="#D9EAD3", alpha=0.7, label="±10 %")
    axis.axhspan(-20.0, -10.0, color="#FFF2CC", alpha=0.55)
    axis.axhspan(10.0, 20.0, color="#FFF2CC", alpha=0.55, label="±20 %")
    axis.axhline(0.0, color="black", linewidth=1.4)
    for method in SIMPLE_METHOD_ORDER:
        predicted = np.asarray(
            simple_results[method]["predicted_test_activity_mbq"], dtype=float
        )
        error = 100.0 * (predicted / reference - 1.0)
        axis.plot(
            x,
            error,
            color=SIMPLE_METHOD_COLORS[method],
            marker=SIMPLE_METHOD_MARKERS[method],
            linewidth=1.8,
            markersize=6.5,
            label=SIMPLE_METHOD_LABELS[method],
        )
    axis.set_xticks(x, [record["timepoint"] for record in records])
    axis.set_xlabel("Timepoint P8")
    axis.set_ylabel("Erreur relative de l’activité prédite (%)")
    axis.set_title("Erreur du transfert verrouillé P11 → P8")
    axis.grid(True, axis="y", linestyle="--", alpha=0.24)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_j0_calibration_predictions(
    datasets_and_results: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]
    ],
    output_path: Path,
) -> Path:
    """Show absolute within-patient J0 calibration and J1/J2 predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    for axis, (dataset, results) in zip(axes, datasets_and_results):
        records = list(dataset["records"])
        x = np.arange(len(records))
        reference = np.asarray(dataset["reference_activity"], dtype=float)
        axis.plot(
            x,
            reference / 1000.0,
            color="black",
            marker="*",
            linewidth=2.3,
            markersize=9,
            label="Activité selon urine",
            zorder=4,
        )
        for method in SIMPLE_METHOD_ORDER:
            predicted = np.asarray(
                results[method]["predicted_activity_mbq"], dtype=float
            )
            axis.plot(
                x,
                predicted / 1000.0,
                color=SIMPLE_METHOD_COLORS[method],
                marker=SIMPLE_METHOD_MARKERS[method],
                linewidth=1.8,
                markersize=6.5,
                label=SIMPLE_METHOD_LABELS[method],
            )
        axis.set_xticks(x, [record["timepoint"] for record in records])
        axis.set_xlabel("J0 calibre; J1 et J2 sont des tests")
        axis.set_ylabel("Activité corporelle (GBq)")
        axis.set_title(dataset["spec"].label)
        axis.grid(True, axis="y", linestyle="--", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        frameon=False,
        fontsize=9.0,
    )
    fig.suptitle(
        "Calibration intra-patient à J0, sans réajustement ensuite", y=0.985
    )
    fig.subplots_adjust(top=0.76, bottom=0.13, wspace=0.20)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_normalized_simple_sensitivity(
    datasets: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    """Compare temporal sensitivity after normalization to patient J0."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    for axis, dataset in zip(axes, datasets):
        activity = np.asarray(dataset["reference_activity"], dtype=float)
        records = list(dataset["records"])
        x = np.arange(activity.size)
        axis.axhspan(0.9, 1.1, color="#D9EAD3", alpha=0.7)
        axis.axhline(1.0, color="black", linewidth=1.4)
        for method in SIMPLE_METHOD_ORDER:
            rate = np.asarray(dataset["simple_rates"][method], dtype=float)
            sensitivity = rate / activity
            axis.plot(
                x,
                sensitivity / sensitivity[0],
                color=SIMPLE_METHOD_COLORS[method],
                marker=SIMPLE_METHOD_MARKERS[method],
                linewidth=1.8,
                markersize=6.5,
                label=SIMPLE_METHOD_LABELS[method],
            )
        axis.set_xticks(x, [record["timepoint"] for record in records])
        axis.set_xlabel("Timepoint")
        axis.set_ylabel("S_eff(t) / S_eff(J0)")
        axis.set_title(dataset["spec"].label)
        axis.grid(True, axis="y", linestyle="--", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
        fontsize=9.0,
    )
    fig.suptitle("Stabilité temporelle de la sensibilité effective", y=0.985)
    fig.subplots_adjust(top=0.76, bottom=0.13, wspace=0.20)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def timing_qc_rows(dataset: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Describe which collection interval surrounds every planar time."""
    collection_times = np.asarray(dataset["urine_balance"]["times_h"], dtype=float)
    output: list[Dict[str, Any]] = []
    for record in dataset["records"]:
        time_h = float(record["hours_after_injection"])
        before = collection_times[collection_times <= time_h]
        after = collection_times[collection_times > time_h]
        output.append(
            {
                "dataset_key": dataset["spec"].key,
                "patient_id": dataset["spec"].patient_id,
                "cycle_id": dataset["spec"].cycle_id,
                "timepoint": record["timepoint"],
                "planar_hours_after_injection": time_h,
                "previous_collection_h": float(before[-1]) if before.size else math.nan,
                "next_collection_h": float(after[0]) if after.size else math.nan,
                "minutes_to_next_collection": (
                    float((after[0] - time_h) * 60.0) if after.size else math.nan
                ),
                "completed_only_activity_mbq": record[
                    "completed_only_activity_mbq"
                ],
                "linear_interval_activity_mbq": record[
                    "linear_interval_activity_mbq"
                ],
                "current_interval_complete_activity_mbq": record[
                    "current_interval_complete_activity_mbq"
                ],
            }
        )
    return output


def plot_imaging_urine_timeline(
    datasets: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    """Show injection, planar acquisitions and urine-collection endpoints."""
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), layout="constrained")
    for axis, dataset in zip(axes, datasets):
        balance = dataset["urine_balance"]
        collection_times = np.asarray(balance["times_h"], dtype=float)
        cumulative_gbq = np.asarray(
            balance["cumulative_injection_equivalent_mbq"], dtype=float
        ) / 1000.0
        step_times = np.concatenate(([0.0], collection_times))
        step_values = np.concatenate(([0.0], cumulative_gbq))
        axis.step(
            step_times,
            step_values,
            where="post",
            color="#1F77B4",
            linewidth=2.2,
            label="Excrétion cumulative équivalente à t=0",
        )
        axis.scatter(
            collection_times,
            cumulative_gbq,
            color="#1F77B4",
            s=42,
            zorder=3,
            label="Fin de collecte",
        )
        maximum = max(float(np.max(cumulative_gbq)), 0.1)
        for record in dataset["records"]:
            time_h = float(record["hours_after_injection"])
            axis.axvline(time_h, color="#E67E22", linestyle="--", alpha=0.65)
            axis.text(
                time_h,
                0.04 * maximum,
                str(record["timepoint"]),
                rotation=90,
                ha="right",
                va="bottom",
                color="#A64B00",
                fontsize=9,
            )
        axis.scatter(
            [0.0], [0.0], marker="*", s=110, color="black", label="Injection"
        )
        axis.set_xlabel("Temps après injection (h)")
        axis.set_ylabel("Activité excrétée cumulée (GBq à t=0)")
        axis.set_title(dataset["spec"].label)
        axis.grid(True, linestyle="--", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Synchronisation des images et des collectes urinaires")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_urine_target_timing_scenarios(
    datasets: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    """Visualize the effect of interval timing assumptions on the target."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), layout="constrained")
    definitions = (
        (
            "completed_only_activity_mbq",
            "Collectes terminées seulement (cible principale)",
            "#111111",
            "o",
        ),
        (
            "linear_interval_activity_mbq",
            "Excrétion uniforme dans l’intervalle",
            "#1F77B4",
            "s",
        ),
        (
            "current_interval_complete_activity_mbq",
            "Intervalle courant entièrement excrété",
            "#D62728",
            "^",
        ),
    )
    for axis, dataset in zip(axes, datasets):
        records = list(dataset["records"])
        times = np.asarray(
            [record["hours_after_injection"] for record in records], dtype=float
        )
        for field, label, color, marker in definitions:
            values = np.asarray([record[field] for record in records], dtype=float)
            axis.plot(
                times,
                values / 1000.0,
                color=color,
                marker=marker,
                linewidth=2.0,
                markersize=6.5,
                label=label,
            )
        for record in records:
            axis.annotate(
                str(record["timepoint"]),
                (
                    float(record["hours_after_injection"]),
                    float(record["completed_only_activity_mbq"]) / 1000.0,
                ),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=9,
            )
        axis.set_xlabel("Temps après injection (h)")
        axis.set_ylabel("Cible d’activité corporelle (GBq)")
        axis.set_title(dataset["spec"].label)
        axis.grid(True, linestyle="--", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Sensibilité de la cible au moment attribué à l’excrétion")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_mode_model_mare(
    metrics_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    models = v3.MODEL_ORDER
    matrix = np.full((len(MODE_ORDER), len(models)), np.nan, dtype=float)
    for mode_index, mode in enumerate(MODE_ORDER):
        for model_index, model in enumerate(models):
            row = next(
                (
                    item
                    for item in metrics_rows
                    if item["rate_mode"] == mode and item["model"] == model
                ),
                None,
            )
            if row is not None and row["fit_status"] == "fitted":
                matrix[mode_index, model_index] = float(
                    row["activity_mare_percent"]
                )
    finite = matrix[np.isfinite(matrix)]
    vmax = max(10.0, float(np.percentile(finite, 90))) if finite.size else 10.0
    fig, axis = plt.subplots(figsize=(12.8, 5.8), layout="constrained")
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
    )
    axis.set_xticks(range(len(models)), [name.split("_")[0] for name in models])
    axis.set_yticks(
        range(len(MODE_ORDER)), [MODE_SHORT_LABELS[mode] for mode in MODE_ORDER]
    )
    axis.set_xlabel("Modèle de sensibilité")
    axis.set_ylabel("Mode planaire pleine image")
    axis.set_title(
        "Transfert verrouillé P11 → P8 — activité corporelle issue du bilan urinaire"
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                "N/E" if not np.isfinite(value) else f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=(
                    "white"
                    if np.isfinite(value) and min(value, vmax) > 0.58 * vmax
                    else "black"
                ),
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.88)
    colorbar.set_label("MARE de l’activité corporelle sur les 3 timepoints P8 (%)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_primary_predictions(
    prediction_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.0), layout="constrained")
    for axis, mode in zip(axes.flat, MODE_ORDER):
        selected = [
            row
            for row in prediction_rows
            if row["rate_mode"] == mode and row["model"] in COMMON_PLOT_MODELS
        ]
        labels = list(dict.fromkeys(str(row["timepoint"]) for row in selected))
        x = np.arange(len(labels), dtype=float)
        axis.axhspan(0.9, 1.1, color="#D9EAD3", alpha=0.65, zorder=0)
        axis.axhline(1.0, color="black", linewidth=1.4, zorder=1)
        for model in COMMON_PLOT_MODELS:
            rows = [row for row in selected if row["model"] == model]
            if not rows:
                continue
            by_label = {str(row["timepoint"]): row for row in rows}
            values = np.asarray(
                [by_label[label]["predicted_over_urine_reference"] for label in labels],
                dtype=float,
            )
            extrapolated = np.asarray(
                [by_label[label]["feature_extrapolation"] for label in labels],
                dtype=bool,
            )
            axis.plot(
                x,
                values,
                color=v3.MODEL_COLORS[model],
                marker=v3.MODEL_MARKERS[model],
                linewidth=1.8,
                markersize=6.5,
                label=model.split("_")[0],
            )
            if np.any(extrapolated):
                axis.scatter(
                    x[extrapolated],
                    values[extrapolated],
                    s=70,
                    facecolors="white",
                    edgecolors=v3.MODEL_COLORS[model],
                    linewidths=1.5,
                    zorder=5,
                )
        axis.set_xticks(x, labels)
        axis.set_title(MODE_SHORT_LABELS[mode].replace("\n", " — "))
        axis.set_ylabel("Activité prédite / activité selon urine")
        axis.grid(True, axis="y", linestyle="--", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=6,
        frameon=False,
    )
    fig.suptitle(
        "Prédictions P8 sans réajustement — activité corporelle par bilan urinaire",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_observed_sensitivity(
    training_dataset: Mapping[str, Any],
    test_dataset: Mapping[str, Any],
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6), layout="constrained")
    for axis, mode in zip(axes.flat, MODE_ORDER):
        for dataset, color, marker in (
            (training_dataset, "#1F77B4", "o"),
            (test_dataset, "#E67E22", "D"),
        ):
            observation = dataset["mode_observations"][mode]
            times = np.asarray(
                [record["hours_after_injection"] for record in observation["records"]],
                dtype=float,
            )
            sensitivity = np.asarray(observation["target_sensitivity"], dtype=float)
            axis.plot(
                times,
                sensitivity,
                marker=marker,
                color=color,
                linewidth=1.8,
                markersize=6.5,
                label=dataset["spec"].label,
            )
        axis.set_title(MODE_SHORT_LABELS[mode].replace("\n", " — "))
        axis.set_xlabel("Temps après injection (h)")
        axis.set_ylabel("S_eff selon urine (cps/MBq)")
        axis.grid(True, linestyle="--", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Sensibilité effective définie par le bilan urinaire", fontsize=15)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_methodology_report(
    simple_results: Mapping[str, Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]],
    stability_rows: Sequence[Mapping[str, Any]] = (),
    j0_summary_rows: Sequence[Mapping[str, Any]] = (),
    timing_rows: Sequence[Mapping[str, Any]] = (),
) -> str:
    lines = [
        "SENSIBILITÉ PLANAIRE AVEC CIBLE DÉRIVÉE DES URINES",
        "=" * 58,
        "",
        "Question",
        "--------",
        "Peut-on calibrer une mesure planaire sur P11, puis estimer sans",
        "réajustement l'activité corporelle de P8 en utilisant le bilan urinaire",
        "comme référence indépendante?",
        "",
        "Cible urinaire",
        "--------------",
        "À chaque heure planaire t:",
        "",
        "  A_physique(t) = A_injectée * 2^(-t/T_physique)",
        "",
        "Chaque intervalle d'urine/couche mesuré est d'abord ramené au temps de",
        "l'injection, puis la somme des collectes terminées est ramenée au temps t",
        "par décroissance physique.",
        "",
        "  A_corps,urine(t) = A_physique(t) - A_excrétée_mesurée(t)",
        "",
        "Le sang n'est pas utilisé dans cette cible de rétention pancorporelle.",
        "L'heure d'injection, l'activité injectée et les collectes proviennent du CSV.",
        "",
        "Séparation des données",
        "----------------------",
        "  Entraînement : P11 mai, trois premiers timepoints.",
        "  Test verrouillé : P8 juin, trois timepoints, sans réajustement.",
        "  P11 juillet est exclu parce qu'aucun fichier d'excrétion n'est disponible.",
        "",
        "Entrées planaires",
        "------------------",
        "Toutes les mesures utilisent l'image complète. Aucun CT, ACCT, Q/SPECT,",
        "crop Q/SPECT ou activité Q/SPECT n'est chargé.",
        "",
        "Analyse principale: six définitions de comptes bruts:",
    ]
    lines.extend(f"  - {SIMPLE_METHOD_LABELS[method]}" for method in SIMPLE_METHOD_ORDER)
    lines.extend(
        [
            "",
            "GM pixel signifie: somme de sqrt(AP_pixel * PA_pixel_aligné).",
            "La TEW et les modèles M0-M9 sont conservés comme analyses",
            "exploratoires secondaires.",
            "",
            "Modélisation",
            "------------",
            "Pour chaque mode:",
            "",
            "  S_eff,urine(t) = R_planaire(t) / A_corps,urine(t)",
            "  A_prédite(t) = R_planaire(t) / S_eff,prédite(t)",
            "",
            "M0 utilise la moyenne arithmétique des trois sensibilités de P11.",
            "M1, M4, M5, M6, M7 et M8 sont des régressions log-linéaires",
            "univariées avec les mêmes définitions planaires que v3. M5 existe",
            "seulement dans les modes TEW. M2 et M9 demandent deux variables plus",
            "un intercept; avec trois points d'entraînement, ils sont non estimables.",
            "",
            "Les ratios spectraux sont calculés après sommation des pixels de chaque",
            "fenêtre. La TEW et la moyenne géométrique sont appliquées pixel par pixel",
            "avant l'intégration globale, comme dans v3.",
            "",
            "Test intra-patient prioritaire:",
            "  S_J0 = R(J0) / A_corps,urine(J0)",
            "  A_prédite(J1,J2) = R(J1,J2) / S_J0, sans réajustement.",
            "",
            "Résultat simple: R = k*A sur P11, puis test sur P8",
            "--------------------------------------------------",
            " Méthode                  | sensibilité P11 | MARE P8 | biais P8 | erreur max",
            "------------------------- | --------------- | ------- | -------- | ----------",
        ]
    )
    for name in SIMPLE_METHOD_ORDER:
        row = simple_results[name]
        lines.append(
            f"{SIMPLE_METHOD_LABELS[name]:25s} | "
            f"{row['sensitivity_cps_per_mbq']:15.4f} | "
            f"{row['activity_mare_percent']:7.2f}% | "
            f"{row['activity_bias_percent']:+8.2f}% | "
            f"{row['maximum_activity_error_percent']:10.2f}%"
        )
    if j0_summary_rows:
        lines.extend(
            [
                "",
                "Test intra-patient: calibration J0, test J1-J2",
                "------------------------------------------------",
                " Patient | méthode                   | MARE suivi | biais suivi | erreur max",
                "------- | ------------------------- | ---------- | ----------- | ----------",
            ]
        )
        for row in j0_summary_rows:
            lines.append(
                f"{str(row['patient_id']).upper():7s} | "
                f"{str(row['method_label']):25s} | "
                f"{float(row['followup_mare_percent']):10.2f}% | "
                f"{float(row['followup_bias_percent']):+11.2f}% | "
                f"{float(row['followup_max_error_percent']):10.2f}%"
            )
    if timing_rows:
        finite_next = [
            row
            for row in timing_rows
            if np.isfinite(float(row["minutes_to_next_collection"]))
        ]
        closest = min(
            finite_next,
            key=lambda row: float(row["minutes_to_next_collection"]),
        )
        lines.extend(
            [
                "",
                "Contrôle temporel des collectes",
                "---------------------------------",
                "La cible principale inclut uniquement les collectes terminées au",
                "moment du planaire. Deux scénarios de sensibilité sont aussi fournis:",
                "excrétion uniforme dans l'intervalle et intervalle courant entièrement",
                "excrété.",
                f"Collecte future la plus proche: {str(closest['patient_id']).upper()} "
                f"{closest['timepoint']}, "
                f"{float(closest['minutes_to_next_collection']):.1f} min après le planaire.",
            ]
        )
    if stability_rows:
        lines.extend(
            [
                "",
                "Stabilité intra-patient de S_eff,urine",
                "--------------------------------------",
                " Mode                       | P11 moyenne | P11 CV | P8 moyenne | P8 CV",
                "--------------------------- | ----------- | ------ | ---------- | -----",
            ]
        )
        for mode in MODE_ORDER:
            p11 = next(
                row
                for row in stability_rows
                if row["rate_mode"] == mode and row["dataset_role"] == "training"
            )
            p8 = next(
                row
                for row in stability_rows
                if row["rate_mode"] == mode and row["dataset_role"] == "test"
            )
            lines.append(
                f"{MODE_SHORT_LABELS[mode].replace(chr(10), ' '):27s} | "
                f"{float(p11['mean_sensitivity_cps_per_mbq']):11.4f} | "
                f"{float(p11['cv_percent']):6.2f}% | "
                f"{float(p8['mean_sensitivity_cps_per_mbq']):10.4f} | "
                f"{float(p8['cv_percent']):5.2f}%"
            )
    lines.extend(
        [
            "",
            "Résultats exploratoires des modèles",
            "-----------------------------------",
            " Mode                       | meilleur modèle estimable | MARE P8",
            "--------------------------- | -------------------------- | -------",
        ]
    )
    for mode in MODE_ORDER:
        candidates = [
            row
            for row in metrics_rows
            if row["rate_mode"] == mode and row["fit_status"] == "fitted"
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: float(row["activity_mare_percent"]))
        lines.append(
            f"{MODE_SHORT_LABELS[mode].replace(chr(10), ' '):27s} | "
            f"{best['model'].split('_')[0]:26s} | "
            f"{float(best['activity_mare_percent']):7.2f}%"
        )
    lines.extend(
        [
            "",
            "Interprétation et limites",
            "-------------------------",
            "* La stabilité de S_eff et la calibration J0->J1/J2 constituent le",
            "  test principal directement lié à la question intra-patient.",
            "* La droite R=k*A sur P11 puis P8 est le test interpatient simple.",
            "* Les ratios M1-M9 sont exploratoires; leurs définitions ont été héritées",
            "  du travail v3 et ne constituent pas une sélection indépendante.",
            "* Trois points d'entraînement donnent seulement un degré de liberté",
            "  résiduel aux modèles univariés. Une faible MARE peut être instable.",
            "* Deux patients ne suffisent pas pour conclure à la généralisation.",
            "* La cible suppose que les collectes urine/couches sont complètes et",
            "  correctement synchronisées. Les voies d'excrétion non mesurées et les",
            "  collectes futures ne sont pas incluses.",
            "* Cette référence est un bilan d'activité corporelle, pas une mesure",
            "  d'imagerie volumique. Elle répond directement à la rétention globale.",
            "* Le Q/SPECT demeure nécessaire plus tard pour comparer la précision des",
            "  deux références et pour les questions spatiales, mais il n'entre pas",
            "  dans les calculs de ce module.",
            "",
        ]
    )
    return "\n".join(lines)


def run_urine_target_analysis(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timepoint_count: int = DEFAULT_TIMEPOINT_COUNT,
) -> Dict[str, Any]:
    """Run the urine-target P11-to-P8 transfer analysis and save artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    interpatient_dir = output_dir / "interpatient_p11_to_p8"
    intrapatient_dir = output_dir / "intrapatient_j0_to_j1_j2"
    exploratory_dir = output_dir / "exploratory_models"
    timing_dir = output_dir / "qc_timing"
    for directory in (
        interpatient_dir,
        intrapatient_dir,
        exploratory_dir,
        timing_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    training_dataset = build_urine_dataset(TRAINING_DATASET, timepoint_count)
    test_dataset = build_urine_dataset(TEST_DATASET, timepoint_count)

    metrics_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    fits_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in MODE_ORDER:
        result = fit_and_test_mode(
            mode,
            training_dataset["mode_observations"][mode],
            test_dataset["mode_observations"][mode],
        )
        metrics_rows.extend(result["metrics"])
        prediction_rows.extend(result["predictions"])
        fits_by_mode[mode] = result["fits"]

    simple_results = simple_zero_origin_transfer(training_dataset, test_dataset)
    j0_training = j0_calibration_results(training_dataset)
    j0_test = j0_calibration_results(test_dataset)
    j0_prediction_rows: list[Dict[str, Any]] = []
    j0_summary_rows: list[Dict[str, Any]] = []
    for dataset, j0_results in (
        (training_dataset, j0_training),
        (test_dataset, j0_test),
    ):
        predictions, summaries = j0_calibration_rows(dataset, j0_results)
        j0_prediction_rows.extend(predictions)
        j0_summary_rows.extend(summaries)

    simple_metrics_rows: list[Dict[str, Any]] = []
    simple_prediction_rows: list[Dict[str, Any]] = []
    for method in SIMPLE_METHOD_ORDER:
        result = simple_results[method]
        simple_metrics_rows.append(
            {
                "method": method,
                "method_label": SIMPLE_METHOD_LABELS[method],
                "training_dataset": TRAINING_DATASET.key,
                "test_dataset": TEST_DATASET.key,
                "sensitivity_p11_cps_per_mbq": result[
                    "sensitivity_cps_per_mbq"
                ],
                "activity_bias_percent": result["activity_bias_percent"],
                "activity_mare_percent": result["activity_mare_percent"],
                "activity_rmse_percent": result["activity_rmse_percent"],
                "maximum_activity_error_percent": result[
                    "maximum_activity_error_percent"
                ],
            }
        )
        prediction = np.asarray(
            result["predicted_test_activity_mbq"], dtype=float
        )
        reference = np.asarray(test_dataset["reference_activity"], dtype=float)
        for index, record in enumerate(test_dataset["records"]):
            simple_prediction_rows.append(
                {
                    "method": method,
                    "method_label": SIMPLE_METHOD_LABELS[method],
                    "patient_id": test_dataset["spec"].patient_id,
                    "cycle_id": test_dataset["spec"].cycle_id,
                    "timepoint": record["timepoint"],
                    "hours_after_injection": record["hours_after_injection"],
                    "urine_reference_activity_mbq": float(reference[index]),
                    "predicted_activity_mbq": float(prediction[index]),
                    "predicted_over_reference": float(
                        prediction[index] / reference[index]
                    ),
                    "activity_error_percent": float(
                        100.0 * (prediction[index] / reference[index] - 1.0)
                    ),
                }
            )

    timing_rows = timing_qc_rows(training_dataset) + timing_qc_rows(test_dataset)
    stability_rows = sensitivity_stability_rows(training_dataset, test_dataset)
    observation_rows: list[Dict[str, Any]] = []
    for dataset in (training_dataset, test_dataset):
        for mode in MODE_ORDER:
            observation = dataset["mode_observations"][mode]
            for index, record in enumerate(observation["records"]):
                observation_rows.append(
                    {
                        "rate_mode": mode,
                        "rate_mode_label": v3.RATE_MODES[mode].label,
                        **record,
                        "measurement_rate_cps": float(
                            observation["measurement_rate"][index]
                        ),
                        "urine_reference_activity_mbq": float(
                            observation["reference_activity"][index]
                        ),
                        "observed_sensitivity_cps_per_mbq": float(
                            observation["target_sensitivity"][index]
                        ),
                    }
                )

    paths = {
        "simple_metrics_csv": _write_csv(
            interpatient_dir / "simple_transfer_metrics.csv", simple_metrics_rows
        ),
        "simple_predictions_csv": _write_csv(
            interpatient_dir / "simple_transfer_predictions.csv",
            simple_prediction_rows,
        ),
        "simple_transfer_figure": plot_simple_transfer(
            training_dataset,
            test_dataset,
            simple_results,
            interpatient_dir / "simple_count_transfer.png",
        ),
        "simple_parity_figure": plot_simple_prediction_parity(
            test_dataset,
            simple_results,
            interpatient_dir / "predicted_vs_urine_reference.png",
        ),
        "simple_error_figure": plot_simple_relative_error(
            test_dataset,
            simple_results,
            interpatient_dir / "relative_error_vs_time.png",
        ),
        "j0_predictions_csv": _write_csv(
            intrapatient_dir / "j0_calibration_predictions.csv",
            j0_prediction_rows,
        ),
        "j0_summary_csv": _write_csv(
            intrapatient_dir / "j0_calibration_summary.csv",
            j0_summary_rows,
        ),
        "j0_prediction_figure": plot_j0_calibration_predictions(
            (
                (training_dataset, j0_training),
                (test_dataset, j0_test),
            ),
            intrapatient_dir / "j0_calibration_predictions.png",
        ),
        "normalized_sensitivity_figure": plot_normalized_simple_sensitivity(
            (training_dataset, test_dataset),
            intrapatient_dir / "normalized_sensitivity_vs_time.png",
        ),
        "timing_csv": _write_csv(
            timing_dir / "imaging_urine_timing_qc.csv", timing_rows
        ),
        "timeline_figure": plot_imaging_urine_timeline(
            (training_dataset, test_dataset),
            timing_dir / "imaging_urine_timeline.png",
        ),
        "timing_scenarios_figure": plot_urine_target_timing_scenarios(
            (training_dataset, test_dataset),
            timing_dir / "urine_target_timing_scenarios.png",
        ),
        "observations_csv": _write_csv(
            exploratory_dir / "urine_target_observations.csv", observation_rows
        ),
        "metrics_csv": _write_csv(
            exploratory_dir / "urine_target_model_metrics.csv", metrics_rows
        ),
        "predictions_csv": _write_csv(
            exploratory_dir / "urine_target_model_predictions.csv", prediction_rows
        ),
        "stability_csv": _write_csv(
            exploratory_dir / "urine_target_sensitivity_stability.csv",
            stability_rows,
        ),
        "observed_sensitivity_figure": plot_observed_sensitivity(
            training_dataset,
            test_dataset,
            exploratory_dir / "urine_target_observed_sensitivity.png",
        ),
        "model_mare_figure": plot_mode_model_mare(
            metrics_rows, exploratory_dir / "urine_target_mode_model_mare.png"
        ),
        "prediction_figure": plot_primary_predictions(
            prediction_rows,
            exploratory_dir / "urine_target_model_predictions.png",
        ),
    }
    methodology_path = output_dir / "urine_target_methodology_and_results.txt"
    methodology_path.write_text(
        build_methodology_report(
            simple_results,
            metrics_rows,
            stability_rows,
            j0_summary_rows,
            timing_rows,
        ),
        encoding="utf-8",
    )
    paths["methodology_report"] = methodology_path
    return {
        "training_dataset": training_dataset,
        "test_dataset": test_dataset,
        "simple_results": simple_results,
        "j0_training": j0_training,
        "j0_test": j0_test,
        "j0_summaries": j0_summary_rows,
        "timing_qc": timing_rows,
        "stability": stability_rows,
        "fits_by_mode": fits_by_mode,
        "metrics": metrics_rows,
        "predictions": prediction_rows,
        "paths": paths,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--timepoints", type=int, default=DEFAULT_TIMEPOINT_COUNT
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_urine_target_analysis(args.output_dir, args.timepoints)
    print("Urine-target planar sensitivity analysis")
    print(
        f"Training: {TRAINING_DATASET.label}; test: {TEST_DATASET.label}; "
        f"n={args.timepoints}+{args.timepoints}"
    )
    for name, values in result["simple_results"].items():
        print(
            f"  {name}: S={values['sensitivity_cps_per_mbq']:.4f} cps/MBq, "
            f"P8 MARE={values['activity_mare_percent']:.2f}%"
        )
    for name, path in result["paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
