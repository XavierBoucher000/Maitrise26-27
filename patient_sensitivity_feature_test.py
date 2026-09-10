"""One-patient univariate feasibility tests for planar attenuation predictors.

Each candidate planar feature is evaluated alone with leave-one-timepoint-out
validation. The script compares every univariate model with a training-mean
baseline. Results are exploratory because five timepoints from one patient are
not independent validation data.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import attenuation_global_model_v2 as global_model
import patient_effective_sensitivity as sensitivity


OUTPUT_DIR = Path(__file__).resolve().parent / "fig" / "patient_sensitivity_features"
VALUES_PATH = OUTPUT_DIR / "univariate_feature_results.csv"
TIMEPOINT_VALUES_PATH = OUTPUT_DIR / "timepoint_feature_values.csv"
REPORT_PATH = OUTPUT_DIR / "univariate_feature_report.txt"
SUMMARY_FIGURE_PATH = OUTPUT_DIR / "univariate_feature_summary.png"
PREDICTION_FIGURE_PATH = OUTPUT_DIR / "best_univariate_predictions.png"

FEATURE_NAMES = (
    *global_model.ALL_FEATURE_NAMES,
    "TEW retained fraction",
    "log photopeak count rate",
)

TARGET_LABELS = {
    "tew_effective_sensitivity": "Sensibilité effective TEW (cps/MBq)",
    "ct_effective_factor": "Facteur effectif CT",
}


def leave_one_out_log_linear(feature: Sequence[float], target: Sequence[float]) -> Dict[str, Any]:
    """Fit log(target) = intercept + slope*feature in five LOO folds."""
    x = np.asarray(feature, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 3:
        raise ValueError("feature and target must be same-length 1D arrays with >=3 points")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(y <= 0.0):
        raise ValueError("feature must be finite and target must be finite and positive")

    predictions = np.empty_like(y)
    baseline_predictions = np.empty_like(y)
    slopes = np.empty_like(y)
    for held_out in range(y.size):
        training = np.arange(y.size) != held_out
        x_train = x[training]
        log_y_train = np.log(y[training])
        baseline_predictions[held_out] = float(np.mean(y[training]))
        if float(np.std(x_train)) <= np.finfo(float).eps:
            slopes[held_out] = 0.0
            predictions[held_out] = float(np.exp(np.mean(log_y_train)))
            continue
        design = np.column_stack([np.ones(x_train.size), x_train])
        intercept, slope = np.linalg.lstsq(design, log_y_train, rcond=None)[0]
        slopes[held_out] = float(slope)
        predictions[held_out] = float(np.exp(intercept + slope * x[held_out]))

    relative_errors = 100.0 * (predictions - y) / y
    baseline_relative_errors = 100.0 * (baseline_predictions - y) / y
    correlation = (
        np.nan
        if float(np.std(x)) <= np.finfo(float).eps
        else float(np.corrcoef(x, y)[0, 1])
    )
    return {
        "predictions": predictions,
        "baseline_predictions": baseline_predictions,
        "relative_errors_percent": relative_errors,
        "baseline_relative_errors_percent": baseline_relative_errors,
        "mare_percent": float(np.mean(np.abs(relative_errors))),
        "baseline_mare_percent": float(np.mean(np.abs(baseline_relative_errors))),
        "rmse": float(np.sqrt(np.mean((predictions - y) ** 2))),
        "baseline_rmse": float(np.sqrt(np.mean((baseline_predictions - y) ** 2))),
        "maximum_absolute_error_percent": float(np.max(np.abs(relative_errors))),
        "correlation": correlation,
        "mean_fold_slope": float(np.mean(slopes)),
    }


def build_analysis() -> Dict[str, Any]:
    sensitivity_rows = sensitivity.calculate_patient_effective_sensitivity()
    dataset = global_model.build_global_dataset()
    if len(sensitivity_rows) != len(dataset["spectral_rows"]):
        raise ValueError("Sensitivity and spectral datasets have different timepoint counts")

    feature_values: Dict[str, np.ndarray] = {}
    for name in global_model.ALL_FEATURE_NAMES:
        feature_values[name] = np.asarray(
            [row["features"][name] for row in dataset["spectral_rows"]],
            dtype=np.float64,
        )
    feature_values["TEW retained fraction"] = np.asarray(
        [row["tew_retained_fraction"] for row in sensitivity_rows], dtype=np.float64
    )
    feature_values["log photopeak count rate"] = np.log(
        np.asarray(
            [row["photopeak_rate_cps"] for row in dataset["spectral_rows"]],
            dtype=np.float64,
        )
    )

    targets = {
        "tew_effective_sensitivity": np.asarray(
            [row["tew_effective_sensitivity_cps_per_mbq"] for row in sensitivity_rows],
            dtype=np.float64,
        ),
        "ct_effective_factor": np.asarray(dataset["target_factor"], dtype=np.float64),
    }
    results: List[Dict[str, Any]] = []
    for target_name, target in targets.items():
        for feature_name, feature in feature_values.items():
            metrics = leave_one_out_log_linear(feature, target)
            results.append(
                {
                    "target": target_name,
                    "feature": feature_name,
                    **metrics,
                }
            )
    return {
        "days": np.asarray([row["planar_day"] for row in sensitivity_rows]),
        "labels": [row["label"] for row in sensitivity_rows],
        "crop_bounds": [
            (int(row["crop_top"]), int(row["crop_bottom"]))
            for row in sensitivity_rows
        ],
        "feature_values": feature_values,
        "targets": targets,
        "results": results,
    }


def best_result(analysis: Dict[str, Any], target_name: str) -> Dict[str, Any]:
    candidates = [row for row in analysis["results"] if row["target"] == target_name]
    return min(candidates, key=lambda row: row["mare_percent"])


def write_outputs(analysis: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with VALUES_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target", "feature", "pearson_r", "loo_mare_percent",
                "constant_loo_mare_percent", "loo_rmse", "constant_loo_rmse",
                "maximum_absolute_error_percent", "mean_fold_slope",
            ]
        )
        for row in analysis["results"]:
            writer.writerow(
                [
                    row["target"], row["feature"], row["correlation"],
                    row["mare_percent"], row["baseline_mare_percent"], row["rmse"],
                    row["baseline_rmse"], row["maximum_absolute_error_percent"],
                    row["mean_fold_slope"],
                ]
            )

    with TIMEPOINT_VALUES_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["label", "day", "crop_top", "crop_bottom", *FEATURE_NAMES, *TARGET_LABELS]
        )
        for index, label in enumerate(analysis["labels"]):
            writer.writerow(
                [
                    label,
                    analysis["days"][index],
                    *analysis["crop_bounds"][index],
                    *[analysis["feature_values"][name][index] for name in FEATURE_NAMES],
                    *[analysis["targets"][name][index] for name in TARGET_LABELS],
                ]
            )

    lines = [
        "One-patient univariate planar-feature feasibility test",
        "=======================================================",
        "",
        "Method:",
        "  - profile_qspect crop for every timepoint",
        "  - one planar feature at a time",
        "  - log-linear leave-one-timepoint-out prediction",
        "  - comparison against the mean of the four training timepoints",
        "  - no p-values and no claim of independent validation",
        "",
    ]
    for target_name, target_label in TARGET_LABELS.items():
        baseline = next(
            row["baseline_mare_percent"]
            for row in analysis["results"]
            if row["target"] == target_name
        )
        lines.extend([target_label, "-" * len(target_label)])
        ranked = sorted(
            (row for row in analysis["results"] if row["target"] == target_name),
            key=lambda row: row["mare_percent"],
        )
        for row in ranked:
            lines.append(
                f"  {row['feature']}: r={row['correlation']:+.3f}, "
                f"LOO MARE={row['mare_percent']:.2f}%, "
                f"constant={baseline:.2f}%, max error={row['maximum_absolute_error_percent']:.2f}%"
            )
        winner = ranked[0]
        verdict = "better" if winner["mare_percent"] < baseline else "not better"
        lines.extend(
            [
                f"  Best: {winner['feature']} ({verdict} than constant baseline).",
                "",
            ]
        )
    lines.extend(
        [
            "Interpretation:",
            "  A strong full-data correlation can still fail LOO validation with five points.",
            "  The absolute count-rate feature is exploratory and may encode activity/time,",
            "  not attenuation. Confirmation requires the second patient.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(analysis: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), facecolor="white", layout="constrained")
    for axis, target_name in zip(axes, TARGET_LABELS):
        rows = [row for row in analysis["results"] if row["target"] == target_name]
        x = np.arange(len(rows))
        model_error = [row["mare_percent"] for row in rows]
        baseline = rows[0]["baseline_mare_percent"]
        axis.bar(x, model_error, color="#1f77b4", alpha=0.85)
        axis.axhline(baseline, color="#d62728", linestyle="--", linewidth=2, label=f"Moyenne constante: {baseline:.1f} %")
        axis.set_xticks(x, [row["feature"] for row in rows], rotation=18, ha="right")
        axis.set_ylabel("Erreur absolue relative LOO moyenne (%)")
        axis.set_title(TARGET_LABELS[target_name])
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Une variable planaire à la fois — validation leave-one-timepoint-out")
    fig.savefig(SUMMARY_FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_best_predictions(analysis: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), facecolor="white", layout="constrained")
    days = analysis["days"]
    for axis, target_name in zip(axes, TARGET_LABELS):
        row = best_result(analysis, target_name)
        measured = analysis["targets"][target_name]
        axis.plot(days, measured, "ko-", linewidth=2.2, label="Cible mesurée")
        axis.plot(days, row["predictions"], "s--", linewidth=2, label=f"LOO: {row['feature']}")
        axis.plot(days, row["baseline_predictions"], "^:", linewidth=1.8, label="Moyenne des jours d’entraînement")
        axis.set_xlabel("Temps après la première acquisition (jours)")
        axis.set_ylabel(TARGET_LABELS[target_name])
        axis.set_title(f"Meilleure variable — MARE {row['mare_percent']:.1f} %")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(PREDICTION_FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_analysis() -> Dict[str, Any]:
    analysis = build_analysis()
    write_outputs(analysis)
    plot_summary(analysis)
    plot_best_predictions(analysis)
    for target_name in TARGET_LABELS:
        winner = best_result(analysis, target_name)
        print(
            f"{target_name}: best={winner['feature']} | "
            f"LOO MARE={winner['mare_percent']:.2f}% | "
            f"constant={winner['baseline_mare_percent']:.2f}%"
        )
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved summary: {SUMMARY_FIGURE_PATH}")
    print(f"Saved predictions: {PREDICTION_FIGURE_PATH}")
    return analysis


if __name__ == "__main__":
    run_analysis()
