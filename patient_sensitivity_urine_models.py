"""Organized interpatient M0-M9 urine-target sensitivity analyses.

The simple J0 calibration remains the primary result. This module is limited
to bidirectional strict three-point interpatient M0-M9 transfers. No CT or
Q/SPECT enters the models.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figure_layout
import patient_sensitivity_multicycle_v3 as v3
import patient_sensitivity_urine_target as urine


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    figure_layout.analysis_dir("patient_sensitivity_urine_target.py")
    / "3_sensibilite_modeles"
    / "4_ancienne_analyse_p11_p8"
    / "modeles"
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _subset_observation(
    observation: Mapping[str, Any], indices: Sequence[int]
) -> Dict[str, Any]:
    selected = np.asarray(indices, dtype=int)
    return {
        "records": [observation["records"][int(index)] for index in selected],
        "features": {
            name: np.asarray(values, dtype=float)[selected]
            for name, values in observation["features"].items()
        },
        "measurement_rate": np.asarray(
            observation["measurement_rate"], dtype=float
        )[selected],
        "reference_activity": np.asarray(
            observation["reference_activity"], dtype=float
        )[selected],
        "target_sensitivity": np.asarray(
            observation["target_sensitivity"], dtype=float
        )[selected],
    }


def _extrapolation_flags(
    fit: Mapping[str, Any], test: Mapping[str, Any]
) -> np.ndarray:
    flags = np.zeros(len(test["records"]), dtype=bool)
    for name, bounds in fit.get("feature_ranges", {}).items():
        values = np.asarray(test["features"][name], dtype=float)
        flags |= (values < float(bounds[0])) | (values > float(bounds[1]))
    return flags


def _activity_metrics(
    reference: Sequence[float], predicted: Sequence[float]
) -> Dict[str, float]:
    reference_array = np.asarray(reference, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    errors = 100.0 * (predicted_array / reference_array - 1.0)
    return {
        "activity_bias_percent": float(np.mean(errors)),
        "activity_mare_percent": float(np.mean(np.abs(errors))),
        "activity_rmse_percent": float(np.sqrt(np.mean(errors**2))),
        "maximum_activity_error_percent": float(np.max(np.abs(errors))),
    }


def _model_status_row(
    analysis: str,
    split: str,
    mode: str,
    model: str,
    n_train: int,
    n_test: int,
    status: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "analysis": analysis,
        "split": split,
        "rate_mode": mode,
        "rate_mode_label": v3.RATE_MODES[mode].label,
        "model": model,
        "model_label": v3.MODEL_LABELS[model],
        "n_train": n_train,
        "n_test": n_test,
        "fit_status": status,
        "fit_reason": reason,
    }


def intrapatient_loto_results(
    fitted_p11_five: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Leave one P11 timepoint out; fit on four and predict the held-out point."""
    metric_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    for mode in urine.MODE_ORDER:
        observation = fitted_p11_five["mode_observations"][mode]
        allowed = set(urine._mode_model_names(mode))
        point_count = len(observation["records"])
        for model in v3.MODEL_ORDER:
            base = _model_status_row(
                "intrapatient_loto",
                "P11 cinq points, leave-one-timepoint-out",
                mode,
                model,
                point_count - 1,
                point_count,
                "not_run",
                "",
            )
            if model not in allowed:
                metric_rows.append(
                    {
                        **base,
                        "fit_status": "not_applicable",
                        "fit_reason": "M5 requires a TEW measurement mode",
                        "activity_bias_percent": math.nan,
                        "activity_mare_percent": math.nan,
                        "activity_rmse_percent": math.nan,
                        "maximum_activity_error_percent": math.nan,
                        "extrapolated_predictions": 0,
                    }
                )
                continue

            predicted_activity: list[float] = []
            reference_activity: list[float] = []
            fold_rows: list[Dict[str, Any]] = []
            failed_reason = ""
            for test_index in range(point_count):
                train_indices = [
                    index for index in range(point_count) if index != test_index
                ]
                train = _subset_observation(observation, train_indices)
                test = _subset_observation(observation, [test_index])
                fit = v3.fit_model(model, train)
                if fit["status"] != "fitted":
                    failed_reason = str(fit["reason"])
                    break
                predicted_sensitivity = v3.predict_model(fit, test)
                rate = float(test["measurement_rate"][0])
                reference = float(test["reference_activity"][0])
                prediction = rate / float(predicted_sensitivity[0])
                flag = bool(_extrapolation_flags(fit, test)[0])
                predicted_activity.append(prediction)
                reference_activity.append(reference)
                record = test["records"][0]
                fold_rows.append(
                    {
                        "analysis": "intrapatient_loto",
                        "split": "P11 cinq points, leave-one-timepoint-out",
                        "rate_mode": mode,
                        "rate_mode_label": v3.RATE_MODES[mode].label,
                        "model": model,
                        "model_label": v3.MODEL_LABELS[model],
                        "patient_id": record["patient_id"],
                        "timepoint": record["timepoint"],
                        "hours_after_injection": record["hours_after_injection"],
                        "target_status": record.get(
                            "excretion_target_status", "unknown"
                        ),
                        "urine_reference_activity_mbq": reference,
                        "predicted_activity_mbq": prediction,
                        "activity_error_percent": 100.0
                        * (prediction / reference - 1.0),
                        "feature_extrapolation": flag,
                    }
                )

            if failed_reason:
                metric_rows.append(
                    {
                        **base,
                        "fit_status": "not_estimable",
                        "fit_reason": failed_reason,
                        "activity_bias_percent": math.nan,
                        "activity_mare_percent": math.nan,
                        "activity_rmse_percent": math.nan,
                        "maximum_activity_error_percent": math.nan,
                        "extrapolated_predictions": 0,
                    }
                )
                continue
            prediction_rows.extend(fold_rows)
            metric_rows.append(
                {
                    **base,
                    "fit_status": "fitted",
                    "fit_reason": "",
                    **_activity_metrics(reference_activity, predicted_activity),
                    "extrapolated_predictions": int(
                        sum(bool(row["feature_extrapolation"]) for row in fold_rows)
                    ),
                }
            )
    return metric_rows, prediction_rows


def intrapatient_j0_results(
    fitted_p11_five: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Calibrate one constant sensitivity at P11-J0 and test J1-J6."""
    metric_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    for mode in urine.MODE_ORDER:
        observation = fitted_p11_five["mode_observations"][mode]
        rates = np.asarray(observation["measurement_rate"], dtype=float)
        references = np.asarray(observation["reference_activity"], dtype=float)
        sensitivity = float(rates[0] / references[0])
        predictions = rates / sensitivity
        metric_rows.append(
            {
                "analysis": "intrapatient_J0_only",
                "split": "P11 J0→J1/J2/J3/J6",
                "rate_mode": mode,
                "rate_mode_label": v3.RATE_MODES[mode].label,
                "model": "J0_constant",
                "model_label": "Sensibilité calibrée uniquement à J0",
                "n_train": 1,
                "n_test": len(references) - 1,
                "fit_status": "calibrated",
                "fit_reason": "",
                "sensitivity_j0_cps_per_mbq": sensitivity,
                **_activity_metrics(references[1:], predictions[1:]),
            }
        )
        for index, record in enumerate(observation["records"]):
            prediction_rows.append(
                {
                    "analysis": "intrapatient_J0_only",
                    "split": "P11 J0→J1/J2/J3/J6",
                    "rate_mode": mode,
                    "rate_mode_label": v3.RATE_MODES[mode].label,
                    "model": "J0_constant",
                    "model_label": "Sensibilité calibrée uniquement à J0",
                    "patient_id": record["patient_id"],
                    "timepoint": record["timepoint"],
                    "hours_after_injection": record["hours_after_injection"],
                    "used_for_calibration": index == 0,
                    "target_status": record.get(
                        "excretion_target_status", "unknown"
                    ),
                    "sensitivity_j0_cps_per_mbq": sensitivity,
                    "urine_reference_activity_mbq": float(references[index]),
                    "predicted_activity_mbq": float(predictions[index]),
                    "activity_error_percent": float(
                        100.0 * (predictions[index] / references[index] - 1.0)
                    ),
                }
            )
    return metric_rows, prediction_rows


def plot_intrapatient_j0_errors(
    prediction_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    timepoints: list[str] = []
    for row in prediction_rows:
        timepoint = str(row["timepoint"])
        if timepoint != "J0" and timepoint not in timepoints:
            timepoints.append(timepoint)
    x = np.arange(len(timepoints), dtype=float)
    colors = plt.get_cmap("tab10").colors
    fig, axis = plt.subplots(figsize=(12.8, 6.8))
    for mode_index, mode in enumerate(urine.MODE_ORDER):
        rows = [
            row
            for row in prediction_rows
            if row["rate_mode"] == mode and row["timepoint"] != "J0"
        ]
        by_timepoint = {str(row["timepoint"]): row for row in rows}
        values = [
            abs(float(by_timepoint[timepoint]["activity_error_percent"]))
            for timepoint in timepoints
        ]
        axis.plot(
            x,
            values,
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=colors[mode_index],
            label=v3.RATE_MODES[mode].label,
        )
    axis.set_xticks(x, timepoints)
    axis.set_xlabel("Timepoint prédit sans réajustement")
    axis.set_ylabel("Erreur absolue relative (%)")
    axis.set_title("P11 — sensibilité calibrée uniquement à J0", fontsize=17)
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend(frameon=False, fontsize=9.5, ncol=2)
    fig.text(
        0.5,
        0.015,
        "Cible urinaire continue; J0 calibre et les quatre points suivants sont des tests.",
        ha="center",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_intrapatient_j0_mare(
    metric_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    rows_by_mode = {str(row["rate_mode"]): row for row in metric_rows}
    values = [
        float(rows_by_mode[mode]["activity_mare_percent"])
        for mode in urine.MODE_ORDER
    ]
    labels = [urine.MODE_SHORT_LABELS[mode] for mode in urine.MODE_ORDER]
    colors = plt.get_cmap("tab10").colors[: len(values)]
    fig, axis = plt.subplots(figsize=(11.8, 6.4))
    bars = axis.bar(np.arange(len(values)), values, color=colors)
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    axis.set_xticks(np.arange(len(values)), labels, rotation=18, ha="right")
    axis.set_ylabel("MARE sur J1, J2, J3 et J6 (%)")
    axis.set_title("P11 — performance d’une sensibilité calibrée uniquement à J0")
    axis.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _interpatient_direction(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    direction: str,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    metric_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    for mode in urine.MODE_ORDER:
        training = source["mode_observations"][mode]
        test = target["mode_observations"][mode]
        allowed = set(urine._mode_model_names(mode))
        for model in v3.MODEL_ORDER:
            base = _model_status_row(
                "interpatient_transfer",
                direction,
                mode,
                model,
                len(training["records"]),
                len(test["records"]),
                "not_run",
                "",
            )
            if model not in allowed:
                metric_rows.append(
                    {
                        **base,
                        "fit_status": "not_applicable",
                        "fit_reason": "M5 requires a TEW measurement mode",
                        "activity_bias_percent": math.nan,
                        "activity_mare_percent": math.nan,
                        "activity_rmse_percent": math.nan,
                        "maximum_activity_error_percent": math.nan,
                        "extrapolated_predictions": 0,
                    }
                )
                continue
            fit = v3.fit_model(model, training)
            if fit["status"] != "fitted":
                metric_rows.append(
                    {
                        **base,
                        "fit_status": fit["status"],
                        "fit_reason": fit["reason"],
                        "activity_bias_percent": math.nan,
                        "activity_mare_percent": math.nan,
                        "activity_rmse_percent": math.nan,
                        "maximum_activity_error_percent": math.nan,
                        "extrapolated_predictions": 0,
                    }
                )
                continue
            predicted_sensitivity = v3.predict_model(fit, test)
            rates = np.asarray(test["measurement_rate"], dtype=float)
            references = np.asarray(test["reference_activity"], dtype=float)
            predictions = rates / predicted_sensitivity
            flags = _extrapolation_flags(fit, test)
            metric_rows.append(
                {
                    **base,
                    "fit_status": "fitted",
                    "fit_reason": "",
                    **_activity_metrics(references, predictions),
                    "extrapolated_predictions": int(np.count_nonzero(flags)),
                }
            )
            for index, record in enumerate(test["records"]):
                prediction_rows.append(
                    {
                        "analysis": "interpatient_transfer",
                        "split": direction,
                        "rate_mode": mode,
                        "rate_mode_label": v3.RATE_MODES[mode].label,
                        "model": model,
                        "model_label": v3.MODEL_LABELS[model],
                        "patient_id": record["patient_id"],
                        "timepoint": record["timepoint"],
                        "hours_after_injection": record["hours_after_injection"],
                        "urine_reference_activity_mbq": float(references[index]),
                        "predicted_activity_mbq": float(predictions[index]),
                        "predicted_over_urine_reference": float(
                            predictions[index] / references[index]
                        ),
                        "activity_error_percent": float(
                            100.0 * (predictions[index] / references[index] - 1.0)
                        ),
                        "feature_extrapolation": bool(flags[index]),
                    }
                )
    return metric_rows, prediction_rows


def interpatient_results(
    p11_three: Mapping[str, Any], p8_three: Mapping[str, Any]
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    metrics: list[Dict[str, Any]] = []
    predictions: list[Dict[str, Any]] = []
    for source, target, direction in (
        (p11_three, p8_three, "P11→P8"),
        (p8_three, p11_three, "P8→P11"),
    ):
        rows, points = _interpatient_direction(source, target, direction)
        metrics.extend(rows)
        predictions.extend(points)
    return metrics, predictions


def _plot_metric_heatmaps(
    metric_rows: Sequence[Mapping[str, Any]],
    splits: Sequence[str],
    title: str,
    output_path: Path,
) -> Path:
    matrices: list[np.ndarray] = []
    for split in splits:
        matrix = np.full((len(v3.MODEL_ORDER), len(urine.MODE_ORDER)), np.nan)
        for row_index, model in enumerate(v3.MODEL_ORDER):
            for column_index, mode in enumerate(urine.MODE_ORDER):
                matches = [
                    row
                    for row in metric_rows
                    if row["split"] == split
                    and row["model"] == model
                    and row["rate_mode"] == mode
                ]
                if len(matches) == 1 and matches[0]["fit_status"] == "fitted":
                    matrix[row_index, column_index] = float(
                        matches[0]["activity_mare_percent"]
                    )
        matrices.append(matrix)
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    vmax = max(10.0, float(np.percentile(finite, 90))) if finite.size else 10.0
    fig, axes = plt.subplots(
        1, len(splits), figsize=(8.2 * len(splits), 8.0), squeeze=False
    )
    image = None
    for axis, split, matrix in zip(axes[0], splits, matrices):
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            cmap="YlOrRd",
            vmin=0.0,
            vmax=vmax,
            aspect="auto",
        )
        axis.set_xticks(
            range(len(urine.MODE_ORDER)),
            [urine.MODE_SHORT_LABELS[mode] for mode in urine.MODE_ORDER],
            rotation=20,
            ha="right",
        )
        axis.set_yticks(
            range(len(v3.MODEL_ORDER)),
            [v3.MODEL_LABELS[model].split(" — ")[0] for model in v3.MODEL_ORDER],
        )
        axis.set_title(split, fontsize=14)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    "N/E" if not np.isfinite(value) else f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=(
                        "white"
                        if np.isfinite(value) and min(value, vmax) > 0.58 * vmax
                        else "black"
                    ),
                )
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.subplots_adjust(
        top=0.90,
        bottom=0.20,
        left=0.08,
        right=0.86,
        wspace=0.24,
    )
    if image is not None:
        colorbar_axis = fig.add_axes((0.89, 0.23, 0.022, 0.60))
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("MARE sur l’activité (%)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_error_evolution(
    prediction_rows: Sequence[Mapping[str, Any]],
    split: str,
    output_path: Path,
) -> Path:
    """Plot absolute activity error at each test timepoint for every model."""
    selected_rows = [row for row in prediction_rows if row["split"] == split]
    if not selected_rows:
        raise ValueError(f"No prediction is available for {split}")

    panel_modes = (
        v3.MODE_TEW_FULL_GM,
        v3.MODE_TEW_FULL_AP_PA,
        v3.MODE_RAW_FULL_GM,
        v3.MODE_RAW_FULL_AP_PA,
    )
    timepoints = list(
        dict.fromkeys(str(row["timepoint"]) for row in selected_rows)
    )
    x = np.arange(len(timepoints), dtype=float)
    maximum_error = max(
        abs(float(row["activity_error_percent"])) for row in selected_rows
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.5, 10.2),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    legend_handles: Dict[str, Any] = {}
    for axis, mode in zip(axes.flat, panel_modes):
        mode_rows = [row for row in selected_rows if row["rate_mode"] == mode]
        for model in v3.MODEL_ORDER:
            model_rows = [row for row in mode_rows if row["model"] == model]
            if not model_rows:
                continue
            by_timepoint = {
                str(row["timepoint"]): row for row in model_rows
            }
            if any(timepoint not in by_timepoint for timepoint in timepoints):
                continue
            errors = [
                abs(float(by_timepoint[timepoint]["activity_error_percent"]))
                for timepoint in timepoints
            ]
            label = v3.MODEL_LABELS[model].split(" — ")[0]
            (line,) = axis.plot(
                x,
                errors,
                color=v3.MODEL_COLORS[model],
                marker=v3.MODEL_MARKERS[model],
                linewidth=2.0,
                markersize=7,
                label=label,
            )
            legend_handles[label] = line
        axis.set_title(
            urine.MODE_SHORT_LABELS[mode].replace("\n", " — "),
            fontsize=15,
        )
        axis.set_xticks(x, timepoints)
        axis.set_xlabel("Timepoint évalué")
        axis.set_ylabel("Erreur absolue relative (%)")
        axis.set_ylim(0.0, max(10.0, 1.10 * maximum_error))
        axis.grid(alpha=0.25, linestyle="--")
        axis.spines[["top", "right"]].set_visible(False)

    ordered_labels = [
        v3.MODEL_LABELS[model].split(" — ")[0]
        for model in v3.MODEL_ORDER
        if v3.MODEL_LABELS[model].split(" — ")[0] in legend_handles
    ]
    fig.legend(
        [legend_handles[label] for label in ordered_labels],
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(ordered_labels),
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        f"Évolution de l’erreur des modèles — transfert {split}",
        fontsize=19,
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "Chaque point = |activité prédite / activité urinaire − 1| × 100. "
        "La moyenne des trois points correspond à la MARE.",
        ha="center",
        fontsize=10.5,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.84), h_pad=2.4, w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_methodology() -> str:
    return """MODÈLES DE SENSIBILITÉ M0-M9 — CIBLE URINAIRE
=================================================

Position dans le projet
-----------------------
La calibration simple à J0 demeure l'analyse principale. Cette section teste
si des ratios spectraux planaires prédisent mieux la sensibilité effective.
Le CT et le Q/SPECT ne sont jamais utilisés.

Modèles
-------
M0 est conservé comme référence à sensibilité constante. M1, M2 et M4-M9
reprennent les définitions verrouillées de v3. M3 demeure supprimé puisqu'il
dupliquait M1. M5 est applicable uniquement lorsque le taux de comptage est
corrigé par TEW.

Interpatients
-------------
Les cibles principales par collectes terminées sont utilisées sur trois
points. Les deux directions P11→P8 et P8→P11 sont calculées sans réajustement
sur le patient cible. Avec trois points d'entraînement, M2 et M9 (deux
variables plus intercept) ne sont pas estimables et sont indiqués N/E.

Interprétation
--------------
Une MARE faible n'est convaincante que si elle est obtenue sur le patient non
utilisé pour l'ajustement. Les deux patients ne permettent pas d'établir une
généralisation; cette section demeure un test de transfert exploratoire.

Figures principales
-------------------
urine_target_model_predictions.png montre, pour P11→P8, le rapport entre
l'activité prédite et la cible urinaire à chacun des trois timepoints.
urine_target_six_main_modes_model_mare.png compare les MARE des modèles et
des principaux traitements planaires. modeles_M0_M9_interpatient_MARE.png
résume finalement les deux directions P11→P8 et P8→P11.
urine_target_model_error_vs_time*.png décompose la MARE en montrant l'erreur
absolue séparément à J0, J1 et J2 dans chaque direction de transfert.
"""


def build_readme() -> str:
    return """SECTION 2 — MODÈLES DE SENSIBILITÉ
===================================

1_interpatient
  Transferts stricts P11→P8 et P8→P11 sur trois points.

2_comparaison
  Figures principales de performance:
  - prédictions P11→P8 pour chaque modèle et timepoint;
  - MARE P11→P8 par modèle pour les quatre modes pleine image;
  - évolution de l'erreur à J0, J1 et J2 dans les deux directions;
  - matrice MARE résumant P11→P8 et P8→P11.

3_methodologie
  Définitions, séparation entraînement/test et limites.

4_annexes_ancienne_analyse
  Ancienne analyse exploratoire conservée pour traçabilité.
"""


def run_model_section(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    inter_dir = output_dir / "1_interpatient"
    comparison_dir = output_dir / "2_comparaison"
    methodology_dir = output_dir / "3_methodologie"
    for directory in (output_dir, inter_dir, comparison_dir, methodology_dir):
        directory.mkdir(parents=True, exist_ok=True)

    p11_three = urine.build_urine_dataset(urine.TRAINING_DATASET, 3)
    p8_three = urine.build_urine_dataset(urine.TEST_DATASET, 3)
    inter_metrics, inter_predictions = interpatient_results(p11_three, p8_three)
    p11_to_p8_metrics = [
        row for row in inter_metrics if row["split"] == "P11→P8"
    ]
    p11_to_p8_predictions = [
        row for row in inter_predictions if row["split"] == "P11→P8"
    ]

    paths = {
        "readme": output_dir / "README.txt",
        "interpatient_metrics": _write_csv(
            inter_dir / "modeles_M0_M9_metriques.csv", inter_metrics
        ),
        "interpatient_predictions": _write_csv(
            inter_dir / "modeles_M0_M9_predictions.csv", inter_predictions
        ),
        "interpatient_figure": _plot_metric_heatmaps(
            inter_metrics,
            ("P11→P8", "P8→P11"),
            "Modèles de sensibilité — transferts interpatients",
            comparison_dir / "modeles_M0_M9_interpatient_MARE.png",
        ),
        "p11_to_p8_prediction_figure": urine.plot_primary_predictions(
            p11_to_p8_predictions,
            comparison_dir / "urine_target_model_predictions.png",
        ),
        "p11_to_p8_six_mode_mare_figure": (
            urine.plot_urine_six_mode_primary_mare(
                p11_to_p8_metrics,
                comparison_dir / "urine_target_six_main_modes_model_mare.png",
                "Cible principale par collectes terminées — P11 3 → P8 3",
            )
        ),
        "p11_to_p8_error_evolution_figure": plot_model_error_evolution(
            inter_predictions,
            "P11→P8",
            comparison_dir / "urine_target_model_error_vs_time.png",
        ),
        "p8_to_p11_error_evolution_figure": plot_model_error_evolution(
            inter_predictions,
            "P8→P11",
            comparison_dir / "urine_target_model_error_vs_time_P8_to_P11.png",
        ),
    }
    paths["readme"].write_text(build_readme(), encoding="utf-8")
    methodology_path = methodology_dir / "modeles_M0_M9_methodologie.txt"
    methodology_path.write_text(build_methodology(), encoding="utf-8")
    paths["methodology"] = methodology_path
    return {
        "interpatient_metrics": inter_metrics,
        "paths": paths,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_model_section(args.output_dir)
    print("Section des modèles de sensibilité terminée")
    for name, path in result["paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
