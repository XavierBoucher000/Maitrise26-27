"""Generalized urine-reference planar sensitivity analyses.

The runner uses the canonical DICOM folders ``Data/p8``, ``Data/p11`` and
``Data/p17``.  A directory whose name ends in ``copy`` is never discovered as
an imaging source.  The three sections deliberately answer different
questions:

1. one-point J0 calibration;
2. descriptive zero-origin sensitivity fitted on J0--J2;
3. M0--M9 models evaluated with leave-one-patient-out validation.

All targets are urine-derived remaining body activities evaluated at the
planar acquisition time.  No Q/SPECT activity, CT or ACCT is used to define
the target.  The planar counts are evaluated on two supports with the same
analysis engine: the complete planar image and the longitudinal crop matched
to the Q/SPECT coverage.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figure_layout
import patient_sensitivity_multicycle_v3 as v3
import patient_sensitivity_qspect_slopes as slope_engine
import patient_sensitivity_urine_target as urine
import planar_total_counts_retention as retention


DEFAULT_OUTPUT_DIR = figure_layout.analysis_dir(
    "patient_sensitivity_urine_target.py"
)
DATASET_SPECS = urine.URINE_DATASETS

J0_METHODS = (
    urine.SIMPLE_PHOTO_AP_PA,
    urine.SIMPLE_PHOTO_GM_PIXEL,
    v3.MODE_TEW_FULL_AP_PA,
    v3.MODE_TEW_FULL_GM,
)
SLOPE_METHODS = urine.SIMPLE_METHOD_ORDER
METHOD_LABELS = {
    **urine.SIMPLE_METHOD_LABELS,
    v3.MODE_TEW_FULL_AP_PA: "TEW — AP+PA",
    v3.MODE_TEW_FULL_GM: "TEW — GM pixel",
}

ENERGY_WINDOW_FIGURES = (
    ("lower_scatter_counts", "Fenêtre de diffusé inférieure", "01_lower_scatter.png"),
    ("photopeak_counts", "Photopeak", "02_photopeak.png"),
    ("upper_scatter_counts", "Fenêtre de diffusé supérieure", "03_upper_scatter.png"),
    ("general_scatter_counts", "Diffusé général (basse énergie)", "04_general_scatter.png"),
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _rates(dataset: Mapping[str, Any], method: str) -> np.ndarray:
    if method in dataset["simple_rates"]:
        return np.asarray(dataset["simple_rates"][method], dtype=float)
    return np.asarray(
        dataset["mode_observations"][method]["measurement_rate"], dtype=float
    )


def _activity_metrics(
    reference: Sequence[float], predicted: Sequence[float], start: int = 0
) -> Dict[str, float]:
    reference_array = np.asarray(reference, dtype=float)[start:]
    predicted_array = np.asarray(predicted, dtype=float)[start:]
    errors = 100.0 * (predicted_array / reference_array - 1.0)
    return {
        "n_evaluated": int(errors.size),
        "bias_percent": float(np.mean(errors)),
        "mare_percent": float(np.mean(np.abs(errors))),
        "rmse_percent": float(np.sqrt(np.mean(errors**2))),
        "maximum_absolute_error_percent": float(np.max(np.abs(errors))),
    }


def load_datasets(timepoint_count: int = 3) -> Dict[str, Dict[str, Any]]:
    """Load every configured urine dataset from its canonical patient folder."""
    datasets: Dict[str, Dict[str, Any]] = {}
    ordered_specs = sorted(
        DATASET_SPECS,
        key=lambda spec: figure_layout.patient_cycle_sort_key(
            spec.patient_id, spec.cycle_id
        ),
    )
    for spec in ordered_specs:
        if " copy" in str(spec.planar_dir.parent.name).lower():
            raise ValueError(f"Copy imaging directory is not allowed: {spec.planar_dir}")
        datasets[spec.patient_id] = urine.build_urine_dataset(
            spec, timepoint_count
        )
    return datasets


def prepare_urine_reference_rows(
    datasets: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Joindre la cible urinaire au meme tableau planaire que l'analyse Q/SPECT.

    Le champ commun ``reference_activity_mbq`` est la seule difference donnee
    au moteur de pente.  Les comptes, durees, corrections de temps mort,
    fenetres et combinaisons de detecteurs sont donc strictement identiques.
    """
    references: dict[tuple[str, str, str], float] = {}
    for dataset in datasets.values():
        spec = dataset["spec"]
        activity = np.asarray(dataset["reference_activity"], dtype=float)
        for record, value in zip(dataset["records"], activity):
            planar_datetime = record["planar_datetime"].strftime("%Y-%m-%d %H:%M:%S")
            references[(spec.patient_id, spec.cycle_id, planar_datetime)] = float(value)

    prepared: list[dict[str, Any]] = []
    for source in slope_engine.load_master_table(slope_engine.DEFAULT_MASTER_TABLE):
        key = (
            str(source["patient_id"]),
            str(source["cycle_id"]),
            str(source["planar_datetime"]),
        )
        if key not in references:
            continue
        row = dict(source)
        row["reference_activity_mbq"] = references[key]
        prepared.append(row)
    if not prepared:
        raise ValueError("Aucune cible urinaire ne correspond a la table planaire")
    return prepared


def _remove_legacy_slope_artifacts(section_slope: Path) -> None:
    """Retirer seulement les anciennes sorties remplacees par le moteur commun."""
    legacy = [
        section_slope / "pentes_J0_J2.csv",
        section_slope / "predictions_intrapatient.csv",
        section_slope / "transferts_interpatient_metriques.csv",
        section_slope / "transferts_interpatient_MARE.png",
    ]
    for folder in ("AP_PA", "GM", "GM_globale"):
        legacy.extend(
            [
                section_slope / folder / "construction_pente_J0_J2_photopeak.png",
                section_slope / folder / "pentes_J0_J2_par_patient.png",
                section_slope / folder / "04_general_scatter.png",
            ]
        )
    for path in legacy:
        if path.is_file():
            path.unlink()


def _remove_legacy_j0_artifacts(section_j0: Path) -> None:
    """Retirer les anciennes sorties J0 qui repondaient a d'autres questions."""
    legacy_directories = (
        section_j0 / "5_ancienne_analyse_p11_p8",
        section_j0 / "1_activites_reference",
    )
    legacy_files = (
        section_j0 / "predictions_intrapatient.csv",
        section_j0 / "sensibilites_J0_par_patient.png",
        section_j0 / "transferts_interpatient_MARE.png",
        section_j0 / "transferts_interpatient_metriques.csv",
    )
    for directory in legacy_directories:
        if directory.is_dir():
            shutil.rmtree(directory)
    for path in legacy_files:
        if path.is_file():
            path.unlink()


def _pair_directions(patient_ids: Sequence[str]) -> list[tuple[str, str]]:
    return [
        (source, target)
        for source in patient_ids
        for target in patient_ids
        if source != target
    ]


def j0_results(
    datasets: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    sensitivities: list[Dict[str, Any]] = []
    intrapatient: list[Dict[str, Any]] = []
    transfer: list[Dict[str, Any]] = []
    for patient_id, dataset in datasets.items():
        reference = np.asarray(dataset["reference_activity"], dtype=float)
        for method in J0_METHODS:
            rates = _rates(dataset, method)
            sensitivity = float(rates[0] / reference[0])
            predicted = rates / sensitivity
            sensitivities.append(
                {
                    "patient_id": patient_id,
                    "dataset_key": dataset["spec"].key,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "sensitivity_j0_cps_per_mbq": sensitivity,
                }
            )
            for index, record in enumerate(dataset["records"]):
                intrapatient.append(
                    {
                        "patient_id": patient_id,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "timepoint": record["timepoint"],
                        "hours_after_injection": record["hours_after_injection"],
                        "used_for_calibration": index == 0,
                        "reference_activity_mbq": float(reference[index]),
                        "predicted_activity_mbq": float(predicted[index]),
                        "absolute_error_percent": float(
                            abs(100.0 * (predicted[index] / reference[index] - 1.0))
                        ),
                    }
                )

    sensitivity_map = {
        (row["patient_id"], row["method"]): float(
            row["sensitivity_j0_cps_per_mbq"]
        )
        for row in sensitivities
    }
    for source, target in _pair_directions(tuple(datasets)):
        target_dataset = datasets[target]
        reference = np.asarray(target_dataset["reference_activity"], dtype=float)
        for method in J0_METHODS:
            predicted = _rates(target_dataset, method) / sensitivity_map[(source, method)]
            metric = _activity_metrics(reference, predicted, start=1)
            transfer.append(
                {
                    "source_patient": source,
                    "test_patient": target,
                    "direction": f"{source}_to_{target}",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    **metric,
                }
            )
    return sensitivities, intrapatient, transfer


def zero_origin_slope(activity: Sequence[float], rates: Sequence[float]) -> float:
    activity_array = np.asarray(activity, dtype=float)
    rate_array = np.asarray(rates, dtype=float)
    denominator = float(np.dot(activity_array, activity_array))
    if denominator <= 0.0:
        raise ValueError("Cannot fit a zero-origin sensitivity")
    return float(np.dot(activity_array, rate_array) / denominator)


def _raw_count_series(
    count_results: Mapping[tuple[str, str], Mapping[str, Any]],
    patient_id: str,
    method: str,
) -> np.ndarray:
    if method.endswith("_ap_pa"):
        combination = retention.COMBINATION_AP_PA
    elif method.endswith("_gm_pixel"):
        combination = retention.COMBINATION_GM
    else:
        raise ValueError(f"Unsupported raw-count slope method: {method}")
    field_by_method = {
        urine.SIMPLE_PHOTO_AP_PA: "photopeak_counts",
        urine.SIMPLE_PHOTO_GM_PIXEL: "photopeak_counts",
        urine.SIMPLE_GENERAL_AP_PA: "general_scatter_counts",
        urine.SIMPLE_GENERAL_GM_PIXEL: "general_scatter_counts",
        urine.SIMPLE_ALL_AP_PA: "all_four_windows_counts",
        urine.SIMPLE_ALL_GM_PIXEL: "all_four_windows_counts",
    }
    result = count_results[(patient_id, combination)]
    rows = result["rows"][: result["comparison_timepoints"]]
    return np.asarray([row[field_by_method[method]] for row in rows], dtype=float)


def slope_results(
    datasets: Mapping[str, Mapping[str, Any]],
    count_results: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    slopes: list[Dict[str, Any]] = []
    intrapatient: list[Dict[str, Any]] = []
    transfer: list[Dict[str, Any]] = []
    for patient_id, dataset in datasets.items():
        reference = np.asarray(dataset["reference_activity"], dtype=float)
        for method in SLOPE_METHODS:
            counts = _raw_count_series(count_results, patient_id, method)
            slope = zero_origin_slope(reference, counts)
            predicted = counts / slope
            metric = _activity_metrics(reference, predicted)
            slopes.append(
                {
                    "patient_id": patient_id,
                    "dataset_key": dataset["spec"].key,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "slope_counts_per_mbq": slope,
                    **metric,
                }
            )
            for index, record in enumerate(dataset["records"]):
                intrapatient.append(
                    {
                        "patient_id": patient_id,
                        "method": method,
                        "timepoint": record["timepoint"],
                        "reference_activity_mbq": float(reference[index]),
                        "predicted_activity_mbq": float(predicted[index]),
                        "absolute_error_percent": float(
                            abs(100.0 * (predicted[index] / reference[index] - 1.0))
                        ),
                        "used_for_fit": True,
                    }
                )

    slope_map = {
        (row["patient_id"], row["method"]): float(row["slope_counts_per_mbq"])
        for row in slopes
    }
    for source, target in _pair_directions(tuple(datasets)):
        target_dataset = datasets[target]
        reference = np.asarray(target_dataset["reference_activity"], dtype=float)
        for method in SLOPE_METHODS:
            target_counts = _raw_count_series(count_results, target, method)
            predicted = target_counts / slope_map[(source, method)]
            transfer.append(
                {
                    "source_patient": source,
                    "test_patient": target,
                    "direction": f"{source}_to_{target}",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    **_activity_metrics(reference, predicted),
                }
            )
    return slopes, intrapatient, transfer


def _combine_observations(
    datasets: Sequence[Mapping[str, Any]], mode: str
) -> Dict[str, Any]:
    observations = [dataset["mode_observations"][mode] for dataset in datasets]
    return {
        "records": [record for item in observations for record in item["records"]],
        "features": {
            name: np.concatenate(
                [np.asarray(item["features"][name], dtype=float) for item in observations]
            )
            for name in v3.FEATURE_ORDER
        },
        "measurement_rate": np.concatenate(
            [np.asarray(item["measurement_rate"], dtype=float) for item in observations]
        ),
        "reference_activity": np.concatenate(
            [np.asarray(item["reference_activity"], dtype=float) for item in observations]
        ),
        "target_sensitivity": np.concatenate(
            [np.asarray(item["target_sensitivity"], dtype=float) for item in observations]
        ),
    }


def model_results(
    datasets: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Run the shared Q/SPECT/urine model engine with urine targets."""
    metrics: list[Dict[str, Any]] = []
    predictions: list[Dict[str, Any]] = []
    patient_ids = tuple(datasets)
    for mode in urine.MODE_ORDER:
        observations = {
            patient: datasets[patient]["mode_observations"][mode]
            for patient in patient_ids
        }
        for test_patient in patient_ids:
            training_patients = tuple(
                patient for patient in patient_ids if patient != test_patient
            )
            result = v3.fit_and_evaluate(
                training_patients,
                (test_patient,),
                observations,
                split_id=f"patient_holdout_{test_patient}",
                split_type="leave_one_patient_out",
                split_label=f"{test_patient.upper()} entièrement exclu",
                rate_mode=mode,
                model_names=urine._mode_model_names(mode),
                include_j0_personalized=True,
                reference_name="activité corporelle dérivée des urines",
            )
            for row in result["metrics"]:
                metrics.append(
                    {
                        **row,
                        "test_patient": test_patient,
                        "training_patients": "+".join(training_patients),
                        "mare_percent": row.get("activity_mare_percent", math.nan),
                        "bias_percent": row.get("activity_bias_percent", math.nan),
                        "rmse_percent": row.get("activity_rmse_percent", math.nan),
                        "maximum_absolute_error_percent": row.get(
                            "maximum_activity_error_percent", math.nan
                        ),
                    }
                )
            for row in result["predictions"]:
                predictions.append(
                    {
                        **row,
                        "test_patient": test_patient,
                        "training_patients": "+".join(training_patients),
                        "error_percent": row["activity_error_percent"],
                    }
                )
    return metrics, predictions


def plot_grouped_sensitivities(
    rows: Sequence[Mapping[str, Any]],
    value_field: str,
    title: str,
    path: Path,
    ylabel: str = "Sensibilité (cps/MBq)",
) -> Path:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    patients = list(dict.fromkeys(str(row["patient_id"]) for row in rows))
    if len(methods) == 1:
        method = methods[0]
        values = [
            float(
                next(
                    row[value_field]
                    for row in rows
                    if row["patient_id"] == patient and row["method"] == method
                )
            )
            for patient in patients
        ]
        colors = {"p11": "#1f77b4", "p8": "#ff7f0e", "p17": "#2ca02c"}
        fig, axis = plt.subplots(figsize=(9.2, 5.8))
        bars = axis.bar(
            np.arange(len(patients), dtype=float),
            values,
            width=0.65,
            color=[colors.get(patient, "#777777") for patient in patients],
        )
        axis.set_xticks(
            np.arange(len(patients), dtype=float),
            [patient.upper() for patient in patients],
        )
        axis.bar_label(bars, fmt="%.1f", padding=4, fontsize=11)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25, linestyle="--")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_ylim(0.0, max(values) * 1.15)
        fig.tight_layout()
        return _save(fig, path)

    x = np.arange(len(methods), dtype=float)
    width = 0.78 / len(patients)
    fig, axis = plt.subplots(figsize=(13.2, 6.2))
    for index, patient in enumerate(patients):
        values = [
            float(
                next(
                    row[value_field]
                    for row in rows
                    if row["patient_id"] == patient and row["method"] == method
                )
            )
            for method in methods
        ]
        axis.bar(
            x + (index - (len(patients) - 1) / 2) * width,
            values,
            width,
            label=patient.upper(),
        )
    axis.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=18, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25, linestyle="--")
    axis.legend(frameon=False, ncol=len(patients))
    fig.tight_layout()
    return _save(fig, path)


def plot_j0_j2_photopeak_fits(
    datasets: Mapping[str, Mapping[str, Any]],
    count_results: Mapping[tuple[str, str], Mapping[str, Any]],
    combination: str,
    combination_label: str,
    path: Path,
) -> Path:
    """Show how each J0--J2 photopeak slope is obtained for one combination.

    The displayed straight line is the same zero-origin fit used by
    :func:`slope_results`.  Counts are shown in millions and activity in GBq
    for readability, while the annotated slope keeps the analysis unit
    counts/MBq.
    """
    patient_ids = tuple(datasets)
    fig, axes = plt.subplots(
        1,
        len(patient_ids),
        figsize=(16.2, 5.0),
        squeeze=False,
        layout="constrained",
    )
    patient_colors = {"p11": "#1f77b4", "p8": "#d62728", "p17": "#2ca02c"}

    for axis, patient_id in zip(axes[0], patient_ids):
        dataset = datasets[patient_id]
        activity_mbq = np.asarray(dataset["reference_activity"], dtype=float)
        activity_gbq = activity_mbq / 1000.0
        labels = [str(record["timepoint"]) for record in dataset["records"]]
        result = count_results[(patient_id, combination)]
        count_rows = list(result["rows"][: result["comparison_timepoints"]])
        counts = np.asarray(
            [row["photopeak_counts"] for row in count_rows], dtype=float
        )
        fit = retention.through_origin_summary(activity_mbq, counts)
        slope = float(fit["slope_counts_per_mbq"])
        r_squared = float(fit["r_squared_zero_origin"])
        color = patient_colors.get(patient_id, "#444444")

        axis.scatter(
            activity_gbq,
            counts / 1e6,
            s=70,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        axis.scatter([0.0], [0.0], s=58, marker="X", color="#222222", zorder=4)
        for label, x_value, y_value in zip(labels, activity_gbq, counts / 1e6):
            axis.annotate(
                label,
                (x_value, y_value),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=10,
            )
        x_line_gbq = np.linspace(0.0, max(activity_gbq) * 1.06, 150)
        y_line_million = slope * (x_line_gbq * 1000.0) / 1e6
        axis.plot(x_line_gbq, y_line_million, color=color, linewidth=2.0)
        axis.text(
            0.96,
            0.95,
            r"$C=S\,A$" "\n" f"S = {slope:.1f} comptes/MBq\n" f"R² = {r_squared:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        axis.set_title(patient_id.upper(), fontsize=12)
        axis.set_xlabel("Activité corporelle estimée par bilan urinaire (GBq)")
        axis.set_ylabel("Comptes photopeak totaux (millions)")
        axis.invert_xaxis()
        axis.grid(True, linestyle="--", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.suptitle(
        f"Construction de la pente comptes–activité — Photopeak — {combination_label}",
        fontsize=16,
        fontweight="bold",
    )
    return _save(fig, path)


def plot_j0_j2_single_window_fits(
    datasets: Mapping[str, Mapping[str, Any]],
    count_results: Mapping[tuple[str, str], Mapping[str, Any]],
    combination: str,
    count_field: str,
    window_label: str,
    path: Path,
) -> Path:
    """Plot one energy window for all three patients and its J0--J2 fit."""
    patient_ids = tuple(datasets)
    combination_label = {
        retention.COMBINATION_AP_PA: "AP+PA",
        retention.COMBINATION_GM: "GM pixel",
        retention.COMBINATION_GM_GLOBAL: "GM globale",
    }[combination]
    patient_colors = {"p11": "#1f77b4", "p8": "#d62728", "p17": "#2ca02c"}
    fig, axes = plt.subplots(
        1,
        len(patient_ids),
        figsize=(16.2, 5.0),
        squeeze=False,
        layout="constrained",
    )

    for axis, patient_id in zip(axes[0], patient_ids):
        dataset = datasets[patient_id]
        result = count_results[(patient_id, combination)]
        rows = list(result["rows"][: result["comparison_timepoints"]])
        activity_mbq = np.asarray(dataset["reference_activity"], dtype=float)
        activity_gbq = activity_mbq / 1000.0
        counts = np.asarray([row[count_field] for row in rows], dtype=float)
        labels = [str(row["timepoint"]) for row in rows]
        fit = retention.through_origin_summary(activity_mbq, counts)
        slope = float(fit["slope_counts_per_mbq"])
        r_squared = float(fit["r_squared_zero_origin"])
        color = patient_colors.get(patient_id, "#444444")

        axis.scatter(
            activity_gbq,
            counts / 1e6,
            s=74,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        axis.scatter([0.0], [0.0], s=58, marker="X", color="#222222", zorder=4)
        for label, x_value, y_value in zip(labels, activity_gbq, counts / 1e6):
            axis.annotate(
                label,
                (x_value, y_value),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=10,
            )

        x_line_gbq = np.linspace(0.0, max(activity_gbq) * 1.06, 150)
        y_line_million = slope * (x_line_gbq * 1000.0) / 1e6
        axis.plot(x_line_gbq, y_line_million, color=color, linewidth=2.0)
        axis.text(
            0.96,
            0.95,
            f"S = {slope:.1f} comptes/MBq\nR² = {r_squared:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        axis.set_title(patient_id.upper(), fontsize=13, fontweight="bold")
        axis.set_xlabel("Activité corporelle estimée par bilan urinaire (GBq)")
        axis.set_ylabel("Comptes totaux (millions)")
        axis.invert_xaxis()
        axis.grid(True, linestyle="--", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.suptitle(
        f"{window_label} — {combination_label} — ajustement J0 à J2",
        fontsize=16,
        fontweight="bold",
    )
    return _save(fig, path)


def energy_window_slope_rows(
    datasets: Mapping[str, Mapping[str, Any]],
    count_results: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Return the J0--J2 zero-origin slope for each individual energy window."""
    rows: list[Dict[str, Any]] = []
    for combination in retention.COMBINATIONS:
        for patient_id, dataset in datasets.items():
            reference = np.asarray(dataset["reference_activity"], dtype=float)
            result = count_results[(patient_id, combination)]
            count_rows = list(result["rows"][: result["comparison_timepoints"]])
            for count_field, window_label, _ in ENERGY_WINDOW_FIGURES:
                counts = np.asarray(
                    [row[count_field] for row in count_rows], dtype=float
                )
                fit = retention.through_origin_summary(reference, counts)
                slope = float(fit["slope_counts_per_mbq"])
                predicted_counts = slope * reference
                residuals = counts - predicted_counts
                degrees_of_freedom = max(int(reference.size) - 1, 1)
                residual_variance = float(
                    np.sum(residuals**2) / degrees_of_freedom
                )
                slope_standard_error = float(
                    np.sqrt(residual_variance / np.sum(reference**2))
                )
                predicted_activity = counts / slope
                activity_errors_percent = 100.0 * (
                    predicted_activity / reference - 1.0
                )
                rows.append(
                    {
                        "patient_id": patient_id,
                        "combination": combination,
                        "window": count_field,
                        "window_label": window_label,
                        "slope_counts_per_mbq": slope,
                        "slope_standard_error_counts_per_mbq": (
                            slope_standard_error
                        ),
                        "r_squared_zero_origin": float(
                            fit["r_squared_zero_origin"]
                        ),
                        "mare_activity_percent": float(
                            np.mean(np.abs(activity_errors_percent))
                        ),
                    }
                )
    return rows


def plot_energy_window_slopes(
    rows: Sequence[Mapping[str, Any]],
    combination: str,
    path: Path,
) -> Path:
    """Plot patient slopes in four panels, one per individual energy window."""
    selected = [row for row in rows if row["combination"] == combination]
    patients = list(dict.fromkeys(str(row["patient_id"]) for row in selected))
    colors = {"p11": "#1f77b4", "p8": "#ff7f0e", "p17": "#2ca02c"}
    combination_label = {
        retention.COMBINATION_AP_PA: "AP+PA",
        retention.COMBINATION_GM: "GM pixel",
        retention.COMBINATION_GM_GLOBAL: "GM globale",
    }[combination]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4))
    for axis, (count_field, window_label, _) in zip(
        axes.flat, ENERGY_WINDOW_FIGURES
    ):
        values = [
            float(
                next(
                    row["slope_counts_per_mbq"]
                    for row in selected
                    if row["patient_id"] == patient
                    and row["window"] == count_field
                )
            )
            for patient in patients
        ]
        standard_errors = [
            float(
                next(
                    row["slope_standard_error_counts_per_mbq"]
                    for row in selected
                    if row["patient_id"] == patient
                    and row["window"] == count_field
                )
            )
            for patient in patients
        ]
        r_squared = [
            float(
                next(
                    row["r_squared_zero_origin"]
                    for row in selected
                    if row["patient_id"] == patient
                    and row["window"] == count_field
                )
            )
            for patient in patients
        ]
        mare = [
            float(
                next(
                    row["mare_activity_percent"]
                    for row in selected
                    if row["patient_id"] == patient
                    and row["window"] == count_field
                )
            )
            for patient in patients
        ]
        bars = axis.bar(
            np.arange(len(patients), dtype=float),
            values,
            width=0.65,
            color=[colors.get(patient, "#777777") for patient in patients],
            yerr=standard_errors,
            capsize=4,
            error_kw={"ecolor": "#222222", "elinewidth": 1.2, "capthick": 1.2},
        )
        axis.set_xticks(
            np.arange(len(patients), dtype=float),
            [patient.upper() for patient in patients],
        )
        label_offset = 0.025 * max(values)
        for bar, value, standard_error, r2_value, mare_value in zip(
            bars, values, standard_errors, r_squared, mare
        ):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + standard_error + label_offset,
                f"{value:.1f}\nR²={r2_value:.3f} | MARE={mare_value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        axis.set_title(window_label)
        axis.set_ylabel("Pente (comptes/MBq)")
        axis.set_ylim(
            0.0,
            max(value + error for value, error in zip(values, standard_errors))
            * 1.30,
        )
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        f"Pentes des quatre fenêtres énergétiques — {combination_label}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 0.955), h_pad=2.2, w_pad=1.4)
    fig.text(
        0.5,
        0.018,
        "Moustache : ±1 erreur-type de la pente. R² et MARE sont descriptifs (J0–J2 utilisés pour l’ajustement).",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    return _save(fig, path)


def plot_transfer_heatmaps(
    rows: Sequence[Mapping[str, Any]], methods: Sequence[str], title: str, path: Path
) -> Path:
    directions = list(dict.fromkeys(str(row["direction"]) for row in rows))
    column_count = min(3, len(methods))
    row_count = int(math.ceil(len(methods) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.1 * column_count, 4.7 * row_count + 0.7),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, method in zip(flat_axes, methods):
        values = np.asarray(
            [
                float(
                    next(
                        row["mare_percent"]
                        for row in rows
                        if row["direction"] == direction and row["method"] == method
                    )
                )
                for direction in directions
            ]
        )[:, None]
        image = axis.imshow(values, cmap="YlOrRd", vmin=0.0, aspect="auto")
        for row_index, value in enumerate(values[:, 0]):
            axis.text(0, row_index, f"{value:.1f}", ha="center", va="center")
        axis.set_xticks([0], ["MARE (%)"])
        axis.set_yticks(
            range(len(directions)),
            [direction.replace("_to_", "→").upper() for direction in directions],
        )
        axis.set_title(METHOD_LABELS[method], fontsize=11)
        fig.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    for axis in flat_axes[len(methods) :]:
        axis.set_visible(False)
    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def plot_model_mare(
    rows: Sequence[Mapping[str, Any]], mode: str, path: Path
) -> Path:
    patients = list(dict.fromkeys(str(row["test_patient"]) for row in rows))
    models = [model for model in v3.MODEL_ORDER if model in urine._mode_model_names(mode)]
    matrix = np.full((len(patients), len(models)), np.nan)
    for row in rows:
        if (
            row["rate_mode"] != mode
            or row["fit_status"] != "fitted"
            or row.get("metric_scope") != "combined_test"
        ):
            continue
        matrix[patients.index(str(row["test_patient"])), models.index(str(row["model"]))] = float(row["mare_percent"])
    fig, axis = plt.subplots(figsize=(12.5, 4.8))
    image = axis.imshow(matrix, cmap="YlOrRd", vmin=0.0, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            axis.text(j, i, "--" if not np.isfinite(value) else f"{value:.1f}", ha="center", va="center")
    axis.set_xticks(range(len(models)), [model.split("_", 1)[0] for model in models])
    axis.set_yticks(range(len(patients)), [patient.upper() for patient in patients])
    axis.set_xlabel("Modèle")
    axis.set_ylabel("Patient entièrement exclu")
    axis.set_title(f"Modèles urinaires leave-one-patient-out — {urine.MODE_SHORT_LABELS[mode].replace(chr(10), ' ')}")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("MARE sur l’activité (%)")
    fig.tight_layout()
    return _save(fig, path)


def _generate_reference_count_figures(
    output_dir: Path,
) -> tuple[Dict[str, Path], Dict[tuple[str, str], Dict[str, Any]]]:
    paths: Dict[str, Path] = {}
    results: Dict[tuple[str, str], Dict[str, Any]] = {}
    for spec in DATASET_SPECS:
        configured = retention.DATASETS[(spec.patient_id, spec.cycle_id)]
        for combination in retention.COMBINATIONS:
            target = (
                output_dir
                / spec.patient_id
                / spec.cycle_id
                / combination
            )
            result = retention.run_analysis(
                planar_dir=spec.planar_dir,
                urine_csv=spec.urine_csv,
                output_dir=target,
                comparison_timepoints=3,
                patient_label=str(configured["patient_label"]),
                cycle_label=str(configured["cycle_label"]),
                combination=combination,
            )
            paths[f"{spec.patient_id}_{combination}"] = result["paths"][
                "counts_vs_retention_figure"
            ]
            results[(spec.patient_id, combination)] = result
    return paths, results


def build_readme() -> str:
    return """ANALYSES PLANAIRES AVEC ACTIVITE DE REFERENCE URINAIRE
========================================================

Sources
-------
Les DICOM proviennent exclusivement de Data/p8, Data/p11 et Data/p17.
Les dossiers nommes 'copy' ne sont pas utilises comme source d'imagerie.

1_sensibilite_J0
----------------
Le meme moteur planaire que la section J0-J2 calcule S_J0 = R_J0/A_J0. La
droite est donc definie uniquement par (0,0) et J0. Les acquisitions suivantes
ne participent jamais a la calibration et servent seulement a mesurer l'erreur
sur l'activite. Pour chacun des supports crop et complet, les sorties AP+PA,
GM par pixel et GM globale contiennent les quatre fenetres, la synthese de
l'erreur post-J0, les sensibilites J0 par cycle et leur tableau. Une calibration
sur un seul point ne permet pas d'estimer une erreur-type de pente.

2_sensibilite_J0_J2
-------------------
Le meme moteur que pour la cible Q/SPECT ajuste R = S*A a l'origine sur les
trois premiers points. Les comptes sont divises par la duree pour obtenir des
cps, puis corriges par le facteur global de temps mort. Les noms de fichiers,
fenetres, supports crop/complet et figures sont identiques entre les cibles
urinaire et Q/SPECT. Seule la colonne d'activite de reference change. La
performance intracycle sur les memes trois points est descriptive; les
transferts cycle vers cycle sont sans reajustement.
P11-juillet n'est pas inclus dans cette section, car les seules collectes de
P11 sont rattachees a l'injection du 19 mai 2026. Aucune cible urinaire n'est
disponible pour l'injection de juin/juillet.

3_sensibilite_modeles
---------------------
Pour chaque patient test, M0, M0b et les modeles M1-M9 sont ajustes sur les
deux autres patients puis verrouilles. M0 est la moyenne des sensibilites;
M0b est la pente a l'origine R=S*A sur l'entrainement. La reference J0 utilise
le premier point du cycle teste et est evaluee seulement sur les points
suivants. Une figure dediee compare toutes les methodes sur ce meme sous-
ensemble post-J0. Il s'agit d'une validation leave-one-patient-out pour les
modeles appris, mais pas pour la reference personnalisee J0.

Aucun CT, ACCT ni activite Q/SPECT n'est utilise pour definir la cible
urinaire. Le dossier crop utilise toutefois le Q/SPECT pour definir la
couverture longitudinale; il constitue une analyse de developpement et non un
pipeline final autonome sans Q/SPECT. Pour la retention pancorporelle urinaire,
le support complet demeure la comparaison physiquement la plus directe.
"""


def run_generalized_urine_analysis(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    section_j0 = output_dir / "1_sensibilite_J0"
    section_slope = output_dir / "2_sensibilite_J0_J2"
    section_models = output_dir / "3_sensibilite_modeles"
    for directory in (section_j0, section_slope, section_models):
        directory.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(3)
    urine_slope_rows = prepare_urine_reference_rows(datasets)

    _remove_legacy_j0_artifacts(section_j0)
    _remove_legacy_slope_artifacts(section_slope)
    sensitivity_outputs = slope_engine.run_reference_sensitivity_sections(
        urine_slope_rows,
        output_dir,
        reference_label="urinaire",
        reference_description=(
            "l'activite corporelle restante derivee des collectes urine/couches "
            "au moment du planaire"
        ),
    )
    j0_outputs = {
        name: path
        for name, path in sensitivity_outputs.items()
        if name.startswith("j0_")
    }
    j0_figures = [
        path for path in j0_outputs.values() if Path(path).suffix.lower() == ".png"
    ]
    slope_outputs = {
        name: path
        for name, path in sensitivity_outputs.items()
        if name.startswith("slope_")
    }
    slope_figures = [
        path for path in slope_outputs.values() if Path(path).suffix.lower() == ".png"
    ]

    model_metrics, model_predictions = model_results(datasets)
    _write_csv(section_models / "modeles_M0_M9_metriques.csv", model_metrics)
    _write_csv(section_models / "modeles_M0_M9_predictions.csv", model_predictions)
    _write_csv(section_models / "modeles_sensibilite_metriques.csv", model_metrics)
    _write_csv(
        section_models / "modeles_sensibilite_predictions.csv", model_predictions
    )
    methodology_path = section_models / "modeles_methodologie_commune.txt"
    methodology_path.write_text(
        v3.build_model_methodology_document(
            reference_symbol="A_urine",
            reference_label="bilan urinaire",
        ),
        encoding="utf-8",
    )
    model_figures = [
        plot_model_mare(
            model_metrics,
            mode,
            section_models / f"modeles_M0_M9_MARE_{mode}.png",
        )
        for mode in urine.MODE_ORDER
    ]
    post_j0_dir = section_models / "post_J0_comparison"
    model_figures.extend(
        v3.plot_post_j0_model_mare(
            model_metrics,
            mode,
            post_j0_dir / f"post_J0_MARE_{mode}.png",
            model_names=urine._mode_model_names(mode),
            title_prefix="Cible urine — comparaison après J0",
        )
        for mode in urine.MODE_ORDER
    )

    readme = output_dir / "README.txt"
    readme.write_text(build_readme(), encoding="utf-8")
    return {
        "datasets": datasets,
        "count_figures": {},
        "j0_figures": j0_figures,
        "slope_figures": slope_figures,
        "model_figures": model_figures,
        "model_methodology": methodology_path,
        "output_dir": output_dir,
        "readme": readme,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_generalized_urine_analysis(args.output_dir)
    print(f"Urine analysis completed in {result['output_dir']}")


if __name__ == "__main__":
    main()
