"""Analyse comptes--Q/SPECT par pente forcee par l'origine.

Cette analyse reproduit, avec l'activite Q/SPECT comme reference, les figures
J0--J2 auparavant produites avec le bilan urinaire.  Pour chaque cycle, les
trois premiers timepoints disponibles sont utilises pour ajuster

    R = S * A_QSPECT,

ou R est un taux de comptage planaire corrige globalement pour le temps mort
et S est une sensibilite en cps/MBq.  Les quatre fenetres d'energie sont
analysees avec AP+PA, la somme de la GM pixel par pixel et la GM globale.

Le fit intracycle est descriptif.  Les transferts diriges cycle -> cycle sont
egalement calcules sans reajustement afin de separer qualite d'ajustement et
generalisation.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import numpy as np

import figure_layout


DEFAULT_MASTER_TABLE = (
    figure_layout.shared_dir("patient_sensitivity_generalized")
    / "analysis_master_table.csv"
)
DEFAULT_OUTPUT_DIR = figure_layout.shared_dir(__file__)

WINDOWS = (
    ("lower_scatter", "Fenetre inferieure", "#9467bd"),
    ("photopeak", "Photopeak", "#1f77b4"),
    ("upper_scatter", "Fenetre superieure", "#ff7f0e"),
    ("low_energy_scatter", "Diffusion generale", "#d62728"),
)

WINDOW_ENGLISH_LABELS = {
    "lower_scatter": "Lower Scatter",
    "photopeak": "Photopeak",
    "upper_scatter": "Upper Scatter",
    "low_energy_scatter": "Low Energy Scatter",
}

COMBINATIONS = (
    ("AP_PA", "AP + PA", "ap_pa"),
    ("GM", "GM pixel", "gm"),
    ("GM_globale", "GM globale", "gm_global"),
)

COUNT_SUPPORTS = {
    "crop": {
        "column_prefix": "crop",
        "folder": "crop",
        "label": "crop Q/SPECT",
        "description": (
            "crop planaire par profil longitudinal, de meme couverture "
            "que le volume Q/SPECT"
        ),
    },
    "full": {
        "column_prefix": "full",
        "folder": "complet",
        "label": "image complete",
        "description": "image planaire complete, sans crop longitudinal",
    },
}

# Couleurs verrouillees par patient-cycle. Elles sont partagees par les
# analyses Q/SPECT et urinaires afin qu'une couleur conserve toujours la meme
# signification d'une figure a l'autre.
CYCLE_COLORS = {
    ("p11", "2026-05__Studies"): "#1f77b4",  # bleu
    ("p11", "2026-07__Studies"): "#6f4c9b",  # violet
    ("p8", "2026-06__Studies"): "#d62728",   # rouge
    ("p17", "2026-09__Studies"): "#2ca02c",  # vert
}
DEFAULT_CYCLE_COLOR = "#777777"


def _count_rate_relation_title(
    window: str,
    reference_label: str,
    *,
    j0_calibration: bool,
) -> str:
    """Construire un titre descriptif uniforme pour les figures comptes--activité."""
    reference = reference_label.split(",", 1)[0].strip()
    window_phrases = {
        "lower_scatter": "de la fenêtre de diffusé inférieure",
        "photopeak": "du photopeak",
        "upper_scatter": "de la fenêtre de diffusé supérieure",
        "low_energy_scatter": "du diffusé général à basse énergie",
    }
    phrase = window_phrases.get(window, f"de la fenêtre {window}")
    if j0_calibration:
        return (
            f"Relation entre l’activité {reference} et le taux de comptage "
            f"{phrase} — calibration à J0"
        )
    return (
        f"Ajustement linéaire entre l’activité {reference} et le taux de "
        f"comptage {phrase}"
    )


def cycle_color(row: Mapping[str, Any]) -> str:
    """Retourner la couleur stable associee au patient et a son cycle."""
    key = (str(row["patient_id"]), str(row["cycle_id"]))
    return CYCLE_COLORS.get(key, DEFAULT_CYCLE_COLOR)


def _float(row: Mapping[str, Any], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Valeur non finie dans {field}: {row[field]!r}")
    return value


def format_value_with_uncertainty(value: float, uncertainty: float) -> str:
    """Formater S ± EC avec un chiffre significatif sur l'incertitude.

    La valeur centrale est arrondie au meme rang decimal que l'incertitude.
    Exemple: 0.745 ± 0.032 devient 0.75 ± 0.03.
    """
    value = float(value)
    uncertainty = abs(float(uncertainty))
    if not (math.isfinite(value) and math.isfinite(uncertainty)):
        raise ValueError("Valeur ou incertitude non finie")
    if uncertainty == 0.0:
        return f"{value:.3g} ± 0"

    decimal_places = -math.floor(math.log10(uncertainty))
    quantum = Decimal("1").scaleb(-decimal_places)
    rounded_value = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    rounded_uncertainty = Decimal(str(uncertainty)).quantize(
        quantum, rounding=ROUND_HALF_UP
    )
    displayed_decimals = max(0, decimal_places)
    return (
        f"{rounded_value:.{displayed_decimals}f} ± "
        f"{rounded_uncertainty:.{displayed_decimals}f}"
    )


def load_master_table(path: Path) -> list[dict[str, Any]]:
    """Charger et valider la table unique generee par le pipeline generalise."""
    if not path.exists():
        raise FileNotFoundError(
            f"Table maitresse absente: {path}. "
            "Executer d'abord patient_sensitivity_generalized.py."
        )
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    required = {
        "dataset_key",
        "patient_id",
        "cycle_id",
        "dataset_label",
        "timepoint",
        "planar_day",
        "actual_frame_duration_s",
        "dead_time_dtcf",
    }
    for support in ("full", "crop"):
        for window, _, _ in WINDOWS:
            required.update(
                {
                    f"counts_{support}_{window}_ap_counts",
                    f"counts_{support}_{window}_pa_counts",
                    f"counts_{support}_{window}_ap_pa_counts",
                    f"counts_{support}_{window}_gm_counts",
                }
            )
    missing = sorted(required.difference(rows[0] if rows else {}))
    if missing:
        raise ValueError("Colonnes manquantes: " + ", ".join(missing))
    rows.sort(
        key=lambda row: (
            *figure_layout.patient_cycle_sort_key(
                str(row["patient_id"]), str(row["cycle_id"])
            ),
            float(row["planar_day"]),
        )
    )
    return rows


def select_count_support(
    rows: Sequence[Mapping[str, Any]], count_support: str
) -> list[dict[str, Any]]:
    """Associer un support de comptes a toutes les lignes d'une analyse."""
    if count_support not in COUNT_SUPPORTS:
        raise ValueError(
            f"Support inconnu {count_support!r}; choix: "
            + ", ".join(COUNT_SUPPORTS)
        )
    prefix = str(COUNT_SUPPORTS[count_support]["column_prefix"])
    selected: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["count_support"] = count_support
        for window, _, _ in WINDOWS:
            for suffix in ("ap_counts", "pa_counts", "ap_pa_counts", "gm_counts"):
                field = f"counts_{prefix}_{window}_{suffix}"
                if field not in row:
                    raise ValueError(f"Colonne manquante pour {count_support}: {field}")
        selected.append(row)
    return selected


def prepare_qspect_reference_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normaliser la cible Q/SPECT vers le champ commun des analyses de pente."""
    prepared: list[dict[str, Any]] = []
    for source in rows:
        if "qspect_activity_at_planar_time_mbq" not in source:
            raise ValueError("Colonne Q/SPECT absente de la table maitresse")
        row = dict(source)
        row["reference_activity_mbq"] = _float(
            source, "qspect_activity_at_planar_time_mbq"
        )
        prepared.append(row)
    return prepared


def group_first_timepoints(
    rows: Sequence[Mapping[str, Any]], number: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """Regrouper par cycle et garder les premiers points chronologiques."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset_key"]), []).append(dict(row))
    selected: dict[str, list[dict[str, Any]]] = {}
    for key, cycle_rows in grouped.items():
        ordered = sorted(cycle_rows, key=lambda row: _float(row, "planar_day"))
        if len(ordered) >= number:
            selected[key] = ordered[:number]
    if not selected:
        raise ValueError("Aucun cycle ne contient au moins trois points apparies")
    return selected


def corrected_rate(
    row: Mapping[str, Any], window: str, combination_code: str
) -> float:
    """Retourner le taux corrige pour le temps mort dans une fenetre."""
    duration = _float(row, "actual_frame_duration_s")
    dtcf = _float(row, "dead_time_dtcf")
    if duration <= 0.0 or dtcf <= 0.0:
        raise ValueError("Duree et DTCF doivent etre positifs")
    count_support = str(row.get("count_support", "full"))
    if count_support not in COUNT_SUPPORTS:
        raise ValueError(f"Support de comptes inconnu: {count_support}")
    column_prefix = str(COUNT_SUPPORTS[count_support]["column_prefix"])
    prefix = f"counts_{column_prefix}_{window}"
    if combination_code == "ap_pa":
        counts = _float(row, f"{prefix}_ap_pa_counts")
    elif combination_code == "gm":
        counts = _float(row, f"{prefix}_gm_counts")
    elif combination_code == "gm_global":
        ap = _float(row, f"{prefix}_ap_counts")
        pa = _float(row, f"{prefix}_pa_counts")
        counts = math.sqrt(max(ap, 0.0) * max(pa, 0.0))
    else:
        raise ValueError(f"Combinaison inconnue: {combination_code}")
    return counts / duration * dtcf


def zero_origin_fit(activity: Sequence[float], rate: Sequence[float]) -> dict[str, float]:
    """Ajuster R=S*A et retourner les statistiques descriptives."""
    x = np.asarray(activity, dtype=float)
    y = np.asarray(rate, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("activity et rate doivent contenir au moins deux points")
    denominator = float(np.dot(x, x))
    if denominator <= 0.0:
        raise ValueError("Impossible d'ajuster une pente avec activite nulle")
    slope = float(np.dot(x, y) / denominator)
    predicted_rate = slope * x
    residuals = y - predicted_rate
    residual_sum = float(np.sum(residuals**2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else math.nan
    degrees_of_freedom = max(int(x.size) - 1, 1)
    slope_se = float(
        np.sqrt((residual_sum / degrees_of_freedom) / denominator)
    )
    pointwise_sensitivity = y / x
    m0_mean = float(np.mean(pointwise_sensitivity))
    predicted_activity = y / slope
    errors_percent = 100.0 * (predicted_activity / x - 1.0)
    return {
        "slope_cps_per_mbq": slope,
        "slope_standard_error_cps_per_mbq": slope_se,
        "r_squared_zero_origin": r_squared,
        "mare_activity_percent": float(np.mean(np.abs(errors_percent))),
        "bias_activity_percent": float(np.mean(errors_percent)),
        "max_abs_error_activity_percent": float(np.max(np.abs(errors_percent))),
        "m0_mean_sensitivity_cps_per_mbq": m0_mean,
        "m0_mare_activity_percent": float(
            np.mean(np.abs(100.0 * (y / m0_mean / x - 1.0)))
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build_fit_tables(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construire les pentes et predictions descriptives intracycle."""
    fit_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for key, cycle_rows in grouped.items():
        example = cycle_rows[0]
        activity = np.asarray(
            [_float(row, "reference_activity_mbq") for row in cycle_rows]
        )
        for folder, combination_label, combination_code in COMBINATIONS:
            for window, window_label, _ in WINDOWS:
                rates = np.asarray(
                    [corrected_rate(row, window, combination_code) for row in cycle_rows]
                )
                fit = zero_origin_fit(activity, rates)
                common = {
                    "dataset_key": key,
                    "patient_id": example["patient_id"],
                    "cycle_id": example["cycle_id"],
                    "dataset_label": example["dataset_label"],
                    "count_support": example.get("count_support", "full"),
                    "combination": folder,
                    "combination_label": combination_label,
                    "window": window,
                    "window_label": window_label,
                    "number_of_points": len(cycle_rows),
                    **fit,
                }
                fit_rows.append(common)
                for row, true_activity, rate in zip(cycle_rows, activity, rates):
                    predicted = rate / fit["slope_cps_per_mbq"]
                    relative_error = 100.0 * (
                        predicted / true_activity - 1.0
                    )
                    prediction_rows.append(
                        {
                            **{field: common[field] for field in (
                                "dataset_key", "patient_id", "cycle_id",
                                "dataset_label", "count_support", "combination",
                                "combination_label", "window", "window_label",
                            )},
                            "timepoint": row["timepoint"],
                            "planar_day": row["planar_day"],
                            "reference_activity_mbq": true_activity,
                            "reference_qspect_activity_mbq": true_activity,
                            "dead_time_corrected_rate_cps": rate,
                            "predicted_activity_mbq": predicted,
                            "relative_error_percent": relative_error,
                            "absolute_error_percent": abs(relative_error),
                        }
                    )
    return fit_rows, prediction_rows


def build_j0_tables(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calibrer chaque sensibilite uniquement avec J0 et evaluer les autres jours."""
    sensitivity_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for key, cycle_rows in grouped.items():
        example = cycle_rows[0]
        activity = np.asarray(
            [_float(row, "reference_activity_mbq") for row in cycle_rows], dtype=float
        )
        for folder, combination_label, combination_code in COMBINATIONS:
            for window, window_label, _ in WINDOWS:
                rates = np.asarray(
                    [corrected_rate(row, window, combination_code) for row in cycle_rows],
                    dtype=float,
                )
                sensitivity = float(rates[0] / activity[0])
                predicted = rates / sensitivity
                errors = 100.0 * (predicted / activity - 1.0)
                test_errors = errors[1:]
                common = {
                    "dataset_key": key,
                    "patient_id": example["patient_id"],
                    "cycle_id": example["cycle_id"],
                    "dataset_label": example["dataset_label"],
                    "count_support": example.get("count_support", "full"),
                    "combination": folder,
                    "combination_label": combination_label,
                    "window": window,
                    "window_label": window_label,
                    "calibration_timepoint": cycle_rows[0]["timepoint"],
                    "sensitivity_j0_cps_per_mbq": sensitivity,
                    "number_of_test_points": int(test_errors.size),
                    "test_mare_percent": float(np.mean(np.abs(test_errors))),
                    "test_bias_percent": float(np.mean(test_errors)),
                    "test_max_abs_error_percent": float(
                        np.max(np.abs(test_errors))
                    ),
                }
                sensitivity_rows.append(common)
                for index, (row, true_activity, rate, estimate, error) in enumerate(
                    zip(cycle_rows, activity, rates, predicted, errors)
                ):
                    prediction_rows.append(
                        {
                            **common,
                            "timepoint": row["timepoint"],
                            "planar_day": row["planar_day"],
                            "used_for_calibration": index == 0,
                            "reference_activity_mbq": true_activity,
                            "dead_time_corrected_rate_cps": rate,
                            "predicted_activity_mbq": estimate,
                            "relative_error_percent": error,
                            "absolute_error_percent": abs(error),
                        }
                    )
    return sensitivity_rows, prediction_rows


def build_point_error_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    calibration_method: str,
) -> list[dict[str, Any]]:
    """Normaliser les erreurs point par point et leurs deux moyennes.

    ``mare_all_points_percent`` porte sur les trois points utilisés dans ces
    analyses. ``mare_after_first_percent`` exclut le premier point
    chronologique. Les mêmes définitions sont utilisées pour les cibles urine
    et Q/SPECT, y compris lorsqu'un cycle commence à J1 plutôt qu'à J0.
    """
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in prediction_rows:
        key = (
            str(row["dataset_key"]),
            str(row["combination"]),
            str(row["window"]),
        )
        groups.setdefault(key, []).append(row)

    normalized: list[dict[str, Any]] = []
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda row: float(row["planar_day"]))
        absolute_errors = np.asarray(
            [abs(float(row["relative_error_percent"])) for row in ordered],
            dtype=float,
        )
        mare_all = float(np.mean(absolute_errors))
        mare_post_j0 = (
            float(np.mean(absolute_errors[1:]))
            if absolute_errors.size > 1
            else math.nan
        )
        for index, row in enumerate(ordered):
            reference_activity = float(
                row.get(
                    "reference_activity_mbq",
                    row.get("reference_qspect_activity_mbq"),
                )
            )
            rate = float(row["dead_time_corrected_rate_cps"])
            predicted_activity = float(row["predicted_activity_mbq"])
            signed_error = float(row["relative_error_percent"])
            sensitivity = rate / predicted_activity
            if calibration_method == "J0_only":
                used_for_sensitivity = index == 0
                point_role = "calibration" if index == 0 else "test"
            elif calibration_method == "J0_J2_fit":
                used_for_sensitivity = True
                point_role = "ajustement descriptif"
            else:
                raise ValueError(
                    f"Methode de calibration inconnue: {calibration_method}"
                )
            normalized.append(
                {
                    "dataset_key": row["dataset_key"],
                    "patient_id": row["patient_id"],
                    "cycle_id": row["cycle_id"],
                    "dataset_label": row["dataset_label"],
                    "count_support": row["count_support"],
                    "combination": row["combination"],
                    "combination_label": row["combination_label"],
                    "window": row["window"],
                    "window_label": row["window_label"],
                    "calibration_method": calibration_method,
                    "timepoint": row["timepoint"],
                    "planar_day": float(row["planar_day"]),
                    "used_to_estimate_sensitivity": used_for_sensitivity,
                    "point_role": point_role,
                    "reference_activity_mbq": reference_activity,
                    "planar_rate_cps": rate,
                    "sensitivity_cps_per_mbq": sensitivity,
                    "predicted_activity_mbq": predicted_activity,
                    "signed_relative_error_percent": signed_error,
                    "mare_all_points_percent": mare_all,
                    "mare_after_first_percent": mare_post_j0,
                }
            )
    return normalized


def _light_cycle_color(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(cycle_color(row)), dtype=float)
    light = 0.82 + 0.18 * rgb
    return float(light[0]), float(light[1]), float(light[2]), 1.0


def _format_signed_percent(value: float) -> str:
    """Afficher une erreur signée sans produire l'étiquette déroutante -0.0."""
    value = float(value)
    if abs(value) < 0.05:
        value = 0.0
    return f"{value:+.1f}"


def plot_point_error_table(
    point_rows: Sequence[Mapping[str, Any]],
    combination: str,
    window: str,
    path: Path,
    reference_label: str,
    *,
    calibration_method: str,
) -> Path:
    """Créer un tableau lisible des erreurs d'une fenêtre à chaque acquisition."""
    selected = [
        row
        for row in point_rows
        if row["combination"] == combination and row["window"] == window
    ]
    if not selected:
        raise ValueError(f"Aucune erreur {window} pour {combination}")

    reference = reference_label.split(",", 1)[0].strip()
    window_phrases = {
        "lower_scatter": "la fenêtre de diffusé inférieure",
        "photopeak": "le photopeak",
        "upper_scatter": "la fenêtre de diffusé supérieure",
        "low_energy_scatter": "le diffusé général à basse énergie",
    }
    window_phrase = window_phrases.get(window, str(selected[0]["window_label"]))
    method_title = (
        "calibration au premier point"
        if calibration_method == "J0_only"
        else "ajustement sur trois points"
    )
    columns = (
        "Patient / cycle",
        "Point",
        "Activité réf.\n(GBq)",
        "Activité préd.\n(GBq)",
        "Erreur\n(%)",
        "MARE 3 points\n(%)",
        "MARE 2 derniers\n(%)",
    )
    table_values: list[list[str]] = []
    row_colors: list[tuple[float, float, float, float]] = []
    seen_datasets: set[str] = set()
    for row in selected:
        dataset_key = str(row["dataset_key"])
        is_first = dataset_key not in seen_datasets
        seen_datasets.add(dataset_key)
        timepoint = str(row["timepoint"]).replace("Day", "J")
        table_values.append(
            [
                _cycle_label(row).replace("\n", " — "),
                timepoint,
                f"{float(row['reference_activity_mbq']) / 1000.0:.3f}",
                f"{float(row['predicted_activity_mbq']) / 1000.0:.3f}",
                _format_signed_percent(row["signed_relative_error_percent"]),
                f"{float(row['mare_all_points_percent']):.1f}" if is_first else "",
                f"{float(row['mare_after_first_percent']):.1f}" if is_first else "",
            ]
        )
        row_colors.append(_light_cycle_color(row))

    figure_height = max(6.4, 1.15 + 0.53 * len(table_values))
    figure, axis = plt.subplots(figsize=(16.0, figure_height), facecolor="white")
    axis.axis("off")
    table = axis.table(
        cellText=table_values,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=(0.19, 0.07, 0.14, 0.14, 0.14, 0.16, 0.16),
        bbox=(0.0, 0.02, 1.0, 0.88),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for column_index in range(len(columns)):
        cell = table[(0, column_index)]
        cell.set_facecolor("#E8E8E8")
        cell.set_text_props(weight="bold", fontsize=12.0)
        cell.set_edgecolor("#666666")
    for row_index, color in enumerate(row_colors, start=1):
        for column_index in range(len(columns)):
            cell = table[(row_index, column_index)]
            cell.set_edgecolor("#B8B8B8")
            cell.set_facecolor(color if column_index == 0 else "white")
        table[(row_index, 0)].set_text_props(weight="bold")

    axis.set_title(
        f"Erreurs sur l’activité {reference} à partir de {window_phrase} — "
        f"{method_title}",
        fontsize=21,
        fontweight="bold",
        pad=18,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def build_j0_fit_comparison_rows(
    j0_sensitivities: Sequence[Mapping[str, Any]],
    fitted_slopes: Sequence[Mapping[str, Any]],
    j0_point_errors: Sequence[Mapping[str, Any]],
    fit_point_errors: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aligner les sensibilités et erreurs des deux méthodes."""
    fit_lookup = {
        (str(row["dataset_key"]), str(row["combination"]), str(row["window"])): row
        for row in fitted_slopes
    }
    fit_point_lookup = {
        (
            str(row["dataset_key"]),
            str(row["combination"]),
            str(row["window"]),
            str(row["timepoint"]),
        ): row
        for row in fit_point_errors
    }
    sensitivity_rows: list[dict[str, Any]] = []
    for j0_row in j0_sensitivities:
        key = (
            str(j0_row["dataset_key"]),
            str(j0_row["combination"]),
            str(j0_row["window"]),
        )
        fit_row = fit_lookup.get(key)
        if fit_row is None:
            continue
        s_j0 = float(j0_row["sensitivity_j0_cps_per_mbq"])
        s_fit = float(fit_row["slope_cps_per_mbq"])
        sensitivity_rows.append(
            {
                "dataset_key": j0_row["dataset_key"],
                "patient_id": j0_row["patient_id"],
                "cycle_id": j0_row["cycle_id"],
                "dataset_label": j0_row["dataset_label"],
                "count_support": j0_row["count_support"],
                "combination": j0_row["combination"],
                "combination_label": j0_row["combination_label"],
                "window": j0_row["window"],
                "window_label": j0_row["window_label"],
                "sensitivity_first_point_cps_per_mbq": s_j0,
                "sensitivity_three_point_fit_cps_per_mbq": s_fit,
                "fit_slope_standard_error_cps_per_mbq": float(
                    fit_row["slope_standard_error_cps_per_mbq"]
                ),
                "relative_difference_fit_vs_first_percent": 100.0
                * (s_fit / s_j0 - 1.0),
                "mare_first_point_all_percent": float(
                    next(
                        row["mare_all_points_percent"]
                        for row in j0_point_errors
                        if (
                            str(row["dataset_key"]),
                            str(row["combination"]),
                            str(row["window"]),
                        )
                        == key
                    )
                ),
                "mare_fit_all_percent": float(fit_row["mare_activity_percent"]),
            }
        )

    point_rows: list[dict[str, Any]] = []
    sensitivity_lookup = {
        (str(row["dataset_key"]), str(row["combination"]), str(row["window"])): row
        for row in sensitivity_rows
    }
    for j0_row in j0_point_errors:
        point_key = (
            str(j0_row["dataset_key"]),
            str(j0_row["combination"]),
            str(j0_row["window"]),
            str(j0_row["timepoint"]),
        )
        fit_row = fit_point_lookup.get(point_key)
        summary = sensitivity_lookup.get(point_key[:3])
        if fit_row is None or summary is None:
            continue
        point_rows.append(
            {
                "dataset_key": j0_row["dataset_key"],
                "patient_id": j0_row["patient_id"],
                "cycle_id": j0_row["cycle_id"],
                "dataset_label": j0_row["dataset_label"],
                "count_support": j0_row["count_support"],
                "combination": j0_row["combination"],
                "combination_label": j0_row["combination_label"],
                "window": j0_row["window"],
                "window_label": j0_row["window_label"],
                "timepoint": j0_row["timepoint"],
                "planar_day": float(j0_row["planar_day"]),
                "reference_activity_mbq": float(
                    j0_row["reference_activity_mbq"]
                ),
                "first_point_predicted_activity_mbq": float(
                    j0_row["predicted_activity_mbq"]
                ),
                "first_point_error_percent": float(
                    j0_row["signed_relative_error_percent"]
                ),
                "fit_predicted_activity_mbq": float(
                    fit_row["predicted_activity_mbq"]
                ),
                "fit_error_percent": float(
                    fit_row["signed_relative_error_percent"]
                ),
                "mare_first_point_all_percent": float(
                    summary["mare_first_point_all_percent"]
                ),
                "mare_fit_all_percent": float(summary["mare_fit_all_percent"]),
            }
        )
    return sensitivity_rows, point_rows


def plot_j0_fit_error_table(
    point_rows: Sequence[Mapping[str, Any]],
    combination: str,
    window: str,
    path: Path,
    reference_label: str,
) -> Path:
    """Comparer les erreurs signées des deux méthodes, point par point."""
    selected = [
        row
        for row in point_rows
        if row["combination"] == combination and row["window"] == window
    ]
    if not selected:
        raise ValueError(f"Aucune comparaison {combination}/{window}")
    reference = reference_label.split(",", 1)[0].strip()
    window_label = WINDOW_ENGLISH_LABELS.get(window, str(selected[0]["window_label"]))
    values = []
    colors = []
    for row in selected:
        values.append(
            [
                _cycle_label(row).replace("\n", " — "),
                str(row["timepoint"]).replace("Day", "J"),
                f"{float(row['reference_activity_mbq']) / 1000.0:.3f}",
                f"{float(row['first_point_predicted_activity_mbq']) / 1000.0:.3f}",
                _format_signed_percent(row["first_point_error_percent"]),
                f"{float(row['fit_predicted_activity_mbq']) / 1000.0:.3f}",
                _format_signed_percent(row["fit_error_percent"]),
            ]
        )
        colors.append(_light_cycle_color(row))
    figure_height = max(6.0, 1.2 + 0.50 * len(values))
    figure, axis = plt.subplots(figsize=(17.0, figure_height), facecolor="white")
    axis.axis("off")
    table = axis.table(
        cellText=values,
        colLabels=(
            "Patient / cycle",
            "Point",
            "Activité réf.\n(GBq)",
            "Activité préd.\npremier point (GBq)",
            "Erreur premier\npoint (%)",
            "Activité préd.\npente (GBq)",
            "Erreur pente\n(%)",
        ),
        cellLoc="center",
        colLoc="center",
        colWidths=(0.20, 0.07, 0.13, 0.16, 0.14, 0.16, 0.14),
        bbox=(0.0, 0.02, 1.0, 0.87),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.0)
    for column in range(7):
        header = table[(0, column)]
        header.set_facecolor("#E8E8E8")
        header.set_text_props(weight="bold", fontsize=12.5)
        header.set_edgecolor("#666666")
    for row_index, color in enumerate(colors, start=1):
        for column in range(7):
            cell = table[(row_index, column)]
            cell.set_edgecolor("#B8B8B8")
            cell.set_facecolor(color if column == 0 else "white")
        table[(row_index, 0)].set_text_props(weight="bold")
    axis.set_title(
        f"Erreurs sur l’activité {reference} : calibration ou ajustement — "
        f"{window_label}",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_four_window_error_table(
    point_rows: Sequence[Mapping[str, Any]],
    combination: str,
    path: Path,
    reference_label: str,
) -> Path:
    """Réunir les erreurs J0/pente des quatre fenêtres dans un seul tableau."""
    selected = [row for row in point_rows if row["combination"] == combination]
    if not selected:
        raise ValueError(f"Aucune erreur pour {combination}")
    ordered_points = list(
        dict.fromkeys(
            (str(row["dataset_key"]), str(row["timepoint"])) for row in selected
        )
    )
    lookup = {
        (str(row["dataset_key"]), str(row["timepoint"]), str(row["window"])): row
        for row in selected
    }
    values: list[list[str]] = []
    colors: list[tuple[float, float, float, float]] = []
    for dataset_key, timepoint in ordered_points:
        rows = [lookup[(dataset_key, timepoint, window)] for window, _, _ in WINDOWS]
        example = rows[0]
        values.append(
            [
                _cycle_label(example).replace("\n", " — "),
                timepoint.replace("Day", "J"),
                *[
                    formatted
                    for row in rows
                    for formatted in (
                        _format_signed_percent(row["first_point_error_percent"]),
                        _format_signed_percent(row["fit_error_percent"]),
                    )
                ],
            ]
        )
        colors.append(_light_cycle_color(example))

    columns = ["Patient / cycle", "Point"]
    for window, _, _ in WINDOWS:
        label = WINDOW_ENGLISH_LABELS[window]
        columns.extend((f"{label}\npremier point (%)", f"{label}\npente (%)"))
    figure_height = max(7.0, 1.3 + 0.52 * len(values))
    figure, axis = plt.subplots(figsize=(22.0, figure_height), facecolor="white")
    axis.axis("off")
    table = axis.table(
        cellText=values,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=(0.16, 0.055, *([0.098] * 8)),
        bbox=(0.0, 0.02, 1.0, 0.87),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.3)
    for column in range(len(columns)):
        header = table[(0, column)]
        header.set_facecolor("#E8E8E8")
        header.set_text_props(weight="bold", fontsize=10.3)
        header.set_edgecolor("#666666")
    for row_index, color in enumerate(colors, start=1):
        for column in range(len(columns)):
            cell = table[(row_index, column)]
            cell.set_edgecolor("#B8B8B8")
            cell.set_facecolor(color if column == 0 else "white")
        table[(row_index, 0)].set_text_props(weight="bold")
    reference = reference_label.split(",", 1)[0].strip()
    axis.set_title(
        f"Erreurs sur l’activité {reference} : premier point ou pente ajustée",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_four_window_sensitivity_table(
    sensitivity_rows: Sequence[Mapping[str, Any]],
    combination: str,
    path: Path,
    reference_label: str,
) -> Path:
    """Réunir les sensibilités J0/pente des quatre fenêtres dans un tableau."""
    selected = [row for row in sensitivity_rows if row["combination"] == combination]
    if not selected:
        raise ValueError(f"Aucune sensibilité pour {combination}")
    reference = reference_label.split(",", 1)[0].strip()
    dataset_keys = list(dict.fromkeys(str(row["dataset_key"]) for row in selected))
    lookup = {
        (str(row["dataset_key"]), str(row["window"])): row for row in selected
    }
    values: list[list[str]] = []
    colors: list[tuple[float, float, float, float]] = []
    for dataset_key in dataset_keys:
        rows = [lookup[(dataset_key, window)] for window, _, _ in WINDOWS]
        example = rows[0]
        values.append(
            [
                _cycle_label(example).replace("\n", " — "),
                *[
                    (
                        f"Premier point : "
                        f"{float(row['sensitivity_first_point_cps_per_mbq']):.3f}\n"
                        f"Pente : "
                        f"{format_value_with_uncertainty(float(row['sensitivity_three_point_fit_cps_per_mbq']), float(row['fit_slope_standard_error_cps_per_mbq']))}"
                    )
                    for row in rows
                ],
            ]
        )
        colors.append(_light_cycle_color(example))

    figure_height = max(5.8, 1.8 + 1.05 * len(values))
    figure, axis = plt.subplots(figsize=(18.0, figure_height), facecolor="white")
    axis.axis("off")
    table = axis.table(
        cellText=values,
        colLabels=("Patient / cycle", *[WINDOW_ENGLISH_LABELS[w] for w, _, _ in WINDOWS]),
        cellLoc="center",
        colLoc="center",
        colWidths=(0.18, 0.205, 0.205, 0.205, 0.205),
        bbox=(0.0, 0.04, 1.0, 0.79),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for column in range(5):
        header = table[(0, column)]
        header.set_facecolor("#E8E8E8")
        header.set_text_props(weight="bold", fontsize=12.0)
        header.set_edgecolor("#666666")
    for row_index, color in enumerate(colors, start=1):
        for column in range(5):
            cell = table[(row_index, column)]
            cell.set_edgecolor("#B8B8B8")
            cell.set_facecolor(color if column == 0 else "white")
        table[(row_index, 0)].set_text_props(weight="bold")
    axis.set_title(
        f"Sensibilités planaires relatives à l’activité {reference}",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _comparison_method_legend() -> list[Patch]:
    return [
        Patch(facecolor="#777777", edgecolor="#222222", label="Premier point"),
        Patch(
            facecolor="#777777",
            edgecolor="#222222",
            hatch="///",
            label="Ajustement sur trois points",
        ),
    ]


def plot_four_window_error_comparison(
    sensitivity_rows: Sequence[Mapping[str, Any]],
    combination: str,
    path: Path,
    reference_label: str,
) -> Path:
    """Comparer la MARE des deux méthodes dans quatre panneaux énergétiques."""
    selected = [row for row in sensitivity_rows if row["combination"] == combination]
    dataset_keys = list(dict.fromkeys(str(row["dataset_key"]) for row in selected))
    max_value = max(
        max(
            float(row["mare_first_point_all_percent"]),
            float(row["mare_fit_all_percent"]),
        )
        for row in selected
    )
    figure, axes = plt.subplots(2, 2, figsize=(16.5, 11.2), facecolor="white")
    x = np.arange(len(dataset_keys), dtype=float)
    width = 0.34
    for axis, (window, _, _) in zip(axes.flat, WINDOWS):
        rows = [
            next(
                row for row in selected
                if row["dataset_key"] == key and row["window"] == window
            )
            for key in dataset_keys
        ]
        first_values = [float(row["mare_first_point_all_percent"]) for row in rows]
        fit_values = [float(row["mare_fit_all_percent"]) for row in rows]
        colors = [cycle_color(row) for row in rows]
        bars_first = axis.bar(
            x - width / 2,
            first_values,
            width,
            color=colors,
            edgecolor="#222222",
            linewidth=0.8,
            zorder=3,
        )
        bars_fit = axis.bar(
            x + width / 2,
            fit_values,
            width,
            color=colors,
            edgecolor="#222222",
            linewidth=0.8,
            hatch="///",
            zorder=3,
        )
        for bars, values in ((bars_first, first_values), (bars_fit, fit_values)):
            axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=11)
        axis.set_xticks(
            x,
            [_cycle_label(row).replace("\n", "\n") for row in rows],
        )
        axis.set_title(WINDOW_ENGLISH_LABELS[window], fontsize=18, fontweight="bold")
        axis.set_ylabel("MARE sur trois points (%)", fontsize=14)
        axis.tick_params(axis="both", labelsize=12)
        axis.set_ylim(0.0, max_value * 1.25)
        axis.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    figure.legend(
        handles=_comparison_method_legend(),
        frameon=False,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
    )
    reference = reference_label.split(",", 1)[0].strip()
    figure.suptitle(
        f"Erreur sur l’activité {reference} selon la méthode de sensibilité",
        fontsize=22,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), h_pad=2.2, w_pad=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_four_window_sensitivity_comparison(
    sensitivity_rows: Sequence[Mapping[str, Any]],
    combination: str,
    path: Path,
    reference_label: str,
) -> Path:
    """Comparer S au premier point et S ajustée sur trois points."""
    selected = [row for row in sensitivity_rows if row["combination"] == combination]
    dataset_keys = list(dict.fromkeys(str(row["dataset_key"]) for row in selected))
    figure, axes = plt.subplots(2, 2, figsize=(16.5, 11.2), facecolor="white")
    x = np.arange(len(dataset_keys), dtype=float)
    width = 0.34
    for axis, (window, _, _) in zip(axes.flat, WINDOWS):
        rows = [
            next(
                row for row in selected
                if row["dataset_key"] == key and row["window"] == window
            )
            for key in dataset_keys
        ]
        first_values = [
            float(row["sensitivity_first_point_cps_per_mbq"]) for row in rows
        ]
        fit_values = [
            float(row["sensitivity_three_point_fit_cps_per_mbq"]) for row in rows
        ]
        colors = [cycle_color(row) for row in rows]
        bars_first = axis.bar(
            x - width / 2,
            first_values,
            width,
            color=colors,
            edgecolor="#222222",
            linewidth=0.8,
            zorder=3,
        )
        bars_fit = axis.bar(
            x + width / 2,
            fit_values,
            width,
            color=colors,
            edgecolor="#222222",
            linewidth=0.8,
            hatch="///",
            zorder=3,
        )
        decimals = 3 if max(first_values + fit_values) < 10.0 else 1
        for bars, values in ((bars_first, first_values), (bars_fit, fit_values)):
            axis.bar_label(
                bars,
                labels=[f"{value:.{decimals}f}" for value in values],
                padding=3,
                fontsize=11,
            )
        axis.set_xticks(x, [_cycle_label(row) for row in rows])
        axis.set_title(WINDOW_ENGLISH_LABELS[window], fontsize=18, fontweight="bold")
        axis.set_ylabel("Sensibilité (cps/MBq)", fontsize=14)
        axis.tick_params(axis="both", labelsize=12)
        axis.margins(y=0.20)
        axis.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    figure.legend(
        handles=_comparison_method_legend(),
        frameon=False,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
    )
    reference = reference_label.split(",", 1)[0].strip()
    figure.suptitle(
        f"Sensibilité planaire {reference} : premier point ou ajustement",
        fontsize=22,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), h_pad=2.2, w_pad=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def run_j0_fit_comparison(
    j0_dir: Path,
    fit_dir: Path,
    output_dir: Path,
    *,
    reference_label: str,
) -> dict[str, Path]:
    """Généraliser la section 4 pour les trois combinaisons et quatre fenêtres."""
    j0_sensitivities = _read_csv(j0_dir / "sensibilites_J0.csv")
    fitted_slopes = _read_csv(fit_dir / "pentes_fenetres_J0_J2.csv")
    j0_errors = _read_csv(j0_dir / "erreurs_par_point.csv")
    fit_errors = _read_csv(fit_dir / "erreurs_par_point.csv")
    sensitivity_rows, point_rows = build_j0_fit_comparison_rows(
        j0_sensitivities,
        fitted_slopes,
        j0_errors,
        fit_errors,
    )
    outputs: dict[str, Path] = {}
    for combination, _, _ in COMBINATIONS:
        combination_dir = output_dir / combination
        selected_sensitivities = [
            row for row in sensitivity_rows if row["combination"] == combination
        ]
        selected_points = [
            row for row in point_rows if row["combination"] == combination
        ]
        for legacy_path in combination_dir.glob(
            "tableau_comparaison_sensibilites_*.png"
        ):
            legacy_path.unlink()
        legacy_sensitivity_figure = (
            combination_dir / "comparaison_sensibilites_quatre_fenetres.png"
        )
        if legacy_sensitivity_figure.is_file():
            legacy_sensitivity_figure.unlink()
        outputs[f"{combination}_sensitivities_csv"] = _write_csv(
            combination_dir / "comparaison_sensibilites.csv",
            selected_sensitivities,
        )
        outputs[f"{combination}_point_errors_csv"] = _write_csv(
            combination_dir / "comparaison_erreurs_par_point.csv",
            selected_points,
        )
        for window, _, _ in WINDOWS:
            outputs[f"{combination}_{window}_table"] = plot_j0_fit_error_table(
                point_rows,
                combination,
                window,
                combination_dir / f"tableau_comparaison_erreurs_{window}.png",
                reference_label,
            )
        outputs[f"{combination}_four_window_error_table"] = (
            plot_four_window_error_table(
                point_rows,
                combination,
                combination_dir / "tableau_comparaison_erreurs_quatre_fenetres.png",
                reference_label,
            )
        )
        outputs[f"{combination}_four_window_sensitivity_table"] = (
            plot_four_window_sensitivity_table(
                sensitivity_rows,
                combination,
                combination_dir
                / "tableau_comparaison_sensibilites_quatre_fenetres.png",
                reference_label,
            )
        )
        outputs[f"{combination}_error_mega"] = plot_four_window_error_comparison(
            sensitivity_rows,
            combination,
            combination_dir / "comparaison_erreurs_quatre_fenetres.png",
            reference_label,
        )
    return outputs


def plot_j0_window_evaluation(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
    combination_folder: str,
    combination_code: str,
    window: str,
    window_label: str,
    path: Path,
    reference_label: str,
) -> Path:
    """Montrer la droite (0,0)--J0 et les erreurs des points ulterieurs."""
    keys = list(grouped)
    lookup = {
        (row["dataset_key"], row["combination"], row["window"]): row
        for row in sensitivity_rows
    }
    fig, axes = plt.subplots(
        1, len(keys), figsize=(4.35 * len(keys), 5.25), squeeze=False,
        layout="constrained",
    )
    for axis, key in zip(axes[0], keys):
        cycle_rows = grouped[key]
        color = cycle_color(cycle_rows[0])
        activity = np.asarray(
            [_float(row, "reference_activity_mbq") for row in cycle_rows], dtype=float
        )
        rates = np.asarray(
            [corrected_rate(row, window, combination_code) for row in cycle_rows],
            dtype=float,
        )
        result = lookup[(key, combination_folder, window)]
        sensitivity = float(result["sensitivity_j0_cps_per_mbq"])
        predicted_activity = rates / sensitivity
        errors = 100.0 * (predicted_activity / activity - 1.0)
        x_gbq = activity / 1000.0
        y_kcps = rates / 1000.0

        axis.scatter(
            [x_gbq[0]], [y_kcps[0]], s=82, color=color,
            edgecolor="white", linewidth=1.0, zorder=4, label="Calibration J0",
        )
        if len(cycle_rows) > 1:
            axis.scatter(
                x_gbq[1:], y_kcps[1:], s=76, facecolor="white",
                edgecolor=color, linewidth=2.1, zorder=4, label="Evaluation",
            )
        axis.scatter([0.0], [0.0], marker="X", s=55, color="#222222", zorder=5)
        x_line = np.linspace(0.0, float(np.max(x_gbq)) * 1.06, 150)
        axis.plot(x_line, sensitivity * x_line, color=color, linewidth=2.1)

        for index, (row, x_value, y_value) in enumerate(
            zip(cycle_rows, x_gbq, y_kcps)
        ):
            label = str(row["timepoint"])
            if index > 0:
                label += f"\n|ecart S| {abs(errors[index]):.1f}%"
            axis.annotate(
                label, (x_value, y_value), xytext=(6, 5),
                textcoords="offset points", fontsize=9.5,
            )
        axis.text(
            0.96, 0.95,
            f"S_J0 = {sensitivity:.3f} cps/MBq\n"
            f"MARE J>0 = {float(result['test_mare_percent']):.1f} %",
            transform=axis.transAxes, ha="right", va="top", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        )
        axis.set_title(_cycle_label(cycle_rows[0]), fontsize=12, fontweight="bold")
        axis.set_xlabel(f"Activite {reference_label} (GBq)")
        axis.set_ylabel("Taux de comptage (kcps)")
        axis.invert_xaxis()
        axis.grid(True, linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        _count_rate_relation_title(
            window,
            reference_label,
            j0_calibration=True,
        ),
        fontsize=16, fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_j0_error_summary(
    prediction_rows: Sequence[Mapping[str, Any]], combination: str, path: Path,
    reference_label: str,
) -> Path:
    """Afficher l'erreur absolue aux jours non utilises pour la calibration."""
    selected = [
        row for row in prediction_rows
        if row["combination"] == combination and not row["used_for_calibration"]
    ]
    keys = list(dict.fromkeys(str(row["dataset_key"]) for row in selected))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4), layout="constrained")
    for axis, (window, window_label, _) in zip(axes.flat, WINDOWS):
        label_offsets = ((5, 7), (5, -15), (5, 7), (5, -15))
        for key_index, key in enumerate(keys):
            rows = sorted(
                [
                    row for row in selected
                    if row["dataset_key"] == key and row["window"] == window
                ],
                key=lambda row: float(row["planar_day"]),
            )
            if not rows:
                continue
            axis.plot(
                [float(row["planar_day"]) for row in rows],
                [float(row["absolute_error_percent"]) for row in rows],
                marker="o", markersize=7, linewidth=2.0,
                color=cycle_color(rows[0]), label=_cycle_label(rows[0]).replace("\n", " "),
            )
            for row in rows:
                axis.annotate(
                    str(row["timepoint"]),
                    (float(row["planar_day"]), float(row["absolute_error_percent"])),
                    xytext=label_offsets[key_index % len(label_offsets)],
                    textcoords="offset points", fontsize=8.5,
                )
        axis.set_title(window_label, fontweight="bold")
        axis.set_xlabel("Temps depuis la premiere acquisition planaire (jours)")
        axis.set_ylabel("Ecart absolu de sensibilite par rapport a J0 (%)")
        axis.grid(True, linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    combination_label = next(
        label for folder, label, _ in COMBINATIONS if folder == combination
    )
    fig.suptitle(
        f"Variation de la sensibilite apres J0 — cible {reference_label} — {combination_label}",
        fontsize=16, fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _format_sensitivity_value(value: float) -> str:
    """Afficher une sensibilite J0 avec trois chiffres significatifs."""
    return f"{float(value):.3g}"


def plot_j0_sensitivity_summary(
    sensitivity_rows: Sequence[Mapping[str, Any]],
    combination: str,
    path: Path,
    reference_label: str,
) -> Path:
    """Comparer les sensibilites J0 avec le style de la figure J0--J2."""
    selected = [
        row for row in sensitivity_rows if row["combination"] == combination
    ]
    keys = list(dict.fromkeys(str(row["dataset_key"]) for row in selected))
    labels = [
        _cycle_label(next(row for row in selected if row["dataset_key"] == key))
        for key in keys
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.5), layout="constrained")
    for axis, (window, _window_label, _) in zip(axes.flat, WINDOWS):
        rows = [
            next(
                row for row in selected
                if row["dataset_key"] == key and row["window"] == window
            )
            for key in keys
        ]
        values = np.asarray(
            [float(row["sensitivity_j0_cps_per_mbq"]) for row in rows]
        )
        bars = axis.bar(
            np.arange(len(keys)), values,
            color=[cycle_color(row) for row in rows], alpha=0.9,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                _format_sensitivity_value(float(value)),
                ha="center", va="bottom", fontsize=14,
            )
        axis.set_xticks(np.arange(len(keys)), labels)
        axis.set_ylabel("Sensibilité S_J0 (cps/MBq)", fontsize=16)
        axis.set_title(
            WINDOW_ENGLISH_LABELS[window], fontsize=18, fontweight="bold"
        )
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(y=0.20)
    combination_label = next(
        label for folder, label, _ in COMBINATIONS if folder == combination
    )
    fig.suptitle(
        f"Sensibilité {reference_label} calibrée à J0 — {combination_label}",
        fontsize=23,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_j0_sensitivity_table(
    sensitivity_rows: Sequence[Mapping[str, Any]],
    combination: str,
    path: Path,
    reference_label: str,
) -> Path:
    """Creer le tableau J0 analogue au tableau des pentes J0--J2."""
    selected = [
        row for row in sensitivity_rows if row["combination"] == combination
    ]
    keys = list(dict.fromkeys(str(row["dataset_key"]) for row in selected))
    row_labels = [
        _cycle_label(next(row for row in selected if row["dataset_key"] == key))
        for key in keys
    ]
    table_values: list[list[str]] = []
    for key in keys:
        table_values.append(
            [
                _format_sensitivity_value(
                    float(
                        next(
                            row for row in selected
                            if row["dataset_key"] == key and row["window"] == window
                        )["sensitivity_j0_cps_per_mbq"]
                    )
                )
                for window, _, _ in WINDOWS
            ]
        )
    cv_values: list[str] = []
    for window, _, _ in WINDOWS:
        values = np.asarray(
            [
                float(row["sensitivity_j0_cps_per_mbq"])
                for row in selected if row["window"] == window
            ],
            dtype=float,
        )
        cv = 100.0 * float(np.std(values, ddof=1) / np.mean(values))
        cv_values.append(f"{cv:.1f} %")
    table_values.append(cv_values)
    row_labels.append("CV intercycle")
    column_labels = [WINDOW_ENGLISH_LABELS[window] for window, _, _ in WINDOWS]

    figure, axis = plt.subplots(figsize=(15.5, 5.2), facecolor="white")
    axis.axis("off")
    table = axis.table(
        cellText=table_values,
        rowLabels=row_labels,
        colLabels=column_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=(0.02, 0.15, 0.96, 0.73),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 1.55)
    for column in range(len(column_labels)):
        cell = table[(0, column)]
        cell.set_facecolor("#4A4A4A")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    final_row = len(row_labels)
    for column in range(-1, len(column_labels)):
        if (final_row, column) in table.get_celld():
            table[(final_row, column)].set_facecolor("#E8E8E8")
            table[(final_row, column)].get_text().set_fontweight("bold")
    combination_label = next(
        label for folder, label, _ in COMBINATIONS if folder == combination
    )
    axis.set_title(
        f"Tableau des sensibilités {reference_label} à J0 — {combination_label}",
        fontsize=20,
        fontweight="bold",
        pad=14,
    )
    figure.text(
        0.5,
        0.045,
        "Une calibration J0 ne permet pas d’estimer une erreur-type de pente.",
        ha="center",
        fontsize=13,
        color="#444444",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def build_transfer_table(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    fit_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Appliquer chaque pente a tous les autres cycles sans reajustement."""
    fit_lookup = {
        (row["dataset_key"], row["combination"], row["window"]): row
        for row in fit_rows
    }
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for source_key, source_rows in grouped.items():
        source_example = source_rows[0]
        for target_key, target_rows in grouped.items():
            if source_key == target_key:
                continue
            target_example = target_rows[0]
            activity = np.asarray(
                [_float(row, "reference_activity_mbq") for row in target_rows]
            )
            for folder, combination_label, combination_code in COMBINATIONS:
                for window, window_label, _ in WINDOWS:
                    slope = float(
                        fit_lookup[(source_key, folder, window)]["slope_cps_per_mbq"]
                    )
                    m0_sensitivity = float(
                        fit_lookup[(source_key, folder, window)][
                            "m0_mean_sensitivity_cps_per_mbq"
                        ]
                    )
                    rates = np.asarray(
                        [corrected_rate(row, window, combination_code) for row in target_rows]
                    )
                    predicted = rates / slope
                    errors = 100.0 * (predicted / activity - 1.0)
                    m0_predicted = rates / m0_sensitivity
                    m0_errors = 100.0 * (m0_predicted / activity - 1.0)
                    metric = {
                        "count_support": source_example.get("count_support", "full"),
                        "training_dataset": source_key,
                        "training_patient": source_example["patient_id"],
                        "training_cycle": source_example["cycle_id"],
                        "test_dataset": target_key,
                        "test_patient": target_example["patient_id"],
                        "test_cycle": target_example["cycle_id"],
                        "same_patient": (
                            source_example["patient_id"] == target_example["patient_id"]
                        ),
                        "combination": folder,
                        "combination_label": combination_label,
                        "window": window,
                        "window_label": window_label,
                        "slope_cps_per_mbq": slope,
                        "m0_mean_sensitivity_cps_per_mbq": m0_sensitivity,
                        "mare_percent": float(np.mean(np.abs(errors))),
                        "bias_percent": float(np.mean(errors)),
                        "rmse_mbq": float(np.sqrt(np.mean((predicted - activity) ** 2))),
                        "max_abs_error_percent": float(np.max(np.abs(errors))),
                        "m0_mare_percent": float(np.mean(np.abs(m0_errors))),
                        "m0_bias_percent": float(np.mean(m0_errors)),
                        "m0_rmse_mbq": float(
                            np.sqrt(np.mean((m0_predicted - activity) ** 2))
                        ),
                        "m0_max_abs_error_percent": float(
                            np.max(np.abs(m0_errors))
                        ),
                        "delta_mare_slope_minus_m0_percent_points": float(
                            np.mean(np.abs(errors)) - np.mean(np.abs(m0_errors))
                        ),
                        "number_of_test_points": len(target_rows),
                    }
                    metrics.append(metric)
                    for (
                        row,
                        true_activity,
                        rate,
                        estimate,
                        error,
                        m0_estimate,
                        m0_error,
                    ) in zip(
                        target_rows,
                        activity,
                        rates,
                        predicted,
                        errors,
                        m0_predicted,
                        m0_errors,
                    ):
                        predictions.append(
                            {
                                **metric,
                                "timepoint": row["timepoint"],
                                "planar_day": row["planar_day"],
                                "reference_qspect_activity_mbq": true_activity,
                                "dead_time_corrected_rate_cps": rate,
                                "predicted_activity_mbq": estimate,
                                "relative_error_percent": error,
                                "m0_predicted_activity_mbq": m0_estimate,
                                "m0_relative_error_percent": m0_error,
                            }
                        )
    return metrics, predictions


def _cycle_label(row: Mapping[str, Any]) -> str:
    return str(row["dataset_label"]).replace(" 2026-", "\n2026-")


def plot_window_fits(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    fit_rows: Sequence[Mapping[str, Any]],
    combination_folder: str,
    combination_code: str,
    window: str,
    window_label: str,
    path: Path,
    reference_label: str,
) -> Path:
    """Montrer visuellement la construction de chaque pente de cycle."""
    keys = list(grouped)
    lookup = {
        (row["dataset_key"], row["combination"], row["window"]): row
        for row in fit_rows
    }
    fig, axes = plt.subplots(
        1, len(keys), figsize=(4.35 * len(keys), 5.2), squeeze=False,
        layout="constrained",
    )
    for axis, key in zip(axes[0], keys):
        cycle_rows = grouped[key]
        activity = np.asarray(
            [_float(row, "reference_activity_mbq") for row in cycle_rows]
        )
        rates = np.asarray(
            [corrected_rate(row, window, combination_code) for row in cycle_rows]
        )
        fit = lookup[(key, combination_folder, window)]
        color = cycle_color(cycle_rows[0])
        x_gbq = activity / 1000.0
        y_kcps = rates / 1000.0
        axis.scatter(
            x_gbq, y_kcps, s=72, color=color, edgecolor="white",
            linewidth=0.9, zorder=3,
        )
        axis.scatter([0.0], [0.0], marker="X", s=55, color="#222222", zorder=4)
        for row, x_value, y_value in zip(cycle_rows, x_gbq, y_kcps):
            axis.annotate(
                str(row["timepoint"]), (x_value, y_value), xytext=(5, 5),
                textcoords="offset points", fontsize=10,
            )
        x_line = np.linspace(0.0, float(np.max(x_gbq)) * 1.06, 150)
        y_line = float(fit["slope_cps_per_mbq"]) * x_line
        axis.plot(x_line, y_line, color=color, linewidth=2.1)
        axis.text(
            0.96, 0.95,
            f"S = {float(fit['slope_cps_per_mbq']):.3f} cps/MBq\n"
            f"R² = {float(fit['r_squared_zero_origin']):.3f}\n"
            f"MARE = {float(fit['mare_activity_percent']):.1f} %",
            transform=axis.transAxes, ha="right", va="top", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        )
        axis.set_title(_cycle_label(cycle_rows[0]), fontsize=12, fontweight="bold")
        axis.set_xlabel(f"Activite {reference_label} (GBq)")
        axis.set_ylabel("Taux de comptage (kcps)")
        axis.invert_xaxis()
        axis.grid(True, linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        _count_rate_relation_title(
            window,
            reference_label,
            j0_calibration=False,
        ),
        fontsize=16, fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_slope_summary(
    fit_rows: Sequence[Mapping[str, Any]], combination: str, path: Path,
    reference_label: str,
) -> Path:
    """Comparer les sensibilites par cycle sans surcharger avec les QC du fit."""
    selected = [row for row in fit_rows if row["combination"] == combination]
    keys = list(dict.fromkeys(str(row["dataset_key"]) for row in selected))
    labels = [
        _cycle_label(next(row for row in selected if row["dataset_key"] == key))
        for key in keys
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.5), layout="constrained")
    for axis, (window, _window_label, _) in zip(axes.flat, WINDOWS):
        rows = [
            next(
                row for row in selected
                if row["dataset_key"] == key and row["window"] == window
            )
            for key in keys
        ]
        values = np.asarray([float(row["slope_cps_per_mbq"]) for row in rows])
        errors = np.asarray(
            [float(row["slope_standard_error_cps_per_mbq"]) for row in rows]
        )
        bars = axis.bar(
            np.arange(len(keys)), values, yerr=errors, capsize=4,
            color=[cycle_color(row) for row in rows], alpha=0.9,
        )
        for bar, row in zip(bars, rows):
            slope = float(row["slope_cps_per_mbq"])
            standard_error = float(row["slope_standard_error_cps_per_mbq"])
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + standard_error,
                format_value_with_uncertainty(slope, standard_error),
                ha="center", va="bottom", fontsize=14,
            )
        axis.set_xticks(np.arange(len(keys)), labels)
        axis.set_ylabel("Sensibilité S (cps/MBq)", fontsize=16)
        axis.set_title(
            WINDOW_ENGLISH_LABELS[window], fontsize=18, fontweight="bold"
        )
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(y=0.22)
    combination_label = next(
        label for folder, label, _ in COMBINATIONS if folder == combination
    )
    fig.suptitle(
        f"Sensibilité {reference_label} par cycle — {combination_label}",
        fontsize=23,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sensitivity_table(
    fit_rows: Sequence[Mapping[str, Any]],
    combination: str,
    path: Path,
    reference_label: str,
) -> Path:
    """Créer un tableau de présentation des sensibilités S ± SE par cycle."""
    selected = [row for row in fit_rows if row["combination"] == combination]
    keys = list(dict.fromkeys(str(row["dataset_key"]) for row in selected))
    cycle_labels = [
        _cycle_label(next(row for row in selected if row["dataset_key"] == key))
        for key in keys
    ]
    table_values: list[list[str]] = []
    for key in keys:
        cycle_values: list[str] = []
        for window, _window_label, _color in WINDOWS:
            row = next(
                item
                for item in selected
                if item["dataset_key"] == key and item["window"] == window
            )
            slope = float(row["slope_cps_per_mbq"])
            standard_error = float(row["slope_standard_error_cps_per_mbq"])
            cycle_values.append(format_value_with_uncertainty(slope, standard_error))
        table_values.append(cycle_values)

    cv_values: list[str] = []
    for window, _window_label, _color in WINDOWS:
        slopes = np.asarray(
            [
                float(row["slope_cps_per_mbq"])
                for row in selected
                if row["window"] == window
            ],
            dtype=float,
        )
        cv = 100.0 * float(np.std(slopes, ddof=1) / np.mean(slopes))
        cv_values.append(f"{cv:.1f} %")
    table_values.append(cv_values)
    row_labels = [*cycle_labels, "CV intercycle"]
    column_labels = [WINDOW_ENGLISH_LABELS[window] for window, _, _ in WINDOWS]

    figure, axis = plt.subplots(figsize=(15.5, 5.2), facecolor="white")
    axis.axis("off")
    table = axis.table(
        cellText=table_values,
        rowLabels=row_labels,
        colLabels=column_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=(0.02, 0.15, 0.96, 0.73),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 1.55)
    for column in range(len(column_labels)):
        cell = table[(0, column)]
        cell.set_facecolor("#4A4A4A")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    final_row = len(row_labels)
    for column in range(-1, len(column_labels)):
        if (final_row, column) in table.get_celld():
            table[(final_row, column)].set_facecolor("#E8E8E8")
            table[(final_row, column)].get_text().set_fontweight("bold")
    axis.set_title(
        f"Tableau des sensibilités {reference_label} — "
        + next(label for folder, label, _ in COMBINATIONS if folder == combination),
        fontsize=20,
        fontweight="bold",
        pad=14,
    )
    figure.text(
        0.5,
        0.045,
        "Valeurs : S ± 1 erreur-type, en cps/MBq. Le CV résume la variation entre les quatre cycles.",
        ha="center",
        fontsize=13,
        color="#444444",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_m0_vs_slope(
    fit_rows: Sequence[Mapping[str, Any]], combination: str, path: Path,
    reference_label: str,
) -> Path:
    """Comparer l'estimateur M0 moyen a la pente forcee par l'origine."""
    selected = [row for row in fit_rows if row["combination"] == combination]
    keys = list(dict.fromkeys(str(row["dataset_key"]) for row in selected))
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4), layout="constrained")
    for axis, (window, window_label, _) in zip(axes.flat, WINDOWS):
        rows = [
            next(
                row for row in selected
                if row["dataset_key"] == key and row["window"] == window
            )
            for key in keys
        ]
        x = np.arange(len(keys), dtype=float)
        width = 0.36
        axis.bar(
            x - width / 2,
            [float(row["m0_mare_activity_percent"]) for row in rows],
            width, label="M0: moyenne des S_i", color="#9aa0a6",
        )
        axis.bar(
            x + width / 2,
            [float(row["mare_activity_percent"]) for row in rows],
            width, label="Pente forcee par l'origine", color="#c8102e",
        )
        axis.set_xticks(x, [_cycle_label(row) for row in rows])
        axis.set_ylabel("MARE intracycle (%)")
        axis.set_title(window_label, fontweight="bold")
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    combination_label = next(
        label for folder, label, _ in COMBINATIONS if folder == combination
    )
    fig.suptitle(
        f"M0 versus pente {reference_label} — {combination_label}\n"
        "Ajustement et evaluation sur les memes trois points",
        fontsize=16, fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_transfer_heatmaps(
    metrics: Sequence[Mapping[str, Any]], combination: str, path: Path,
    reference_label: str,
) -> Path:
    """Afficher la MARE de tous les transferts diriges cycle vers cycle."""
    selected = [row for row in metrics if row["combination"] == combination]
    keys = list(dict.fromkeys(str(row["training_dataset"]) for row in selected))
    labels = []
    for key in keys:
        row = next(row for row in selected if row["training_dataset"] == key)
        labels.append(f"{str(row['training_patient']).upper()}\n{str(row['training_cycle'])[:7]}")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.4), layout="constrained")
    finite_values = [float(row["mare_percent"]) for row in selected]
    vmax = max(finite_values) if finite_values else 1.0
    image = None
    for axis, (window, window_label, _) in zip(axes.flat, WINDOWS):
        matrix = np.full((len(keys), len(keys)), np.nan)
        for row in selected:
            if row["window"] != window:
                continue
            i = keys.index(str(row["training_dataset"]))
            j = keys.index(str(row["test_dataset"]))
            matrix[i, j] = float(row["mare_percent"])
        image = axis.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=vmax)
        for i in range(len(keys)):
            for j in range(len(keys)):
                if math.isfinite(matrix[i, j]):
                    axis.text(
                        j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center",
                        color="white" if matrix[i, j] > 0.55 * vmax else "#222222",
                        fontsize=9, fontweight="bold",
                    )
                else:
                    axis.text(j, i, "—", ha="center", va="center", color="#777777")
        axis.set_xticks(np.arange(len(keys)), labels)
        axis.set_yticks(np.arange(len(keys)), labels)
        axis.set_xlabel("Cycle teste")
        axis.set_ylabel("Cycle de calibration")
        axis.set_title(window_label, fontweight="bold")
    if image is not None:
        fig.colorbar(image, ax=axes, label="MARE du transfert (%)", shrink=0.82)
    combination_label = next(
        label for folder, label, _ in COMBINATIONS if folder == combination
    )
    fig.suptitle(
        f"Transfert sans reajustement de la pente {reference_label} — {combination_label}",
        fontsize=16, fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_transfer_m0_vs_slope(
    metrics: Sequence[Mapping[str, Any]], combination: str, path: Path,
    reference_label: str,
) -> Path:
    """Comparer M0 et la pente sur les transferts diriges hors cycle."""
    selected = [row for row in metrics if row["combination"] == combination]
    labels = [window_label for _, window_label, _ in WINDOWS]
    m0_values = []
    slope_values = []
    for window, _, _ in WINDOWS:
        rows = [row for row in selected if row["window"] == window]
        m0_values.append(float(np.mean([float(row["m0_mare_percent"]) for row in rows])))
        slope_values.append(float(np.mean([float(row["mare_percent"]) for row in rows])))
    x = np.arange(len(WINDOWS), dtype=float)
    width = 0.36
    fig, axis = plt.subplots(figsize=(11.8, 6.3), layout="constrained")
    m0_bars = axis.bar(
        x - width / 2, m0_values, width, label="M0: moyenne des S_i",
        color="#9aa0a6",
    )
    slope_bars = axis.bar(
        x + width / 2, slope_values, width,
        label="Pente forcee par l'origine", color="#c8102e",
    )
    axis.bar_label(m0_bars, fmt="%.1f %%", padding=3, fontsize=10)
    axis.bar_label(slope_bars, fmt="%.1f %%", padding=3, fontsize=10)
    axis.set_xticks(x, labels)
    axis.set_ylabel("MARE moyenne des transferts diriges (%)")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    combination_label = next(
        label for folder, label, _ in COMBINATIONS if folder == combination
    )
    axis.set_title(
        f"Generalisation hors cycle: M0 versus pente {reference_label} — {combination_label}",
        fontsize=15, fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def write_methodology(
    path: Path,
    number_of_cycles: int,
    reference_label: str,
    reference_description: str,
    count_support: str,
) -> Path:
    support_description = str(COUNT_SUPPORTS[count_support]["description"])
    text = f"""Pentes {reference_label} sur les trois premiers timepoints
========================================================

Objectif
--------
Estimer une sensibilite constante a partir des comptes planaires et de
{reference_description}.

Donnees
-------
- {number_of_cycles} cycles complets;
- trois premiers timepoints chronologiques de chaque cycle;
- support des comptes: {support_description};
- quatre fenetres: inferieure, photopeak, superieure et diffusion generale;
- trois combinaisons: AP+PA, GM pixel et GM globale.

Pretraitement
-------------
Les comptes sont divises par Actual Frame Duration pour obtenir des cps. Le
facteur global de correction du temps mort DTCF est ensuite applique. Aucun CT,
aucune correction TEW et aucun Q/SPECT ne servent d'entree planaire.

Fit
---
Pour chaque cycle, fenetre et combinaison:

    R_i = S * A_reference,i
    S = somme(A_i * R_i) / somme(A_i^2)

L'origine est forcee: zero activite implique zero taux de comptage. La pente S
est en cps/MBq. Les barres d'erreur representent l'erreur-type de la pente.

M0 versus pente
---------------
M0, dans les modeles de sensibilite, utilise la moyenne arithmetique des
sensibilites individuelles S_i = R_i/A_i des donnees d'entrainement. La pente
forcee par l'origine est une autre facon d'estimer une sensibilite constante;
elle pondere davantage les points a forte activite parce que les A_i entrent au
carre. Elle ne remplace donc pas automatiquement M0.

Interpretation
--------------
Le R2 et la MARE intracycle sont descriptifs, car la pente est ajustee et
evaluee sur les memes trois points. La matrice de transfert est le test le plus
pertinent: une pente apprise sur un cycle est appliquee a un autre cycle sans
reajustement. Les deux cycles de P11 ne sont pas deux patients independants.

Erreur par point
----------------
Pour chaque acquisition i:

    A_predite,i = R_i / S
    erreur_i (%) = 100 * (A_predite,i / A_reference,i - 1)
    erreur_absolue_i (%) = valeur_absolue(erreur_i)

La MARE sur trois points est la moyenne des trois erreurs absolues. La MARE
sur les deux derniers points exclut simplement le premier point chronologique.
Les deux valeurs sont rapportees dans chaque tableau d'erreurs; elles restent
descriptives ici, puisque les trois points ont servi a ajuster S.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_j0_methodology(
    path: Path,
    number_of_cycles: int,
    reference_label: str,
    reference_description: str,
    count_support: str,
) -> Path:
    support_description = str(COUNT_SUPPORTS[count_support]["description"])
    text = f"""Sensibilite calibree uniquement avec J0
=========================================

Objectif
--------
Evaluer la stabilite temporelle d'une sensibilite obtenue avec un seul point:
J0. La reference est {reference_description}.

Donnees et pretraitement
------------------------
- {number_of_cycles} cycles avec une reference disponible;
- trois premiers timepoints chronologiques de chaque cycle;
- support des comptes: {support_description};
- quatre fenetres energetiques;
- AP+PA, GM pixel et GM globale;
- taux de comptage en cps avec correction globale du temps mort.

Calibration
-----------
Pour chaque cycle, fenetre et combinaison:

    S_J0 = R_J0 / A_J0
    A_predite(t) = R(t) / S_J0

Cette calibration equivaut a tracer l'unique droite passant par (0,0) et J0.
J1 et J2 ne servent ni a choisir ni a ajuster la sensibilite.

Evaluation
----------
L'erreur aux jours ulterieurs est:

    erreur(t) = 100 * [A_predite(t) / A_reference(t) - 1]

Comme A_predite(t) = R(t)/S_J0 et S_effective(t) = R(t)/A_reference(t), cette
quantite est exactement egale a 100 * [S_effective(t)/S_J0 - 1]. Elle peut
donc etre lue comme l'ecart relatif de sensibilite par rapport a J0.

Les figures individuelles montrent J0 comme point plein et les points de test
comme cercles blancs. La figure de synthese exclut J0 et montre uniquement
l'erreur absolue des acquisitions suivantes. Aucun transfert interpatient et
aucun modele M0-M9 ne font partie de cette section.

Le tableau d'erreurs rapporte deux moyennes:

    MARE_3_points = moyenne(|erreur_1|, |erreur_2|, |erreur_3|)
    MARE_2_derniers = moyenne(|erreur_2|, |erreur_3|)

Comme le premier point definit exactement S, son erreur vaut zero. Lorsque le
cycle commence a J0, MARE_3_points = (2/3) * MARE_J1_J2. Les libelles generaux
emploient « premier point » et « deux derniers » parce que certains cycles
commencent a J1 plutot qu'a J0.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_j0_analysis(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    reference_label: str,
    reference_description: str,
    count_support: str = "full",
) -> dict[str, Path]:
    """Executer la calibration J0 avec le meme moteur planaire que les pentes."""
    supported_rows = select_count_support(rows, count_support)
    grouped = group_first_timepoints(supported_rows)
    sensitivities, predictions = build_j0_tables(grouped)
    point_errors = build_point_error_rows(
        predictions,
        calibration_method="J0_only",
    )
    plot_reference_label = (
        f"{reference_label}, {COUNT_SUPPORTS[count_support]['label']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sensitivities": _write_csv(
            output_dir / "sensibilites_J0.csv", sensitivities
        ),
        "predictions": _write_csv(
            output_dir / "predictions_post_J0.csv", predictions
        ),
        "point_errors": _write_csv(
            output_dir / "erreurs_par_point.csv", point_errors
        ),
        "methodology": write_j0_methodology(
            output_dir / "methodologie.txt", len(grouped), reference_label,
            reference_description, count_support,
        ),
    }
    for folder, _, combination_code in COMBINATIONS:
        combination_dir = output_dir / folder
        combination_point_errors = [
            row for row in point_errors if row["combination"] == folder
        ]
        outputs[f"{folder}_point_errors_csv"] = _write_csv(
            combination_dir / "erreurs_par_point.csv",
            combination_point_errors,
        )
        for index, (window, window_label, _) in enumerate(WINDOWS, start=1):
            outputs[f"{folder}_{window}_point_errors_table"] = (
                plot_point_error_table(
                    point_errors,
                    folder,
                    window,
                    combination_dir
                    / f"tableau_erreurs_{window}_par_point.png",
                    plot_reference_label,
                    calibration_method="J0_only",
                )
            )
            outputs[f"{folder}_{window}"] = plot_j0_window_evaluation(
                grouped,
                sensitivities,
                folder,
                combination_code,
                window,
                window_label,
                combination_dir / f"{index:02d}_{window}.png",
                plot_reference_label,
            )
        outputs[f"{folder}_errors"] = plot_j0_error_summary(
            predictions,
            folder,
            combination_dir / "erreurs_post_J0_par_cycle.png",
            plot_reference_label,
        )
        outputs[f"{folder}_summary"] = plot_j0_sensitivity_summary(
            sensitivities,
            folder,
            combination_dir / "sensibilites_J0_par_cycle.png",
            plot_reference_label,
        )
        outputs[f"{folder}_table"] = plot_j0_sensitivity_table(
            sensitivities,
            folder,
            combination_dir / "tableau_sensibilites_J0.png",
            plot_reference_label,
        )
    return outputs


def run_prepared_analysis(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    reference_label: str,
    reference_description: str,
    count_support: str = "full",
) -> dict[str, Path]:
    """Executer le moteur commun avec un champ ``reference_activity_mbq``."""
    supported_rows = select_count_support(rows, count_support)
    grouped = group_first_timepoints(supported_rows)
    fit_rows, intracycle_predictions = build_fit_tables(grouped)
    point_errors = build_point_error_rows(
        intracycle_predictions,
        calibration_method="J0_J2_fit",
    )
    transfer_metrics, transfer_predictions = build_transfer_table(grouped, fit_rows)
    plot_reference_label = (
        f"{reference_label}, {COUNT_SUPPORTS[count_support]['label']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fits": _write_csv(output_dir / "pentes_fenetres_J0_J2.csv", fit_rows),
        "intracycle": _write_csv(
            output_dir / "predictions_intracycle.csv", intracycle_predictions
        ),
        "point_errors": _write_csv(
            output_dir / "erreurs_par_point.csv", point_errors
        ),
        "transfer_metrics": _write_csv(
            output_dir / "transferts_intercycle_metriques.csv", transfer_metrics
        ),
        "transfer_predictions": _write_csv(
            output_dir / "transferts_intercycle_predictions.csv", transfer_predictions
        ),
        "methodology": write_methodology(
            output_dir / "methodologie.txt", len(grouped), reference_label,
            reference_description, count_support,
        ),
    }
    for folder, _, combination_code in COMBINATIONS:
        combination_dir = output_dir / folder
        combination_point_errors = [
            row for row in point_errors if row["combination"] == folder
        ]
        outputs[f"{folder}_point_errors_csv"] = _write_csv(
            combination_dir / "erreurs_par_point.csv",
            combination_point_errors,
        )
        for index, (window, window_label, color) in enumerate(WINDOWS, start=1):
            outputs[f"{folder}_{window}_point_errors_table"] = (
                plot_point_error_table(
                    point_errors,
                    folder,
                    window,
                    combination_dir
                    / f"tableau_erreurs_{window}_par_point.png",
                    plot_reference_label,
                    calibration_method="J0_J2_fit",
                )
            )
            outputs[f"{folder}_{window}"] = plot_window_fits(
                grouped, fit_rows, folder, combination_code, window,
                window_label,
                combination_dir / f"{index:02d}_{window}.png",
                plot_reference_label,
            )
        outputs[f"{folder}_summary"] = plot_slope_summary(
            fit_rows, folder, combination_dir / "pentes_J0_J2_par_cycle.png",
            plot_reference_label,
        )
        outputs[f"{folder}_table"] = plot_sensitivity_table(
            fit_rows,
            folder,
            combination_dir / "tableau_sensibilites_J0_J2.png",
            plot_reference_label,
        )
        outputs[f"{folder}_m0"] = plot_m0_vs_slope(
            fit_rows, folder, combination_dir / "comparaison_M0_pente.png",
            plot_reference_label,
        )
        outputs[f"{folder}_transfer"] = plot_transfer_heatmaps(
            transfer_metrics, folder,
            combination_dir / "transferts_intercycle_MARE.png",
            plot_reference_label,
        )
        outputs[f"{folder}_transfer_comparison"] = plot_transfer_m0_vs_slope(
            transfer_metrics, folder,
            combination_dir / "comparaison_transfert_M0_pente.png",
            plot_reference_label,
        )
    return outputs


def _remove_legacy_supportless_outputs(section_dir: Path) -> None:
    """Retirer les anciennes sorties sans sous-dossier crop/complet."""
    for folder, _, _ in COMBINATIONS:
        legacy_directory = section_dir / folder
        if legacy_directory.is_dir():
            shutil.rmtree(legacy_directory)
    for filename in (
        "sensibilites_J0.csv",
        "predictions_post_J0.csv",
        "pentes_fenetres_J0_J2.csv",
        "predictions_intracycle.csv",
        "transferts_intercycle_metriques.csv",
        "transferts_intercycle_predictions.csv",
        "methodologie.txt",
    ):
        legacy_file = section_dir / filename
        if legacy_file.is_file():
            legacy_file.unlink()


def run_reference_sensitivity_sections(
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    *,
    reference_label: str,
    reference_description: str,
) -> dict[str, Path]:
    """Produire les memes sections J0 et J0--J2 pour une cible quelconque."""
    output_root = Path(output_root)
    section_j0 = output_root / "1_sensibilite_J0"
    section_slope = output_root / "2_sensibilite_J0_J2"
    section_comparison = output_root / "4_comparaison_J0_force_vs_pente"
    for section in (section_j0, section_slope, section_comparison):
        section.mkdir(parents=True, exist_ok=True)
    for section in (section_j0, section_slope):
        _remove_legacy_supportless_outputs(section)

    outputs: dict[str, Path] = {}
    for count_support, support_spec in COUNT_SUPPORTS.items():
        support_folder = str(support_spec["folder"])
        j0_outputs = run_j0_analysis(
            rows,
            section_j0 / support_folder,
            reference_label=reference_label,
            reference_description=reference_description,
            count_support=count_support,
        )
        slope_outputs = run_prepared_analysis(
            rows,
            section_slope / support_folder,
            reference_label=reference_label,
            reference_description=reference_description,
            count_support=count_support,
        )
        outputs.update(
            {
                f"j0_{count_support}_{name}": path
                for name, path in j0_outputs.items()
            }
        )
        outputs.update(
            {
                f"slope_{count_support}_{name}": path
                for name, path in slope_outputs.items()
            }
        )
        comparison_outputs = run_j0_fit_comparison(
            section_j0 / support_folder,
            section_slope / support_folder,
            section_comparison / support_folder,
            reference_label=(
                f"{reference_label}, {COUNT_SUPPORTS[count_support]['label']}"
            ),
        )
        outputs.update(
            {
                f"comparison_{count_support}_{name}": path
                for name, path in comparison_outputs.items()
            }
        )
    return outputs


def run_analysis(master_table: Path, output_dir: Path) -> dict[str, Path]:
    rows = prepare_qspect_reference_rows(load_master_table(master_table))
    return run_reference_sensitivity_sections(
        rows,
        output_dir,
        reference_label="Q/SPECT",
        reference_description=(
            "l'activite Q/SPECT ramenee au moment de l'acquisition planaire"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-table", type=Path, default=DEFAULT_MASTER_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_analysis(args.master_table, args.output_dir)
    print(f"Analyse terminee: {len(outputs)} fichiers principaux")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
