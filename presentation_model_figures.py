"""Create three concise, slide-ready figures from the model result tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figure_layout


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "fig/models/presentation_summary"
QSPECT_DIR = PROJECT_DIR / "fig/models/patient_sensitivity_generalized/shared"
URINE_SLOPE_DIR = (
    PROJECT_DIR
    / "fig/models/patient_sensitivity_urine_target/2_sensibilite_J0_J2/complet"
)

PATIENT_ORDER = figure_layout.PATIENT_ORDER
PATIENT_LABELS = {"p11": "P11", "p17": "P17", "p8": "P8"}
CYCLE_COLORS = {"p11": "#1F77B4", "p8": "#D62728", "p17": "#2CA02C"}
PATIENT_COLORS = CYCLE_COLORS


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: Mapping[str, Any], field: str) -> float:
    return float(row[field])


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_simple_urine_relation(output_path: Path) -> Path:
    """Show the descriptive three-point urine slope for photopeak GM global."""
    slopes = _read_csv(URINE_SLOPE_DIR / "pentes_fenetres_J0_J2.csv")
    predictions = _read_csv(URINE_SLOPE_DIR / "predictions_intracycle.csv")
    slope_rows = {
        row["patient_id"]: row
        for row in slopes
        if row["combination"] == "GM_globale" and row["window"] == "photopeak"
    }
    point_rows = {
        patient: [
            row
            for row in predictions
            if row["patient_id"] == patient
            and row["combination"] == "GM_globale"
            and row["window"] == "photopeak"
        ]
        for patient in PATIENT_ORDER
    }

    figure, axes = plt.subplots(
        1, 3, figsize=(16.5, 6.2), facecolor="white", layout="constrained"
    )
    for axis, patient in zip(axes, PATIENT_ORDER):
        rows = point_rows[patient]
        activity_gbq = np.asarray(
            [_float(row, "reference_qspect_activity_mbq") for row in rows]
        ) / 1000.0
        rate_kcps = np.asarray(
            [_float(row, "dead_time_corrected_rate_cps") for row in rows]
        ) / 1000.0
        slope = _float(slope_rows[patient], "slope_cps_per_mbq")
        r_squared = _float(slope_rows[patient], "r_squared_zero_origin")
        mare = _float(slope_rows[patient], "mare_activity_percent")
        color = CYCLE_COLORS[patient]

        x_line = np.linspace(0.0, 1.04 * float(np.max(activity_gbq)), 150)
        axis.plot(x_line, slope * x_line, color=color, linewidth=2.4)
        axis.scatter(
            activity_gbq,
            rate_kcps,
            s=78,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        axis.scatter([0.0], [0.0], marker="X", s=62, color="#222222", zorder=4)
        for row, x_value, y_value in zip(rows, activity_gbq, rate_kcps):
            axis.annotate(
                row["timepoint"].replace("Day", "J"),
                (x_value, y_value),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=13,
            )
        axis.text(
            0.96,
            0.94,
            f"S = {slope:.3f} cps/MBq\nR²₀ = {r_squared:.3f}\nMARE = {mare:.1f} %",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=14,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        cycle = slope_rows[patient]["cycle_id"].split("__", 1)[0]
        axis.set_title(
            f"{PATIENT_LABELS[patient]} — {cycle}", fontsize=17, weight="bold"
        )
        axis.set_xlabel(
            "Activité corporelle estimée par bilan urinaire (GBq)", fontsize=14
        )
        axis.set_ylabel("Taux de comptage du photopeak (kcps)", fontsize=14)
        axis.tick_params(axis="both", labelsize=13)
        axis.invert_xaxis()
        axis.grid(True, linestyle="--", alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Ajustement linéaire entre l’activité urinaire et le taux de comptage du photopeak",
        fontsize=21,
        weight="bold",
    )
    return _save(figure, output_path)


def _selected_model_metrics() -> list[Dict[str, str]]:
    rows = _read_csv(QSPECT_DIR / "all_model_metrics.csv")
    return [
        row
        for row in rows
        if row["rate_mode"] == "raw_photopeak_full_gm"
        and row["split_type"] == "leave_one_patient_out"
        and row["metric_scope"] == "combined_test"
        and row["fit_status"] == "fitted"
    ]


def plot_primary_model_comparison(output_path: Path) -> Path:
    """Compare the principal models in the deployable full-image GM mode."""
    rows = _selected_model_metrics()
    models = (
        ("M0_constant", "M0\nmoyenne"),
        ("M0b_origin_slope", "M0b\npente"),
        ("M1_locked_spectral", "M1\nratio"),
        ("M2_spectral_plus_ap_pa", "M2\nratio + AP/PA"),
        ("M9_raw_ratio_plus_all_ap_pa", "M9\nratio inverse + AP/PA"),
    )
    x = np.arange(len(models), dtype=float)
    width = 0.23
    figure, axis = plt.subplots(figsize=(11.8, 6.0), facecolor="white")
    for patient_index, patient in enumerate(PATIENT_ORDER):
        split_id = f"patient_holdout_{patient}"
        values = [
            _float(
                next(
                    row
                    for row in rows
                    if row["split_id"] == split_id and row["model"] == model
                ),
                "activity_mare_percent",
            )
            for model, _ in models
        ]
        offsets = x + (patient_index - 1) * width
        bars = axis.bar(
            offsets,
            values,
            width=width,
            color=PATIENT_COLORS[patient],
            label=PATIENT_LABELS[patient],
        )
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=10, rotation=90)

    axis.set_xticks(x, [label for _, label in models], fontsize=13)
    axis.set_ylabel("MARE sur l’activité Q/SPECT (%)", fontsize=13)
    axis.set_xlabel("Référence ou modèle", fontsize=13)
    axis.set_title(
        "Validation interpatient — photopeak complet, GM",
        fontsize=18,
        weight="bold",
    )
    axis.legend(title="Patient test", frameon=False, ncol=3, loc="upper right")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0.0, max(axis.get_ylim()[1], 55.0))
    figure.text(
        0.5,
        0.01,
        "Chaque patient est entièrement exclu de l’entraînement. Une MARE plus faible indique une meilleure prédiction.",
        ha="center",
        fontsize=11,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    return _save(figure, output_path)


def plot_primary_predictions(output_path: Path) -> Path:
    """Show M2 held-out activity predictions at full and zoomed scales."""
    rows = _read_csv(QSPECT_DIR / "all_model_predictions.csv")
    selected = [
        row
        for row in rows
        if row["rate_mode"] == "raw_photopeak_full_gm"
        and row["split_type"] == "leave_one_patient_out"
        and row["model"] == "M2_spectral_plus_ap_pa"
        and row["uses_test_reference_j0"] == "False"
    ]
    reference = np.asarray([_float(row, "reference_activity_mbq") for row in selected])
    predicted = np.asarray([_float(row, "predicted_activity_mbq") for row in selected])
    global_mare = float(
        np.mean(np.abs(100.0 * (predicted / reference - 1.0)))
    )

    figure, axes = plt.subplots(
        1, 2, figsize=(13.5, 6.1), facecolor="white", layout="constrained"
    )
    limits = (
        (0.0, 1.08 * float(max(np.max(reference), np.max(predicted)))),
        (0.0, 2500.0),
    )
    panel_titles = ("Étendue complète", "Agrandissement ≤ 2,5 GBq")
    for axis, (lower, upper), panel_title in zip(axes, limits, panel_titles):
        x_line = np.linspace(lower, upper, 200)
        axis.fill_between(
            x_line,
            0.8 * x_line,
            1.2 * x_line,
            color="#D9D9D9",
            alpha=0.55,
            label="±20 %",
        )
        axis.plot(x_line, x_line, "--", color="#222222", linewidth=1.7, label="Identité")
        for patient in PATIENT_ORDER:
            patient_rows = [row for row in selected if row["patient_id"] == patient]
            x_values = np.asarray(
                [_float(row, "reference_activity_mbq") for row in patient_rows]
            )
            y_values = np.asarray(
                [_float(row, "predicted_activity_mbq") for row in patient_rows]
            )
            patient_mare = float(
                np.mean(np.abs(100.0 * (y_values / x_values - 1.0)))
            )
            axis.scatter(
                x_values,
                y_values,
                s=70,
                color=PATIENT_COLORS[patient],
                edgecolor="white",
                linewidth=0.9,
                label=f"{PATIENT_LABELS[patient]} — MARE {patient_mare:.1f} %",
                zorder=3,
            )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(panel_title, fontsize=14, weight="bold")
        axis.set_xlabel("Activité Q/SPECT (MBq)", fontsize=12)
        axis.set_ylabel("Activité planaire prédite (MBq)", fontsize=12)
        axis.grid(True, linestyle="--", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="lower right", fontsize=10)
    axes[0].text(
        0.04,
        0.94,
        f"MARE globale = {global_mare:.1f} %",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    figure.suptitle(
        "Prédictions interpatient sans réajustement — M2, photopeak complet GM",
        fontsize=18,
        weight="bold",
    )
    return _save(figure, output_path)


def run(output_dir: Path = OUTPUT_DIR) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "simple_urine": plot_simple_urine_relation(
            output_dir / "01_relation_simple_comptes_urine.png"
        ),
        "model_comparison": plot_primary_model_comparison(
            output_dir / "02_validation_interpatient_modeles.png"
        ),
        "predictions": plot_primary_predictions(
            output_dir / "03_predictions_interpatient_M2.png"
        ),
    }
    readme = output_dir / "README.txt"
    readme.write_text(
        """FIGURES RECOMMANDÉES POUR LA PRÉSENTATION
========================================

1. Relation simple comptes–activité urinaire
   Répond directement à la demande d'analyse simple. Le R² décrit la linéarité;
   la MARE décrit la précision quantitative. Ajustement descriptif sur J0–J2.

2. Validation interpatient des modèles
   Figure principale. Photopeak complet GM, donc aucune TEW et aucun crop guidé
   par Q/SPECT. Chaque patient est entièrement exclu de l'entraînement.

3. Activité prédite versus Q/SPECT
   Traduction clinique des erreurs du modèle M2. Le panneau de droite agrandit
   les activités inférieures à 2,5 GBq; la bande grise correspond à ±20 %.

Si seulement deux figures sont permises, présenter 1 et 2. La figure 3 peut
servir à expliquer concrètement la MARE ou aller en annexe.
""",
        encoding="utf-8",
    )
    paths["readme"] = readme
    return paths


if __name__ == "__main__":
    for name, path in run().items():
        print(f"{name}: {path}")
