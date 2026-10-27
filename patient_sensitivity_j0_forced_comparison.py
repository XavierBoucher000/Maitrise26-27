"""Compare J0-only calibration with the zero-origin J0--J2 fitted slope.

The two methods are evaluated on J1--J2 for the same raw-photopeak global-GM
planar signal. Urine uses the complete planar image; Q/SPECT uses the matched
crop. J0-only is a genuine temporal test, whereas the J0--J2 fit is descriptive
because J1 and J2 participate in estimating its slope.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import patient_sensitivity_qspect_slopes as slopes


PROJECT_DIR = Path(__file__).resolve().parent
COMBINATION = "GM_globale"
WINDOW = "photopeak"
COMMON_KEYS = (
    "p11_2026_05_studies",
    "p8_2026_06_studies",
    "p17_2026_09_studies",
)

REFERENCE_SPECS = {
    "urine": {
        "label": "Urine — image complète",
        "short_label": "Urine",
        "j0_dir": PROJECT_DIR
        / "fig/models/patient_sensitivity_urine_target/1_sensibilite_J0/complet",
        "fit_dir": PROJECT_DIR
        / "fig/models/patient_sensitivity_urine_target/2_sensibilite_J0_J2/complet",
        "output_dir": PROJECT_DIR
        / "fig/models/patient_sensitivity_urine_target/4_comparaison_J0_force_vs_pente/complet/GM_globale",
    },
    "qspect": {
        "label": "Q/SPECT — crop",
        "short_label": "Q/SPECT",
        "j0_dir": PROJECT_DIR
        / "fig/models/patient_sensitivity_qspect_slopes/shared/1_sensibilite_J0/crop",
        "fit_dir": PROJECT_DIR
        / "fig/models/patient_sensitivity_qspect_slopes/shared/2_sensibilite_J0_J2/crop",
        "output_dir": PROJECT_DIR
        / "fig/models/patient_sensitivity_qspect_slopes/shared/4_comparaison_J0_force_vs_pente/crop/GM_globale",
    },
}
PRESENTATION_PATH = (
    PROJECT_DIR
    / "fig/models/presentation_summary/07_erreur_J0_force_vs_pente_J0_J2.png"
)
SUMMARY_DIR = (
    PROJECT_DIR
    / "fig/models/patient_sensitivity_reference_comparison/shared/j0_force_vs_pente"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _cycle_label(row: Mapping[str, Any]) -> str:
    patient = str(row["patient_id"]).upper()
    month = str(row["cycle_id"])[5:7]
    month_label = {"05": "mai", "06": "juin", "09": "sept."}.get(
        month, month
    )
    return f"{patient} — {month_label}"


def load_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference_key, spec in REFERENCE_SPECS.items():
        j0_rows = [
            row
            for row in _read_csv(Path(spec["j0_dir"]) / "sensibilites_J0.csv")
            if row["combination"] == COMBINATION
            and row["window"] == WINDOW
            and row["dataset_key"] in COMMON_KEYS
        ]
        j0_predictions = [
            row
            for row in _read_csv(
                Path(spec["j0_dir"]) / "predictions_post_J0.csv"
            )
            if row["combination"] == COMBINATION
            and row["window"] == WINDOW
            and row["dataset_key"] in COMMON_KEYS
        ]
        fit_rows = [
            row
            for row in _read_csv(
                Path(spec["fit_dir"]) / "pentes_fenetres_J0_J2.csv"
            )
            if row["combination"] == COMBINATION
            and row["window"] == WINDOW
            and row["dataset_key"] in COMMON_KEYS
        ]
        fit_predictions = [
            row
            for row in _read_csv(
                Path(spec["fit_dir"]) / "predictions_intracycle.csv"
            )
            if row["combination"] == COMBINATION
            and row["window"] == WINDOW
            and row["dataset_key"] in COMMON_KEYS
        ]
        for dataset_key in COMMON_KEYS:
            j0_row = next(row for row in j0_rows if row["dataset_key"] == dataset_key)
            fit_row = next(
                row for row in fit_rows if row["dataset_key"] == dataset_key
            )
            j0_point_rows = [
                row for row in j0_predictions if row["dataset_key"] == dataset_key
            ]
            fit_point_rows = [
                row for row in fit_predictions if row["dataset_key"] == dataset_key
            ]
            if len(j0_point_rows) != 3 or len(fit_point_rows) != 3:
                raise ValueError(
                    f"Expected three J0--J2 rows for {reference_key}/{dataset_key}"
                )
            j0_errors = np.asarray(
                [abs(float(row["relative_error_percent"])) for row in j0_point_rows],
                dtype=float,
            )
            fit_errors = np.asarray(
                [abs(float(row["relative_error_percent"])) for row in fit_point_rows],
                dtype=float,
            )
            fit_mare_all = float(np.mean(fit_errors))
            fit_mare_source = float(fit_row["mare_activity_percent"])
            if not np.isclose(fit_mare_all, fit_mare_source, atol=1e-10):
                raise ValueError(
                    "Pointwise MARE does not match fitted MARE for "
                    f"{reference_key}/{dataset_key}: "
                    f"{fit_mare_all} versus {fit_mare_source}"
                )
            rows.append(
                {
                    "reference": reference_key,
                    "reference_label": spec["label"],
                    "dataset_key": dataset_key,
                    "patient_id": j0_row["patient_id"],
                    "cycle_id": j0_row["cycle_id"],
                    "cycle_label": _cycle_label(j0_row),
                    "support": j0_row["count_support"],
                    "combination": COMBINATION,
                    "window": WINDOW,
                    "j0_sensitivity_cps_per_mbq": float(
                        j0_row["sensitivity_j0_cps_per_mbq"]
                    ),
                    "j0_forced_mare_j0_j2_percent": float(np.mean(j0_errors)),
                    "j0_forced_mare_j1_j2_percent": float(np.mean(j0_errors[1:])),
                    "j0_j2_fit_mare_j0_j2_percent": fit_mare_all,
                    "j0_j2_fit_mare_j1_j2_percent": float(
                        np.mean(fit_errors[1:])
                    ),
                    "displayed_timepoints": "Day0;Day1;Day2",
                }
            )
    return rows


def _plot_reference(rows: Sequence[Mapping[str, Any]], reference: str, path: Path) -> Path:
    selected = [row for row in rows if row["reference"] == reference]
    methods = (
        ("j0_forced_mare_j0_j2_percent", "Calibration à J0"),
        ("j0_j2_fit_mare_j0_j2_percent", "Ajustement J0–J2"),
    )
    x = np.arange(len(methods), dtype=float)
    width = 0.22
    offsets = (np.arange(len(selected)) - 1.0) * width
    figure, axis = plt.subplots(figsize=(10.0, 7.0), facecolor="white")
    for row, offset in zip(selected, offsets):
        values = [float(row[field]) for field, _ in methods]
        bars = axis.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=slopes.cycle_color(row),
            edgecolor="#222222",
            linewidth=0.9,
            label=row["cycle_label"],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.7,
                f"{value:.1f} %",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )
    axis.set_xticks(x, [label for _, label in methods])
    axis.set_ylabel("MARE sur J0–J2 (%)", fontsize=17)
    axis.set_title(
        "Erreur d’activité après calibration à J0 ou ajustement J0–J2 — "
        f"{REFERENCE_SPECS[reference]['label']}",
        fontsize=21,
        fontweight="bold",
        pad=18,
    )
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0.0, 43.0)
    axis.legend(
        title="Patient / cycle",
        title_fontsize=14,
        fontsize=13,
        frameon=False,
        loc="upper right",
    )
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.86)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_combined(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    categories = (
        ("urine", "j0_forced_mare_j0_j2_percent", "Urine\nCalibration à J0"),
        ("urine", "j0_j2_fit_mare_j0_j2_percent", "Urine\nAjustement J0–J2"),
        ("qspect", "j0_forced_mare_j0_j2_percent", "Q/SPECT\nCalibration à J0"),
        ("qspect", "j0_j2_fit_mare_j0_j2_percent", "Q/SPECT\nAjustement J0–J2"),
    )
    centers = np.asarray([0.0, 1.05, 2.65, 3.70])
    width = 0.22
    figure, axis = plt.subplots(figsize=(14.5, 7.3), facecolor="white")
    patient_rows = [
        next(
            row
            for row in rows
            if row["reference"] == "urine" and row["dataset_key"] == dataset_key
        )
        for dataset_key in COMMON_KEYS
    ]
    offsets = (np.arange(len(patient_rows)) - 1.0) * width
    for patient_row, offset in zip(patient_rows, offsets):
        dataset_key = patient_row["dataset_key"]
        values = []
        for reference, field, _ in categories:
            row = next(
                item
                for item in rows
                if item["reference"] == reference
                and item["dataset_key"] == dataset_key
            )
            values.append(float(row[field]))
        bars = axis.bar(
            centers + offset,
            values,
            width=width * 0.92,
            color=slopes.cycle_color(patient_row),
            edgecolor="#222222",
            linewidth=0.9,
            label=patient_row["cycle_label"],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.65,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
            )
    for center, (reference, field, _) in zip(centers, categories):
        mean_value = float(
            np.mean(
                [
                    float(row[field])
                    for row in rows
                    if row["reference"] == reference
                ]
            )
        )
        axis.text(
            center,
            40.0,
            f"Moyenne : {mean_value:.1f} %",
            ha="center",
            va="bottom",
            fontsize=13.5,
            fontweight="bold",
        )
    axis.set_xticks(centers, [label for _, _, label in categories])
    axis.set_ylabel("MARE sur J0–J2 (%)", fontsize=17)
    axis.set_title(
        "Erreur d’activité après calibration à J0 ou ajustement J0–J2",
        fontsize=22,
        fontweight="bold",
        pad=18,
    )
    axis.tick_params(axis="both", labelsize=14)
    axis.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlim(-0.65, 4.35)
    axis.set_ylim(0.0, 44.0)
    axis.legend(
        title="Patient / cycle",
        title_fontsize=13,
        fontsize=12.5,
        frameon=False,
        loc="center right",
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.86)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def run() -> dict[str, Path]:
    rows = load_metrics()
    outputs = {
        "urine": _plot_reference(
            rows,
            "urine",
            Path(REFERENCE_SPECS["urine"]["output_dir"])
            / "MARE_J1_J2_J0_force_vs_pente.png",
        ),
        "qspect": _plot_reference(
            rows,
            "qspect",
            Path(REFERENCE_SPECS["qspect"]["output_dir"])
            / "MARE_J1_J2_J0_force_vs_pente.png",
        ),
        "combined": plot_combined(rows, PRESENTATION_PATH),
        "summary_figure": plot_combined(
            rows, SUMMARY_DIR / "MARE_J1_J2_J0_force_vs_pente.png"
        ),
        "metrics": _write_csv(SUMMARY_DIR / "metrics.csv", rows),
    }
    readme = SUMMARY_DIR / "README.txt"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        """COMPARAISON J0 FORCE VERSUS PENTE J0--J2\n"
        "=========================================\n\n"
        "Signal planaire commun: photopeak brut et moyenne geometrique globale.\n"
        "Urine: image complete. Q/SPECT: crop correspondant.\n\n"
        "Calibration J0: S=R_J0/A_J0; l'erreur de J0 est donc nulle.\n"
        "Ajustement J0--J2: S est ajustee a l'origine avec J0, J1 et J2.\n"
        "La figure compare la meme MARE sur J0--J2 pour les deux methodes, comme\n"
        "la figure de comparaison urine/QSPECT. Les colonnes J1--J2 restent dans\n"
        "metrics.csv pour verifier la performance aux jours suivant J0.\n"
        """,
        encoding="utf-8",
    )
    outputs["readme"] = readme
    return outputs


if __name__ == "__main__":
    for name, path in run().items():
        print(f"{name}: {path}")
