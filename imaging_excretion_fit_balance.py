"""Compare planar and Q/SPECT activity closure against measured excretion.

This module applies the same independent bi-exponential time-alignment method
to both imaging modalities.  Detailed figures are written to separate Planar
and QSPECT subfolders, while a single grouped comparison is produced for
presentation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import planar_excretion_fit_balance as pefb
import sang_urine


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "fig" / "imaging_excretion_fit_balance"
PLANAR_DIR = OUTPUT_DIR / "Planar"
QSPECT_DIR = OUTPUT_DIR / "QSPECT"
COMPARISON_FIGURE = OUTPUT_DIR / "planar_qspect_recovered_injection.png"
COMPARISON_VALUES = OUTPUT_DIR / "planar_qspect_recovered_injection_values.csv"
COMPARISON_REPORT = OUTPUT_DIR / "planar_qspect_recovered_injection_report.txt"


MODALITIES = {
    "planar": {
        "display": "Planaire",
        "folder": "Planar",
        "time_key": "planar_times_h",
        "activity_key": "planar_ctac_mbq",
        "color": "#1f77b4",
        "marker": "s",
    },
    "qspect": {
        "display": "Q/SPECT",
        "folder": "QSPECT",
        "time_key": "qspect_times_h",
        "activity_key": "qspect_mbq",
        "color": "#111111",
        "marker": "D",
    },
}


def build_modality_imaging(
    imaging: Dict[str, np.ndarray], modality: str
) -> Dict[str, np.ndarray]:
    """Map a modality to the interface used by the existing fit functions."""
    if modality not in MODALITIES:
        raise ValueError(f"Unknown imaging modality: {modality}")
    configuration = MODALITIES[modality]
    return {
        "labels": np.asarray(imaging["labels"]),
        "planar_times_h": np.asarray(imaging[configuration["time_key"]], dtype=float),
        "planar_ctac_mbq": np.asarray(
            imaging[configuration["activity_key"]], dtype=float
        ),
    }


def calculate_modality_balance(
    data: Dict[str, Any],
    balance: Dict[str, np.ndarray],
    imaging: Dict[str, np.ndarray],
    modality: str,
) -> Dict[str, Any]:
    """Fit one imaging modality and calculate both reciprocal comparisons."""
    modality_imaging = build_modality_imaging(imaging, modality)
    fits = pefb.fit_planar_and_excretion(data, balance, modality_imaging)
    cross_balances = pefb.calculate_cross_time_balances(fits)
    for row in cross_balances["planar_rows"]:
        row["comparison"] = f"measured_{modality}_plus_fitted_excretion"
    for row in cross_balances["excretion_rows"]:
        row["comparison"] = f"measured_excretion_plus_fitted_{modality}"
    return {
        "modality": modality,
        "configuration": MODALITIES[modality],
        "imaging": modality_imaging,
        "fits": fits,
        "cross_balances": cross_balances,
    }


def _write_modality_outputs(
    result: Dict[str, Any], output_dir: Path
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    display = str(result["configuration"]["display"])
    fits = result["fits"]
    cross_balances = result["cross_balances"]
    return {
        "measured_imaging_plus_excretion_fit": (
            pefb.plot_measured_imaging_plus_fitted_excretion(
                cross_balances,
                output_dir / "01_measured_imaging_plus_excretion_fit.png",
                modality_label=display,
            )
        ),
        "measured_excretion_plus_imaging_fit": (
            pefb.plot_measured_excretion_plus_fitted_imaging(
                cross_balances,
                output_dir / "02_measured_excretion_plus_imaging_fit.png",
                modality_label=display,
            )
        ),
        "continuous_sum": pefb.plot_sum_of_biexponential_fits(
            fits,
            output_dir / "03_sum_of_imaging_and_excretion_fits.png",
            modality_label=display,
        ),
        "values": pefb.write_values(
            cross_balances, output_dir / "fit_balance_values.csv"
        ),
        "report": pefb.write_report(
            fits,
            cross_balances,
            output_dir / "fit_balance_report.txt",
            modality_label=display,
        ),
    }


def comparison_rows(
    planar_result: Dict[str, Any], qspect_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Flatten the direct imaging-plus-excretion comparison for both modalities."""
    rows: List[Dict[str, Any]] = []
    for result in (planar_result, qspect_result):
        display = str(result["configuration"]["display"])
        color = str(result["configuration"]["color"])
        measured_actual_mbq = np.asarray(
            result["fits"]["planar_measured_mbq"], dtype=float
        )
        for index, source in enumerate(result["cross_balances"]["planar_rows"]):
            rows.append(
                {
                    "modality": display,
                    "color": color,
                    "label": source["label"],
                    "elapsed_h": source["elapsed_h"],
                    "imaging_activity_at_acquisition_mbq": float(
                        measured_actual_mbq[index]
                    ),
                    "imaging_injection_equivalent_mbq": (
                        source["measured_component_injection_equivalent_mbq"]
                    ),
                    "fitted_excretion_injection_equivalent_mbq": (
                        source["fitted_component_injection_equivalent_mbq"]
                    ),
                    "recovered_injection_equivalent_mbq": (
                        source["sum_injection_equivalent_mbq"]
                    ),
                    "closure_ratio": source["closure_ratio"],
                    "closure_error_percent": source["closure_error_percent"],
                    "excretion_fit_extrapolated": source[
                        "fit_is_extrapolated_after_last_collection"
                    ],
                }
            )
    return rows


def plot_planar_qspect_comparison(
    rows: List[Dict[str, Any]],
    injected_mbq: float,
    output_path: Path = COMPARISON_FIGURE,
) -> Path:
    """Create the compact presentation figure comparing recovered injection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(dict.fromkeys(str(row["label"]) for row in rows))
    positions = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, axis = plt.subplots(figsize=(9.4, 5.8), facecolor="white", layout="constrained")
    offsets = {"Planaire": -width / 2.0, "Q/SPECT": width / 2.0}
    hatches_used = False
    for modality in ("Planaire", "Q/SPECT"):
        modality_rows = [row for row in rows if row["modality"] == modality]
        body_gbq = np.asarray(
            [row["imaging_injection_equivalent_mbq"] for row in modality_rows]
        ) / 1000.0
        excretion_gbq = np.asarray(
            [
                row["fitted_excretion_injection_equivalent_mbq"]
                for row in modality_rows
            ]
        ) / 1000.0
        total_gbq = body_gbq + excretion_gbq
        body_bars = axis.bar(
            positions + offsets[modality],
            body_gbq,
            width=width,
            color=modality_rows[0]["color"],
            label=f"Activité corporelle mesurée — {modality}",
            zorder=3,
        )
        excretion_bars = axis.bar(
            positions + offsets[modality],
            excretion_gbq,
            width=width,
            bottom=body_gbq,
            color="#ff7f0e",
            label="Excrétion cumulative ajustée"
            if modality == "Planaire"
            else None,
            zorder=3,
        )
        for body_bar, excretion_bar, row, total in zip(
            body_bars, excretion_bars, modality_rows, total_gbq
        ):
            if row["excretion_fit_extrapolated"]:
                excretion_bar.set_hatch("//")
                excretion_bar.set_edgecolor("#9a4f00")
                hatches_used = True
            axis.text(
                body_bar.get_x() + body_bar.get_width() / 2.0,
                total + 0.055,
                f"{row['closure_error_percent']:+.1f} %",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.axhline(
        injected_mbq / 1000.0,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Activité injectée ({injected_mbq / 1000.0:.1f} GBq)",
        zorder=4,
    )
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Temps d'imagerie")
    axis.set_ylabel("Activité retrouvée, équivalente à l'injection (GBq)")
    axis.set_title("Contributions au bilan : activité corporelle + excrétion")
    axis.set_ylim(0.0, max(row["recovered_injection_equivalent_mbq"] for row in rows) / 1000.0 * 1.14)
    axis.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    handles, legend_labels = axis.get_legend_handles_labels()
    if hatches_used:
        handles.append(
            Patch(
                facecolor="#ff7f0e",
                edgecolor="#9a4f00",
                hatch="//",
                label="Extrapolation après la dernière collecte (46,62 h)",
            )
        )
        legend_labels.append("Extrapolation après la dernière collecte (46,62 h)")
    axis.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_comparison_values(
    rows: List[Dict[str, Any]], output_path: Path = COMPARISON_VALUES
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [name for name in rows[0] if name != "color"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: round(float(value), 6)
                    if isinstance(value, (float, np.floating))
                    else value
                    for key, value in row.items()
                    if key in fieldnames
                }
            )
    return output_path


def write_comparison_report(
    rows: List[Dict[str, Any]], output_path: Path = COMPARISON_REPORT
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Planar and Q/SPECT comparison using the same excretion fit",
        "==========================================================",
        "",
        "All imaging activities and the fitted cumulative excretion are placed",
        "on an injection-time-equivalent scale before addition. The excretion",
        "fit is identical for both modalities and is evaluated at each exact",
        "acquisition time.",
        "",
        " modality | label | time h | recovered MBq | error % | extrapolated",
        "--------- | ----- | ------ | ------------- | ------- | ------------",
    ]
    for row in rows:
        lines.append(
            f"{row['modality']:9s} | {row['label']:5s} | {row['elapsed_h']:6.2f} | "
            f"{row['recovered_injection_equivalent_mbq']:13.2f} | "
            f"{row['closure_error_percent']:+7.2f} | "
            f"{str(row['excretion_fit_extrapolated']):>12s}"
        )
    lines.append("")
    for modality in ("Planaire", "Q/SPECT"):
        errors = np.asarray(
            [
                row["closure_error_percent"]
                for row in rows
                if row["modality"] == modality
            ],
            dtype=float,
        )
        lines.append(
            f"{modality}: mean absolute closure error = "
            f"{np.mean(np.abs(errors)):.2f} %, maximum = {np.max(np.abs(errors)):.2f} %."
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "  The direct measured-imaging plus fitted-excretion comparison is the",
            "  preferred presentation because only excretion is fitted for temporal",
            "  alignment. The reciprocal and two-fit figures remain diagnostics.",
            "  Hatched Day2-Day6 bars depend on excretion extrapolation beyond the",
            "  final 46.62 h collection endpoint.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_imaging_excretion_fit_balance(
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    """Generate Planar, QSPECT, and direct-comparison output sections."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = sang_urine.load_sang_urine_csv()
    balance = sang_urine.calculate_urine_mass_balance(data)
    imaging = sang_urine.load_imaging_comparison(data["injection_datetime"])
    if imaging is None:
        raise FileNotFoundError("The profile-crop planar/QSPECT data are unavailable")

    planar = calculate_modality_balance(data, balance, imaging, "planar")
    qspect = calculate_modality_balance(data, balance, imaging, "qspect")
    planar["paths"] = _write_modality_outputs(planar, output_dir / "Planar")
    qspect["paths"] = _write_modality_outputs(qspect, output_dir / "QSPECT")
    rows = comparison_rows(planar, qspect)
    comparison_paths = {
        "figure": plot_planar_qspect_comparison(
            rows,
            float(data["injected_activity_mbq"]),
            output_dir / COMPARISON_FIGURE.name,
        ),
        "values": write_comparison_values(
            rows, output_dir / COMPARISON_VALUES.name
        ),
        "report": write_comparison_report(
            rows, output_dir / COMPARISON_REPORT.name
        ),
    }
    return {
        "data": data,
        "balance": balance,
        "imaging": imaging,
        "planar": planar,
        "qspect": qspect,
        "comparison_rows": rows,
        "comparison_paths": comparison_paths,
    }


def main() -> None:
    result = run_imaging_excretion_fit_balance()
    for modality_key in ("planar", "qspect"):
        modality_result = result[modality_key]
        print(f"{modality_result['configuration']['display']}:")
        for row in modality_result["cross_balances"]["planar_rows"]:
            print(
                f"  {row['label']}: {row['sum_injection_equivalent_mbq']:.2f} MBq "
                f"({row['closure_error_percent']:+.2f} %)"
            )
    print("Outputs:")
    for key, path in result["comparison_paths"].items():
        print(f"  comparison_{key}: {path}")
    for modality_key in ("planar", "qspect"):
        for key, path in result[modality_key]["paths"].items():
            print(f"  {modality_key}_{key}: {path}")


if __name__ == "__main__":
    main()
