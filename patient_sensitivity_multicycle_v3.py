"""Grouped multi-cycle validation of planar effective-sensitivity models.

V2 is intentionally left unchanged.  This module generalizes the locked
planar spectral model to explicit lists of training and test cycles.  A cycle
can never occur in both groups.  The default run evaluates:

* every directed single-cycle transfer;
* every train-two-cycles/test-one-cycle split;
* the reverse patient-level transfer, P8 -> both P11 cycles.

Seven measurement-rate definitions are evaluated: TEW and raw photopeak with
crop+GM, full-image+GM, or full-image+AP+PA, plus an exploratory full-image
AP+PA rate summing all four windows.  Every predictor is recomputed on the
spatial support and AP/PA combination selected by its rate mode.
The Q/SPECT activity is used only to construct the training target and to score
held-out predictions.  No CT voxel value is a model input.  The current profile
crop is still Q/SPECT-guided and is therefore a development-stage preprocessing
step rather than a final planar-only deployment solution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import attenuation_correction as ac
import attenuation_global_model_v2 as spectral_model
import figure_layout
import patient_effective_sensitivity as sensitivity
import patient_sensitivity_feature_test_v2 as v2
import planar_processing


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = figure_layout.shared_dir(__file__)

LOCKED_FEATURE = spectral_model.BROAD_FEATURE_NAME
ASYMMETRY_FEATURE = "abs AP/PA photopeak asymmetry"
LOWER_FEATURE = "log lower/photopeak"
UPPER_FEATURE = "log upper/photopeak"
TEW_FRACTION_FEATURE = "TEW retained fraction"
RAW_PHOTO_GENERAL_FEATURE = "log raw full-image photopeak/general scatter"
ADJACENT_FEATURE = "log raw adjacent scatter/photopeak"
PHOTO_ALL_FRACTION_FEATURE = "raw photopeak fraction of all four windows"
ALL_WINDOW_ASYMMETRY_FEATURE = "abs AP/PA all-window asymmetry"
RAW_FULL_PHOTOPEAK_RATE = "raw full-image AP+PA photopeak rate cps"
RAW_FULL_ALL_WINDOWS_RATE = "raw full-image AP+PA four-window rate cps"
TEW_FULL_GM_RATE = "TEW full-image geometric-mean rate cps"
RAW_FULL_GM_RATE = "raw full-image geometric-mean photopeak rate cps"
TEW_FULL_AP_PA_RATE = "TEW full-image AP+PA rate cps"

FEATURE_ORDER = (
    LOCKED_FEATURE,
    RAW_PHOTO_GENERAL_FEATURE,
    ADJACENT_FEATURE,
    PHOTO_ALL_FRACTION_FEATURE,
    TEW_FRACTION_FEATURE,
    LOWER_FEATURE,
    UPPER_FEATURE,
    ASYMMETRY_FEATURE,
    ALL_WINDOW_ASYMMETRY_FEATURE,
)
FEATURE_LABELS = {
    LOCKED_FEATURE: "General/photopeak\n(adapté au mode)",
    RAW_PHOTO_GENERAL_FEATURE: "Photopeak/general\n(adapté au mode)",
    ADJACENT_FEATURE: "Diffusé adjacent/photopeak\n(adapté au mode)",
    PHOTO_ALL_FRACTION_FEATURE: "Fraction photopeak/4 fenêtres\n(adaptée au mode)",
    TEW_FRACTION_FEATURE: "Fraction conservée après TEW\n(adaptée au mode)",
    LOWER_FEATURE: "Fenêtre inférieure/photopeak\n(adapté au mode)",
    UPPER_FEATURE: "Fenêtre supérieure/photopeak\n(adapté au mode)",
    ASYMMETRY_FEATURE: "Asymétrie AP/PA du photopeak\n(même support)",
    ALL_WINDOW_ASYMMETRY_FEATURE: "Asymétrie AP/PA des 4 fenêtres\n(même support)",
}

MODEL_CONSTANT = "M0_constant"
MODEL_LOCKED = "M1_locked_spectral"
MODEL_ASYMMETRY = "M2_spectral_plus_ap_pa"
MODEL_ADJACENT = "M4_adjacent_scatter"
MODEL_TEW_FRACTION = "M5_tew_fraction"
MODEL_PHOTO_ALL = "M6_photo_all_fraction"
MODEL_LOWER = "M7_lower_scatter"
MODEL_UPPER = "M8_upper_scatter"
MODEL_RAW_ASYMMETRY = "M9_raw_ratio_plus_all_ap_pa"
MODEL_ORDER = (
    MODEL_CONSTANT,
    MODEL_LOCKED,
    MODEL_ASYMMETRY,
    MODEL_ADJACENT,
    MODEL_TEW_FRACTION,
    MODEL_PHOTO_ALL,
    MODEL_LOWER,
    MODEL_UPPER,
    MODEL_RAW_ASYMMETRY,
)
MODEL_LABELS = {
    MODEL_CONSTANT: "M0 — sensibilité constante",
    MODEL_LOCKED: "M1 — ratio spectral verrouillé",
    MODEL_ASYMMETRY: "M2 — ratio spectral + asymétrie AP/PA",
    MODEL_ADJACENT: "M4 — diffusé adjacent/photopeak",
    MODEL_TEW_FRACTION: "M5 — fraction conservée après TEW",
    MODEL_PHOTO_ALL: "M6 — fraction photopeak/4 fenêtres",
    MODEL_LOWER: "M7 — fenêtre inférieure/photopeak",
    MODEL_UPPER: "M8 — fenêtre supérieure/photopeak",
    MODEL_RAW_ASYMMETRY: "M9 — ratio brut + asymétrie totale AP/PA",
}
MODEL_COLORS = {
    MODEL_CONSTANT: "#7A7A7A",
    MODEL_LOCKED: "#2E9F55",
    MODEL_ASYMMETRY: "#8C6BB1",
    MODEL_ADJACENT: "#FF7F0E",
    MODEL_TEW_FRACTION: "#D62728",
    MODEL_PHOTO_ALL: "#17BECF",
    MODEL_LOWER: "#BCBD22",
    MODEL_UPPER: "#E377C2",
    MODEL_RAW_ASYMMETRY: "#D62728",
}
MODEL_MARKERS = {
    MODEL_CONSTANT: "o",
    MODEL_LOCKED: "D",
    MODEL_ASYMMETRY: "s",
    MODEL_ADJACENT: "v",
    MODEL_TEW_FRACTION: "P",
    MODEL_PHOTO_ALL: "X",
    MODEL_LOWER: "<",
    MODEL_UPPER: ">",
    MODEL_RAW_ASYMMETRY: "h",
}

PRIMARY_PLOT_MODELS = (
    MODEL_CONSTANT,
    MODEL_LOCKED,
    MODEL_ASYMMETRY,
    MODEL_ADJACENT,
    MODEL_PHOTO_ALL,
    MODEL_LOWER,
    MODEL_UPPER,
    MODEL_RAW_ASYMMETRY,
)

MODE_TEW_CROP_GM = "tew_crop_gm"
MODE_RAW_CROP_GM = "raw_photopeak_crop_gm"
MODE_TEW_FULL_GM = "tew_full_gm"
MODE_RAW_FULL_GM = "raw_photopeak_full_gm"
MODE_TEW_FULL_AP_PA = "tew_full_ap_pa"
MODE_RAW_FULL_AP_PA = "raw_photopeak_full_ap_pa"
MODE_ALL_FOUR_FULL_AP_PA = "all_four_full_ap_pa"
GEOMETRY_CROP_GM = "crop_gm"
GEOMETRY_FULL_GM = "full_gm"
GEOMETRY_FULL_AP_PA = "full_ap_pa"
CORE_COMPARISON_MODES = (
    MODE_TEW_CROP_GM,
    MODE_RAW_CROP_GM,
    MODE_RAW_FULL_AP_PA,
)
FACTORIAL_COMPARISON_MODES = (
    MODE_TEW_CROP_GM,
    MODE_TEW_FULL_GM,
    MODE_TEW_FULL_AP_PA,
    MODE_RAW_CROP_GM,
    MODE_RAW_FULL_GM,
    MODE_RAW_FULL_AP_PA,
)


@dataclass(frozen=True)
class RateModeSpec:
    key: str
    label: str
    short_label: str
    description: str
    feature_geometry: str
    model_names: tuple[str, ...]


MODELS_WITHOUT_TEW_INPUT = tuple(
    model for model in MODEL_ORDER if model != MODEL_TEW_FRACTION
)
COMMON_MODE_MODELS = MODELS_WITHOUT_TEW_INPUT

RATE_MODES: Dict[str, RateModeSpec] = {
    MODE_TEW_CROP_GM: RateModeSpec(
        key=MODE_TEW_CROP_GM,
        label="TEW — crop et moyenne géométrique",
        short_label="TEW crop GM",
        description=(
            "Taux TEW calculé dans le crop après correction AP et PA séparée, "
            "puis moyenne géométrique."
        ),
        feature_geometry=GEOMETRY_CROP_GM,
        model_names=MODEL_ORDER,
    ),
    MODE_RAW_CROP_GM: RateModeSpec(
        key=MODE_RAW_CROP_GM,
        label="Photopeak brut — même crop et moyenne géométrique",
        short_label="Photopeak brut crop GM",
        description=(
            "Photopeak brut dans exactement le même crop et avec la même "
            "moyenne géométrique que le mode TEW."
        ),
        feature_geometry=GEOMETRY_CROP_GM,
        model_names=MODELS_WITHOUT_TEW_INPUT,
    ),
    MODE_TEW_FULL_GM: RateModeSpec(
        key=MODE_TEW_FULL_GM,
        label="TEW — image complète et moyenne géométrique",
        short_label="TEW complet GM",
        description=(
            "Taux TEW sur l'image complète après correction AP et PA "
            "séparée, puis moyenne géométrique; aucun crop."
        ),
        feature_geometry=GEOMETRY_FULL_GM,
        model_names=MODEL_ORDER,
    ),
    MODE_RAW_FULL_GM: RateModeSpec(
        key=MODE_RAW_FULL_GM,
        label="Photopeak brut — image complète et moyenne géométrique",
        short_label="Photopeak complet GM",
        description=(
            "Photopeak brut sur l'image complète, combiné par moyenne "
            "géométrique; aucun crop et aucune TEW."
        ),
        feature_geometry=GEOMETRY_FULL_GM,
        model_names=MODELS_WITHOUT_TEW_INPUT,
    ),
    MODE_TEW_FULL_AP_PA: RateModeSpec(
        key=MODE_TEW_FULL_AP_PA,
        label="TEW — image complète AP+PA",
        short_label="TEW complet AP+PA",
        description=(
            "Somme AP+PA du photopeak corrigé par TEW sur l'image complète; "
            "aucun crop et aucune moyenne géométrique."
        ),
        feature_geometry=GEOMETRY_FULL_AP_PA,
        model_names=MODEL_ORDER,
    ),
    MODE_RAW_FULL_AP_PA: RateModeSpec(
        key=MODE_RAW_FULL_AP_PA,
        label="Photopeak brut — image complète AP+PA",
        short_label="Photopeak complet AP+PA",
        description=(
            "Somme brute AP+PA du photopeak sur l'image complète, sans crop, "
            "sans moyenne géométrique et sans TEW."
        ),
        feature_geometry=GEOMETRY_FULL_AP_PA,
        model_names=MODELS_WITHOUT_TEW_INPUT,
    ),
    MODE_ALL_FOUR_FULL_AP_PA: RateModeSpec(
        key=MODE_ALL_FOUR_FULL_AP_PA,
        label="Quatre fenêtres — image complète AP+PA",
        short_label="4 fenêtres AP+PA",
        description=(
            "Somme brute AP+PA des quatre fenêtres sur l'image complète; "
            "analyse exploratoire."
        ),
        feature_geometry=GEOMETRY_FULL_AP_PA,
        model_names=MODELS_WITHOUT_TEW_INPUT,
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    patient_id: str
    cycle_id: str
    label: str
    short_label: str
    planar_dir: Path
    qspect_dir: Path


@dataclass(frozen=True)
class SplitDefinition:
    split_id: str
    split_type: str
    training_datasets: tuple[str, ...]
    test_datasets: tuple[str, ...]
    label: str


def _configured_spec(
    key: str,
    patient_id: str,
    cycle_id: str,
    short_label: str,
) -> DatasetSpec:
    configured = v2.CYCLE_DATASETS[(patient_id, cycle_id)]
    return DatasetSpec(
        key=key,
        patient_id=patient_id,
        cycle_id=cycle_id,
        label=str(configured["patient_label"]),
        short_label=short_label,
        planar_dir=Path(configured["planar_dir"]),
        qspect_dir=Path(configured["qspect_dir"]),
    )


DATASETS: Dict[str, DatasetSpec] = {
    "p11_may": _configured_spec(
        "p11_may", "p11", "2026-05__Studies", "P11-mai"
    ),
    "p11_july": _configured_spec(
        "p11_july", "p11", "2026-07__Studies", "P11-juillet"
    ),
    "p8_june": _configured_spec(
        "p8_june", "p8", "2026-06__Studies", "P8-juin"
    ),
}


def default_split_definitions() -> list[SplitDefinition]:
    """Return the non-random grouped transfers used by the v3 audit."""
    keys = tuple(DATASETS)
    splits: list[SplitDefinition] = []
    for training_key in keys:
        for test_key in keys:
            if training_key == test_key:
                continue
            splits.append(
                SplitDefinition(
                    split_id=f"pair_{training_key}_to_{test_key}",
                    split_type="pairwise_cycle_transfer",
                    training_datasets=(training_key,),
                    test_datasets=(test_key,),
                    label=(
                        f"{DATASETS[training_key].short_label} → "
                        f"{DATASETS[test_key].short_label}"
                    ),
                )
            )

    for test_key in keys:
        training_keys = tuple(key for key in keys if key != test_key)
        split_type = "leave_one_cycle_out"
        if test_key == "p8_june":
            split_type += "+patient_holdout"
        splits.append(
            SplitDefinition(
                split_id=f"loco_holdout_{test_key}",
                split_type=split_type,
                training_datasets=training_keys,
                test_datasets=(test_key,),
                label=(
                    " + ".join(DATASETS[key].short_label for key in training_keys)
                    + f" → {DATASETS[test_key].short_label}"
                ),
            )
        )

    splits.append(
        SplitDefinition(
            split_id="patient_holdout_p8_to_p11_all",
            split_type="patient_holdout_reverse",
            training_datasets=("p8_june",),
            test_datasets=("p11_may", "p11_july"),
            label="P8-juin → P11-mai + P11-juillet",
        )
    )
    return splits


def _validate_dataset_keys(keys: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(key) for key in keys)
    if not normalized:
        raise ValueError("At least one dataset is required")
    unknown = sorted(set(normalized) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset key(s): {', '.join(unknown)}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Dataset lists must not contain duplicates")
    return normalized


def validate_split(
    training_datasets: Sequence[str], test_datasets: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate and return disjoint training and test dataset keys."""
    training = _validate_dataset_keys(training_datasets)
    test = _validate_dataset_keys(test_datasets)
    overlap = sorted(set(training) & set(test))
    if overlap:
        raise ValueError(
            "Training and test datasets must be disjoint; overlap: "
            + ", ".join(overlap)
        )
    return training, test


def _positive_image(image: Any) -> np.ndarray:
    return np.clip(
        np.nan_to_num(
            np.asarray(image, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        None,
    )


def mode_matched_measurements(
    scan: Mapping[str, Any], crop_bounds: tuple[int, int]
) -> Dict[str, Any]:
    """Build rates and predictors on support matched to each mode geometry."""
    groups = ac.group_by_energy_and_view(scan["images"])
    energies = tuple(planar_processing.ENERGY_ORDER)
    widths = planar_processing.window_widths(scan["images"])
    missing = [
        f"{energy} {view}"
        for energy in energies
        for view in ("AP", "PA")
        if energy not in groups
        or view not in groups[energy]
        or energy not in widths
    ]
    if missing:
        raise ValueError(
            "Mode-matched features require every AP/PA window and width; missing: "
            + ", ".join(missing)
        )
    duration_seconds = float(
        planar_processing.acquisition_duration_seconds(scan["images"])
    )
    if duration_seconds <= 0.0:
        raise ValueError("A positive planar frame duration is required")

    images = {
        energy: {
            view: _positive_image(groups[energy][view]["image"])
            for view in ("AP", "PA")
        }
        for energy in energies
    }
    reference_shape = images["Photopeak"]["AP"].shape
    if any(
        images[energy][view].shape != reference_shape
        for energy in energies
        for view in ("AP", "PA")
    ):
        raise ValueError("All planar energy-window images must share one shape")
    crop_top, crop_bottom = (int(crop_bounds[0]), int(crop_bounds[1]))
    if crop_top < 0 or crop_bottom <= crop_top or crop_bottom > reference_shape[0]:
        raise ValueError(
            f"Crop {crop_bounds} is invalid for planar shape {reference_shape}"
        )
    crop_slice = (slice(crop_top, crop_bottom), slice(None))
    full_slice = (slice(None), slice(None))

    gm_images = {
        energy: ac.geometric_mean(
            images[energy]["AP"], images[energy]["PA"]
        )
        for energy in energies
    }
    tew_ap = ac.tew_correct_view_image(groups, "AP")
    tew_pa = ac.tew_correct_view_image(groups, "PA")
    raw_photo_gm = gm_images["Photopeak"]
    tew_gm = ac.geometric_mean(tew_ap, tew_pa)

    def support_slice(geometry: str) -> tuple[slice, slice]:
        return crop_slice if geometry == GEOMETRY_CROP_GM else full_slice

    def spectral_density(energy: str, geometry: str) -> float:
        width = float(widths[energy])
        if width <= 0.0:
            raise ValueError(f"Invalid width for {energy}: {width}")
        region = support_slice(geometry)
        if geometry in (GEOMETRY_CROP_GM, GEOMETRY_FULL_GM):
            counts = float(np.sum(gm_images[energy][region]))
        elif geometry == GEOMETRY_FULL_AP_PA:
            counts = float(
                np.sum(images[energy]["AP"][region])
                + np.sum(images[energy]["PA"][region])
            )
        else:
            raise ValueError(f"Unknown feature geometry: {geometry}")
        return counts / (width * duration_seconds)

    def view_density(energy: str, geometry: str, view: str) -> float:
        region = support_slice(geometry)
        return float(np.sum(images[energy][view][region])) / (
            float(widths[energy]) * duration_seconds
        )

    raw_rates = {
        GEOMETRY_CROP_GM: float(np.sum(raw_photo_gm[crop_slice]))
        / duration_seconds,
        GEOMETRY_FULL_GM: float(np.sum(raw_photo_gm)) / duration_seconds,
        GEOMETRY_FULL_AP_PA: float(
            np.sum(images["Photopeak"]["AP"])
            + np.sum(images["Photopeak"]["PA"])
        )
        / duration_seconds,
    }
    tew_rates = {
        GEOMETRY_CROP_GM: float(np.sum(tew_gm[crop_slice])) / duration_seconds,
        GEOMETRY_FULL_GM: float(np.sum(tew_gm)) / duration_seconds,
        GEOMETRY_FULL_AP_PA: float(np.sum(tew_ap) + np.sum(tew_pa))
        / duration_seconds,
    }

    features_by_geometry: Dict[str, Dict[str, float]] = {}
    for geometry in (
        GEOMETRY_CROP_GM,
        GEOMETRY_FULL_GM,
        GEOMETRY_FULL_AP_PA,
    ):
        density = {
            energy: spectral_density(energy, geometry) for energy in energies
        }
        photopeak = density["Photopeak"]
        general = density["Low Energy Scatter"]
        adjacent = density["Lower Scatter"] + density["Upper Scatter"]
        all_density = float(sum(density.values()))
        all_ap = float(
            sum(view_density(energy, geometry, "AP") for energy in energies)
        )
        all_pa = float(
            sum(view_density(energy, geometry, "PA") for energy in energies)
        )
        photo_ap = view_density("Photopeak", geometry, "AP")
        photo_pa = view_density("Photopeak", geometry, "PA")
        # Numerical protection only: it must not measurably alter non-zero
        # spectral ratios.
        epsilon = max(1e-12, 1e-12 * abs(photopeak))
        features_by_geometry[geometry] = {
            LOCKED_FEATURE: float(
                np.log((general + epsilon) / (photopeak + epsilon))
            ),
            RAW_PHOTO_GENERAL_FEATURE: float(
                np.log((photopeak + epsilon) / (general + epsilon))
            ),
            ADJACENT_FEATURE: float(
                np.log((adjacent + epsilon) / (photopeak + epsilon))
            ),
            PHOTO_ALL_FRACTION_FEATURE: float(
                (photopeak + epsilon) / (all_density + 4.0 * epsilon)
            ),
            TEW_FRACTION_FEATURE: float(
                tew_rates[geometry] / raw_rates[geometry]
            ),
            LOWER_FEATURE: float(
                np.log(
                    (density["Lower Scatter"] + epsilon)
                    / (photopeak + epsilon)
                )
            ),
            UPPER_FEATURE: float(
                np.log(
                    (density["Upper Scatter"] + epsilon)
                    / (photopeak + epsilon)
                )
            ),
            ASYMMETRY_FEATURE: float(
                abs(0.5 * np.log((photo_ap + epsilon) / (photo_pa + epsilon)))
            ),
            ALL_WINDOW_ASYMMETRY_FEATURE: float(
                abs(0.5 * np.log((all_ap + epsilon) / (all_pa + epsilon)))
            ),
        }

    all_four_full_ap_pa_rate = float(
        sum(
            np.sum(images[energy]["AP"]) + np.sum(images[energy]["PA"])
            for energy in energies
        )
    ) / duration_seconds
    return {
        "duration_seconds": duration_seconds,
        "features_by_geometry": features_by_geometry,
        "raw_rates": raw_rates,
        "tew_rates": tew_rates,
        "all_four_full_ap_pa_rate": all_four_full_ap_pa_rate,
    }


def raw_full_image_features(scan: Mapping[str, Any]) -> Dict[str, float]:
    """Backward-compatible full-image AP+PA feature/rate summary."""
    groups = ac.group_by_energy_and_view(scan["images"])
    photopeak = groups.get("Photopeak", {})
    if "AP" not in photopeak:
        raise ValueError("Photopeak AP image is required")
    rows = int(np.asarray(photopeak["AP"]["image"]).shape[0])
    matched = mode_matched_measurements(scan, (0, rows))
    features = matched["features_by_geometry"][GEOMETRY_FULL_AP_PA]
    return {
        **features,
        RAW_FULL_PHOTOPEAK_RATE: matched["raw_rates"][GEOMETRY_FULL_AP_PA],
        RAW_FULL_ALL_WINDOWS_RATE: matched["all_four_full_ap_pa_rate"],
        TEW_FULL_GM_RATE: matched["tew_rates"][GEOMETRY_FULL_GM],
        RAW_FULL_GM_RATE: matched["raw_rates"][GEOMETRY_FULL_GM],
        TEW_FULL_AP_PA_RATE: matched["tew_rates"][GEOMETRY_FULL_AP_PA],
    }


def build_dataset_observations(spec: DatasetSpec) -> Dict[str, Any]:
    """Build all planar-only features and Q/SPECT targets for one cycle."""
    rows = sensitivity.calculate_patient_effective_sensitivity(
        planar_dir=spec.planar_dir,
        qspect_dir=spec.qspect_dir,
        crop_strategy="profile_qspect",
        skip_incomplete_qspect=True,
        allow_unpaired_planar=True,
    )
    scans_by_name = {
        str(scan["scan_name"]): scan
        for scan in ac.sorted_planar_scans(spec.planar_dir)
    }
    scans = [scans_by_name[str(row["planar_scan_name"])] for row in rows]
    matched_rows = [
        mode_matched_measurements(
            scan,
            (int(row["crop_top"]), int(row["crop_bottom"])),
        )
        for scan, row in zip(scans, rows)
    ]
    features_by_geometry = {
        geometry: {
            name: np.asarray(
                [
                    item["features_by_geometry"][geometry][name]
                    for item in matched_rows
                ],
                dtype=np.float64,
            )
            for name in FEATURE_ORDER
        }
        for geometry in (
            GEOMETRY_CROP_GM,
            GEOMETRY_FULL_GM,
            GEOMETRY_FULL_AP_PA,
        )
    }
    features = features_by_geometry[GEOMETRY_CROP_GM]
    tew_target = np.asarray(
        [row["tew_effective_sensitivity_cps_per_mbq"] for row in rows],
        dtype=np.float64,
    )
    tew_rate = np.asarray([row["tew_gm_cps"] for row in rows], dtype=np.float64)
    raw_crop_rate = np.asarray(
        [row["raw_photopeak_gm_cps"] for row in rows], dtype=np.float64
    )
    raw_full_rate = np.asarray(
        [item["raw_rates"][GEOMETRY_FULL_AP_PA] for item in matched_rows],
        dtype=np.float64,
    )
    tew_full_gm_rate = np.asarray(
        [item["tew_rates"][GEOMETRY_FULL_GM] for item in matched_rows],
        dtype=np.float64,
    )
    raw_full_gm_rate = np.asarray(
        [item["raw_rates"][GEOMETRY_FULL_GM] for item in matched_rows],
        dtype=np.float64,
    )
    tew_full_ap_pa_rate = np.asarray(
        [item["tew_rates"][GEOMETRY_FULL_AP_PA] for item in matched_rows],
        dtype=np.float64,
    )
    all_four_full_rate = np.asarray(
        [item["all_four_full_ap_pa_rate"] for item in matched_rows],
        dtype=np.float64,
    )
    if tew_target.size == 0:
        raise ValueError(f"No valid paired observations for {spec.key}")
    if np.any(~np.isfinite(tew_target)) or np.any(tew_target <= 0.0):
        raise ValueError(f"Invalid effective-sensitivity target in {spec.key}")
    reference_activity = tew_rate / tew_target
    rates = {
        MODE_TEW_CROP_GM: tew_rate,
        MODE_RAW_CROP_GM: raw_crop_rate,
        MODE_TEW_FULL_GM: tew_full_gm_rate,
        MODE_RAW_FULL_GM: raw_full_gm_rate,
        MODE_TEW_FULL_AP_PA: tew_full_ap_pa_rate,
        MODE_RAW_FULL_AP_PA: raw_full_rate,
        MODE_ALL_FOUR_FULL_AP_PA: all_four_full_rate,
    }
    for mode_key, rate in rates.items():
        if np.any(~np.isfinite(rate)) or np.any(rate <= 0.0):
            raise ValueError(f"Invalid measurement rate for {spec.key}/{mode_key}")

    records = []
    for row in rows:
        records.append(
            {
                "dataset_key": spec.key,
                "patient_id": spec.patient_id,
                "cycle_id": spec.cycle_id,
                "dataset_label": spec.label,
                "timepoint": str(row["label"]),
                "planar_day": float(row["planar_day"]),
                "planar_datetime": row["planar_datetime"],
                "qspect_datetime": row["qspect_datetime"],
                "crop_top": int(row["crop_top"]),
                "crop_bottom": int(row["crop_bottom"]),
                "profile_match_accepted": bool(row["profile_match_accepted"]),
            }
        )
    return {
        "spec": spec,
        "records": records,
        "features": features,
        "target_sensitivity": tew_target,
        "tew_rate": tew_rate,
        "measurement_rate": tew_rate,
        "reference_activity": reference_activity,
        "rate_modes": {
            mode_key: {
                "measurement_rate": rate,
                "target_sensitivity": rate / reference_activity,
                "features": features_by_geometry[
                    RATE_MODES[mode_key].feature_geometry
                ],
            }
            for mode_key, rate in rates.items()
        },
    }


def select_rate_mode_observations(
    base_observations: Mapping[str, Mapping[str, Any]],
    rate_mode: str,
) -> Dict[str, Dict[str, Any]]:
    """Select one measurement-rate target without recomputing image features."""
    if rate_mode not in RATE_MODES:
        raise ValueError(f"Unknown rate mode: {rate_mode}")
    selected: Dict[str, Dict[str, Any]] = {}
    for dataset_key, observation in base_observations.items():
        if "rate_modes" not in observation:
            if rate_mode != MODE_TEW_CROP_GM:
                raise ValueError(
                    "Legacy observation cache only supports the TEW rate mode"
                )
            legacy_rate = (
                observation["measurement_rate"]
                if "measurement_rate" in observation
                else observation["tew_rate"]
            )
            measurement_rate = np.asarray(legacy_rate, dtype=np.float64)
            target = np.asarray(
                observation["target_sensitivity"], dtype=np.float64
            )
        else:
            mode_values = observation["rate_modes"][rate_mode]
            measurement_rate = np.asarray(
                mode_values["measurement_rate"], dtype=np.float64
            )
            target = np.asarray(
                mode_values["target_sensitivity"], dtype=np.float64
            )
        mode_features = (
            observation["features"]
            if "rate_modes" not in observation
            else observation["rate_modes"][rate_mode].get(
                "features", observation["features"]
            )
        )
        selected[dataset_key] = {
            **observation,
            "rate_mode": rate_mode,
            "rate_mode_label": RATE_MODES[rate_mode].label,
            "measurement_rate": measurement_rate,
            "target_sensitivity": target,
            "features": mode_features,
        }
    return selected


def load_base_observation_cache(
    dataset_keys: Iterable[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    keys = (
        tuple(DATASETS)
        if dataset_keys is None
        else _validate_dataset_keys(tuple(dataset_keys))
    )
    return {key: build_dataset_observations(DATASETS[key]) for key in keys}


def load_observation_cache(
    dataset_keys: Iterable[str] | None = None,
    rate_mode: str = MODE_TEW_CROP_GM,
) -> Dict[str, Dict[str, Any]]:
    """Load observations for one rate mode; TEW remains the legacy default."""
    return select_rate_mode_observations(
        load_base_observation_cache(dataset_keys), rate_mode
    )


def _combine_observations(
    observations: Mapping[str, Mapping[str, Any]], dataset_keys: Sequence[str]
) -> Dict[str, Any]:
    keys = _validate_dataset_keys(dataset_keys)
    missing = sorted(set(keys) - set(observations))
    if missing:
        raise ValueError(f"Observation cache is missing: {', '.join(missing)}")
    feature_names = FEATURE_ORDER
    return {
        "dataset_keys": keys,
        "records": [
            record
            for key in keys
            for record in observations[key]["records"]
        ],
        "features": {
            name: np.concatenate(
                [np.asarray(observations[key]["features"][name]) for key in keys]
            )
            for name in feature_names
        },
        "target_sensitivity": np.concatenate(
            [np.asarray(observations[key]["target_sensitivity"]) for key in keys]
        ),
        "measurement_rate": np.concatenate(
            [
                np.asarray(
                    observations[key]["measurement_rate"]
                    if "measurement_rate" in observations[key]
                    else observations[key]["tew_rate"]
                )
                for key in keys
            ]
        ),
        "reference_activity": np.concatenate(
            [np.asarray(observations[key]["reference_activity"]) for key in keys]
        ),
    }


def _model_features(model_name: str) -> tuple[str, ...]:
    if model_name == MODEL_CONSTANT:
        return ()
    if model_name == MODEL_LOCKED:
        return (LOCKED_FEATURE,)
    if model_name == MODEL_ASYMMETRY:
        return (LOCKED_FEATURE, ASYMMETRY_FEATURE)
    if model_name == MODEL_ADJACENT:
        return (ADJACENT_FEATURE,)
    if model_name == MODEL_TEW_FRACTION:
        return (TEW_FRACTION_FEATURE,)
    if model_name == MODEL_PHOTO_ALL:
        return (PHOTO_ALL_FRACTION_FEATURE,)
    if model_name == MODEL_LOWER:
        return (LOWER_FEATURE,)
    if model_name == MODEL_UPPER:
        return (UPPER_FEATURE,)
    if model_name == MODEL_RAW_ASYMMETRY:
        return (RAW_PHOTO_GENERAL_FEATURE, ALL_WINDOW_ASYMMETRY_FEATURE)
    raise ValueError(f"Unknown model: {model_name}")


def fit_model(model_name: str, training: Mapping[str, Any]) -> Dict[str, Any]:
    """Fit one pre-specified model on training observations only."""
    target = np.asarray(training["target_sensitivity"], dtype=np.float64)
    feature_names = _model_features(model_name)
    if model_name == MODEL_CONSTANT:
        return {
            "model": model_name,
            "status": "fitted",
            "reason": "",
            "feature_names": feature_names,
            "coefficients": np.asarray([float(np.mean(target))]),
            "rank": 1,
            "condition_number": 1.0,
            "feature_ranges": {},
        }

    parameter_count = len(feature_names) + 1
    if target.size <= parameter_count:
        return {
            "model": model_name,
            "status": "not_estimable",
            "reason": (
                f"n_train={target.size} must exceed {parameter_count} fitted parameters"
            ),
            "feature_names": feature_names,
            "coefficients": np.full(parameter_count, np.nan),
            "rank": 0,
            "condition_number": np.nan,
            "feature_ranges": {},
        }

    columns = [
        np.asarray(training["features"][name], dtype=np.float64)
        for name in feature_names
    ]
    design = np.column_stack([np.ones(target.size), *columns])
    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        design, np.log(target), rcond=None
    )
    if int(rank) < parameter_count:
        return {
            "model": model_name,
            "status": "not_estimable",
            "reason": "rank-deficient training design",
            "feature_names": feature_names,
            "coefficients": coefficients,
            "rank": int(rank),
            "condition_number": np.inf,
            "feature_ranges": {},
        }
    condition_number = float(singular_values[0] / singular_values[-1])
    return {
        "model": model_name,
        "status": "fitted",
        "reason": "",
        "feature_names": feature_names,
        "coefficients": coefficients,
        "rank": int(rank),
        "condition_number": condition_number,
        "feature_ranges": {
            name: (
                float(np.min(training["features"][name])),
                float(np.max(training["features"][name])),
            )
            for name in feature_names
        },
    }


def predict_model(fit: Mapping[str, Any], test: Mapping[str, Any]) -> np.ndarray:
    if fit["status"] != "fitted":
        raise ValueError(f"Cannot predict with model status {fit['status']}")
    coefficients = np.asarray(fit["coefficients"], dtype=np.float64)
    if fit["model"] == MODEL_CONSTANT:
        return np.full(
            np.asarray(test["target_sensitivity"]).shape,
            coefficients[0],
            dtype=np.float64,
        )
    columns = [
        np.asarray(test["features"][name], dtype=np.float64)
        for name in fit["feature_names"]
    ]
    design = np.column_stack(
        [np.ones(np.asarray(test["target_sensitivity"]).size), *columns]
    )
    return np.exp(design @ coefficients)


def _extrapolation_flags(
    fit: Mapping[str, Any], test: Mapping[str, Any]
) -> np.ndarray:
    count = np.asarray(test["target_sensitivity"]).size
    flags = np.zeros(count, dtype=bool)
    for name, bounds in fit.get("feature_ranges", {}).items():
        values = np.asarray(test["features"][name], dtype=np.float64)
        flags |= (values < bounds[0]) | (values > bounds[1])
    return flags


def _metrics(
    target_sensitivity: np.ndarray, predicted_sensitivity: np.ndarray
) -> Dict[str, float]:
    return v2._prediction_metrics(target_sensitivity, predicted_sensitivity)


def fit_and_evaluate(
    training_datasets: Sequence[str],
    test_datasets: Sequence[str],
    observations: Mapping[str, Mapping[str, Any]],
    *,
    split_id: str = "custom",
    split_type: str = "custom",
    split_label: str = "custom split",
    rate_mode: str = MODE_TEW_CROP_GM,
    model_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Fit on X datasets and evaluate unchanged on disjoint Y datasets."""
    if rate_mode not in RATE_MODES:
        raise ValueError(f"Unknown rate mode: {rate_mode}")
    fitted_model_names = tuple(
        RATE_MODES[rate_mode].model_names if model_names is None else model_names
    )
    invalid_models = sorted(set(fitted_model_names) - set(MODEL_ORDER))
    if invalid_models:
        raise ValueError(f"Unknown model(s): {', '.join(invalid_models)}")
    training_keys, test_keys = validate_split(training_datasets, test_datasets)
    training = _combine_observations(observations, training_keys)
    test = _combine_observations(observations, test_keys)
    metrics_rows: list[Dict[str, Any]] = []
    prediction_rows: list[Dict[str, Any]] = []
    fits: Dict[str, Dict[str, Any]] = {}

    for model_name in fitted_model_names:
        fit = fit_model(model_name, training)
        fits[model_name] = fit
        base_metric_row = {
            "rate_mode": rate_mode,
            "rate_mode_label": RATE_MODES[rate_mode].label,
            "split_id": split_id,
            "split_type": split_type,
            "split_label": split_label,
            "training_datasets": "+".join(training_keys),
            "test_datasets": "+".join(test_keys),
            "n_train": len(training["records"]),
            "n_test": len(test["records"]),
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "fit_status": fit["status"],
            "fit_reason": fit["reason"],
            "rank": fit["rank"],
            "condition_number": fit["condition_number"],
        }
        if fit["status"] != "fitted":
            metrics_rows.append(
                {
                    **base_metric_row,
                    "metric_scope": "combined_test",
                    "metric_dataset": "+".join(test_keys),
                    "activity_mare_percent": np.nan,
                    "activity_bias_percent": np.nan,
                    "activity_rmse_percent": np.nan,
                    "maximum_activity_error_percent": np.nan,
                    "sensitivity_mare_percent": np.nan,
                    "sensitivity_bias_percent": np.nan,
                }
            )
            continue

        predicted_sensitivity = predict_model(fit, test)
        extrapolated = _extrapolation_flags(fit, test)
        target_sensitivity = np.asarray(test["target_sensitivity"], dtype=np.float64)
        measurement_rate = np.asarray(
            test["measurement_rate"], dtype=np.float64
        )
        reference_activity = np.asarray(test["reference_activity"], dtype=np.float64)
        predicted_activity = measurement_rate / predicted_sensitivity
        activity_ratio = predicted_activity / reference_activity

        combined_metrics = _metrics(target_sensitivity, predicted_sensitivity)
        metrics_rows.append(
            {
                **base_metric_row,
                "metric_scope": "combined_test",
                "metric_dataset": "+".join(test_keys),
                "extrapolated_predictions": int(np.count_nonzero(extrapolated)),
                "prediction_count": int(extrapolated.size),
                **combined_metrics,
            }
        )

        dataset_array = np.asarray(
            [record["dataset_key"] for record in test["records"]], dtype=object
        )
        for dataset_key in test_keys:
            selected = dataset_array == dataset_key
            scoped_metrics = _metrics(
                target_sensitivity[selected], predicted_sensitivity[selected]
            )
            metrics_rows.append(
                {
                    "rate_mode": rate_mode,
                    "rate_mode_label": RATE_MODES[rate_mode].label,
                    **base_metric_row,
                    "metric_scope": "test_dataset",
                    "metric_dataset": dataset_key,
                    "extrapolated_predictions": int(
                        np.count_nonzero(extrapolated[selected])
                    ),
                    "prediction_count": int(np.count_nonzero(selected)),
                    **scoped_metrics,
                }
            )

        coefficient_values = np.asarray(fit["coefficients"], dtype=np.float64)
        coefficient_by_name = {"intercept_or_constant": coefficient_values[0]}
        coefficient_by_name.update(
            {
                f"coefficient_{name}": coefficient_values[index + 1]
                for index, name in enumerate(fit["feature_names"])
            }
        )
        for index, record in enumerate(test["records"]):
            feature_values = {
                name: float(test["features"][name][index])
                for name in FEATURE_ORDER
            }
            prediction_rows.append(
                {
                    "rate_mode": rate_mode,
                    "rate_mode_label": RATE_MODES[rate_mode].label,
                    "split_id": split_id,
                    "split_type": split_type,
                    "split_label": split_label,
                    "training_datasets": "+".join(training_keys),
                    "test_datasets": "+".join(test_keys),
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    **record,
                    **feature_values,
                    "measurement_rate_cps": float(measurement_rate[index]),
                    "observed_sensitivity_cps_per_mbq": float(
                        target_sensitivity[index]
                    ),
                    "predicted_sensitivity_cps_per_mbq": float(
                        predicted_sensitivity[index]
                    ),
                    "qspect_activity_mbq": float(reference_activity[index]),
                    "predicted_activity_mbq": float(predicted_activity[index]),
                    "predicted_over_qspect": float(activity_ratio[index]),
                    "activity_error_percent": float(
                        100.0 * (activity_ratio[index] - 1.0)
                    ),
                    "feature_extrapolation": bool(extrapolated[index]),
                    "fit_rank": fit["rank"],
                    "fit_condition_number": fit["condition_number"],
                    **coefficient_by_name,
                }
            )

    return {
        "split_id": split_id,
        "split_type": split_type,
        "split_label": split_label,
        "rate_mode": rate_mode,
        "model_names": fitted_model_names,
        "training": training,
        "test": test,
        "fits": fits,
        "metrics": metrics_rows,
        "predictions": prediction_rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _combined_metric_lookup(
    metrics_rows: Sequence[Mapping[str, Any]], split_id: str, model_name: str
) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in metrics_rows
            if row["split_id"] == split_id
            and row["model"] == model_name
            and row["metric_scope"] == "combined_test"
        ),
        None,
    )


def plot_pairwise_transfer_matrix(
    metrics_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    model_names: Sequence[str] = MODEL_ORDER,
    title_prefix: str = "",
) -> None:
    plotted_models = tuple(model_names)
    directed_splits = [
        (training_key, test_key)
        for training_key in DATASETS
        for test_key in DATASETS
        if training_key != test_key
    ]
    finite_values = [
        float(row["activity_mare_percent"])
        for row in metrics_rows
        if row["split_type"] == "pairwise_cycle_transfer"
        and row["metric_scope"] == "combined_test"
        and np.isfinite(float(row["activity_mare_percent"]))
    ]
    vmax = max(
        10.0,
        float(np.percentile(finite_values, 90)) if finite_values else 10.0,
    )
    matrix = np.full(
        (len(plotted_models), len(directed_splits)), np.nan, dtype=np.float64
    )
    for row_index, model_name in enumerate(plotted_models):
        for column_index, (training_key, test_key) in enumerate(directed_splits):
            row = _combined_metric_lookup(
                metrics_rows,
                f"pair_{training_key}_to_{test_key}",
                model_name,
            )
            if row is not None and row["fit_status"] == "fitted":
                matrix[row_index, column_index] = float(
                    row["activity_mare_percent"]
                )

    fig, axis = plt.subplots(
        figsize=(13.8, 8.2), layout="constrained", facecolor="white"
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix), cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto"
    )
    axis.set_xticks(
        range(len(directed_splits)),
        [
            f"{DATASETS[training].short_label}\n→ {DATASETS[test].short_label}"
            for training, test in directed_splits
        ],
    )
    axis.set_yticks(
        range(len(plotted_models)),
        [MODEL_LABELS[model] for model in plotted_models],
    )
    axis.set_xlabel("Cycle d’entraînement → cycle de test")
    axis.set_title(
        (f"{title_prefix}\n" if title_prefix else "")
        + "Transferts cycle à cycle — erreur moyenne absolue"
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
                color=(
                    "white"
                    if np.isfinite(value) and min(value, vmax) > 0.58 * vmax
                    else "black"
                ),
                fontsize=8.5,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("MARE sur l’activité (%)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_leave_one_cycle_out(
    metrics_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    model_names: Sequence[str] = MODEL_ORDER,
    title_prefix: str = "",
) -> None:
    plotted_models = tuple(model_names)
    test_keys = tuple(DATASETS)
    matrix = np.full((len(plotted_models), len(test_keys)), np.nan, dtype=np.float64)
    for row_index, model_name in enumerate(plotted_models):
        for column_index, test_key in enumerate(test_keys):
            row = _combined_metric_lookup(
                metrics_rows, f"loco_holdout_{test_key}", model_name
            )
            if row is not None and row["fit_status"] == "fitted":
                matrix[row_index, column_index] = float(
                    row["activity_mare_percent"]
                )
    finite = matrix[np.isfinite(matrix)]
    vmax = max(10.0, float(np.percentile(finite, 90)) if finite.size else 10.0)
    fig, axis = plt.subplots(
        figsize=(9.8, 8.1), layout="constrained", facecolor="white"
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix), cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto"
    )
    axis.set_xticks(
        range(len(test_keys)),
        [f"Test : {DATASETS[key].short_label}" for key in test_keys],
    )
    axis.set_yticks(
        range(len(plotted_models)),
        [MODEL_LABELS[model] for model in plotted_models],
    )
    axis.set_title(
        (f"{title_prefix}\n" if title_prefix else "")
        + "Validation leave-one-cycle-out"
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
                color=(
                    "white"
                    if np.isfinite(value) and min(value, vmax) > 0.58 * vmax
                    else "black"
                ),
                fontsize=9.5,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("MARE sur l’activité (%)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_feature_overview(
    observations: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    sensitivity_label: str = "Sensibilité effective TEW (cps/MBq)",
    title_prefix: str = "",
) -> None:
    """Show every candidate feature against the Q/SPECT-derived target."""
    combined = _combine_observations(observations, tuple(DATASETS))
    targets = np.asarray(combined["target_sensitivity"], dtype=np.float64)
    dataset_keys = np.asarray(
        [record["dataset_key"] for record in combined["records"]], dtype=object
    )
    dataset_colors = {
        "p11_may": "#1F77B4",
        "p11_july": "#FF7F0E",
        "p8_june": "#2CA02C",
    }
    dataset_markers = {"p11_may": "o", "p11_july": "s", "p8_june": "D"}
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13.8, 11.2),
        layout="constrained",
        sharey=True,
        facecolor="white",
    )
    for axis, feature_name in zip(axes.flat, FEATURE_ORDER):
        values = np.asarray(combined["features"][feature_name], dtype=np.float64)
        for dataset_key in DATASETS:
            selected = dataset_keys == dataset_key
            axis.scatter(
                values[selected],
                targets[selected],
                s=54,
                marker=dataset_markers[dataset_key],
                color=dataset_colors[dataset_key],
                edgecolor="white",
                linewidth=0.7,
                label=DATASETS[dataset_key].short_label,
                zorder=3,
            )
        correlation = (
            np.nan
            if float(np.ptp(values)) <= np.finfo(float).eps
            else float(np.corrcoef(values, np.log(targets))[0, 1])
        )
        axis.set_title(FEATURE_LABELS[feature_name], fontsize=10.5)
        axis.set_xlabel("Valeur de la variable")
        axis.grid(True, linestyle="--", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.04,
            0.94,
            f"r avec log(S_eff) = {correlation:+.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#111111",
        )
    for axis in axes[:, 0]:
        axis.set_ylabel(sensitivity_label)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        (f"{title_prefix}\n" if title_prefix else "")
        + "Variables planaires candidates et sensibilité effective observée",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_feature_correlation_matrix(
    observations: Mapping[str, Mapping[str, Any]], output_path: Path
) -> None:
    """Visualize redundancy among candidate planar variables."""
    combined = _combine_observations(observations, tuple(DATASETS))
    values = np.vstack(
        [
            np.asarray(combined["features"][name], dtype=np.float64)
            for name in FEATURE_ORDER
        ]
    )
    matrix = np.corrcoef(values)
    labels = [FEATURE_LABELS[name].replace("\n", " ") for name in FEATURE_ORDER]
    fig, axis = plt.subplots(
        figsize=(11.8, 9.6), layout="constrained", facecolor="white"
    )
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(
        range(len(labels)), labels, rotation=42, ha="right", fontsize=8.5
    )
    axis.set_yticks(range(len(labels)), labels, fontsize=8.5)
    axis.set_title("Corrélation entre les variables planaires candidates")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(value) > 0.62 else "black",
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Corrélation de Pearson")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_primary_model_mare(
    metrics_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    model_names: Sequence[str] = MODEL_ORDER,
    title_prefix: str = "",
) -> None:
    """Compare every model for the primary P11-to-P8 patient holdout."""
    plotted_models = tuple(model_names)
    values = []
    for model_name in plotted_models:
        row = _combined_metric_lookup(
            metrics_rows, "loco_holdout_p8_june", model_name
        )
        values.append(
            np.nan
            if row is None or row["fit_status"] != "fitted"
            else float(row["activity_mare_percent"])
        )
    y = np.arange(len(plotted_models), dtype=np.float64)
    fig, axis = plt.subplots(
        figsize=(10.8, 7.6), layout="constrained", facecolor="white"
    )
    bars = axis.barh(
        y,
        np.nan_to_num(values, nan=0.0),
        color=[MODEL_COLORS[model] for model in plotted_models],
    )
    for bar, value in zip(bars, values):
        label = "N/E" if not np.isfinite(value) else f"{value:.1f} %"
        axis.annotate(
            label,
            (bar.get_width(), bar.get_y() + bar.get_height() / 2.0),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#111111",
        )
    axis.set_yticks(y, [MODEL_LABELS[model] for model in plotted_models])
    axis.invert_yaxis()
    axis.set_xlabel("MARE sur l’activité (%)")
    axis.set_title(
        (f"{title_prefix}\n" if title_prefix else "")
        + "Entraînement P11-mai + P11-juillet — test P8-juin"
    )
    axis.grid(axis="x", linestyle="--", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_primary_patient_holdout(
    prediction_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    model_names: Sequence[str] = PRIMARY_PLOT_MODELS,
    title_prefix: str = "",
) -> None:
    split_id = "loco_holdout_p8_june"
    selected = [row for row in prediction_rows if row["split_id"] == split_id]
    if not selected:
        return
    timepoints = [
        str(row["timepoint"])
        for row in selected
        if row["model"] == MODEL_CONSTANT
    ]
    x = np.arange(len(timepoints), dtype=np.float64)
    fig, axis = plt.subplots(figsize=(10.2, 6.1), layout="constrained")
    axis.axhspan(0.90, 1.10, color="#2E9F55", alpha=0.10, label="Écart de ±10 %")
    axis.axhline(1.0, color="black", linewidth=1.4, label="Q/SPECT")
    for model_name in model_names:
        model_rows = [row for row in selected if row["model"] == model_name]
        model_rows.sort(key=lambda row: float(row["planar_day"]))
        if not model_rows:
            continue
        ratios = [float(row["predicted_over_qspect"]) for row in model_rows]
        axis.plot(
            x,
            ratios,
            marker=MODEL_MARKERS[model_name],
            linewidth=2.2,
            markersize=8,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
        for position, value, row in zip(x, ratios, model_rows):
            if bool(row["feature_extrapolation"]):
                axis.scatter(
                    [position],
                    [value],
                    marker=MODEL_MARKERS[model_name],
                    s=82,
                    facecolor="white",
                    edgecolor=MODEL_COLORS[model_name],
                    linewidth=2.0,
                    zorder=5,
                )
            axis.annotate(
                f"{value:.3f}",
                (position, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=MODEL_COLORS[model_name],
            )
    axis.set_xticks(x, timepoints)
    axis.set_xlabel("Timepoint de P8 — cycle de juin")
    axis.set_ylabel("Activité prédite / activité Q/SPECT")
    axis.set_title(
        (f"{title_prefix}\n" if title_prefix else "")
        + "Entraînement sur les deux cycles de P11 — test externe sur P8"
    )
    axis.grid(axis="y", linestyle="--", alpha=0.28)
    axis.legend(frameon=False, ncol=2)
    axis.text(
        0.995,
        0.015,
        "Marqueur ouvert : variable hors de l’intervalle d’entraînement",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#111111",
    )
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _format_metric(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/E"
    return "N/E" if not np.isfinite(numeric) else f"{numeric:.2f}"


def build_report(
    observations: Mapping[str, Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]],
    rate_mode: str = MODE_TEW_CROP_GM,
    model_names: Sequence[str] | None = None,
) -> str:
    mode_spec = RATE_MODES[rate_mode]
    reported_models = tuple(
        mode_spec.model_names if model_names is None else model_names
    )
    lines = [
        "Multi-cycle planar effective-sensitivity model — v3",
        "====================================================",
        "",
        f"Rate mode: {mode_spec.label}",
        f"Definition: {mode_spec.description}",
        "Target: S_eff(mode) = R_mode / A_Q/SPECT.",
        "",
        "Datasets:",
    ]
    for key, spec in DATASETS.items():
        lines.append(
            f"  {key}: {spec.label}, n={len(observations[key]['records'])}"
        )
    lines.extend(
        [
            "",
            "Models:",
        ]
    )
    for model_name in reported_models:
        feature_names = _model_features(model_name)
        description = (
            "moyenne de la sensibilité d’entraînement"
            if not feature_names
            else " + ".join(feature_names)
        )
        lines.append(f"  {MODEL_LABELS[model_name]}: {description}.")
    lines.extend(
        [
            "  Models with two predictors are not estimated when n_train does not",
            "  exceed their 3 fitted parameters.",
            "",
            "Primary grouped validations (MARE on activity, %):",
            "  model | P11-mai held out | P11-juillet held out | P8-juin held out",
            "  ----- | ---------------- | -------------------- | ----------------",
        ]
    )
    for model_name in reported_models:
        values = []
        for test_key in DATASETS:
            row = _combined_metric_lookup(
                metrics_rows, f"loco_holdout_{test_key}", model_name
            )
            values.append("N/E" if row is None else _format_metric(row["activity_mare_percent"]))
        lines.append(
            f"  {MODEL_LABELS[model_name].split(' — ')[0]:>5} | "
            + " | ".join(f"{value:>16}" for value in values)
        )

    primary_rows = {
        model: _combined_metric_lookup(metrics_rows, "loco_holdout_p8_june", model)
        for model in reported_models
    }
    lines.extend(
        [
            "",
            "Primary inter-patient result:",
            "  Training = P11 May + P11 July; test = P8 June.",
        ]
    )
    for model_name in reported_models:
        row = primary_rows[model_name]
        if row is None or row["fit_status"] != "fitted":
            lines.append(f"  {MODEL_LABELS[model_name]}: not estimable")
            continue
        lines.append(
            f"  {MODEL_LABELS[model_name]}: "
            f"MARE={float(row['activity_mare_percent']):.2f}%, "
            f"bias={float(row['activity_bias_percent']):+.2f}%, "
            f"RMSE={float(row['activity_rmse_percent']):.2f}%, "
            f"max={float(row['maximum_activity_error_percent']):.2f}%, "
            f"extrapolation={int(row.get('extrapolated_predictions', 0))}/"
            f"{int(row.get('prediction_count', 0))}"
        )

    lines.extend(
        [
            "",
            "Interpretation limits:",
            "  - There are 11 timepoints but only two patients.",
            "  - P11 May and P11 July are two cycles from the same patient.",
            "  - The locked spectral feature was historically discovered on P11 May.",
            "    Splits that test P11 May are exploratory, not fully independent.",
            "  - The current crop is guided by Q/SPECT profile correspondence.",
            "    The model inputs are planar, but preprocessing is not yet deployable",
            "    without Q/SPECT.",
            "  - No random timepoint split is used; every split is grouped by cycle.",
            "  - M2 and M9 are exploratory and limited to two predictors.",
            "  - Several candidate variables are strongly correlated. Their models",
            "    are compared separately rather than combined into a large regression.",
            f"  - Rate mode evaluated here: {mode_spec.label}.",
            "  - Modes must be compared with the same grouped train/test split and",
            "    the same predictor model; choosing the best combination afterward",
            "    remains exploratory with only two patients.",
            "",
        ]
    )
    return "\n".join(lines)


def build_model_methodology_document() -> str:
    """Return a standalone description of the ten v3 candidate models."""
    return """MODÈLES DE SENSIBILITÉ EFFECTIVE PLANAIRE — V3
=================================================

OBJECTIF COMMUN
---------------

Les neuf modèles conservés de M0 à M9 ne sont pas neuf algorithmes statistiques
différents. Ils représentent neuf choix de variables planaires appliqués aux
différentes définitions du taux de comptage. Pour chaque mode m, la cible est :

    S_eff(m) = R_planaire(m) / A_QSPECT

où :

    S_eff(m)       = sensibilité effective du mode, en cps/MBq;
    R_planaire(m)  = taux de comptage défini par le mode, en cps;
    A_QSPECT       = activité Q/SPECT de référence, en MBq.

Pour M1 à M9, l'ajustement est une régression linéaire par moindres carrés
sur le logarithme de la sensibilité :

    log(S_eff(m)) = β0 + β1*x1 [+ β2*x2]

La sensibilité prédite est ensuite ramenée à son échelle originale :

    S_eff(m) prédite = exp(β0 + β1*x1 [+ β2*x2])

L'activité planaire prédite est finalement :

    A_planaire prédite = R_planaire(m) / S_eff(m) prédite

Le Q/SPECT est utilisé comme cible pendant l'entraînement et comme référence
pendant le test. Il n'est pas une variable d'entrée du modèle final. Le CT
n'est pas une variable d'entrée. Le crop par profil demeure toutefois guidé
par le Q/SPECT pour les deux modes utilisant un crop.

Principe de cohérence spatiale : le mode m définit aussi la géométrie de toutes
les variables planaires. Chaque fenêtre est donc mesurée sur le même support
que le taux cible : crop ou image complète, puis moyenne géométrique GM ou
somme AP+PA. Les comptes spectraux sont divisés par la largeur de leur fenêtre
et par la durée d'acquisition avant de former les ratios. On note cette densité
spectrale Q_m(E). Ainsi, un modèle évalué en mode image complète n'emploie plus
une variable calculée dans le crop.

MODES DE TAUX DE COMPTAGE
-------------------------

    tew_crop_gm
        TEW appliquée séparément sur AP et PA dans le crop, puis moyenne
        géométrique. Il s'agit du mode historique.

    raw_photopeak_crop_gm
        Photopeak brut avec exactement le même crop et la même moyenne
        géométrique. Sa comparaison au mode précédent isole l'effet TEW.

    tew_full_gm
        TEW appliquée séparément sur AP et PA sur l'image complète, puis
        moyenne géométrique. Sa comparaison à tew_crop_gm isole le crop.

    raw_photopeak_full_gm
        Photopeak brut sur l'image complète avec moyenne géométrique. Sa
        comparaison à raw_photopeak_crop_gm isole le crop sans TEW.

    tew_full_ap_pa
        Somme AP+PA du photopeak corrigé par TEW sur l'image complète. Sa
        comparaison au mode brut AP+PA isole la TEW sans moyenne géométrique.

    raw_photopeak_full_ap_pa
        Somme brute AP+PA du photopeak sur l'image complète. Ce mode change
        aussi le crop et la combinaison AP/PA.

    all_four_full_ap_pa
        Somme brute AP+PA des quatre fenêtres sur l'image complète. Ce mode est
        exploratoire, notamment parce que les fenêtres ont différentes largeurs.

M1 à M9 sont recalculés selon la géométrie de chaque mode. M5 est disponible
dans les trois modes TEW, avec une fraction TEW/photopeak adaptée au crop, à
l'image complète GM ou à l'image complète AP+PA. Il est volontairement exclu
des modes bruts et du mode quatre fenêtres afin que ces pipelines n'utilisent
aucune information provenant d'une correction TEW.

NOTATION DES FENÊTRES
---------------------

    P = fenêtre photopeak autour de 208 keV;
    G = grande fenêtre de basse énergie, appelée General Scatter;
    L = petite fenêtre de diffusé inférieure adjacente au photopeak;
    U = petite fenêtre de diffusé supérieure adjacente au photopeak;
    GM = moyenne géométrique des images AP et PA;
    TEW = correction de diffusé Triple Energy Window.
    Q_m(E) = densité de comptes de la fenêtre E dans la géométrie du mode m.


ORDRE EXACT DES OPÉRATIONS POUR LES RATIOS
-------------------------------------------

Les variables spectrales sont des ratios globaux de comptes intégrés. Aucun
modèle ne construit une carte de ratios pixel par pixel, et aucun modèle ne
somme ensuite des ratios locaux. L'ordre commun des opérations est le suivant.

1. Définir le support spatial Ω_m :

       * lignes du crop par profil pour les modes « crop »;
       * tous les pixels pour les modes « image complète ».

2. Combiner les vues AP et PA selon la géométrie du mode :

       Modes GM :

           I_m,E(p) = sqrt(I_AP,E(p) * I_PA,E(p))

           C_m(E) = somme sur p dans Ω_m de I_m,E(p)

       Modes AP+PA :

           C_m(E) = somme sur p dans Ω_m de I_AP,E(p)
                    + somme sur p dans Ω_m de I_PA,E(p)

   La moyenne géométrique est donc bien une opération pixel par pixel effectuée
   avant la somme. En revanche, le ratio entre fenêtres n'est jamais calculé
   pixel par pixel.

3. Normaliser le total de chaque fenêtre par sa largeur énergétique ΔE_E et par
   la durée T de l'acquisition :

       Q_m(E) = C_m(E) / (ΔE_E * T)

   Les Q_m(E) ont ainsi la forme d'une densité spectrale globale de taux de
   comptage. Cette normalisation est importante parce que les quatre fenêtres
   n'ont pas toutes la même largeur.

4. Former les variables à partir de ces totaux normalisés :

       M1 : log(Q_m(G) / Q_m(P))
       M4 : log((Q_m(L) + Q_m(U)) / Q_m(P))
       M6 : Q_m(P) / (Q_m(L) + Q_m(P) + Q_m(U) + Q_m(G))
       M7 : log(Q_m(L) / Q_m(P))
       M8 : log(Q_m(U) / Q_m(P))

   Ainsi, M4 utilise bien les informations de M7 et M8, mais sur l'échelle
   non logarithmique :

       (Q_m(L) + Q_m(U)) / Q_m(P)
           = Q_m(L)/Q_m(P) + Q_m(U)/Q_m(P)

   Après logarithme :

       x_M4 = log(exp(x_M7) + exp(x_M8)), et non x_M7 + x_M8.

Cas particuliers :

    * M0 ne contient aucun ratio spectral. Il utilise la moyenne arithmétique
      des valeurs S_eff des observations d'entraînement pour le mode considéré.
    * M2 reprend le ratio global de M1 et ajoute une asymétrie AP/PA. Pour cette
      asymétrie, les comptes AP et PA du photopeak sont sommés séparément sur
      Ω_m, normalisés, puis leur ratio est calculé. Ce n'est pas une carte de
      ratios AP/PA.
    * M9 utilise le ratio global photopeak/General Scatter et une asymétrie AP/PA
      fondée sur les densités spectrales globales des quatre fenêtres, encore
      après sommation dans chaque vue.
    * M5 est le ratio de deux taux globaux : R_TEW/R_photopeak brut. La TEW est
      d'abord appliquée pixel par pixel séparément aux images AP et PA; les vues
      sont ensuite combinées selon le mode, les pixels sont sommés, puis les deux
      taux globaux sont divisés.

La cible suit le même principe d'intégration globale :

    S_eff(m) = R_planaire(m) / A_QSPECT

où R_planaire(m) est le taux total du support et de la combinaison AP/PA choisis
par le mode. Le seul calcul local commun aux modes GM est donc la moyenne
géométrique AP/PA; la TEW est également locale lorsqu'elle est utilisée. Les
ratios servant de variables aux régressions sont toujours calculés après
l'intégration spatiale.


M0 — SENSIBILITÉ CONSTANTE
--------------------------

Forme :

    S_eff prédite = moyenne de S_eff dans les données d'entraînement

Entrées :

    Aucune variable planaire prédictive.

Caractéristiques :

    * Modèle de référence le plus simple.
    * Aucune variable spectrale supplémentaire n'est nécessaire pour estimer
      la sensibilité; le taux reste toutefois défini par le mode choisi.
    * Suppose qu'une seule sensibilité convient à tous les timepoints du test.
    * Très rigide et très interprétable.

Rôle :

    Les autres modèles doivent produire une erreur prédictive inférieure à M0
    pour démontrer que leurs variables planaires apportent de l'information.

Limite principale :

    Il ne peut pas représenter les différences d'atténuation, de diffusé ou de
    gabarit entre patients.


M1 — RATIO SPECTRAL VERROUILLÉ
------------------------------

Variable :

    x1 = log(Q_m(G) / Q_m(P))

Forme :

    log(S_eff) = β0 + β1*x1

Construction :

    * Support spatial et combinaison AP/PA déterminés par le mode m.
    * Comptes normalisés par la largeur de la fenêtre énergétique et la durée.
    * Une seule variable prédictive.

Interprétation :

    Une plus grande proportion de photons de basse énergie peut refléter une
    contribution plus importante du diffusé et du gabarit du patient.

Caractéristique particulière :

    C'est le modèle principal historique de v2. Sa variable a été verrouillée
    avant les comparaisons multi-cycles de v3.

Limites :

    La relation imposée est log-linéaire. Dans les modes avec crop, ce crop est
    encore guidé par le Q/SPECT.


M2 — RATIO SPECTRAL ET ASYMÉTRIE AP/PA DU PHOTOPEAK
---------------------------------------------------

Variables :

    x1 = log(Q_m(G) / Q_m(P))

    x2 = |0,5*log(Q_m(P_AP) / Q_m(P_PA))|

Forme :

    log(S_eff) = β0 + β1*x1 + β2*x2

Construction :

    * Reprend la variable spectrale de M1.
    * Ajoute l'écart absolu entre les détecteurs AP et PA dans le photopeak.
    * Les deux variables utilisent le même support spatial que le mode.

Interprétation :

    x1 décrit la composition spectrale. x2 pourrait contenir de l'information
    sur la profondeur ou l'asymétrie de la distribution de l'activité.

Avantage :

    Combine une information spectrale et une information géométrique planaire.

Limites :

    Modèle exploratoire à deux variables. Il estime trois paramètres avec
    l'intercept et devient rapidement instable lorsque l'entraînement contient
    peu de timepoints.


M4 — DIFFUSÉ ADJACENT/PHOTOPEAK
-------------------------------

Variable :

    x1 = log((Q_m(L) + Q_m(U)) / Q_m(P))

Forme :

    log(S_eff) = β0 + β1*x1

Construction :

    * Support spatial et combinaison AP/PA déterminés par le mode m.
    * Combine les deux petites fenêtres adjacentes au photopeak.
    * N'utilise pas la grande fenêtre General Scatter.

Interprétation :

    Mesure le diffusé spectral immédiatement voisin du photopeak et se rapproche
    de l'information utilisée dans la correction TEW.

Avantage :

    Peut être plus représentatif de la contamination du photopeak que la grande
    fenêtre de basse énergie.

Limite :

    Les petites fenêtres contiennent moins de comptes et peuvent donc être plus
    sensibles au bruit statistique.


M5 — FRACTION CONSERVÉE APRÈS TEW
---------------------------------

Variable :

    x1 = R_TEW / R_photopeak brut

Forme :

    log(S_eff) = β0 + β1*x1

Construction :

    * Même support spatial et même combinaison AP/PA que le mode m.
    * Numérateur : taux après TEW; dénominateur : photopeak brut.
    * Disponible dans les modes TEW seulement.

Interprétation :

    Une faible fraction signifie que la TEW retire une grande proportion des
    comptes du photopeak. Une fraction élevée correspond à une correction de
    diffusé plus faible.

Avantage :

    Résume directement l'importance relative de la correction TEW.

Limite :

    R_TEW apparaît aussi dans la définition de la cible S_eff,TEW. La relation
    peut donc être partiellement mécanique et doit être interprétée prudemment.


M6 — FRACTION DU PHOTOPEAK DANS LES QUATRE FENÊTRES
---------------------------------------------------

Variable :

    x1 = Q_m(P) / (Q_m(L) + Q_m(P) + Q_m(U) + Q_m(G))

Forme :

    log(S_eff) = β0 + β1*x1

Construction :

    * Support spatial et combinaison AP/PA déterminés par le mode m.
    * Utilise simultanément les quatre fenêtres acquises.
    * Normalise chaque fenêtre par sa largeur énergétique et la durée.

Interprétation :

    Représente la fraction de tous les comptes détectés qui appartient au
    photopeak.

Avantage :

    Variable bornée, simple et indépendante du calcul TEW.

Limite :

    La variable résume les quatre fenêtres en une seule fraction et peut masquer
    des comportements différents de la fenêtre inférieure et supérieure.


M7 — FENÊTRE INFÉRIEURE/PHOTOPEAK
---------------------------------

Variable :

    x1 = log(Q_m(L) / Q_m(P))

Forme :

    log(S_eff) = β0 + β1*x1

Construction :

    * Support spatial et combinaison AP/PA déterminés par le mode m.
    * Normalisation par largeur énergétique et durée.
    * Utilise seulement la petite fenêtre inférieure.

Interprétation :

    Teste spécifiquement l'information de diffusé située sous le photopeak.

Avantage :

    Permet d'isoler l'apport de la fenêtre inférieure.

Limite :

    Plus faible statistique de comptage qu'une grande fenêtre énergétique.


M8 — FENÊTRE SUPÉRIEURE/PHOTOPEAK
---------------------------------

Variable :

    x1 = log(Q_m(U) / Q_m(P))

Forme :

    log(S_eff) = β0 + β1*x1

Construction :

    * Support spatial et combinaison AP/PA déterminés par le mode m.
    * Normalisation par largeur énergétique et durée.
    * Utilise seulement la petite fenêtre supérieure.

Interprétation :

    Teste spécifiquement l'information spectrale située au-dessus du photopeak.

Avantage :

    Comparé à M7, permet de déterminer quel côté du photopeak paraît le plus
    informatif.

Limite :

    Plus faible statistique de comptage et sensibilité possible au bruit ou à
    d'autres émissions dans cette région énergétique.


M9 — RATIO BRUT ET ASYMÉTRIE TOTALE AP/PA
-----------------------------------------

Variables :

    x1 = log(Q_m(P) / Q_m(G))

    x2 = |0,5*log(Q_m(C_AP,4 fenêtres) / Q_m(C_PA,4 fenêtres))|

Forme :

    log(S_eff) = β0 + β1*x1 + β2*x2

Construction :

    * Support spatial déterminé par le mode m.
    * x1 utilise le rapport photopeak/General Scatter.
    * x2 compare les comptes AP et PA sommés dans les quatre fenêtres.

Interprétation :

    Combine la composition spectrale avec une mesure globale de l'asymétrie
    entre les deux détecteurs.

Avantage :

    Combine deux informations planaires tout en respectant la géométrie choisie
    par le mode.

Limites :

    Modèle exploratoire à deux variables. L'asymétrie globale peut être
    influencée par le positionnement, la distribution de l'activité et les
    différences instrumentales entre les deux détecteurs.


RÉSUMÉ DES FAMILLES
-------------------

    M0                  : référence constante, aucun prédicteur.
    M1, M4, M7, M8      : ratios de densité spectrale adaptés au mode.
    M2                  : M1 plus asymétrie AP/PA du photopeak.
    M5                  : importance relative de la correction TEW, modes TEW.
    M6                  : fraction photopeak des quatre densités spectrales.
    M9                  : rapport photopeak/General Scatter plus asymétrie AP/PA
                          des quatre fenêtres.

    Modèles à 0 variable : M0.
    Modèles à 1 variable : M1, M4, M5, M6, M7 et M8.
    Modèles à 2 variables : M2 et M9.

Tous les modèles demeurent peu flexibles et relativement interprétables. M0
sert de référence. M1 est le modèle historique verrouillé. M2 et M9 testent
l'ajout d'une information AP/PA. Aucun de ces modèles n'utilise le Lasso, un
arbre, un SVM ou un réseau neuronal.
Aucun de ces modèles n'est un modèle flexible de type arbre ou réseau profond.


LIMITES COMMUNES ACTUELLES
--------------------------

    * Seulement 11 observations provenant de trois cycles et deux patients.
    * Les timepoints d'un même patient ne sont pas indépendants.
    * Les modèles doivent être évalués sur des cycles ou patients exclus de
      l'entraînement; un partage aléatoire des timepoints serait optimiste.
    * Choisir le meilleur modèle sur les mêmes données utilisées pour annoncer
      sa performance introduirait un biais de sélection.
    * Les modèles utilisant le crop par profil ne constituent pas encore un
      pipeline entièrement planaire, puisque ce crop est guidé par le Q/SPECT.
    * Une validation sur des patients supplémentaires demeure nécessaire avant
      de conclure à une capacité de généralisation.
"""


def _mode_metric_lookup(
    metrics_rows: Sequence[Mapping[str, Any]],
    rate_mode: str,
    split_id: str,
    model_name: str,
) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in metrics_rows
            if row.get("rate_mode") == rate_mode
            and row["split_id"] == split_id
            and row["model"] == model_name
            and row["metric_scope"] == "combined_test"
        ),
        None,
    )


def plot_mode_comparison_primary_mare(
    metrics_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    """Compare rate modes with predictor models available in every mode."""
    modes = tuple(RATE_MODES)
    models = COMMON_MODE_MODELS
    matrix = np.full((len(modes), len(models)), np.nan, dtype=np.float64)
    for row_index, rate_mode in enumerate(modes):
        for column_index, model_name in enumerate(models):
            row = _mode_metric_lookup(
                metrics_rows, rate_mode, "loco_holdout_p8_june", model_name
            )
            if row is not None and row["fit_status"] == "fitted":
                matrix[row_index, column_index] = float(
                    row["activity_mare_percent"]
                )
    finite = matrix[np.isfinite(matrix)]
    vmax = max(10.0, float(np.percentile(finite, 90)) if finite.size else 10.0)
    fig, axis = plt.subplots(
        figsize=(15.5, max(6.6, 1.05 * len(modes) + 2.0)),
        layout="constrained",
        facecolor="white",
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
    )
    axis.set_xticks(range(len(models)), [model.split("_")[0] for model in models])
    axis.set_yticks(
        range(len(modes)), [RATE_MODES[mode].short_label for mode in modes]
    )
    axis.set_xlabel(
        "Modèles comparables dans tous les modes (M5 est réservé aux modes TEW)",
        fontsize=9.5,
    )
    axis.set_title(
        "Comparaison des modes de comptage\n"
        "Entraînement P11-mai + P11-juillet — test P8-juin"
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
                fontsize=10,
                fontweight="bold",
                color=(
                    "white"
                    if np.isfinite(value) and min(value, vmax) > 0.58 * vmax
                    else "black"
                ),
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("MARE sur l’activité (%)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tew_vs_raw_crop_primary_mare(
    metrics_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    """Isolate TEW by comparing matched crop/GM pipelines."""
    modes = (MODE_TEW_CROP_GM, MODE_RAW_CROP_GM)
    models = MODELS_WITHOUT_TEW_INPUT
    matrix = np.full((len(modes), len(models)), np.nan, dtype=np.float64)
    for row_index, rate_mode in enumerate(modes):
        for column_index, model_name in enumerate(models):
            row = _mode_metric_lookup(
                metrics_rows, rate_mode, "loco_holdout_p8_june", model_name
            )
            if row is not None and row["fit_status"] == "fitted":
                matrix[row_index, column_index] = float(
                    row["activity_mare_percent"]
                )
    finite = matrix[np.isfinite(matrix)]
    vmax = max(10.0, float(np.percentile(finite, 90)) if finite.size else 10.0)
    fig, axis = plt.subplots(
        figsize=(15.0, 4.7), layout="constrained", facecolor="white"
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
    )
    axis.set_xticks(
        range(len(models)),
        [MODEL_LABELS[model].split(" — ")[0] for model in models],
    )
    axis.set_yticks(
        range(len(modes)), [RATE_MODES[mode].short_label for mode in modes]
    )
    axis.set_xlabel("Même modèle prédictif; M5 exclu puisqu’il utilise la TEW")
    axis.set_title(
        "Effet isolé de la TEW : même crop et même moyenne géométrique\n"
        "Entraînement P11-mai + P11-juillet — test P8-juin"
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
                fontsize=9.5,
                fontweight="bold",
                color=(
                    "white"
                    if np.isfinite(value) and min(value, vmax) > 0.58 * vmax
                    else "black"
                ),
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.80)
    colorbar.set_label("MARE sur l’activité (%)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_factorial_m0_mare(
    metrics_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    """Isolate crop, TEW and AP/PA-combination effects for the M0 baseline."""
    processing_labels = ("Crop + GM", "Image complète + GM", "Image complète + AP+PA")
    tew_modes = (MODE_TEW_CROP_GM, MODE_TEW_FULL_GM, MODE_TEW_FULL_AP_PA)
    raw_modes = (MODE_RAW_CROP_GM, MODE_RAW_FULL_GM, MODE_RAW_FULL_AP_PA)

    def values_for(modes: Sequence[str]) -> list[float]:
        values: list[float] = []
        for rate_mode in modes:
            row = _mode_metric_lookup(
                metrics_rows,
                rate_mode,
                "loco_holdout_p8_june",
                MODEL_CONSTANT,
            )
            values.append(
                np.nan
                if row is None or row["fit_status"] != "fitted"
                else float(row["activity_mare_percent"])
            )
        return values

    tew_values = values_for(tew_modes)
    raw_values = values_for(raw_modes)
    x = np.arange(len(processing_labels), dtype=np.float64)
    width = 0.36
    fig, axis = plt.subplots(
        figsize=(10.8, 6.4), layout="constrained", facecolor="white"
    )
    tew_bars = axis.bar(
        x - width / 2.0,
        tew_values,
        width,
        color="#2E9F55",
        label="Avec TEW",
    )
    raw_bars = axis.bar(
        x + width / 2.0,
        raw_values,
        width,
        color="#1F77B4",
        label="Photopeak brut",
    )
    for bars, values in ((tew_bars, tew_values), (raw_bars, raw_values)):
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            axis.annotate(
                f"{value:.1f} %",
                (bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    axis.set_xticks(x, processing_labels)
    axis.set_ylabel("MARE sur l’activité (%)")
    axis.set_title(
        "Décomposition du pipeline avec la sensibilité constante M0\n"
        "Entraînement P11-mai + P11-juillet — test P8-juin"
    )
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_mode_comparison_primary_predictions(
    prediction_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    """Show P8 activity ratios for the common M0 and M1 models by rate mode."""
    modes = tuple(RATE_MODES)
    compared_models = (MODEL_CONSTANT, MODEL_LOCKED)
    n_columns = 2
    n_rows = (len(modes) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12.8, 3.9 * n_rows),
        facecolor="white",
    )
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    for axis, rate_mode in zip(flat_axes, modes):
        selected = [
            row
            for row in prediction_rows
            if row.get("rate_mode") == rate_mode
            and row["split_id"] == "loco_holdout_p8_june"
        ]
        axis.axhspan(0.90, 1.10, color="#2E9F55", alpha=0.10)
        axis.axhline(1.0, color="black", linewidth=1.3, label="Q/SPECT")
        timepoints = [
            str(row["timepoint"])
            for row in selected
            if row["model"] == MODEL_CONSTANT
        ]
        x = np.arange(len(timepoints), dtype=np.float64)
        for model_name in compared_models:
            model_rows = [row for row in selected if row["model"] == model_name]
            model_rows.sort(key=lambda row: float(row["planar_day"]))
            if not model_rows:
                continue
            axis.plot(
                x,
                [float(row["predicted_over_qspect"]) for row in model_rows],
                marker=MODEL_MARKERS[model_name],
                linewidth=2.0,
                markersize=7,
                color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name],
            )
        axis.set_xticks(x, timepoints)
        axis.set_title(RATE_MODES[rate_mode].short_label)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in flat_axes[len(modes) :]:
        axis.set_visible(False)
    axes_array = np.asarray(axes, dtype=object).reshape(n_rows, n_columns)
    for axis in axes_array[:, 0]:
        axis.set_ylabel("Activité prédite / activité Q/SPECT")
    for axis in axes_array[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Timepoint de P8")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Comparaison des modes sur le patient exclu P8\n"
        "M0 constant et M1 ratio General Scatter/photopeak",
        fontsize=15,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.09,
        top=0.91,
        hspace=0.38,
        wspace=0.10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_six_mode_primary_mare(
    metrics_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    """Place the six factorial mode-specific MARE summaries in a 2x3 grid."""
    modes = FACTORIAL_COMPARISON_MODES
    processing_labels = (
        "Crop + GM",
        "Image complète + GM",
        "Image complète + AP+PA",
    )
    row_labels = ("Avec TEW", "Photopeak brut")
    row_colors = ("#2E9F55", "#1F77B4")
    y = np.arange(len(MODEL_ORDER), dtype=np.float64)
    values_by_mode: Dict[str, Dict[str, float]] = {}
    finite_values: list[float] = []
    for rate_mode in modes:
        mode_values: Dict[str, float] = {}
        for model_name in RATE_MODES[rate_mode].model_names:
            row = _mode_metric_lookup(
                metrics_rows,
                rate_mode,
                "loco_holdout_p8_june",
                model_name,
            )
            if row is not None and row["fit_status"] == "fitted":
                value = float(row["activity_mare_percent"])
                mode_values[model_name] = value
                finite_values.append(value)
        values_by_mode[rate_mode] = mode_values

    x_max = max(finite_values, default=10.0) * 1.20
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(19.5, 14.0),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    for panel_index, (axis, rate_mode) in enumerate(zip(flat_axes, modes)):
        mode_values = values_by_mode[rate_mode]
        for row_index, model_name in enumerate(MODEL_ORDER):
            if model_name not in mode_values:
                continue
            value = mode_values[model_name]
            axis.barh(
                row_index,
                value,
                color=MODEL_COLORS[model_name],
                height=0.68,
            )
            axis.annotate(
                f"{value:.1f} %",
                (value, row_index),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=11.5,
                fontweight="bold",
            )
        axis.set_xlim(0.0, x_max)
        row_index, column_index = divmod(panel_index, 3)
        axis.set_title(
            processing_labels[column_index], fontsize=17, pad=34
        )
        axis.text(
            0.5,
            1.01,
            row_labels[row_index],
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=14.5,
            fontweight="bold",
            color=row_colors[row_index],
        )
        axis.set_xlabel("MARE sur l’activité (%)", fontsize=14)
        axis.tick_params(axis="both", labelsize=12.5)
        axis.grid(axis="x", linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    flat_axes[0].set_yticks(
        y, [MODEL_LABELS[model].split(" — ")[0] for model in MODEL_ORDER]
    )
    flat_axes[0].invert_yaxis()
    axes_array = np.asarray(axes, dtype=object).reshape(2, 3)
    for axis in axes_array[:, 0]:
        axis.tick_params(labelleft=True)
    fig.suptitle(
        "Performance des modèles selon les six modes principaux\n"
        "Entraînement P11-mai + P11-juillet — test P8-juin",
        fontsize=21,
        y=0.985,
    )
    fig.text(
        0.5,
        0.02,
        "Les lignes vides indiquent un modèle non compatible avec ce mode. "
        "Le mode quatre fenêtres demeure dans l’analyse exploratoire séparée.",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.085,
        top=0.84,
        hspace=0.31,
        wspace=0.12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_six_mode_primary_predictions(
    prediction_rows: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    """Place the six factorial mode-specific P8 predictions in a 2x3 grid."""
    modes = FACTORIAL_COMPARISON_MODES
    processing_labels = (
        "Crop + GM",
        "Image complète + GM",
        "Image complète + AP+PA",
    )
    row_labels = ("Avec TEW", "Photopeak brut")
    row_colors = ("#2E9F55", "#1F77B4")
    split_id = "loco_holdout_p8_june"
    plotted_ratios: list[float] = []
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(20.0, 10.4),
        sharey=True,
        facecolor="white",
    )
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    for panel_index, (axis, rate_mode) in enumerate(zip(flat_axes, modes)):
        selected = [
            row
            for row in prediction_rows
            if row.get("rate_mode") == rate_mode
            and row["split_id"] == split_id
        ]
        timepoints = [
            str(row["timepoint"])
            for row in selected
            if row["model"] == MODEL_CONSTANT
        ]
        x = np.arange(len(timepoints), dtype=np.float64)
        axis.axhspan(
            0.90,
            1.10,
            color="#2E9F55",
            alpha=0.10,
            label="Écart de ±10 %",
        )
        axis.axhline(1.0, color="black", linewidth=1.3, label="Q/SPECT")
        plotted_models = tuple(
            model
            for model in PRIMARY_PLOT_MODELS
            if model in RATE_MODES[rate_mode].model_names
        )
        for model_name in plotted_models:
            model_rows = [
                row for row in selected if row["model"] == model_name
            ]
            model_rows.sort(key=lambda row: float(row["planar_day"]))
            if not model_rows:
                continue
            ratios = [
                float(row["predicted_over_qspect"]) for row in model_rows
            ]
            plotted_ratios.extend(ratios)
            axis.plot(
                x,
                ratios,
                marker=MODEL_MARKERS[model_name],
                linewidth=2.1,
                markersize=7.5,
                color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name].split(" — ")[0],
            )
            for position, value, row in zip(x, ratios, model_rows):
                if bool(row["feature_extrapolation"]):
                    axis.scatter(
                        [position],
                        [value],
                        marker=MODEL_MARKERS[model_name],
                        s=82,
                        facecolor="white",
                        edgecolor=MODEL_COLORS[model_name],
                        linewidth=1.9,
                        zorder=5,
                    )
        axis.set_xticks(x, timepoints)
        axis.set_xlabel("Timepoint de P8", fontsize=14.5)
        axis.tick_params(axis="both", labelsize=13)
        row_index, column_index = divmod(panel_index, 3)
        axis.set_title(
            processing_labels[column_index], fontsize=17, pad=34
        )
        axis.text(
            0.5,
            1.01,
            row_labels[row_index],
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=14.5,
            fontweight="bold",
            color=row_colors[row_index],
        )
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes_array = np.asarray(axes, dtype=object).reshape(2, 3)
    for axis in axes_array[:, 0]:
        axis.set_ylabel(
            "Activité prédite / activité Q/SPECT", fontsize=14.5
        )
        axis.tick_params(labelleft=True)
    if plotted_ratios:
        y_min = min(0.88, min(plotted_ratios) - 0.035)
        y_max = max(1.12, max(plotted_ratios) + 0.035)
        flat_axes[0].set_ylim(y_min, y_max)
    handles: list[Any] = []
    labels: list[str] = []
    for axis in flat_axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        fontsize=11.8,
    )
    fig.suptitle(
        "Prédictions sur le patient exclu P8 selon les six modes principaux\n"
        "Entraînement sur les deux cycles de P11",
        fontsize=21,
        y=0.985,
    )
    fig.text(
        0.985,
        0.115,
        "Marqueur ouvert : variable hors de l’intervalle d’entraînement",
        ha="right",
        va="bottom",
        fontsize=11.5,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.805,
        hspace=0.58,
        wspace=0.10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_mode_comparison_report(
    metrics_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "Comparaison des modes de comptage — v3",
        "=======================================",
        "",
        "Primary split: training P11 May + P11 July; test P8 June.",
        "Values are activity MARE (%). Comparisons use the same predictor model.",
        "",
        "mode | " + " | ".join(MODEL_LABELS[m].split(" — ")[0] for m in COMMON_MODE_MODELS),
        "---- | " + " | ".join("-----" for _ in COMMON_MODE_MODELS),
    ]
    for rate_mode, mode_spec in RATE_MODES.items():
        values = []
        for model_name in COMMON_MODE_MODELS:
            row = _mode_metric_lookup(
                metrics_rows, rate_mode, "loco_holdout_p8_june", model_name
            )
            values.append(
                "N/E"
                if row is None
                else _format_metric(row["activity_mare_percent"])
            )
        lines.append(f"{mode_spec.short_label} | " + " | ".join(values))
    lines.extend(
        [
            "",
            "Best observed model within each mode (post hoc, not an unbiased model selection):",
        ]
    )
    for rate_mode, mode_spec in RATE_MODES.items():
        candidates = [
            row
            for row in metrics_rows
            if row.get("rate_mode") == rate_mode
            and row["split_id"] == "loco_holdout_p8_june"
            and row["metric_scope"] == "combined_test"
            and row["fit_status"] == "fitted"
            and row["model"] in mode_spec.model_names
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: float(row["activity_mare_percent"]))
        lines.append(
            f"- {mode_spec.short_label}: {best['model_label']} = "
            f"{float(best['activity_mare_percent']):.2f}% MARE; "
            f"{int(best['extrapolated_predictions'])}/{int(best['prediction_count'])} "
            "predictions in extrapolation."
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- TEW crop GM versus raw photopeak crop GM isolates the TEW step.",
            "- TEW full GM versus raw full GM isolates TEW without a crop.",
            "- TEW full AP+PA versus raw full AP+PA isolates TEW without GM.",
            "- Crop GM versus full GM isolates the crop for the same scatter mode.",
            "- Full GM versus full AP+PA isolates the AP/PA combination.",
            "- Four-window AP+PA is exploratory because energy-window widths differ.",
            "- Every predictor uses the support and AP/PA combination of its mode.",
            "- M5 is available only in TEW modes and is matched to each geometry.",
            "- These are grouped transfer results from only two patients.",
            "",
        ]
    )
    return "\n".join(lines)


def run_default_analysis(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_observations = load_base_observation_cache()
    shared_dir = output_dir / "shared_inputs"
    comparison_dir = output_dir / "comparison_modes"
    shared_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows = [
        {
            "dataset_key": key,
            "patient_id": DATASETS[key].patient_id,
            "cycle_id": DATASETS[key].cycle_id,
            "label": DATASETS[key].label,
            "n_observations": len(base_observations[key]["records"]),
            "planar_dir": str(DATASETS[key].planar_dir),
            "qspect_dir": str(DATASETS[key].qspect_dir),
        }
        for key in DATASETS
    ]
    _write_csv(shared_dir / "v3_dataset_inventory.csv", inventory_rows)
    methodology_path = output_dir / "v3_modeles_M0_M9_methodologie.txt"
    methodology_path.write_text(
        build_model_methodology_document(), encoding="utf-8"
    )
    all_results: Dict[str, list[Dict[str, Any]]] = {}
    observations_by_mode: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_metrics: list[Dict[str, Any]] = []
    all_predictions: list[Dict[str, Any]] = []

    for rate_mode, mode_spec in RATE_MODES.items():
        observations = select_rate_mode_observations(
            base_observations, rate_mode
        )
        observations_by_mode[rate_mode] = observations
        mode_results: list[Dict[str, Any]] = []
        metrics_rows: list[Dict[str, Any]] = []
        prediction_rows: list[Dict[str, Any]] = []
        for split in default_split_definitions():
            result = fit_and_evaluate(
                split.training_datasets,
                split.test_datasets,
                observations,
                split_id=split.split_id,
                split_type=split.split_type,
                split_label=split.label,
                rate_mode=rate_mode,
                model_names=mode_spec.model_names,
            )
            mode_results.append(result)
            metrics_rows.extend(result["metrics"])
            prediction_rows.extend(result["predictions"])
        all_results[rate_mode] = mode_results
        all_metrics.extend(metrics_rows)
        all_predictions.extend(prediction_rows)

        mode_dir = output_dir / rate_mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        feature_rows = []
        for dataset_key in DATASETS:
            observation = observations[dataset_key]
            for index, record in enumerate(observation["records"]):
                feature_rows.append(
                    {
                        "rate_mode": rate_mode,
                        "rate_mode_label": mode_spec.label,
                        **record,
                        **{
                            name: float(observation["features"][name][index])
                            for name in FEATURE_ORDER
                        },
                        "measurement_rate_cps": float(
                            observation["measurement_rate"][index]
                        ),
                        "observed_effective_sensitivity_cps_per_mbq": float(
                            observation["target_sensitivity"][index]
                        ),
                        "qspect_activity_mbq": float(
                            observation["reference_activity"][index]
                        ),
                    }
                )
        _write_csv(mode_dir / "v3_feature_values.csv", feature_rows)
        _write_csv(mode_dir / "v3_split_metrics.csv", metrics_rows)
        _write_csv(mode_dir / "v3_predictions.csv", prediction_rows)
        (mode_dir / "v3_report.txt").write_text(
            build_report(
                observations,
                metrics_rows,
                rate_mode=rate_mode,
                model_names=mode_spec.model_names,
            ),
            encoding="utf-8",
        )
        plot_pairwise_transfer_matrix(
            metrics_rows,
            mode_dir / "v3_pairwise_transfer_matrix.png",
            model_names=mode_spec.model_names,
            title_prefix=mode_spec.short_label,
        )
        plot_leave_one_cycle_out(
            metrics_rows,
            mode_dir / "v3_leave_one_cycle_out.png",
            model_names=mode_spec.model_names,
            title_prefix=mode_spec.short_label,
        )
        primary_plot_models = tuple(
            model
            for model in PRIMARY_PLOT_MODELS
            if model in mode_spec.model_names
        )
        plot_primary_patient_holdout(
            prediction_rows,
            mode_dir / "v3_primary_p11_to_p8.png",
            model_names=primary_plot_models,
            title_prefix=mode_spec.short_label,
        )
        plot_primary_model_mare(
            metrics_rows,
            mode_dir / "v3_primary_p11_to_p8_model_mare.png",
            model_names=mode_spec.model_names,
            title_prefix=mode_spec.short_label,
        )
        plot_feature_overview(
            observations,
            mode_dir / "v3_candidate_feature_overview.png",
            sensitivity_label="Sensibilité effective du mode (cps/MBq)",
            title_prefix=mode_spec.short_label,
        )

    plot_feature_correlation_matrix(
        observations_by_mode[MODE_TEW_CROP_GM],
        shared_dir / "v3_candidate_feature_correlations.png",
    )
    _write_csv(comparison_dir / "v3_all_modes_metrics.csv", all_metrics)
    _write_csv(comparison_dir / "v3_all_modes_predictions.csv", all_predictions)
    (comparison_dir / "v3_mode_comparison_report.txt").write_text(
        build_mode_comparison_report(all_metrics), encoding="utf-8"
    )
    plot_mode_comparison_primary_mare(
        all_metrics, comparison_dir / "v3_mode_comparison_primary_mare.png"
    )
    plot_tew_vs_raw_crop_primary_mare(
        all_metrics, comparison_dir / "v3_tew_vs_raw_crop_primary_mare.png"
    )
    plot_factorial_m0_mare(
        all_metrics, comparison_dir / "v3_factorial_m0_mare.png"
    )
    plot_mode_comparison_primary_predictions(
        all_predictions,
        comparison_dir / "v3_mode_comparison_primary_predictions.png",
    )
    plot_six_mode_primary_mare(
        all_metrics,
        comparison_dir / "v3_six_main_modes_primary_model_mare.png",
    )
    plot_six_mode_primary_predictions(
        all_predictions,
        comparison_dir / "v3_six_main_modes_primary_predictions.png",
    )
    readme_lines = [
        "V3 multimode output organization",
        "================================",
        "",
        "Each rate-mode folder contains its own values, metrics, predictions,",
        "report and validation figures.",
        "",
    ]
    for rate_mode, mode_spec in RATE_MODES.items():
        readme_lines.append(f"{rate_mode}: {mode_spec.description}")
    readme_lines.extend(
        [
            "",
            "comparison_modes: fair cross-mode figures and combined CSV files.",
            "shared_inputs: dataset inventory and predictor correlation figure.",
            "",
            "The two six-panel figures compare the six primary modes in a 2x3",
            "grid. The four-window mode remains a separate exploratory mode.",
            "",
            "The added full-image modes form a factorial comparison of crop",
            "versus full image, TEW versus raw photopeak, and geometric mean",
            "versus AP+PA sum. The all-mode figures include these additions.",
            "",
            "The TEW-versus-raw-crop comparison changes only the scatter",
            "correction. Crop/full-image and GM/AP+PA comparisons isolate their",
            "respective processing choices when the scatter mode is fixed.",
            "Every predictor is recomputed on the geometry selected by its mode.",
            "M5 is available only in TEW modes.",
        ]
    )
    (output_dir / "README.txt").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8"
    )
    return {
        "base_observations": base_observations,
        "observations_by_mode": observations_by_mode,
        "results_by_mode": all_results,
        "metrics": all_metrics,
        "predictions": all_predictions,
        "output_dir": output_dir,
        "comparison_dir": comparison_dir,
        "methodology_path": methodology_path,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--train",
        action="append",
        choices=tuple(DATASETS),
        help="training dataset key; repeat to pool cycles",
    )
    parser.add_argument(
        "--test",
        action="append",
        choices=tuple(DATASETS),
        help="test dataset key; repeat to combine held-out cycles",
    )
    parser.add_argument(
        "--rate-mode",
        choices=tuple(RATE_MODES),
        default=MODE_TEW_CROP_GM,
        help="measurement-rate definition for a custom train/test run",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (args.train is None) != (args.test is None):
        raise ValueError("--train and --test must be provided together")
    if args.train is None:
        result = run_default_analysis(args.output_dir)
        print(f"Saved v3 outputs to {result['output_dir']}")
        return

    training, test = validate_split(args.train, args.test)
    observations = load_observation_cache(
        (*training, *test), rate_mode=args.rate_mode
    )
    mode_spec = RATE_MODES[args.rate_mode]
    result = fit_and_evaluate(
        training,
        test,
        observations,
        split_id="custom",
        split_type="custom",
        split_label=(
            " + ".join(DATASETS[key].short_label for key in training)
            + " → "
            + " + ".join(DATASETS[key].short_label for key in test)
        ),
        rate_mode=args.rate_mode,
        model_names=mode_spec.model_names,
    )
    custom_dir = args.output_dir / args.rate_mode
    custom_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(custom_dir / "v3_custom_metrics.csv", result["metrics"])
    _write_csv(custom_dir / "v3_custom_predictions.csv", result["predictions"])
    (args.output_dir / "v3_modeles_M0_M9_methodologie.txt").write_text(
        build_model_methodology_document(), encoding="utf-8"
    )
    print(f"Saved custom v3 outputs to {custom_dir}")


if __name__ == "__main__":
    main()
