from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.special import lambertw


PARALYZABLE_DOMAIN_LIMIT = 1.0 / np.e
SMALL_A_THRESHOLD = 1e-12


@dataclass
class DeadTimeCorrectionResult:
    dtcf: float
    rpo_observed_cps: float
    rpt_corrected_cps: float
    recovered_activity_mbq: float
    count_loss_fraction: float
    count_loss_percent: float
    corrected_image: Optional[np.ndarray]
    warnings: list[str]


def _validate_finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _validate_finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def dead_time_correction_factor(
    rwo_cps: float,
    tau_us: float,
    max_calibrated_rwo_cps: float | None = None,
) -> tuple[float, list[str]]:
    """Return Frezza paralyzable dead-time correction factor.

    DTCF solves 1 = DTCF * exp(-DTCF * RWo * tau), using the principal
    Lambert W branch. RWo and tau must come from the same wide-window
    calibration definition.
    """
    warnings: list[str] = []
    rwo_cps = _validate_finite_nonnegative("rwo_cps", rwo_cps)
    tau_us = _validate_finite_positive("tau_us", tau_us)

    if max_calibrated_rwo_cps is not None:
        max_calibrated_rwo_cps = _validate_finite_positive(
            "max_calibrated_rwo_cps",
            max_calibrated_rwo_cps,
        )
        if rwo_cps > max_calibrated_rwo_cps:
            raise ValueError(
                f"rwo_cps={rwo_cps:g} exceeds max_calibrated_rwo_cps={max_calibrated_rwo_cps:g}"
            )

    tau_seconds = tau_us * 1e-6
    a = rwo_cps * tau_seconds
    if a < SMALL_A_THRESHOLD:
        return 1.0, warnings
    if a >= PARALYZABLE_DOMAIN_LIMIT:
        raise ValueError(
            "Dead-time inversion is outside the usable paralyzable-model domain: "
            f"RWo*tau={a:.6g} >= 1/e={PARALYZABLE_DOMAIN_LIMIT:.6g}"
        )

    dtcf = float((-lambertw(-a, k=0).real) / a)
    if not np.isfinite(dtcf) or dtcf < 1.0:
        raise RuntimeError(f"Invalid DTCF computed from Lambert W: {dtcf}")

    if a > 0.2:
        warnings.append(
            f"High dead-time load: RWo*tau={a:.3g}. Verify this is inside the calibrated range."
        )

    return dtcf, warnings


def correct_dead_time(
    rpo_cps: float,
    rwo_cps: float,
    tau_us: float,
    calibration_factor_cps_per_mbq: float,
    image_primary: np.ndarray | None = None,
    max_calibrated_rwo_cps: float | None = None,
) -> DeadTimeCorrectionResult:
    """Correct primary Lu-177 planar count rate using Frezza et al. DTCF.

    The DTCF is estimated from the observed wide-spectrum rate RWo, then applied
    to the observed primary rate RPo and optionally to the primary image. It does
    not directly correct RWo.
    """
    rpo_cps = _validate_finite_nonnegative("rpo_cps", rpo_cps)
    calibration_factor_cps_per_mbq = _validate_finite_positive(
        "calibration_factor_cps_per_mbq",
        calibration_factor_cps_per_mbq,
    )
    dtcf, warnings = dead_time_correction_factor(
        rwo_cps,
        tau_us,
        max_calibrated_rwo_cps=max_calibrated_rwo_cps,
    )

    corrected_image = None
    if image_primary is not None:
        image = np.asarray(image_primary, dtype=np.float64)
        if not np.all(np.isfinite(image)):
            raise ValueError("image_primary must contain only finite values")
        if np.any(image < 0):
            raise ValueError("image_primary contains negative values")
        corrected_image = image * dtcf

    rpt_corrected_cps = dtcf * rpo_cps
    recovered_activity_mbq = rpt_corrected_cps / calibration_factor_cps_per_mbq
    count_loss_fraction = 1.0 - 1.0 / dtcf

    return DeadTimeCorrectionResult(
        dtcf=dtcf,
        rpo_observed_cps=rpo_cps,
        rpt_corrected_cps=rpt_corrected_cps,
        recovered_activity_mbq=recovered_activity_mbq,
        count_loss_fraction=count_loss_fraction,
        count_loss_percent=100.0 * count_loss_fraction,
        corrected_image=corrected_image,
        warnings=warnings,
    )


def correct_dead_time_dataframe(
    data: pd.DataFrame,
    max_calibrated_rwo_cps: float | None = None,
) -> pd.DataFrame:
    """Correct multiple time points from columns timepoint, rpo_cps, rwo_cps, tau_us, CF_cps_per_MBq."""
    required_columns = {"timepoint", "rpo_cps", "rwo_cps", "tau_us", "CF_cps_per_MBq"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for _idx, row in data.iterrows():
        result = correct_dead_time(
            rpo_cps=row["rpo_cps"],
            rwo_cps=row["rwo_cps"],
            tau_us=row["tau_us"],
            calibration_factor_cps_per_mbq=row["CF_cps_per_MBq"],
            max_calibrated_rwo_cps=max_calibrated_rwo_cps,
        )
        rows.append(
            {
                "timepoint": row["timepoint"],
                "rpo_observed_cps": result.rpo_observed_cps,
                "rwo_cps": float(row["rwo_cps"]),
                "tau_us": float(row["tau_us"]),
                "CF_cps_per_MBq": float(row["CF_cps_per_MBq"]),
                "dtcf": result.dtcf,
                "rpt_corrected_cps": result.rpt_corrected_cps,
                "recovered_activity_mbq": result.recovered_activity_mbq,
                "count_loss_fraction": result.count_loss_fraction,
                "count_loss_percent": result.count_loss_percent,
                "warnings": "; ".join(result.warnings),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    example = pd.DataFrame(
        {
            "timepoint": ["D0", "D1", "D3"],
            "rpo_cps": [3301.7, 819.3, 265.1],
            "rwo_cps": [95000.0, 24000.0, 8000.0],
            "tau_us": [0.632, 0.632, 0.632],
            "CF_cps_per_MBq": [9.36, 9.36, 9.36],
        }
    )
    corrected = correct_dead_time_dataframe(example)
    print(corrected.to_string(index=False))
