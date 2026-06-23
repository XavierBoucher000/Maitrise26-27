"""
Utilities for scatter correction using TEW/DEW approaches.

This module provides:
- helpers to extract counts per energy window from loaded image dicts
- TEW and DEW estimators that compute the scatter contribution under
  the photopeak and return a corrected photopeak count

NOTE: Widths (W_L, W_M, W_U) are expected as inputs to TEW. This file
intentionally does NOT implement automatic extraction/estimation of the
window widths — leave that section to be implemented later or provided
by the caller.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np


def extract_counts_from_images(images: List[Dict]) -> Dict[str, float]:
    """Aggregate counts from a list of image metadata dicts.

    Expects each item to contain `energy_window` or `energy_window_name`
    and `pixel_sum` (total counts). Returns a mapping from normalized
    window labels to summed counts.

    Normalized labels used: 'Lower Scatter', 'Photopeak', 'Upper Scatter',
    or the original label if it does not match known names.
    """
    counts: Dict[str, float] = {}
    for item in images:
        label = item.get("energy_window") or item.get("energy_window_name") or "Unknown"
        l = label
        if isinstance(label, str):
            l_low = label.lower()
            if "lower" in l_low:
                l = "Lower Scatter"
            elif "upper" in l_low:
                l = "Upper Scatter"
            elif "photo" in l_low or "lutetium" in l_low or "177" in l_low or "main" in l_low:
                l = "Photopeak"

        value = item.get("pixel_sum") or 0.0
        try:
            value = float(value)
        except Exception:
            value = 0.0
        counts[l] = counts.get(l, 0.0) + value

    return counts


def tew_scatter_estimate(
    C_L: float,
    C_M: float,
    C_U: float,
    W_L: float,
    W_M: float,
    W_U: float,
) -> Tuple[float, float]:
    """Estimate scatter under the photopeak using the TEW formula.

    S = ((C_L/W_L + C_U/W_U) / 2) * W_M
    C_M_corr = C_M - S

    Parameters must be provided in counts and keV.

    Returns (S, C_M_corr).
    """
    if any(w is None for w in (W_L, W_M, W_U)):
        raise ValueError("Window widths W_L, W_M, W_U are required for TEW estimation")
    if W_L <= 0 or W_U <= 0 or W_M <= 0:
        raise ValueError("Window widths must be positive")

    S = ((C_L / W_L) + (C_U / W_U)) / 2.0 * W_M
    C_M_corr = C_M - S
    return float(S), float(C_M_corr)


def dew_scatter_estimate(C_L: float, C_M: float, k: float) -> Tuple[float, float]:
    """Estimate scatter with a Dual-Energy-Window (DEW) simple scaling.

    S = k * C_L
    C_M_corr = C_M - S

    `k` is an empirically determined factor from calibration.
    """
    S = float(k) * float(C_L)
    C_M_corr = float(C_M) - S
    return float(S), float(C_M_corr)


def build_discrete_spectrum(
    counts: Dict[str, float],
    centers: Optional[Dict[str, float]] = None,
    widths: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Build a simple discrete spectrum representation from window counts.

    - `counts`: mapping window label -> counts (as returned by extract_counts_from_images)
    - `centers`: optional mapping window label -> centre energy (keV). If not
      provided, centre energies will be set to None for each window.
    - `widths`: optional mapping window label -> width (keV). If not provided,
      `counts_per_keV` will be left as None.

    Returns a list of dicts with keys: 'label','counts','center','width','counts_per_keV'
    """
    out: List[Dict] = []
    for label, cnt in counts.items():
        center = None if centers is None else centers.get(label)
        width = None if widths is None else widths.get(label)
        counts_per_keV = None
        if width and width > 0:
            counts_per_keV = float(cnt) / float(width)
        out.append({
            "label": label,
            "counts": float(cnt),
            "center": center,
            "width": width,
            "counts_per_keV": counts_per_keV,
        })
    out_sorted = sorted(out, key=lambda x: (float('inf') if x['center'] is None else x['center']))
    return out_sorted


if __name__ == "__main__":
    sample_images = [
        {"energy_window": "Lower Scatter", "pixel_sum": 10000},
        {"energy_window": "Photopeak", "pixel_sum": 50000},
        {"energy_window": "Upper Scatter", "pixel_sum": 12000},
    ]
    counts = extract_counts_from_images(sample_images)
    print("Counts per window:", counts)
    try:
        S, Ccorr = tew_scatter_estimate(
            counts.get('Lower Scatter', 0.0),
            counts.get('Photopeak', 0.0),
            counts.get('Upper Scatter', 0.0),
            1.0,
            1.0,
            1.0,
        )
        print(f"TEW estimate with W=1: S={S}, corrected={Ccorr}")
    except Exception as exc:
        print("TEW estimate unavailable:", exc)
