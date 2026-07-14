from pathlib import Path
import csv
import math
import random
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Data" / "Lu177_approx_spectrum_from_image.csv"
OUT_DIR = ROOT / "fig" / "spectrum"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NOISE_SEED = 177
NOISE_LEVEL = 0.045


def load_csv_spectrum(path: Path):
    energies = []
    counts = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        count_column = next(
            (name for name in ["Approx_Counts", "Counts_approx", "Counts", "Intensity"] if name in fieldnames),
            fieldnames[1] if len(fieldnames) > 1 else None,
        )
        if count_column is None or "Energy_keV" not in fieldnames:
            raise ValueError(f"CSV must contain Energy_keV and a counts/intensity column: {path}")

        for row in reader:
            energies.append(float(row["Energy_keV"]))
            counts.append(float(row[count_column]))
    return energies, counts


def gaussian(x, center, sigma, amplitude):
    return amplitude * math.exp(-0.5 * ((x - center) / sigma) ** 2)


def build_presentation_spectrum(energies, counts):
    """Build a clean Lu-177 spectrum for presentation, using the CSV energy grid."""
    if not energies:
        return []

    max_count = max(counts) if counts else 1.0
    csv_relative = [c / max_count for c in counts]
    spectrum = []

    for energy, csv_value in zip(energies, csv_relative):
        if energy < 55:
            scatter = 1.1 + 3.4 * max(0.0, energy - 26.0) / 29.0
        elif energy <= 165:
            scatter = 4.5 - 0.018 * (energy - 55)
        elif energy <= 187.2:
            scatter = 2.6 - 0.035 * (energy - 165)
        elif energy <= 208.4:
            scatter = 1.7 - 0.02 * (energy - 187.2)
        else:
            scatter = 1.2 * math.exp(-(energy - 208.4) / 22.0)

        scatter += gaussian(energy, 73.0, 25.0, 0.35)
        left_83_dip = gaussian(energy, 70.0, 9.0, 0.45)
        mini_peak_83 = gaussian(energy, 83.0, 3.6, 0.9)
        separation_gap = gaussian(energy, 98.0, 6.0, 0.95)
        peak_113 = gaussian(energy, 112.9, 5.2, 3.15)
        peak_208 = gaussian(energy, 208.4, 9.2, 9.4)

        if energy >= 208.4:
            right_tail = 0.9 * math.exp(-(energy - 208.4) / 18.0)
        else:
            right_tail = 0.0

        # Keep a very small imprint of the CSV shape without reproducing its sharp artifacts.
        csv_texture = 1.0 + 0.04 * (csv_value - 0.25)
        value = max(0.0, (scatter + mini_peak_83 + peak_113 + peak_208 + right_tail - separation_gap - left_83_dip) * csv_texture)

        if energy > 255:
            value *= math.exp(-(energy - 255.0) / 14.0)

        spectrum.append(value)

    peak_scale = max(spectrum) if spectrum else 1.0
    return [value / peak_scale * 11.0 for value in spectrum]


def add_measurement_noise(values, noise_level=NOISE_LEVEL, seed=NOISE_SEED):
    """Add reproducible counting-like noise for a more realistic presentation curve."""
    rng = random.Random(seed)
    noisy = []

    for index, value in enumerate(values):
        previous_value = values[index - 1] if index > 0 else value
        next_value = values[index + 1] if index < len(values) - 1 else value
        local_slope = abs(next_value - previous_value) / 2.0
        relative_noise = noise_level * (0.55 + 0.45 / math.sqrt(max(value, 0.25)))
        absolute_noise = rng.gauss(0.0, relative_noise * value + 0.015 + 0.012 * local_slope)
        noisy.append(max(0.0, value + absolute_noise))

    peak_scale = max(noisy) if noisy else 1.0
    return [value / peak_scale * 11.0 for value in noisy]


energies, counts = load_csv_spectrum(DATA_PATH)
relative_intensity = add_measurement_noise(build_presentation_spectrum(energies, counts))

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
})

fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Clean black spectrum line for readability in PowerPoint.
ax.plot(energies, relative_intensity, color="#111111", linewidth=2.0, zorder=2)

# Energy windows shown as simple colored bands for a cleaner presentation look.
windows = [
    (55, 165, "#e5e5e5"),
    (166.4, 187.2, "#dceeff"),
    (187.2, 228.8, "#ffd6d6"),
    (228.8, 249.6, "#dff5df"),
]
for start, end, color in windows:
    ax.axvspan(start, end, color=color, alpha=0.75, zorder=0)

# Window labels, in dark text.
ax.text(110, 11.35, "General scatter\n55-165 keV", ha="center", va="center", fontsize=10.0, color="#111111", fontweight="bold")
ax.text(176.8, 0.65, "Low scatter\n166.4-187.2 keV", ha="center", va="bottom", fontsize=8.7, color="#111111", fontweight="bold")
ax.text(208.0, 11.35, "Photopeak 208 keV +/-10%\n187.2-228.8 keV", ha="center", va="center", fontsize=9.2, color="#111111", fontweight="bold")
ax.text(239.2, 0.65, "Upper scatter\n228.8-249.6 keV", ha="center", va="bottom", fontsize=8.7, color="#111111", fontweight="bold")

# Axis styling
ax.set_xlim(0, 300)
ax.set_ylim(0, 12.6)
ax.set_xlabel("Energy (keV)")
ax.set_ylabel("Relative intensity")
ax.grid(False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#111111")
    ax.spines[spine].set_linewidth(1.0)
ax.tick_params(axis="both", colors="#111111", labelsize=10)

plt.tight_layout()

png_path = OUT_DIR / "177Lu_gamma_spectrum_windows.png"
svg_path = OUT_DIR / "177Lu_gamma_spectrum_windows.svg"
fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {svg_path}")
