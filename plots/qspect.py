"""
Visualisation des coupes Q/SPECT et génération des 3 projections MIP
(axiale, coronale, sagittale).

Réutilise vos modules existants (qspect_processing) pour charger le volume,
donc placez ce script dans le même dossier que votre autre script
(ou ajustez PROJECT_ROOT / les imports ci-dessous).

Usage:
    python visualize_qspect.py [chemin_vers_dossier_qspect] [Day0]
    python visualize_qspect.py                      # utilise les valeurs par défaut
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import qspect_processing  # noqa: E402
import re  # noqa: E402
import pydicom  # noqa: E402


def extract_day_label(series_name: str) -> str:
    """Extrait 'Day0', 'Day2', etc. directement du nom du dossier de la série."""
    match = re.search(r"Day\d+", series_name, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Impossible de trouver 'DayX' dans le nom du dossier : {series_name!r}"
        )
    return match.group(0)


def load_qspect_volume_raw(series_dir: Path) -> np.ndarray:
    """
    Charge un volume Q/SPECT directement avec pydicom, SANS passer par la
    vérification de complétude de qspect_processing. Utile pour inspecter
    une série incomplète (fichiers DICOM manquants).
    """
    files = sorted(series_dir.glob("*.dcm"), key=qspect_processing.sort_key_for_slice)
    if not files:
        raise ValueError(f"Aucun fichier DICOM trouvé dans {series_dir}")

    print(f"Chargement brut : {len(files)} fichiers DICOM trouvés dans {series_dir.name}")

    slices = []
    for path in files:
        ds = pydicom.dcmread(path, force=True)
        pixel_array = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(pixel_array * slope + intercept)

    return np.stack(slices, axis=0)


# ---------------------------------------------------------------------------
# À MODIFIER SELON VOS BESOINS
# ---------------------------------------------------------------------------
# Dossier parent contenant toutes les séries (ne change pas d'un jour à l'autre)
BASE_DIR = Path(
    "/Users/xavierboucher/Desktop/maitrise/2.0/Code/Data/p8/2026-06__Studies"
)

# Nom COMPLET du dossier de la série à visualiser (collez-le tel quel à
# chaque fois — c'est le seul paramètre à changer).
SERIES_NAME = "2.16.840_969.6535_PT_2026-06-16_122714_MN.LU177.POST.TRAITEMENT-EN_WB.QSPECT.Day0_n232__00000"
# ---------------------------------------------------------------------------


def display_limits(image: np.ndarray) -> tuple[float, float]:
    """Bornes d'affichage robustes (0 -> 99.5e percentile des valeurs positives)."""
    finite = image[np.isfinite(image)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return 0.0, 1.0
    return 0.0, float(np.percentile(positive, 99.5))


def make_projections(volume: np.ndarray) -> dict[str, np.ndarray]:
    """
    Calcule les 3 projections MIP (Maximum Intensity Projection) à partir
    d'un volume Q/SPECT de forme (slices, rows, cols) = (z, y, x).

    - Axiale   : vue de dessus  -> MIP le long de l'axe z (slices)
    - Coronale : vue de face    -> MIP le long de l'axe y (rows)
    - Sagittale: vue de profil  -> MIP le long de l'axe x (cols)
    """
    axial_mip = volume.max(axis=0)                     # (rows, cols)
    coronal_mip = np.flipud(volume.max(axis=1))         # (slices, cols), tête en haut
    sagittal_mip = np.flipud(volume.max(axis=2))        # (slices, rows), tête en haut
    return {
        "Axiale (MIP)": axial_mip,
        "Coronale (MIP)": coronal_mip,
        "Sagittale (MIP)": sagittal_mip,
    }


def plot_three_projections(volume: np.ndarray, title: str, output_path: Path) -> None:
    projections = make_projections(volume)

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), facecolor="white")
    for ax, (name, image) in zip(axes, projections.items()):
        vmin, vmax = display_limits(image)
        ax.imshow(image, cmap="magma", vmin=vmin, vmax=vmax, aspect="equal", origin="upper")
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show(block=False)
    fig.canvas.flush_events()
    plt.close("all")
    print(f"Figure des 3 projections sauvegardée : {output_path}")


def slice_viewer(volume: np.ndarray, title: str = "Coupes Q/SPECT") -> None:
    """
    Visualiseur interactif simple: navigation dans les coupes axiales
    du volume avec les touches flèche haut / bas (ou molette de la souris).
    """
    n_slices = volume.shape[0]
    current = {"index": n_slices // 2}

    vmin, vmax = display_limits(volume)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    plt.subplots_adjust(bottom=0.1)
    img = ax.imshow(
        volume[current["index"]],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        origin="upper",
    )
    ax.set_title(f"{title} - coupe {current['index'] + 1}/{n_slices}")
    ax.axis("off")

    def update(index: int) -> None:
        index = max(0, min(n_slices - 1, index))
        current["index"] = index
        img.set_data(volume[index])
        ax.set_title(f"{title} - coupe {index + 1}/{n_slices}")
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key == "up":
            update(current["index"] + 1)
        elif event.key == "down":
            update(current["index"] - 1)

    def on_scroll(event) -> None:
        step = 1 if event.button == "up" else -1
        update(current["index"] + step)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    print("Visualiseur interactif : flèches haut/bas ou molette de la souris pour naviguer.")
    plt.show()


def main() -> None:
    qspect_dir = BASE_DIR / SERIES_NAME
    selected_day = extract_day_label(SERIES_NAME)

    try:
        series = qspect_processing.load_qspect_study(qspect_dir)
        qspect = qspect_processing.selected_series(series, selected_day)
        volume = qspect["volume"]
    except ValueError as error:
        print(f"AVERTISSEMENT : {error}")
        print("-> Basculement sur le chargement brut (série incomplète tolérée).")
        volume = load_qspect_volume_raw(qspect_dir)

    print(f"Volume chargé : {qspect_dir}")
    print(f"Forme du volume (slices, rows, cols) : {volume.shape}")

    output_path = qspect_dir.parent / f"qspect_projections_{selected_day.lower()}.png"
    plot_three_projections(volume, title=f"Q/SPECT - {selected_day}", output_path=output_path)

    # Visualiseur interactif : flèches haut/bas ou molette pour naviguer entre les coupes
    slice_viewer(volume, title=f"Q/SPECT - {selected_day}")


if __name__ == "__main__":
    main()
