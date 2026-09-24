from pathlib import Path
import re
import sys
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional

import pydicom


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_DIR = ROOT / "Data"
FIG_ROOT = ROOT / "fig" / "metadonné"
CT_GUIDE_PATH = FIG_ROOT / "shared" / "ct_series_guide.txt"

# Un dossier de cycle "QSPECT" ressemble a "2026-06__Studies" (jamais "_WBP").
# Le dossier planar correspondant est le meme nom + "_WBP".
CYCLE_DIR_RE = re.compile(r"^.+__Studies(-\d+)?$")
PATIENT_DIR_RE = re.compile(r"^p\d+$", re.IGNORECASE)


class Cycle(NamedTuple):
    name: str            # nom du dossier de cycle, ex: "2026-06__Studies-2"
    qspect_root: Path    # .../patient 2/2026-06__Studies-2
    planar_root: Path    # .../patient 2/2026-06__Studies-2_WBP


def patient_dirs(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and PATIENT_DIR_RE.fullmatch(p.name) is not None
    )


def cycles_for_patient(patient_dir: Path) -> List[Cycle]:
    cycles = []
    for sub in sorted(patient_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.endswith("_WBP"):
            continue
        if not CYCLE_DIR_RE.match(sub.name):
            continue
        planar_dir = patient_dir / f"{sub.name}_WBP"
        cycles.append(Cycle(name=sub.name, qspect_root=sub, planar_root=planar_dir))
    return cycles


def first_dicom_file(series_dir: Path) -> Optional[Path]:
    return next(iter(sorted(series_dir.glob("*.dcm"))), None)


def read_header(path: Path) -> pydicom.dataset.FileDataset:
    return pydicom.dcmread(path, stop_before_pixels=True, force=True)


def series_dirs(root_dir: Path) -> Iterable[Path]:
    if not root_dir.exists():
        return []
    return sorted(path for path in root_dir.iterdir() if path.is_dir() and first_dicom_file(path) is not None)


def find_series(
    root_dir: Path,
    predicate: Callable[[Path, pydicom.dataset.FileDataset], bool],
) -> Optional[tuple[Path, Path, pydicom.dataset.FileDataset]]:
    for series_dir in series_dirs(root_dir):
        dicom_path = first_dicom_file(series_dir)
        if dicom_path is None:
            continue
        try:
            header = read_header(dicom_path)
        except Exception:
            continue
        if predicate(series_dir, header):
            return series_dir, dicom_path, header
    return None


def modality(header: pydicom.dataset.FileDataset) -> str:
    return str(getattr(header, "Modality", "") or "")


def description(header: pydicom.dataset.FileDataset) -> str:
    return str(getattr(header, "SeriesDescription", "") or "")


def save_header_dump(out_dir: Path, name: str, series_dir: Path, dicom_path: Path, header: pydicom.dataset.FileDataset) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{name}.txt"
    lines = [
        f"Header dump: {name}",
        "=" * (13 + len(name)),
        "",
        f"Series directory: {series_dir}",
        f"DICOM file: {dicom_path}",
        "",
        "Note: PixelData is intentionally not loaded/printed. This is the complete readable DICOM header.",
        "",
        str(header),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def representative_cases(qspect_root: Path, planar_root: Path) -> Dict[str, Optional[tuple[Path, Path, pydicom.dataset.FileDataset]]]:
    return {
        "planar_wb_rapide_nm_header": find_series(
            planar_root,
            lambda _series_dir, header: modality(header) == "NM",
        ),
        "qspect_pt_bqml_header": find_series(
            qspect_root,
            lambda _series_dir, header: modality(header) == "PT" and "QSPECT" in description(header).upper(),
        ),
        "spect_reconstructed_nm_header": find_series(
            qspect_root,
            lambda _series_dir, header: modality(header) == "NM" and description(header).upper().startswith("SPECT"),
        ),
        "spect_tomo_raw_nm_header": find_series(
            qspect_root,
            lambda _series_dir, header: modality(header) == "NM" and "TOMO" in description(header).upper(),
        ),
        "ct_header": find_series(
            qspect_root,
            lambda _series_dir, header: modality(header) == "CT",
        ),
    }


def process_cycle(patient_name: str, cycle: Cycle) -> None:
    out_dir = FIG_ROOT / patient_name / cycle.name

    saved_paths = []
    missing = []
    for name, match in representative_cases(cycle.qspect_root, cycle.planar_root).items():
        if match is None:
            missing.append(name)
            continue
        series_dir, dicom_path, header = match
        saved_paths.append(save_header_dump(out_dir, name, series_dir, dicom_path, header))

    index_lines = [
        "DICOM Metadata Header Dumps",
        "===========================",
        "",
        f"Patient: {patient_name}",
        f"Cycle: {cycle.name}",
        "",
        "These files are full pydicom header prints for representative data types.",
        "PixelData is not loaded or printed, so the files stay readable.",
        "",
        "Saved headers:",
    ]
    index_lines.extend(f"- {path.name}" for path in saved_paths)
    if missing:
        index_lines.extend(["", "Missing representative cases:"])
        index_lines.extend(f"- {name}" for name in missing)
    index_path = out_dir / "README.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    for path in saved_paths:
        print(f"Saved: {path}")
    print(f"Saved: {index_path}")


def write_ct_series_guide(output_path: Path = CT_GUIDE_PATH) -> Path:
    """Write a durable summary of the CT-family objects used in this project."""
    lines = [
        "Guide des séries CT du projet Lu-177",
        "====================================",
        "",
        "1. Topogram / localizer",
        "  Image de repérage utilisée pour planifier l'étendue du CT.",
        "  Ce n'est ni un volume anatomique 3D ni une carte d'atténuation.",
        "  Elle n'est pas utilisée dans le pipeline de correction planaire.",
        "",
        "2. CT",
        "  Reconstruction CT anatomique originale, généralement en matrice 512 x 512.",
        "  Les pixels sont convertis en HU avec RescaleSlope et RescaleIntercept.",
        "  Elle conserve la meilleure résolution anatomique, mais n'est pas directement",
        "  dans la matrice du Q/SPECT.",
        "",
        "3. ACCT",
        "  Reconstruction CT préparée par Siemens pour la correction d'atténuation.",
        "  Elle demeure souvent dans une grille CT, par exemple 224 coupes de 512 x 512.",
        "  Elle peut servir de repli lorsque l'objet transformé est absent, mais nécessite",
        "  alors un rééchantillonnage explicite vers la géométrie NM.",
        "",
        "4. ACCT [Transformed Object]",
        "  Objet CT/AC recalé et rééchantillonné dans la grille NM/Q/SPECT.",
        "  Il partage normalement le FrameOfReferenceUID, la matrice 128 x 128,",
        "  l'espacement d'environ 4,7952 mm et le nombre de coupes du Q/SPECT.",
        "  C'est l'objet prioritaire du pipeline, car sa géométrie correspond déjà au",
        "  volume Q/SPECT utilisé comme référence.",
        "  Dans les exports examinés, RescaleSlope=1 et RescaleIntercept=-1024 donnent",
        "  des valeurs de type HU. Il ne faut donc pas présumer que les pixels exportés",
        "  sont directement des coefficients mu_208 en cm^-1 sans validation additionnelle.",
        "",
        "5. ACCT [Resampled]",
        "  Objet intermédiaire rééchantillonné rencontré dans certains examens.",
        "  Sa couverture peut différer de celle du Q/SPECT, par exemple 81 coupes.",
        "  Il n'est pas choisi lorsque ACCT [Transformed Object] est disponible.",
        "",
        "6. Patient Protocol",
        "  Objet de protocole ou de planification, pas un volume CT quantitatif destiné",
        "  à la correction d'atténuation.",
        "",
        "Ordre de sélection du pipeline",
        "  1) ACCT [Transformed Object]",
        "  2) ACCT original comme repli",
        "  3) autre CT compatible seulement après vérification de la géométrie",
        "",
        "Disponibilité vérifiée le 21 septembre 2026",
        "  P11, cycle de mai : objet transformé présent à J0, J1, J3 et J6;",
        "    absent à J2, où ACCT et CT sont présents.",
        "  P11, cycle de juillet : CT, ACCT et objet transformé présents aux trois jours.",
        "  P8, cycle de juin : J0 possède le CT brut, mais ACCT et objet transformé",
        "    sont absents; J1 et J2 possèdent les objets nécessaires.",
        "  Le Q/SPECT J0 de P8 est aussi incomplet : 53 fichiers sur 232.",
        "",
        "Portée",
        "  Le CT sert au développement, à la compréhension de l'atténuation et à la",
        "  validation. Il ne doit pas être une entrée du modèle planaire final sans CT.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    write_ct_series_guide()
    for patient_dir in patient_dirs(DATA_DIR):
        patient_name = patient_dir.name
        cycles = cycles_for_patient(patient_dir)
        if not cycles:
            print(f"No cycle found for {patient_name}, skipped.")
            continue
        for cycle in cycles:
            print(f"--- {patient_name} / {cycle.name} ---")
            process_cycle(patient_name, cycle)


if __name__ == "__main__":
    main()
