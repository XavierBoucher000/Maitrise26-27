from pathlib import Path

import pydicom


ROOT = Path(__file__).resolve().parents[1] / "data" / "2026-05_studies" / "2026-05__Studies"


def main() -> None:
    for path in sorted(ROOT.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as exc:
            print(f"unreadable | {path} | {exc}")
            continue

        frames = int(getattr(ds, "NumberOfFrames", 1))
        if frames == 1:
            continue

        print(f"frames={frames} | {path}")


if __name__ == "__main__":
    main()
