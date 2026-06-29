import sys
from pathlib import Path

import planar_processing


def main() -> None:
    scan_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else planar_processing.default_rapid_scan_dir()
    scan_dir = scan_dir.expanduser().resolve()
    if not scan_dir.exists():
        raise FileNotFoundError(f"Scan folder not found: {scan_dir}")

    planar_processing.run_planar_workflow(scan_dir)


if __name__ == "__main__":
    main()
