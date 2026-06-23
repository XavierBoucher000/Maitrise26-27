from pathlib import Path
import dicom_loader
import correction_3DEW as c3


def estimate_widths_from_images(images):
    """Try to extract center and width per normalized label from image dicts."""
    centers = {}
    widths = {}
    for im in images:
        label = im.get("energy_window") or im.get("energy_window_name") or "Unknown"
        l = label
        if isinstance(label, str):
            l_low = label.lower()
            if "lower" in l_low:
                l = "Lower Scatter"
            elif "upper" in l_low:
                l = "Upper Scatter"
            elif "photo" in l_low or "lutetium" in l_low or "177" in l_low or "main" in l_low:
                l = "Photopeak"

        low = im.get("energy_window_lower_limit")
        high = im.get("energy_window_upper_limit")
        try:
            if low is not None and high is not None:
                lowf = float(low)
                highf = float(high)
                centers[l] = (lowf + highf) / 2.0
                widths[l] = highf - lowf
        except Exception:
            continue
    return centers, widths


def fallback_estimate_widths(centers: dict, widths: dict, images: list, peak_pct: float = 0.15):
    """Fill missing widths with heuristics.

    - If center for photopeak exists but width missing: use `peak_pct * center`.
    - If none exist, try to detect radionuclide name and set typical E_peak.
    - Finally, set scatter windows to half the photopeak width if missing.
    """
    centers = dict(centers)
    widths = dict(widths)

    # detect radionuclide from images
    radionuclide = None
    for im in images:
        name = (im.get("energy_window_name") or im.get("energy_window") or "")
        if isinstance(name, str) and name:
            nl = name.lower()
            if "lutetium" in nl or "177" in nl:
                radionuclide = "lu177"
                break
            if "tc" in nl or "99m" in nl or "140" in nl:
                radionuclide = "tc99m"
                break

    # determine a photopeak center if missing
    if "Photopeak" not in centers:
        if radionuclide == "lu177":
            centers["Photopeak"] = 208.4
        elif radionuclide == "tc99m":
            centers["Photopeak"] = 140.0
        else:
            centers["Photopeak"] = 140.0

    # photopeak width
    if "Photopeak" not in widths or not widths.get("Photopeak"):
        widths["Photopeak"] = centers["Photopeak"] * peak_pct

    # scatter widths default to half photopeak width
    if "Lower Scatter" not in widths or not widths.get("Lower Scatter"):
        widths["Lower Scatter"] = widths["Photopeak"] / 2.0
    if "Upper Scatter" not in widths or not widths.get("Upper Scatter"):
        widths["Upper Scatter"] = widths["Photopeak"] / 2.0

    return centers, widths


def main():
    base = Path(__file__).resolve().parent / "1j92g763g"
    patients = dicom_loader.load_patients(base)

    for patient in patients:
        pid = patient["patient_id"]
        print(f"Processing patient: {pid}")
        for scan in patient["scans"]:
            print(f"  Scan: {scan['scan_name']}")
            imgs = scan["images"]
            counts = c3.extract_counts_from_images(imgs)
            print(f"    Counts: {counts}")

            centers, widths = estimate_widths_from_images(imgs)
            centers, widths = fallback_estimate_widths(centers, widths, imgs)

            print(f"    Centers: {centers}")
            print(f"    Widths: {widths}")

            # attempt TEW
            try:
                S, Ccorr = c3.tew_scatter_estimate(
                    counts.get("Lower Scatter", 0.0),
                    counts.get("Photopeak", 0.0),
                    counts.get("Upper Scatter", 0.0),
                    widths.get("Lower Scatter"),
                    widths.get("Photopeak"),
                    widths.get("Upper Scatter"),
                )
                print(f"    TEW: Scatter S={S:.1f}, Photopeak corrected={Ccorr:.1f}")
                # if corrected negative, fallback to DEW
                if Ccorr < 0:
                    raise ValueError("Corrected counts negative, fallback to DEW")
            except Exception as e:
                # If TEW failed due to missing/invalid widths, try with widths=1
                try:
                    wL = widths.get("Lower Scatter") or 1.0
                    wM = widths.get("Photopeak") or 1.0
                    wU = widths.get("Upper Scatter") or 1.0
                    S, Ccorr = c3.tew_scatter_estimate(
                        counts.get("Lower Scatter", 0.0),
                        counts.get("Photopeak", 0.0),
                        counts.get("Upper Scatter", 0.0),
                        float(wL),
                        float(wM),
                        float(wU),
                    )
                    print(f"    TEW retry with W=1 where missing: Scatter S={S:.1f}, Photopeak corrected={Ccorr:.1f} (original error: {e})")
                    if Ccorr < 0:
                        raise ValueError("Corrected counts negative after retry, fallback to DEW")
                except Exception as e2:
                    k = 0.5
                    S, Ccorr = c3.dew_scatter_estimate(counts.get("Lower Scatter", 0.0), counts.get("Photopeak", 0.0), k)
                    print(f"    DEW fallback (k={k}): Scatter S={S:.1f}, Photopeak corrected={Ccorr:.1f} (errors: {e}; {e2})")


if __name__ == "__main__":
    main()
