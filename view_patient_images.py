import sys
from pathlib import Path
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ----- Edit this path to quickly choose a series without passing an argument -----
# Example: set to '2026-05__Studies/DOE^JOHN_ANON62096_CT_2026-05-20_082808_MN.LU177.POST.TRAITEMENT-EN_CT_n224__00'
# Leave as empty string to require a command-line argument.
DEFAULT_FOLDER = '2026-05__Studies/WB'

def find_dicom_files(folder):
    return sorted([p for p in Path(folder).rglob('*.dcm')])


def load_series(dcm_paths):
    # Return a flat list of 2D frames: handle single-frame files and multi-frame DICOMs
    frames = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(p)
        except Exception:
            continue
        if not hasattr(ds, 'pixel_array'):
            continue
        arr = ds.pixel_array.astype(np.float32)
        # apply rescale if present
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        arr = arr * slope + intercept

        # If this is a standard 2D image
        if arr.ndim == 2:
            frames.append((p.name, arr, ds))
        # Multi-frame: expand into individual 2D frames
        elif arr.ndim == 3:
            for i in range(arr.shape[0]):
                frames.append((f"{p.name}[{i}]", arr[i], ds))
        else:
            # Fallback: try to reshape so last two dims are image
            try:
                newshape = (-1, arr.shape[-2], arr.shape[-1])
                arr2 = arr.reshape(newshape)
                for i in range(arr2.shape[0]):
                    frames.append((f"{p.name}[{i}]", arr2[i], ds))
            except Exception:
                # give up for this file
                continue
    return frames


def show_stack(slices, title='Series'):
    imgs = [s[1] for s in slices]
    if not imgs:
        print('No images to show.')
        return
    stack = np.stack(imgs, axis=0)
    n = stack.shape[0]
    fig, ax = plt.subplots(figsize=(6,6))
    plt.subplots_adjust(bottom=0.15)
    idx = 0
    im = ax.imshow(stack[idx], cmap='gray', aspect='auto')
    ax.set_title(f'{title} (slice {idx+1}/{n})')
    ax.axis('off')

    axcolor = 'lightgoldenrodyellow'
    axslice = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor=axcolor)
    slider = Slider(axslice, 'Slice', 1, n, valinit=1, valstep=1)

    def update(val):
        i = int(slider.val) - 1
        im.set_data(stack[i])
        ax.set_title(f'{title} (slice {i+1}/{n})')
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show()


def choose_and_show(patient_folder):
    p = Path(patient_folder)
    if not p.exists():
        print('Folder not found:', patient_folder); return
    # group by top-level subdirs (series)
    series_dirs = [d for d in p.iterdir() if d.is_dir()]
    if not series_dirs:
        series_dirs = [p]
    print('Found series folders:')
    for i,d in enumerate(series_dirs,1):
        print(f'{i}. {d.name}')
    sel = input('Choose series number (or enter to show first): ').strip()
    try:
        sel_i = int(sel)-1 if sel else 0
    except:
        sel_i = 0
    sel_i = max(0, min(sel_i, len(series_dirs)-1))
    sel_dir = series_dirs[sel_i]
    dcm_files = find_dicom_files(sel_dir)
    print('DICOM files found:', len(dcm_files))
    slices = load_series(dcm_files)
    if not slices:
        print('No readable images in series.')
        return
    show_stack(slices, title=sel_dir.name)


if __name__ == '__main__':
    # If a default path is set above, use it when no CLI arg is provided.
    if len(sys.argv) < 2:
        if DEFAULT_FOLDER:
            choose_and_show(DEFAULT_FOLDER)
        else:
            print('Usage: python view_patient_images.py /path/to/patient_folder')
    else:
        choose_and_show(sys.argv[1])
