"""
build_flatfield_ref.py

Builds a per-pixel, per-channel flat-field reference from the particle-free
background calibration file (frames_bg_30.mat), for the --channels pol4_ff
correction (Fraser's suggested fix for the I0/I90 fixed-pattern imbalance
found in scripts/stokes_diagnostic_v2.py's q_ref map: mean q_ref ~ -0.64,
i.e. I0 is systematically much dimmer than I90 across the whole sensor).

METHOD: manifest.csv stores each patch's (x, y) centroid position within
its 1024x1224 demosaiced source frame (confirmed directly: real manifest
and the background manifests both have x,y columns, same coordinate
system, used by --channels pol4_ff to crop the matching region of this
reference) - so a per-PIXEL reference map is used, not a per-channel
scalar. Built the same way as stokes_diagnostic_v2.py's q_ref/u_ref maps:
average the 30 raw mosaic frames first, demosaic once (demosaic() is
fixed-position subsampling, a linear op that commutes with averaging, so
this is equivalent to demosaicing each frame then averaging, but cheaper).

Output: pipeline_output/flatfield_ref.npy, float32 array shape
(1024, 1224, 4), channel order [I0, I45, I90, I135], values on the same
[0,1] scale patches are loaded to (divided by 255) - so --channels pol4_ff
can divide a [0,1]-scaled patch by a crop of this array directly.
"""
import os
import numpy as np
import scipy.io as sio

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
BG_FILE = os.path.join(DATA_DIR, "frames_bg_30.mat")
OUT_PATH = os.path.join(DATA_DIR, "pipeline_output", "flatfield_ref.npy")


def demosaic(snapshot):
    I_0 = snapshot[1::2, 1::2]
    I_45 = snapshot[0::2, 1::2]
    I_90 = snapshot[0::2, 0::2]
    I_135 = snapshot[1::2, 0::2]
    return I_0, I_45, I_90, I_135


def main():
    print(f"Loading {BG_FILE} ...")
    mat = sio.loadmat(BG_FILE)
    var_name = [k for k in mat.keys() if not k.startswith("__")][0]
    bg_raw = mat[var_name]  # (2048, 2448, 1, 30) uint8
    bg_frames = bg_raw[:, :, 0, :] if bg_raw.ndim == 4 else bg_raw
    n_bg = bg_frames.shape[-1]
    print(f"{n_bg} background frames, mosaic shape {bg_frames.shape[:2]}")

    mean_frame = bg_frames.astype(np.float64).mean(axis=-1)  # (2048, 2448)
    I0, I45, I90, I135 = demosaic(mean_frame.astype(np.float32))
    ref = np.stack([I0, I45, I90, I135], axis=-1).astype(np.float32) / 255.0  # (1024, 1224, 4)

    print(f"Reference shape: {ref.shape}")
    print("METHOD USED: per-pixel reference map (manifest.csv stores per-patch x,y "
          "position within the demosaiced frame, so pol4_ff crops this map to match).")
    for i, name in enumerate(["I0", "I45", "I90", "I135"]):
        ch = ref[:, :, i]
        print(f"  {name}: mean={ch.mean():.4f} std={ch.std():.4f} min={ch.min():.4f} max={ch.max():.4f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, ref)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
