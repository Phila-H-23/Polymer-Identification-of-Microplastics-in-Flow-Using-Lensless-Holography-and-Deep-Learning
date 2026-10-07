"""
stokes_diagnostic.py

Cheap pre-check before adding a "stokes" channel mode to the classifiers:
does the polarisation signal (q, u, dolp derived from the four polariser
channels) actually carry contrast between particle and background pixels,
and does that contrast differ across polymer classes? If particle pixels
don't clear the background noise floor, a Stokes-parameter input channel
is unlikely to help the network and the later training runs should be
read with that in mind.

CHANNEL ORDER: [I0, I45, I90, I135], matching holo_pipeline_starter.py's
demosaic() - confirmed directly against that function in an earlier
session, not assumed here.

demosaic() below is copied verbatim from holo_pipeline_starter.py rather
than imported, because that module runs a full pipeline pass (loading
INPUT_FILE, demosaicing, detecting particles, ...) as top-level code with
no __main__ guard - importing it would trigger an expensive, unwanted run.

"Particle" / "background" regions within a patch: the central 24x24 block
approximates where the detected particle sits (patches are cropped around
the detection centroid); the outer 8px border approximates local
background. For patches smaller than 64x64 (edge-of-frame crops - about
14% of the dataset), both regions are computed relative to that patch's
own actual shape and clipped so they never overlap.

Full-frame background reference: 5 random raw frames from one real sample
video (PE1_600.mat), demosaiced, with dolp/q/u computed over the ENTIRE
frame (no particle/background split - it's a broad reference, not
attempting to exclude the ~100 real particles that frame likely contains).
Frames are drawn from a small prefix of the video (first 60) rather than
uniformly across all 600, purely for I/O speed: earlier work in this
project found that scattered individual-frame reads on this OneDrive-
synced file are pathologically slow, while one contiguous block read is
fast - there's no reason background/noise statistics would differ
systematically by frame position within a video, so this doesn't bias
the reference.

Aggregation: |q|, |u|, dolp are pooled across ALL pixels of a given
region (particle or background) across ALL patches in a class, then
mean/std are taken over that pooled pixel population - not a mean of
per-patch means. This preserves genuine pixel-level variability instead
of averaging it away, which matters for a noise-floor comparison.
"""

import os
import numpy as np
import pandas as pd
import h5py

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
MANIFEST_PATH = os.path.join(DATA_DIR, "pipeline_output", "manifest.csv")
PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches")
OUT_CSV = os.path.join(DATA_DIR, "results", "stokes_diagnostic.csv")

PARTICLE_BOX = 24
BORDER_PX = 8
BG_VIDEO = "PE1_600.mat"
N_BG_FRAMES = 5
BG_FRAME_PREFIX = 60  # draw random frames from [0, BG_FRAME_PREFIX)
SEED = 42


def demosaic(snapshot):
    # Copied verbatim from holo_pipeline_starter.py - see module docstring.
    I_0 = snapshot[1::2, 1::2]
    I_45 = snapshot[0::2, 1::2]
    I_90 = snapshot[0::2, 0::2]
    I_135 = snapshot[1::2, 0::2]
    return I_0, I_45, I_90, I_135


def compute_squ_dolp(I0, I45, I90, I135):
    S = (I0 + I45 + I90 + I135) / 4.0
    q = (I0 - I90) / (I0 + I90 + 1e-6)
    u = (I45 - I135) / (I45 + I135 + 1e-6)
    dolp = np.sqrt(q ** 2 + u ** 2)
    return S, q, u, dolp


def particle_background_masks(h, w):
    """Central PARTICLE_BOX x PARTICLE_BOX (clipped to patch bounds) vs the
    outer BORDER_PX-pixel border, with border pixels inside the particle
    box excluded so small patches never double-count or overlap."""
    ph0 = max(0, h // 2 - PARTICLE_BOX // 2)
    ph1 = min(h, h // 2 + PARTICLE_BOX // 2)
    pw0 = max(0, w // 2 - PARTICLE_BOX // 2)
    pw1 = min(w, w // 2 + PARTICLE_BOX // 2)
    particle_mask = np.zeros((h, w), dtype=bool)
    particle_mask[ph0:ph1, pw0:pw1] = True

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:BORDER_PX, :] = True
    border_mask[-BORDER_PX:, :] = True
    border_mask[:, :BORDER_PX] = True
    border_mask[:, -BORDER_PX:] = True
    border_mask &= ~particle_mask
    return particle_mask, border_mask


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest: {len(manifest)} patches, classes: {sorted(manifest['class'].unique())}")

    pooled = {}  # class -> dict of region -> dict of metric -> list of pixel values
    n_skipped = 0

    for _, row in manifest.iterrows():
        patch = np.load(os.path.join(PATCH_DIR, row["patch_file"])).astype(np.float32)
        h, w = patch.shape[:2]
        if h < PARTICLE_BOX or w < PARTICLE_BOX:
            n_skipped += 1
            continue
        I0, I45, I90, I135 = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2], patch[:, :, 3]
        _, q, u, dolp = compute_squ_dolp(I0, I45, I90, I135)
        particle_mask, bg_mask = particle_background_masks(h, w)

        cls = row["class"]
        if cls not in pooled:
            pooled[cls] = {"particle": {"absq": [], "absu": [], "dolp": []},
                            "background": {"absq": [], "absu": [], "dolp": []}}

        pooled[cls]["particle"]["absq"].append(np.abs(q[particle_mask]))
        pooled[cls]["particle"]["absu"].append(np.abs(u[particle_mask]))
        pooled[cls]["particle"]["dolp"].append(dolp[particle_mask])
        pooled[cls]["background"]["absq"].append(np.abs(q[bg_mask]))
        pooled[cls]["background"]["absu"].append(np.abs(u[bg_mask]))
        pooled[cls]["background"]["dolp"].append(dolp[bg_mask])

    if n_skipped:
        print(f"Skipped {n_skipped} patches smaller than {PARTICLE_BOX}x{PARTICLE_BOX} "
              f"(particle box wouldn't fit)")

    rows = []
    for cls in sorted(pooled):
        p = pooled[cls]["particle"]
        b = pooled[cls]["background"]
        row = {"class": cls}
        for metric in ["absq", "absu", "dolp"]:
            pv = np.concatenate(p[metric])
            bv = np.concatenate(b[metric])
            row[f"particle_{metric}_mean"] = pv.mean()
            row[f"particle_{metric}_std"] = pv.std()
            row[f"background_{metric}_mean"] = bv.mean()
            row[f"background_{metric}_std"] = bv.std()
            row[f"ratio_{metric}"] = pv.mean() / bv.mean() if bv.mean() != 0 else np.nan
        row["n_patches"] = len(p["dolp"])
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- Full-frame background reference from one real sample video ---
    print(f"\nLoading {N_BG_FRAMES} random raw frames from {BG_VIDEO} "
          f"(prefix [0,{BG_FRAME_PREFIX})) for a full-frame background reference...")
    rng = np.random.default_rng(SEED)
    frame_idx = sorted(rng.choice(BG_FRAME_PREFIX, size=N_BG_FRAMES, replace=False).tolist())
    print(f"Frames: {frame_idx}")

    with h5py.File(os.path.join(DATA_DIR, BG_VIDEO), "r") as f:
        var_name = [k for k in f.keys() if not k.startswith("#")][0]
        ds = f[var_name]
        lo, hi = min(frame_idx), max(frame_idx) + 1
        block = ds[lo:hi]

    frame_absq, frame_absu, frame_dolp = [], [], []
    for idx in frame_idx:
        raw = block[idx - lo]
        frame = np.transpose(raw, axes=range(raw.ndim)[::-1])
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        I0, I45, I90, I135 = demosaic(frame.astype(np.float32))
        _, q, u, dolp = compute_squ_dolp(I0, I45, I90, I135)
        frame_absq.append(np.abs(q).ravel())
        frame_absu.append(np.abs(u).ravel())
        frame_dolp.append(dolp.ravel())

    frame_absq = np.concatenate(frame_absq)
    frame_absu = np.concatenate(frame_absu)
    frame_dolp = np.concatenate(frame_dolp)
    fullframe_row = {
        "class": "FULLFRAME_REF",
        "particle_absq_mean": np.nan, "particle_absq_std": np.nan,
        "background_absq_mean": frame_absq.mean(), "background_absq_std": frame_absq.std(),
        "ratio_absq": np.nan,
        "particle_absu_mean": np.nan, "particle_absu_std": np.nan,
        "background_absu_mean": frame_absu.mean(), "background_absu_std": frame_absu.std(),
        "ratio_absu": np.nan,
        "particle_dolp_mean": np.nan, "particle_dolp_std": np.nan,
        "background_dolp_mean": frame_dolp.mean(), "background_dolp_std": frame_dolp.std(),
        "ratio_dolp": np.nan,
        "n_patches": N_BG_FRAMES,
    }
    df = pd.concat([df, pd.DataFrame([fullframe_row])], ignore_index=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== Stokes diagnostic ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
