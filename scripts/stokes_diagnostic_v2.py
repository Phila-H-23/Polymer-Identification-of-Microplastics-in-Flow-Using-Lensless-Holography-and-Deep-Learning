"""
stokes_diagnostic_v2.py

Follow-up to scripts/stokes_diagnostic.py, using the FIXED Stokes formula
(q,u normalised by S+0.05, clamped to [-1,1], zeroed where S<0.02 - see
to_stokes() in train_compact_cnn.py / train_baseline_resnet.py) and testing
a sharper question: not "is dolp different between particle and background
pixels in absolute terms" (v1 found no), but "does either region deviate
from a proper spatial background REFERENCE" - i.e. does subtracting out the
sensor's own fixed-pattern q/u (vignetting, per-pixel polariser transmission
non-uniformity) reveal a particle signal that a flat comparison would miss.

REFERENCE SOURCE: the pipeline has no saved "90-frame rolling background" -
grepped holo_pipeline_starter.py and pipeline_output/ directly to confirm.
What it does have is frames_bg_30.mat: a dedicated calibration file the
starter script's own docstring describes as "no real flowing particles in
it (it's background/dust)" - i.e. exactly the "particle-free frames from
one video" fallback this task asked for, and it has exactly 30 frames (its
name is literal). Used here as the sole reference; NOT loaded per-source-
video, so this assumes the sensor's fixed-pattern q/u is stable across
sessions - a real caveat if it isn't, but there's no per-video background
capture in this dataset to test that against.

REFERENCE BUILD: average the 30 raw mosaic frames first, then demosaic
once (equivalent to demosaicing each frame then averaging, since demosaic()
is just fixed-position subsampling - a linear operation that commutes with
averaging - but cheaper).

SPATIAL ALIGNMENT: manifest.csv's x,y columns are each patch's CENTROID in
the demosaiced (1024x1224) frame, and the crop is
[y-32:y+32, x-32:x+32] clipped to frame bounds - copied verbatim from
holo_pipeline_starter.py Step 8 (confirmed by reading that code directly,
not assumed). The same clip is applied to q_ref/u_ref here so the reference
slice lines up pixel-for-pixel with the patch, including at frame edges
where patches are smaller than 64x64.

Particle / background regions within a patch: same central-24x24 vs
outer-8px-border definition as stokes_diagnostic.py, which already
handles undersized (edge-of-frame) patches by clipping to the patch's own
shape - there's no separate zero-padding to exclude at this stage, because
these raw manifest patches are the pipeline's actual variable-size crops,
not the fixed-size zero-padded canvas the training PatchDataset builds
later on the fly.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
MANIFEST_PATH = os.path.join(DATA_DIR, "pipeline_output", "manifest.csv")
PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches")
BG_FILE = os.path.join(DATA_DIR, "frames_bg_30.mat")
OUT_CSV = os.path.join(DATA_DIR, "results", "stokes_diagnostic_v2.csv")
OUT_QREF_PNG = os.path.join(DATA_DIR, "results", "q_ref.png")

PARTICLE_BOX = 24
BORDER_PX = 8
CROP_HALF = 32  # matches holo_pipeline_starter.py Step 8's CROP_SIZE // 2
S_FLOOR = 0.05
S_DARK = 0.02


def demosaic(snapshot):
    # Verbatim from holo_pipeline_starter.py.
    I_0 = snapshot[1::2, 1::2]
    I_45 = snapshot[0::2, 1::2]
    I_90 = snapshot[0::2, 0::2]
    I_135 = snapshot[1::2, 0::2]
    return I_0, I_45, I_90, I_135


def compute_squ(I0, I45, I90, I135):
    """Fixed formula: normalise by S (+0.05 floor), clamp to [-1,1], zero
    q/u in dark regions (S < 0.02) instead of letting a near-zero
    denominator blow up. Matches to_stokes() in both training scripts."""
    S = (I0 + I45 + I90 + I135) / 4.0
    denom = S + S_FLOOR
    q = np.clip((I0 - I90) / denom, -1.0, 1.0)
    u = np.clip((I45 - I135) / denom, -1.0, 1.0)
    dark = S < S_DARK
    q = np.where(dark, 0.0, q)
    u = np.where(dark, 0.0, u)
    return S, q, u


def particle_background_masks(h, w):
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


def five_number_summary(a):
    return {
        "min": float(np.min(a)), "q1": float(np.percentile(a, 25)),
        "median": float(np.median(a)), "q3": float(np.percentile(a, 75)),
        "max": float(np.max(a)),
    }


def main():
    # --- 1. Build the reference q_ref / u_ref map from frames_bg_30.mat ---
    print(f"Loading {BG_FILE} ...")
    mat = sio.loadmat(BG_FILE)
    var_name = [k for k in mat.keys() if not k.startswith("__")][0]
    bg_raw = mat[var_name]  # (2048, 2448, 1, 30) uint8
    bg_frames = bg_raw[:, :, 0, :] if bg_raw.ndim == 4 else bg_raw
    n_bg = bg_frames.shape[-1]
    print(f"{n_bg} background frames, mosaic shape {bg_frames.shape[:2]}")

    mean_frame = bg_frames.astype(np.float64).mean(axis=-1)  # (2048, 2448)
    I0, I45, I90, I135 = demosaic(mean_frame.astype(np.float32))
    I0, I45, I90, I135 = I0 / 255.0, I45 / 255.0, I90 / 255.0, I135 / 255.0
    S_ref, q_ref, u_ref = compute_squ(I0, I45, I90, I135)
    print(f"Reference map shape (demosaiced): {q_ref.shape}")

    for name, arr in [("q_ref", q_ref), ("u_ref", u_ref)]:
        stats = five_number_summary(arr)
        print(f"\n{name}: mean={arr.mean():.4f} std={arr.std():.4f}")
        print(f"  5-number summary: min={stats['min']:.4f} q1={stats['q1']:.4f} "
              f"median={stats['median']:.4f} q3={stats['q3']:.4f} max={stats['max']:.4f}")

    os.makedirs(os.path.dirname(OUT_QREF_PNG), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(q_ref, cmap="coolwarm", vmin=-np.abs(q_ref).max(), vmax=np.abs(q_ref).max())
    ax.set_title(f"q_ref map (from {n_bg}-frame background mean, {BG_FILE.split(chr(92))[-1]})")
    fig.colorbar(im, ax=ax, label="q_ref")
    fig.tight_layout()
    fig.savefig(OUT_QREF_PNG, dpi=110)
    plt.close(fig)
    print(f"\nSaved q_ref map -> {OUT_QREF_PNG}")

    ref_h, ref_w = q_ref.shape

    # --- 2. Per-patch dq/du vs the spatially-aligned reference slice ---
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"\nLoaded manifest: {len(manifest)} patches, classes: {sorted(manifest['class'].unique())}")

    pooled = {}
    n_skipped = 0
    for _, row in manifest.iterrows():
        patch = np.load(os.path.join(PATCH_DIR, row["patch_file"])).astype(np.float32) / 255.0
        h, w = patch.shape[:2]
        if h < PARTICLE_BOX or w < PARTICLE_BOX:
            n_skipped += 1
            continue

        cx, cy = int(row["x"]), int(row["y"])
        y0, y1 = max(0, cy - CROP_HALF), min(ref_h, cy + CROP_HALF)
        x0, x1 = max(0, cx - CROP_HALF), min(ref_w, cx + CROP_HALF)
        q_ref_patch = q_ref[y0:y1, x0:x1]
        u_ref_patch = u_ref[y0:y1, x0:x1]
        if q_ref_patch.shape != (h, w):
            # shape mismatch (shouldn't happen given identical clip logic,
            # but guard rather than silently misaligning pixels)
            n_skipped += 1
            continue

        I0, I45, I90, I135 = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2], patch[:, :, 3]
        _, q_p, u_p = compute_squ(I0, I45, I90, I135)
        dq = q_p - q_ref_patch
        du = u_p - u_ref_patch

        particle_mask, bg_mask = particle_background_masks(h, w)

        cls = row["class"]
        if cls not in pooled:
            pooled[cls] = {"particle": {"absdq": [], "absdu": []},
                            "background": {"absdq": [], "absdu": []}}
        pooled[cls]["particle"]["absdq"].append(np.abs(dq[particle_mask]))
        pooled[cls]["particle"]["absdu"].append(np.abs(du[particle_mask]))
        pooled[cls]["background"]["absdq"].append(np.abs(dq[bg_mask]))
        pooled[cls]["background"]["absdu"].append(np.abs(du[bg_mask]))

    if n_skipped:
        print(f"Skipped {n_skipped} patches (too small for the particle box, or a reference-slice shape mismatch)")

    rows = []
    for cls in sorted(pooled):
        p = pooled[cls]["particle"]
        b = pooled[cls]["background"]
        row = {"class": cls}
        for metric in ["absdq", "absdu"]:
            pv = np.concatenate(p[metric])
            bv = np.concatenate(b[metric])
            row[f"particle_{metric}_mean"] = pv.mean()
            row[f"particle_{metric}_std"] = pv.std()
            row[f"background_{metric}_mean"] = bv.mean()
            row[f"background_{metric}_std"] = bv.std()
            row[f"ratio_mean_{metric}"] = pv.mean() / bv.mean() if bv.mean() != 0 else np.nan
            row[f"ratio_std_{metric}"] = pv.std() / bv.std() if bv.std() != 0 else np.nan
        row["n_patches"] = len(p["absdq"])
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("\n=== Stokes diagnostic v2 (deviation from spatial background reference) ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")

    # --- plain verdict ---
    print("\n=== Verdict ===")
    ratios_dq = df["ratio_mean_absdq"].values
    ratios_du = df["ratio_mean_absdu"].values
    all_ratios = np.concatenate([ratios_dq, ratios_du])
    print(f"particle/background ratio range: |dq| {ratios_dq.min():.3f}-{ratios_dq.max():.3f}, "
          f"|du| {ratios_du.min():.3f}-{ratios_du.max():.3f}")
    if np.all(all_ratios > 1.05):
        print("Particle pixels deviate from the reference MORE than background pixels do, consistently across classes.")
    elif np.all(all_ratios < 0.95):
        print("Particle pixels deviate from the reference LESS than background pixels do.")
    else:
        print("Particle pixels do NOT deviate from the reference more than background pixels do - "
              "ratios cluster near 1.0, same conclusion as v1's flat comparison. The spatial-reference "
              "correction does not surface a particle signal that was hidden before.")

    class_range_dq = df["particle_absdq_mean"].max() - df["particle_absdq_mean"].min()
    class_range_du = df["particle_absdu_mean"].max() - df["particle_absdu_mean"].min()
    print(f"\nAcross-class spread in particle mean |dq|: {class_range_dq:.4f} "
          f"(values: {dict(zip(df['class'], df['particle_absdq_mean'].round(4)))})")
    print(f"Across-class spread in particle mean |du|: {class_range_du:.4f} "
          f"(values: {dict(zip(df['class'], df['particle_absdu_mean'].round(4)))})")


if __name__ == "__main__":
    main()
