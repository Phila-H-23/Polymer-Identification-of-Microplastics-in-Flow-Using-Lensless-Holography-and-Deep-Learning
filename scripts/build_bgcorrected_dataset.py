"""
build_bgcorrected_dataset.py

Implements the background-subtraction preprocessing method the user
supplied directly (from Montandon & Nicolls 2024, no paper access here -
taking the method as given, applied exactly as specified), and builds
BOTH a corrected real-patch dataset and a corrected background-only
dataset from it. Writes ONLY new files - never touches pipeline_output/
patches/, manifest.csv, pipeline_output/patches_background/,
manifest_background.csv, or anything under results/.

METHOD (applied per raw mosaic frame, BEFORE demosaicing):
  background_image = mean of the (up to) 90 raw frames immediately
                      preceding this frame in the SAME video's capture
                      sequence (fewer if fewer than 90 precede it)
  corrected_frame = raw_frame - background_image + mean(background_image)

EDGE CASE (not covered by the method as stated): frame index 0 has ZERO
preceding frames, so "the mean of the preceding frames" is undefined.
Manifest.csv does contain frame=0 particles (checked directly) - for
these, no correction is applied (corrected_frame = raw_frame unchanged),
and this is counted and reported rather than silently done.

EFFICIENCY: background-only patches are drawn from the SAME
(source_file, frame) as their paired real particle (by construction -
see build_background_dataset.py), so the per-frame correction is
computed ONCE per unique (video, frame) and reused for both the real
particle crop and its paired background crop at a different (x, y) in
that same corrected frame. Combined with one bulk contiguous read per
video (matches the timing/approach established in
build_background_dataset.py and stokes_diagnostic.py: scattered per-
frame HDF5 reads on these OneDrive-synced files are pathologically
slow, ~74s/video for one full bulk read is not), this needs exactly 16
video reads total, not 32.

BACKGROUND-MEAN COMPUTATION: averaging is done on the RAW (un-
demosaiced, un-transposed) frame stack first, then transposed/demosaiced
ONCE - equivalent to transposing/demosaicing each of up to 90 frames
individually then averaging (both are linear operations, so they
commute), but far cheaper.

OUTPUTS (new files only):
  pipeline_output/patches_bgcorrected/*.npy            (real patches, corrected)
  pipeline_output/manifest_bgcorrected.csv              (mirrors manifest.csv)
  pipeline_output/patches_background_bgcorrected/*.npy  (background-only, corrected)
  pipeline_output/manifest_background_bgcorrected.csv   (mirrors manifest_background.csv)
"""
import os
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
REAL_MANIFEST_PATH = os.path.join(DATA_DIR, "pipeline_output", "manifest.csv")
BG_MANIFEST_PATH = os.path.join(DATA_DIR, "pipeline_output", "manifest_background.csv")

OUT_REAL_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_bgcorrected")
OUT_REAL_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_bgcorrected.csv")
OUT_BG_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_background_bgcorrected")
OUT_BG_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_background_bgcorrected.csv")

HALF = 32  # matches CROP_SIZE//2, same crop formula used everywhere else in this project
BG_WINDOW = 90


def demosaic(snapshot):
    I_0 = snapshot[1::2, 1::2]
    I_45 = snapshot[0::2, 1::2]
    I_90 = snapshot[0::2, 0::2]
    I_135 = snapshot[1::2, 0::2]
    return I_0, I_45, I_90, I_135


def load_video_block(path):
    """Same dual-format handling as build_background_dataset.py: scipy
    for older-format files (frame axis LAST), h5py fallback for v7.3
    files (frame axis FIRST, needs a transpose per frame - see
    get_raw_frame / get_background_mean_raw below)."""
    try:
        mat = sio.loadmat(path)
        var_name = [k for k in mat.keys() if not k.startswith("__")][0]
        return mat[var_name], "scipy"
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            var_name = [k for k in f.keys() if not k.startswith("#")][0]
            return f[var_name][:], "h5py"


def n_frames_of(block, mode):
    return block.shape[-1] if mode == "scipy" else block.shape[0]


def get_raw_frame(block, mode, idx):
    """Returns frame idx as (H, W), NOT demosaiced, matching the raw
    mosaic orientation demosaic() expects."""
    if mode == "scipy":
        return block[:, :, 0, idx] if block.ndim == 4 else block[:, :, idx]
    else:
        return block[idx, 0].T if block.ndim == 4 else block[idx].T


def get_background_mean_raw(block, mode, lo, hi):
    """Mean of raw frames [lo, hi) - averaged BEFORE the per-frame
    transpose/orientation fix (see module docstring: linear op, commutes,
    far cheaper than transposing every one of up to 90 frames)."""
    if mode == "scipy":
        window = block[:, :, 0, lo:hi] if block.ndim == 4 else block[:, :, lo:hi]
        mean_raw = window.astype(np.float64).mean(axis=-1)  # (H_orig, W_orig) no transpose needed
        return mean_raw
    else:
        window = block[lo:hi, 0] if block.ndim == 4 else block[lo:hi]
        mean_raw = window.astype(np.float64).mean(axis=0)  # (W_orig, H_orig)
        return mean_raw.T  # -> (H, W)


def correct_frame(block, mode, frame_idx):
    """Returns (corrected_2d_uint8, was_corrected_bool)."""
    raw = get_raw_frame(block, mode, frame_idx).astype(np.float64)
    lo, hi = max(0, frame_idx - BG_WINDOW), frame_idx
    if hi <= lo:
        # frame_idx == 0: no preceding frames exist, correction undefined - leave raw
        return np.clip(raw, 0, 255).astype(np.uint8), False
    bg_mean = get_background_mean_raw(block, mode, lo, hi)  # (H, W) float64, ALREADY oriented (H,W)
    # get_background_mean_raw for scipy mode returns (H_orig,W_orig) without transpose since
    # scipy frames are already (H,W) - matches get_raw_frame's scipy branch directly.
    corrected = raw - bg_mean + bg_mean.mean()
    return np.clip(corrected, 0, 255).astype(np.uint8), True


def crop_patch(demosaiced_4ch, cx, cy):
    """demosaiced_4ch: (H, W, 4) uint8. Same half=32 clip formula used by
    holo_pipeline_starter.py Step 8 and build_background_dataset.py."""
    h, w = demosaiced_4ch.shape[:2]
    y0, y1 = max(0, cy - HALF), min(h, cy + HALF)
    x0, x1 = max(0, cx - HALF), min(w, cx + HALF)
    return demosaiced_4ch[y0:y1, x0:x1, :]


def main():
    real_manifest = pd.read_csv(REAL_MANIFEST_PATH)
    bg_manifest = pd.read_csv(BG_MANIFEST_PATH)

    os.makedirs(OUT_REAL_PATCH_DIR, exist_ok=True)
    os.makedirs(OUT_BG_PATCH_DIR, exist_ok=True)

    real_out_rows = []
    bg_out_rows = []
    n_frame0_real = 0
    n_frame0_bg = 0

    videos = sorted(real_manifest["source_file"].unique())
    for source_file in videos:
        video_path = os.path.join(DATA_DIR, source_file)
        print(f"\n{source_file}: loading (bulk contiguous read)...")
        block, mode = load_video_block(video_path)
        n_frames = n_frames_of(block, mode)
        print(f"  shape {block.shape}, {n_frames} total frames available")

        real_rows = real_manifest[real_manifest["source_file"] == source_file]
        bg_rows = bg_manifest[bg_manifest["source_file"] == source_file]
        needed_frames = sorted(set(real_rows["frame"]) | set(bg_rows["frame"]))
        print(f"  correcting {len(needed_frames)} unique frames "
              f"({len(real_rows)} real patches, {len(bg_rows)} background patches)")

        for frame_idx in needed_frames:
            corrected_2d, was_corrected = correct_frame(block, mode, int(frame_idx))
            I0, I45, I90, I135 = demosaic(corrected_2d)
            demosaiced_4ch = np.stack([I0, I45, I90, I135], axis=-1)  # (1024, 1224, 4) uint8

            for _, row in real_rows[real_rows["frame"] == frame_idx].iterrows():
                if not was_corrected:
                    n_frame0_real += 1
                patch = crop_patch(demosaiced_4ch, int(row["x"]), int(row["y"]))
                np.save(os.path.join(OUT_REAL_PATCH_DIR, row["patch_file"]), patch)
                real_out_rows.append(row.to_dict())

            for _, row in bg_rows[bg_rows["frame"] == frame_idx].iterrows():
                if not was_corrected:
                    n_frame0_bg += 1
                patch = crop_patch(demosaiced_4ch, int(row["x"]), int(row["y"]))
                np.save(os.path.join(OUT_BG_PATCH_DIR, row["patch_file"]), patch)
                bg_out_rows.append(row.to_dict())

        del block

    real_out = pd.DataFrame(real_out_rows)
    bg_out = pd.DataFrame(bg_out_rows)
    real_out.to_csv(OUT_REAL_MANIFEST, index=False)
    bg_out.to_csv(OUT_BG_MANIFEST, index=False)

    print(f"\nBuilt {len(real_out)} corrected real patches -> {OUT_REAL_MANIFEST}")
    print(f"Built {len(bg_out)} corrected background patches -> {OUT_BG_MANIFEST}")
    print(f"Frame-0 edge case (no preceding frames, left uncorrected): "
          f"{n_frame0_real} real patches, {n_frame0_bg} background patches")

    # Sanity: row counts and split distributions should exactly match the originals
    print("\nSanity check vs originals:")
    print(f"  real: {len(real_out)} vs original manifest.csv {len(real_manifest)} rows")
    print(f"  background: {len(bg_out)} vs original manifest_background.csv {len(bg_manifest)} rows")
    print("  real split_video_holdout distribution:", real_out["split_video_holdout"].value_counts().to_dict())
    print("  background split distribution:", bg_out["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
