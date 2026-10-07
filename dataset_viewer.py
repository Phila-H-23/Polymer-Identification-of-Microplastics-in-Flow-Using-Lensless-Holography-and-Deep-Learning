"""
Dataset viewer - see your data directly
=========================================

WHERE TO RUN THIS: same as holo_pipeline_starter.py - VS Code, same Python
environment, same folder as your .mat files.

This does NOT run detection or any processing - it just shows you images,
saved as PNG "contact sheets" (a grid of several images at once) so you can
quickly eyeball your data without running the full pipeline.

TWO INDEPENDENT MODES, controlled by the flags below - turn either or both
on:

  MODE 1 (view_raw_frames): shows a grid of frames sampled evenly across a
  raw .mat video, straight off the sensor, before any processing. Useful
  for a quick look at a file - e.g. "does this video actually have visible
  particles in it?" - before spending time running the full pipeline on it.

  MODE 2 (view_patches): shows a grid of the cropped particle patches your
  pipeline already produced (from pipeline_output/manifest.csv), so you can
  browse what got detected and kept. Requires holo_pipeline_starter.py to
  have been run first.
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py

# ---------------------------------------------------------------------------
# CONFIG - edit these, then run the script
# ---------------------------------------------------------------------------
view_raw_frames = True
RAW_MAT_FILE = "frames_bg_30_4.mat"
N_RAW_FRAMES_TO_SHOW = 12  # sampled evenly across the whole video, not just the start

view_patches = True
MANIFEST_PATH = "pipeline_output/manifest.csv"
PATCH_DIR = "pipeline_output/patches"
N_PATCHES_TO_SHOW = 24  # randomly sampled from the manifest
PATCH_CHANNEL = 0  # which polarization channel to display: 0=I_0, 1=I_45, 2=I_90, 3=I_135

view_frame_differences = True
DIFF_MAT_FILE = "PE1_600.mat"
DIFF_FRAME_PAIRS = [(0, 5), (0, 20), (100, 105), (100, 120), (300, 305), (300, 320)]
# each pair is (frame_a, frame_b) - a SHORT gap (e.g. 5 frames apart) shows
# whether anything moves at all in a brief window; a LONGER gap (e.g. 20
# frames) shows whether particles have fully flowed through and been
# replaced. Fixed background debris cancels to near-zero in every pair,
# regardless of gap size - only genuine motion should show up as bright
# blobs in the diff.


def load_mat_array(path, var_name=None):
    """Same loader as holo_pipeline_starter.py - handles both older MATLAB
    formats (via scipy) and v7.3/HDF5 format (via h5py, with the axis
    reversal correction that format requires)."""
    try:
        mat = sio.loadmat(path)
        if var_name is None:
            var_name = [k for k in mat.keys() if not k.startswith("__")][0]
        return mat[var_name]
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            if var_name is None:
                var_name = [k for k in f.keys() if not k.startswith("#")][0]
            raw = f[var_name][()]
            return np.transpose(raw, axes=range(raw.ndim)[::-1])


def make_grid(images, titles, out_path, cols=4, figsize_per_cell=4):
    """Lays out a list of 2D images in a grid and saves it as one PNG."""
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_cell * cols, figsize_per_cell * rows))
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# MODE 1: browse raw frames from a .mat video
# ---------------------------------------------------------------------------
if view_raw_frames:
    print(f"Loading {RAW_MAT_FILE} ...")
    raw = load_mat_array(RAW_MAT_FILE)
    frames = raw[:, :, 0, :] if raw.ndim == 4 else raw
    n_total = frames.shape[-1]

    idx = np.linspace(0, n_total - 1, min(N_RAW_FRAMES_TO_SHOW, n_total)).round().astype(int)
    images = [frames[:, :, i] for i in idx]
    titles = [f"frame {i}" for i in idx]

    out_path = "dataset_view_raw_frames.png"
    make_grid(images, titles, out_path)
    print(f"Saved {out_path} - {len(idx)} frames sampled evenly across {n_total} total frames in {RAW_MAT_FILE}")

# ---------------------------------------------------------------------------
# MODE 2: browse cropped particle patches already produced by the pipeline
# ---------------------------------------------------------------------------
if view_patches:
    if not os.path.exists(MANIFEST_PATH):
        print(f"No manifest found at {MANIFEST_PATH} - run holo_pipeline_starter.py first.")
    else:
        manifest = pd.read_csv(MANIFEST_PATH)
        print(f"Manifest has {len(manifest)} patches across classes: {manifest['class'].unique().tolist()}")

        n_show = min(N_PATCHES_TO_SHOW, len(manifest))
        sample = manifest.sample(n_show, random_state=0).reset_index(drop=True)

        images, titles = [], []
        for _, row in sample.iterrows():
            patch = np.load(os.path.join(PATCH_DIR, row["patch_file"]))
            images.append(patch[:, :, PATCH_CHANNEL])
            titles.append(f"{row['class']} | particle {row['particle_id']}")

        out_path = "dataset_view_patches.png"
        make_grid(images, titles, out_path, cols=6, figsize_per_cell=3)
        print(f"Saved {out_path} - {n_show} randomly sampled patches from the manifest")

        # Also break down counts per class, since this is useful to see at
        # a glance every time you look at your data, not just during the
        # dedicated class balance audit step.
        print("\nPatch counts per class:")
        print(manifest["class"].value_counts().to_string())

# ---------------------------------------------------------------------------
# MODE 3: frame-difference viewer - the reliable way to spot real motion
# ---------------------------------------------------------------------------
# Fixed sensor/optical debris (dust, scratches) sits in the exact same
# position in every frame of every video, so subtracting one frame from
# another cancels it out almost completely. Real flowing particles do NOT
# cancel out this way, since they've moved between the two frames - they
# show up as bright blobs in the difference image. This is the most
# reliable way to tell "is this a real particle or fixed debris" by eye,
# much more reliable than comparing static screenshots side by side.
if view_frame_differences:
    print(f"\nLoading {DIFF_MAT_FILE} for frame-difference comparison...")
    raw = load_mat_array(DIFF_MAT_FILE)
    frames = raw[:, :, 0, :] if raw.ndim == 4 else raw

    images, titles = [], []
    for a, b in DIFF_FRAME_PAIRS:
        diff = np.abs(frames[:, :, a].astype(float) - frames[:, :, b].astype(float))
        images.append(diff)
        titles.append(f"frame {a} vs {b} (gap={b - a})")

    out_path = "dataset_view_frame_differences.png"
    make_grid(images, titles, out_path, cols=3)
    print(f"Saved {out_path} - bright blobs are real motion (particles); "
          f"a near-black result at every pair means nothing detectable moved "
          f"in these particular frames")

    # ------------------------------------------------------------------
    # Glitch scan: some sensors/setups produce occasional corrupted frames
    # (e.g. vertical readout striping). A genuinely corrupted frame makes
    # EVERY neighboring-frame diff involving it look like a flood of
    # motion, which would wrongly get read as "lots of particles." Scan
    # consecutive-frame differences across the whole video and flag any
    # frame whose diff is a statistical outlier compared to the rest -
    # a real particle event affects a small, normal-sized patch of the
    # frame; a corrupted frame affects almost the entire frame at once.
    print("\nScanning all frames for possible sensor glitches (this may take a moment)...")
    diff_totals = []
    for i in range(frames.shape[-1] - 1):
        d = np.abs(frames[:, :, i].astype(float) - frames[:, :, i + 1].astype(float))
        diff_totals.append(d.mean())
    diff_totals = np.array(diff_totals)
    # Use median and MAD (median absolute deviation) rather than mean/std -
    # a genuine glitch is a large outlier, and mean/std are themselves
    # skewed by the very outlier they're supposed to catch, especially with
    # only a few hundred frames. Median/MAD stay stable even with a real
    # outlier present, so the glitch doesn't hide itself.
    median = np.median(diff_totals)
    mad = np.median(np.abs(diff_totals - median)) + 1e-6
    robust_z = 0.6745 * (diff_totals - median) / mad  # 0.6745 makes this comparable to a normal z-score
    glitch_frames = np.where(robust_z > 6)[0]
    print(f"Typical frame-to-frame diff level (median): {median:.2f}, MAD: {mad:.2f}")
    if len(glitch_frames) > 0:
        print(f"Flagged {len(glitch_frames)} frames as likely glitches (diff far above normal): "
              f"{glitch_frames.tolist()[:30]}{'...' if len(glitch_frames) > 30 else ''}")
        print("Consider excluding these frame indices (and their immediate neighbors) from "
              "detection/tracking until the cause is confirmed with Fraser.")
    else:
        print("No frames flagged as likely glitches by this check.")