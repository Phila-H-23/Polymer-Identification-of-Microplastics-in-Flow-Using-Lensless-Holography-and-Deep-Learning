"""
Hologram particle pipeline - starter script
=============================================

WHERE TO RUN THIS: VS Code, with the Python extension installed, using a
Python environment (virtual env or conda) that has these packages:
    pip install scipy numpy matplotlib pandas trackpy opencv-python

HOW TO CONNECT YOUR DATA: put your .mat files in the same folder as this
script (or update the paths below), then just run the script. VS Code's
"Run" button (or F5) runs the whole file top to bottom.

DOES THE CODE DO THIS FOR ME? Yes, almost entirely. You are not manually
opening thousands of images and looking at each one. The code loops through
every frame, detects particles automatically, and saves the results. The
only images you personally look at are a handful of spot-checks (Step 2 and
the end of Step 5) to confirm the automatic detection is actually finding
real particles and not garbage - this is a sanity check, not a sorting task.

WHAT THIS SCRIPT DOES, IN ORDER:
  1. Load a .mat file, demosaic each frame into its 4 polarization channels
     (0/45/90/135 degrees), per Fraser's confirmed sensor mosaic layout
  2. Show you one demosaiced frame, so you can visually confirm it loaded
     and demosaiced correctly
  3. Build an averaged background reference, saved as a spot-check image
     (not currently used by detection itself - see step 5's notes)
  4. Automatically detect particle-like blobs in each frame, using local
     contrast/variance rather than plain brightness deviation, so that
     extended diffraction ring patterns get caught and not just small
     sharp dust specks
  5. Link detections of the same particle across consecutive frames (tracking)
  6. Pick up to N frames per particle (currently N=3, per your supervisor's
     starting suggestion), to avoid keeping many near-duplicate frames of
     the same particle
  7. Crop a small 4-channel patch around each selected detection, save it as
     a .npy array, plus a manifest.csv that records everything about it

THIS VERSION RUNS ON frames_bg_30.mat ONLY, TO SHOW YOU THE MECHANICS.
That file has no real flowing particles in it (it's background/dust), so
the "particles" it detects here are just static dust specks - useful for
proving the code runs correctly, not for building your real dataset. Once
you have a real sample video (e.g. PE1_600.mat), swap the filename in
Step 1 - the rest of the code does not need to change.
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import trackpy as tp
import os
import re
import h5py


def load_mat_array(path, var_name=None):
    """Loads a variable out of a .mat file, whether it's an older-format
    file (readable by scipy) or a MATLAB v7.3 file (HDF5-based - MATLAB
    switches to this automatically once a variable gets large, which is
    why your small background file loads fine with scipy but your big
    sample videos raise NotImplementedError).

    If var_name is not given, assumes the file has one main variable and
    picks it automatically (skipping internal HDF5 bookkeeping keys).
    """
    try:
        mat = sio.loadmat(path)
        if var_name is None:
            var_name = [k for k in mat.keys() if not k.startswith("__")][0]
        return mat[var_name]
    except NotImplementedError:
        # v7.3 file - must use h5py instead
        with h5py.File(path, "r") as f:
            if var_name is None:
                var_name = [k for k in f.keys() if not k.startswith("#")][0]
            raw = f[var_name][()]
            # CRITICAL: MATLAB v7.3 (HDF5) files store arrays with their
            # dimensions reversed relative to MATLAB's own size(). Reverse
            # the axes back, or every dimension (height/width/frame/etc.)
            # ends up silently swapped.
            return np.transpose(raw, axes=range(raw.ndim)[::-1])

# ---------------------------------------------------------------------------
# STEP 1: Load your data
# ---------------------------------------------------------------------------
# Works whether the file is an older MATLAB format or v7.3 (HDF5) - see
# load_mat_array above. Set var_name if the file has more than one variable
# and the wrong one gets picked automatically.
#
# NOTE: this is your SAMPLE video (e.g. a real polymer recording). The
# dedicated background calibration file (frames_bg_30.mat) is loaded
# separately below in Step 2b - it is NOT the same thing as this file, and
# is not swapped in here even for testing.
INPUT_FILE = "PVC2_600.mat"  # <-- point this at your real sample video, one at a time

# Debug-only: saves every rejected detection (crop + coords + which check +
# why, including the actual z-value/displacement it was rejected on) to
# pipeline_output/rejected_for_review/, so they can be manually spot-checked
# for wrongly-discarded real particles. Off by default - it roughly doubles
# this run's I/O and disk use for information you usually don't need. Used
# to audit the Step 6/7 debris-check fixes on PE1_600.mat: spot-checked 21
# rejections across the full z-value range (including the borderline z<10
# tail) and found no false rejections - see project memory for details.
SAVE_REJECTED_FOR_REVIEW = False
raw = load_mat_array(INPUT_FILE)
print(f"Loaded array with shape {raw.shape}")

# Background files are (H, W, 1, num_frames); sample videos may be saved as
# (H, W, num_frames) with no singleton channel dim - handle both.
if raw.ndim == 4:
    frames = raw[:, :, 0, :]
elif raw.ndim == 3:
    frames = raw
else:
    raise ValueError(f"Unexpected array shape {raw.shape} - check var_name and inspect manually")
print(f"Loaded {frames.shape[-1]} frames, each {frames.shape[0]}x{frames.shape[1]} pixels")


# ---------------------------------------------------------------------------
# STEP 1b: Demosaic - split each raw frame into its 4 polarization channels
# ---------------------------------------------------------------------------
# Confirmed with Fraser: the 2x2 sensor mosaic layout is
#     90   45
#     135   0
# MATLAB 1:2:end (odd positions) -> Python [0::2]
# MATLAB 2:2:end (even positions) -> Python [1::2]
def demosaic(snapshot):
    I_0 = snapshot[1::2, 1::2]
    I_45 = snapshot[0::2, 1::2]
    I_90 = snapshot[0::2, 0::2]
    I_135 = snapshot[1::2, 0::2]
    return I_0, I_45, I_90, I_135


# Demosaic every frame up front. Each channel comes out at half the original
# resolution in each dimension (e.g. 2048x2448 -> 1024x1224), since every
# 2x2 block of raw pixels becomes one pixel per polarization channel.
demosaiced = np.stack(
    [np.stack(demosaic(frames[:, :, i]), axis=-1) for i in range(frames.shape[-1])],
    axis=-1,
)  # shape: (H/2, W/2, 4, num_frames) - channel order is [0, 45, 90, 135]
print(f"Demosaiced to shape {demosaiced.shape} (H, W, 4 polarization channels, frames)")

# From here on, "frames" for detection purposes uses just the intensity sum
# across polarization channels (a stand-in for overall brightness/contrast -
# good enough for localization). The individual 4-channel patches are what
# actually get saved and fed to the classifier later.
frames_for_detection = demosaiced.sum(axis=2)  # (H/2, W/2, num_frames)

# ---------------------------------------------------------------------------
# STEP 2: Look at one frame - a sanity check, not manual sorting
# ---------------------------------------------------------------------------
os.makedirs("pipeline_output", exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 2b: Load (or build and cache) the shared background reference
# ---------------------------------------------------------------------------
# This is the piece that was missing before: a proper reference built from
# your dedicated background calibration file (frames_bg_30.mat), computed
# ONCE and reused for every sample video, so static debris that's baked
# into the optical setup doesn't get repeatedly flagged as a "particle" in
# every single sample recording.
#
# Why not just subtract the background frame itself? Because debris causes
# a diffraction ring pattern (high local contrast), not a uniform brightness
# shift - plain intensity subtraction doesn't reliably cancel it. Instead we
# compute LOCAL VARIANCE for each individual background frame (not the
# average frame - see the comment in detect_particles for why that
# distinction matters), then store the per-pixel mean AND standard
# deviation of that variance across the whole background stack. That lets
# us later ask, per pixel, "is this frame's local variance unusually high
# compared to what this exact pixel normally shows in background data" -
# a proper statistical comparison rather than a flat threshold.
BACKGROUND_FILE = "frames_bg_30.mat"
BACKGROUND_CACHE = "pipeline_output/background_reference.npz"
# Bump this any time the local variance computation changes (window size,
# normalization method, etc). The cache stores this alongside the data and
# gets rebuilt automatically if it doesn't match, instead of silently
# reusing a reference built by different, incompatible code - which is
# exactly what caused the "zero detections, wildly negative z-scores" bug.
CACHE_VERSION = "normalized_v1"


def local_variance(frame, window=9):
    """Local variance in a small neighborhood around each pixel - high
    where the image is oscillating (fringes, particle edges), low in flat
    regions, regardless of overall brightness."""
    frame = frame.astype(float)
    local_mean = ndimage.uniform_filter(frame, size=window)
    local_sqmean = ndimage.uniform_filter(frame ** 2, size=window)
    return local_sqmean - local_mean ** 2


def normalized_local_variance(frame, window=9):
    """Local variance, rescaled by this frame's OWN overall noise level.

    This matters because your sample videos and the background calibration
    file were captured in different sessions, likely with slightly
    different exposure, laser power, or optical throughput (e.g. from the
    sample fluid itself). Tested directly: a mismatch as small as 5% in
    overall brightness between a frame and the background reference caused
    over 600 false detections with the unnormalized version, even with zero
    real particles present. Dividing by each frame's own median local
    variance makes the comparison self-relative, so a global brightness or
    contrast difference cancels out instead of being misread as particles
    everywhere."""
    lv = local_variance(frame, window)
    scale = np.median(lv)
    return lv / (scale + 1e-6)


if os.path.exists(BACKGROUND_CACHE) and np.load(BACKGROUND_CACHE).get("version") == CACHE_VERSION:
    print(f"Loading cached background reference from {BACKGROUND_CACHE}")
    cache = np.load(BACKGROUND_CACHE)
    bg_lv_mean, bg_lv_std = cache["bg_lv_mean"], cache["bg_lv_std"]
else:
    if os.path.exists(BACKGROUND_CACHE):
        print("Cached background reference is from an older version of this script - rebuilding.")
    print(f"Building background reference from {BACKGROUND_FILE} (one-time cost)")
    bg_raw = load_mat_array(BACKGROUND_FILE)
    bg_frames = bg_raw[:, :, 0, :] if bg_raw.ndim == 4 else bg_raw
    # demosaic and sum polarization channels the same way sample frames are,
    # so the reference is directly comparable
    bg_frames_summed = np.stack(
        [np.sum(demosaic(bg_frames[:, :, i]), axis=0) for i in range(bg_frames.shape[-1])],
        axis=-1,
    )
    lv_stack = np.stack(
        [normalized_local_variance(bg_frames_summed[:, :, i]) for i in range(bg_frames_summed.shape[-1])],
        axis=-1,
    )
    bg_lv_mean = lv_stack.mean(axis=-1)
    bg_lv_std = lv_stack.std(axis=-1) + 1e-6  # avoid division by zero in flat regions
    np.savez(BACKGROUND_CACHE, bg_lv_mean=bg_lv_mean, bg_lv_std=bg_lv_std, version=CACHE_VERSION)
    print(f"Saved background reference to {BACKGROUND_CACHE} for reuse on future runs")


fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, ch, lab in zip(axes, np.moveaxis(demosaiced[:, :, :, 0], -1, 0), ["I_0", "I_45", "I_90", "I_135"]):
    ax.imshow(ch, cmap="gray")
    ax.set_title(f"Frame 0 - {lab}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("pipeline_output/check_01_demosaiced_frame.png", dpi=100, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, ch, lab in zip(axes, np.moveaxis(demosaiced[:, :, :, 0], -1, 0), ["I_0", "I_45", "I_90", "I_135"]):
    ax.imshow(ch, cmap="gray")
    ax.set_title(f"Frame 0 - {lab}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("pipeline_output/check_01_demosaiced_frame.png", dpi=100, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(bg_lv_mean, cmap="gray")
plt.title("Background reference: typical local variance per pixel")
plt.axis("off")
plt.savefig("pipeline_output/check_02_background_reference.png", dpi=100, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# STEP 5: Detect particle-like blobs in a frame
# ---------------------------------------------------------------------------
# NOTE: this uses LOCAL VARIANCE compared against the background reference
# from Step 2b, not raw intensity deviation. Two distinct ideas combined:
#   1. Local variance (not raw brightness) catches diffraction ring patterns,
#      which oscillate across a neighborhood rather than differing sharply
#      from the background at any single pixel.
#   2. Comparing against the background reference (as a z-score: how many
#      standard deviations above what this exact pixel normally shows in
#      background-only data) stops static debris from being re-flagged in
#      every single frame - only variance that's genuinely unusual relative
#      to the known background gets through.
def compute_z_field(frame, bg_lv_mean, bg_lv_std, var_window=9, smooth=9):
    """The smoothed z-score field itself (how many standard deviations this
    frame's local variance is above what the background reference normally
    shows, per pixel) - everything detect_particles does up to the point
    where it gets thresholded into a binary mask and dilated/labeled.

    Exposed separately from detect_particles so the debris check in Step 6
    can ask "is there still unusually high variance sitting at this exact
    spot in another frame" by looking at the z-value directly, rather than
    at a binarized-then-dilated mask's area coverage. That distinction
    matters: on real PE1_600 data, a known fixed diffraction ring's z-peak
    stayed above z_thresh in a comparison frame 10 frames later (confirming
    it was still there), even though the exact set of pixels crossing
    z_thresh - and therefore the dilated mask's shape and area - had shifted
    enough that an area-overlap-fraction check missed it."""
    lv = normalized_local_variance(frame, var_window)
    z = (lv - bg_lv_mean) / bg_lv_std
    return ndimage.uniform_filter(z, size=smooth)


def detect_particles(frame, bg_lv_mean, bg_lv_std, var_window=9, smooth=9, z_thresh=7.0,
                      dilate_px=10, min_pixels=500, max_pixels=60000):
    """Finds particle-like regions by comparing this frame's local variance
    against the background reference's typical local variance, per pixel.

    Tuning notes (adjust by eye against the check_03 spot-check image):
      - z_thresh: how many standard deviations above the background's own
        typical variance counts as "new particle." Raise it if debris or
        noise is still getting flagged; lower it if real particles are
        being missed. This was validated on synthetic test data at 7.0 -
        expect to retune slightly on real sample videos.
      - smooth: a light blur applied to the comparison itself, since raw
        per-pixel variance is noisy even for identical data - without this,
        isolated noise pixels create spurious detections even after
        background comparison.
      - dilate_px: how far to grow each detected region before merging
        nearby pixels into one connected blob (needed so an extended ring
        pattern becomes one detection rather than several fragments). In
        practice this doesn't fully merge every large ring (real fringe
        gaps have been measured well over 2*dilate_px in places), which is
        why the debris check in Step 6 no longer depends on ring fragments
        staying consistently shaped between frames.
      - min_pixels / max_pixels: filters detections that are implausibly
        small (noise) or large (several particles merged together). Raised
        to 500 after real testing showed extended debris features can throw
        off small (~250-400px), inconsistent noise fragments at their edges
        that don't reliably reappear between the kept frame and its
        comparison frame - since a genuine real particle detected here was
        much larger (~40,000px), this cutoff removes the noise fragments
        without touching real detections. Revisit if real particles turn
        out to be smaller than this in your actual data.
    """
    z = compute_z_field(frame, bg_lv_mean, bg_lv_std, var_window, smooth)
    mask = z > z_thresh
    mask = ndimage.binary_dilation(mask, iterations=dilate_px)

    labeled, num_regions = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, index=range(1, num_regions + 1))
    slices = ndimage.find_objects(labeled)

    detections = []
    for size, sl in zip(sizes, slices):
        if sl is None or size < min_pixels or size > max_pixels:
            continue
        y_slice, x_slice = sl
        detections.append(
            {
                "x": (x_slice.start + x_slice.stop) / 2,
                "y": (y_slice.start + y_slice.stop) / 2,
                "w": x_slice.stop - x_slice.start,
                "h": y_slice.stop - y_slice.start,
                "pixel_count": size,
            }
        )
    return pd.DataFrame(detections)


# ---------------------------------------------------------------------------
# STEP 5b: Choose which frames to actually process
# ---------------------------------------------------------------------------
# Per your supervisor: since each frame already contains roughly 100
# particles, there's no need to run detection on every one of a video's 600
# frames. Instead, sample a small number of frames evenly across the video
# (16-20, per his guidance) and keep every particle found within them -
# rather than tracking individual particles continuously across many frames
# and deduplicating per particle identity, which is what the earlier
# version of this pipeline did.
#
# This also makes the pipeline substantially faster: detection now only
# needs to run on roughly 2x N_KEPT_FRAMES frames (each kept frame, plus one
# nearby comparison frame used purely for the debris check below), instead
# of on all 600 frames.
N_KEPT_FRAMES = 18  # per supervisor's guidance of "16-20 per video"
total_frames = frames_for_detection.shape[-1]
kept_frame_indices = np.unique(
    np.linspace(0, total_frames - 1, min(N_KEPT_FRAMES, total_frames)).round().astype(int)
)
print(f"Selected {len(kept_frame_indices)} frames out of {total_frames} to process: {kept_frame_indices.tolist()}")

# ---------------------------------------------------------------------------
# STEP 6: Detect particles in each kept frame, and reject likely debris
# ---------------------------------------------------------------------------
# search_range=15 was checked against real tracked data and confirmed not
# to fragment genuine particle tracks, so it's kept as-is here rather than
# re-tuned.
SEARCH_RANGE = 15

# How far away (in frames) to look for a comparison detection when checking
# whether something moved. Needs to be far enough that real motion is
# clearly visible, but still within the same general region of the video.
COMPARISON_OFFSET = 10

# Per your supervisor: the actual flow rate isn't known, so this is set
# from direct observation using dataset_viewer.py's frame-difference tool
# rather than derived from flow-rate physics - real motion showed up
# clearly as well-separated bright spots at gaps of 5-20 frames. Revisit
# this by eye on your actual data if detections look wrong either way.
MIN_DISPLACEMENT_PX = 5

# How the debris check decides whether a detection's own footprint is still
# "unusually variant" in the comparison frame - checked against the smoothed
# z-field directly (compute_z_field), not a binarized-then-dilated mask's
# area coverage. That distinction was necessary: an earlier version of this
# check used mask area-overlap, but on real PE1_600 data a known fixed
# diffraction ring kept leaking through even in the comparison frame's mask
# only covered 0-20% of the detection's box (below any reasonable overlap
# threshold) while its PEAK z-value there was still 7.7-9.0 - i.e. clearly
# still present, just fragmented differently by per-frame noise right at the
# fringe edge, where z sits close to z_thresh. Checking the max z-value
# instead catches this directly. Set below z_thresh (7.0) rather than equal
# to it, since we're not asking "would this independently count as a new
# detection" - just "is there still meaningfully elevated variance here" -
# and the observed ring leakage's weakest case had a comparison-frame max
# z of 5.71, comfortably above the noise floor elsewhere in the frame.
DEBRIS_CHECK_Z_THRESH = 5.0

final_rows = []
n_debris_rejected = 0
n_no_comparison = 0

# Debug-only (see SAVE_REJECTED_FOR_REVIEW): every rejected detection's
# crop/coords/frame/reason, so a human can spot-check whether real particles
# are being wrongly thrown out. Written to pipeline_output/rejected_for_review/
# in Step 8b below, only when SAVE_REJECTED_FOR_REVIEW is True.
rejected_rows = []

for kept_idx in kept_frame_indices:
    dets = detect_particles(frames_for_detection[:, :, kept_idx], bg_lv_mean, bg_lv_std)
    if len(dets) == 0:
        continue
    dets["frame"] = 0  # relabeled to the real frame index once kept below

    # a comparison frame a fixed offset away, purely to test whether each
    # detection is static (debris) or has moved (a real particle)
    comp_idx = kept_idx + COMPARISON_OFFSET
    if comp_idx >= total_frames:
        comp_idx = kept_idx - COMPARISON_OFFSET
    comp_idx = int(max(0, min(total_frames - 1, comp_idx)))

    comp_frame = frames_for_detection[:, :, comp_idx]
    comp_dets = detect_particles(comp_frame, bg_lv_mean, bg_lv_std)
    # Smoothed z-field for the comparison frame - lets the checks below ask
    # "is there still meaningfully elevated variance at this exact spot",
    # instead of trusting that a static feature fragments into the same
    # blobs in both frames (see DEBRIS_CHECK_Z_THRESH comment above).
    comp_z = compute_z_field(comp_frame, bg_lv_mean, bg_lv_std)

    def _comp_still_elevated(row):
        y0 = max(0, int(row["y"] - row["h"] / 2))
        y1 = min(comp_z.shape[0], int(row["y"] + row["h"] / 2))
        x0 = max(0, int(row["x"] - row["w"] / 2))
        x1 = min(comp_z.shape[1], int(row["x"] + row["w"] / 2))
        region = comp_z[y0:y1, x0:x1]
        return region.max() if region.size else -np.inf

    if len(comp_dets) == 0:
        # No labeled blobs in the comparison frame, but the z-field itself
        # can still be elevated there (e.g. a ring fragment too small to
        # pass min_pixels on its own) - so check each detection individually
        # rather than assuming "no comparison detections" means "keep all".
        dets["frame"] = kept_idx
        dets["displacement"] = np.nan
        z_vals = dets.apply(_comp_still_elevated, axis=1)
        still_present = z_vals > DEBRIS_CHECK_Z_THRESH
        n_debris_rejected += int(still_present.sum())
        n_no_comparison += int((~still_present).sum())
        if SAVE_REJECTED_FOR_REVIEW:
            for idx2, r in dets[still_present].iterrows():
                rejected_rows.append({**r.to_dict(), "reason": "motion_check",
                                       "detail": "no_comp_dets_but_still_elevated",
                                       "z_value": z_vals[idx2]})
        final_rows.append(dets[~still_present])
        continue

    comp_dets["frame"] = 1
    pair = pd.concat([dets, comp_dets], ignore_index=True)
    linked = tp.link(pair, search_range=SEARCH_RANGE, memory=0)

    kept_this_frame = []
    for particle_id, group in linked.groupby("particle"):
        row_kept = group[group["frame"] == 0]
        if len(row_kept) == 0:
            continue  # only appears in the comparison frame, not the one we're keeping
        row_kept = row_kept.iloc[0].copy()

        # Diagnostic displacement, computed regardless of whether the
        # z-check below already rejects this detection - purely so a
        # rejected-for-review dump can show the actual displacement value
        # a "still_elevated" rejection had, not just that it was rejected.
        diag_displacement = np.nan
        if len(group) > 1:
            row_comp_diag = group[group["frame"] == 1].iloc[0]
            diag_displacement = np.hypot(row_kept["x"] - row_comp_diag["x"], row_kept["y"] - row_comp_diag["y"])

        # Checked first, regardless of whether trackpy matched this detection
        # to anything in the comparison frame. Originally this was only
        # checked in the unmatched case below, on the assumption that a
        # matched pair's own displacement was a reliable enough signal on
        # its own - but a static ring fragment can also get matched to
        # ANOTHER fragment of itself with an apparent displacement of more
        # than MIN_DISPLACEMENT_PX, purely from fragmentation-boundary noise
        # right at the fringe edge (where z sits close to z_thresh and is
        # noise-sensitive) - not real motion. Confirmed on real PE1_600 data:
        # several ring-edge detections were still leaking through via the
        # matched branch even after the unmatched-only version of this check.
        z_val = _comp_still_elevated(row_kept)
        if z_val > DEBRIS_CHECK_Z_THRESH:
            n_debris_rejected += 1
            if SAVE_REJECTED_FOR_REVIEW:
                r = row_kept.copy()
                r["frame"] = kept_idx
                rejected_rows.append({**r.to_dict(), "reason": "motion_check",
                                       "detail": "still_elevated_in_comparison_frame",
                                       "z_value": z_val, "displacement": diag_displacement})
            continue  # still present in the comparison frame - static debris

        if len(group) == 1:
            # Unmatched by trackpy's centroid linking, and cleared by the
            # z-field check above - most likely a real particle that already
            # flowed out of the search radius.
            row_kept["displacement"] = np.nan
        else:
            row_comp = group[group["frame"] == 1].iloc[0]
            displacement = np.hypot(row_kept["x"] - row_comp["x"], row_kept["y"] - row_comp["y"])
            row_kept["displacement"] = displacement
            if displacement < MIN_DISPLACEMENT_PX:
                n_debris_rejected += 1
                if SAVE_REJECTED_FOR_REVIEW:
                    r = row_kept.copy()
                    r["frame"] = kept_idx
                    rejected_rows.append({**r.to_dict(), "reason": "motion_check",
                                           "detail": "low_displacement",
                                           "z_value": z_val, "displacement": displacement})
                continue  # matched with near-zero movement - likely debris

        row_kept["frame"] = kept_idx
        kept_this_frame.append(row_kept)

    if kept_this_frame:
        final_rows.append(pd.DataFrame(kept_this_frame))

if not final_rows:
    print(
        "\nNo particles survived detection and debris rejection across any "
        "kept frame. Check z_thresh (see the diagnostic printed if raw "
        "detection was zero) and MIN_DISPLACEMENT_PX - and confirm this "
        "video genuinely has particles using dataset_viewer.py first."
    )
    raise SystemExit(0)

selected_df = pd.concat(final_rows, ignore_index=True).reset_index(drop=True)
n_kept = len(selected_df)
print(
    f"Kept {n_kept} particle detections across {len(kept_frame_indices)} sampled frames "
    f"(rejected {n_debris_rejected} as likely static debris; {n_no_comparison} kept "
    f"with no labeled comparison-frame blob nearby, cleared via raw mask check)"
)

# ---------------------------------------------------------------------------
# STEP 7: Final cross-frame recurrence check (closes a blind spot in Step 6)
# ---------------------------------------------------------------------------
# The per-frame debris check above only ever compares a kept frame against
# ONE other frame COMPARISON_OFFSET frames away - so a small, weak static
# feature that persists longer than that window, but never happens to land
# within it, can keep getting kept under different particle_ids every time
# it's sampled. Confirmed on real PE1_600 data: 4 pairs of "kept" detections
# turned out to sit within RECURRENCE_DISTANCE_PX of each other despite
# being 35-212 sampled frames apart - well outside COMPARISON_OFFSET's
# reach, so Step 6 never had a chance to compare them directly.
#
# This pass looks across ALL kept detections in this run, regardless of
# frame gap, and drops the later (higher-frame) occurrence of any pair
# within RECURRENCE_DISTANCE_PX - keeping the earliest sighting as the
# representative. Same logic as Step 6, just without the ±COMPARISON_OFFSET
# limit: something genuinely real shouldn't keep reappearing at the same
# pixel position many frames apart.
RECURRENCE_DISTANCE_PX = 15

selected_df = selected_df.sort_values("frame").reset_index(drop=True)
kept_xy = []  # (x, y) of detections confirmed to survive so far, earliest-frame first
keep_mask = []
for _, row in selected_df.iterrows():
    is_recurrence = any(
        np.hypot(row["x"] - kx, row["y"] - ky) < RECURRENCE_DISTANCE_PX for kx, ky in kept_xy
    )
    keep_mask.append(not is_recurrence)
    if not is_recurrence:
        kept_xy.append((row["x"], row["y"]))

n_recurrence_dropped = len(selected_df) - sum(keep_mask)
if SAVE_REJECTED_FOR_REVIEW:
    for _, r in selected_df[[not k for k in keep_mask]].iterrows():
        rejected_rows.append({**r.to_dict(), "reason": "recurrence_check",
                               "detail": f"within {RECURRENCE_DISTANCE_PX}px of an earlier-frame detection"})
selected_df = selected_df[keep_mask].reset_index(drop=True)
selected_df["particle_id"] = selected_df.index  # each kept detection is independent, so a simple running ID is enough
n_kept = len(selected_df)
print(
    f"Cross-frame recurrence check: dropped {n_recurrence_dropped} detection(s) that "
    f"reappeared within {RECURRENCE_DISTANCE_PX}px of an earlier-frame detection "
    f"(likely persistent debris outside the +/-{COMPARISON_OFFSET}-frame check's reach) "
    f"- {n_kept} remain"
)

# Spot-check: draw kept detections on the first sampled frame that actually
# has any, so you can visually confirm the detector is finding real blobs.
first_check_frame = int(selected_df["frame"].iloc[0])
frame_dets = selected_df[selected_df["frame"] == first_check_frame]
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(frames_for_detection[:, :, first_check_frame], cmap="gray")
for _, d in frame_dets.iterrows():
    ax.add_patch(
        plt.Rectangle(
            (d["x"] - d["w"] / 2, d["y"] - d["h"] / 2),
            max(d["w"], 8),
            max(d["h"], 8),
            fill=False,
            edgecolor="red",
            linewidth=1,
        )
    )
ax.set_title(f"Kept detections on frame {first_check_frame} (n={len(frame_dets)}) - spot-check only")
ax.axis("off")
plt.savefig("pipeline_output/check_03_detections.png", dpi=100, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# STEP 8: Crop a patch around each selected detection, save it, log it
# ---------------------------------------------------------------------------
# Each patch is saved as a .npy array with shape (CROP_SIZE, CROP_SIZE, 4) -
# one layer per polarization channel (0, 45, 90, 135 in that order). This is
# what your classifier will eventually take as input. A PNG can't hold 4
# channels meaningfully, so we also save a quick-look PNG (just the 0-degree
# channel) purely so you can visually browse patches in a file explorer.
#
# IMPORTANT: you will run this script once per sample video (PE1_600.mat,
# PE2_600.mat, PET1_600.mat, ...). Two things make that safe rather than
# destructive:
#   1. The class is auto-detected from the filename (e.g. "PE1_600.mat" ->
#      class "PE"), instead of being left as "UNLABELED" for you to fix by
#      hand later across dozens of files.
#   2. Every patch filename is prefixed with the source video's name, and
#      the manifest is APPENDED to (loaded, combined, deduplicated, saved),
#      not overwritten - so running this on PP1_600.mat after PE1_600.mat
#      adds to your dataset instead of erasing PE1's results.
CROP_SIZE = 64  # pixels, at the demosaiced (half) resolution; adjust once you know your minimum particle size
patch_dir = "pipeline_output/patches"
os.makedirs(patch_dir, exist_ok=True)

video_tag = os.path.splitext(os.path.basename(INPUT_FILE))[0]  # e.g. "PE1_600"
# strip a trailing "_<number>" (frame count suffix) then a trailing digit
# (replicate number) to get the polymer class, e.g. "PE1_600" -> "PE1" -> "PE"
class_label = re.sub(r"_\d+$", "", video_tag)
class_label = re.sub(r"\d+$", "", class_label)
print(f"Auto-detected class '{class_label}' from filename '{INPUT_FILE}' - double check this is right.")

manifest_rows = []
for _, row in selected_df.iterrows():
    frame_stack = demosaiced[:, :, :, int(row["frame"])]  # (H, W, 4)
    cx, cy = int(row["x"]), int(row["y"])
    half = CROP_SIZE // 2
    y0, y1 = max(0, cy - half), min(frame_stack.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(frame_stack.shape[1], cx + half)
    patch = frame_stack[y0:y1, x0:x1, :]

    # prefixed with video_tag so patches from different videos never collide,
    # even if trackpy happens to reuse the same particle ID across separate runs
    base_name = f"{video_tag}_particle{int(row['particle_id'])}_frame{int(row['frame'])}"
    npy_path = os.path.join(patch_dir, base_name + ".npy")
    np.save(npy_path, patch)
    plt.imsave(os.path.join(patch_dir, base_name + "_preview.png"), patch[:, :, 0], cmap="gray")

    manifest_rows.append(
        {
            "patch_file": base_name + ".npy",
            "particle_id": int(row["particle_id"]),
            "frame": int(row["frame"]),
            "x": cx,
            "y": cy,
            "source_file": INPUT_FILE,
            "class": class_label,
        }
    )

new_manifest = pd.DataFrame(manifest_rows)
manifest_path = "pipeline_output/manifest.csv"
if os.path.exists(manifest_path):
    existing_manifest = pd.read_csv(manifest_path)
    combined = pd.concat([existing_manifest, new_manifest], ignore_index=True)
    combined = combined.drop_duplicates(subset="patch_file", keep="last")
    print(f"Appending to existing manifest: {len(existing_manifest)} existing + {len(new_manifest)} new "
          f"= {len(combined)} total rows after dropping any duplicates")
else:
    combined = new_manifest
    print(f"Creating new manifest with {len(combined)} rows")

combined.to_csv(manifest_path, index=False)
print(f"Saved {len(new_manifest)} 4-channel patches from this run to '{patch_dir}/'")
print(f"Manifest at '{manifest_path}' now has {len(combined)} total rows across "
      f"{combined['source_file'].nunique()} source video(s) and classes: {sorted(combined['class'].unique())}")

# ---------------------------------------------------------------------------
# STEP 8b (debug-only, gated by SAVE_REJECTED_FOR_REVIEW): save every
# rejected detection for manual review
# ---------------------------------------------------------------------------
# Not part of the permanent pipeline - added to sanity-check the Step 6/7
# debris fixes before trusting them across the rest of the sample videos.
# Same crop/save logic as Step 8, but for every detection either check threw
# out, tagged with which check rejected it and why, so real particles being
# wrongly discarded would show up here rather than silently vanishing. Left
# in as an opt-in toggle (off by default) rather than deleted, in case a
# future video's debris check needs the same audit.
if SAVE_REJECTED_FOR_REVIEW:
    rejected_dir = "pipeline_output/rejected_for_review"
    os.makedirs(rejected_dir, exist_ok=True)

    rejected_manifest_rows = []
    for i, r in enumerate(rejected_rows):
        frame_stack = demosaiced[:, :, :, int(r["frame"])]
        cx, cy = int(r["x"]), int(r["y"])
        half = CROP_SIZE // 2
        y0, y1 = max(0, cy - half), min(frame_stack.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(frame_stack.shape[1], cx + half)
        patch = frame_stack[y0:y1, x0:x1, :]

        base_name = f"{video_tag}_rejected{i}_frame{int(r['frame'])}_{r['reason']}"
        np.save(os.path.join(rejected_dir, base_name + ".npy"), patch)
        plt.imsave(os.path.join(rejected_dir, base_name + "_preview.png"), patch[:, :, 0], cmap="gray")

        rejected_manifest_rows.append(
            {
                "patch_file": base_name + ".npy",
                "frame": int(r["frame"]),
                "x": cx,
                "y": cy,
                "reason": r["reason"],
                "detail": r["detail"],
                # z_value: the comparison-frame z-field peak this was actually
                # rejected on (motion_check only - the value DEBRIS_CHECK_Z_THRESH
                # was compared against). displacement: trackpy's matched-centroid
                # distance, computed for diagnostic purposes even when the
                # z-check already decided the rejection (see diag_displacement
                # in Step 6) - NaN where no comparison-frame match existed.
                "z_value": r.get("z_value", np.nan),
                "displacement": r.get("displacement", np.nan),
                "source_file": INPUT_FILE,
            }
        )

    rejected_manifest = pd.DataFrame(rejected_manifest_rows)
    rejected_manifest.to_csv(os.path.join(rejected_dir, "rejected_manifest.csv"), index=False)
    print(f"\n[DEBUG] Saved {len(rejected_manifest_rows)} rejected detections for review to "
          f"'{rejected_dir}/' ({sum(r['reason'] == 'motion_check' for r in rejected_rows)} from the "
          f"per-frame motion check, {sum(r['reason'] == 'recurrence_check' for r in rejected_rows)} "
          f"from the cross-frame recurrence check)")

print("\nDone. Open pipeline_output/ in VS Code's file explorer to see the results:")
print("  - check_01_demosaiced_frame.png, check_02_background_reference.png, check_03_detections.png : spot-check images (overwritten each run - reflect the MOST RECENT video only)")
print("  - patches/*.npy : 4-channel (0/45/90/135) cropped particle arrays, ready for a classifier")
print("  - patches/*_preview.png : quick-look single-channel images, for browsing only")
print("  - manifest.csv : accumulates across every video you run - one row per patch, open it like a spreadsheet")