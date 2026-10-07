"""
build_background_dataset.py

Builds a background-only control dataset: one 64x64x4 patch per real
manifest.csv row, drawn from the SAME (source_file, frame) that row's real
particle came from, at a random position that contains no detected
particle. If a network trained on this can still discriminate polymer
class, that's a session/video confound (lighting, exposure, sensor
fixed-pattern, video-specific artefacts) riding along with the class
label, not a particle-derived signal - independent of, and complementary
to, the Stokes/dolp findings in scripts/stokes_diagnostic*.py.

DETECTION / EXCLUSION SET: there is no saved full detection database in
this pipeline (checked: SAVE_REJECTED_FOR_REVIEW defaults to False and no
rejected_manifest.csv exists on disk) - manifest.csv's own (x, y) per
(source_file, frame) IS the only recorded detection data, so it is used as
the exclusion set: for a given frame, every manifest row sharing that
(source_file, frame) is excluded, not just the one row being matched.

EXCLUSION RULE: real patches are 64x64 boxes centred on each detection
(CROP_SIZE=64, half=32, copied from holo_pipeline_starter.py Step 8). Two
such boxes are non-overlapping once their centres are >=64px apart (in
Chebyshev/max-norm, since both are axis-aligned squares of the same size).
"no overlap ... and at least 32px from one" is implemented as a single
Chebyshev-distance threshold of 64 + 32 = 96px between the candidate
centre and every real detection centre in that frame - i.e. not just
non-overlapping boxes, but a further 32px clearance beyond that.

FRAME LOADING: videos are read via ONE bulk contiguous read per video
(`ds[:]`, ~3GB/74s measured directly on PET2_600.mat), never scattered
per-frame indexing - a single scattered h5py read on these OneDrive-synced
files was timed at >120s and killed as a background task before it
returned, consistent with earlier findings in this project (see
stokes_diagnostic.py's docstring) that random-access HDF5 reads here are
pathologically slow while one bulk block read is not. Needed frame indices
per video span nearly the full [0,600) range (checked directly), so a
bulk full-video read is required rather than a small prefix.

AXIS ORDERING: load_mat_array's h5py branch reverses ALL axes of the
on-disk array to correct MATLAB v7.3's storage order. A bulk `ds[:]` read
here is done WITHOUT that full reversal (would need a second 3GB copy);
instead each needed frame is pulled out and transposed individually:
block[i, 0].T reverses just that frame's (W, H) pair, which is
mathematically identical to what the full reversal would produce for that
one frame (confirmed: full-array axis reversal of (n,1,W,H) ->
(H,W,1,n), so new[:, :, 0, i] == old[i, 0, :, :].T).

SPLIT ASSIGNMENT: the task asked to copy "the video-level assignment used
in 'split' ... whichever split most of that video's real patches went
to". Checked directly, twice:
  - manifest.csv's 'split' column (the particle-level 70/15/15 stratified
    split) has TRAIN as the per-video MAJORITY for EVERY SINGLE video (it's
    a particle-level split, not a video-level one) - a literal majority
    vote on 'split' puts 100% of background patches in train, zero in
    val/test.
  - 'split_video_holdout' IS genuinely video-clean (every video is either
    100% one split, or a clean train/val mix, never mixed with test) - but
    within a train/val-mix video, train is STILL the majority, so val is
    never any video's mode either. A majority vote on this column instead
    gives train/test but *still* zero val.
Both literal readings of "majority vote" are degenerate: they leave val
completely empty, which makes the "test macro-F1" (and implicitly,
working model selection on val) this control is meant to report
impossible. Fixed by copying split_video_holdout PER ROW instead of
per-video-majority: each background patch inherits the split of the real
particle it was paired with (same source_file/frame/particle_id), stored
under the column name 'split' in the output manifest (matching what
--split particle / the default SPLIT_CONFIG expects). This still ties
every background patch to "the split assignment used for that video" (it
IS that video's split_video_holdout scheme, just applied at row rather
than whole-video granularity for the train/val-mix videos - the 100%-test
videos are unaffected, copying trivially reduces to the same per-video
label there) while actually producing a usable train/val/test partition:
confirmed train=369/val=80/test=434, matching the real video_holdout
totals exactly, with every class present in all three splits.

Because exactly one background patch is generated per real manifest row,
per-video (and hence per-class, since class = f(video)) patch counts match
the real manifest exactly by construction - no separate stratification
needed to satisfy "match the per-video patch count ... so class
proportions are identical".
"""

import os
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
MANIFEST_PATH = os.path.join(DATA_DIR, "pipeline_output", "manifest.csv")
OUT_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_background.csv")
OUT_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_background")

CROP_SIZE = 64
HALF = CROP_SIZE // 2
EXCLUDE_DIST = 96  # 64 (box non-overlap) + 32 (extra buffer), Chebyshev
MAX_ATTEMPTS = 3000
SEED = 42


def demosaic(snapshot):
    I_0 = snapshot[1::2, 1::2]
    I_45 = snapshot[0::2, 1::2]
    I_90 = snapshot[0::2, 0::2]
    I_135 = snapshot[1::2, 0::2]
    return I_0, I_45, I_90, I_135


def load_video_block(path):
    """Mirrors load_mat_array's dual-format handling (holo_pipeline_starter.py):
    older MATLAB files load fine with scipy (frame axis LAST, already H,W
    order); v7.3/HDF5 files need h5py (frame axis FIRST, W,H order, needs a
    per-frame transpose - see get_frame_hw). Returns (block, mode).

    Confirmed directly: PET4_600.mat is scipy-loadable (old format) while
    PE1_600.mat / PET2_600.mat etc raise here and need the h5py fallback -
    same "your small/big files differ" situation load_mat_array's own
    docstring describes, just discovered on a 288-frame file instead of
    the small background calibration file.
    """
    try:
        mat = sio.loadmat(path)
        var_name = [k for k in mat.keys() if not k.startswith("__")][0]
        return mat[var_name], "scipy"  # (H, W, 1, n) or (H, W, n)
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            var_name = [k for k in f.keys() if not k.startswith("#")][0]
            return f[var_name][:], "h5py"  # (n, 1, W, H) or (n, W, H)


def get_frame_hw(block, mode, frame_idx):
    """Returns frame_idx as a (H, W) 2D array regardless of load path."""
    if mode == "scipy":
        return block[:, :, 0, frame_idx] if block.ndim == 4 else block[:, :, frame_idx]
    else:
        return block[frame_idx, 0].T if block.ndim == 4 else block[frame_idx].T


def sample_background_center(rng, exclusions, w, h):
    """Rejection-sample a (cx, cy) at least EXCLUDE_DIST (Chebyshev) from
    every (x, y) in exclusions, with the full 64x64 box in-bounds. Falls
    back to the attempt with the largest minimum distance if none clears
    the threshold within MAX_ATTEMPTS (documented, and counted/reported)."""
    lo_x, hi_x = HALF, w - HALF
    lo_y, hi_y = HALF, h - HALF
    best_c, best_d = None, -1.0
    for _ in range(MAX_ATTEMPTS):
        cx = rng.integers(lo_x, hi_x)
        cy = rng.integers(lo_y, hi_y)
        if exclusions:
            d = min(max(abs(cx - ex), abs(cy - ey)) for ex, ey in exclusions)
        else:
            d = float("inf")
        if d >= EXCLUDE_DIST:
            return cx, cy, True
        if d > best_d:
            best_d, best_c = d, (cx, cy)
    return best_c[0], best_c[1], False


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    os.makedirs(OUT_PATCH_DIR, exist_ok=True)

    # per-video split label, reported for visibility only - actual
    # assignment is per-row (see docstring's SPLIT ASSIGNMENT note: a
    # majority vote on either 'split' or 'split_video_holdout' leaves val
    # empty, so each background patch instead copies its OWN paired real
    # particle's split_video_holdout value further down).
    video_split = manifest.groupby("source_file")["split_video_holdout"].agg(
        lambda s: s.value_counts().idxmax())
    print("Per-video majority split (reference only, NOT what's used - see docstring):")
    print(video_split.to_string())

    rng = np.random.default_rng(SEED)
    manifest_sorted = manifest.sort_values(["source_file", "frame", "particle_id"]).reset_index(drop=True)

    out_rows = []
    n_fallback = 0
    for source_file, video_df in manifest_sorted.groupby("source_file", sort=False):
        video_path = os.path.join(DATA_DIR, source_file)
        print(f"\n{source_file}: loading (bulk contiguous read)...")
        block, mode = load_video_block(video_path)
        print(f"  shape {block.shape}, building {len(video_df)} background patches "
              f"across {video_df['frame'].nunique()} frames")

        cls = video_df["class"].iloc[0]

        for frame_idx, frame_df in video_df.groupby("frame"):
            frame2d = get_frame_hw(block, mode, int(frame_idx))
            I0, I45, I90, I135 = demosaic(frame2d)
            h, w = I0.shape  # (1024, 1224)
            exclusions = list(zip(frame_df["x"].astype(int), frame_df["y"].astype(int)))

            for _, row in frame_df.iterrows():
                cx, cy, ok = sample_background_center(rng, exclusions, w, h)
                if not ok:
                    n_fallback += 1
                y0, y1 = cy - HALF, cy + HALF
                x0, x1 = cx - HALF, cx + HALF
                patch = np.stack(
                    [I0[y0:y1, x0:x1], I45[y0:y1, x0:x1], I90[y0:y1, x0:x1], I135[y0:y1, x0:x1]],
                    axis=-1,
                ).astype(np.uint8)

                base_name = f"{os.path.splitext(source_file)[0]}_bgparticle{int(row['particle_id'])}_frame{int(frame_idx)}"
                npy_path = os.path.join(OUT_PATCH_DIR, base_name + ".npy")
                np.save(npy_path, patch)

                out_rows.append({
                    "patch_file": base_name + ".npy",
                    "particle_id": int(row["particle_id"]),
                    "frame": int(frame_idx),
                    "x": cx, "y": cy,
                    "source_file": source_file,
                    "class": cls,
                    "split": row["split_video_holdout"],
                })
        del block

    out_manifest = pd.DataFrame(out_rows)
    out_manifest.to_csv(OUT_MANIFEST, index=False)

    print(f"\nBuilt {len(out_manifest)} background patches -> {OUT_MANIFEST}")
    print(f"Fallback (no position cleared {EXCLUDE_DIST}px) used for {n_fallback}/{len(out_manifest)} patches")
    print("\nSplit distribution:")
    print(out_manifest["split"].value_counts().to_string())
    print("\nClass distribution (background vs real, should match):")
    print(pd.DataFrame({
        "background": out_manifest["class"].value_counts(),
        "real": manifest["class"].value_counts(),
    }))


if __name__ == "__main__":
    main()
