"""
build_dust_masks.py

Phase 0 + Phase 1 of the dust/artifact-mask feasibility check. READ-ONLY
with respect to the existing dataset: writes new mask .npy/.png files and
prints a feasibility table, does not touch pipeline_output/patches*,
manifest*.csv, or results/, and does NOT retrain anything.

PHASE 0 - per-video dust masks for PE1, PE2, PMMA2, PVC1, PVC2:
  1. local_variance() is copied verbatim from holo_pipeline_starter.py
     (window=9, ndimage.uniform_filter-based) - the same building block
     the pipeline's own particle detector uses.
  2. Applied to RAW (uncorrected) intensity frames - mean of the 4
     demosaiced channels - at exactly the frame indices each video's
     background-only patches were sampled from (manifest_background.csv's
     'frame' column for that source_file), NOT the full detection pass.
     Raw, not flat-fielded/bgcorr-corrected: the dust artifact is a
     property of the optical path itself, present regardless of any
     correction, and using raw data avoids entangling this diagnostic
     with the correction pipeline's own behaviour.
  3. Per video, local-variance maps are computed PER sampled frame, then
     AVERAGED across that video's sampled frames (not: average the frames
     then take one local-variance map) - this highlights pixels that show
     persistent fringe/edge structure across multiple frames, which is
     what a fixed dust/debris artifact looks like, and is more robust to
     transient per-frame noise than a single-frame map would be.
  4. Clean reference video: PET1 (one of the explicitly-allowed choices;
     PET4 and PVA2 excluded per the task's instruction). Its local-
     variance map's 95th and 99th percentile are reported as candidate
     noise floors; the 99th percentile is used as THE noise floor for
     the 2x/3x/4x threshold sweep (more robust to being a genuine
     ceiling than the 95th).
  5. For each dusty video, mask = (local_var_map > multiplier * floor),
     reported at 2x/3x/4x. The 3x mask is treated as the default/primary
     mask (dilated + saved) unless its coverage looks implausible
     (near-0% or near-100%), in which case the nearer-to-reasonable of
     2x/4x is used instead and this is called out explicitly.
  6. Binary-dilated by 3 pixels (3 iterations of scipy's default
     3x3 connectivity structuring element) for margin.
  7. Saved as results/dust_mask_<video>.npy and an overlay PNG
     (mask in semi-transparent red over that video's grayscale averaged
     raw background frame) at results/dust_mask_<video>_overlay.png.

PHASE 1 - feasibility check, no retraining:
  For each of the 5 dusty videos x {real particle patches, background-only
  patches} (the pol4_bgcorr_vnorm dataset - manifest_bgcorrected.csv /
  manifest_background_bgcorrected.csv, positions unchanged from the
  uncorrected originals since correction only changes pixel values):
    - each patch's actual on-disk (h, w) is read from its own .npy file
      (not a hardcoded 64), since edge-of-frame patches can be smaller.
    - overlap fraction = mean(mask[y0:y1, x0:x1]) using the SAME clip
      formula (half=32) used everywhere else in this project to crop
      patches from the demosaiced frame.
    - counts original / surviving at >0% / >10% / >25% overlap tolerance
      are reported per (video, class, patch_type).
"""
import os
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
REAL_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest.csv")
BG_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_background.csv")
REAL_BGCORR_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_bgcorrected.csv")
BG_BGCORR_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_background_bgcorrected.csv")
REAL_BGCORR_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_bgcorrected")
BG_BGCORR_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_background_bgcorrected")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

DUSTY_VIDEOS = ["PE1_600.mat", "PE2_600.mat", "PMMA2_600.mat", "PVC1_600.mat", "PVC2_600.mat"]
CLEAN_VIDEO = "PET1_600.mat"
HALF = 32
WINDOW = 9
DILATE_ITER = 3
THRESH_MULTIPLIERS = [2, 3, 4]
DEFAULT_MULTIPLIER = 3
PLAUSIBLE_COVERAGE_RANGE = (0.01, 0.60)  # outside this, fall back to nearer-reasonable multiplier


def demosaic(snapshot):
    I_0 = snapshot[1::2, 1::2]
    I_45 = snapshot[0::2, 1::2]
    I_90 = snapshot[0::2, 0::2]
    I_135 = snapshot[1::2, 0::2]
    return I_0, I_45, I_90, I_135


def local_variance(frame, window=WINDOW):
    """Verbatim from holo_pipeline_starter.py."""
    frame = frame.astype(float)
    local_mean = ndimage.uniform_filter(frame, size=window)
    local_sqmean = ndimage.uniform_filter(frame ** 2, size=window)
    return local_sqmean - local_mean ** 2


def load_video_block(path):
    try:
        mat = sio.loadmat(path)
        var_name = [k for k in mat.keys() if not k.startswith("__")][0]
        return mat[var_name], "scipy"
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            var_name = [k for k in f.keys() if not k.startswith("#")][0]
            return f[var_name][:], "h5py"


def get_raw_frame(block, mode, idx):
    if mode == "scipy":
        return block[:, :, 0, idx] if block.ndim == 4 else block[:, :, idx]
    else:
        return block[idx, 0].T if block.ndim == 4 else block[idx].T


def video_avg_frame_and_lv(video, bg_manifest):
    """Returns (avg_intensity_frame, avg_local_variance_map) - both
    (1024, 1224) float64 - over this video's background-only sampled
    frames, RAW (uncorrected) intensity = mean of 4 demosaiced channels."""
    frames_needed = sorted(bg_manifest.loc[bg_manifest["source_file"] == video, "frame"].unique())
    block, mode = load_video_block(os.path.join(DATA_DIR, video))
    intensity_frames = []
    lv_maps = []
    for f_idx in frames_needed:
        raw2d = get_raw_frame(block, mode, int(f_idx)).astype(np.float64)
        I0, I45, I90, I135 = demosaic(raw2d)
        intensity = (I0 + I45 + I90 + I135) / 4.0
        intensity_frames.append(intensity)
        lv_maps.append(local_variance(intensity))
    del block
    avg_frame = np.mean(intensity_frames, axis=0)
    avg_lv = np.mean(lv_maps, axis=0)
    return avg_frame, avg_lv, len(frames_needed)


def crop_region(h_total, w_total, cx, cy, half=HALF):
    y0, y1 = max(0, cy - half), min(h_total, cy + half)
    x0, x1 = max(0, cx - half), min(w_total, cx + half)
    return y0, y1, x0, x1


def main():
    bg_manifest = pd.read_csv(BG_MANIFEST)

    print("=== Phase 0: clean reference video ===")
    clean_frame, clean_lv, clean_n = video_avg_frame_and_lv(CLEAN_VIDEO, bg_manifest)
    p95 = np.percentile(clean_lv, 95)
    p99 = np.percentile(clean_lv, 99)
    print(f"{CLEAN_VIDEO}: {clean_n} sampled frames averaged")
    print(f"  local-variance map: 95th pct = {p95:.4f}, 99th pct = {p99:.4f}")
    noise_floor = p99
    print(f"  using 99th percentile ({noise_floor:.4f}) as the noise floor for thresholding")

    print("\n=== Phase 0: dusty videos ===")
    video_masks = {}
    video_frames = {}
    coverage_report = []
    for video in DUSTY_VIDEOS:
        avg_frame, avg_lv, n_frames = video_avg_frame_and_lv(video, bg_manifest)
        video_frames[video] = avg_frame
        h_total, w_total = avg_lv.shape

        coverages = {}
        masks_by_mult = {}
        for mult in THRESH_MULTIPLIERS:
            mask = avg_lv > (mult * noise_floor)
            cov = mask.mean()
            coverages[mult] = cov
            masks_by_mult[mult] = mask

        chosen_mult = DEFAULT_MULTIPLIER
        if not (PLAUSIBLE_COVERAGE_RANGE[0] <= coverages[DEFAULT_MULTIPLIER] <= PLAUSIBLE_COVERAGE_RANGE[1]):
            alt = min([m for m in THRESH_MULTIPLIERS if m != DEFAULT_MULTIPLIER],
                      key=lambda m: abs(coverages[m] - 0.15))
            print(f"  {video}: 3x coverage ({coverages[DEFAULT_MULTIPLIER]*100:.1f}%) looks implausible, "
                  f"falling back to {alt}x ({coverages[alt]*100:.1f}%)")
            chosen_mult = alt

        mask = masks_by_mult[chosen_mult]
        mask_dilated = ndimage.binary_dilation(mask, iterations=DILATE_ITER)
        video_masks[video] = mask_dilated

        print(f"{video}: {n_frames} sampled frames | coverage 2x={coverages[2]*100:.2f}%  "
              f"3x={coverages[3]*100:.2f}%  4x={coverages[4]*100:.2f}%  "
              f"| chosen={chosen_mult}x, dilated coverage={mask_dilated.mean()*100:.2f}%")

        coverage_report.append({
            "video": video, "n_frames": n_frames,
            "coverage_2x": coverages[2], "coverage_3x": coverages[3], "coverage_4x": coverages[4],
            "chosen_multiplier": chosen_mult, "dilated_coverage": mask_dilated.mean(),
        })

        # save mask + overlay
        mask_name = video.replace("_600.mat", "")
        np.save(os.path.join(RESULTS_DIR, f"dust_mask_{mask_name}.npy"), mask_dilated)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(avg_frame, cmap="gray")
        overlay = np.zeros((*mask_dilated.shape, 4))
        overlay[mask_dilated] = [1, 0, 0, 0.4]
        ax.imshow(overlay)
        ax.set_title(f"{mask_name} dust mask ({chosen_mult}x noise floor, dilated)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"dust_mask_{mask_name}_overlay.png"), dpi=110)
        plt.close(fig)

    print("\nSaved masks + overlays for:", ", ".join(v.replace("_600.mat", "") for v in DUSTY_VIDEOS))

    # ------------------------------------------------------------------
    # Phase 1: feasibility check
    # ------------------------------------------------------------------
    print("\n=== Phase 1: patch survival vs dust mask ===")
    real_manifest = pd.read_csv(REAL_BGCORR_MANIFEST)
    bg_bgcorr_manifest = pd.read_csv(BG_BGCORR_MANIFEST)

    h_total, w_total = next(iter(video_frames.values())).shape

    def overlap_fractions(manifest, patch_dir, video):
        sub = manifest[manifest["source_file"] == video]
        mask = video_masks[video]
        fracs = []
        for _, row in sub.iterrows():
            patch_path = os.path.join(patch_dir, row["patch_file"])
            h, w = np.load(patch_path).shape[:2]  # actual on-disk shape, not assumed
            cx, cy = int(row["x"]), int(row["y"])
            y0, y1, x0, x1 = crop_region(h_total, w_total, cx, cy)
            # region size should match (h, w); guard just in case
            region = mask[y0:y1, x0:x1]
            frac = region.mean() if region.size else 0.0
            fracs.append(frac)
        return np.array(fracs)

    rows = []
    for video in DUSTY_VIDEOS:
        cls = real_manifest.loc[real_manifest["source_file"] == video, "class"].iloc[0]
        for patch_type, manifest, patch_dir in [
            ("real", real_manifest, REAL_BGCORR_PATCH_DIR),
            ("background-only", bg_bgcorr_manifest, BG_BGCORR_PATCH_DIR),
        ]:
            fracs = overlap_fractions(manifest, patch_dir, video)
            n_total = len(fracs)
            n_gt0 = int((fracs <= 0.0).sum())
            n_gt10 = int((fracs <= 0.10).sum())
            n_gt25 = int((fracs <= 0.25).sum())
            rows.append({
                "video": video.replace("_600.mat", ""), "class": cls, "patch_type": patch_type,
                "n_total": n_total, "survive_gt0pct": n_gt0, "survive_gt10pct": n_gt10, "survive_gt25pct": n_gt25,
            })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))

    print("\n=== By class (summed across that class's videos) ===")
    cls_df = df.groupby(["class", "patch_type"])[["n_total", "survive_gt0pct", "survive_gt10pct", "survive_gt25pct"]].sum().reset_index()
    print(cls_df.to_string(index=False))

    print("\n=== Flags: classes dropping below 50% survival at >10% tolerance ===")
    cls_df["pct_survive_10"] = cls_df["survive_gt10pct"] / cls_df["n_total"]
    flagged = cls_df[cls_df["pct_survive_10"] < 0.5]
    if len(flagged):
        print(flagged.to_string(index=False))
    else:
        print("None - all classes/patch-types retain >=50% of patches at the >10% overlap tolerance.")

    df.to_csv(os.path.join(RESULTS_DIR, "dust_mask_feasibility.csv"), index=False)
    print(f"\nSaved -> {os.path.join(RESULTS_DIR, 'dust_mask_feasibility.csv')}")
    pd.DataFrame(coverage_report).to_csv(os.path.join(RESULTS_DIR, "dust_mask_coverage.csv"), index=False)
    print(f"Saved -> {os.path.join(RESULTS_DIR, 'dust_mask_coverage.csv')}")


if __name__ == "__main__":
    main()
