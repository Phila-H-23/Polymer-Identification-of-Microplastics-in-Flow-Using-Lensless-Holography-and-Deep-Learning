"""
artifact_exposure_check.py

Phase 0.5 - read-only, no retraining. Tests whether PE/PMMA/PVC's
elevated background-only recall correlates with proximity to the
UNIVERSAL hardware artifact (confirmed present in frames_bg_30.mat, PET1,
and every dusty video alike - see the previous diagnostic), rather than
those videos being differentially "dirtier."

STEP 1: continuous (unthresholded) local-variance artifact map, built
ONLY from frames_bg_30.mat (no sample chamber contents - pure hardware
signal). Same local_variance() building block as before (window=9,
ndimage.uniform_filter), averaged across all 30 calibration frames'
individual local-variance maps (not: average frames then one map -
consistent with the previous diagnostic's methodology).

STEP 2: for every one of the 16 videos, for both real and background-only
patches (pol4_bgcorr_vnorm dataset - manifest_bgcorrected.csv +
patches_bgcorrected/, manifest_background_bgcorrected.csv +
patches_background_bgcorrected/), "artifact exposure" = mean of the Step
1 map within that patch's actual footprint (position from the manifest,
on-disk (h, w) from the patch file itself, same half=32 crop formula
used everywhere else in this project).

STEP 3: per-video real-vs-background-only exposure gap.

STEP 4: per-class (pooled) background-only exposure vs the two recall
vectors supplied by the user (reference data, not recomputed here).
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
BG30_FILE = os.path.join(DATA_DIR, "frames_bg_30.mat")
REAL_BGCORR_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_bgcorrected.csv")
BG_BGCORR_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_background_bgcorrected.csv")
REAL_BGCORR_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_bgcorrected")
BG_BGCORR_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_background_bgcorrected")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

HALF = 32
WINDOW = 9

RECALL_VNORM = {"PE": 0.556, "PET": 0.196, "PMMA": 0.557, "PP": 0.055, "PS": 0.119, "PVA": 0.000, "PVC": 0.847}
RECALL_UNCORRECTED = {"PE": 0.652, "PET": 0.657, "PMMA": 0.043, "PP": 0.400, "PS": 0.465, "PVA": 0.190, "PVC": 0.329}


def demosaic(snapshot):
    I_0 = snapshot[1::2, 1::2]
    I_45 = snapshot[0::2, 1::2]
    I_90 = snapshot[0::2, 0::2]
    I_135 = snapshot[1::2, 0::2]
    return I_0, I_45, I_90, I_135


def local_variance(frame, window=WINDOW):
    frame = frame.astype(float)
    local_mean = ndimage.uniform_filter(frame, size=window)
    local_sqmean = ndimage.uniform_filter(frame ** 2, size=window)
    return local_sqmean - local_mean ** 2


def crop_region(h_total, w_total, cx, cy, half=HALF):
    y0, y1 = max(0, cy - half), min(h_total, cy + half)
    x0, x1 = max(0, cx - half), min(w_total, cx + half)
    return y0, y1, x0, x1


def main():
    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------
    print("=== Step 1: universal artifact map from frames_bg_30.mat ===")
    mat = sio.loadmat(BG30_FILE)
    var_name = [k for k in mat.keys() if not k.startswith("__")][0]
    bg_raw = mat[var_name]
    bg_frames = bg_raw[:, :, 0, :] if bg_raw.ndim == 4 else bg_raw
    n_frames = bg_frames.shape[-1]

    lv_maps = []
    for i in range(n_frames):
        raw2d = bg_frames[:, :, i].astype(np.float64)
        I0, I45, I90, I135 = demosaic(raw2d)
        intensity = (I0 + I45 + I90 + I135) / 4.0
        lv_maps.append(local_variance(intensity))
    artifact_map = np.mean(lv_maps, axis=0)  # (1024, 1224), continuous, unthresholded

    print(f"{n_frames} calibration frames averaged. artifact_map shape {artifact_map.shape}")
    print(f"  mean={artifact_map.mean():.4f} std={artifact_map.std():.4f} "
          f"min={artifact_map.min():.4f} max={artifact_map.max():.4f} "
          f"median={np.median(artifact_map):.4f}")

    np.save(os.path.join(RESULTS_DIR, "artifact_exposure_map.npy"), artifact_map)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.log1p(artifact_map), cmap="inferno")
    ax.set_title("Universal artifact map (log1p local-variance, frames_bg_30.mat)")
    fig.colorbar(im, ax=ax, label="log1p(local variance)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "artifact_exposure_map.png"), dpi=110)
    plt.close(fig)
    print(f"Saved -> results/artifact_exposure_map.npy, results/artifact_exposure_map.png")

    h_total, w_total = artifact_map.shape

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------
    print("\n=== Step 2: per-patch artifact exposure, all videos ===")
    real_manifest = pd.read_csv(REAL_BGCORR_MANIFEST)
    bg_manifest = pd.read_csv(BG_BGCORR_MANIFEST)

    def compute_exposures(manifest, patch_dir):
        exposures = []
        for _, row in manifest.iterrows():
            patch_path = os.path.join(patch_dir, row["patch_file"])
            h, w = np.load(patch_path).shape[:2]
            cx, cy = int(row["x"]), int(row["y"])
            y0, y1, x0, x1 = crop_region(h_total, w_total, cx, cy)
            region = artifact_map[y0:y1, x0:x1]
            exposures.append(region.mean() if region.size else np.nan)
        return np.array(exposures)

    real_manifest["exposure"] = compute_exposures(real_manifest, REAL_BGCORR_PATCH_DIR)
    bg_manifest["exposure"] = compute_exposures(bg_manifest, BG_BGCORR_PATCH_DIR)

    real_manifest["patch_type"] = "real"
    bg_manifest["patch_type"] = "background-only"
    combined = pd.concat([real_manifest[["source_file", "class", "patch_type", "exposure"]],
                           bg_manifest[["source_file", "class", "patch_type", "exposure"]]])

    table = combined.groupby(["source_file", "class", "patch_type"])["exposure"].agg(["count", "mean", "std"]).reset_index()
    table.columns = ["video", "class", "patch_type", "n", "mean_exposure", "std_exposure"]
    pd.set_option("display.width", 160)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(RESULTS_DIR, "artifact_exposure_by_video.csv"), index=False)
    print(f"\nSaved -> results/artifact_exposure_by_video.csv")

    # ------------------------------------------------------------------
    # Step 3
    # ------------------------------------------------------------------
    print("\n=== Step 3: within-video real vs background-only exposure gap ===")
    pivot = table.pivot_table(index=["video", "class"], columns="patch_type", values="mean_exposure").reset_index()
    pivot["gap_bg_minus_real"] = pivot["background-only"] - pivot["real"]
    pivot = pivot.sort_values("gap_bg_minus_real", ascending=False)
    print(pivot.to_string(index=False))

    print(f"\nVideos where background-only sits closer to the artifact than real (gap > 0): "
          f"{(pivot['gap_bg_minus_real'] > 0).sum()}/{len(pivot)}")
    print(f"Mean gap across all videos: {pivot['gap_bg_minus_real'].mean():.4f}")
    print(f"Mean gap for PE/PMMA/PVC videos: {pivot[pivot['class'].isin(['PE','PMMA','PVC'])]['gap_bg_minus_real'].mean():.4f}")
    print(f"Mean gap for PET/PP/PS/PVA videos: {pivot[~pivot['class'].isin(['PE','PMMA','PVC'])]['gap_bg_minus_real'].mean():.4f}")

    # ------------------------------------------------------------------
    # Step 4
    # ------------------------------------------------------------------
    print("\n=== Step 4: per-class background-only exposure vs recall ===")
    bg_only = combined[combined["patch_type"] == "background-only"]
    class_exposure = bg_only.groupby("class")["exposure"].agg(["count", "mean", "std"]).reset_index()
    class_exposure.columns = ["class", "n", "mean_exposure", "std_exposure"]
    class_exposure["recall_vnorm"] = class_exposure["class"].map(RECALL_VNORM)
    class_exposure["recall_uncorrected"] = class_exposure["class"].map(RECALL_UNCORRECTED)
    class_exposure = class_exposure.sort_values("mean_exposure", ascending=False)
    print(class_exposure.to_string(index=False))

    exp = class_exposure["mean_exposure"].values
    rec_vnorm = class_exposure["recall_vnorm"].values
    rec_unc = class_exposure["recall_uncorrected"].values

    corr_vnorm = np.corrcoef(exp, rec_vnorm)[0, 1]
    corr_unc = np.corrcoef(exp, rec_unc)[0, 1]
    print(f"\nPearson correlation, exposure vs vnorm recall: {corr_vnorm:+.4f}")
    print(f"Pearson correlation, exposure vs uncorrected recall: {corr_unc:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, rec, title, corr in [
        (axes[0], rec_vnorm, "vnorm recall", corr_vnorm),
        (axes[1], rec_unc, "uncorrected recall", corr_unc),
    ]:
        ax.scatter(exp, rec)
        for i, c in enumerate(class_exposure["class"]):
            ax.annotate(c, (exp[i], rec[i]), fontsize=9, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("mean background-only artifact exposure")
        ax.set_ylabel("background-only recall")
        ax.set_title(f"{title} (r={corr:+.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "artifact_exposure_vs_recall.png"), dpi=110)
    plt.close(fig)
    print(f"\nSaved -> results/artifact_exposure_vs_recall.png")

    print("\n=== Verdict ===")
    if abs(corr_vnorm) > 0.5 or abs(corr_unc) > 0.5:
        print("Meaningful correlation found - supports the exposure hypothesis (background-only "
              "recall tracks proximity to the universal hardware artifact).")
    else:
        print("No strong correlation (|r| <= 0.5 both conditions) - the exposure hypothesis is "
              "NOT well supported by this test; the PE/PMMA/PVC recall pattern likely needs a "
              "different explanation than proximity to this specific artifact map.")


if __name__ == "__main__":
    main()
