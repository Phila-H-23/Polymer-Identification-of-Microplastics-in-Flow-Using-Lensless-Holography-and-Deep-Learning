"""
localvar_bgcorr_recheck.py

Read-only, no masking/filtering/retraining. Re-tests the ORIGINAL
per-video local-variance signature (the one that first flagged PE/PMMA/PVC,
up to a 36x spread between videos) on the data it was actually measured
on - pol4_bgcorr and pol4_bgcorr_vnorm patches - after the last two
checks (raw-frame threshold mask, then a raw-frame universal-artifact
correlation) both came back null/contradictory on RAW data.

Step 1 metric: local_var_proxy(img, window=5) - reconstructed to be
IDENTICAL to the function used in the original background-patch-gallery
diagnostic (not saved as a script at the time, run inline): mean of the
5x5-window local-variance map, i.e. mean(uniform_filter(x**2, 5) -
uniform_filter(x, 5)**2), applied to the mean-of-4-channels image of
each patch - same window, same channel-reduction, same aggregation
(mean over the whole patch's map -> one scalar per patch) as before.

Two stages:
  (a) pol4_bgcorr        - patches loaded directly from patches_bgcorrected/
                            or patches_background_bgcorrected/, /255 scaled,
                            no further transform.
  (b) pol4_bgcorr_vnorm   - the SAME patches with apply_vnorm's per-video
                            (patch - video_bg_mean) / video_bg_std applied
                            first (video_background_stats.json), exactly
                            matching what --channels pol4_bgcorr_vnorm
                            feeds the network, before taking the 4-channel
                            mean and computing local_var_proxy.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
BG_BGCORR_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_background_bgcorrected.csv")
BG_BGCORR_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_background_bgcorrected")
VIDEO_BG_STATS_PATH = os.path.join(DATA_DIR, "pipeline_output", "video_background_stats.json")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

RECALL_VNORM = {"PE": 0.556, "PET": 0.196, "PMMA": 0.557, "PP": 0.055, "PS": 0.119, "PVA": 0.000, "PVC": 0.847}

DUSTY_EXAMPLE_VIDEOS = ["PE1_600.mat", "PVC1_600.mat", "PMMA2_600.mat"]
CLEAN_EXAMPLE_VIDEOS = ["PET1_600.mat", "PP1_600.mat", "PS1_600.mat"]


def local_var_proxy(img, window=5):
    img = img.astype(np.float64)
    m = ndimage.uniform_filter(img, size=window)
    m2 = ndimage.uniform_filter(img ** 2, size=window)
    return (m2 - m ** 2).mean()


def main():
    with open(VIDEO_BG_STATS_PATH) as f:
        video_stats = json.load(f)

    bg = pd.read_csv(BG_BGCORR_MANIFEST)

    # ------------------------------------------------------------------
    # Step 2 FIRST (visual check before trusting any number)
    # ------------------------------------------------------------------
    print("=== Step 2: visual side-by-side check (pol4_bgcorr stage, mean of 4 channels) ===")
    n_examples = 4
    videos_shown = DUSTY_EXAMPLE_VIDEOS + CLEAN_EXAMPLE_VIDEOS
    fig, axes = plt.subplots(len(videos_shown), n_examples, figsize=(3 * n_examples, 3 * len(videos_shown)))
    for vi, video in enumerate(videos_shown):
        sub = bg[bg["source_file"] == video].sample(n=min(n_examples, len(bg[bg["source_file"] == video])), random_state=0)
        for j in range(n_examples):
            ax = axes[vi, j]
            if j < len(sub):
                row = sub.iloc[j]
                patch = np.load(os.path.join(BG_BGCORR_PATCH_DIR, row["patch_file"])).astype(np.float32)
                mean_img = patch.mean(axis=-1)
                ax.imshow(mean_img, cmap="gray", vmin=mean_img.min(), vmax=mean_img.max())
            ax.axis("off")
        label = video.replace("_600.mat", "") + (" (flagged dusty)" if video in DUSTY_EXAMPLE_VIDEOS else " (clean candidate)")
        fig.text(0.005, 1 - (vi + 0.5) / len(videos_shown), label, fontsize=10, fontweight="bold", va="center")
    plt.tight_layout(rect=[0.09, 0, 1, 1])
    fig.savefig(os.path.join(RESULTS_DIR, "localvar_bgcorr_visual_check.png"), dpi=110)
    plt.close(fig)
    print("Saved -> results/localvar_bgcorr_visual_check.png - VIEW THIS BEFORE TRUSTING THE NUMBERS BELOW")

    # ------------------------------------------------------------------
    # Step 1 + 3
    # ------------------------------------------------------------------
    print("\n=== Step 1/3: local_var_proxy per patch, both stages ===")
    lv_bgcorr = []
    lv_vnorm = []
    for _, row in bg.iterrows():
        patch = np.load(os.path.join(BG_BGCORR_PATCH_DIR, row["patch_file"])).astype(np.float32) / 255.0
        mean_img_bgcorr = patch.mean(axis=-1)
        lv_bgcorr.append(local_var_proxy(mean_img_bgcorr))

        s = video_stats[row["source_file"]]
        patch_vnorm = (patch - s["mean"]) / (s["std"] + 1e-6)
        mean_img_vnorm = patch_vnorm.mean(axis=-1)
        lv_vnorm.append(local_var_proxy(mean_img_vnorm))

    bg = bg.copy()
    bg["lv_bgcorr"] = lv_bgcorr
    bg["lv_vnorm"] = lv_vnorm

    per_video = bg.groupby(["source_file", "class"])[["lv_bgcorr", "lv_vnorm"]].agg(["count", "mean"])
    per_video.columns = ["_".join(c) for c in per_video.columns]
    per_video = per_video.reset_index().sort_values("lv_bgcorr_mean", ascending=False)
    pd.set_option("display.width", 160)
    print("\nPer-video mean local_var_proxy (background-only), both stages:")
    print(per_video.to_string(index=False))

    print(f"\nRange across videos, pol4_bgcorr stage: {per_video['lv_bgcorr_mean'].min():.4f} - "
          f"{per_video['lv_bgcorr_mean'].max():.4f}  ({per_video['lv_bgcorr_mean'].max()/per_video['lv_bgcorr_mean'].min():.1f}x spread)")
    print(f"Range across videos, pol4_bgcorr_vnorm stage: {per_video['lv_vnorm_mean'].min():.4f} - "
          f"{per_video['lv_vnorm_mean'].max():.4f}  ({per_video['lv_vnorm_mean'].max()/per_video['lv_vnorm_mean'].min():.1f}x spread)")

    per_class = bg.groupby("class")[["lv_bgcorr", "lv_vnorm"]].agg(["count", "mean"])
    per_class.columns = ["_".join(c) for c in per_class.columns]
    per_class = per_class.reset_index()
    per_class["recall_vnorm"] = per_class["class"].map(RECALL_VNORM)
    per_class = per_class.sort_values("lv_vnorm_mean", ascending=False)
    print("\nPer-class mean local_var_proxy (background-only), both stages, vs vnorm recall:")
    print(per_class.to_string(index=False))

    corr_bgcorr = np.corrcoef(per_class["lv_bgcorr_mean"], per_class["recall_vnorm"])[0, 1]
    corr_vnorm = np.corrcoef(per_class["lv_vnorm_mean"], per_class["recall_vnorm"])[0, 1]
    print(f"\nPearson r, pol4_bgcorr-stage variance vs vnorm recall: {corr_bgcorr:+.4f}")
    print(f"Pearson r, pol4_bgcorr_vnorm-stage variance vs vnorm recall: {corr_vnorm:+.4f}")

    per_video.to_csv(os.path.join(RESULTS_DIR, "localvar_bgcorr_recheck_by_video.csv"), index=False)
    per_class.to_csv(os.path.join(RESULTS_DIR, "localvar_bgcorr_recheck_by_class.csv"), index=False)
    print(f"\nSaved -> results/localvar_bgcorr_recheck_by_video.csv, results/localvar_bgcorr_recheck_by_class.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(per_class["lv_vnorm_mean"], per_class["recall_vnorm"])
    for _, r in per_class.iterrows():
        ax.annotate(r["class"], (r["lv_vnorm_mean"], r["recall_vnorm"]), fontsize=9, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean local_var_proxy (pol4_bgcorr_vnorm stage)")
    ax.set_ylabel("background-only recall (vnorm)")
    ax.set_title(f"local variance vs recall (r={corr_vnorm:+.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "localvar_bgcorr_recheck_scatter.png"), dpi=110)
    plt.close(fig)

    print("\n=== Verdict ===")
    if abs(corr_vnorm) > 0.5:
        print("The original finding HOLDS UP on the data it was measured on: per-class local "
              "variance at the vnorm stage correlates meaningfully with vnorm background-only recall.")
    else:
        print("The original finding does NOT correlate with recall even measured on its own data "
              "(pol4_bgcorr / pol4_bgcorr_vnorm) - the per-video local-variance spread is real, but "
              "it is not what's driving the PE/PMMA/PVC recall pattern either.")


if __name__ == "__main__":
    main()
