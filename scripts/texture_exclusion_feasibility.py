"""
texture_exclusion_feasibility.py

Phase 2a - read-only feasibility check, no retraining, no filtered
dataset written. Checks whether uniformly excluding high local-variance-
texture patches (vnorm stage, --channels pol4_bgcorr_vnorm) leaves enough
data per class to be worth actually retraining on.

Step 0: video-level correlation (n~16, each video's own mean texture
score vs its CLASS's known vnorm background-only recall - recall is a
per-class metric so repeats within a class's videos; this still tests
whether within-class video-to-video texture variation tracks recall,
with more points than the n=7 class-level test).

Step 1: threshold = 95th/99th percentile of PER-PATCH vnorm-stage texture
score, pooled over the four "clean" classes' (PET, PP, PS, PVA)
BACKGROUND-ONLY patches only - matches what the previous script actually
computed ("already computed" = background-only, not real).

Step 2: apply that threshold to EVERY patch - all 7 classes, BOTH real
and background-only (real patches' per-patch scores are computed fresh
here, extending the previous background-only-only run) - report
before/after counts and %retained.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import ndimage

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
REAL_BGCORR_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_bgcorrected.csv")
BG_BGCORR_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_background_bgcorrected.csv")
REAL_BGCORR_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_bgcorrected")
BG_BGCORR_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_background_bgcorrected")
VIDEO_BG_STATS_PATH = os.path.join(DATA_DIR, "pipeline_output", "video_background_stats.json")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PER_VIDEO_CSV = os.path.join(RESULTS_DIR, "localvar_bgcorr_recheck_by_video.csv")

RECALL_VNORM = {"PE": 0.556, "PET": 0.196, "PMMA": 0.557, "PP": 0.055, "PS": 0.119, "PVA": 0.000, "PVC": 0.847}
CLEAN_CLASSES = ["PET", "PP", "PS", "PVA"]


def local_var_proxy(img, window=5):
    img = img.astype(np.float64)
    m = ndimage.uniform_filter(img, size=window)
    m2 = ndimage.uniform_filter(img ** 2, size=window)
    return (m2 - m ** 2).mean()


def compute_vnorm_lv(manifest, patch_dir, video_stats):
    scores = []
    for _, row in manifest.iterrows():
        patch = np.load(os.path.join(patch_dir, row["patch_file"])).astype(np.float32) / 255.0
        s = video_stats[row["source_file"]]
        patch_vnorm = (patch - s["mean"]) / (s["std"] + 1e-6)
        mean_img = patch_vnorm.mean(axis=-1)
        scores.append(local_var_proxy(mean_img))
    return np.array(scores)


def main():
    with open(VIDEO_BG_STATS_PATH) as f:
        video_stats = json.load(f)

    # ------------------------------------------------------------------
    # Step 0: video-level correlation
    # ------------------------------------------------------------------
    print("=== Step 0: video-level local-variance vs recall correlation ===")
    per_video = pd.read_csv(PER_VIDEO_CSV)
    per_video["recall_vnorm"] = per_video["class"].map(RECALL_VNORM)
    x = per_video["lv_vnorm_mean"].values
    y = per_video["recall_vnorm"].values
    r_video = np.corrcoef(x, y)[0, 1]
    n_video = len(per_video)
    print(f"n = {n_video} videos")
    print(per_video[["source_file", "class", "lv_vnorm_mean", "recall_vnorm"]].to_string(index=False))
    print(f"\nPearson r (video-level): {r_video:+.4f}  (n={n_video})")
    print(f"(for reference, class-level was r=+0.9544, n=7)")

    # ------------------------------------------------------------------
    # Step 1: threshold from clean-class background-only per-patch scores
    # ------------------------------------------------------------------
    print("\n=== Step 1: threshold from PET/PP/PS/PVA background-only per-patch texture scores ===")
    bg_manifest = pd.read_csv(BG_BGCORR_MANIFEST)
    bg_manifest["lv_vnorm"] = compute_vnorm_lv(bg_manifest, BG_BGCORR_PATCH_DIR, video_stats)

    clean_scores = bg_manifest.loc[bg_manifest["class"].isin(CLEAN_CLASSES), "lv_vnorm"].values
    p95 = np.percentile(clean_scores, 95)
    p99 = np.percentile(clean_scores, 99)
    print(f"Pooled n = {len(clean_scores)} background-only patches from {CLEAN_CLASSES}")
    print(f"95th percentile texture score: {p95:.4f}")
    print(f"99th percentile texture score: {p99:.4f}")

    # ------------------------------------------------------------------
    # Step 2: apply threshold to everything
    # ------------------------------------------------------------------
    print("\n=== Step 2: retention after exclusion, all classes, both patch types ===")
    real_manifest = pd.read_csv(REAL_BGCORR_MANIFEST)
    real_manifest["lv_vnorm"] = compute_vnorm_lv(real_manifest, REAL_BGCORR_PATCH_DIR, video_stats)

    real_manifest["patch_type"] = "real"
    bg_manifest["patch_type"] = "background-only"
    combined = pd.concat([
        real_manifest[["class", "patch_type", "lv_vnorm"]],
        bg_manifest[["class", "patch_type", "lv_vnorm"]],
    ])

    rows = []
    for (cls, ptype), sub in combined.groupby(["class", "patch_type"]):
        n_total = len(sub)
        n_keep_95 = int((sub["lv_vnorm"] <= p95).sum())
        n_keep_99 = int((sub["lv_vnorm"] <= p99).sum())
        rows.append({
            "class": cls, "patch_type": ptype, "n_total": n_total,
            "keep_at_95th": n_keep_95, "pct_retained_95th": n_keep_95 / n_total * 100,
            "keep_at_99th": n_keep_99, "pct_retained_99th": n_keep_99 / n_total * 100,
        })

    df = pd.DataFrame(rows).sort_values(["class", "patch_type"])
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "texture_exclusion_feasibility.csv"), index=False)
    print(f"\nSaved -> results/texture_exclusion_feasibility.csv")

    print("\n=== By class (real + background-only combined) ===")
    cls_df = df.groupby("class")[["n_total", "keep_at_95th", "keep_at_99th"]].sum().reset_index()
    cls_df["pct_retained_95th"] = cls_df["keep_at_95th"] / cls_df["n_total"] * 100
    cls_df["pct_retained_99th"] = cls_df["keep_at_99th"] / cls_df["n_total"] * 100
    print(cls_df.to_string(index=False))

    print("\n=== Flags: classes/patch-types below 50% retention at the 95th-percentile threshold ===")
    flagged = df[df["pct_retained_95th"] < 50]
    if len(flagged):
        print(flagged.to_string(index=False))
    else:
        print("None - all classes/patch-types retain >=50% at the 95th-percentile threshold.")

    print("\n=== Recommendation ===")
    overall_pct_95 = df["keep_at_95th"].sum() / df["n_total"].sum() * 100
    print(f"Overall retention at 95th-percentile threshold: {overall_pct_95:.1f}%")
    if len(flagged) == 0 and overall_pct_95 > 50:
        print("Retention looks reasonable - majority of patches survive per class. "
              "Feasible to proceed to Phase 2b if desired.")
    else:
        print("Retention does NOT clear the bar - do not proceed to Phase 2b on this threshold "
              "without reconsidering the exclusion criteria.")


if __name__ == "__main__":
    main()
