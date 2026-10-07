"""Phase 2b report: pol4_bgcorr_vnorm_texfilt (95th-percentile texture
exclusion applied uniformly to all 7 classes, real + background-only),
compact CNN, split_video_holdout, seeds 0-4.
"""
import json
import statistics as stats
from pathlib import Path
import pandas as pd

CLASSES = ["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"]


def load(out_dir):
    p = Path(out_dir) / "results.json"
    if not p.exists():
        return None
    return json.load(open(p))


def group_stats(out_dirs):
    ds = [load(d) for d in out_dirs]
    missing = [d for d, x in zip(out_dirs, ds) if x is None]
    ds = [x for x in ds if x is not None]
    f1s = [d["test_report"]["macro avg"]["f1-score"] for d in ds]
    mean_f1 = stats.mean(f1s) if f1s else float("nan")
    std_f1 = stats.pstdev(f1s) if len(f1s) > 1 else 0.0
    recalls = {c: [d["test_report"].get(c, {}).get("recall", float("nan")) for d in ds] for c in CLASSES}
    mean_recalls = {c: stats.mean(recalls[c]) for c in CLASSES}
    return dict(n=len(ds), missing=missing, mean_f1=mean_f1, std_f1=std_f1, f1s=f1s, mean_recalls=mean_recalls)


def main():
    real = group_stats([f"results/real_vh_texfilt_s{s}" for s in range(5)])
    bg = group_stats([f"results/bg_texfilt_s{s}" for s in range(5)])

    print("=== pol4_bgcorr_vnorm_texfilt (95th pct texture exclusion, video_holdout, n=5) ===")
    for name, g in [("real", real), ("background-only", bg)]:
        if g["missing"]:
            print(f"{name:20s} INCOMPLETE - missing: {g['missing']}")
        else:
            print(f"{name:20s} n={g['n']} mean={g['mean_f1']:.4f} std={g['std_f1']:.4f}  {[round(v,4) for v in g['f1s']]}")

    if real["missing"] or bg["missing"]:
        return

    gap = real["mean_f1"] - bg["mean_f1"]
    larger_std = max(real["std_f1"], bg["std_f1"])
    print(f"\ngap (real - background-only) = {gap:+.4f}, larger std = {larger_std:.4f}  "
          f"[{'EXCEEDS larger std -> gap likely real' if abs(gap) > larger_std else 'within larger std -> not distinguishable from noise'}]")

    print("\n=== Comparison vs pol4_bgcorr_vnorm (unfiltered) ===")
    print(f"{'variant':30s} {'real':>8s} {'background':>11s}")
    print(f"{'pol4_bgcorr_vnorm':30s} {0.3129:>8.4f} {0.3017:>11.4f}")
    print(f"{'pol4_bgcorr_vnorm_texfilt':30s} {real['mean_f1']:>8.4f} {bg['mean_f1']:>11.4f}")

    print("\n=== Per-class recall, real vs background-only (mean of 5 seeds) ===")
    print(f"{'class':6s} {'real':>8s} {'background':>11s} {'gap':>8s}")
    for c in CLASSES:
        r, b = real["mean_recalls"][c], bg["mean_recalls"][c]
        print(f"{c:6s} {r:>8.4f} {b:>11.4f} {r-b:>+8.4f}")

    # sample counts used
    print("\n=== Per-class/patch-type sample counts after filtering (train/val/test) ===")
    real_m = pd.read_csv("pipeline_output/manifest_bgcorrected_texfilt.csv")
    bg_m = pd.read_csv("pipeline_output/manifest_background_bgcorrected_texfilt.csv")
    real_counts = real_m.groupby(["class", "split_video_holdout"]).size().unstack(fill_value=0)
    bg_counts = bg_m.groupby(["class", "split"]).size().unstack(fill_value=0)
    print("\nREAL (split_video_holdout):")
    print(real_counts.to_string())
    print("\nBACKGROUND-ONLY (split):")
    print(bg_counts.to_string())


if __name__ == "__main__":
    main()
