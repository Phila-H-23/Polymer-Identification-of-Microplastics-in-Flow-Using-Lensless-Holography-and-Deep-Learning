"""Final raw-input comparison table: uncorrected pol4/intensity and
flat-fielded pol4_ff/intensity_ff, each real vs background-only, all on
split=video_holdout, n=5. Flags any comparison where the real/background
gap exceeds the larger of the two groups' std (a rough "is this gap
outside noise" check, not a formal significance test).
"""
import json
import statistics as stats
from pathlib import Path

CLASSES = ["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"]
CHANCE = 1 / len(CLASSES)


def load(out_dir):
    p = Path(out_dir) / "results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    tr = d["test_report"]
    return {"test_macro_f1": tr["macro avg"]["f1-score"]}


def group_stats(out_dirs):
    runs = [load(d) for d in out_dirs]
    missing = [d for d, r in zip(out_dirs, runs) if r is None]
    runs = [r for r in runs if r is not None]
    f1s = [r["test_macro_f1"] for r in runs]
    mean_f1 = stats.mean(f1s) if f1s else float("nan")
    std_f1 = stats.pstdev(f1s) if len(f1s) > 1 else 0.0
    return dict(n=len(runs), missing=missing, mean_f1=mean_f1, std_f1=std_f1, f1s=f1s)


def main():
    pairs = [
        ("pol4 (uncorrected)", [f"results/real_vh_pol4_s{s}" for s in range(5)],
         [f"results/bg_pol4_s{s}" for s in range(5)]),
        ("intensity (uncorrected)", [f"results/real_vh_intensity_s{s}" for s in range(5)],
         [f"results/bg_intensity_s{s}" for s in range(5)]),
        ("pol4_ff (flat-fielded)", [f"results/real_vh_pol4ff_s{s}" for s in range(5)],
         [f"results/bg_pol4ff_s{s}" for s in range(5)]),
        ("intensity_ff (flat-fielded)", [f"results/real_vh_intensityff_s{s}" for s in range(5)],
         [f"results/bg_intensityff_s{s}" for s in range(5)]),
    ]

    print("=== Final raw-input comparison table (split=video_holdout, n=5) ===")
    print(f"{'channels':30s} {'side':16s} {'n':3s} {'mean_f1':>9s} {'std':>7s}  values")
    summary = []
    for name, real_dirs, bg_dirs in pairs:
        real_g = group_stats(real_dirs)
        bg_g = group_stats(bg_dirs)
        for side, g in [("real", real_g), ("background-only", bg_g)]:
            if g["missing"]:
                print(f"{name:30s} {side:16s}  INCOMPLETE - missing: {g['missing']}")
            else:
                print(f"{name:30s} {side:16s} {g['n']:<3d} {g['mean_f1']:>9.4f} {g['std_f1']:>7.4f}  {[round(v,4) for v in g['f1s']]}")
        summary.append((name, real_g, bg_g))
        print()

    print("=== Gap analysis: real - background-only, vs larger std ===")
    for name, real_g, bg_g in summary:
        if real_g["missing"] or bg_g["missing"]:
            print(f"{name:30s}  INCOMPLETE")
            continue
        gap = real_g["mean_f1"] - bg_g["mean_f1"]
        larger_std = max(real_g["std_f1"], bg_g["std_f1"])
        exceeds = abs(gap) > larger_std
        flag = "EXCEEDS larger std -> gap likely real" if exceeds else "within larger std -> gap NOT distinguishable from noise"
        print(f"{name:30s} gap={gap:+.4f}  larger_std={larger_std:.4f}  [{flag}]")


if __name__ == "__main__":
    main()
