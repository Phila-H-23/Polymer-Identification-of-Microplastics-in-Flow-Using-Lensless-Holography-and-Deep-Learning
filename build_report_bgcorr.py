"""Report for the background-subtraction (pol4_bgcorr) re-test: does the
90-frame rolling-background correction (Montandon & Nicolls method, as
supplied by the user) open up the real-vs-background-only gap, or
suppress background-only's absolute score, compared to uncorrected pol4
and flat-fielded pol4_ff?
"""
import json
import statistics as stats
from pathlib import Path

CLASSES = ["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"]


def load(out_dir):
    p = Path(out_dir) / "results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["test_report"]["macro avg"]["f1-score"]


def group_stats(out_dirs):
    vals = [load(d) for d in out_dirs]
    missing = [d for d, v in zip(out_dirs, vals) if v is None]
    vals = [v for v in vals if v is not None]
    mean_f1 = stats.mean(vals) if vals else float("nan")
    std_f1 = stats.pstdev(vals) if len(vals) > 1 else 0.0
    return dict(n=len(vals), missing=missing, mean_f1=mean_f1, std_f1=std_f1, f1s=vals)


def main():
    real = group_stats([f"results/real_vh_pol4bgcorr_s{s}" for s in range(5)])
    bg = group_stats([f"results/bg_pol4bgcorr_s{s}" for s in range(5)])

    print("=== pol4_bgcorr (background-subtracted, video_holdout, n=5) ===")
    for name, g in [("real", real), ("background-only", bg)]:
        if g["missing"]:
            print(f"{name:20s} INCOMPLETE - missing: {g['missing']}")
        else:
            print(f"{name:20s} n={g['n']} mean={g['mean_f1']:.4f} std={g['std_f1']:.4f}  {[round(v,4) for v in g['f1s']]}")

    if real["missing"] or bg["missing"]:
        return

    gap = real["mean_f1"] - bg["mean_f1"]
    print(f"\ngap (real - background-only) = {gap:+.4f}")

    print("\n=== Comparison across all three preprocessing variants ===")
    rows = [
        ("uncorrected pol4", 0.4179, 0.3589),
        ("flat-fielded pol4_ff", 0.5348, 0.5303),
        ("background-subtracted pol4_bgcorr", real["mean_f1"], bg["mean_f1"]),
    ]
    print(f"{'variant':35s} {'real':>8s} {'background':>11s} {'gap':>8s}")
    for name, r, b in rows:
        print(f"{name:35s} {r:>8.4f} {b:>11.4f} {r-b:>+8.4f}")

    print()
    uncorrected_gap = 0.4179 - 0.3589
    print(f"Gap vs uncorrected pol4 (+{uncorrected_gap:.4f}): "
          f"{'OPENS UP' if gap > uncorrected_gap + 0.01 else ('SHRINKS' if gap < uncorrected_gap - 0.01 else 'ABOUT THE SAME')} "
          f"({uncorrected_gap:+.4f} -> {gap:+.4f})")

    bg_uncorrected = 0.3589
    bg_ff = 0.5303
    print(f"Background-only absolute score: bgcorr={bg['mean_f1']:.4f} vs uncorrected={bg_uncorrected:.4f} "
          f"({'LOWER (suppressed)' if bg['mean_f1'] < bg_uncorrected - 0.01 else ('HIGHER' if bg['mean_f1'] > bg_uncorrected + 0.01 else 'ABOUT THE SAME')}), "
          f"vs flat-fielded={bg_ff:.4f} ({'LOWER' if bg['mean_f1'] < bg_ff - 0.01 else ('HIGHER' if bg['mean_f1'] > bg_ff + 0.01 else 'ABOUT THE SAME')})")


if __name__ == "__main__":
    main()
