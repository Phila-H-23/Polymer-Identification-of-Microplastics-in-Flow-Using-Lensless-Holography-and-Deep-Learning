"""Report for the per-video brightness-normalisation diagnostic
(--channels pol4_bgcorr_vnorm): does removing each video's background-
derived mean/std collapse the pol4_bgcorr confound, or does dust/ring
texture alone (independent of brightness) still separate classes?
"""
import json
import statistics as stats
from pathlib import Path

CHANCE = 1 / 7


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
    real = group_stats([f"results/real_vh_pol4bgcorrvnorm_s{s}" for s in range(5)])
    bg = group_stats([f"results/bg_pol4bgcorrvnorm_s{s}" for s in range(5)])

    print("=== pol4_bgcorr_vnorm (per-video brightness-normalised, video_holdout, n=5) ===")
    for name, g in [("real", real), ("background-only", bg)]:
        if g["missing"]:
            print(f"{name:20s} INCOMPLETE - missing: {g['missing']}")
        else:
            print(f"{name:20s} n={g['n']} mean={g['mean_f1']:.4f} std={g['std_f1']:.4f}  {[round(v,4) for v in g['f1s']]}")

    if real["missing"] or bg["missing"]:
        return

    print(f"\nChance level: {CHANCE:.4f}")
    print(f"real vs chance: {real['mean_f1']/CHANCE:.2f}x")
    print(f"background-only vs chance: {bg['mean_f1']/CHANCE:.2f}x")

    print("\n=== Comparison vs pol4_bgcorr (no vnorm) ===")
    print(f"{'variant':30s} {'real':>8s} {'background':>11s}")
    print(f"{'pol4_bgcorr':30s} {0.6066:>8.4f} {0.6192:>11.4f}")
    print(f"{'pol4_bgcorr_vnorm':30s} {real['mean_f1']:>8.4f} {bg['mean_f1']:>11.4f}")

    gap = real["mean_f1"] - bg["mean_f1"]
    larger_std = max(real["std_f1"], bg["std_f1"])
    print(f"\ngap (real - background-only) = {gap:+.4f}, larger std = {larger_std:.4f}  "
          f"[{'EXCEEDS larger std -> gap likely real' if abs(gap) > larger_std else 'within larger std -> not distinguishable from noise'}]")

    print("\n=== Verdict ===")
    real_near_chance = abs(real["mean_f1"] - CHANCE) < 0.05
    bg_near_chance = abs(bg["mean_f1"] - CHANCE) < 0.05
    if real_near_chance and bg_near_chance:
        print("(a) BOTH collapse toward chance - the brightness offset was the whole story, "
              "nothing else in this input is separable.")
    elif bg_near_chance and not real_near_chance:
        print("(b) Background collapses but real does NOT - the brightness offset was masking "
              "a real particle signal that is now visible.")
    elif not bg_near_chance:
        print("(c) Background stays well above chance - dust/ring texture alone, independent "
              "of brightness, is enough on its own to separate classes.")
    else:
        print("Ambiguous - inspect the numbers directly.")


if __name__ == "__main__":
    main()
