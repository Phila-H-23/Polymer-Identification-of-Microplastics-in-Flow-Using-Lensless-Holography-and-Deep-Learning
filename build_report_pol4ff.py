"""Report for the flat-field (pol4_ff) re-test: does real pol4 still beat
its own background-only control once the fixed I0/I90 imbalance is
removed (scripts/build_flatfield_ref.py, --channels pol4_ff)?

Compares directly against the uncorrected pol4 numbers from Table A in
build_report_final.py (results/real_vh_pol4_s0-4, results/bg_pol4_s0-4),
both already on split=video_holdout, n=5 - the same comparison, just with
the flat-field-corrected input instead of raw pol4.
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
    return {
        "test_macro_f1": tr["macro avg"]["f1-score"],
        "test_accuracy": tr["accuracy"],
        "recall": {c: tr.get(c, {}).get("recall", float("nan")) for c in CLASSES},
    }


def group_stats(out_dirs):
    runs = [load(d) for d in out_dirs]
    missing = [d for d, r in zip(out_dirs, runs) if r is None]
    runs = [r for r in runs if r is not None]
    f1s = [r["test_macro_f1"] for r in runs]
    mean_f1 = stats.mean(f1s) if f1s else float("nan")
    std_f1 = stats.pstdev(f1s) if len(f1s) > 1 else 0.0
    return dict(n=len(runs), missing=missing, mean_f1=mean_f1, std_f1=std_f1, f1s=f1s)


def main():
    rows = [
        ("real pol4 (uncorrected, video_holdout)", [f"results/real_vh_pol4_s{s}" for s in range(5)]),
        ("background-only pol4 (uncorrected, video_holdout)", [f"results/bg_pol4_s{s}" for s in range(5)]),
        ("real pol4_ff (flat-fielded, video_holdout)", [f"results/real_vh_pol4ff_s{s}" for s in range(5)]),
        ("background-only pol4_ff (flat-fielded, video_holdout)", [f"results/bg_pol4ff_s{s}" for s in range(5)]),
    ]

    print("=== Flat-field re-test: real pol4_ff vs background-only pol4_ff (video_holdout, n=5) ===")
    print(f"{'row':52s} {'n':3s} {'mean_test_f1':>13s} {'std':>7s}  values")
    results = {}
    for name, dirs in rows:
        g = group_stats(dirs)
        results[name] = g
        if g["missing"]:
            print(f"{name:52s}  INCOMPLETE - missing: {g['missing']}")
        else:
            print(f"{name:52s} {g['n']:<3d} {g['mean_f1']:>13.4f} {g['std_f1']:>7.4f}  {[round(v,4) for v in g['f1s']]}")

    print()
    if not any(results[n]["missing"] for n, _ in rows):
        unc_real = results["real pol4 (uncorrected, video_holdout)"]["mean_f1"]
        unc_bg = results["background-only pol4 (uncorrected, video_holdout)"]["mean_f1"]
        ff_real = results["real pol4_ff (flat-fielded, video_holdout)"]["mean_f1"]
        ff_bg = results["background-only pol4_ff (flat-fielded, video_holdout)"]["mean_f1"]

        gap_before = unc_real - unc_bg
        gap_after = ff_real - ff_bg

        print(f"Uncorrected pol4: real={unc_real:.4f} background-only={unc_bg:.4f} gap={gap_before:+.4f}")
        print(f"Flat-fielded pol4_ff: real={ff_real:.4f} background-only={ff_bg:.4f} gap={gap_after:+.4f}")
        print()
        if gap_after > gap_before + 0.01:
            verdict = "GROWS"
        elif gap_after < gap_before - 0.01:
            verdict = "SHRINKS"
        else:
            verdict = "STAYS ABOUT THE SAME"
        print(f"Gap between real and background-only {verdict} after flat-fielding "
              f"({gap_before:+.4f} -> {gap_after:+.4f}, delta={gap_after-gap_before:+.4f})")
        print(f"Does real pol4_ff beat its own background-only control? "
              f"{'YES' if ff_real > ff_bg else 'NO'} (by {ff_real-ff_bg:+.4f})")


if __name__ == "__main__":
    main()
