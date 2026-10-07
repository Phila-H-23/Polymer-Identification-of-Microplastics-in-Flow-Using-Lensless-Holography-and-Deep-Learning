"""
1b. Item-P evaluation from saved best.pt (no training) on (i) every *_cwvh
background-only checkpoint (R and R2) and (ii) the stokes_bgcorr_vnorm real runs.
The real runs of all other arms are re-evaluated too, from best.pt, so the PET and
per-class F1 gap tables (real minus new background-only) come from one code path.
Uses evaluate() from p_eval_checkpoints.py unchanged (input-channel counts added
for the modes it did not cover). Writes results/p_report_cwvh.txt and
results/p_report_cwvh.csv (mean and standard deviation over seeds).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p_eval_checkpoints as p

p.IN_CH.update({"pol4": 4, "pol4_ff": 4, "i0_bgcorr_vnorm": 1, "i0i90_bgcorr_vnorm": 2, "stokes_bgcorr_vnorm": 3})
ARMS = [("intensity_bgcorr_vnorm", "intbv_c849"), ("z0", "recons0z0_c849"), ("recon_s0", "recons0_r180"),
        ("recon_biref", "reconbiref_c849"), ("pol4_bgcorr_vnorm", "pol4bgcorrvnorm_c849"), ("I45", "i45bv_c849"),
        ("Stokes", "stokes_c849"), ("stokes_bgcorr_vnorm", "stokesbv_c849"), ("pol4_bgcorr", "pol4bgcorr_c849"),
        ("pol4", "pol4_c849"), ("pol4_ff", "pol4ff_c849"), ("I0", "i0bv_c849"), ("I0+I90", "i0i90bv_c849")]
FULL_BLOCKS_REAL = {"stokes_bgcorr_vnorm"}  # (ii): real runs never evaluated before


def run_side(prefix):
    out = [p.evaluate(f"results/{prefix}_s{s}") for s in range(5)]
    classes = out[0][0]
    cms = np.array([o[1] for o in out], float)
    P, R, F = (np.array([o[k] for o in out]) for k in (2, 3, 4))
    mism = sum(not o[5] for o in out)
    return classes, cms, P, R, F, mism


def main():
    lines, rows, res, mism = [], [], {}, 0
    for name, tag in ARMS:
        for side, prefix in (("real", f"real_vh_{tag}"), ("bg_cwvh", f"bg_{tag}_cwvh")):
            classes, cms, P, R, F, mm = run_side(prefix)
            mism += mm
            res[(name, side)] = (classes, cms, P, R, F)
            for i, c in enumerate(classes):
                for met, arr in (("precision", P), ("recall", R), ("f1", F)):
                    rows.append(dict(arm=name, side=side, run_prefix=prefix, kind="metric", true=c, item=met,
                                     mean=arr[:, i].mean(), std=arr[:, i].std()))
                for j, cp in enumerate(classes):
                    rows.append(dict(arm=name, side=side, run_prefix=prefix, kind="confusion", true=c, item=cp,
                                     mean=cms[:, i, j].mean(), std=cms[:, i, j].std()))
            print(f"done {prefix}", flush=True)
    pd.DataFrame(rows).to_csv("results/p_report_cwvh.csv", index=False)

    classes = res[(ARMS[0][0], "real")][0]
    pet = classes.index("PET")
    lines.append(f"Recomputed confusion matrices differing from results.json: {mism} of {len(ARMS) * 10}")
    lines.append("Real = original runs (unchanged). bg_cwvh = background-only re-run with the video-holdout weights.")
    lines.append("\nPET precision / recall / F1 (mean +/- std over 5 seeds)")
    lines.append(f"{'arm':22s} {'real P':>13s} {'real R':>13s} {'real F1':>13s}   {'bg P':>13s} {'bg R':>13s} {'bg F1':>13s}")
    for name, _ in ARMS:
        _, _, rP, rR, rF = res[(name, "real")]
        _, _, bP, bR, bF = res[(name, "bg_cwvh")]
        f = lambda a: f"{a[:, pet].mean():.3f}+/-{a[:, pet].std():.3f}"
        lines.append(f"{name:22s} {f(rP):>13s} {f(rR):>13s} {f(rF):>13s}   {f(bP):>13s} {f(bR):>13s} {f(bF):>13s}")

    lines.append("\nPer-class F1 gap: real minus new background-only (means over 5 seeds)")
    lines.append(f"{'arm':22s} " + " ".join(f"{c:>7s}" for c in classes) + f" {'macro':>7s}")
    for name, _ in ARMS:
        rF, bF = res[(name, "real")][4], res[(name, "bg_cwvh")][4]
        g = rF.mean(0) - bF.mean(0)
        lines.append(f"{name:22s} " + " ".join(f"{v:>+7.3f}" for v in g) + f" {rF.mean(1).mean() - bF.mean(1).mean():>+7.3f}")

    def block(name, side, prefix):
        classes, cms, P, R, F = res[(name, side)]
        cm = cms.mean(0)
        nonpet = [i for i in range(len(classes)) if i != pet]
        frac = np.mean([c[nonpet, pet].sum() / c[nonpet].sum() for c in cms])
        lines.append(f"\n{name} - {side} ({prefix}, 5 seeds, {int(cms[0].sum())} test patches); "
                     f"non-PET predicted PET {frac:.3f}")
        lines.append(f"  {'class':5s} {'precision':>9s} {'recall':>7s} {'F1':>6s}   confusion matrix, mean counts "
                     f"(rows true, cols predicted: {' '.join(classes)})")
        for i, c in enumerate(classes):
            lines.append(f"  {c:5s} {P.mean(0)[i]:9.3f} {R.mean(0)[i]:7.3f} {F.mean(0)[i]:6.3f}   "
                         + " ".join(f"{v:6.1f}" for v in cm[i]))

    lines.append("\n(i) all *_cwvh background-only checkpoints; (ii) stokes_bgcorr_vnorm real runs")
    for name, tag in ARMS:
        if name in FULL_BLOCKS_REAL:
            block(name, "real", f"real_vh_{tag}")
        block(name, "bg_cwvh", f"bg_{tag}_cwvh")
    text = "\n".join(lines)
    open("results/p_report_cwvh.txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
