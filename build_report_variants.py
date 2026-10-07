"""Real vs background-only, compact CNN, video-holdout, for any variant below.
Same gap / noise-floor rule and per-class table as build_report_rung1.py.
Works on partial results: uses whichever seeds have finished and prints n.
Usage: python build_report_variants.py                 (all queue v2 variants)
       python build_report_variants.py i0 i0_orig       (selected)"""
import sys
import json
import statistics as stats
from pathlib import Path

CLASSES = ["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"]

# Current queue (queue v2): corrected 0/180 deg augmentation, common 849 patches.
QUEUE = {
    "recon_s0": "recons0_r180", "intensity_bgcorr_vnorm": "intbv_c849",
    "pol4_bgcorr_vnorm": "pol4bgcorrvnorm_c849", "i0": "i0bv_c849", "i45": "i45bv_c849",
    "recon_s0_z0": "recons0z0_c849", "recon_biref": "reconbiref_c849",
    "pol4": "pol4_c849", "i0i90": "i0i90bv_c849", "pol4_ff": "pol4ff_c849",
    "pol4_bgcorr": "pol4bgcorr_c849", "stokes": "stokes_c849",
}
VARIANTS = {k: (f"real_vh_{v}", f"bg_{v}") for k, v in QUEUE.items()}
VARIANTS.update({  # queue v3 (D)
    "stokes_bgcorr_vnorm": ("real_vh_stokesbv_c849", "bg_stokesbv_c849"),
})
VARIANTS.update({  # superseded: corrected augmentation but full 883 patches (stopped mid-group)
    "intensity_bgcorr_vnorm_883": ("real_vh_intbv_r180", "bg_intbv_r180"),
})
VARIANTS.update({  # earlier runs (pre-fix augmentation), for side-by-side
    "i0_orig": ("real_vh_i0bv", "bg_i0bv"), "i45_orig": ("real_vh_i45bv", "bg_i45bv"),
    "i0i90_orig": ("real_vh_i0i90bv", "bg_i0i90bv"), "intensity_bgcorr_vnorm_orig": ("real_vh_intbv", "bg_intbv"),
    "pol4_bgcorr_vnorm_orig": ("real_vh_pol4bgcorrvnorm", "bg_pol4bgcorrvnorm"),
})


def load_all(prefix):
    out = []
    for s in range(5):
        p = Path(f"results/{prefix}_s{s}/results.json")
        if p.exists():
            out.append(json.load(open(p)))
    return out


def side_stats(runs):
    f1 = [r["test_report"]["macro avg"]["f1-score"] for r in runs]
    rec = {c: [r["test_report"].get(c, {}).get("recall", float("nan")) for r in runs] for c in CLASSES}
    sd = lambda v: stats.pstdev(v) if len(v) > 1 else float("nan")
    return dict(n=len(runs), f1=f1, mean=stats.mean(f1) if f1 else float("nan"), std=sd(f1),
                rec_mean={c: stats.mean(v) for c, v in rec.items()} if runs else {},
                rec_std={c: sd(v) for c, v in rec.items()} if runs else {})


def report(name, real_prefix, bg_prefix):
    real, bg = side_stats(load_all(real_prefix)), side_stats(load_all(bg_prefix))
    print(f"\n{'=' * 78}\n{name}   (real n={real['n']}/5, background-only n={bg['n']}/5)\n{'=' * 78}")
    if real["n"] == 0 or bg["n"] == 0:
        print("no completed runs on one side yet")
        return None
    for label, g in (("real", real), ("background-only", bg)):
        print(f"{label:16s} macro-F1 mean={g['mean']:.4f} std={g['std']:.4f} {[round(v, 4) for v in g['f1']]}")
    gap, floor = real["mean"] - bg["mean"], max(real["std"], bg["std"])
    exceeds = abs(gap) > floor
    partial = " [PARTIAL - fewer than 5 seeds]" if min(real["n"], bg["n"]) < 5 else ""
    print(f"gap (real - background-only) = {gap:+.4f}; noise floor (larger std) = {floor:.4f} -> "
          f"{'EXCEEDS floor' if exceeds else 'within floor'}{partial}")
    print(f"\n{'class':6s} {'real':>15s} {'background':>15s} {'gap':>8s} {'floor':>7s}  clears?")
    clears = []
    for c in CLASSES:
        rm, rs, bm, bs = real["rec_mean"][c], real["rec_std"][c], bg["rec_mean"][c], bg["rec_std"][c]
        g, f = rm - bm, max(rs, bs)
        ok = g > f
        clears += [c] if ok else []
        print(f"{c:6s} {rm:.3f} +/- {rs:.3f} {bm:.3f} +/- {bs:.3f} {g:>+8.3f} {f:>7.3f}  {'YES' if ok else 'no'}")
    print(f"classes where real > background-only beyond the floor: {clears or 'none'}{partial}")
    return dict(gap=gap, floor=floor, exceeds=exceeds, clears=clears, n=(real["n"], bg["n"]))


def main():
    names = sys.argv[1:] or list(QUEUE)
    for n in names:
        report(n, *VARIANTS[n])


if __name__ == "__main__":
    main()
