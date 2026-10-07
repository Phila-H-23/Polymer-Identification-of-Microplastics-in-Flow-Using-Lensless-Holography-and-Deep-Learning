"""Surround augmentation report -> results/report_aug.txt (layout of report_R2_reweighted_bg.txt).
Per configuration (BG, BG+TEX, INTERIOR) and arm (z0, intensity_bgcorr_vnorm): macro-F1 real vs
particle-free, gap, floor (larger seed std), above/within; per-class recall table; PET P/R/F1.
Summary: no-augmentation numbers parsed from results/report_R_reweighted_bg.txt (real = original
runs, particle-free = matched *_cwvh runs), then BG, BG+TEX, INTERIOR, and the chance macro-F1.
Readings (fixed before the runs):
  within 0.05 of chance = |particle-free - chance| <= 0.05
  real not more than its seed std below no-aug = real_aug >= real_noaug - std(real_aug)
  R1: within chance AND real not below AND gap > floor
  R2: within chance AND (real below OR gap <= floor)
  R3: particle-free > chance + 0.05"""
import os
import re
import json
import statistics as stats
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLASSES = ["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"]
CONFIGS = [("BG", "bg"), ("BG+TEX", "bgtex"), ("INTERIOR", "interior")]
ARMS = [("z0", "recons0z0_c849"), ("intensity_bgcorr_vnorm", "intbv_c849")]
CW = "pipeline_output/class_weights_video_holdout.json"
HEADER = """SURROUND AUGMENTATION (new runs: results/real_vh_<tag>_s<seed>_<bg|bgtex|interior> and bg_<tag>_s<seed>_<...>)
Regime copied from the corrected queue (scripts/run_queue_r180.sh -> run_real_bg_seeds.sh): video-holdout split,
  common 849 patches, epochs 150, patience 30 (best-validation checkpoint), batch 32, AdamW lr 1e-3, weight
  decay 1e-4, cosine annealing, width 32, dropout 0.3, rotations 0/180, translation +/-4 px, gain +/-5%,
  noise 0.01; 5 seeds; class weights pipeline_output/class_weights_video_holdout.json for both classifiers.
BG and BG+TEX are augmentations of training patches only (validation and test never augmented; applied before
the rotation). INTERIOR is an input representation applied to training, validation and test patches of both
classifiers, with the per-channel normalisation computed on the masked training patches.
  BG: keep the disc of radius 12 px at the particle position (detection centre; actual position for clipped
      patches), edge feathered with a Gaussian of sigma 2 px; surround from a random training-split background
      patch of a different video, any class, same domain, redrawn each time. Background patches: own central
      disc, surround from a donor.  TEX: after BG, each with p=0.5: Gaussian blur sigma U(0,1.5) px; Gaussian
      noise std U(0,0.1) x patch std.  INTERIOR: same disc, surround = mean of all training background
      patches of the arm's domain (no donors), on every split.
"""


def load(prefix, suffix):
    runs, missing = [], []
    for s in range(5):
        d = f"results/{prefix}_s{s}_{suffix}"
        if os.path.exists(f"{d}/results.json"):
            runs.append(json.load(open(f"{d}/results.json")))
        elif os.path.isdir(d):
            missing.append(d)
    return runs, missing


def ms(v):
    return (stats.mean(v), stats.pstdev(v)) if len(v) > 1 else ((v[0], float("nan")) if v else (float("nan"),) * 2)


def side(runs):
    g = lambda c, k: ms([r["test_report"][c][k] for r in runs])
    return dict(n=len(runs), rec={c: g(c, "recall") for c in CLASSES},
                pet={k: g("PET", k) for k in ("precision", "recall", "f1-score")},
                f1=ms([r["test_report"]["macro avg"]["f1-score"] for r in runs]),
                support={c: int(runs[0]["test_report"][c]["support"]) for c in CLASSES})


def no_aug():
    text = open("results/report_R_reweighted_bg.txt").read()
    out = {}
    for m in re.finditer(r"^(\S+)\s+\(real n=.*?\n=+\nmacro-F1: real ([\d.]+)\+/-([\d.]+) \| old bg .*? \| "
                         r"new bg ([\d.]+)\+/-([\d.]+)", text, re.M):
        out[m.group(1)] = ((float(m.group(2)), float(m.group(3))), (float(m.group(4)), float(m.group(5))))
    return out


def chance():
    r = pd.read_csv("pipeline_output/manifest_bgcorrected_common849.csv")
    tr, te = r[r.split_video_holdout == "train"]["class"], r[r.split_video_holdout == "test"]["class"].values
    p = tr.value_counts(normalize=True).reindex(CLASSES).values
    rng = np.random.default_rng(0)
    s = [f1_score(te, rng.choice(CLASSES, size=len(te), p=p), labels=CLASSES, average="macro", zero_division=0)
         for _ in range(1000)]
    return float(np.mean(s)), float(np.std(s))


def main():
    f = lambda t: f"{t[0]:.4f}+/-{t[1]:.4f}"
    ch, ch_sd = chance()
    base = no_aug()
    lines, res, failures, cw_bad, n_runs = [HEADER], {}, [], [], 0
    for cname, suffix in CONFIGS:
        for arm, tag in ARMS:
            rr, rm = load(f"real_vh_{tag}", suffix)
            br, bm = load(f"bg_{tag}", suffix)
            failures += rm + bm
            n_runs += len(rr) + len(br)
            cw_bad += [r["args"]["out"] for r in rr + br
                       if r["args"].get("class_weights") != CW or r["args"].get("aug") != suffix]
            if not rr or not br:
                lines.append(f"\n{arm} + {cname}: not run yet (real n={len(rr)}, particle-free n={len(br)})")
                continue
            real, pf = side(rr), side(br)
            gap, floor = real["f1"][0] - pf["f1"][0], max(real["f1"][1], pf["f1"][1])
            res[(arm, cname)] = dict(real=real["f1"], pf=pf["f1"], gap=gap, floor=floor, n=(real["n"], pf["n"]))
            lines += ["", "=" * 92, f"{arm} + {cname}  (real n={real['n']}, particle-free n={pf['n']})", "=" * 92,
                      f"macro-F1: real {f(real['f1'])} | particle-free {f(pf['f1'])} | gap {gap:+.4f} "
                      f"floor {floor:.4f} {'ABOVE' if gap > floor else 'within'}",
                      "PET precision / recall / F1: real "
                      + " / ".join(f(real['pet'][k]) for k in ('precision', 'recall', 'f1-score'))
                      + " | particle-free " + " / ".join(f(pf['pet'][k]) for k in ('precision', 'recall', 'f1-score')),
                      f"{'class':6s} {'n test':>6s} {'real':>15s} {'particle-free':>15s} {'gap':>8s} {'floor':>6s} clears?"]
            clears = []
            for c in CLASSES:
                r, p = real["rec"][c], pf["rec"][c]
                g, fl = r[0] - p[0], max(r[1], p[1])
                clears += [c] if g > fl else []
                lines.append(f"{c:6s} {real['support'][c]:>6d} {r[0]:.3f}+/-{r[1]:.3f} {p[0]:.3f}+/-{p[1]:.3f} "
                             f"{g:>+8.3f} {fl:>6.3f}  {'YES' if g > fl else 'no'}")
            lines.append(f"classes clearing the floor: {clears or 'none'}")

    lines += ["", "=" * 92, "SUMMARY: macro-F1 (mean +/- std over 5 seeds)", "=" * 92,
              f"{'arm':24s} {'configuration':14s} {'real':>15s} {'particle-free':>15s} {'gap':>8s} {'floor':>7s}",
              f"{'(reference)':24s} {'chance':14s} {'':>15s} {f((ch, ch_sd)):>15s}"]
    for arm, _ in ARMS:
        (br_, bp_) = base[arm]
        g0, fl0 = br_[0] - bp_[0], max(br_[1], bp_[1])
        lines.append(f"{arm:24s} {'no aug (R)':14s} {f(br_):>15s} {f(bp_):>15s} {g0:>+8.4f} {fl0:>7.4f}")
        for cname, _ in CONFIGS:
            s = res.get((arm, cname))
            lines.append(f"{arm:24s} {cname:14s} " + (f"{f(s['real']):>15s} {f(s['pf']):>15s} {s['gap']:>+8.4f} "
                                                      f"{s['floor']:>7.4f}" if s else "not run yet"))

    lines += ["", f"Pre-registered readings (chance macro-F1 {ch:.4f}):"]
    for cname, _ in CONFIGS:
        for arm, _ in ARMS:
            s = res.get((arm, cname))
            if not s or s["n"] != (5, 5):
                lines.append(f"{arm} + {cname}: not evaluated (runs incomplete)")
                continue
            near = abs(s["pf"][0] - ch) <= 0.05
            real_ok = s["real"][0] >= base[arm][0][0] - s["real"][1]
            above = s["gap"] > s["floor"]
            if near and real_ok and above:
                rd = "R1  <<< FLAG: the augmentation works; reverse-fold test would be next >>>"
            elif near:
                rd = "R2"
            elif s["pf"][0] > ch + 0.05:
                rd = "R3"
            else:
                rd = "none of R1-R3 (particle-free more than 0.05 BELOW chance)"
            lines.append(f"{arm} + {cname}: {rd}  [particle-free {s['pf'][0]:.4f} vs chance {ch:.4f}; real "
                         f"{s['real'][0]:.4f} vs no-aug {base[arm][0][0]:.4f} (std {s['real'][1]:.4f}); gap "
                         f"{s['gap']:+.4f} vs floor {s['floor']:.4f}]")

    lines += ["", f"runs finished: {n_runs} of 60",
              f"class-weights file {CW} and matching --aug recorded in every finished run: "
              f"{'yes' if not cw_bad else 'NO: ' + str(cw_bad)}",
              f"failed / incomplete runs: {failures or 'none'}"]
    text = "\n".join(lines)
    open("results/report_aug.txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
