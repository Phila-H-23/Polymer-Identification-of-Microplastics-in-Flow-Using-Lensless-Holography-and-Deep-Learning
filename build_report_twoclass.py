"""Two-class test report -> results/report_twoclass.txt (refreshed after every pair)."""
import os
import json
import statistics as stats
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PO = "pipeline_output"
PAIRS = ["PS", "PE", "PMMA", "PVC", "PP", "PVA"]
ARMS = ["intensity_bgcorr_vnorm", "z0"]
MARGIN = 0.10
HEADER = """TWO-CLASS TEST: PET versus one other polymer, real and particle-free classifiers
Split: the stored video-holdout assignment (column split_video_holdout of the common-849 manifests; particle-free
  patches carry the same assignment in 'split'), restricted to the two classes. Its validation rule: about 17.5%
  of each class's patches from that class's training video(s), one patch per particle (no particle in two splits);
  PET trains on PET1, PET3, PET4 and is tested on PET2; every other class has one training and one test video.
Class weights: n_train / (2 x n_c) from the pair's own training split, pipeline_output/class_weights_twoclass_<pair>.json,
  the same file for the real and the particle-free classifier (same construction as the seven-class file, n_train / (7 x n_c)).
Regime: common 849 patches; 150 epochs, patience 30 on validation macro-F1 (best-validation checkpoint); batch 32;
  AdamW lr 1e-3, weight decay 1e-4, cosine annealing; CompactHoloCNN width 32, dropout 0.3, 2 outputs (--classes);
  rotation 0/180, translation +/-4 px, gain +/-5%, noise 0.01; CPU; seeds 0-4 (Part 2: 0-2).
Chance macro-F1: mean (+/- std) over 1000 draws of random guessing in proportion to the pair's training class
  frequencies, scored on the pair's test patches (numpy default_rng(0)); the method used for the seven-class 0.1373.
Readings (pre-registered, margin 0.10 over the pair's own chance; floor = larger seed std of macro-F1):
  T1 gap > floor AND real > chance + 0.10;  T2 gap <= floor AND particle-free > chance + 0.10;
  T3 real <= chance + 0.10 AND particle-free <= chance + 0.10;  T4 none of these.
  Only PET vs PS is a named test; the other pairs are descriptive. With 12 pair-arm comparisons one or two T1 can
  occur by chance, so no reading is described as a finding.
"""


def specs():
    out = []
    for o in PAIRS:
        for arm in ARMS:
            out.append(dict(part=1, pair=f"PETvs{o}", cls=["PET", o], arm=arm, real=f"real_tc_PETvs{o}_{arm}",
                            bg=f"bg_tc_PETvs{o}_{arm}", seeds=5, cw=f"{PO}/class_weights_twoclass_PETvs{o}.json", rev=False))
    for arm in ARMS:
        out.append(dict(part=2, pair="PETvsPS", cls=["PET", "PS"], arm=arm, real=f"real_tc_PETvsPS_{arm}_rev",
                        bg=f"bg_tc_PETvsPS_{arm}_rev", seeds=3, cw=f"{PO}/class_weights_twoclass_PETvsPS_rev.json", rev=True))
    out.append(dict(part=3, pair="PETvsPS", cls=["PET", "PS"], arm="z0 + interior", real="real_tc_PETvsPS_z0_interior",
                    bg="bg_tc_PETvsPS_z0_interior", seeds=5, cw=f"{PO}/class_weights_twoclass_PETvsPS.json", rev=False))
    return out


def chance(cls, rev):
    m = pd.read_csv(f"{PO}/manifest_bgcorrected_common849{'_rev' if rev else ''}.csv")
    m = m[m["class"].isin(cls)]
    tr, te = m[m.split_video_holdout == "train"]["class"], m[m.split_video_holdout == "test"]["class"].values
    labels = sorted(cls)
    p = tr.value_counts(normalize=True).reindex(labels).values
    rng = np.random.default_rng(0)
    s = [f1_score(te, rng.choice(labels, size=len(te), p=p), labels=labels, average="macro", zero_division=0)
         for _ in range(1000)]
    return float(np.mean(s)), float(np.std(s))


def load(prefix, seeds):
    runs, missing = [], []
    for s in range(seeds):
        d = f"results/{prefix}_s{s}"
        if os.path.exists(f"{d}/results.json"):
            runs.append(json.load(open(f"{d}/results.json")))
        elif os.path.isdir(d):
            missing.append(d)
    return runs, missing


def ms(v):
    return (stats.mean(v), stats.pstdev(v)) if len(v) > 1 else ((v[0], float("nan")) if v else (float("nan"),) * 2)


def side(runs, cls):
    rep = lambda r, c, k: r["test_report"][c][k]
    return dict(n=len(runs), f1=ms([r["test_report"]["macro avg"]["f1-score"] for r in runs]),
                rec={c: ms([rep(r, c, "recall") for r in runs]) for c in cls},
                bacc=ms([np.mean([rep(r, c, "recall") for c in cls]) for r in runs]),
                pet={k: ms([rep(r, "PET", k) for r in runs]) for k in ("precision", "recall", "f1-score")},
                cm=np.mean([np.array(r["confusion_matrix"], float) for r in runs], axis=0),
                support={c: int(rep(runs[0], c, "support")) for c in cls}, classes=runs[0]["classes"])


def reading(real, pf, gap, floor, ch):
    if gap > floor and real > ch + MARGIN:
        return "T1"
    if gap <= floor and pf > ch + MARGIN:
        return "T2"
    if real <= ch + MARGIN and pf <= ch + MARGIN:
        return "T3"
    return "T4"


def main():
    f = lambda t: f"{t[0]:.4f}+/-{t[1]:.4f}"
    rows, blocks, failures, cw_bad = [], [], [], []
    done = {1: 0, 2: 0, 3: 0}
    counts = {arm: {t: 0 for t in ("T1", "T2", "T3", "T4")} for arm in ARMS}
    chances = {}
    for sp in specs():
        key = (tuple(sp["cls"]), sp["rev"])
        if key not in chances:
            chances[key] = chance(sp["cls"], sp["rev"])
        ch = chances[key]
        rr, rm = load(sp["real"], sp["seeds"])
        br, bm = load(sp["bg"], sp["seeds"])
        failures += rm + bm
        done[sp["part"]] += len(rr) + len(br)
        cw_bad += [r["args"]["out"] for r in rr + br
                   if r["args"].get("class_weights") != sp["cw"] or r["args"].get("classes") != ",".join(sp["cls"])]
        label = f"{sp['pair']:10s} {sp['arm']:24s}" + {1: "", 2: " [Part 2, reverse fold]", 3: " [Part 3, interior]"}[sp["part"]]
        if not rr or not br:
            rows.append(f"{label}  not run yet (real n={len(rr)}, particle-free n={len(br)})")
            continue
        real, pf = side(rr, sp["cls"]), side(br, sp["cls"])
        gap, floor = real["f1"][0] - pf["f1"][0], max(real["f1"][1], pf["f1"][1])
        complete = real["n"] == pf["n"] == sp["seeds"]
        rd = reading(real["f1"][0], pf["f1"][0], gap, floor, ch[0]) if complete else "(incomplete)"
        if complete and sp["part"] == 1:
            counts[sp["arm"]][rd] += 1
        rows.append(f"{label}  {f(real['f1'])} {f(pf['f1'])} {gap:+.4f} {floor:.4f} {ch[0]:.4f}  "
                    f"{real['bacc'][0]:.4f} {pf['bacc'][0]:.4f}  {rd}")
        b = ["", "=" * 96, f"{label.strip()}  (real n={real['n']}, particle-free n={pf['n']})", "=" * 96,
             f"macro-F1: real {f(real['f1'])} | particle-free {f(pf['f1'])} | gap {gap:+.4f} floor {floor:.4f} "
             f"{'ABOVE' if gap > floor else 'within'} | chance {f(ch)} | reading {rd}",
             f"balanced accuracy: real {f(real['bacc'])} | particle-free {f(pf['bacc'])}",
             "PET precision / recall / F1: real " + " / ".join(f(real['pet'][k]) for k in ('precision', 'recall', 'f1-score'))
             + " | particle-free " + " / ".join(f(pf['pet'][k]) for k in ('precision', 'recall', 'f1-score')),
             f"{'class':6s} {'n test':>6s} {'real recall':>15s} {'p-free recall':>15s} {'gap':>8s} {'floor':>6s} clears?"]
        for c in real["classes"]:
            r, p = real["rec"][c], pf["rec"][c]
            g, fl = r[0] - p[0], max(r[1], p[1])
            b.append(f"{c:6s} {real['support'][c]:>6d} {r[0]:.3f}+/-{r[1]:.3f} {p[0]:.3f}+/-{p[1]:.3f} {g:>+8.3f} {fl:>6.3f}  "
                     f"{'YES' if g > fl else 'no'}")
        cl = real["classes"]
        b.append(f"mean confusion matrix over seeds (rows true, columns predicted: {' '.join(cl)})")
        for name, s_ in (("real", real), ("particle-free", pf)):
            b.append(f"  {name:13s} " + " | ".join(f"{cl[i]}: " + " ".join(f"{v:6.1f}" for v in s_["cm"][i]) for i in range(len(cl))))
        blocks += b

    lines = [HEADER, "=" * 96, "SUMMARY (macro-F1 mean +/- std over seeds; balanced accuracy = mean of the two recalls)", "=" * 96,
             f"{'pair':10s} {'arm':24s}  {'real':>15s} {'particle-free':>15s} {'gap':>7s} {'floor':>6s} {'chance':>6s}  "
             f"{'bacc R':>6s} {'bacc PF':>7s}  reading"] + rows + blocks
    total = {1: 120, 2: 12, 3: 10}
    lines += ["", "=" * 96,
              "runs finished: " + ", ".join(f"Part {k} {done[k]} of {total[k]}" for k in (1, 2, 3)),
              f"class-weights file (pipeline_output/class_weights_twoclass_<pair>[_rev].json) and --classes recorded as "
              f"expected in every finished run: {'yes' if not cw_bad else 'NO: ' + str(cw_bad)}"]
    failed_txt = "results/logs_twoclass/failed.txt"
    logged = open(failed_txt).read().split() if os.path.exists(failed_txt) else []
    lines.append(f"failed / incomplete runs: {failures or 'none'}" + (f"; logged failures: {logged}" if logged else ""))
    lines.append("reading counts, Part 1 (complete pair-arm comparisons only): "
                 + "; ".join(f"{arm}: " + ", ".join(f"{t} {n}" for t, n in counts[arm].items()) for arm in ARMS))
    text = "\n".join(lines)
    open("results/report_twoclass.txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
