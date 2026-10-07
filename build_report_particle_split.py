"""Particle-identity split comparison report -> results/report_particle_split.txt.
Same layout as results/report_R2_reweighted_bg.txt, per arm: macro-F1 real vs
particle-free, gap, floor (larger of the two seed stds), above/within; per-class
recall table with gap, floor, clears; PET precision/recall/F1; test class counts.
Summary: this split next to the video-holdout numbers parsed from
results/report_R_reweighted_bg.txt and results/report_R2_reweighted_bg.txt (real =
original runs, particle-free = matched *_cwvh runs). Regenerated after every arm."""
import os
import re
import json
import statistics as stats

CLASSES = ["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"]
ARMS = [("intensity_bgcorr_vnorm", "intbv"), ("pol4_bgcorr_vnorm", "pol4bgcorrvnorm"), ("pol4_bgcorr", "pol4bgcorr"),
        ("pol4", "pol4"), ("z0", "recons0z0"), ("recon_s0", "recons0"), ("recon_biref", "reconbiref")]
CW = "pipeline_output/class_weights_psplit.json"
HEADER = """PARTICLE-IDENTITY SPLIT COMPARISON (new runs, folders results/{real,bg}_<arm>_s<seed>_psplit)
Split: the earlier particle-identity split (column 'split' of pipeline_output/manifest.csv, identical to
  manifest_backup_pre_video_holdout.csv) restricted to the common 849 patches; saved as
  pipeline_output/split_particle_identity_849.csv. Background-only patches take their particle's split.
Class weights: pipeline_output/class_weights_psplit.json (n_train/(7*n_c) from this split), both classifiers.
Regime copied from the corrected queue (scripts/run_queue_r180.sh -> scripts/run_real_bg_seeds.sh):
  epochs 150, early-stopping patience 30 on validation macro-F1 (best-validation checkpoint kept),
  batch size 32, AdamW lr 1e-3, weight decay 1e-4, cosine annealing over the epochs, CompactHoloCNN
  width 32, dropout 0.3 (train_compact_cnn.py defaults, unchanged); augmentation: rotation 0/180 only,
  translation +/-4 px, gain +/-5%, noise 0.01; per-channel normalisation from the training split; CPU.
  Common 849 patches (frame-0 patches excluded). Same channel modes, correction and reconstruction
  patch sets (caches) as the corrected queue. 5 seeds (0-4) per classifier.
"""


def load(prefix):
    runs, missing = [], []
    for s in range(5):
        p = f"results/{prefix}_s{s}_psplit/results.json"
        if os.path.exists(p):
            runs.append(json.load(open(p)))
        elif os.path.isdir(os.path.dirname(p)):
            missing.append(os.path.dirname(p))
    return runs, missing


def ms(v):
    return (stats.mean(v), stats.pstdev(v)) if len(v) > 1 else ((v[0], float("nan")) if v else (float("nan"),) * 2)


def side(runs):
    g = lambda c, k: ms([r["test_report"][c][k] for r in runs])
    return dict(n=len(runs), rec={c: g(c, "recall") for c in CLASSES},
                pet={k: g("PET", k) for k in ("precision", "recall", "f1-score")},
                f1=ms([r["test_report"]["macro avg"]["f1-score"] for r in runs]),
                support={c: int(runs[0]["test_report"][c]["support"]) for c in CLASSES})


def video_holdout():
    out = {}
    for path in ("results/report_R_reweighted_bg.txt", "results/report_R2_reweighted_bg.txt"):
        text = open(path).read()
        for m in re.finditer(r"^(\S+)\s+\(real n=.*?\n=+\nmacro-F1: real ([\d.]+)\+/-([\d.]+) \| old bg .*? \| "
                             r"new bg ([\d.]+)\+/-([\d.]+)", text, re.M):
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
    return out


def main():
    f = lambda t: f"{t[0]:.4f}+/-{t[1]:.4f}"
    lines, summary, failures, cw_bad, n_runs = [HEADER], {}, [], [], 0
    for name, tag in ARMS:
        rr, rmiss = load(f"real_{tag}")
        br, bmiss = load(f"bg_{tag}")
        failures += rmiss + bmiss
        n_runs += len(rr) + len(br)
        cw_bad += [f"{name}: {r['args'].get('class_weights')}" for r in rr + br if r["args"].get("class_weights") != CW]
        if not rr or not br:
            lines.append(f"\n{name}: not run yet (real n={len(rr)}, particle-free n={len(br)})")
            continue
        real, pf = side(rr), side(br)
        gap, floor = real["f1"][0] - pf["f1"][0], max(real["f1"][1], pf["f1"][1])
        summary[name] = dict(real=real["f1"], pf=pf["f1"], gap=gap, floor=floor, n=(real["n"], pf["n"]))
        lines += ["", "=" * 92, f"{name}  (real n={real['n']}, particle-free n={pf['n']})", "=" * 92,
                  f"macro-F1: real {f(real['f1'])} | particle-free {f(pf['f1'])} | gap {gap:+.4f} floor {floor:.4f} "
                  f"{'ABOVE' if gap > floor else 'within'}",
                  "PET precision / recall / F1: real " + " / ".join(f(real['pet'][k]) for k in ('precision', 'recall', 'f1-score'))
                  + " | particle-free " + " / ".join(f(pf['pet'][k]) for k in ('precision', 'recall', 'f1-score')),
                  f"{'class':6s} {'n test':>6s} {'real':>15s} {'particle-free':>15s} {'gap':>8s} {'floor':>6s} clears?"]
        clears = []
        for c in CLASSES:
            r, p = real["rec"][c], pf["rec"][c]
            g, fl = r[0] - p[0], max(r[1], p[1])
            clears += [c] if g > fl else []
            lines.append(f"{c:6s} {real['support'][c]:>6d} {r[0]:.3f}+/-{r[1]:.3f} {p[0]:.3f}+/-{p[1]:.3f} "
                         f"{g:>+8.3f} {fl:>6.3f}  {'YES' if g > fl else 'no'}"
                         + ("   (fewer than 5 test patches)" if real["support"][c] < 5 else ""))
        lines.append(f"classes clearing the floor: {clears or 'none'}")

    vh = video_holdout()
    lines += ["", "=" * 92, "SUMMARY: macro-F1, this split vs video-holdout", "=" * 92,
              "video-holdout: real = original runs, particle-free = matched-weight *_cwvh runs, parsed from",
              "results/report_R_reweighted_bg.txt and results/report_R2_reweighted_bg.txt (not retrained)",
              f"{'arm':24s} {'psplit real':>15s} {'psplit p-free':>15s} {'gap':>8s} {'floor':>7s} {'':6s} "
              f"{'vh real':>15s} {'vh p-free':>15s}"]
    for name, _ in ARMS:
        if name not in summary:
            lines.append(f"{name:24s} not run yet")
            continue
        s, v = summary[name], vh.get(name)
        vtxt = f"{v[0]:.4f}+/-{v[1]:.4f} {v[2]:.4f}+/-{v[3]:.4f}" if v else "not found in R/R2 reports"
        lines.append(f"{name:24s} {f(s['real']):>15s} {f(s['pf']):>15s} {s['gap']:>+8.4f} {s['floor']:>7.4f} "
                     f"{'ABOVE' if s['gap'] > s['floor'] else 'within':6s} {vtxt}")

    lines += ["", "Pre-registered readings:"]
    if len(summary) == len(ARMS) and all(s["n"] == (5, 5) for s in summary.values()):
        n_within = sum(s["gap"] <= s["floor"] for s in summary.values())
        n_above = sum(s["gap"] > s["floor"] for s in summary.values())
        n_pf_above = [k for k, s in summary.items() if s["gap"] < -s["floor"]]
        lines.append(f"(A) gap <= floor in >= 5 of 7 arms: {'APPLIES' if n_within >= 5 else 'does not apply'} ({n_within} of 7)")
        lines.append(f"(B) real exceeds particle-free by more than the floor in >= 5 of 7 arms: "
                     f"{'APPLIES  <<< FLAG >>>' if n_above >= 5 else 'does not apply'} ({n_above} of 7)")
        if n_within < 5 and n_above < 5:
            lines.append("(C) mixed: see the per-arm blocks")
        lines.append(f"arms counted in (A) where particle-free exceeds real by more than the floor: {n_pf_above or 'none'}")
        higher = [k for k, s in summary.items() if k in vh and s["real"][0] > vh[k][0]]
        lines.append(f"real macro-F1 higher in this split than under video-holdout: {len(higher)} of 7 arms {higher}")
    else:
        lines.append("not evaluated yet: not every arm has 5 + 5 finished runs")

    lines += ["", f"runs finished: {n_runs} of 70",
              f"class-weights file recorded in every finished run = {CW}: {'yes' if not cw_bad else 'NO: ' + str(cw_bad)}",
              f"failed / incomplete runs: {failures or 'none'}"]
    text = "\n".join(lines)
    open("results/report_particle_split.txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
