"""Item 49 report: width 64, 10 layers, reverse direction (test recordings PET1 + PS2), arms Wn and Wd; plus the capacity
grid of all eight arms x two directions. No training.
--partial  writes results/capacity_grid_summary.csv only, from result files: the saved prediction files of the earlier
           arms (scored there from each run's best.pt and checked against its results.json) re-checked here against
           each run folder's results.json (4 decimals), and, for the new item 49 runs, the test macro-F1 in results.json.
full       also scores best.pt of the finished item 49 runs on their test patches with scripts/item46_analysis.py
           evaluate() (the per-patch predictions the paired bootstrap needs; 4-decimal check), and writes
           results/report_w64_reverse_exp.txt (once; never overwritten) and results/w64_reverse_predictions_exp.csv.
Statistics: scripts/item44_chain.py stats() and scripts/item46_analysis.py cmp() (1000 resamples of whole test tracks,
default_rng(0), the same resampled tracks for every arm, mean over seeds per resample). Seeds 0 to 2 for every arm.
usage: python scripts/item49_report.py [--partial] [--runs P] [--csv P] [--report P] [--pred P]"""
import os
import sys
import json
import time
import hashlib

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import item44_chain as c44  # noqa: E402  (stats)
import item46_analysis as a46  # noqa: E402  (evaluate, cmp)
prio41.set_priority(prio41.NORMAL)

RES = "results"
READ = f"{RES}/item49_readings.txt"
B = 1000
SEEDS = (0, 1, 2)
CL = ["PET", "PS"]
# arm, run folder prefix, weighted layers, width, depth-aug, saved predictions of the earlier runs, arm label in that file
ARMS = [("z0", "real_tc_PETvsPS_z0_exp", 10, 32, "none", f"{RES}/z0_predictions_exp.csv", None),
        ("D", "real_tc_PETvsPS_z0depthD_exp", 10, 32, "all21", f"{RES}/depth_aug_predictions_exp.csv", "D"),
        ("Wn", "real_tc_PETvsPS_z0w64_exp", 10, 64, "none", f"{RES}/width_depth_predictions_exp.csv", "Wn"),
        ("Wd", "real_tc_PETvsPS_z0depthD_w64_exp", 10, 64, "all21", f"{RES}/width_depth_predictions_exp.csv", "Wd"),
        ("Ln", "real_tc_PETvsPS_z0deep18_exp", 18, 32, "none", f"{RES}/layers_depth_predictions_exp.csv", "Ln"),
        ("Ld", "real_tc_PETvsPS_z0deep18depthD_exp", 18, 32, "all21", f"{RES}/layers_depth_predictions_exp.csv", "Ld"),
        ("WLn", "real_tc_PETvsPS_z0wl_exp", 18, 64, "none", f"{RES}/width_layers_predictions_exp.csv", "WLn"),
        ("WLd", "real_tc_PETvsPS_z0wl_depthD_exp", 18, 64, "all21", f"{RES}/width_layers_predictions_exp.csv", "WLd")]
NEW = {"Wn", "Wd"}  # reverse direction of these two arms comes from item 49


def clean(text):
    low = text.lower()
    assert "\u2014" not in text and ("signif" + "icant") not in low and ("valid" + "ated") not in low
    for p in ("capacity is ruled out", "the cnn is not too simple", "deeper is worse"):
        assert p not in low, p
    return text


def saved_preds(path, label, direction):
    P = pd.read_csv(path)
    if label is None:
        P = P[P.kind == "real"].rename(columns={"patch_id": "patch_file"})
    else:
        P = P[P.arm == label]
    return P[(P.direction == direction) & P.seed.isin(SEEDS)]


def f1_saved(run):
    return float(json.load(open(f"{RES}/{run}/results.json"))["test_report"]["macro avg"]["f1-score"])


def main():
    partial = "--partial" in sys.argv
    g = lambda k, dflt: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else dflt
    runs_path, csv_path = g("--runs", f"{RES}/logs_item49/runs.json"), g("--csv", f"{RES}/capacity_grid_summary.csv")
    rep_path, pred_path = g("--report", f"{RES}/report_w64_reverse_exp.txt"), g("--pred", f"{RES}/w64_reverse_predictions_exp.csv")
    if not partial:
        for p in (rep_path, pred_path):
            if os.path.exists(p):
                raise FileExistsError(p)
    rj = json.load(open(runs_path)) if os.path.exists(runs_path) else dict(runs={}, timing=[], not_run=[], failed=[])
    new_runs = {a: {int(s): d for s, d in rj["runs"].get(a, {}).get("reverse", {}).items()} for a in NEW}
    z = saved_preds(ARMS[0][5], None, "reverse")
    tracks = {dr: sorted(set(saved_preds(ARMS[0][5], None, dr).track_id)) for dr in ("original", "reverse")}
    Ws = {}
    for dr, tr in tracks.items():
        rng = np.random.default_rng(0)
        Ws[dr] = np.stack([np.bincount(rng.integers(0, len(tr), len(tr)), minlength=len(tr)) for _ in range(B)])
    rows, excl, frames, info = [], [], {}, {}
    for arm, prefix, layers, width, depth, path, label in ARMS:
        npar = next((json.load(open(f"{RES}/{prefix}{sfx}_s{s}/results.json")).get("n_params") for sfx in ("", "_rev") for s in SEEDS
                     if os.path.exists(f"{RES}/{prefix}{sfx}_s{s}/results.json")), "not run")
        for dr, sfx in (("original", ""), ("reverse", "_rev")):
            per, src = {}, ""
            if arm in NEW and dr == "reverse":
                if partial:
                    per = {s: f1_saved(os.path.basename(d)) for s, d in sorted(new_runs[arm].items())}
                    src = "item 49 runs, results.json test macro-F1 (not yet scored)"
                else:
                    P = []
                    for s, d in sorted(new_runs[arm].items()):
                        df, f1, f1s = a46.evaluate(arm, d)
                        sv = json.load(open(f"{d}/results.json"))
                        info[os.path.basename(d)] = dict(epochs=len(sv["history"]), best=sv["best_epoch"],
                                                         tr=sv["history"][sv["best_epoch"] - 1].get("train_macro_f1"))
                        if round(f1, 4) != round(f1s, 4):
                            excl.append(f"{os.path.basename(d)} (recomputed {f1:.4f} vs saved {f1s:.4f})")
                            continue
                        P.append(df.assign(seed=s, run=os.path.basename(d), arm=arm, direction=dr))
                    src = "item 49 runs, scored from best.pt (scripts/item46_analysis.py evaluate())"
                    if P:
                        frames[(arm, dr)] = pd.concat(P)
                        st = c44.stats(frames[(arm, dr)], tracks[dr], Ws[dr])
                        per = dict(zip(sorted(set(frames[(arm, dr)].seed)), st["f1s"]))
            else:
                P = saved_preds(path, label, dr)
                if len(P):
                    keep = []
                    for s in sorted(set(P.seed)):
                        run = P[P.seed == s].run.iloc[0]
                        st1 = c44.stats(P[P.seed == s], tracks[dr], Ws[dr])["f1s"][0]
                        if os.path.exists(f"{RES}/{run}/results.json") and round(st1, 4) == round(f1_saved(run), 4):
                            keep.append(s)
                        else:
                            excl.append(f"{run} (saved predictions {st1:.4f} vs results.json "
                                        f"{f1_saved(run) if os.path.exists(f'{RES}/{run}/results.json') else 'missing'})")
                    P = P[P.seed.isin(keep)]
                    frames[(arm, dr)] = P
                    st = c44.stats(P, tracks[dr], Ws[dr])
                    per = dict(zip(sorted(set(P.seed)), st["f1s"]))
                    src = f"saved predictions {path}, checked against each run folder's results.json"
            if per:
                v = np.array(list(per.values()))
                rows.append(dict(arm=arm, direction=dr, status="run", seeds_present=" ".join(map(str, per)), n_seeds=len(per),
                                 macro_f1_mean=round(float(v.mean()), 4), macro_f1_std_pop=round(float(v.std()), 4),
                                 per_seed="; ".join(f"s{s} {x:.4f}" for s, x in per.items()), parameters=npar, layers=layers,
                                 width=width, depth_aug=depth, source=src))
            else:
                rows.append(dict(arm=arm, direction=dr, status="not run", seeds_present="not run", n_seeds=0,
                                 macro_f1_mean="not run", macro_f1_std_pop="not run", per_seed="not run", parameters=npar,
                                 layers=layers, width=width, depth_aug=depth, source="not run"))
            print(f"{arm:4s} {dr:9s} {rows[-1]['status']:8s} {rows[-1]['per_seed']}  mean {rows[-1]['macro_f1_mean']} "
                  f"std {rows[-1]['macro_f1_std_pop']}")
    grid = pd.DataFrame(rows)
    grid["excluded_by_4_decimal_check"] = "; ".join(excl) if excl else "none"
    grid.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}; excluded: {excl or 'none'}")
    if partial:
        return

    sha = hashlib.sha256(open(READ, "rb").read()).hexdigest()
    dr, W, tr = "reverse", Ws["reverse"], tracks["reverse"]
    A = {k: frames[(k, dr)] for k in ("z0", "D", "Wn", "Wd") if (k, dr) in frames and len(frames[(k, dr)])}
    nseed = {k: len(set(v.seed)) for k, v in A.items()}
    for k in ("Wn", "Wd"):
        if k in A:
            A[k].to_csv(pred_path, mode="a", header=not os.path.exists(pred_path), index=False,
                        columns=["run", "arm", "direction", "seed", "patch_file", "track_id", "true_class", "predicted_class"])
    L = ["ITEM 49: WIDTH 64, 10 LAYERS, REVERSE DIRECTION (test recordings PET1 + PS2), z0 EXPANDED SET, PET vs PS, REAL RUNS ONLY",
         "Completeness run for an empty grid cell, decided after items 47 and 48; every reading here is descriptive.",
         f"Readings: {READ}, sha256 {sha}.",
         "Commands: scripts/item44_chain.py cmd() for the reverse direction, changed only to --width 64; Wn additionally",
         "--depth-aug none with the flat z0 patch folder of results/real_tc_PETvsPS_z0_exp_rev_s0 (as item 46 Wn; the depth",
         "folder holds only depth subfolders), Wd --depth-aug all21 with item 44's depth folder.",
         "No particle-free runs, so no T reading.",
         "Seeds per arm (reverse): " + ", ".join(f"{k} {nseed.get(k, 0)}" for k in ("z0", "D", "Wn", "Wd")) + ".",
         "The bootstrap resamples whole test tracks only (1000 resamples, default_rng(0)); it does not cover seeds or recordings.",
         "Margin for each comparison = the larger real seed population std of the two arms over their shared seeds.", ""]
    for k in ("Wn", "Wd"):
        if k not in A:
            L.append(f"{k}: no finished run.")
            continue
        st = c44.stats(A[k], tr, W)
        f1c = []
        for s in sorted(set(A[k].seed)):
            C = A[k][A[k].seed == s]
            p = np.array([((C.predicted_class == c) & (C.true_class == c)).sum() / max((C.predicted_class == c).sum(), 1) for c in CL])
            r = np.array([((C.predicted_class == c) & (C.true_class == c)).sum() / max((C.true_class == c).sum(), 1) for c in CL])
            f1c.append(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0))
        f1c = np.array(f1c)
        L.append(f"{k} (reverse, seeds {sorted(set(A[k].seed))}): macro-F1 {st['f1'][0]:.4f} +/- {st['f1'][1]:.4f} "
                 f"[track CI {st['ci'][0]:.4f}, {st['ci'][1]:.4f}]; per seed " + ", ".join(f"{x:.4f}" for x in st["f1s"]))
        L.append(f"  {'class':5s} {'precision mean +/- std [track CI]':>36s} {'recall mean +/- std [track CI]':>36s} {'F1 mean +/- std':>16s}")
        for i, c in enumerate(CL):
            pr, rc = st["prec"][i], st["rec"][i]
            L.append(f"  {c:5s} {pr[0][0]:.3f} +/- {pr[0][1]:.3f} [{pr[1][0]:.3f}, {pr[1][1]:.3f}]       "
                     f"{rc[0][0]:.3f} +/- {rc[0][1]:.3f} [{rc[1][0]:.3f}, {rc[1][1]:.3f}]       {f1c[:, i].mean():.3f} +/- {f1c[:, i].std():.3f}")
        L.append("")
    c = {}
    for a_, b_ in (("Wd", "D"), ("Wd", "Wn"), ("Wn", "z0")):
        if a_ in A and b_ in A:
            c[f"{a_} - {b_}"] = a46.cmp(A[a_], A[b_], W, tr)
    L.append("Comparisons (reverse):")
    for k, v in c.items():
        vd = "above" if v["diff"] > v["margin"] else ("below" if v["diff"] < -v["margin"] else "within")
        L.append(f"  {k:8s} {v['diff']:+.4f} (margin {v['margin']:.4f}, {vd}; paired track CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}], "
                 f"seeds {v['seeds']})")
    n = min(nseed.get("Wn", 0), nseed.get("Wd", 0))
    above = lambda k: k in c and c[k]["diff"] > c[k]["margin"]
    below = lambda k: k in c and c[k]["diff"] < -c[k]["margin"]
    within = lambda k: k in c and abs(c[k]["diff"]) <= c[k]["margin"]
    if n <= 1:
        R = "no X reading (fewer than 2 seeds for Wn or Wd); only the per-seed values above are given"
    else:
        R = ("X1r" if above("Wd - D") and above("Wd - Wn") else "X2r" if above("Wn - z0") else
             "X3r" if within("Wd - D") and within("Wn - z0") else "X4r" if below("Wd - D") else "X5r")
        if n == 2:
            R += " (reduced design: 2 seeds, margins over those 2 seeds)"
    L += ["", f"READING (rules of {READ}, sha256 {sha}; applied in order, the first that holds): {R}", "",
          "Runs (epochs run, best epoch, run seconds, training macro-F1 at the saved checkpoint as logged in the run's history,",
          "train mode during that epoch, with the script's augmentation and dropout; no extra evaluation pass):"]
    tsec = {t["run"]: t for t in rj.get("timing", [])}
    for k in ("Wn", "Wd"):
        for s, d in sorted(new_runs[k].items()):
            b = os.path.basename(d)
            i_ = info.get(b, {})
            t_ = tsec.get(b, {})
            L.append(f"  {b}: {i_.get('epochs')} epochs, best {i_.get('best')}, {t_.get('seconds', 'n/a')} s, training macro-F1 at "
                     f"the saved checkpoint {i_.get('tr') if i_.get('tr') is None else format(i_.get('tr'), '.4f')}")
    L += [f"Runs not run (stop rule): {rj.get('not_run') or 'none'}. Runs failed twice: {rj.get('failed') or 'none'}.",
          f"Runs excluded by the 4-decimal check: {excl or 'none'}.",
          f"Grid summary: {csv_path}. Test predictions of the item 49 runs: {pred_path}.",
          f"Written {time.strftime('%Y-%m-%d %H:%M')}"]
    open(rep_path, "w", encoding="utf-8").write(clean("\n".join(L)) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
