"""CNN-level track intervals (read-only; no training). For every run of an arm (real and particle-free, both directions):
rebuilds the test inputs from the run's results.json args (evaluation code of scripts/p_eval_checkpoints.py, extended
with --classes and the --meannorm flag of train_compact_cnn_bright.py), loads best.pt, predicts every test patch,
checks the recomputed macro-F1 against results.json to 4 decimals (a run that differs is reported and excluded),
saves the predictions (run, patch id, track id, true class, predicted class), then per direction bootstraps TRACKS of
the test videos (1000 resamples, numpy default_rng(0); the same resampled tracks for the real and the particle-free
classifier, whose test patches carry the track of their paired real patch) and reports, per resample, the mean over
seeds of macro-F1, per-class recall and precision for each classifier and the gap (real minus particle-free).
usage: python scripts/eval_track_ci_exp.py z0 | armG | armN | PETvsPP
"""
import os
import sys
import json
import importlib.util
import statistics as st
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
PO, RES = "pipeline_output", "results"
KEY = ["source_file", "frame", "particle_id"]
B = 1000
_mods = {}


def module(path):
    if path not in _mods:
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _mods[path] = m
    return _mods[path]


def evaluate(run_dir):
    """Per-patch test predictions of one run. Returns (DataFrame, recomputed macro-F1, saved macro-F1)."""
    saved = json.load(open(os.path.join(run_dir, "results.json")))
    a = saved["args"]
    bright = "gain_range" in a or "meannorm" in a
    m = module("train_compact_cnn_bright.py" if bright else "train_compact_cnn.py")
    split_col = m.SPLIT_CONFIG[a["split"]]["split_column"]
    man = pd.read_csv(a["manifest"])
    if a.get("classes"):
        man = man[man["class"].isin(a["classes"].split(","))]
    classes = saved["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    rep = {"meannorm": True} if a.get("meannorm") else {}
    train_df, test_df = man[man[split_col] == "train"], man[man[split_col] == "test"].reset_index(drop=True)
    mean, std = m.compute_channel_stats(train_df, a["patch_dir"], c2i, a["channels"], patch_norm=a.get("patch_norm", False), **rep)
    in_ch = {"recon_s0": 1, "intensity_bgcorr_vnorm": 1}.get(a["channels"], 4)
    model = m.CompactHoloCNN(in_ch, len(classes), a["width"], a["dropout"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    model.eval()
    ds = m.PatchDataset(test_df, a["patch_dir"], c2i, channels=a["channels"], mean=mean, std=std,
                        patch_norm=a.get("patch_norm", False), **rep)
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=64, shuffle=False):
            preds += model(x).argmax(1).tolist()
    out = test_df[KEY + ["patch_file", "class", "track_id"]].copy()
    out["pred"] = [classes[p] for p in preds]
    f1 = f1_score(out["class"], out["pred"], labels=classes, average="macro", zero_division=0)
    return out, float(f1), float(saved["test_report"]["macro avg"]["f1-score"]), classes


def per_track_counts(df, tracks, classes):
    """(T, K, K) confusion counts per track, rows true, columns predicted."""
    ti = {t: i for i, t in enumerate(tracks)}
    ci = {c: i for i, c in enumerate(classes)}
    C = np.zeros((len(tracks), len(classes), len(classes)))
    for t, y, p in zip(df.track_id, df["class"], df.pred):
        C[ti[t], ci[y], ci[p]] += 1
    return C


def metrics(cm):
    """cm (..., K, K) -> macro-F1, recall (..., K), precision (..., K); zero division -> 0 (as sklearn)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    rows, cols = cm.sum(-1), cm.sum(-2)
    rec = np.divide(tp, rows, out=np.zeros_like(tp), where=rows > 0)
    prec = np.divide(tp, cols, out=np.zeros_like(tp), where=cols > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return f1.mean(-1), rec, prec


def chance(man, col, classes):
    tr, te = man[man[col] == "train"]["class"], man[man[col] == "test"]["class"].values
    p = tr.value_counts(normalize=True).reindex(classes).values
    rng = np.random.default_rng(0)
    s = [f1_score(te, rng.choice(classes, size=len(te), p=p), labels=classes, average="macro", zero_division=0)
         for _ in range(1000)]
    return float(np.mean(s))


def t_reading(real, pf, gap, floor, ch, margin=0.10):
    if gap > floor and real > ch + margin:
        return "T1"
    if gap <= floor and pf > ch + margin:
        return "T2"
    if real <= ch + margin and pf <= ch + margin:
        return "T3"
    return "T4"


def ms(v):
    return (st.mean(v), st.pstdev(v)) if len(v) > 1 else ((v[0], float("nan")) if v else (float("nan"),) * 2)


def spec_from_prefixes(prefixes, seeds, retries=None):
    """{direction: {kind: {seed: run dir}}} from name prefixes; a recorded retry dir replaces a missing run."""
    retries = retries or {}
    out = {}
    for direction, (rp, bp) in prefixes.items():
        out[direction] = {}
        for kind, pref in (("real", rp), ("particle-free", bp)):
            out[direction][kind] = {}
            for s in seeds[direction]:
                d = f"{RES}/{pref}_s{s}"
                out[direction][kind][s] = retries.get(d, d)
    return out


def analyse(name, spec, pred_csv):
    """spec: {direction: {kind: {seed: run dir}}}. Returns (lines, summary dict, excluded runs)."""
    L, summ, all_preds, excluded = [], {}, [], []
    for direction, kinds in spec.items():
        runs = {}
        seeds = {direction: sorted(set(int(s) for k in kinds.values() for s in k))}
        for kind in ("real", "particle-free"):
            for s_, d in kinds.get(kind, {}).items():
                s = int(s_)
                if not os.path.exists(f"{d}/results.json"):
                    excluded.append(f"{d} (no results.json)")
                    continue
                df, f1, f1_saved, classes = evaluate(d)
                if round(f1, 4) != round(f1_saved, 4):
                    excluded.append(f"{d} (recomputed macro-F1 {f1:.4f} vs saved {f1_saved:.4f})")
                    continue
                runs[(kind, s)] = df
                x = df.assign(run=os.path.basename(d), kind=kind, direction=direction, seed=s)
                all_preds.append(x[["run", "kind", "direction", "seed", "patch_file", "track_id", "class", "pred"]].rename(
                    columns={"patch_file": "patch_id", "class": "true_class", "pred": "predicted_class"}))
        if not runs:
            continue
        classes = sorted(next(iter(runs.values()))["class"].unique())
        tracks = sorted(set().union(*[set(df.track_id) for df in runs.values()]))
        rng = np.random.default_rng(0)
        W = np.stack([np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks)) for _ in range(B)])
        res, full = {}, {}
        for kind in ("real", "particle-free"):
            ss = [s for s in seeds[direction] if (kind, s) in runs]
            Cs = [per_track_counts(runs[(kind, s)], tracks, classes) for s in ss]
            f1b = np.mean([metrics(np.einsum("bt,tij->bij", W, C))[0] for C in Cs], axis=0)
            recb = np.mean([metrics(np.einsum("bt,tij->bij", W, C))[1] for C in Cs], axis=0)
            precb = np.mean([metrics(np.einsum("bt,tij->bij", W, C))[2] for C in Cs], axis=0)
            res[kind] = dict(f1=f1b, rec=recb, prec=precb, n=len(ss))
            cms = [C.sum(0) for C in Cs]
            f1s = [metrics(cm)[0] for cm in cms]
            full[kind] = dict(cm=np.mean(cms, axis=0), f1=ms(f1s), rec=[ms([metrics(cm)[1][k] for cm in cms]) for k in range(len(classes))],
                              prec=[ms([metrics(cm)[2][k] for cm in cms]) for k in range(len(classes))])
        if "particle-free" not in res or "real" not in res:
            continue
        gapb = res["real"]["f1"] - res["particle-free"]["f1"]
        q = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        real, pf = full["real"]["f1"], full["particle-free"]["f1"]
        gap, floor = real[0] - pf[0], max(real[1], pf[1])
        rd0 = [kinds["real"][s_] for s_ in kinds["real"] if os.path.exists(f"{kinds['real'][s_]}/results.json")][0]
        man = pd.read_csv(json.load(open(f"{rd0}/results.json"))["args"]["manifest"])
        man = man[man["class"].isin(classes)]
        ch = chance(man, "split_video_holdout", classes)
        T = t_reading(real[0], pf[0], gap, floor, ch) if res["real"]["n"] == res["particle-free"]["n"] == len(seeds[direction]) \
            else "(incomplete)"
        summ[direction] = dict(real=real, pf=pf, gap=gap, floor=floor, chance=ch, T=T, real_ci=q(res["real"]["f1"]),
                               pf_ci=q(res["particle-free"]["f1"]), gap_ci=q(gapb), n_tracks=len(tracks),
                               n_real=res["real"]["n"], n_pf=res["particle-free"]["n"], classes=classes,
                               rec={k: [full[k]["rec"][i] for i in range(len(classes))] for k in full},
                               prec={k: [full[k]["prec"][i] for i in range(len(classes))] for k in full})
        L += ["", f"{name}, {direction} direction: {res['real']['n']} real and {res['particle-free']['n']} particle-free runs "
                  f"(seeds {seeds[direction]}), {len(tracks)} test tracks, chance {ch:.4f}",
              "  pre-registered reading (seed-based floor) first:",
              f"  macro-F1 real {real[0]:.4f}+/-{real[1]:.4f} | particle-free {pf[0]:.4f}+/-{pf[1]:.4f} | gap {gap:+.4f} | "
              f"floor {floor:.4f} | T reading {T}",
              "  track bootstrap (extra column; mean of the 1000 resamples and 2.5 / 97.5 percentiles, mean over seeds per resample):",
              f"  real {q(res['real']['f1'])[0]:.4f} [{q(res['real']['f1'])[1]:.4f}, {q(res['real']['f1'])[2]:.4f}]; particle-free "
              f"{q(res['particle-free']['f1'])[0]:.4f} [{q(res['particle-free']['f1'])[1]:.4f}, {q(res['particle-free']['f1'])[2]:.4f}]; "
              f"gap {q(gapb)[0]:+.4f} [{q(gapb)[1]:+.4f}, {q(gapb)[2]:+.4f}]",
              f"  {'class':5s} {'kind':13s} {'recall':>15s} {'recall track CI':>17s} {'precision':>15s} {'precision track CI':>19s}"]
        for kind in ("real", "particle-free"):
            for i, c in enumerate(classes):
                rq, pq = q(res[kind]["rec"][:, i]), q(res[kind]["prec"][:, i])
                fr, fp = full[kind]["rec"][i], full[kind]["prec"][i]
                L.append(f"  {c:5s} {kind:13s} {fr[0]:.3f}+/-{fr[1]:.3f} [{rq[1]:.3f}, {rq[2]:.3f}]   {fp[0]:.3f}+/-{fp[1]:.3f} "
                         f"[{pq[1]:.3f}, {pq[2]:.3f}]")
        L.append(f"  mean-over-seeds confusion matrices (rows true, columns predicted: {' '.join(classes)}):")
        for kind in ("real", "particle-free"):
            cm = full[kind]["cm"]
            L.append(f"    {kind:13s} " + " | ".join(f"true {c}: " + " ".join(f"{v:7.1f}" for v in cm[i]) for i, c in enumerate(classes)))
    if all_preds:
        pd.concat(all_preds).to_csv(pred_csv, index=False)
    return L, summ, excluded


def size_row(direction, part3="results/expanded_PETPS_exp/part3_results.json"):
    r = json.load(open(part3))
    for x in r["results"]:
        if x["model"].startswith("3a") and x["direction"] == direction and x["subset"] == "all":
            return x
    return None


def z0_seeds():
    p = json.load(open("results/expanded_PETPS_exp/part4_projection.json"))
    return {"original": [int(s) for s in p["original_seeds"].split()], "reverse": [0, 1, 2]}


def main(what):
    if what == "z0":
        seeds = z0_seeds()
        prefixes = {"original": ("real_tc_PETvsPS_z0_exp", "bg_tc_PETvsPS_z0_exp"),
                    "reverse": ("real_tc_PETvsPS_z0_exp_rev", "bg_tc_PETvsPS_z0_exp_rev")}
        rt = f"{RES}/overnight_z0_retries_exp.json"
        retries = json.load(open(rt)) if os.path.exists(rt) else {}
        L, summ, excl = analyse("z0", spec_from_prefixes(prefixes, seeds, retries), f"{RES}/z0_predictions_exp.csv")
        head = ["CNN-LEVEL TRACK INTERVALS, z0 PET vs PS, expanded set (step 1 of the overnight order; read-only)",
                "Predictions regenerated from best.pt with the evaluation code of scripts/p_eval_checkpoints.py; saved to "
                "results/z0_predictions_exp.csv (run, patch id, track id, true class, predicted class).",
                f"Runs excluded (recomputed macro-F1 differs from results.json at 4 decimals, or missing): {excl or 'none'}"]
        P3 = json.load(open("results/expanded_PETPS_exp/part3_results.json"))["results"]
        for direction in ("original", "reverse"):
            sz = size_row(direction)
            if sz:
                L.append(f"size-only (Part 3a, all patches), {direction}: macro-F1 {sz['f1']:.4f}, patch CI [{sz['lo']:.4f}, "
                         f"{sz['hi']:.4f}], track CI [{sz['lo_track']:.4f}, {sz['hi_track']:.4f}]")
            if direction in summ:
                R = summ[direction]["real"][0]
                L.append(f"Y / Z readings, {direction} (R = z0 real macro-F1 {R:.4f}; Y1 F >= R - 0.05, Y2 F <= R - 0.15, Y3 between):")
                for x in P3:
                    if x["direction"] != direction:
                        continue
                    F = x["f1"]
                    y = "1" if F >= R - 0.05 else ("2" if F <= R - 0.15 else "3")
                    L.append(f"  {x['model']:22s} {x['kind']:13s} {x['subset']:9s} F {F:.4f} track CI [{x['lo_track']:.4f}, "
                             f"{x['hi_track']:.4f}] -> " + (f"Z{y} / " if x["model"].startswith("3a") else "") + f"Y{y}")
        out = head + L
        json.dump({k: {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) for kk, vv in v.items()} for k, v in summ.items()},
                  open(f"{RES}/z0_trackCI_summary_exp.json", "w"), indent=1, default=float)
        open(f"{RES}/report_z0_trackCI_exp.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
        print("\n".join(out))
    else:
        cfg = json.load(open(f"{RES}/overnight_arm_{what}_exp.json"))
        L, summ, excl = analyse(what, cfg["runs"], f"{RES}/{what}_predictions_exp.csv")
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in summ.items()},
                  open(f"{RES}/{what}_trackCI_summary_exp.json", "w"), indent=1, default=float)
        open(f"{RES}/{what}_trackCI_exp.txt", "w", encoding="utf-8").write(
            "\n".join([f"Runs excluded: {excl or 'none'}"] + L) + "\n")
        print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
