"""Item 47 analysis (run by scripts/item47_waiter.py after the item 47 chain exits; idle priority; no model calls).
Reading of "the item 46 analysis (scripts/eval_track_ci_exp.py analyse())": the item 46 real-only procedure
(scripts/item46_analysis.py: test predictions from best.pt, 4-decimal check against results.json, then
scripts/item44_chain.py stats(): 1000 resamples of whole test tracks with default_rng(0), the same resampled tracks for
every arm, mean over seeds per resample; margin = the larger real seed population std of the two arms over their shared
seeds), because analyse() itself expects particle-free runs and there are none here.
Ln and Ld runs are evaluated with train_compact_cnn_deep.py (the run's own --width and --extra-convs) on unshifted
patches (Ld: <patch_dir>/2.250, byte copies of the z0 patches), channel statistics from the training split. References
from saved predictions: D (results/depth_aug_predictions_exp.csv) and z0 (results/z0_predictions_exp.csv, real, seeds 0
to 2) at width 32; Wn and Wd (results/width_depth_predictions_exp.csv) at width 64, original direction only.
usage: python scripts/item47_analysis.py part0 --runs P --out P
       python scripts/item47_analysis.py report --runs P --report P --pred P --part0 P1 [P2 ...]"""
import os
import sys
import json
import time

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import item44_chain as c44  # noqa: E402
import eval_track_ci_exp as ev  # noqa: E402
prio41.set_priority(prio41.IDLE)

RES = "results"
B = 1000
LAB = {"Ln": "Ln (18L, none)", "Ld": "Ld (18L, all21)", "D": "D (w32, all21)", "z0": "z0 (w32, none)",
       "Wn": "Wn (w64, none)", "Wd": "Wd (w64, all21)"}
REF = {("Ln", "original"): (0.907, "z0 original"), ("Ln", "reverse"): (0.918, "z0 reverse"),
       ("Ld", "original"): (0.899, "D original"), ("Ld", "reverse"): (0.950, "D reverse")}
REFRUN = {"z0": f"{RES}/real_tc_PETvsPS_z0_exp_s0", "D": f"{RES}/real_tc_PETvsPS_z0depthD_exp_s0",
          "Wn": f"{RES}/real_tc_PETvsPS_z0w64_exp_s0", "Wd": f"{RES}/real_tc_PETvsPS_z0depthD_w64_exp_s0"}


def clean(text):
    assert "\u2014" not in text and ("signif" + "icant") not in text.lower() and ("valid" + "ated") not in text.lower()
    return text


def predict(d, split):
    """best.pt in eval mode on one split of the run's manifest, unshifted patches, no augmentation."""
    import torch
    from torch.utils.data import DataLoader
    m = ev.module("train_compact_cnn_deep.py")
    saved = json.load(open(os.path.join(d, "results.json")))
    a = saved["args"]
    col = m.SPLIT_CONFIG[a["split"]]["split_column"]
    man = pd.read_csv(a["manifest"])
    man = man[man["class"].isin(a["classes"].split(","))]
    classes = saved["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    pdir = os.path.join(a["patch_dir"], "2.250") if a.get("depth_aug", "none") != "none" else a["patch_dir"]
    tr, df = man[man[col] == "train"].reset_index(drop=True), man[man[col] == split].reset_index(drop=True)
    mean, std = m.compute_channel_stats(tr, pdir, c2i, a["channels"])
    net = m.CompactHoloCNN(1, len(classes), a["width"], a["dropout"], extra_convs=a.get("extra_convs", 0))
    net.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cpu"))
    net.eval()
    ds = m.PatchDataset(df, pdir, c2i, channels=a["channels"], mean=mean, std=std)
    p = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=64, shuffle=False):
            p += net(x).argmax(1).tolist()
    out = df[["patch_file", "class", "track_id"]].copy()
    out["pred"] = [classes[i] for i in p]
    return out.rename(columns={"class": "true_class", "pred": "predicted_class"}), classes, saved


def n0(f, c):
    return "N0a" if f >= 0.98 else ("N0b" if f > c + 0.03 else "N0c")


def run_list(rj):
    return [(arm, direction, int(s), d) for arm in ("Ln", "Ld") for direction in ("original", "reverse")
            for s, d in sorted(rj["runs"].get(arm, {}).get(direction, {}).items(), key=lambda x: int(x[0]))]


def part0(runs_path, out):
    """Item 47 Part 0 on the finished Ln and Ld runs (same procedure as scripts/item47_part0.py; seeds taken from the
    runs file, so retry folders are handled)."""
    from sklearn.metrics import f1_score, recall_score
    if os.path.exists(out):
        raise FileExistsError(out)
    rj = json.load(open(runs_path))
    L = ["ITEM 47, PART 0 (deep): TRAINING-SET FIT OF Ln AND Ld (best.pt, eval mode, training split, unshifted, no augmentation)", "",
         f"{'run':45s} {'network':21s} {'direction':9s} {'train F1':>8s} {'recall PET':>10s} {'recall PS':>9s} {'val F1':>7s} {'test F1':>7s} {'n train':>7s}"]
    got = {}
    for arm, direction, s, d in run_list(rj):
        df, classes, saved = predict(d, "train")
        f1 = f1_score(df.true_class, df.predicted_class, labels=classes, average="macro", zero_division=0)
        rec = recall_score(df.true_class, df.predicted_class, labels=classes, average=None, zero_division=0)
        got.setdefault((arm, direction), {})[s] = f1
        L.append(f"{os.path.basename(d):45s} {LAB[arm]:21s} {direction:9s} {f1:8.4f} {rec[0]:10.3f} {rec[1]:9.3f} "
                 f"{saved['best_val_macro_f1']:7.4f} {saved['test_report']['macro avg']['f1-score']:7.4f} {len(df):7d}")
    for x in rj.get("not_run", []) + rj.get("failed", []):
        L.append(f"{x:45s} not run or failed: skipped")
    L += ["", "N0 readings (results/item47_readings.md; seed 0 against its seed-0 width-32 counterpart; descriptive, the 0.03 is arbitrary):"]
    for (arm, direction), v in got.items():
        cref, cname = REF[(arm, direction)]
        r = n0(v[0], cref) if 0 in v else "not available (seed 0 not finished)"
        L.append(f"  {LAB[arm]:21s} {direction:9s} seed 0 {v.get(0, float('nan')):.4f} against {cname} {cref:.3f}: {r}; "
                 f"mean over seeds {sorted(v)} {sum(v.values()) / len(v):.4f}")
    if not got:
        L.append("  no Ln or Ld run finished")
    L.append(f"Written {time.strftime('%Y-%m-%d %H:%M')}.")
    open(out, "w", encoding="utf-8").write(clean("\n".join(L)) + "\n")
    print("\n".join(L))


def cmp(A, B_, W, tracks):
    """Paired comparison of arm A minus arm B over their shared seeds (as scripts/item46_analysis.py)."""
    s = sorted(set(A.seed) & set(B_.seed))
    a, b = c44.stats(A[A.seed.isin(s)], tracks, W), c44.stats(B_[B_.seed.isin(s)], tracks, W)
    dd = a["boot"] - b["boot"]
    return dict(seeds=s, diff=a["f1"][0] - b["f1"][0], margin=max(a["f1"][1], b["f1"][1]),
                ci=(float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))))


def reading(c):
    above = lambda k: k in c and c[k]["diff"] > c[k]["margin"]
    below = lambda k: k in c and c[k]["diff"] < -c[k]["margin"]
    within = lambda k: k in c and abs(c[k]["diff"]) <= c[k]["margin"]
    if above("Ld - D") and above("Ld - Ln"):
        return "N1"
    if above("Ln - z0"):
        return "N2"
    if within("Ld - D") and within("Ln - z0"):
        return "N3"
    if below("Ld - D"):
        return "N4"
    return "N5"


def script_lines(path):
    pats = ("CrossEntropyLoss(", "va_f1 = ", "if va_f1 > best_f1", "bad_epochs >= args.patience", "torch.save(")
    src = open(path, encoding="utf-8").read()
    lines = [f"    {path}:{i}: {l.strip()}" for i, l in enumerate(src.splitlines(), 1) if any(p in l for p in pats)]
    loss = "class-weighted cross-entropy (nn.CrossEntropyLoss with the --class-weights file)" if "nn.CrossEntropyLoss(weight=weights)" in src else "see the lines below"
    es = ("validation macro-F1 (f1_score average='macro' on the validation split; best.pt kept at its highest value; stop after "
          "--patience epochs without a rise)") if ('va_f1 = f1_score(va_l, va_p, average="macro")' in src and "if va_f1 > best_f1" in src) else "see the lines below"
    return loss, es, lines


def report(runs_path, report_path, pred_path, part0_paths):
    from sklearn.metrics import f1_score
    for p in (report_path, pred_path):
        if os.path.exists(p):
            raise FileExistsError(p)
    rj = json.load(open(runs_path))
    runs, timing, dirs = rj["runs"], rj["timing"], rj["dirs"]
    d32 = pd.read_csv(f"{RES}/depth_aug_predictions_exp.csv")
    z32 = pd.read_csv(f"{RES}/z0_predictions_exp.csv")
    z32 = z32[(z32.kind == "real") & z32.seed.isin([0, 1, 2])].rename(columns={"patch_id": "patch_file"})
    w64 = pd.read_csv(f"{RES}/width_depth_predictions_exp.csv")
    allp, excl, res, info = [], [], {}, {}
    for direction in ("original", "reverse"):
        arms = {}
        for arm in ("Ln", "Ld"):
            P = []
            for a_, dr, s, d in [x for x in run_list(rj) if x[0] == arm and x[1] == direction]:
                df, classes, saved = predict(d, "test")
                f1 = f1_score(df.true_class, df.predicted_class, labels=classes, average="macro", zero_division=0)
                f1s = float(saved["test_report"]["macro avg"]["f1-score"])
                info[os.path.basename(d)] = dict(n_params=saved.get("n_params"), best_epoch=saved.get("best_epoch"),
                                                 epochs=len(saved["history"]))
                if round(f1, 4) != round(f1s, 4):
                    excl.append(f"{d} (recomputed {f1:.4f} vs saved {f1s:.4f})")
                    continue
                P.append(df.assign(seed=s, run=os.path.basename(d), arm=arm, direction=direction))
            if P:
                arms[arm] = pd.concat(P)
                allp.append(arms[arm])
        if not arms:
            continue
        arms["D"] = d32[(d32.arm == "D") & (d32.direction == direction)]
        arms["z0"] = z32[z32.direction == direction]
        if direction == "original":
            for k in ("Wn", "Wd"):
                arms[k] = w64[(w64.arm == k) & (w64.direction == direction)]
        arms = {k: v for k, v in arms.items() if len(v)}
        ref = set(arms["z0"].patch_file)
        same = all(set(v[v.seed == s].patch_file) == ref for v in arms.values() for s in set(v.seed))
        tracks = sorted(set(arms["z0"].track_id))
        rng = np.random.default_rng(0)
        W = np.stack([np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks)) for _ in range(B)])
        st = {k: c44.stats(v, tracks, W) for k, v in arms.items()}
        c = {}
        if "Ld" in arms:
            c["Ld - D"] = cmp(arms["Ld"], arms["D"], W, tracks)
        if "Ld" in arms and "Ln" in arms:
            c["Ld - Ln"] = cmp(arms["Ld"], arms["Ln"], W, tracks)
        if "Ln" in arms:
            c["Ln - z0"] = cmp(arms["Ln"], arms["z0"], W, tracks)
        res[direction] = dict(stats=st, cmp=c, seeds={k: sorted(set(v.seed)) for k, v in arms.items()}, same=same,
                              nt=len(tracks), npatch=len(ref), N=reading(c))
    if allp:
        pd.concat(allp)[["run", "arm", "direction", "seed", "patch_file", "track_id", "true_class", "predicted_class"]].to_csv(pred_path, index=False)

    Nd = {k: v["N"] for k, v in res.items()}
    if len(Nd) == 2:
        overall = Nd["original"] if Nd["original"] == Nd["reverse"] else f"N6 (directions disagree: original {Nd['original']}, reverse {Nd['reverse']})"
    elif len(Nd) == 1:
        (k, v), = Nd.items()
        overall = f"reduced design for one test pair: only the {k} direction finished, reading {v} on that pair"
    else:
        overall = "no Ln or Ld run finished, so no reading"
    n0l = []
    for p in part0_paths:
        if os.path.exists(p):
            lines = open(p, encoding="utf-8").read().splitlines()
            i = next((j for j, l in enumerate(lines) if l.startswith("N0 readings")), len(lines))
            n0l += [l for l in lines[i + 1:] if l.startswith("  ")] + [f"    (from {p})"]
        else:
            n0l.append(f"  {p}: not written")
    L = ["ITEM 47: DEEPER CNN (18 WEIGHTED LAYERS) WITH AND WITHOUT DEPTH AUGMENTATION, z0 EXPANDED SET, PET vs PS, REAL RUNS ONLY", "",
         "Reading of the instructions: 'the item 46 analysis (scripts/eval_track_ci_exp.py analyse())' is taken as the item 46",
         "real-only procedure (scripts/item46_analysis.py with scripts/item44_chain.py stats()), because analyse() expects",
         "particle-free runs and none were made. Ln and Ld are evaluated with train_compact_cnn_deep.py; D, z0, Wn and Wd come",
         "from their saved test predictions.", "",
         f"READINGS (results/item47_readings.md): per direction " + (", ".join(f"{k} {v}" for k, v in Nd.items()) or "none") + f"; overall {overall}.",
         "N0 readings (Part 0, training-set macro-F1 of best.pt):"] + n0l + [
         "No T reading",
         "No particle-free runs were made. Nothing here shows that the CNN identifies polymers.",
         "Margin for each comparison = the larger real seed std (population std) of the two arms over their shared seeds.", ""]
    for direction, r in res.items():
        L.append(f"{direction.upper()} DIRECTION ({r['npatch']} test patches, {r['nt']} tracks; identical test patches in every arm and seed: "
                 f"{'yes' if r['same'] else 'NO'}; seeds: " + ", ".join(f"{k} {v}" for k, v in r["seeds"].items()) + ")")
        L.append(f"  {'arm':16s} {'real macro-F1':>15s} {'track CI':>17s}  recall PET / PS [CI]                 precision PET / PS [CI]")
        for k in ("Ln", "Ld", "D", "z0", "Wn", "Wd"):
            if k not in r["stats"]:
                continue
            x = r["stats"][k]
            L.append(f"  {LAB[k]:16s} {x['f1'][0]:.4f}+/-{x['f1'][1]:.4f} [{x['ci'][0]:.4f}, {x['ci'][1]:.4f}]  "
                     f"{x['rec'][0][0][0]:.3f} [{x['rec'][0][1][0]:.3f}, {x['rec'][0][1][1]:.3f}] / {x['rec'][1][0][0]:.3f} "
                     f"[{x['rec'][1][1][0]:.3f}, {x['rec'][1][1][1]:.3f}]   {x['prec'][0][0][0]:.3f} [{x['prec'][0][1][0]:.3f}, "
                     f"{x['prec'][0][1][1]:.3f}] / {x['prec'][1][0][0]:.3f} [{x['prec'][1][1][0]:.3f}, {x['prec'][1][1][1]:.3f}]")
        for k, c in r["cmp"].items():
            v = "above" if c["diff"] > c["margin"] else ("below" if c["diff"] < -c["margin"] else "within")
            L.append(f"  {k:8s} {c['diff']:+.4f} (margin {c['margin']:.4f}, {v}; paired track CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}], seeds {c['seeds']})")
        L.append(f"  N reading for this direction: {r['N']}")
        L.append("")
    pc = {k: json.load(open(f"{d}/results.json")).get("n_params", "not saved") for k, d in REFRUN.items() if os.path.exists(f"{d}/results.json")}
    for k in ("Ln", "Ld"):
        v = [info[os.path.basename(d)]["n_params"] for a_, dr, s, d in run_list(rj) if a_ == k and os.path.basename(d) in info]
        pc[k] = v[0] if v else "no finished run"
    loss, es, lines = script_lines("train_compact_cnn_deep.py")
    loss0, es0, lines0 = script_lines("train_compact_cnn.py")
    L += ["Parameter counts: " + ", ".join(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}" for k, v in pc.items()) + ".",
          "Run times and epochs: " + (", ".join(f"{t['run']} {t['seconds']} s, {t['epochs']} epochs"
                                               + (f" (best {info[t['run']]['best_epoch']})" if t['run'] in info else "")
                                               + ("" if t["ok"] else " FAILED") for t in timing) or "none") + ".",
          f"Runs not run (stop rule 2026-10-14 10:00): {rj.get('not_run') or 'none'}. Runs failed twice: {rj.get('failed') or 'none'}.",
          f"Runs excluded by the 4-decimal check: {excl or 'none'}. Test predictions: {pred_path}.",
          "Open check V7, loss function and early-stopping metric as read from the scripts:",
          f"  train_compact_cnn_deep.py (Ln, Ld): loss {loss}; early-stopping metric {es}.", *lines,
          f"  train_compact_cnn.py (the original script): loss {loss0}; early-stopping metric {es0}.", *lines0,
          f"Written {time.strftime('%Y-%m-%d %H:%M')}"]
    open(report_path, "w", encoding="utf-8").write(clean("\n".join(L)) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    g = lambda k: sys.argv[sys.argv.index(k) + 1]
    if sys.argv[1] == "part0":
        part0(g("--runs"), g("--out"))
    else:
        report(g("--runs"), g("--report"), g("--pred"), sys.argv[sys.argv.index("--part0") + 1:])
