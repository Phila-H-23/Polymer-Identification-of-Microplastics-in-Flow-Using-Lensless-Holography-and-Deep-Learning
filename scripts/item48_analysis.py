"""Item 48 analysis (no model calls; above-normal priority after the imports). Modes:
fitcheck  Test A: last.pt (final-epoch weights) of the two fit-check runs in eval mode on the unshifted training patches
          (macro-F1, per-class recall) and test patches (macro-F1, descriptive), final training loss from results.json,
          FIT reading. Writes results/item48_fit_check.txt.
part0     best.pt of every finished WLn and WLd run in eval mode on the unshifted training patches; per run and mean,
          no letter. Writes results/item48_part0_trainfit.txt.
report    the item 47 real-only analysis procedure (scripts/item46_analysis.py with scripts/item44_chain.py stats(), as
          in scripts/item47_analysis.py: test predictions from best.pt with train_compact_cnn_deep.py, 4-decimal check,
          1000 resamples of whole test tracks with default_rng(0), the same resampled tracks for every arm, mean over
          seeds per resample; margin = the larger real seed population std of the two arms over their shared seeds).
          References from saved test predictions: Ln and Ld (results/layers_depth_predictions_exp.csv), D and z0, and,
          original direction only, Wn and Wd. Writes results/report_width_layers_exp.txt and
          results/width_layers_predictions_exp.csv.
usage: python scripts/item48_analysis.py fitcheck --runs P --out P
       python scripts/item48_analysis.py part0 --runs P --out P
       python scripts/item48_analysis.py report --runs P --report P --pred P --fit P --part0 P"""
import os
import sys
import json
import time

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
import item47_analysis as a47  # noqa: E402  (cmp, clean, c44, ev; its imports set idle priority)
prio41.set_priority(0x8000)
np, pd, c44, ev = a47.np, a47.pd, a47.c44, a47.ev

RES = "results"
B = 1000
ARMS = ("WLn", "WLd")
LAB = {"WLn": "WLn (w64 18L, none)", "WLd": "WLd (w64 18L, all21)", "Ln": "Ln (w32 18L, none)",
       "Ld": "Ld (w32 18L, all21)", "D": "D (w32 10L, all21)", "z0": "z0 (w32 10L, none)",
       "Wn": "Wn (w64 10L, none)", "Wd": "Wd (w64 10L, all21)"}
REFRUN = {"z0": f"{RES}/real_tc_PETvsPS_z0_exp_s0", "D": f"{RES}/real_tc_PETvsPS_z0depthD_exp_s0",
          "Wn": f"{RES}/real_tc_PETvsPS_z0w64_exp_s0", "Wd": f"{RES}/real_tc_PETvsPS_z0depthD_w64_exp_s0",
          "Ln": f"{RES}/real_tc_PETvsPS_z0deep18_exp_s0", "Ld": f"{RES}/real_tc_PETvsPS_z0deep18depthD_exp_s0"}


def predict(d, split, ckpt="best.pt"):
    """Weights ckpt in eval mode on one split of the run's manifest, unshifted patches, no augmentation, channel
    statistics from the training split (as scripts/item47_analysis.py predict(), with the checkpoint as a parameter)."""
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
    net.load_state_dict(torch.load(os.path.join(d, ckpt), map_location="cpu"))
    net.eval()
    ds = m.PatchDataset(df, pdir, c2i, channels=a["channels"], mean=mean, std=std)
    p = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=64, shuffle=False):
            p += net(x).argmax(1).tolist()
    out = df[["patch_file", "class", "track_id"]].copy()
    out["pred"] = [classes[i] for i in p]
    return out.rename(columns={"class": "true_class", "pred": "predicted_class"}), classes, saved


def f1r(df, classes):
    from sklearn.metrics import f1_score, recall_score
    return (f1_score(df.true_class, df.predicted_class, labels=classes, average="macro", zero_division=0),
            recall_score(df.true_class, df.predicted_class, labels=classes, average=None, zero_division=0))


def wl_runs(rj):
    return [(arm, direction, int(s), d) for direction in ("original", "reverse") for arm in ARMS
            for s, d in sorted(rj["runs"].get(arm, {}).get(direction, {}).items(), key=lambda x: int(x[0]))]


def write(path, L):
    if os.path.exists(path):
        raise FileExistsError(path)
    open(path, "w", encoding="utf-8").write(a47.clean("\n".join(L)) + "\n")
    print("\n".join(L))


def fitcheck(runs_path, out):
    if os.path.exists(out):
        raise FileExistsError(out)
    A = json.load(open(runs_path))["runs"].get("A", {})
    L = ["ITEM 48, TEST A: FIT CHECK (results/item48_readings.md)", "",
         "train_compact_cnn_fit.py, width 32, original direction, seed 0, --depth-aug none, --dropout 0 --weight-decay 0",
         "--no-aug --save-last --patience 1000 --epochs 150 (early stopping off, cosine schedule over the 150 epochs), every",
         "other argument as results/real_tc_PETvsPS_z0_exp_s0. Scored from last.pt (final-epoch weights) in eval mode on the",
         "unshifted patches. The test macro-F1 is descriptive only.", "",
         f"{'run':36s} {'layers':>6s} {'parameters':>10s} {'epochs':>6s} {'final train loss':>16s} {'train F1 (log)':>14s} "
         f"{'train F1 (eval)':>15s} {'recall PET':>10s} {'recall PS':>9s} {'test F1':>7s}"]
    f = {}
    for key in ("10", "18"):
        d = A.get(key)
        if not d or not os.path.exists(f"{d}/last.pt"):
            L.append(f"{key}-layer network: not run or last.pt missing")
            continue
        saved = json.load(open(f"{d}/results.json"))
        h = saved["history"]
        tr, classes, _ = predict(d, "train", "last.pt")
        te, _, _ = predict(d, "test", "last.pt")
        (f1, rec), (tf1, _) = f1r(tr, classes), f1r(te, classes)
        f[key] = f1
        L.append(f"{os.path.basename(d):36s} {key:>6s} {saved['n_params']:>10,d} {len(h):>6d} {h[-1]['train_loss']:>16.4f} "
                 f"{h[-1]['train_macro_f1']:>14.4f} {f1:>15.4f} {rec[0]:>10.3f} {rec[1]:>9.3f} {tf1:>7.4f}")
    if "10" in f and f["10"] >= 0.98:
        R = "FITa"
    elif "10" in f and "18" in f:
        R = "FITb" if f["18"] >= 0.98 else "FITc"
    else:
        R = "incomplete (a run is missing)"
    L += ["", "train F1 (log) = the script's own training macro-F1 of the final epoch (train mode, during the epoch); train F1",
          "(eval) = last.pt in eval mode, the number the reading uses.",
          "FIT readings, first that holds: FITa the 10-layer network reaches training macro-F1 of at least 0.98; FITb the",
          "10-layer network is below 0.98 and the 18-layer network is at least 0.98; FITc neither reaches 0.98. Guess written",
          "before the run: FITa.", f"Reading: {R}", f"Written {time.strftime('%Y-%m-%d %H:%M')}."]
    write(out, L)


def part0(runs_path, out):
    rj = json.load(open(runs_path))
    L = ["ITEM 48, PART 0: TRAINING-SET FIT OF THE WL RUNS (best.pt, eval mode, training split, unshifted, no augmentation;",
         "no letter)", "",
         f"{'run':40s} {'arm':20s} {'direction':9s} {'train F1':>8s} {'recall PET':>10s} {'recall PS':>9s} {'val F1':>7s} {'test F1':>7s} {'n train':>7s}"]
    got = {}
    for arm, direction, s, d in wl_runs(rj):
        df, classes, saved = predict(d, "train")
        f1, rec = f1r(df, classes)
        got.setdefault((arm, direction), []).append(f1)
        L.append(f"{os.path.basename(d):40s} {LAB[arm]:20s} {direction:9s} {f1:8.4f} {rec[0]:10.3f} {rec[1]:9.3f} "
                 f"{saved['best_val_macro_f1']:7.4f} {saved['test_report']['macro avg']['f1-score']:7.4f} {len(df):7d}")
    L += ["", "Means over the finished seeds:"] + [f"  {LAB[a]:20s} {dr:9s} {sum(v) / len(v):.4f} ({len(v)} runs)" for (a, dr), v in got.items()]
    if not got:
        L.append("  no WL run finished")
    L.append(f"Written {time.strftime('%Y-%m-%d %H:%M')}.")
    write(out, L)


def reading(c):
    above = lambda k: k in c and c[k]["diff"] > c[k]["margin"]
    below = lambda k: k in c and c[k]["diff"] < -c[k]["margin"]
    within = lambda k: k in c and abs(c[k]["diff"]) <= c[k]["margin"]
    if above("WLd - D") and above("WLd - WLn"):
        return "O1"
    if above("WLn - z0"):
        return "O2"
    if within("WLd - D") and within("WLn - z0"):
        return "O3"
    if below("WLd - D"):
        return "O4"
    return "O5"


def lines_of(path, start, stop=None):
    if not os.path.exists(path):
        return [f"  {path}: not written"]
    L = open(path, encoding="utf-8").read().splitlines()
    i = next((j for j, l in enumerate(L) if l.startswith(start)), 0)
    j = next((j for j, l in enumerate(L) if stop and l.startswith(stop) and j > i), len(L))
    return ["  " + l for l in L[i:j] if l.strip()] + [f"    (from {path})"]


def report(runs_path, report_path, pred_path, fit_path, part0_path):
    for p in (report_path, pred_path):
        if os.path.exists(p):
            raise FileExistsError(p)
    rj = json.load(open(runs_path))
    d32 = pd.read_csv(f"{RES}/depth_aug_predictions_exp.csv")
    z32 = pd.read_csv(f"{RES}/z0_predictions_exp.csv")
    z32 = z32[(z32.kind == "real") & z32.seed.isin([0, 1, 2])].rename(columns={"patch_id": "patch_file"})
    w64 = pd.read_csv(f"{RES}/width_depth_predictions_exp.csv")
    l47 = pd.read_csv(f"{RES}/layers_depth_predictions_exp.csv")
    allp, excl, res, info = [], [], {}, {}
    for direction in ("original", "reverse"):
        arms = {}
        for arm in ARMS:
            P = []
            for a_, dr, s, d in [x for x in wl_runs(rj) if x[0] == arm and x[1] == direction]:
                df, classes, saved = predict(d, "test")
                f1, _ = f1r(df, classes)
                f1s = float(saved["test_report"]["macro avg"]["f1-score"])
                info[os.path.basename(d)] = dict(n_params=saved.get("n_params"), best_epoch=saved.get("best_epoch"))
                if round(f1, 4) != round(f1s, 4):
                    excl.append(f"{d} (recomputed {f1:.4f} vs saved {f1s:.4f})")
                    continue
                P.append(df.assign(seed=s, run=os.path.basename(d), arm=arm, direction=direction))
            if P:
                arms[arm] = pd.concat(P)
                allp.append(arms[arm])
        if not arms:
            continue
        for k in ("Ln", "Ld"):
            arms[k] = l47[(l47.arm == k) & (l47.direction == direction)]
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
        c, x = {}, {}
        for a_, b_, tgt in (("WLd", "D", c), ("WLd", "WLn", c), ("WLn", "z0", c),
                            ("WLn", "Ln", x), ("WLd", "Ld", x), ("WLn", "Wn", x), ("WLd", "Wd", x)):
            if a_ in arms and b_ in arms:
                tgt[f"{a_} - {b_}"] = a47.cmp(arms[a_], arms[b_], W, tracks)
        res[direction] = dict(stats=st, cmp=c, desc=x, seeds={k: sorted(set(v.seed)) for k, v in arms.items()}, same=same,
                              nt=len(tracks), npatch=len(ref), O=reading(c))
    if allp:
        pd.concat(allp)[["run", "arm", "direction", "seed", "patch_file", "track_id", "true_class", "predicted_class"]].to_csv(pred_path, index=False)

    Od = {k: v["O"] for k, v in res.items()}
    if len(Od) == 2:
        overall = Od["original"] if Od["original"] == Od["reverse"] else f"O6 (the directions give different letters: original {Od['original']}, reverse {Od['reverse']})"
    elif len(Od) == 1:
        (k, v), = Od.items()
        overall = f"reduced design for one test pair: only the {k} direction finished, reading {v} on that pair"
    else:
        overall = "no WL run finished, so no reading"
    fitR = next((l.strip() for l in open(fit_path, encoding="utf-8") if l.startswith("Reading:")), "Reading: not available") if os.path.exists(fit_path) else "Reading: not available (Test A file not written)"
    L = ["ITEM 48: (A) FIT CHECK AND (B) WIDTH 64 WITH 18 LAYERS, z0 EXPANDED SET, PET vs PS, REAL RUNS ONLY", "",
         "Reading of the instructions: the item 47 real-only analysis procedure (scripts/item46_analysis.py with",
         "scripts/item44_chain.py stats(), as implemented in scripts/item47_analysis.py). WLn and WLd are evaluated from best.pt",
         "with train_compact_cnn_deep.py; Ln, Ld, D, z0, Wn and Wd come from their saved test predictions.", "",
         "READINGS (results/item48_readings.md): Part 1 per direction " + (", ".join(f"{k} {v}" for k, v in Od.items()) or "none")
         + f"; overall {overall}.", f"Test A {fitR}", "",
         "TEST A (fit check, results/item48_fit_check.txt):"] + lines_of(fit_path, "run ", "train F1 (log)") + [
         "", "PART 0 (training-set macro-F1 of the WL best.pt files, no letter):"] + lines_of(part0_path, "run ", "Written") + [
         "", "No T reading",
         "No particle-free runs were made. Nothing here shows that the CNN identifies polymers.",
         "Margin for each comparison = the larger real seed std (population std) of the two arms over their shared seeds.", ""]
    for direction, r in res.items():
        L.append(f"{direction.upper()} DIRECTION ({r['npatch']} test patches, {r['nt']} tracks; identical test patches in every arm and seed: "
                 f"{'yes' if r['same'] else 'NO'}; seeds: " + ", ".join(f"{k} {v}" for k, v in r["seeds"].items()) + ")")
        L.append(f"  {'arm':21s} {'real macro-F1':>15s} {'track CI':>17s}  recall PET / PS [CI]                 precision PET / PS [CI]")
        for k in ("WLn", "WLd", "Ln", "Ld", "D", "z0", "Wn", "Wd"):
            if k not in r["stats"]:
                continue
            s_ = r["stats"][k]
            L.append(f"  {LAB[k]:21s} {s_['f1'][0]:.4f}+/-{s_['f1'][1]:.4f} [{s_['ci'][0]:.4f}, {s_['ci'][1]:.4f}]  "
                     f"{s_['rec'][0][0][0]:.3f} [{s_['rec'][0][1][0]:.3f}, {s_['rec'][0][1][1]:.3f}] / {s_['rec'][1][0][0]:.3f} "
                     f"[{s_['rec'][1][1][0]:.3f}, {s_['rec'][1][1][1]:.3f}]   {s_['prec'][0][0][0]:.3f} [{s_['prec'][0][1][0]:.3f}, "
                     f"{s_['prec'][0][1][1]:.3f}] / {s_['prec'][1][0][0]:.3f} [{s_['prec'][1][1][0]:.3f}, {s_['prec'][1][1][1]:.3f}]")
        for title, cc in (("Comparisons for the O reading:", r["cmp"]), ("Without a letter:", r["desc"])):
            L.append(f"  {title}")
            for k, c in cc.items():
                v = "above" if c["diff"] > c["margin"] else ("below" if c["diff"] < -c["margin"] else "within")
                L.append(f"    {k:10s} {c['diff']:+.4f} (margin {c['margin']:.4f}, {v}; paired track CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}], seeds {c['seeds']})")
        L.append(f"  O reading for this direction: {r['O']}")
        L.append("")
    pc = {k: json.load(open(f"{d}/results.json")).get("n_params", "not saved") for k, d in REFRUN.items() if os.path.exists(f"{d}/results.json")}
    for k in ARMS:
        v = [info[os.path.basename(d)]["n_params"] for a_, dr, s, d in wl_runs(rj) if a_ == k and os.path.basename(d) in info]
        pc[k] = v[0] if v else "no finished run"
    for key, d in rj["runs"].get("A", {}).items():
        if os.path.exists(f"{d}/results.json"):
            pc[f"Test A {key} layers"] = json.load(open(f"{d}/results.json")).get("n_params")
    tm = rj.get("timing", [])
    L += ["Parameter counts: " + ", ".join(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}" for k, v in pc.items()) + ".",
          "Run times and epochs: " + (", ".join(f"{t['run']} {t['seconds']} s, {t['epochs']} epochs"
                                               + (f" (best {info[t['run']]['best_epoch']})" if t['run'] in info else "")
                                               + ("" if t["ok"] else " FAILED") for t in tm) or "none") + ".",
          "Training processes ran at above-normal priority with Windows power throttling off (results/logs_item48/chain.log);",
          "run times are not comparable with items 44 to 47, which ran at idle or mixed priority.",
          f"Runs not run (stop rule 2026-10-14 22:00): {rj.get('not_run') or 'none'}. Runs failed twice: {rj.get('failed') or 'none'}.",
          f"Runs excluded by the 4-decimal check: {excl or 'none'}. Test predictions: {pred_path}.",
          f"Written {time.strftime('%Y-%m-%d %H:%M')}"]
    write(report_path, L)


if __name__ == "__main__":
    g = lambda k: sys.argv[sys.argv.index(k) + 1]
    if sys.argv[1] == "fitcheck":
        fitcheck(g("--runs"), g("--out"))
    elif sys.argv[1] == "part0":
        part0(g("--runs"), g("--out"))
    else:
        report(g("--runs"), g("--report"), g("--pred"), g("--fit"), g("--part0"))
