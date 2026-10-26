"""Subset re-analysis from SAVED per-patch test predictions (read-only; no model is loaded, nothing is trained).
Used for (1) z0, expanded set, PET vs PS, unclipped test patches only -> results/report_z0_unclipped_exp.txt
(python scripts/unclipped_tracks.py), and (2) the recon_V3 arm (called from scripts/stage3_V3_report.py).
For a set of dropped test patches it recomputes, per direction, exactly as scripts/eval_track_ci_exp.py analyse() does
on the full set: macro-F1 per seed (real and particle-free), mean and population std, gap, floor = larger seed std,
chance = mean macro-F1 of 1000 random label draws (numpy default_rng(0), training class proportions) on the reduced
real test labels, per-class recall, and the track bootstrap: tracks = all track ids left in the reduced real and
particle-free test sets (a particle-free patch carries the track id of its paired real patch), 1000 resamples of whole
tracks (default_rng(0)), the same resampled tracks for both classifiers, mean over seeds per resample, 2.5 and 97.5
percentiles. Reading (fixed in the task before this was run): U1 if gap > floor and real > chance + 0.10 in both
directions; U2 otherwise."""
import os
import sys
import statistics

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

KEY = ["source_file", "frame", "particle_id"]
B = 1000
H_DEM, W_DEM = 1024, 1224


def clipped(x, y):
    """Part 3b / extract_PETPS_exp.py definition: the 64 x 64 crop at the detection centre leaves the frame."""
    return (y - 32 < 0) or (y + 32 > H_DEM) or (x - 32 < 0) or (x + 32 > W_DEM)


def per_track_counts(df, tracks, classes):
    ti = {t: i for i, t in enumerate(tracks)}
    ci = {c: i for i, c in enumerate(classes)}
    C = np.zeros((len(tracks), len(classes), len(classes)))
    for t, y, p in zip(df.track_id, df.true_class, df.predicted_class):
        C[ti[t], ci[y], ci[p]] += 1
    return C


def metrics(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    rows, cols = cm.sum(-1), cm.sum(-2)
    rec = np.divide(tp, rows, out=np.zeros_like(tp), where=rows > 0)
    prec = np.divide(tp, cols, out=np.zeros_like(tp), where=cols > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return f1.mean(-1), rec, prec


def ms(v):
    return (statistics.mean(v), statistics.pstdev(v)) if len(v) > 1 else (v[0], float("nan"))


def chance(train_labels, test_labels, classes):
    p = pd.Series(train_labels).value_counts(normalize=True).reindex(classes).values
    rng = np.random.default_rng(0)
    return float(np.mean([f1_score(test_labels, rng.choice(classes, size=len(test_labels), p=p), labels=classes,
                                   average="macro", zero_division=0) for _ in range(B)]))


def analyse(pred, direction, train_labels, test_labels, drop_real=(), drop_pf=(), classes=("PET", "PS")):
    """pred: saved predictions (run, kind, direction, seed, patch_id, track_id, true_class, predicted_class).
    test_labels: the reduced real test labels in manifest order (as eval_track_ci_exp.chance draws them)."""
    classes = list(classes)
    P = pred[pred.direction == direction]
    sub = {"real": P[(P.kind == "real") & ~P.patch_id.isin(set(drop_real))],
           "particle-free": P[(P.kind == "particle-free") & ~P.patch_id.isin(set(drop_pf))]}
    seeds = sorted(set(P.seed))
    tracks = sorted(set(sub["real"].track_id) | set(sub["particle-free"].track_id))
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks)) for _ in range(B)])
    out = dict(seeds=seeds, n_tracks=len(tracks))
    for kind, df in sub.items():
        f1s, recs, boot_f1, boot_rec = [], [], [], []
        for s in seeds:
            d = df[df.seed == s]
            C = per_track_counts(d, tracks, classes)
            f, r, _ = metrics(C.sum(0))
            f1s.append(float(f))
            recs.append(r)
            bf, br, _ = metrics(np.einsum("bt,tij->bij", W, C))
            boot_f1.append(bf)
            boot_rec.append(br)
        out[kind] = dict(f1=ms(f1s), f1_seeds=f1s, rec=[ms([float(r[i]) for r in recs]) for i in range(len(classes))],
                         boot_f1=np.mean(boot_f1, axis=0), boot_rec=np.mean(boot_rec, axis=0),
                         n=int(len(df) / len(seeds)), per_class_n={c: int((df[df.seed == seeds[0]].true_class == c).sum()) for c in classes})
    assert len(test_labels) == out["real"]["n"], (len(test_labels), out["real"]["n"])
    out["chance"] = chance(train_labels, test_labels, classes)
    out["gap"] = out["real"]["f1"][0] - out["particle-free"]["f1"][0]
    out["floor"] = max(out["real"]["f1"][1], out["particle-free"]["f1"][1])
    gb = out["real"]["boot_f1"] - out["particle-free"]["boot_f1"]
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    out["ci"] = dict(real=q(out["real"]["boot_f1"]), pf=q(out["particle-free"]["boot_f1"]), gap=q(gb),
                     rec={k: [q(out[k]["boot_rec"][:, i]) for i in range(len(classes))] for k in ("real", "particle-free")})
    out["U_cond"] = bool(out["gap"] > out["floor"] and out["real"]["f1"][0] > out["chance"] + 0.10)
    out["classes"] = classes
    return out


def dropped_counts(pred, direction, drop_real, drop_pf, classes=("PET", "PS")):
    P = pred[(pred.direction == direction) & (pred.seed == pred[pred.direction == direction].seed.min())]
    out = {}
    for c in classes:
        rr, pp = P[(P.kind == "real") & (P.true_class == c)], P[(P.kind == "particle-free") & (P.true_class == c)]
        t_full = set(rr.track_id) | set(pp.track_id)
        t_left = set(rr[~rr.patch_id.isin(set(drop_real))].track_id) | set(pp[~pp.patch_id.isin(set(drop_pf))].track_id)
        out[c] = dict(real=int(rr.patch_id.isin(set(drop_real)).sum()), real_of=len(rr), pf=int(pp.patch_id.isin(set(drop_pf)).sum()),
                      pf_of=len(pp), tracks=len(t_full - t_left), tracks_of=len(t_full))
    return out


def table(name, full, red, drops, label_red):
    """Lines: reduced set beside the full set, per direction."""
    L = []
    for d in ("original", "reverse"):
        f, r = full[d], red[d]
        L += [f"{name}, {d} direction ({len(r['seeds'])} seeds; {r['n_tracks']} test tracks left of {f['n_tracks']})",
              f"  {'':22s} {'real macro-F1':>27s} {'particle-free macro-F1':>27s} {'gap':>26s} {'floor':>7s} {'chance':>7s}  U condition"]
        for lab, x in ((label_red, r), ("full test set", f)):
            L.append(f"  {lab:22s} {x['real']['f1'][0]:.4f}+/-{x['real']['f1'][1]:.4f} [{x['ci']['real'][0]:.4f}, {x['ci']['real'][1]:.4f}] "
                     f"{x['particle-free']['f1'][0]:.4f}+/-{x['particle-free']['f1'][1]:.4f} [{x['ci']['pf'][0]:.4f}, {x['ci']['pf'][1]:.4f}] "
                     f"{x['gap']:+.4f} [{x['ci']['gap'][0]:+.4f}, {x['ci']['gap'][1]:+.4f}] {x['floor']:.4f} {x['chance']:.4f}  "
                     f"{'met' if x['U_cond'] else 'not met'}")
        L.append(f"  per-class recall (mean +/- std over seeds [track 95% interval]), {label_red} | full:")
        for k in ("real", "particle-free"):
            for i, c in enumerate(r["classes"]):
                rr, ff = r[k]["rec"][i], f[k]["rec"][i]
                L.append(f"    {c:4s} {k:13s} {rr[0]:.3f}+/-{rr[1]:.3f} [{r['ci']['rec'][k][i][0]:.3f}, {r['ci']['rec'][k][i][1]:.3f}] | "
                         f"{ff[0]:.3f}+/-{ff[1]:.3f} [{f['ci']['rec'][k][i][0]:.3f}, {f['ci']['rec'][k][i][1]:.3f}]")
        L.append("  dropped from the test set: " + "; ".join(
            f"{c}: real {v['real']} of {v['real_of']}, particle-free {v['pf']} of {v['pf_of']}, tracks gone {v['tracks']} of {v['tracks_of']}"
            for c, v in drops[d].items()))
    return L


def same_as_saved(x, d, path="results/z0_trackCI_summary_exp.json"):
    import json
    s = json.load(open(path))[d]
    mine = [x["real"]["f1"][0], x["real"]["f1"][1], x["particle-free"]["f1"][0], x["particle-free"]["f1"][1], x["gap"],
            x["floor"], x["chance"], *x["ci"]["real"], *x["ci"]["pf"], *x["ci"]["gap"]]
    saved = [s["real"][0], s["real"][1], s["pf"][0], s["pf"][1], s["gap"], s["floor"], s["chance"], s["real_ci"][1],
             s["real_ci"][2], s["pf_ci"][1], s["pf_ci"][2], s["gap_ci"][1], s["gap_ci"][2]]
    return all(round(a, 4) == round(b, 4) for a, b in zip(mine, saved))


def z0_main():
    OUT = "results/report_z0_unclipped_exp.txt"
    if os.path.exists(OUT):
        raise FileExistsError(OUT)
    pred = pd.read_csv("results/z0_predictions_exp.csv")
    man = {"original": pd.read_csv("pipeline_output/manifest_recon_s0_z0_PETPS_exp.csv"),
           "reverse": pd.read_csv("pipeline_output/manifest_recon_s0_z0_PETPS_rev_exp.csv")}
    bg = {"original": pd.read_csv("pipeline_output/manifest_background_recon_s0_z0_PETPS_exp.csv"),
          "reverse": pd.read_csv("pipeline_output/manifest_background_recon_s0_z0_PETPS_rev_exp.csv")}
    full, red, drops, info = {}, {}, {}, []
    for d in ("original", "reverse"):
        m, b = man[d], bg[d]
        assert not m.duplicated(KEY).any() and not b.duplicated(KEY).any()
        fn = np.array([clipped(x, y) for x, y in zip(m.x, m.y)])
        assert (fn == m.clipped.values).all(), "clip flag differs from the Part 3b definition"
        clip_real = set(m.patch_file[m.clipped])
        pairs = b.merge(m[KEY + ["patch_file"]], on=KEY, suffixes=("", "_real"))
        assert len(pairs) == len(b)
        clip_pf = set(pairs.patch_file[pairs.patch_file_real.isin(clip_real)])
        train = m[m.split_video_holdout == "train"]["class"].values
        te = m[m.split_video_holdout == "test"]
        full[d] = analyse(pred, d, train, te["class"].values)
        red[d] = analyse(pred, d, train, te[~te.patch_file.isin(clip_real)]["class"].values, clip_real, clip_pf)
        drops[d] = dropped_counts(pred, d, clip_real, clip_pf)
        test_ids = set(pred[(pred.direction == d) & (pred.kind == "real")].patch_id)
        info.append(f"{d}: {len(clip_real & test_ids)} clipped real test patches and their {len(clip_pf & set(pred[(pred.direction == d) & (pred.kind == 'particle-free')].patch_id))} "
                    f"paired particle-free test patches dropped")
    U = "U1" if all(red[d]["U_cond"] for d in red) else "U2"
    L = ["z0, EXPANDED SET, PET vs PS: UNCLIPPED TEST PATCHES ONLY (read-only, from the saved per-patch predictions "
         "results/z0_predictions_exp.csv; nothing retrained)", "",
         "Runs: the z0 expanded-set runs results/{real,bg}_tc_PETvsPS_z0_exp[_rev]_s<seed>, original seeds 0 to 4, reverse seeds 0 to 2",
         "(the runs of results/report_z0_trackCI_exp.txt).",
         "Dropped: every real test patch clipped at a frame edge (the 64 x 64 crop at the detection centre leaves the 1024 x 1224",
         "frame; the 'clipped' column of the manifest, the same definition as the Part 3b scalar table, checked equal to the",
         "function), and, to keep the real / particle-free pairing of the existing track bootstrap, every particle-free test",
         "patch whose paired real patch (same source_file, frame and particle_id) was dropped. The bootstrap then resamples the",
         "tracks left in the two reduced test sets (a particle-free patch carries its pair's track id), the same resampled",
         "tracks for both classifiers, exactly as scripts/eval_track_ci_exp.py does on the full set.",
         "  " + "; ".join(info),
         "Chance: 1000 random label draws (default_rng(0)) with the training class proportions, scored on the reduced real",
         "test labels. Floor: larger population std over seeds of the real and particle-free macro-F1.",
         "Reading (fixed before this was run): U1 if gap > floor and real > chance + 0.10 in both directions; U2 otherwise.", ""]
    L += table("z0", full, red, drops, "unclipped only")
    L += ["", f"Reading: {U} (original: gap {red['original']['gap']:+.4f} vs floor {red['original']['floor']:.4f}, real "
              f"{red['original']['real']['f1'][0]:.4f} vs chance + 0.10 = {red['original']['chance'] + 0.10:.4f}; reverse: gap "
              f"{red['reverse']['gap']:+.4f} vs floor {red['reverse']['floor']:.4f}, real {red['reverse']['real']['f1'][0]:.4f} vs "
              f"chance + 0.10 = {red['reverse']['chance'] + 0.10:.4f})",
          "Check: the full-set rows recompute results/z0_trackCI_summary_exp.json from the same predictions with the same "
          "procedure; equal to 4 decimals: " + ", ".join(f"{d} {same_as_saved(full[d], d)}" for d in ("original", "reverse"))]
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    open(OUT, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return U


if __name__ == "__main__":
    z0_main()
