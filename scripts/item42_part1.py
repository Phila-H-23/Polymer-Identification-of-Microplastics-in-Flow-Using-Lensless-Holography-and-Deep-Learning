"""Item 42, part 1 (evaluation only, no training; idle priority). Starts results/report_z0_pairs_exp.txt.
1a old 561 set, PET vs PS, both directions: z0, as-built recon_s0 (V1) and V3: real and particle-free macro-F1,
   gap, per-class recall and precision (real), and paired track-bootstrap intervals for V3 - z0 and V3 - V1 real
   macro-F1. z0 and V1 test predictions are regenerated from best.pt with scripts/eval_track_ci_exp.py analyse()
   (4-decimal check against results.json); V3 uses its saved predictions and summary.
1b expanded set: per class and video, patches, tracks and the split in each direction.
1c seed ensembles (majority vote over the saved seed predictions, real only): z0 expanded and V3."""
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
import item42_status as st  # noqa: E402
import unclipped_tracks as ut  # noqa: E402  (per_track_counts, metrics, ms)

PO, RES = "pipeline_output", "results"
REPORT = f"{RES}/report_z0_pairs_exp.txt"
KEY = ["source_file", "frame", "particle_id"]
B = 1000
CL = ["PET", "PS"]
ARMS = {"z0": "z0", "V1": "recon_s0", "V3": "recon_V3"}
PRED = {"z0": f"{RES}/z0_old_predictions_PETvsPS.csv", "V1": f"{RES}/recon_s0_old_predictions_PETvsPS.csv",
        "V3": f"{RES}/recon_V3_predictions_PETvsPS.csv"}


def spec(pref):
    return {d: {k: {s: f"{RES}/{pk}_tc_PETvsPS_{pref}{'_rev' if d == 'reverse' else ''}_s{s}" for s in seeds}
                for k, pk in (("real", "real"), ("particle-free", "bg"))}
            for d, seeds in (("original", range(5)), ("reverse", range(3)))}


def old_track_map():
    """Old-set track ids (track_id_oldset of the expanded manifest), for real patch names and, through
    (source_file, frame, particle_id), for every old background manifest."""
    e = pd.read_csv(f"{PO}/manifest_PETPS_exp.csv")
    e = e[e.old_patch.astype(bool)]
    real = dict(zip(e.patch_file, e.track_id_oldset))
    bykey = e.set_index(KEY).track_id_oldset
    return real, bykey


def patched_read_csv(real, bykey):
    _read = pd.read_csv

    def read(p, *a, **k):
        df = _read(p, *a, **k)
        name = os.path.basename(str(p))
        if "track_id" not in df.columns and "patch_file" in df.columns and name.startswith("manifest_") and "_exp" not in name:
            if "background" in name:
                idx = pd.MultiIndex.from_frame(df[KEY])
                df["track_id"] = bykey.reindex(idx).values
            else:
                df["track_id"] = df.patch_file.map(real)
        return df
    return _read, read


def paired_diff(preds, a, b, direction):
    """Bootstrap over whole test tracks (default_rng(0), 1000 resamples) of real macro-F1 (mean over seeds) for arm a
    minus arm b, the same resampled tracks for both arms."""
    P = {k: v[(v.direction == direction) & (v.kind == "real")] for k, v in preds.items() if k in (a, b)}
    ids = {k: set(v.patch_id) for k, v in P.items()}
    assert ids[a] == ids[b], "test patches differ between arms"
    tracks = sorted(set(P[a].track_id) | set(P[b].track_id))
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks)) for _ in range(B)])
    boot, point = {}, {}
    for k in (a, b):
        bs, ps = [], []
        for s in sorted(set(P[k].seed)):
            C = ut.per_track_counts(P[k][P[k].seed == s], tracks, CL)
            bs.append(ut.metrics(np.einsum("bt,tij->bij", W, C))[0])
            ps.append(float(ut.metrics(C.sum(0))[0]))
        boot[k], point[k] = np.mean(bs, axis=0), np.mean(ps)
    d = boot[a] - boot[b]
    return point[a] - point[b], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(tracks), len(ids[a])


def ensemble(pred, direction, label):
    P = pred[(pred.direction == direction) & (pred.kind == "real")]
    seeds = sorted(set(P.seed))
    votes = P.groupby("patch_id").predicted_class.agg(lambda s: s.value_counts().idxmax() if s.value_counts().iloc[0] * 2 > len(s)
                                                      else sorted(s.value_counts()[s.value_counts() == s.value_counts().max()].index)[0])
    one = P[P.seed == seeds[0]].set_index("patch_id")
    E = pd.DataFrame(dict(track_id=one.track_id, true_class=one.true_class, predicted_class=votes.reindex(one.index)))
    tracks = sorted(set(E.track_id))
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks)) for _ in range(B)])
    C = ut.per_track_counts(E, tracks, CL)
    f, r, _ = ut.metrics(C.sum(0))
    bf, br, _ = ut.metrics(np.einsum("bt,tij->bij", W, C))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    per_seed = [float(ut.metrics(ut.per_track_counts(P[P.seed == s], tracks, CL).sum(0))[0]) for s in seeds]
    return (f"  {label:22s} {direction:9s} {len(seeds)} seeds  ensemble macro-F1 {float(f):.4f} [{q(bf)[0]:.4f}, {q(bf)[1]:.4f}]  "
            f"recall PET {float(r[0]):.3f} [{q(br[:, 0])[0]:.3f}, {q(br[:, 0])[1]:.3f}], PS {float(r[1]):.3f} "
            f"[{q(br[:, 1])[0]:.3f}, {q(br[:, 1])[1]:.3f}]  (single seeds: mean {np.mean(per_seed):.4f}, range "
            f"{min(per_seed):.4f} to {max(per_seed):.4f}; {len(E)} patches, {len(tracks)} tracks)")


def main():
    t0 = time.time()
    st.update("part1", status="running", started=time.strftime("%Y-%m-%d %H:%M"), pid=os.getpid())
    if os.path.exists(REPORT):
        raise FileExistsError(REPORT)
    import eval_track_ci_exp as ev
    real, bykey = old_track_map()
    _read, read = patched_read_csv(real, bykey)
    summ, excl_all, Lev = {}, {}, {}
    pd.read_csv = read
    try:
        for arm in ("z0", "V1"):
            if os.path.exists(PRED[arm]):
                raise FileExistsError(PRED[arm])
            Lev[arm], summ[arm], excl_all[arm] = ev.analyse(arm, spec(ARMS[arm]), PRED[arm])
            st.update("part1", note=f"{arm} old-set evaluation done ({time.time() - t0:.0f} s)")
    finally:
        pd.read_csv = _read
    summ["V3"] = json.load(open(f"{RES}/recon_V3_trackCI_PETvsPS.json"))
    preds = {k: pd.read_csv(v) for k, v in PRED.items()}
    L = ["Z0 VIDEO-TO-VIDEO SPREAD AND EVALUATION EXTRAS, PET vs PS (item 42, overnight). Part 1 is evaluation only.", "",
         "PART 1a. OLD 561 SET, PET vs PS: z0, as-built recon_s0 (V1), V3 (extra columns; the R and M readings of the V3 report",
         "stay as pre-registered). z0 and V1 test predictions regenerated from best.pt with scripts/eval_track_ci_exp.py analyse()",
         f"(saved to {PRED['z0']} and {PRED['V1']}); V3 from {PRED['V3']} and results/recon_V3_trackCI_PETvsPS.json.",
         f"Runs excluded by the 4-decimal check: z0 {excl_all['z0'] or 'none'}, V1 {excl_all['V1'] or 'none'}.",
         "Tracks: the old-set track ids (track_id_oldset of pipeline_output/manifest_PETPS_exp.csv; a particle-free patch carries",
         "its pair's).",
         f"{'arm':4s} {'direction':9s} {'real macro-F1':>16s} {'particle-free':>16s} {'gap':>8s}  real recall PET / PS          real precision PET / PS"]
    for d in ("original", "reverse"):
        for arm in ("z0", "V1", "V3"):
            s = summ[arm][d]
            rec, prec = s["rec"]["real"], s["prec"]["real"]
            L.append(f"{arm:4s} {d:9s} {s['real'][0]:.4f}+/-{s['real'][1]:.4f} {s['pf'][0]:.4f}+/-{s['pf'][1]:.4f} {s['gap']:+.4f}  "
                     f"{rec[0][0]:.3f}+/-{rec[0][1]:.3f} / {rec[1][0]:.3f}+/-{rec[1][1]:.3f}   "
                     f"{prec[0][0]:.3f}+/-{prec[0][1]:.3f} / {prec[1][0]:.3f}+/-{prec[1][1]:.3f}")
    L += ["Paired track bootstrap of the real macro-F1 difference. Alignment: the three arms are tested on the same old-set real",
          "test patches (same patch ids and splits, checked equal per direction); each resample draws whole test tracks once",
          "(numpy default_rng(0), 1000 resamples) and applies the same track weights to both arms; each arm's macro-F1 per",
          "resample is the mean over its seeds (5 original, 3 reverse); the interval is the 2.5 and 97.5 percentiles of the",
          "difference."]
    for d in ("original", "reverse"):
        for a, b in (("V3", "z0"), ("V3", "V1")):
            diff, lo, hi, nt, n = paired_diff(preds, a, b, d)
            L.append(f"  {d:9s} {a} - {b}: {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]  ({n} test patches, {nt} tracks)")
    # 1b
    m = pd.read_csv(f"{PO}/manifest_PETPS_exp.csv")
    m["video"] = m.source_file.str.replace("_600.mat", "", regex=False)
    L += ["", "PART 1b. EXPANDED SET (pipeline_output/manifest_PETPS_exp.csv): patches and tracks per video, and the split of each",
          "video in the original direction (split_video_holdout) and the reverse direction (split_rev).",
          f"{'class':5s} {'video':5s} {'patches':>7s} {'tracks':>6s}  original (train/val/test)   reverse (train/val/test)"]
    for (c, v), g in m.groupby(["class", "video"]):
        so, sr = g.split_video_holdout.value_counts(), g.split_rev.value_counts()
        fmt = lambda x: "/".join(str(int(x.get(k, 0))) for k in ("train", "val", "test"))
        role = lambda x: "TEST" if x.get("test", 0) else "train"
        L.append(f"{c:5s} {v:5s} {len(g):7d} {g.track_id.nunique():6d}  {role(so):5s} {fmt(so):15s}      {role(sr):5s} {fmt(sr)}")
    L.append("Test videos: original PET2 + PS1, reverse PET1 + PS2.")
    # 1c
    z0e = pd.read_csv(f"{RES}/z0_predictions_exp.csv")
    L += ["", "PART 1c. SEED ENSEMBLES (extra rows, not a pre-registered arm): majority vote over the saved seed predictions of",
          "the real classifier, same test sets; track intervals from 1000 resamples of whole test tracks (default_rng(0)).",
          "With 5 or 3 seeds and two classes no vote is tied."]
    for d in ("original", "reverse"):
        L.append(ensemble(z0e, d, "z0 expanded set"))
        L.append(ensemble(preds["V3"], d, "V3 old set"))
    L += ["", f"Part 1 time {time.time() - t0:.0f} s.", ""]
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    open(REPORT, "w", encoding="utf-8").write(text + "\n")
    st.update("part1", status="done", finished=time.strftime("%Y-%m-%d %H:%M"), note=f"{time.time() - t0:.0f} s",
              files=[REPORT, PRED["z0"], PRED["V1"], "scripts/item42_part1.py"])
    print(text)


if __name__ == "__main__":
    main()
