"""Item 42, part 2 (detached, idle priority; this process and every child): z0, expanded set, PET vs PS, video-to-video
spread. Every (PET video, PS video) test pair other than the original (PET2, PS1) and reverse (PET1, PS2) ones: 6 new
pairs. Splits built the way the _rev fold was built (scripts/assemble_PETPS_exp.py + scripts/tracks_PETPS_exp.py):
test = the pair's two videos; a training video that was a training video in the original direction keeps its
original-direction train / val labels; a training video that was a test video in the original direction (PET2, PS1)
gets validation by whole tracks (track_val, numpy default_rng(0), 0.175 of that video's patches), falling back to whole
frames (frame_val, same fraction and seed) if its largest track exceeds 15% of the class's training patches; a
particle-free patch follows its real partner's split (same source_file, frame, particle_id). Class weights
n_train / (2 x n_c) from the real training split, 4 decimals. The builder is checked to reproduce the stored original
and reverse splits and weights exactly before any pair is built.
Runs: train_compact_cnn.py (unchanged) with every setting of the z0 expanded runs passed explicitly (read from
results/{real,bg}_tc_PETvsPS_z0_exp_s0/results.json); only --manifest, --class-weights and --out differ. One pair at
a time: real seeds 0 to 2, then particle-free seeds 0 to 2; a failed run is retried once, then skipped. A pair is
started only if it can finish by 07:30 at the rate of the previous z0 runs (median run time x 6); no run starts
after 07:30. After each pair: scripts/eval_track_ci_exp.py analyse() -> section appended to
results/report_z0_pairs_exp.txt. Then the W reading and the summary at the top of results/item42_status.md.
usage: python scripts/item42_chain.py | test"""
import os
import sys
import json
import time
import shutil
import ctypes
import datetime as dt
import statistics
import subprocess
import traceback

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import item42_status as st  # noqa: E402

PO, RES = "pipeline_output", "results"
KEY = ["source_file", "frame", "particle_id"]
REPORT = f"{RES}/report_z0_pairs_exp.txt"
LOGD = f"{RES}/logs_item42"
IDLE_FLAG = 0x40
DEADLINE = dt.datetime(2026, 10, 9, 7, 30)
PET, PS = ["PET1", "PET2", "PET3", "PET4"], ["PS1", "PS2"]
ORIG_TRAIN = {"PET1", "PET3", "PET4", "PS2"}  # training videos of the original direction
EXISTING = {("PET2", "PS1"): ("original", 0.9052), ("PET1", "PS2"): ("reverse", 0.8477)}
NEW_PAIRS = [(a, b) for a in PET for b in PS if (a, b) not in EXISTING]
CFG = dict(test=False, out_prefix="", report=REPORT)


def log(msg):
    os.makedirs(LOGD, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(f"{LOGD}/chain{'_test' if CFG['test'] else ''}.log", "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


def status(key, **kw):
    if not CFG["test"]:
        st.update(key, **kw)


def append_report(lines):
    text = "\n".join(lines)
    assert "\u2014" not in text and "significant" not in text.lower()
    with open(CFG["report"], "a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def avail_mb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 20


def gate(label):
    """Wait for 1.5 GB of free RAM and 20 GB of disk; give up (return False) at the 07:30 deadline."""
    t0, t_log, waited = time.time(), 0.0, False
    while True:
        if dt.datetime.now() >= DEADLINE:
            return False
        a, free = avail_mb(), shutil.disk_usage(D).free / 2 ** 30
        if a >= 1500 and free >= 20:
            if waited:
                log(f"gate passed before {label} after {time.time() - t0:.0f} s")
            return True
        if time.time() - t_log > 1800:
            log(f"waiting before {label}: available RAM {a:.0f} MB, disk {free:.0f} GB (need 1500 MB and 20 GB)")
            status("part2", note=f"waiting before {label} since {time.strftime('%H:%M', time.localtime(t0))}: RAM {a:.0f} MB")
            t_log = time.time()
        waited = True
        time.sleep(60)


# ----------------------------------------------------------------------------------------------- splits
def frame_val(df, frac, seed=0):  # scripts/assemble_PETPS_exp.py, unchanged
    fr = df.groupby(["source_file", "frame"]).size().sort_index()
    order = np.random.default_rng(seed).permutation(len(fr))
    cum = np.cumsum(fr.values[order])
    target = frac * len(df)
    j = int(np.argmin(np.abs(cum - target)))
    chosen = set(fr.index[order[: j + 1]])
    return pd.Series([(s, f) in chosen for s, f in zip(df.source_file, df.frame)], index=df.index)


def track_val(pool, frac, seed=0):  # scripts/tracks_PETPS_exp.py, unchanged
    t = pool.groupby("track_id").size().sort_index()
    order = np.random.default_rng(seed).permutation(len(t))
    cum = np.cumsum(t.values[order])
    j = int(np.argmin(np.abs(cum - frac * len(pool))))
    chosen = set(t.index[order[: j + 1]])
    return pool.track_id.isin(chosen)


def pair_split(m, pet, ps):
    """Split of the real z0 expanded manifest m (original-direction labels in split_video_holdout) for test pair."""
    v = m.source_file.str.replace("_600.mat", "", regex=False)
    s = pd.Series("train", index=m.index, dtype=object)
    s[v.isin([pet, ps])] = "test"
    keep = v.isin(ORIG_TRAIN - {pet, ps})
    s[keep] = m.split_video_holdout[keep]
    decisions = []
    for cls, vids in (("PET", PET), ("PS", PS)):
        train_vids = [x for x in vids if x not in (pet, ps)]
        n_class_train = int(v.isin(train_vids).sum())
        for vid in train_vids:
            if vid in ORIG_TRAIN:
                continue
            pool = m[v == vid]
            largest = int(pool.groupby("track_id").size().max())
            fallback = largest > 0.15 * n_class_train
            is_val = frame_val(pool, 0.175) if fallback else track_val(pool, 0.175)
            s[pool.index] = np.where(is_val, "val", "train")
            decisions.append(dict(cls=cls, video=vid, pool=len(pool), largest_track=largest, fallback_frames=bool(fallback),
                                  val=int(is_val.sum())))
    return s, decisions


def weights(m, split):
    n = m[split == "train"]["class"].value_counts()
    return {c: round(float(n.sum() / (2 * n[c])), 4) for c in sorted(n.index)}


def self_check():
    """The builder must reproduce the stored original and reverse splits and weights exactly."""
    m = pd.read_csv(f"{PO}/manifest_recon_s0_z0_PETPS_exp.csv")
    mr = pd.read_csv(f"{PO}/manifest_recon_s0_z0_PETPS_rev_exp.csv")
    out = []
    for (pet, ps), ref, wfile in ((("PET2", "PS1"), m.split_video_holdout, "class_weights_twoclass_PETvsPS_exp.json"),
                                  (("PET1", "PS2"), mr.split_video_holdout, "class_weights_twoclass_PETvsPS_rev_exp.json")):
        s, _ = pair_split(m, pet, ps)
        same_split = bool((s.values == ref.values).all())
        same_w = weights(m, s) == json.load(open(f"{PO}/{wfile}"))
        out.append(dict(pair=f"{pet}+{ps}", same_split=same_split, n_diff=int((s.values != ref.values).sum()), same_weights=same_w))
    return out


def build_pair(pet, ps):
    tag = f"pair_{pet}_{ps}"
    m = pd.read_csv(f"{PO}/manifest_recon_s0_z0_PETPS_exp.csv")
    b = pd.read_csv(f"{PO}/manifest_background_recon_s0_z0_PETPS_exp.csv")
    s, dec = pair_split(m, pet, ps)
    mp = m.copy()
    mp["split_video_holdout"] = s.values
    bp = b.copy()
    k = mp.set_index(KEY).split_video_holdout
    bp["split"] = k.reindex(pd.MultiIndex.from_frame(b[KEY])).values
    assert bp.split.notna().all()
    w = weights(mp, mp.split_video_holdout)
    files = dict(real=f"{PO}/manifest_recon_s0_z0_PETPS_{tag}_exp.csv", bg=f"{PO}/manifest_background_recon_s0_z0_PETPS_{tag}_exp.csv",
                 cw=f"{PO}/class_weights_twoclass_PETvsPS_{tag}_exp.json")
    if CFG["test"]:
        files = {k_: os.path.join(CFG["out_prefix"], os.path.basename(v_)) for k_, v_ in files.items()}
    for f in files.values():
        if os.path.exists(f):
            raise FileExistsError(f)
    mp.to_csv(files["real"], index=False)
    bp.to_csv(files["bg"], index=False)
    json.dump(w, open(files["cw"], "w"), indent=2)
    counts = mp.groupby(["class", "split_video_holdout"]).size().to_dict()
    return files, dict(weights=w, decisions=dec, counts={f"{c} {sp}": int(n) for (c, sp), n in counts.items()})


# ----------------------------------------------------------------------------------------------- runs
def base_args(kind):
    return json.load(open(f"{RES}/{'real' if kind == 'real' else 'bg'}_tc_PETvsPS_z0_exp_s0/results.json"))["args"]


def cmd(kind, seed, man, cw, out):
    a = base_args(kind)
    assert a["aug"] is None and not a["patch_norm"] and a["donor_manifest"] is None
    return [sys.executable, "train_compact_cnn.py", "--channels", a["channels"], "--split", a["split"], "--manifest", man,
            "--patch-dir", a["patch_dir"], "--class-weights", cw, "--classes", a["classes"], "--epochs", str(a["epochs"]),
            "--patience", str(a["patience"]), "--batch-size", str(a["batch_size"]), "--lr", str(a["lr"]), "--weight-decay",
            str(a["weight_decay"]), "--width", str(a["width"]), "--dropout", str(a["dropout"]), "--seed", str(seed), "--out", out]


def run_once(kind, seed, files, out):
    status("part2", note=f"running {os.path.basename(out)}")
    t0 = time.time()
    with open(f"{LOGD}/{os.path.basename(out)}.log", "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd(kind, seed, files["real"] if kind == "real" else files["bg"], files["cw"], out), stdout=fh,
                            stderr=subprocess.STDOUT, creationflags=IDLE_FLAG).returncode
    ok = rc == 0 and os.path.exists(f"{out}/results.json")
    same = None
    if ok:
        a_new, a_old = json.load(open(f"{out}/results.json"))["args"], base_args(kind)
        same = all(a_new.get(k) == v for k, v in a_old.items() if k not in ("manifest", "class_weights", "out", "seed"))
    dur = time.time() - t0
    log(f"{os.path.basename(out)} exit={rc} {dur:.0f}s {'ok' if ok else 'FAILED'}; other settings as the z0 expanded runs: {same}")
    return ok, dur, same


Z0_EARLIER = [629, 817, 816, 586, 845, 443, 532, 634, 551, 732, 780, 1127, 899, 867, 1021, 1102]  # z0 expanded runs (log)


def rate(durs):
    return statistics.median(Z0_EARLIER + durs)


def analyse_pair(name, runs, pred_csv):
    import eval_track_ci_exp as ev
    if os.path.exists(pred_csv):
        raise FileExistsError(pred_csv)
    L, summ, excl = ev.analyse(name, runs, pred_csv)
    return L, summ["pair"] if "pair" in summ else summ[list(summ)[0]], excl


def pair_section(pet, ps, files, info, runs, failed, L_ev, s, excl, durs):
    return ["", "=" * 118, f"PAIR test {pet} + {ps} (z0, expanded set; real seeds {sorted(runs['pair']['real'])}, particle-free seeds "
            f"{sorted(runs['pair']['particle-free'])})", "=" * 118,
            f"Manifests {files['real']}, {files['bg']}; class weights {files['cw']} = {info['weights']}",
            f"Split counts (real): {info['counts']}",
            f"Validation of the added training videos: {info['decisions'] or 'none (all training videos keep their original labels)'}",
            f"Runs skipped after a failed retry: {failed or 'none'}; excluded by the 4-decimal check: {excl or 'none'}; run times "
            f"{', '.join(f'{x:.0f} s' for x in durs)}",
            f"Reading for this pair: {s['T']} (real {s['real'][0]:.4f}+/-{s['real'][1]:.4f}, particle-free {s['pf'][0]:.4f}+/-"
            f"{s['pf'][1]:.4f}, gap {s['gap']:+.4f}, floor {s['floor']:.4f}, chance {s['chance']:.4f} from 1000 random label draws "
            "with this pair's training class proportions)"] + L_ev


def w_reading(done):
    """done: list of dict(pair, real, T, rec) for the new complete pairs."""
    z0 = json.load(open(f"{RES}/z0_trackCI_summary_exp.json"))
    allp = [dict(pair="PET2+PS1 (original)", real=z0["original"]["real"][0], T=z0["original"]["T"], rec=z0["original"]["rec"]["real"]),
            dict(pair="PET1+PS2 (reverse)", real=z0["reverse"]["real"][0], T=z0["reverse"]["T"], rec=z0["reverse"]["rec"]["real"])] + done
    rng = max(x["real"] for x in allp) - min(x["real"] for x in allp)
    if len(done) < 4:
        W = "W4 (fewer than 4 new complete pairs: no reading)"
    elif all(x["T"] == "T1" for x in allp):
        W = "W1" if rng <= 0.10 else "W2"
    elif any(x["T"] in ("T2", "T3") for x in allp):
        W = "W3"
    else:
        W = "none of W1 to W3 (a pair reads T4)"
    n90 = sum(x["real"] >= 0.90 for x in allp)
    L = ["", "=" * 118, "W READING (fixed before the runs): all pairs of this job plus the z0 expanded original (0.9052) and reverse (0.8477)",
         "values. W1 T1 in every pair and range of real macro-F1 <= 0.10; W2 T1 in every pair and range > 0.10; W3 T2 or T3 in",
         "at least one pair; W4 fewer than 4 new complete pairs (no reading).", "=" * 118,
         f"{'test pair':22s} {'real macro-F1':>13s}  T    real recall PET / PS"]
    L += [f"{x['pair']:22s} {x['real']:13.4f}  {x['T']:4s} {x['rec'][0][0]:.3f} / {x['rec'][1][0]:.3f}" for x in allp]
    L += [f"New complete pairs: {len(done)} of {len(NEW_PAIRS)}. Range of real macro-F1 over all {len(allp)} pairs: {rng:.4f} "
          f"(max {max(x['real'] for x in allp):.4f}, min {min(x['real'] for x in allp):.4f}). Pairs with real macro-F1 >= 0.90: {n90} of "
          f"{len(allp)}.", f"Reading: {W}",
          "The pairs share videos (each video is test in some pairs and training in others, and every pair trains on most of",
          "the same patches), so they are not independent samples: the mean over pairs is a summary of these splits, not a",
          "sample mean with a standard error, and the range is not a sampling interval."]
    return W, L, allp, rng, n90


# ----------------------------------------------------------------------------------------------- main
def main():
    os.makedirs(LOGD, exist_ok=True)
    log(f"item 42 part 2 started, pid {os.getpid()}, idle priority; deadline {DEADLINE:%Y-%m-%d %H:%M}")
    status("part2", status="running", started=time.strftime("%Y-%m-%d %H:%M"), pid=os.getpid())
    chk = self_check()
    log(f"builder self-check against the stored splits: {chk}")
    append_report(["", "PART 2. z0, EXPANDED SET, VIDEO-TO-VIDEO SPREAD (the reading of 'the way the _rev ones were built' used here):",
                   *[x.strip() for x in __doc__.split("spread.")[1].split("Runs:")[0].strip().split("\n")],
                   f"Builder check, reproducing the stored splits and weights: {chk}",
                   "Runs: train_compact_cnn.py (unchanged) with every setting of the z0 expanded runs passed explicitly; only manifest,",
                   "class weights and out differ. Pairs in this order: " + ", ".join(f"{a}+{b}" for a, b in NEW_PAIRS)])
    if not all(x["same_split"] and x["same_weights"] for x in chk):
        log("builder does not reproduce the stored splits: stopped")
        status("part2", status="stopped: the split builder did not reproduce the stored splits")
        append_report(["STOPPED: the split builder does not reproduce the stored original and reverse splits; no pair was run."])
        return
    durs, done = [], []
    for pet, ps in NEW_PAIRS:
        need = 6 * rate(durs)
        if dt.datetime.now() + dt.timedelta(seconds=need) > DEADLINE:
            log(f"pair {pet}+{ps} not started: at the rate of the previous runs ({rate(durs):.0f} s per run) it would end after 07:30")
            append_report(["", f"PAIR test {pet} + {ps}: not started (at {rate(durs):.0f} s per run it would end after 07:30)."])
            continue
        try:
            files, info = build_pair(pet, ps)
        except Exception:
            log(f"pair {pet}+{ps}: manifest build failed\n{traceback.format_exc()}")
            continue
        runs = {"pair": {"real": {}, "particle-free": {}}}
        failed, pdurs, stop = [], [], False
        for kind in ("real", "particle-free"):
            for seed in (0, 1, 2):
                base = f"{RES}/{'real' if kind == 'real' else 'bg'}_tc_PETvsPS_z0_exp_pair_{pet}_{ps}_s{seed}"
                ok = False
                for out in (base, base + "_retry1"):
                    if not gate(os.path.basename(out)):
                        stop = True
                        break
                    if os.path.exists(out):
                        log(f"{out} exists; not reused (new files only)")
                        continue
                    ok, dur, _ = run_once(kind, seed, files, out)
                    pdurs.append(dur)
                    durs.append(dur)
                    json.dump(durs, open(f"{LOGD}/durations.json", "w"))
                    if ok:
                        runs["pair"][kind][seed] = out
                        break
                if stop:
                    break
                if not ok:
                    failed.append(base)
                    log(f"{base}: failed twice, skipped")
            if stop:
                break
        complete = all(len(runs["pair"][k]) == 3 for k in runs["pair"])
        if stop and not complete:
            log(f"pair {pet}+{ps} incomplete at 07:30")
            append_report(["", f"PAIR test {pet} + {ps}: incomplete at 07:30 (runs done: {runs['pair']}); not counted."])
            break
        if not all(len(runs["pair"][k]) >= 2 for k in runs["pair"]):
            append_report(["", f"PAIR test {pet} + {ps}: too few successful runs ({runs['pair']}; failed {failed}); not counted."])
            continue
        try:
            L_ev, s, excl = analyse_pair(f"z0 pair {pet}+{ps}", runs, f"{RES}/z0_pair_{pet}_{ps}_predictions_exp.csv")
            append_report(pair_section(pet, ps, files, info, runs, failed, L_ev, s, excl, pdurs))
            if complete:
                done.append(dict(pair=f"{pet}+{ps}", real=s["real"][0], T=s["T"], rec=s["rec"]["real"], pf=s["pf"][0], gap=s["gap"]))
            log(f"pair {pet}+{ps} done: real {s['real'][0]:.4f}, particle-free {s['pf'][0]:.4f}, {s['T']}")
            status("part2", progress=f"{len(done)} of {len(NEW_PAIRS)} new pairs complete; last {pet}+{ps}: real {s['real'][0]:.4f}, "
                   f"particle-free {s['pf'][0]:.4f}, {s['T']}")
        except Exception:
            log(f"pair {pet}+{ps}: evaluation failed\n{traceback.format_exc()}")
        if dt.datetime.now() >= DEADLINE:
            break
    status("part2", status="done", finished=time.strftime("%Y-%m-%d %H:%M"))
    W, L, allp, rng, n90 = w_reading(done)
    append_report(L)
    log(f"W reading: {W}")
    summary = [f"Part 1 (evaluation only) in results/report_z0_pairs_exp.txt: old set V3 - z0 real +0.0039 [-0.014, +0.021] original, "
               f"-0.0406 [-0.081, -0.002] reverse (paired track bootstrap); seed ensembles z0 0.9082 / 0.8596, V3 0.8811 / 0.7215.",
               f"New (PET video, PS video) test pairs complete: {len(done)} of {len(NEW_PAIRS)}; every run with the z0 expanded settings.",
               "Real macro-F1 per pair: " + ", ".join(f"{x['pair'].split(' ')[0]} {x['real']:.3f} ({x['T']})" for x in allp) + ".",
               f"Range of real macro-F1 over {len(allp)} pairs {rng:.4f}; pairs with real >= 0.90: {n90} of {len(allp)}.",
               f"W reading: {W}. The pairs share videos, so they are not independent samples."]
    st.top(summary, [])
    st.update("wreading", status="done", finished=time.strftime("%Y-%m-%d %H:%M"), reading=W,
              files=[REPORT, "scripts/item42_chain.py", "scripts/item42_status.py"])
    log("finished")


def test():
    """Dry test into the scratchpad: builder self-check, one pair's manifests, the commands, and the per-pair
    analysis and W code on the z0 original runs (seeds 0 to 2) standing in for a pair."""
    S = r"C:\Users\haany\AppData\Local\Temp\claude\c--Users-haany-OneDrive-research-project-dataset-10jan23\075ab033-0168-4afa-a675-c38e4f73b481\scratchpad\item42_test"
    os.makedirs(S, exist_ok=True)
    for f in os.listdir(S):
        os.remove(os.path.join(S, f))
    CFG.update(test=True, out_prefix=S, report=os.path.join(S, "report.txt"))
    print("self-check", self_check())
    for pet, ps in NEW_PAIRS:
        files, info = build_pair(pet, ps)
        print(pet, ps, info["weights"], info["counts"], info["decisions"])
    print(" ".join(cmd("real", 0, files["real"], files["cw"], "results/OUT")[1:]))
    print(" ".join(cmd("particle-free", 2, files["bg"], files["cw"], "results/OUT")[1:]))
    runs = {"pair": {k: {s: f"{RES}/{pk}_tc_PETvsPS_z0_exp_s{s}" for s in (0, 1, 2)} for k, pk in (("real", "real"), ("particle-free", "bg"))}}
    L_ev, s, excl = analyse_pair("stand-in", runs, os.path.join(S, "pred.csv"))
    append_report(pair_section("PET2", "PS1", files, info, runs, [], L_ev, s, excl, [1.0]))
    done = [dict(pair=f"stand-in {i}", real=s["real"][0], T=s["T"], rec=s["rec"]["real"]) for i in range(4)]
    W, L, allp, rng, n90 = w_reading(done)
    append_report(L)
    print(open(CFG["report"], encoding="utf-8").read())
    print("rate now", rate([]), "pair estimate", 6 * rate([]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        main()
