"""Item 43 chain (detached, idle priority; one run at a time; 1.5 GB RAM wait before every run).
STEP 2 (arm Pz): 12 real runs, seeds 0 to 2, on the declared pairs PET2+PS1 (original), PET1+PS2 (reverse), PET1+PS1,
PET2+PS2: train_compact_cnn.py (unchanged) with every setting of the z0 expanded real runs passed explicitly, only
--channels pol4_bgcorr, --patch-dir pipeline_output/patches_recon_z0pol4_PETPS_exp and --out differ (each pair keeps
its z0 manifest and class weights). Evaluation from best.pt (scripts/eval_track_ci_exp.py evaluate(), 4-decimal check),
readings Pz1 to Pz4 against z0 real over the shared seeds, per-class recall and precision of both arms, paired track
bootstrap of the real macro-F1 difference.
STEP 3: pair PET4+PS2 built and run with the item 42 code (scripts/item42_chain.py build_pair and cmd), real seeds 0
to 2 then particle-free seeds 0 to 2, analysed with eval_track_ci_exp.analyse(); then the step 1 table and figure for
8 pairs (new files). Appends to results/report_z0_pol4_exp.txt; status in results/item43_status.md.
usage: python scripts/item43_chain.py | test"""
import os
import sys
import json
import time
import shutil
import ctypes
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
import item43_status as st  # noqa: E402
import unclipped_tracks as ut  # noqa: E402

PO, RES = "pipeline_output", "results"
REPORT = f"{RES}/report_z0_pol4_exp.txt"
LOGD = f"{RES}/logs_item43"
PZDIR = f"{PO}/patches_recon_z0pol4_PETPS_exp"
IDLE_FLAG = 0x40
CL = ["PET", "PS"]
B = 1000
PAIRS = [  # name, z0 manifest, z0 class weights, z0 real run prefix, z0 predictions file, direction label in that file
    ("PET2+PS1 (original)", f"{PO}/manifest_recon_s0_z0_PETPS_exp.csv", f"{PO}/class_weights_twoclass_PETvsPS_exp.json",
     f"{RES}/real_tc_PETvsPS_z0_exp", f"{RES}/z0_predictions_exp.csv", "original", "orig"),
    ("PET1+PS2 (reverse)", f"{PO}/manifest_recon_s0_z0_PETPS_rev_exp.csv", f"{PO}/class_weights_twoclass_PETvsPS_rev_exp.json",
     f"{RES}/real_tc_PETvsPS_z0_exp_rev", f"{RES}/z0_predictions_exp.csv", "reverse", "rev"),
    ("PET1+PS1", f"{PO}/manifest_recon_s0_z0_PETPS_pair_PET1_PS1_exp.csv", f"{PO}/class_weights_twoclass_PETvsPS_pair_PET1_PS1_exp.json",
     f"{RES}/real_tc_PETvsPS_z0_exp_pair_PET1_PS1", f"{RES}/z0_pair_PET1_PS1_predictions_exp.csv", "pair", "pair_PET1_PS1"),
    ("PET2+PS2", f"{PO}/manifest_recon_s0_z0_PETPS_pair_PET2_PS2_exp.csv", f"{PO}/class_weights_twoclass_PETvsPS_pair_PET2_PS2_exp.json",
     f"{RES}/real_tc_PETvsPS_z0_exp_pair_PET2_PS2", f"{RES}/z0_pair_PET2_PS2_predictions_exp.csv", "pair", "pair_PET2_PS2")]
CFG = dict(test=False, report=REPORT)


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


def gate(key, label):
    t0, t_log, waited = time.time(), 0.0, False
    while True:
        a, free = avail_mb(), shutil.disk_usage(D).free / 2 ** 30
        if a >= 1500 and free >= 20:
            if waited:
                log(f"gate passed before {label} after {time.time() - t0:.0f} s")
            return
        if time.time() - t_log > 1800:
            log(f"waiting before {label}: available RAM {a:.0f} MB (need 1500 MB), disk {free:.0f} GB")
            status(key, note=f"waiting before {label} since {time.strftime('%H:%M', time.localtime(t0))}: RAM {a:.0f} MB")
            t_log = time.time()
        waited = True
        time.sleep(60)


def z0_args(kind="real"):
    return json.load(open(f"{RES}/{'real' if kind == 'real' else 'bg'}_tc_PETvsPS_z0_exp_s0/results.json"))["args"]


def pz_cmd(man, cw, seed, out):
    a = z0_args()
    assert a["aug"] is None and not a["patch_norm"] and a["donor_manifest"] is None
    return [sys.executable, "train_compact_cnn.py", "--channels", "pol4_bgcorr", "--split", a["split"], "--manifest", man,
            "--patch-dir", PZDIR, "--class-weights", cw, "--classes", a["classes"], "--epochs", str(a["epochs"]),
            "--patience", str(a["patience"]), "--batch-size", str(a["batch_size"]), "--lr", str(a["lr"]), "--weight-decay",
            str(a["weight_decay"]), "--width", str(a["width"]), "--dropout", str(a["dropout"]), "--seed", str(seed), "--out", out]


def run(key, args, out):
    """One run with retry once into <out>_retry1; returns the run folder or None."""
    for o in (out, out + "_retry1"):
        if os.path.exists(f"{o}/results.json"):
            return o
        if os.path.exists(o):
            log(f"{o} exists without results.json; not reused (new files only)")
            continue
        gate(key, os.path.basename(o))
        status(key, note=f"running {os.path.basename(o)}")
        t0 = time.time()
        a = args[:-1] + [o]
        with open(f"{LOGD}/{os.path.basename(o)}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run(a, stdout=fh, stderr=subprocess.STDOUT, creationflags=IDLE_FLAG).returncode
        ok = rc == 0 and os.path.exists(f"{o}/results.json")
        log(f"{os.path.basename(o)} exit={rc} {time.time() - t0:.0f}s {'ok' if ok else 'FAILED'}")
        if ok:
            return o
    log(f"{out}: failed twice, skipped")
    return None


# ----------------------------------------------------------------------------------------------- Pz analysis
def boot_weights(tracks):
    rng = np.random.default_rng(0)
    return np.stack([np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks)) for _ in range(B)])


def arm_stats(P, tracks, W):
    f1, rec, prec, bf = [], [], [], []
    for s in sorted(set(P.seed)):
        C = ut.per_track_counts(P[P.seed == s], tracks, CL)
        f, r, p = ut.metrics(C.sum(0))
        f1.append(float(f))
        rec.append(r)
        prec.append(p)
        bf.append(ut.metrics(np.einsum("bt,tij->bij", W, C))[0])
    return dict(f1s=f1, f1=ut.ms(f1), rec=[ut.ms([float(x[i]) for x in rec]) for i in range(2)],
                prec=[ut.ms([float(x[i]) for x in prec]) for i in range(2)], boot=np.mean(bf, axis=0))


def pz_analysis(pz_runs):
    import eval_track_ci_exp as ev
    preds, rows, L, excl = [], [], [], []
    for name, man, cw, zpref, zpred, zdir, tag in PAIRS:
        P = []
        for s, d in sorted(pz_runs.get(tag, {}).items()):
            df, f1, f1s, classes = ev.evaluate(d)
            if round(f1, 4) != round(f1s, 4):
                excl.append(f"{d} (recomputed {f1:.4f} vs saved {f1s:.4f})")
                continue
            P.append(df.assign(seed=s, run=os.path.basename(d)))
        if len(P) < 2:
            L.append(f"{name}: fewer than 2 usable Pz runs; no comparison")
            continue
        P = pd.concat(P).rename(columns={"class": "true_class", "pred": "predicted_class"})
        P["pair"] = name
        preds.append(P[["run", "pair", "seed", "patch_file", "track_id", "true_class", "predicted_class"]])
        Z = pd.read_csv(zpred)
        Z = Z[(Z.direction == zdir) & (Z.kind == "real")].rename(columns={"patch_id": "patch_file"})
        seeds = sorted(set(P.seed) & set(Z.seed))
        P, Z = P[P.seed.isin(seeds)], Z[Z.seed.isin(seeds)]
        assert set(P.patch_file) == set(Z.patch_file), f"{name}: test patches differ"
        tracks = sorted(set(Z.track_id))
        W = boot_weights(tracks)
        a, z = arm_stats(P, tracks, W), arm_stats(Z, tracks, W)
        margin = max(a["f1"][1], z["f1"][1])
        diff = a["f1"][0] - z["f1"][0]
        verdict = "above" if diff > margin else ("below" if diff < -margin else "within")
        dboot = a["boot"] - z["boot"]
        rows.append(dict(pair=name, seeds=seeds, pz=a["f1"], z0=z["f1"], margin=margin, diff=diff, verdict=verdict,
                         ci=(float(np.percentile(dboot, 2.5)), float(np.percentile(dboot, 97.5))), a=a, z=z,
                         n=len(set(Z.patch_file)), n_tracks=len(tracks)))
    if preds:
        out = f"{RES}/z0pol4_predictions_exp.csv"
        if not os.path.exists(out) and not CFG["test"]:
            pd.concat(preds).to_csv(out, index=False)
    ab = sum(r["verdict"] == "above" for r in rows)
    be = sum(r["verdict"] == "below" for r in rows)
    if len(rows) < 4:
        R = f"incomplete ({len(rows)} of 4 pairs compared)"
    elif ab >= 2 and be == 0:
        R = "Pz1"
    elif all(r["verdict"] == "within" for r in rows):
        R = "Pz2"
    elif be >= 2 and ab == 0:
        R = "Pz3"
    else:
        R = "Pz4"
    L = ["", "STEP 2. ARM Pz (per-channel z0, 4 channels, real only, seeds 0 to 2): pipeline_output/patches_recon_z0pol4_PETPS_exp,",
         "--channels pol4_bgcorr (ratios stored x 255, see readings), every other setting of the z0 expanded real runs.",
         "Predictions regenerated from best.pt (4-decimal check against results.json); saved to results/z0pol4_predictions_exp.csv.",
         f"Runs excluded by the 4-decimal check: {excl or 'none'}. No particle-free runs, so no T reading for this arm.",
         "Readings (fixed in the task): per pair, Pz real macro-F1 against z0 real over the shared seeds; margin = larger real seed",
         "std of the two; above if Pz > z0 + margin, below if Pz < z0 - margin, else within. Pz1 above in >= 2 pairs and below in",
         "none; Pz2 within in every pair; Pz3 below in >= 2 pairs and above in none; Pz4 none of these.",
         "Paired track bootstrap (extra column): the same 1000 resampled test tracks for both arms (default_rng(0)), each arm the",
         "mean over its seeds per resample; patches aligned by patch id (identical test sets checked).",
         f"{'test pair':22s} {'seeds':9s} {'Pz real':>15s} {'z0 real':>15s} {'Pz - z0':>8s} {'margin':>7s} {'':6s} paired track CI"]
    for r in rows:
        L.append(f"{r['pair']:22s} {str(r['seeds']):9s} {r['pz'][0]:.4f}+/-{r['pz'][1]:.4f} {r['z0'][0]:.4f}+/-{r['z0'][1]:.4f} "
                 f"{r['diff']:+.4f} {r['margin']:.4f} {r['verdict']:6s} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  ({r['n']} test patches, "
                 f"{r['n_tracks']} tracks)")
    L.append("Per-class recall and precision (real, mean +/- seed std over the shared seeds):")
    for r in rows:
        for arm, x in (("Pz", r["a"]), ("z0", r["z"])):
            L.append(f"  {r['pair']:22s} {arm}: recall PET {x['rec'][0][0]:.3f}+/-{x['rec'][0][1]:.3f}, PS {x['rec'][1][0]:.3f}+/-"
                     f"{x['rec'][1][1]:.3f}; precision PET {x['prec'][0][0]:.3f}+/-{x['prec'][0][1]:.3f}, PS {x['prec'][1][0]:.3f}+/-"
                     f"{x['prec'][1][1]:.3f}")
    L += [f"Reading: {R} (above in {ab}, below in {be}, within in {sum(r['verdict'] == 'within' for r in rows)} of {len(rows)} pairs)"]
    return R, L, rows


# ----------------------------------------------------------------------------------------------- main
def step2():
    status("pz", status="running: 12 runs", note="runs start")
    pz_runs = {}
    for name, man, cw, zpref, zpred, zdir, tag in PAIRS:
        for s in (0, 1, 2):
            out = f"{RES}/real_tc_PETvsPS_z0pol4_exp_{tag}_s{s}"
            d = run("pz", pz_cmd(man, cw, s, out), out)
            if d:
                pz_runs.setdefault(tag, {})[s] = d
            json.dump(pz_runs, open(f"{LOGD}/pz_runs.json", "w"), indent=1)
            status("pz", progress=f"{sum(len(v) for v in pz_runs.values())} of 12 runs done")
    gate("pz", "Pz evaluation")
    R, L, rows = pz_analysis(pz_runs)
    append_report(L)
    status("pz", status="done", finished=time.strftime("%Y-%m-%d %H:%M"), reading=R,
           summary="; ".join(f"{r['pair']}: Pz {r['pz'][0]:.4f} vs z0 {r['z0'][0]:.4f} ({r['verdict']})" for r in rows),
           files=[REPORT, f"{RES}/z0pol4_predictions_exp.csv", PZDIR + "/", "scripts/item43_pz_build.py", "scripts/item43_chain.py"])
    log(f"Pz reading {R}")
    return R, rows


def step3():
    import item42_chain as c42
    import eval_track_ci_exp as ev
    status("pair6", status="running", started=time.strftime("%Y-%m-%d %H:%M"))
    files, info = c42.build_pair("PET4", "PS2")
    runs = {"pair": {"real": {}, "particle-free": {}}}
    failed = []
    for kind in ("real", "particle-free"):
        for s in (0, 1, 2):
            out = f"{RES}/{'real' if kind == 'real' else 'bg'}_tc_PETvsPS_z0_exp_pair_PET4_PS2_s{s}"
            d = run("pair6", c42.cmd(kind, s, files["real"] if kind == "real" else files["bg"], files["cw"], out), out)
            if d:
                runs["pair"][kind][s] = d
            else:
                failed.append(out)
            status("pair6", progress=f"{sum(len(v) for v in runs['pair'].values())} of 6 runs done")
    gate("pair6", "pair evaluation")
    pred = f"{RES}/z0_pair_PET4_PS2_predictions_exp.csv"
    L_ev, summ, excl = ev.analyse("z0 pair PET4+PS2", runs, pred)
    s = summ["pair"]
    append_report(["", "STEP 3. PAIR test PET4 + PS2 (built and run with the item 42 code; real seeds 0 to 2, then particle-free 0 to 2)",
                   f"Manifests {files['real']}, {files['bg']}; class weights {files['cw']} = {info['weights']}; split counts {info['counts']};",
                   f"validation of the added training videos {info['decisions']}; runs skipped after a failed retry: {failed or 'none'}; "
                   f"excluded by the 4-decimal check: {excl or 'none'}.",
                   f"Reading for this pair: {s['T']} (real {s['real'][0]:.4f}+/-{s['real'][1]:.4f}, particle-free {s['pf'][0]:.4f}, gap "
                   f"{s['gap']:+.4f}, floor {s['floor']:.4f}, chance {s['chance']:.4f})"] + L_ev)
    import item43_pairs_table as pt
    df = pt.main(f"{RES}/pairs_table_z0_8pairs_exp.csv", f"{RES}/pairs_figure_z0_8pairs_exp.png",
                 "STEP 3 (continued). ALL 8 z0 TEST PAIRS (descriptive; the item 42 W reading stays as pre-registered: W2)")
    rng = df.real_f1.max() - df.real_f1.min()
    append_report([f"8 pairs, descriptive only: real macro-F1 {df.real_f1.min():.4f} to {df.real_f1.max():.4f} (range {rng:.4f}); "
                   f"T1 in {int((df['T'] == 'T1').sum())} of 8; real >= 0.90 in {int((df.real_f1 >= 0.90).sum())} of 8."])
    status("pair6", status="done", finished=time.strftime("%Y-%m-%d %H:%M"), reading=f"{s['T']} (real {s['real'][0]:.4f}, particle-free {s['pf'][0]:.4f})",
           files=[pred, f"{RES}/pairs_table_z0_8pairs_exp.csv", f"{RES}/pairs_figure_z0_8pairs_exp.png"])
    return s, df, rng


def main():
    os.makedirs(LOGD, exist_ok=True)
    log(f"item 43 chain started, pid {os.getpid()}, idle priority")
    R, rows, s6, df, rng = None, [], None, None, None
    try:
        R, rows = step2()
    except Exception:
        log("STEP 2 FAILED\n" + traceback.format_exc())
        status("pz", status="failed", note=traceback.format_exc()[-500:])
        append_report(["", "STEP 2 failed while running (see results/logs_item43/chain.log); going on to step 3."])
    try:
        s6, df, rng = step3()
    except Exception:
        log("STEP 3 FAILED\n" + traceback.format_exc())
        status("pair6", status="failed", note=traceback.format_exc()[-500:])
        append_report(["", "STEP 3 failed while running (see results/logs_item43/chain.log)."])
    j2 = json.load(open(f"{RES}/armN_trackCI_summary_exp.json"))
    summary = [
        f"Step 0: J2 (arm N) finished, K1 (real {j2['original']['real'][0]:.4f} / {j2['reverse']['real'][0]:.4f}, T1 both); J3 (CAM) finished, C1.",
        "Step 2 gate passed: the z0 input is the mean of the four per-channel z0 ratio images; per-channel patches rebuilt in about "
        "15 min (existing build code would need 4 h 40 min); 4 channels through --channels pol4_bgcorr with ratios stored x 255.",
        (f"Arm Pz (4-channel z0, real only): reading {R}; " + "; ".join(f"{r['pair'].split(' ')[0]} {r['pz'][0]:.3f} vs z0 {r['z0'][0]:.3f} "
                                                                       f"({r['verdict']})" for r in rows)) if R else "Arm Pz: did not finish (see the report).",
        (f"Pair PET4+PS2: {s6['T']}, real {s6['real'][0]:.4f}, particle-free {s6['pf'][0]:.4f}; 8 pairs: real macro-F1 "
         f"{df.real_f1.min():.3f} to {df.real_f1.max():.3f} (range {rng:.3f}), descriptive; the W reading stays W2.") if s6 else
        "Pair PET4+PS2: did not finish (see the report).",
        "Report results/report_z0_pol4_exp.txt; tables results/pairs_table_z0_exp.csv (7 pairs) and results/pairs_table_z0_8pairs_exp.csv; "
        "the pairs share videos, so they are not independent samples."]
    st.top(summary, [])
    log("finished")


def test():
    """Dry test (scratchpad): commands, and the Pz analysis with the z0 runs of each pair standing in for Pz."""
    CFG.update(test=True, report=os.path.join(r"C:\Users\haany\AppData\Local\Temp\claude\c--Users-haany-OneDrive-research-project-dataset-10jan23\075ab033-0168-4afa-a675-c38e4f73b481\scratchpad", "item43_test_report.txt"))
    print(" ".join(pz_cmd(PAIRS[0][1], PAIRS[0][2], 0, "results/OUT")[1:]))
    stand = {tag: {s: f"{zpref}_s{s}" for s in (0, 1, 2)} for name, man, cw, zpref, zpred, zdir, tag in PAIRS}
    R, L, rows = pz_analysis(stand)
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        main()
