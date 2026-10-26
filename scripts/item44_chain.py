"""Item 44 chain (detached, idle priority; one run at a time; 1.5 GB RAM wait before every run; retry once).
12 real runs of train_compact_cnn_depth.py (copy of train_compact_cnn.py with --depth-aug; the original is not used or
changed): arm D (--depth-aug all21) then arm S (--depth-aug shifted20), original then reverse direction, seeds 0 to 2;
every setting of the z0 expanded real runs passed explicitly (read from results/real_tc_PETvsPS_z0_exp[_rev]_s0),
only --patch-dir (pipeline_output/patches_depth_z0_PETPS_exp, depth subfolders), --depth-aug and --out differ.
Then the evaluation (test predictions regenerated from best.pt with the original script's evaluation code on the
unshifted 2.250 patches, 4-decimal check), the readings, and results/report_depth_aug_exp.txt plus the summary at the
top of results/item44_status.md."""
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
import item44_status as st  # noqa: E402
import unclipped_tracks as ut  # noqa: E402

PO, RES = "pipeline_output", "results"
ROOT = f"{PO}/patches_depth_z0_PETPS_exp"
REPORT = f"{RES}/report_depth_aug_exp.txt"
LOGD = f"{RES}/logs_item44"
IDLE_FLAG = 0x40
CL = ["PET", "PS"]
B = 1000
ARMS = [("D", "all21"), ("S", "shifted20")]
DIRS = [("original", "", "orig"), ("reverse", "_rev", "rev")]


def log(msg):
    os.makedirs(LOGD, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(f"{LOGD}/chain.log", "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


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
    t0, t_log, waited = time.time(), 0.0, False
    while True:
        a, free = avail_mb(), shutil.disk_usage(D).free / 2 ** 30
        if a >= 1500 and free >= 20:
            if waited:
                log(f"gate passed before {label} after {time.time() - t0:.0f} s")
            return
        if time.time() - t_log > 1800:
            log(f"waiting before {label}: available RAM {a:.0f} MB (need 1500 MB), disk {free:.0f} GB")
            st.update("train", note=f"waiting before {label} since {time.strftime('%H:%M', time.localtime(t0))}")
            t_log = time.time()
        waited = True
        time.sleep(60)


def cmd(direction_sfx, mode, seed, out):
    a = json.load(open(f"{RES}/real_tc_PETvsPS_z0_exp{direction_sfx}_s0/results.json"))["args"]
    assert a["aug"] is None and not a["patch_norm"] and a["donor_manifest"] is None and a["channels"] == "recon_s0"
    return [sys.executable, "train_compact_cnn_depth.py", "--channels", a["channels"], "--split", a["split"],
            "--manifest", a["manifest"], "--patch-dir", ROOT, "--class-weights", a["class_weights"], "--classes", a["classes"],
            "--epochs", str(a["epochs"]), "--patience", str(a["patience"]), "--batch-size", str(a["batch_size"]), "--lr", str(a["lr"]),
            "--weight-decay", str(a["weight_decay"]), "--width", str(a["width"]), "--dropout", str(a["dropout"]),
            "--seed", str(seed), "--depth-aug", mode, "--out", out]


def run_all():
    runs, timing = {}, []
    for arm, mode in ARMS:
        for dname, sfx, tag in DIRS:
            for s in (0, 1, 2):
                base = f"{RES}/real_tc_PETvsPS_z0depth{arm}_exp{sfx}_s{s}"
                done = None
                for out in (base, base + "_retry1"):
                    if os.path.exists(f"{out}/results.json"):
                        done = out
                        break
                    if os.path.exists(out):
                        log(f"{out} exists without results.json; not reused (new files only)")
                        continue
                    gate(os.path.basename(out))
                    st.update("train", note=f"running {os.path.basename(out)}")
                    t0 = time.time()
                    with open(f"{LOGD}/{os.path.basename(out)}.log", "w", encoding="utf-8") as fh:
                        rc = subprocess.run(cmd(sfx, mode, s, out), stdout=fh, stderr=subprocess.STDOUT, creationflags=IDLE_FLAG).returncode
                    ok = rc == 0 and os.path.exists(f"{out}/results.json")
                    dur = time.time() - t0
                    ep = len(json.load(open(f"{out}/results.json"))["history"]) if ok else 0
                    timing.append(dict(run=os.path.basename(out), seconds=round(dur), epochs=ep, ok=ok))
                    log(f"{os.path.basename(out)} exit={rc} {dur:.0f}s epochs={ep} {'ok' if ok else 'FAILED'}")
                    if ok:
                        done = out
                        break
                if done:
                    runs.setdefault(arm, {}).setdefault(dname, {})[s] = done
                else:
                    log(f"{base}: failed twice, skipped")
                json.dump(dict(runs=runs, timing=timing), open(f"{LOGD}/runs.json", "w"), indent=1)
                st.update("train", progress=f"{sum(len(v) for a_ in runs.values() for v in a_.values())} of 12 runs done")
    return runs, timing


def evaluate(run_dir):
    """Test predictions from best.pt with the ORIGINAL script's dataset and model code (eval_track_ci_exp.evaluate()'s
    procedure) on the unshifted patches (ROOT/2.250 = byte copies of the stored z0 patches)."""
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score
    import eval_track_ci_exp as ev
    saved = json.load(open(os.path.join(run_dir, "results.json")))
    a = saved["args"]
    m = ev.module("train_compact_cnn.py")
    split_col = m.SPLIT_CONFIG[a["split"]]["split_column"]
    man = pd.read_csv(a["manifest"])
    man = man[man["class"].isin(a["classes"].split(","))]
    classes = saved["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    train_df, test_df = man[man[split_col] == "train"], man[man[split_col] == "test"].reset_index(drop=True)
    pdir = os.path.join(a["patch_dir"], "2.250")
    mean, std = m.compute_channel_stats(train_df, pdir, c2i, a["channels"])
    model = m.CompactHoloCNN(1, len(classes), a["width"], a["dropout"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    model.eval()
    ds = m.PatchDataset(test_df, pdir, c2i, channels=a["channels"], mean=mean, std=std)
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=64, shuffle=False):
            preds += model(x).argmax(1).tolist()
    out = test_df[["patch_file", "class", "track_id"]].copy()
    out["pred"] = [classes[p] for p in preds]
    f1 = f1_score(out["class"], out["pred"], labels=classes, average="macro", zero_division=0)
    return out, float(f1), float(saved["test_report"]["macro avg"]["f1-score"])


def stats(P, tracks, W):
    f1, rec, prec, bf, br, bp = [], [], [], [], [], []
    for s in sorted(set(P.seed)):
        C = ut.per_track_counts(P[P.seed == s], tracks, CL)
        f, r, p = ut.metrics(C.sum(0))
        f1.append(float(f))
        rec.append(r)
        prec.append(p)
        mb = ut.metrics(np.einsum("bt,tij->bij", W, C))
        bf.append(mb[0])
        br.append(mb[1])
        bp.append(mb[2])
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(f1=ut.ms(f1), f1s=f1, boot=np.mean(bf, axis=0), ci=q(np.mean(bf, axis=0)),
                rec=[(ut.ms([float(x[i]) for x in rec]), q(np.mean(br, axis=0)[:, i])) for i in range(2)],
                prec=[(ut.ms([float(x[i]) for x in prec]), q(np.mean(bp, axis=0)[:, i])) for i in range(2)])


def report(runs, timing):
    z0 = pd.read_csv(f"{RES}/z0_predictions_exp.csv")
    z0 = z0[(z0.kind == "real") & z0.seed.isin([0, 1, 2])].rename(columns={"patch_id": "patch_file"})
    allp, excl, res = [], [], {}
    for arm, mode in ARMS:
        for dname, sfx, tag in DIRS:
            P = []
            for s, d in sorted(runs.get(arm, {}).get(dname, {}).items()):
                df, f1, f1s = evaluate(d)
                if round(f1, 4) != round(f1s, 4):
                    excl.append(f"{d} (recomputed {f1:.4f} vs saved {f1s:.4f})")
                    continue
                P.append(df.assign(seed=s, run=os.path.basename(d)))
            if len(P) < 2:
                continue
            P = pd.concat(P).rename(columns={"class": "true_class", "pred": "predicted_class"})
            allp.append(P.assign(arm=arm, direction=dname))
            Z = z0[z0.direction == dname]
            seeds = sorted(set(P.seed) & set(Z.seed))
            P, Z = P[P.seed.isin(seeds)], Z[Z.seed.isin(seeds)]
            assert set(P.patch_file) == set(Z.patch_file)
            tracks = sorted(set(Z.track_id))
            rng = np.random.default_rng(0)
            W = np.stack([np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks)) for _ in range(B)])
            a, z = stats(P, tracks, W), stats(Z, tracks, W)
            margin = max(a["f1"][1], z["f1"][1])
            diff = a["f1"][0] - z["f1"][0]
            db = a["boot"] - z["boot"]
            res[(arm, dname)] = dict(a=a, z=z, margin=margin, diff=diff, seeds=seeds,
                                     ci=(float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))), n=len(set(Z.patch_file)), nt=len(tracks))
    if allp:
        pd.concat(allp)[["run", "arm", "direction", "seed", "patch_file", "track_id", "true_class", "predicted_class"]].to_csv(
            f"{RES}/depth_aug_predictions_exp.csv", index=False)

    def rd_D():
        if not all(("D", d) in res for d, _, _ in DIRS):
            return "incomplete"
        x = [res[("D", d)] for d, _, _ in DIRS]
        if all(r["diff"] > r["margin"] for r in x):
            return "D1"
        if all(abs(r["diff"]) <= r["margin"] for r in x):
            return "D2"
        if all(r["diff"] < -r["margin"] for r in x):
            return "D3"
        return "D4"

    def rd_S():
        if not all(("S", d) in res for d, _, _ in DIRS):
            return "incomplete"
        x = [res[("S", d)] for d, _, _ in DIRS]
        if all(r["diff"] >= -r["margin"] for r in x):
            return "S1"
        if all(r["diff"] < -r["margin"] for r in x):
            return "S2"
        return "S3"
    Dr, Sr = rd_D(), rd_S()
    b = json.load(open(f"{LOGD}/depth_build.json"))
    u = json.load(open(r"C:\Users\haany\AppData\Local\Temp\claude\c--Users-haany-OneDrive-research-project-dataset-10jan23\075ab033-0168-4afa-a675-c38e4f73b481\scratchpad\item44_unit\unit.json"))
    L = ["DEPTH-SHIFT AUGMENTATION OF THE z0 INPUT, PET vs PS (item 44; the supervisor's depth-propagation request)", "",
         f"READINGS: arm D (all21) {Dr}; arm S (shifted20) {Sr}.",
         "  D1 arm D above z0 + margin in both directions; D2 within the margin in both; D3 below z0 - margin in both; D4 none of",
         "  these. S1 arm S >= z0 - margin in both directions; S2 below z0 - margin in both; S3 otherwise. z0 = the z0 expanded",
         "  real runs, seeds 0 to 2 (0.9087 original, 0.8477 reverse); margin = the larger real seed std of the two over those seeds.",
         "No particle-free runs were made, so there is no T reading for these arms.",
         "NOTE (a property of the method, found in step 1): with a zero-phase field, propagating by +dz and by -dz gives complex",
         "conjugate fields and so exactly the same intensity; the shifted patches at 2.25 - dz and 2.25 + dz are identical",
         "(largest difference 0 over 200 patches; 2.00 against 2.25 differs by up to 0.12). The 21 depths are therefore 11",
         "distinct inputs: all21 draws the unshifted patch with probability 1/21 and each of 10 shifted versions with 2/21;",
         "shifted20 draws each of the 10 shifted versions with 1/10.", "",
         "READINGS TAKEN (no stopping to ask):",
         "1. 'Reflect-pad to 2x before the FFTs and crop back' read as: pad once, do every FFT step on the padded grid, crop",
         "   once at the end; so the explicit chain (object plane, then forward) and the combined propagation by z_k - 2.25 mm",
         "   differ only by rounding (check ii), and the combined form is used.",
         "2. The unshifted depth (2.250) folder holds byte copies of the stored z0 patches, so validation and test inputs, the",
         "   channel statistics and the 2.250 draws in arm D are exactly the z0 runs' inputs.",
         "3. Check (i) to 1e-6 read as relative agreement at float32 precision where the stored value's float32 spacing",
         "   exceeds 1e-6 (as in item 43); the counts are below.",
         "4. 'Per training sample and per epoch': a depth is drawn every time a training patch is loaded (once per epoch per",
         "   patch), with Python's random generator, which the script seeds; epoch size unchanged.",
         "5. Per-class intervals: the analyse() procedure (1000 resamples of whole test tracks, default_rng(0), mean over seeds",
         "   per resample) on the real runs only, because analyse() itself needs particle-free runs.", "",
         f"{'arm':4s} {'direction':9s} {'real macro-F1':>15s} {'track CI':>17s} {'z0 real':>15s} {'arm - z0':>8s} {'margin':>7s} {'paired track CI':>19s}"]
    for (arm, d), r in res.items():
        L.append(f"{arm:4s} {d:9s} {r['a']['f1'][0]:.4f}+/-{r['a']['f1'][1]:.4f} [{r['a']['ci'][0]:.4f}, {r['a']['ci'][1]:.4f}] "
                 f"{r['z']['f1'][0]:.4f}+/-{r['z']['f1'][1]:.4f} {r['diff']:+.4f} {r['margin']:.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  "
                 f"(seeds {r['seeds']}, {r['n']} test patches, {r['nt']} tracks)")
    L.append("Per-class recall and precision, real (mean +/- seed std [track 95% CI]):")
    for (arm, d), r in res.items():
        for lab, x in ((f"arm {arm}", r["a"]), ("z0", r["z"])):
            L.append(f"  {d:9s} {lab:6s} recall PET {x['rec'][0][0][0]:.3f}+/-{x['rec'][0][0][1]:.3f} [{x['rec'][0][1][0]:.3f}, {x['rec'][0][1][1]:.3f}]"
                     f"  PS {x['rec'][1][0][0]:.3f}+/-{x['rec'][1][0][1]:.3f} [{x['rec'][1][1][0]:.3f}, {x['rec'][1][1][1]:.3f}]"
                     f" | precision PET {x['prec'][0][0][0]:.3f}+/-{x['prec'][0][0][1]:.3f} [{x['prec'][0][1][0]:.3f}, {x['prec'][0][1][1]:.3f}]"
                     f"  PS {x['prec'][1][0][0]:.3f}+/-{x['prec'][1][0][1]:.3f} [{x['prec'][1][1][0]:.3f}, {x['prec'][1][1][1]:.3f}]")
    ci = b["check_i"]
    L += ["", "STEP 1 CHECKS (scripts/item44_depth_build.py; results/logs_item44/depth_build.json; sheet results/depth_shift_sheet.png):",
          f"(i) computed z = 2.25 mm output against the stored z0 patch: largest absolute difference {ci['max_abs']:.2e} "
          f"({ci['n_over_1e6']} patches above 1e-6; largest relative difference {ci['max_rel']:.1e})" + (
              "; above 1e-6: " + "; ".join(f"{x[3]} {x[0]:.2e} at value {x[2]:.2f}" for x in ci["over"]) if ci["over"] else ""),
          f"(ii) explicit chain against the combined propagation, first 20 patches, every depth: largest difference {b['check_ii_max']:.1e} (limit 1e-5)",
          "(iii) finite outputs; per depth min / max / mean of the shifted patches (stored z0 patches: min "
          f"{b['ref_min']:.4f}, max {b['ref_max']:.4f}):"]
    L += [f"   {z}: {v['min']:.4f} / {v['max']:.4f} / {v['mean']:.5f}, non-finite {v['nonfinite']}" for z, v in b["check_iii"].items()]
    L += [f"Depth build time {b['seconds'] / 60:.0f} min.",
          f"Unit test of train_compact_cnn_depth.py with --depth-aug none against train_compact_cnn.py (1 epoch, seed 0, z0 original): "
          f"best.pt identical {u['best_pt_identical']}, results.json identical except args {u['results_identical']}, shared args identical "
          f"{u['args_identical']}.",
          f"Runs excluded by the 4-decimal check of the evaluation: {excl or 'none'}. Test predictions: results/depth_aug_predictions_exp.csv.",
          "Run times: " + ", ".join(f"{t['run']} {t['seconds']} s ({t['epochs']} epochs){'' if t['ok'] else ' FAILED'}" for t in timing)]
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    if os.path.exists(REPORT):
        raise FileExistsError(REPORT)
    open(REPORT, "w", encoding="utf-8").write(text + "\n")
    summ = [f"Readings: arm D (all21) {Dr}, arm S (shifted20) {Sr}; real runs only, so no T reading.",
            *[f"Arm {arm}, {d}: real {r['a']['f1'][0]:.4f}+/-{r['a']['f1'][1]:.4f} vs z0 {r['z']['f1'][0]:.4f} (diff {r['diff']:+.4f}, margin "
              f"{r['margin']:.4f}, paired track CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}])" for (arm, d), r in list(res.items())[:4]]]
    st.top(summ[:5], [f"Step 1 checks: (i) max {ci['max_abs']:.1e} ({ci['n_over_1e6']} above 1e-6), (ii) max {b['check_ii_max']:.1e}, (iii) all finite. "
                      f"Full report {REPORT}."])
    return Dr, Sr


def main():
    os.makedirs(LOGD, exist_ok=True)
    log(f"item 44 chain started, pid {os.getpid()}, idle priority")
    st.update("train", status="running", started=time.strftime("%Y-%m-%d %H:%M"), pid=os.getpid())
    runs, timing = run_all()
    st.update("train", status="done", finished=time.strftime("%Y-%m-%d %H:%M"))
    try:
        gate("evaluation")
        st.update("report", status="running")
        Dr, Sr = report(runs, timing)
        st.update("report", status="done", finished=time.strftime("%Y-%m-%d %H:%M"), reading=f"{Dr}, {Sr}",
                  files=[REPORT, f"{RES}/depth_aug_predictions_exp.csv", "train_compact_cnn_depth.py", "scripts/item44_chain.py",
                         "scripts/item44_depth_build.py", "results/depth_shift_sheet.png", ROOT + "/"])
        log(f"report written: {Dr}, {Sr}")
    except Exception:
        log("REPORT FAILED\n" + traceback.format_exc())
        st.update("report", status="failed", note=traceback.format_exc()[-500:])
    log("finished")


if __name__ == "__main__":
    main()
