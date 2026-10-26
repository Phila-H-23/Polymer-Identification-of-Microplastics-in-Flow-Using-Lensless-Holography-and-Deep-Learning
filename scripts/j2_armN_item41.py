"""Item 41 background jobs J2 then J3, at idle priority (this process and every child), with gates between runs.
J2 = arm N: train_compact_cnn_bright.py --meannorm (every input patch divided by its own mean after augmentation),
z0, PET vs PS, expanded set, real and particle-free, seeds 0 to 2 in both directions (12 runs), command line of
scripts/run_arm_exp.py (the z0 baseline's: --classes PET,PS --epochs 150 --patience 30, same manifests, splits and
class-weight files, other settings at their defaults) with no --gain-range flag. A failed run is retried once into
<out>_retry1, then skipped and logged. Unit test first (scripts/test_bright_flags_item41.py, new output file).
After the 12 runs: predictions and track intervals (scripts/eval_track_ci_exp.py analyse(), unchanged) and the K
readings -> results/report_armN_exp.txt. Then J3 (scripts/j3_cam_z0_exp.py).
Gates before every run (never mid-run): available RAM >= 1500 MB, disk >= 20 GB free, and the stage rule: while a
stage runs, a run starts only if the stage has a unit measured with no job run (baseline) and no unit measured during a
job run took more than 1.25 x the median baseline unit; once a stage unit or a training run (time per epoch against
the median of the 12 arm G runs) slowed by more than 25% while a stage ran, the job waits until that stage is done."""
import os
import sys
import json
import time
import shutil
import ctypes
import statistics
import subprocess

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import item41_status as st  # noqa: E402
import run_arm_exp as ra  # noqa: E402  (cmd() and FLAGS only)

LOGD = "results/logs_item41"
CFG = "results/item41_armN_runs.json"
IDLE_FLAG = 0x40  # IDLE_PRIORITY_CLASS for the child processes
STAGES = ("stage1c", "stage2")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(f"{LOGD}/j2_runner.log", "a", encoding="utf-8").write(line + "\n")
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


def alive(pid):
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = ctypes.c_void_p
    k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    h = k.OpenProcess(0x1000, 0, int(pid))
    if not h:
        return False
    code = ctypes.c_uint32()
    k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return code.value == 259


def stage_running(s, key):
    v = s.get(key, {})
    return v.get("status") == "running" and (not v.get("pid") or alive(v["pid"]))


def armG_epoch_seconds():
    """Median seconds per epoch of the 12 arm G runs (same script and data; overnight log and results.json)."""
    v = []
    for line in open("results/logs_overnight_exp/run_arm_exp.log", encoding="utf-8"):
        p = line.split()
        if len(p) >= 6 and p[2] == "armG" and p[4] == "exit=0":
            n = len(json.load(open(f"results/{p[3]}/results.json"))["history"])
            v.append(float(p[5].rstrip("s")) / n)
    return statistics.median(v), len(v)


BASE_EPOCH, N_BASE = armG_epoch_seconds()
slowed_by = {}  # stage key -> reason (wait until that stage is done)


def blockers():
    s = st.load()
    why = []
    for key in list(slowed_by):
        if stage_running(s, key):
            why.append(f"{key}: {slowed_by[key]}")
        else:
            del slowed_by[key]
    for key in STAGES:
        if not stage_running(s, key) or key in slowed_by:
            continue
        u = s[key].get("unit_times", [])
        base = [x["t_compute"] for x in u if not x.get("job_in_run")]
        over = [x["t_compute"] for x in u if x.get("job_in_run")]
        if not base:
            why.append(f"{key}: no stage unit measured without a job run yet")
        elif over and max(over) > 1.25 * statistics.median(base):
            slowed_by[key] = (f"stage unit {max(over):.0f} s during a job run against median {statistics.median(base):.0f} s "
                              f"without (x{max(over) / statistics.median(base):.2f}); waiting until the stage is done")
            log(f"PAUSE: {key} {slowed_by[key]}")
            why.append(f"{key}: {slowed_by[key]}")
    a = avail_mb()
    if a < 1500:
        why.append(f"available RAM {a:.0f} MB < 1500 MB")
    free = shutil.disk_usage(D).free / 2 ** 30
    if free < 20:
        why.append(f"disk free {free:.1f} GB < 20 GB")
    return why


def gate(job, label):
    last = None
    while True:
        why = blockers()
        if not why:
            return
        msg = "; ".join(why)
        if msg != last:
            log(f"waiting before {label}: {msg}")
            st.update(job, in_run=False, note=f"waiting before {label}: {msg}")
            last = msg
        time.sleep(60)


def run_child(job, label, args, logfile):
    gate(job, label)
    s = st.load()
    stages_at_start = [k for k in STAGES if stage_running(s, k)]
    st.update(job, in_run=True, note=f"running {label}", current=label)
    t0 = time.time()
    with open(logfile, "w", encoding="utf-8") as fh:
        rc = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT, creationflags=IDLE_FLAG).returncode
    dt = time.time() - t0
    s = st.load()
    stages = sorted(set(stages_at_start + [k for k in STAGES if stage_running(s, k)]))
    st.update(job, in_run=False, note=f"finished {label} (exit {rc}, {dt:.0f} s)")
    return rc, dt, stages


def train_all():
    runs = {"original": {"real": {}, "particle-free": {}}, "reverse": {"real": {}, "particle-free": {}}}
    failed, timing = [], []
    for direction in ("original", "reverse"):
        for seed in (0, 1, 2):
            for kind, pre in (("real", "real"), ("particle-free", "bg")):
                base = f"results/{pre}_tc_PETvsPS_z0_armN_exp{'_rev' if direction == 'reverse' else ''}_s{seed}"
                done = None
                for out in (base, base + "_retry1"):
                    if os.path.exists(f"{out}/results.json"):
                        done = out
                        break
                    if os.path.exists(out):
                        log(f"{out} exists without results.json; not reused (new files only)")
                        continue
                    name = os.path.basename(out)
                    rc, dt, stages = run_child("j2", name, ra.cmd(kind, direction, seed, out, "armN"), f"{LOGD}/{name}.log")
                    ok = rc == 0 and os.path.exists(f"{out}/results.json")
                    ep = len(json.load(open(f"{out}/results.json"))["history"]) if ok else 0
                    per = dt / ep if ep else float("nan")
                    timing.append(dict(run=name, seconds=round(dt), epochs=ep, s_per_epoch=round(per, 2), stages=stages))
                    log(f"armN {name} exit={rc} {dt:.0f}s epochs={ep} {per:.2f} s/epoch (arm G median {BASE_EPOCH:.2f}) "
                        f"stages during run: {stages or 'none'} {'ok' if ok else 'FAILED'}")
                    if ok and stages and per > 1.25 * BASE_EPOCH:
                        for k in stages:
                            slowed_by[k] = (f"training run {name} {per:.2f} s/epoch against arm G median {BASE_EPOCH:.2f} "
                                            f"(x{per / BASE_EPOCH:.2f}); waiting until the stage is done")
                            log(f"PAUSE: {k} {slowed_by[k]}")
                    if ok:
                        done = out
                        break
                if done:
                    runs[direction][kind][seed] = done
                else:
                    failed.append(base)
                    log(f"armN {base}: failed twice, skipped")
                json.dump(dict(arm="armN", runs=runs, failed=failed, timing=timing,
                               armG_median_s_per_epoch=BASE_EPOCH), open(CFG, "w"), indent=1)
                st.update("j2", runs_done=sum(len(v) for d in runs.values() for v in d.values()), failed=failed)
    return runs, failed


def f1_of(d):
    return json.load(open(f"{d}/results.json"))["test_report"]["macro avg"]["f1-score"]


def report(runs, failed):
    import eval_track_ci_exp as ev
    for p in ("results/armN_predictions_exp.csv", "results/armN_trackCI_summary_exp.json", "results/armN_trackCI_exp.txt",
              "results/report_armN_exp.txt"):
        if os.path.exists(p):
            raise FileExistsError(p)
    L, summ, excl = ev.analyse("armN", runs, "results/armN_predictions_exp.csv")
    json.dump(summ, open("results/armN_trackCI_summary_exp.json", "w"), indent=1, default=float)
    open("results/armN_trackCI_exp.txt", "w", encoding="utf-8").write("\n".join([f"Runs excluded: {excl or 'none'}"] + L) + "\n")
    given = {"original": (0.9087, 0.0112), "reverse": (0.8477, 0.0124)}
    ut = open("results/test_bright_flags_item41.txt", encoding="utf-8").read().splitlines() \
        if os.path.exists("results/test_bright_flags_item41.txt") else ["(unit test output missing)"]
    R = ["ARM N (--meannorm), z0, PET vs PS, expanded set, track-based validation, seeds 0 to 2 per direction (item 41, J2)",
         "Script train_compact_cnn_bright.py with --meannorm: after all augmentation every input patch (training, validation",
         "and test) is divided by its own mean, so its mean is 1 and its sum is constant. No --gain-range flag (the arm G gain",
         "is not used); everything else as the z0 baseline: --classes PET,PS --epochs 150 --patience 30, the same manifests,",
         "splits and class-weight files (class_weights_twoclass_PETvsPS[_rev]_exp.json), other settings at their defaults,",
         "real and particle-free alike. 'No gain' read as no --gain-range flag: the baseline's default +/-5% training gain is",
         "part of 'everything else as the z0 baseline' and is kept; it multiplies the whole patch before the noise, so the",
         "mean division removes it and it changes only the relative size of the added noise (by at most 5%).",
         "Training normalisation statistics: compute_channel_stats() on the training split with the same transform, i.e. the",
         "per-channel mean and std over the training patches after each patch is divided by its own mean (no augmentation),",
         "as the baseline computes its statistics on its own (untransformed) training inputs; these statistics then normalise",
         "every split.",
         "Unit test (scripts/test_bright_flags_item41.py -> results/test_bright_flags_item41.txt):"] + ["  " + x for x in ut]
    R += ["", "Readings fixed before the result: margin per direction = larger real seed std (population std) of this arm and of",
          "the z0 baseline over seeds 0 to 2 (given: original 0.9087 std 0.0112, reverse 0.8477 std 0.0124); K1 arm real",
          "macro-F1 >= z0 real - margin in both directions AND T1 in both; K2 T2 or T3 in either direction; K3 otherwise.",
          f"Runs skipped after a failed retry: {failed or 'none'}. Runs excluded by the 4-decimal check: {excl or 'none'}."]
    ok_dirs, T = [], {}
    for direction in ("original", "reverse"):
        s = summ.get(direction)
        seeds = sorted(int(k) for k in runs[direction]["real"])
        zpref = f"results/real_tc_PETvsPS_z0_exp{'_rev' if direction == 'reverse' else ''}"
        a = [f1_of(runs[direction]["real"][k]) for k in seeds]
        z = [f1_of(f"{zpref}_s{k}") for k in (0, 1, 2)]
        zm, zs = statistics.mean(z), statistics.pstdev(z)
        chk = (round(zm, 4), round(zs, 4)) == given[direction]
        if not s or len(a) < 2:
            R.append(f"{direction}: no complete result (real seeds {seeds})")
            continue
        T[direction] = s["T"]
        am, asd = statistics.mean(a), statistics.pstdev(a)
        margin = max(asd, zs)
        ok = am >= zm - margin
        ok_dirs.append(ok)
        R.append(f"{direction}: arm real {am:.4f} (std {asd:.4f}, seeds {seeds}); z0 real seeds 0 to 2 {zm:.4f} (std {zs:.4f}; "
                 f"matches the given numbers: {chk}); margin {margin:.4f}; arm >= z0 - margin: {ok}; arm T reading {s['T']} (real "
                 f"{s['real'][0]:.4f}, particle-free {s['pf'][0]:.4f}, gap {s['gap']:+.4f}, floor {s['floor']:.4f}, chance "
                 f"{s['chance']:.4f}); track CI real [{s['real_ci'][1]:.4f}, {s['real_ci'][2]:.4f}], particle-free "
                 f"[{s['pf_ci'][1]:.4f}, {s['pf_ci'][2]:.4f}], gap [{s['gap_ci'][1]:+.4f}, {s['gap_ci'][2]:+.4f}]")
    if len(T) == 2 and len(ok_dirs) == 2:
        K = "K1" if all(ok_dirs) and all(t == "T1" for t in T.values()) else (
            "K2" if any(t in ("T2", "T3") for t in T.values()) else "K3")
    else:
        K = "(incomplete: a direction has no complete result)"
    R += [f"Reading: {K}",
          "Note: K1 means only that the z0 result survives dividing each patch by its own mean; it does not rule out brightness",
          "as a cue in other forms (the pattern of brightness inside the patch is kept).",
          "", "Test predictions: results/armN_predictions_exp.csv (same layout as results/z0_predictions_exp.csv).",
          "Per-class recall and precision with track intervals (scripts/eval_track_ci_exp.py analyse(), 1000 resamples of",
          "whole test tracks, numpy default_rng(0)):"]
    R += open("results/armN_trackCI_exp.txt", encoding="utf-8").read().splitlines()
    text = "\n".join(R)
    assert "\u2014" not in text and "significant" not in text.lower()
    open("results/report_armN_exp.txt", "w", encoding="utf-8").write(text + "\n")
    return K, summ


def main():
    os.makedirs(LOGD, exist_ok=True)
    log(f"J2 runner started, pid {os.getpid()}, idle priority; arm G median {BASE_EPOCH:.2f} s/epoch over {N_BASE} runs")
    st.update("j2", status="running", started=time.strftime("%Y-%m-%d %H:%M"), pid=os.getpid(), in_run=False)
    if not os.path.exists("results/test_bright_flags_item41.txt"):
        rc, dt, _ = run_child("j2", "unit test", [sys.executable, "scripts/test_bright_flags_item41.py"], f"{LOGD}/j2_unit_test.log")
        log(f"unit test exit={rc} {dt:.0f}s")
    runs, failed = train_all()
    gate("j2", "evaluation")
    st.update("j2", in_run=True, note="evaluation and report")
    K, summ = report(runs, failed)
    st.update("j2", status="done", in_run=False, finished=time.strftime("%Y-%m-%d %H:%M"),
              reading=f"{K}; " + "; ".join(f"{d}: real {v['real'][0]:.4f}, particle-free {v['pf'][0]:.4f}, {v['T']}"
                                           for d, v in summ.items()),
              files=["results/report_armN_exp.txt", "results/armN_predictions_exp.csv", "results/armN_trackCI_exp.txt",
                     "results/armN_trackCI_summary_exp.json", "results/test_bright_flags_item41.txt", CFG,
                     "scripts/j2_armN_item41.py", "scripts/test_bright_flags_item41.py"])
    log(f"J2 done: {K}")
    # J3 after J2
    st.update("j3", status="waiting", note="starts after J2")
    rc, dt, _ = run_child("j3", "J3 CAM", [sys.executable, "scripts/j3_cam_z0_exp.py"], f"{LOGD}/j3.log")
    if rc != 0 and not os.path.exists("results/report_cam_z0_exp.txt"):
        log(f"J3 failed (exit {rc}); retry once")
        rc, dt, _ = run_child("j3", "J3 CAM retry", [sys.executable, "scripts/j3_cam_z0_exp.py"], f"{LOGD}/j3_retry1.log")
    if rc != 0:
        st.update("j3", status="failed twice, skipped", in_run=False)
    log(f"J3 exit={rc} {dt:.0f}s; runner finished")


if __name__ == "__main__":
    main()
