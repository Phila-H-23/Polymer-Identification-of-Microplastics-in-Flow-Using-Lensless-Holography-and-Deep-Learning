"""Item 47: deeper CNN (18 weighted layers: train_compact_cnn_deep.py --extra-convs 2, width 32) with and without depth
augmentation, z0 expanded set, PET vs PS, real runs only.
Ln: --depth-aug none, every other argument from results/real_tc_PETvsPS_z0_exp[_rev]_s0.
Ld: --depth-aug all21, every other argument from results/real_tc_PETvsPS_z0depthD_exp[_rev]_s0.
Only --extra-convs, --seed and --out differ from those runs. Order seed by seed: Ln original, Ld original, Ln reverse,
Ld reverse. Idle priority (scripts/prio41.py), 1.5 GB RAM wait before every run, a failed run retried once into
<out>_retry1, no run started at or after 2026-10-14 10:00 local time (a run in progress finishes). No existing script is
changed.
timing (Step 4, report only, never used to drop runs): 7 epochs of K=0 and K=2 at width 32, one after the other, same
data (Ln original arguments, seed 0), idle priority; folders kept in results/_timing_item47. Epoch k time = time of the
"ep k" line minus time of the "ep k-1" line (unbuffered output); r = mean of epochs 3 to 7 for K=2 / same for K=0.
Projection per run = width-32 time of the same arm (none: the z0 expanded real runs, results/logs_expanded_PETPS_exp_queue.txt;
all21: item 44 arm D, results/logs_item44/runs.json), seed and direction, times r. Writes results/item47_timing.txt.
usage: python scripts/item47_chain.py timing | run --seeds 0 1 2 --dirs original reverse"""
import os
import re
import sys
import json
import time
import shutil
import ctypes
import subprocess

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)

RES = "results"
LOGD = f"{RES}/logs_item47"
SRC = {("Ln", "original"): f"{RES}/real_tc_PETvsPS_z0_exp_s0", ("Ln", "reverse"): f"{RES}/real_tc_PETvsPS_z0_exp_rev_s0",
       ("Ld", "original"): f"{RES}/real_tc_PETvsPS_z0depthD_exp_s0", ("Ld", "reverse"): f"{RES}/real_tc_PETvsPS_z0depthD_exp_rev_s0"}
NAME = {"Ln": "real_tc_PETvsPS_z0deep18_exp", "Ld": "real_tc_PETvsPS_z0deep18depthD_exp"}
STOP = time.mktime(time.strptime("2026-10-14 10:00", "%Y-%m-%d %H:%M"))
IDLE_FLAG = 0x40


def log(msg, logf=None):
    os.makedirs(LOGD, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(logf or f"{LOGD}/chain.log", "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


def out_name(arm, direction, seed):
    return f"{RES}/{NAME[arm]}{'_rev' if direction == 'reverse' else ''}_s{seed}"


def cmd(arm, direction, seed, out, epochs=None, extra=2, unbuffered=False):
    a = json.load(open(f"{SRC[(arm, direction)]}/results.json"))["args"]
    assert a["aug"] is None and not a["patch_norm"] and a["donor_manifest"] is None and a["width"] == 32
    depth = "none" if arm == "Ln" else a["depth_aug"]
    assert depth in ("none", "all21")
    return [sys.executable] + (["-u"] if unbuffered else []) + [
        "train_compact_cnn_deep.py", "--channels", a["channels"], "--split", a["split"],
        "--manifest", a["manifest"], "--patch-dir", a["patch_dir"], "--class-weights", a["class_weights"],
        "--classes", a["classes"], "--epochs", str(epochs or a["epochs"]), "--patience", str(a["patience"]),
        "--batch-size", str(a["batch_size"]), "--lr", str(a["lr"]), "--weight-decay", str(a["weight_decay"]),
        "--width", str(a["width"]), "--dropout", str(a["dropout"]), "--seed", str(seed), "--depth-aug", depth,
        "--extra-convs", str(extra), "--out", out]


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


def gate(label, stop=True, logf=None):
    """Wait for 1.5 GB free RAM (and 20 GB free disk). With stop=True, returns False once the stop time is reached."""
    t0, t_log, waited = time.time(), 0.0, False
    while True:
        if stop and time.time() >= STOP:
            return False
        disk = shutil.disk_usage(D).free / 2 ** 30
        if avail_mb() >= 1500 and disk >= 20:
            if waited:
                log(f"gate passed before {label} after {time.time() - t0:.0f} s", logf)
            return True
        if time.time() - t_log > 1800:
            log(f"waiting before {label}: available RAM {avail_mb():.0f} MB (need 1500), free disk {disk:.0f} GB (need 20)", logf)
            t_log = time.time()
        waited = True
        time.sleep(60)


def timing():
    tdir = f"{RES}/_timing_item47"
    os.makedirs(tdir, exist_ok=True)
    ep = {}
    for k in (0, 2):
        out = f"{tdir}/K{k}"
        assert not os.path.exists(out)
        gate(f"timing K={k}", stop=False)
        t0, ts = time.time(), {}
        with open(f"{tdir}/K{k}.log", "w", encoding="utf-8") as fh:
            p = subprocess.Popen(cmd("Ln", "original", 0, out, epochs=7, extra=k, unbuffered=True), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, creationflags=IDLE_FLAG)
            for line in p.stdout:
                t = time.time()
                fh.write(f"{t - t0:9.2f} {line}")
                mm = re.match(r"ep\s+(\d+) \|", line)
                if mm:
                    ts[int(mm.group(1))] = t
            rc = p.wait()
        assert rc == 0 and sorted(ts) == list(range(1, 8)), (k, rc, sorted(ts))
        ep[k] = dict(dt={e: ts[e] - ts[e - 1] for e in range(2, 8)}, first=ts[1] - t0, total=time.time() - t0,
                     n_params=json.load(open(f"{out}/results.json"))["n_params"])
    m = {k: sum(v["dt"][e] for e in range(3, 8)) / 5 for k, v in ep.items()}
    r = m[2] / m[0]
    base = {}
    for line in open(f"{RES}/logs_expanded_PETPS_exp_queue.txt", encoding="utf-8"):
        mm = re.match(r"real_tc_PETvsPS_z0_exp(_rev)?_s(\d) exit=0 (\d+)s", line)
        if mm:
            base[("Ln", "reverse" if mm.group(1) else "original", int(mm.group(2)))] = int(mm.group(3))
    for t in json.load(open(f"{RES}/logs_item44/runs.json"))["timing"]:
        mm = re.match(r"real_tc_PETvsPS_z0depthD_exp(_rev)?_s(\d)$", t["run"])
        if mm:
            base[("Ld", "reverse" if mm.group(1) else "original", int(mm.group(2)))] = t["seconds"]
    L = ["ITEM 47, STEP 4: TIMING OF THE K=0 AND K=2 NETWORKS (report only; never used to drop runs)", "",
         "Same data and conditions for both: train_compact_cnn_deep.py with every argument of results/real_tc_PETvsPS_z0_exp_s0",
         "(z0 expanded real, original direction, depth-aug none), seed 0, --epochs 7, width 32, idle priority, run one after the",
         f"other. Folders kept: {tdir}/K0 and {tdir}/K2 (stdout with elapsed seconds in K0.log and K2.log).",
         "Epoch k time = time of the 'ep k' line minus time of the 'ep k-1' line; epoch 1 (start-up) and epoch 2 not used.", ""]
    for k, v in ep.items():
        L.append(f"K={k} ({v['n_params']:,} parameters): start to 'ep 1' {v['first']:.1f} s; epochs 2 to 7: "
                 + ", ".join(f"{v['dt'][e]:.1f}" for e in range(2, 8)) + f" s; mean of epochs 3 to 7 {m[k]:.2f} s; whole run {v['total']:.0f} s")
    L += [f"r = {m[2]:.2f} / {m[0]:.2f} = {r:.3f}", "",
          "Projection per run = width-32 time of the same arm, seed and direction times r (none: z0 expanded real runs from",
          "results/logs_expanded_PETPS_exp_queue.txt; all21: item 44 arm D from results/logs_item44/runs.json):"]
    tot = 0
    for s in (0, 1, 2):
        for direction in ("original", "reverse"):
            for arm in ("Ln", "Ld"):
                b = base[(arm, direction, s)]
                tot += b * r
                L.append(f"  {os.path.basename(out_name(arm, direction, s)):45s} base {b:5d} s x r = {b * r:7.0f} s")
    L += [f"Projected total for the 12 runs: {tot:.0f} s = {tot / 3600:.2f} h (sum of base times {sum(base.values())} s).",
          f"Written {time.strftime('%Y-%m-%d %H:%M')}."]
    text = "\n".join(L)
    assert "\u2014" not in text and ("signif" + "icant") not in text.lower() and ("valid" + "ated") not in text.lower()
    o = f"{RES}/item47_timing.txt"
    if os.path.exists(o):
        raise FileExistsError(o)
    open(o, "w", encoding="utf-8").write(text + "\n")
    print(text)


def run(seeds, dirs):
    log(f"item 47 chain started, pid {os.getpid()}, idle priority; seeds {seeds}, directions {dirs}; "
        f"no run starts at or after 2026-10-14 10:00")
    runs, timing_, not_run, failed = {}, [], [], []
    for s in seeds:
        for direction in ("original", "reverse"):
            if direction not in dirs:
                continue
            for arm in ("Ln", "Ld"):
                base = out_name(arm, direction, s)
                done, stopped = None, False
                for out in (base, base + "_retry1"):
                    if os.path.exists(f"{out}/results.json"):
                        done = out
                        break
                    if os.path.exists(out):
                        log(f"{out} exists without results.json; not reused")
                        continue
                    if not gate(os.path.basename(out)):
                        stopped = True
                        log(f"{os.path.basename(out)} not started: stop time 2026-10-14 10:00 reached")
                        break
                    t0 = time.time()
                    log(f"{os.path.basename(out)} started")
                    with open(f"{LOGD}/{os.path.basename(out)}.log", "w", encoding="utf-8") as fh:
                        rc = subprocess.run(cmd(arm, direction, s, out), stdout=fh, stderr=subprocess.STDOUT,
                                            creationflags=IDLE_FLAG).returncode
                    ok = rc == 0 and os.path.exists(f"{out}/results.json")
                    ep = len(json.load(open(f"{out}/results.json"))["history"]) if ok else 0
                    timing_.append(dict(run=os.path.basename(out), arm=arm, direction=direction, seed=s,
                                        seconds=round(time.time() - t0), epochs=ep, ok=ok))
                    log(f"{os.path.basename(out)} exit={rc} {time.time() - t0:.0f}s epochs={ep} {'ok' if ok else 'FAILED'}")
                    if ok:
                        done = out
                        break
                if done:
                    runs.setdefault(arm, {}).setdefault(direction, {})[s] = done
                elif stopped:
                    not_run.append(os.path.basename(base))
                else:
                    failed.append(os.path.basename(base))
                    log(f"{base}: failed twice, skipped")
                json.dump(dict(runs=runs, timing=timing_, seeds=seeds, dirs=dirs, not_run=not_run, failed=failed),
                          open(f"{LOGD}/runs.json", "w"), indent=1)
    log("finished")


if __name__ == "__main__":
    os.makedirs(LOGD, exist_ok=True)
    if sys.argv[1] == "timing":
        timing()
    else:
        i, j = sys.argv.index("--seeds"), sys.argv.index("--dirs")
        run([int(x) for x in sys.argv[i + 1:j]], sys.argv[j + 1:])
