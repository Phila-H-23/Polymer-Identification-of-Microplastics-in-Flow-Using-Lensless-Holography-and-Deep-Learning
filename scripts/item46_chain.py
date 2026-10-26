"""Item 46: wider CNN (--width 64) with and without depth augmentation, z0 expanded set, PET vs PS, real runs only.
Wn: train_compact_cnn_depth.py --width 64 --depth-aug none, every other argument from results/real_tc_PETvsPS_z0_exp[_rev]_s0.
Wd: train_compact_cnn_depth.py --width 64 --depth-aug all21, every other argument from
    results/real_tc_PETvsPS_z0depthD_exp[_rev]_s0. Only --width, --seed and --out differ from those runs.
Order seed by seed: Wn original, Wd original, Wn reverse, Wd reverse. Idle priority (scripts/prio41.py), 1.5 GB RAM
wait before every run, a failed run retried once into <out>_retry1. No existing script is changed.
usage: python scripts/item46_chain.py params | time1 | run --seeds 0 1 2 --dirs original reverse"""
import os
import sys
import json
import time
import shutil
import ctypes
import subprocess
import importlib.util

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)

RES = "results"
LOGD = f"{RES}/logs_item46"
SCRATCH = r"C:\Users\haany\AppData\Local\Temp\claude\c--Users-haany-OneDrive-research-project-dataset-10jan23\075ab033-0168-4afa-a675-c38e4f73b481\scratchpad"
SRC = {("Wn", "original"): f"{RES}/real_tc_PETvsPS_z0_exp_s0", ("Wn", "reverse"): f"{RES}/real_tc_PETvsPS_z0_exp_rev_s0",
       ("Wd", "original"): f"{RES}/real_tc_PETvsPS_z0depthD_exp_s0", ("Wd", "reverse"): f"{RES}/real_tc_PETvsPS_z0depthD_exp_rev_s0"}
NAME = {"Wn": "real_tc_PETvsPS_z0w64_exp", "Wd": "real_tc_PETvsPS_z0depthD_w64_exp"}
IDLE_FLAG = 0x40


def log(msg):
    os.makedirs(LOGD, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(f"{LOGD}/chain.log", "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


def out_name(arm, direction, seed):
    return f"{RES}/{NAME[arm]}{'_rev' if direction == 'reverse' else ''}_s{seed}"


def cmd(arm, direction, seed, out, epochs=None):
    a = json.load(open(f"{SRC[(arm, direction)]}/results.json"))["args"]
    assert a["aug"] is None and not a["patch_norm"] and a["donor_manifest"] is None and a["width"] == 32
    depth = "none" if arm == "Wn" else a["depth_aug"]
    assert depth in ("none", "all21")
    return [sys.executable, "train_compact_cnn_depth.py", "--channels", a["channels"], "--split", a["split"],
            "--manifest", a["manifest"], "--patch-dir", a["patch_dir"], "--class-weights", a["class_weights"],
            "--classes", a["classes"], "--epochs", str(epochs or a["epochs"]), "--patience", str(a["patience"]),
            "--batch-size", str(a["batch_size"]), "--lr", str(a["lr"]), "--weight-decay", str(a["weight_decay"]),
            "--width", "64", "--dropout", str(a["dropout"]), "--seed", str(seed), "--depth-aug", depth, "--out", out]


def params():
    spec = importlib.util.spec_from_file_location("tcd", "train_compact_cnn_depth.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = {}
    for w in (32, 64):
        net = m.CompactHoloCNN(1, 2, w, 0.3)
        out[w] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(json.dumps(out))
    return out


def time1():
    o = os.path.join(SCRATCH, "item46_time1_Wd_orig_s0")
    t0 = time.time()
    r = subprocess.run(cmd("Wd", "original", 0, o, epochs=1), capture_output=True, text=True, creationflags=IDLE_FLAG)
    dt = time.time() - t0
    ok = r.returncode == 0 and os.path.exists(os.path.join(o, "results.json"))
    print(json.dumps(dict(seconds=round(dt, 1), ok=ok, err=r.stderr[-300:] if not ok else "")))


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
        if avail_mb() >= 1500 and shutil.disk_usage(D).free / 2 ** 30 >= 20:
            if waited:
                log(f"gate passed before {label} after {time.time() - t0:.0f} s")
            return
        if time.time() - t_log > 1800:
            log(f"waiting before {label}: available RAM {avail_mb():.0f} MB (need 1500 MB)")
            t_log = time.time()
        waited = True
        time.sleep(60)


def run(seeds, dirs):
    log(f"item 46 chain started, pid {os.getpid()}, idle priority; seeds {seeds}, directions {dirs}")
    runs, timing = {}, []
    for s in seeds:
        for direction in ("original", "reverse"):
            if direction not in dirs:
                continue
            for arm in ("Wn", "Wd"):
                base = out_name(arm, direction, s)
                done = None
                for out in (base, base + "_retry1"):
                    if os.path.exists(f"{out}/results.json"):
                        done = out
                        break
                    if os.path.exists(out):
                        log(f"{out} exists without results.json; not reused")
                        continue
                    gate(os.path.basename(out))
                    t0 = time.time()
                    with open(f"{LOGD}/{os.path.basename(out)}.log", "w", encoding="utf-8") as fh:
                        rc = subprocess.run(cmd(arm, direction, s, out), stdout=fh, stderr=subprocess.STDOUT,
                                            creationflags=IDLE_FLAG).returncode
                    ok = rc == 0 and os.path.exists(f"{out}/results.json")
                    ep = len(json.load(open(f"{out}/results.json"))["history"]) if ok else 0
                    timing.append(dict(run=os.path.basename(out), arm=arm, direction=direction, seed=s,
                                       seconds=round(time.time() - t0), epochs=ep, ok=ok))
                    log(f"{os.path.basename(out)} exit={rc} {time.time() - t0:.0f}s epochs={ep} {'ok' if ok else 'FAILED'}")
                    if ok:
                        done = out
                        break
                if done:
                    runs.setdefault(arm, {}).setdefault(direction, {})[s] = done
                else:
                    log(f"{base}: failed twice, skipped")
                json.dump(dict(runs=runs, timing=timing, seeds=seeds, dirs=dirs), open(f"{LOGD}/runs.json", "w"), indent=1)
    log("finished")


if __name__ == "__main__":
    os.makedirs(LOGD, exist_ok=True)
    if sys.argv[1] == "params":
        params()
    elif sys.argv[1] == "time1":
        time1()
    else:
        i, j = sys.argv.index("--seeds"), sys.argv.index("--dirs")
        run([int(x) for x in sys.argv[i + 1:j]], sys.argv[j + 1:])
