"""Item 48 chain (detached; one run at a time; 1.5 GB free RAM wait before each run; a failed run retried once into
<out>_retry1). Every training process starts at ABOVE-NORMAL priority (creationflags 0x8000, read back) with Windows
power throttling (EcoQoS) turned off; thread settings are not touched. No existing script is changed.
Test A (first): train_compact_cnn_fit.py, width 32, original direction, seed 0, --depth-aug none, --dropout 0
  --weight-decay 0 --no-aug --save-last --patience 1000 --epochs 150, --extra-convs 0 (10 layers) and 2 (18 layers),
  every other argument from results/real_tc_PETvsPS_z0_exp_s0; then the fit check (scripts/item48_analysis.py fitcheck)
  writes results/item48_fit_check.txt.
Test B: train_compact_cnn_deep.py --width 64 --extra-convs 2. WLn: --depth-aug none, every other argument from
  results/real_tc_PETvsPS_z0_exp[_rev]_s0. WLd: --depth-aug all21, every other argument from
  results/real_tc_PETvsPS_z0depthD_exp[_rev]_s0. Order: original seeds 0, 1, 2 (WLn then WLd), then reverse seeds
  0, 1, 2 (WLn then WLd). No Test B run starts at or after 2026-10-14 22:00 local time; a run in progress finishes.
timing (Step 4, report only, never used to drop runs): 7 epochs of the width-64, --extra-convs 2 network (WLn original
  arguments, seed 0) under the chain's conditions, kept in results/_timing_item48; epoch k time = time of the "ep k"
  line minus time of the "ep k-1" line; mean of epochs 3 to 7; projection per run = that time x the epoch count of the
  item 47 run of the same arm (Ln for WLn, Ld for WLd), seed and direction. Writes results/item48_timing.txt.
usage: python scripts/item48_chain.py timing | run"""
import os
import re
import sys
import json
import time
import shutil
import ctypes
import subprocess
from ctypes import wintypes

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.NORMAL)

RES = "results"
LOGD = f"{RES}/logs_item48"
SRC = {("n", "original"): f"{RES}/real_tc_PETvsPS_z0_exp_s0", ("n", "reverse"): f"{RES}/real_tc_PETvsPS_z0_exp_rev_s0",
       ("d", "original"): f"{RES}/real_tc_PETvsPS_z0depthD_exp_s0", ("d", "reverse"): f"{RES}/real_tc_PETvsPS_z0depthD_exp_rev_s0"}
NAME = {"WLn": "real_tc_PETvsPS_z0wl_exp", "WLd": "real_tc_PETvsPS_z0wl_depthD_exp"}
ITEM47 = {"WLn": "real_tc_PETvsPS_z0deep18_exp", "WLd": "real_tc_PETvsPS_z0deep18depthD_exp"}
STOP = time.mktime(time.strptime("2026-10-14 22:00", "%Y-%m-%d %H:%M"))
ABOVE = 0x8000

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class PTS(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]


class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
k32.GetPriorityClass.restype = wintypes.DWORD
k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MS)]


def log(msg, logf=None):
    os.makedirs(LOGD, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(logf or f"{LOGD}/chain.log", "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


def avail_mb():
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
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
        time.sleep(30)


def cmd(script, kind, direction, seed, out, width, extra, epochs=None, dropout=None, wd=None, patience=None, flags=()):
    a = json.load(open(f"{SRC[(kind, direction)]}/results.json"))["args"]
    assert a["aug"] is None and not a["patch_norm"] and a["donor_manifest"] is None and a["width"] == 32
    depth = "none" if kind == "n" else a["depth_aug"]
    assert depth in ("none", "all21")
    return [sys.executable, "-u", script, "--channels", a["channels"], "--split", a["split"], "--manifest", a["manifest"],
            "--patch-dir", a["patch_dir"], "--class-weights", a["class_weights"], "--classes", a["classes"],
            "--epochs", str(epochs or a["epochs"]), "--patience", str(patience or a["patience"]),
            "--batch-size", str(a["batch_size"]), "--lr", str(a["lr"]),
            "--weight-decay", str(a["weight_decay"] if wd is None else wd), "--width", str(width),
            "--dropout", str(a["dropout"] if dropout is None else dropout), "--seed", str(seed),
            "--depth-aug", depth, "--extra-convs", str(extra), "--out", out, *flags]


def cmd_fit(extra, out):
    return cmd("train_compact_cnn_fit.py", "n", "original", 0, out, 32, extra, epochs=150, dropout=0, wd=0,
               patience=1000, flags=("--no-aug", "--save-last"))


def cmd_wl(arm, direction, seed, out, epochs=None):
    return cmd("train_compact_cnn_deep.py", "n" if arm == "WLn" else "d", direction, seed, out, 64, 2, epochs=epochs)


def launch(c, **kw):
    """Start c at above-normal priority, turn power throttling off, return (process, priority class read back, ok)."""
    p = subprocess.Popen(c, creationflags=ABOVE, **kw)
    h = ctypes.c_void_p(int(p._handle))
    s = PTS(1, 0x1, 0x0)
    ok = bool(k32.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s)))
    return p, k32.GetPriorityClass(h), ok


def out_name(arm, direction, seed):
    return f"{RES}/{NAME[arm]}{'_rev' if direction == 'reverse' else ''}_s{seed}"


def timing():
    tdir = f"{RES}/_timing_item48"
    os.makedirs(tdir, exist_ok=True)
    out = f"{tdir}/WL"
    assert not os.path.exists(out)
    gate("timing", stop=False)
    t0, ts = time.time(), {}
    with open(f"{tdir}/WL.log", "w", encoding="utf-8") as fh:
        p, cls, thr = launch(cmd_wl("WLn", "original", 0, out, epochs=7), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            t = time.time()
            fh.write(f"{t - t0:9.2f} {line}")
            mm = re.match(r"ep\s+(\d+) \|", line)
            if mm:
                ts[int(mm.group(1))] = t
        rc = p.wait()
    assert rc == 0 and sorted(ts) == list(range(1, 8)), (rc, sorted(ts))
    dt = {e: ts[e] - ts[e - 1] for e in range(2, 8)}
    m = sum(dt[e] for e in range(3, 8)) / 5
    n_params = json.load(open(f"{out}/results.json"))["n_params"]
    ep47 = {t["run"]: t["epochs"] for t in json.load(open(f"{RES}/logs_item47/runs.json"))["timing"] if t["ok"]}
    L = ["ITEM 48, STEP 4: TIMING OF THE WIDTH-64, --extra-convs 2 NETWORK (report only; never used to drop runs)", "",
         "Conditions of the chain: train_compact_cnn_deep.py --width 64 --extra-convs 2 with every argument of",
         "results/real_tc_PETvsPS_z0_exp_s0 (z0 expanded real, original direction, depth-aug none), seed 0, --epochs 7,",
         f"above-normal priority (read back {cls:#x}), power throttling off ({'ok' if thr else 'FAILED'}), thread settings untouched.",
         f"Folder kept: {out} (stdout with elapsed seconds in {tdir}/WL.log).", "",
         f"{n_params:,} parameters; start to 'ep 1' {ts[1] - t0:.1f} s; epochs 2 to 7: "
         + ", ".join(f"{dt[e]:.1f}" for e in range(2, 8)) + f" s; mean of epochs 3 to 7 {m:.2f} s.", "",
         "Projection per run = that epoch time x the epoch count of the item 47 run of the same arm (Ln for WLn, Ld for WLd),",
         "seed and direction (results/logs_item47/runs.json). The two Test A runs are not included."]
    tot = {}
    for direction in ("original", "reverse"):
        for s in (0, 1, 2):
            for arm in ("WLn", "WLd"):
                src = f"{ITEM47[arm]}{'_rev' if direction == 'reverse' else ''}_s{s}"
                e = ep47[src]
                tot[direction] = tot.get(direction, 0) + e * m
                L.append(f"  {os.path.basename(out_name(arm, direction, s)):38s} {e:4d} epochs (as {src}) x {m:.2f} s = {e * m:7.0f} s")
    L += [f"Projected: original {tot['original'] / 3600:.2f} h, reverse {tot['reverse'] / 3600:.2f} h, "
          f"total {sum(tot.values()) / 3600:.2f} h for the 12 Test B runs.", f"Written {time.strftime('%Y-%m-%d %H:%M')}."]
    text = "\n".join(L)
    assert "\u2014" not in text and ("signif" + "icant") not in text.lower() and ("valid" + "ated") not in text.lower()
    o = f"{RES}/item48_timing.txt"
    if os.path.exists(o):
        raise FileExistsError(o)
    open(o, "w", encoding="utf-8").write(text + "\n")
    print(text)


def one(label, c, base, state, stop):
    """One run (retried once into <base>_retry1). Returns the finished folder, or None."""
    for out in (base, base + "_retry1"):
        if os.path.exists(f"{out}/results.json"):
            return out
        if os.path.exists(out):
            log(f"{out} exists without results.json; not reused")
            continue
        if not gate(os.path.basename(out), stop=stop):
            log(f"{os.path.basename(out)} not started: stop time 2026-10-14 22:00 reached")
            state["not_run"].append(os.path.basename(base))
            return None
        t0 = time.time()
        with open(f"{LOGD}/{os.path.basename(out)}.log", "w", encoding="utf-8") as fh:
            p, cls, thr = launch([out if x == base else x for x in c], stdout=fh, stderr=subprocess.STDOUT)
            log(f"{os.path.basename(out)} started ({label}), pid {p.pid}, priority class {cls:#x}, "
                f"power throttling off {'ok' if thr else 'FAILED'}")
            rc = p.wait()
        ok = rc == 0 and os.path.exists(f"{out}/results.json")
        ep = len(json.load(open(f"{out}/results.json"))["history"]) if ok else 0
        state["timing"].append(dict(run=os.path.basename(out), label=label, seconds=round(time.time() - t0), epochs=ep, ok=ok))
        log(f"{os.path.basename(out)} exit={rc} {time.time() - t0:.0f}s epochs={ep} {'ok' if ok else 'FAILED'}")
        if ok:
            return out
    state["failed"].append(os.path.basename(base))
    log(f"{base}: failed twice, skipped")
    return None


def save(state):
    json.dump(state, open(f"{LOGD}/runs.json", "w"), indent=1)


def run():
    log(f"item 48 chain started, pid {os.getpid()}; training at above-normal priority with power throttling off; "
        f"no Test B run starts at or after 2026-10-14 22:00")
    state = dict(runs={}, timing=[], not_run=[], failed=[])
    for key, k in (("10", 0), ("18", 2)):
        base = f"{RES}/real_tc_PETvsPS_z0fit{key}_exp_s0"
        d = one(f"Test A, {key} layers", cmd_fit(k, base), base, state, stop=False)
        if d:
            state["runs"].setdefault("A", {})[key] = d
        save(state)
    if len(state["runs"].get("A", {})) == 2 and not os.path.exists(f"{RES}/item48_fit_check.txt"):
        gate("fit check", stop=False)
        with open(f"{LOGD}/fitcheck.log", "w", encoding="utf-8") as fh:
            p, cls, thr = launch([sys.executable, "scripts/item48_analysis.py", "fitcheck", "--runs", f"{LOGD}/runs.json",
                                  "--out", f"{RES}/item48_fit_check.txt"], stdout=fh, stderr=subprocess.STDOUT)
            rc = p.wait()
        log(f"fit check written (exit {rc}): results/item48_fit_check.txt")
    for direction in ("original", "reverse"):
        for s in (0, 1, 2):
            for arm in ("WLn", "WLd"):
                base = out_name(arm, direction, s)
                d = one(f"Test B, {arm} {direction} seed {s}", cmd_wl(arm, direction, s, base), base, state, stop=True)
                if d:
                    state["runs"].setdefault(arm, {}).setdefault(direction, {})[s] = d
                save(state)
    log("finished")


if __name__ == "__main__":
    os.makedirs(LOGD, exist_ok=True)
    if sys.argv[1] == "timing":
        timing()
    else:
        run()
