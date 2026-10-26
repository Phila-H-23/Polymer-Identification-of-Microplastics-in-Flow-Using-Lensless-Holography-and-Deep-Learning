"""Item 49 chain (detached): width 64, 10 layers, reverse direction, arms Wn (--depth-aug none) and Wd (--depth-aug all21),
real runs only, seeds 0 to 2. Command = scripts/item44_chain.py cmd() for the reverse direction (train_compact_cnn_depth.py,
every setting read from results/real_tc_PETvsPS_z0_exp_rev_s0), changed only to --width 64; Wn also --depth-aug none with
the flat z0 patch folder of that run (the depth folder holds only depth subfolders), as item 46 Wn did.
One run at a time, 1.5 GB free RAM before each, training at above-normal priority with power throttling off
(scripts/item48_chain.py launch()), a failed run retried once into <out>_retry1, no run started at or after STOP_AFTER.
After each run: scripts/item49_report.py --partial; after the last run or at the stop: scripts/item49_report.py.
Order: s0 Wn, s0 Wd, s1 Wn, s1 Wd, s2 Wn, s2 Wd. Log results/logs_item49/chain.log, runs results/logs_item49/runs.json.
usage: python scripts/item49_chain.py run | show"""
import os
import sys
import json
import time
import shutil
import hashlib
import subprocess

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
import item44_chain as c44  # noqa: E402  (cmd)
import item48_chain as c48  # noqa: E402  (launch, avail_mb)
prio41.set_priority(prio41.NORMAL)

RES = "results"
LOGD = f"{RES}/logs_item49"
READ = f"{RES}/item49_readings.txt"
NAME = {"Wn": "real_tc_PETvsPS_z0w64_exp_rev", "Wd": "real_tc_PETvsPS_z0depthD_w64_exp_rev"}
STOP_AFTER = "2026-10-25 12:00"
STOP = time.mktime(time.strptime(STOP_AFTER, "%Y-%m-%d %H:%M"))


def log(msg):
    os.makedirs(LOGD, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(f"{LOGD}/chain.log", "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


def cmd(arm, seed, out):
    c = c44.cmd("_rev", "none" if arm == "Wn" else "all21", seed, out)
    c[c.index("--width") + 1] = "64"
    if arm == "Wn":
        c[c.index("--patch-dir") + 1] = json.load(open(f"{RES}/real_tc_PETvsPS_z0_exp_rev_s0/results.json"))["args"]["patch_dir"]
    return c


def gate(label):
    t0, t_log = time.time(), 0.0
    while True:
        if time.time() >= STOP:
            return False
        if c48.avail_mb() >= 1500 and shutil.disk_usage(D).free / 2 ** 30 >= 20:
            return True
        if time.time() - t_log > 1800:
            log(f"waiting before {label}: available RAM {c48.avail_mb():.0f} MB (need 1500)")
            t_log = time.time()
        time.sleep(30)


def report(partial):
    with open(f"{LOGD}/report{'_partial' if partial else ''}.log", "a", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, "scripts/item49_report.py"] + (["--partial"] if partial else []),
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"item49_report.py{' --partial' if partial else ''} exit {rc}")


def run():
    sha = hashlib.sha256(open(READ, "rb").read()).hexdigest()
    log(f"item 49 chain started, pid {os.getpid()}; readings {READ} sha256 {sha}; STOP_AFTER {STOP_AFTER}")
    for arm in ("Wn", "Wd"):
        log(f"{arm} command (seed 0): " + " ".join(cmd(arm, 0, f"{RES}/{NAME[arm]}_s0")[1:]))
    state = dict(runs={}, timing=[], not_run=[], failed=[], readings_sha256=sha, stop_after=STOP_AFTER)
    stopped = False
    for s in (0, 1, 2):
        for arm in ("Wn", "Wd"):
            base = f"{RES}/{NAME[arm]}_s{s}"
            if stopped:
                state["not_run"].append(os.path.basename(base))
                continue
            done = None
            for out in (base, base + "_retry1"):
                if os.path.exists(out):
                    log(f"{out} exists; not overwritten")
                    continue
                if not gate(os.path.basename(out)):
                    stopped = True
                    log(f"{os.path.basename(out)} not started: STOP_AFTER {STOP_AFTER} reached")
                    break
                t0 = time.time()
                with open(f"{LOGD}/{os.path.basename(out)}.log", "w", encoding="utf-8") as fh:
                    p, cls, thr = c48.launch(cmd(arm, s, out), stdout=fh, stderr=subprocess.STDOUT)
                    log(f"{os.path.basename(out)} started, pid {p.pid}, priority class {cls:#x}, power throttling off "
                        f"{'ok' if thr else 'FAILED'}")
                    rc = p.wait()
                ok = rc == 0 and os.path.exists(f"{out}/results.json")
                ep = len(json.load(open(f"{out}/results.json"))["history"]) if ok else 0
                state["timing"].append(dict(run=os.path.basename(out), arm=arm, seed=s, seconds=round(time.time() - t0), epochs=ep, ok=ok))
                log(f"{os.path.basename(out)} exit={rc} {time.time() - t0:.0f}s epochs={ep} {'ok' if ok else 'FAILED'}")
                if ok:
                    done = out
                    break
            if done:
                state["runs"].setdefault(arm, {}).setdefault("reverse", {})[s] = done
            elif stopped:
                state["not_run"].append(os.path.basename(base))
            else:
                state["failed"].append(os.path.basename(base))
                log(f"{base}: failed twice, skipped")
            json.dump(state, open(f"{LOGD}/runs.json", "w"), indent=1)
            report(partial=True)
    json.dump(state, open(f"{LOGD}/runs.json", "w"), indent=1)
    log("runs finished" if not stopped else "stopped by STOP_AFTER")
    report(partial=False)
    log("finished")


if __name__ == "__main__":
    os.makedirs(LOGD, exist_ok=True)
    if sys.argv[1] == "show":
        for arm in ("Wn", "Wd"):
            print(arm, " ".join(cmd(arm, 0, f"{RES}/{NAME[arm]}_s0")[1:]))
    else:
        for arm in ("Wn", "Wd"):
            for s in (0, 1, 2):
                assert not os.path.exists(f"{RES}/{NAME[arm]}_s{s}"), f"{NAME[arm]}_s{s} exists"
        run()
