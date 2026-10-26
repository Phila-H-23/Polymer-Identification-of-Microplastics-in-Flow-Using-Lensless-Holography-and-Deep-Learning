"""Item 47 waiter (detached, idle priority, no model calls): waits until the item 47 chain process exits, then, each
after a 1.5 GB free RAM wait, runs Part 0 on the finished Ln and Ld best.pt files and the analysis
(scripts/item47_analysis.py part0, then report). Log: results/logs_item47/analysis.log.
usage: python scripts/item47_waiter.py <chain pid> [--dry DIR]
--dry DIR: dry test on the item 46 files: the width-64 runs listed in results/logs_item46/runs.json stand in for Ln (Wn)
and Ld (Wd); every output (runs file, Part 0 table, report, predictions, log) goes to DIR."""
import os
import sys
import json
import time
import ctypes
import subprocess

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import item47_chain as c47  # noqa: E402  (gate, log)


def alive(pid):
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = ctypes.c_void_p
    k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    h = k.OpenProcess(0x1000, 0, pid)
    if not h:
        return False
    code = ctypes.c_uint32()
    k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return code.value == 259


pid = int(sys.argv[1])
if "--dry" in sys.argv:
    out = sys.argv[sys.argv.index("--dry") + 1]
    os.makedirs(out, exist_ok=True)
    rj = json.load(open("results/logs_item46/runs.json"))
    mp = {"Wn": "Ln", "Wd": "Ld"}
    json.dump(dict(runs={mp[k]: v for k, v in rj["runs"].items()}, timing=[dict(t, arm=mp[t["arm"]]) for t in rj["timing"]],
                   seeds=rj["seeds"], dirs=rj["dirs"], not_run=[], failed=[]), open(f"{out}/runs.json", "w"), indent=1)
    runs, p0, rep, pred, logf = (f"{out}/runs.json", f"{out}/part0_trainfit.txt", f"{out}/report.txt",
                                 f"{out}/predictions.csv", f"{out}/analysis.log")
else:
    runs, p0, rep, pred, logf = (f"{c47.LOGD}/runs.json", "results/item47_part0_deep_trainfit.txt",
                                 "results/report_layers_depth_exp.txt", "results/layers_depth_predictions_exp.csv",
                                 f"{c47.LOGD}/analysis.log")
c47.log(f"waiter started, pid {os.getpid()}, watching chain pid {pid}", logf)
while alive(pid):
    time.sleep(120)
c47.log(f"chain {pid} exited; running Part 0 and the analysis", logf)
for label, args in (("Part 0", ["part0", "--runs", runs, "--out", p0]),
                    ("analysis", ["report", "--runs", runs, "--report", rep, "--pred", pred, "--part0",
                                  "results/item47_part0_trainfit.txt", p0])):
    c47.gate(label, stop=False, logf=logf)
    with open(logf, "a", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, "scripts/item47_analysis.py"] + args, stdout=fh, stderr=subprocess.STDOUT,
                            creationflags=0x40).returncode
    c47.log(f"{label} exit {rc}", logf)
c47.log("waiter finished", logf)
