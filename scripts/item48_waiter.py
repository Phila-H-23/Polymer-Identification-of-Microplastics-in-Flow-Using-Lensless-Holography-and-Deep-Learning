"""Item 48 waiter (detached, no model calls): waits until the item 48 chain process exits, then, each after a 1.5 GB
free RAM wait and at above-normal priority, runs the Test A fit check if the chain has not written it, Part 0 on the
finished WL best.pt files and the report (scripts/item48_analysis.py). Log: results/logs_item48/analysis.log.
usage: python scripts/item48_waiter.py <chain pid> [--dry DIR]
--dry DIR: dry test on the item 47 files: the item 47 Ln and Ld runs (results/logs_item47/runs.json) stand in for WLn
and WLd, and the 1-epoch flags-on check run (results/_check_item48/fit_e1_flags) stands in for both Test A runs; every
output (runs file, fit check, Part 0, report, predictions, log) goes to DIR."""
import os
import sys
import json
import time
import ctypes
import subprocess

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import item48_chain as c48  # noqa: E402  (gate, log, launch)


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
    rj = json.load(open("results/logs_item47/runs.json"))
    mp = {"Ln": "WLn", "Ld": "WLd"}
    stand = "results/_check_item48/fit_e1_flags"
    json.dump(dict(runs={"A": {"10": stand, "18": stand}, **{mp[k]: v for k, v in rj["runs"].items()}},
                   timing=rj["timing"], not_run=[], failed=[]), open(f"{out}/runs.json", "w"), indent=1)
    runs, fit, p0, rep, pred, logf = (f"{out}/runs.json", f"{out}/fit_check.txt", f"{out}/part0_trainfit.txt",
                                      f"{out}/report.txt", f"{out}/predictions.csv", f"{out}/analysis.log")
else:
    runs, fit, p0, rep, pred, logf = (f"{c48.LOGD}/runs.json", "results/item48_fit_check.txt",
                                      "results/item48_part0_trainfit.txt", "results/report_width_layers_exp.txt",
                                      "results/width_layers_predictions_exp.csv", f"{c48.LOGD}/analysis.log")
c48.log(f"waiter started, pid {os.getpid()}, watching chain pid {pid}", logf)
while alive(pid):
    time.sleep(60)
c48.log(f"chain {pid} exited; running the analysis", logf)
steps = [] if os.path.exists(fit) else [("fit check", ["fitcheck", "--runs", runs, "--out", fit])]
steps += [("Part 0", ["part0", "--runs", runs, "--out", p0]),
          ("report", ["report", "--runs", runs, "--report", rep, "--pred", pred, "--fit", fit, "--part0", p0])]
for label, args in steps:
    c48.gate(label, stop=False, logf=logf)
    with open(logf, "a", encoding="utf-8") as fh:
        p, cls, thr = c48.launch([sys.executable, "scripts/item48_analysis.py"] + args, stdout=fh, stderr=subprocess.STDOUT)
        rc = p.wait()
    c48.log(f"{label} exit {rc}", logf)
c48.log("waiter finished", logf)
