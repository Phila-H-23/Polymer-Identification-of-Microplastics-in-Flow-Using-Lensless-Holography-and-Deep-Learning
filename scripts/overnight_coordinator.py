"""Overnight coordinator, 2026-10-07/08 (Phila asleep until about 08:00). Runs as a background process, polls every
30 s, launches each step's own script in the order of the overnight message, never starts a step after 07:45, writes
results/overnight_status.md (READ THIS FIRST block on top) after every change, and a final status block at the end.
It does not stop, suspend or change the chain (scripts/run_expanded_PETPS_v2.sh) or the Part C driver; the only link
to the chain is the release flag of its held raw-intensity build (step 6).

Overlap rule: lower-priority work may overlap training only at idle priority and only if training run times rise by no
more than 20%. Measured on the chain's z0 runs: seconds per epoch of each finished run (chain log seconds / epochs in
results.json) against the no-overlap baseline (the timed first run; reverse-fold runs scaled by the training-set ratio
1.258). The verdict is written to results/overnight_overlap_ok.json; the PP yield job (step 3) waits while it is false.
"""
import os
import re
import sys
import json
import time
import shutil
import hashlib
import datetime as dt
import statistics as st
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
RES = "results"
WORK = (r"C:\Users\haany\AppData\Local\Temp\claude\c--Users-haany-OneDrive-research-project-dataset-10jan23"
        r"\075ab033-0168-4afa-a675-c38e4f73b481\scratchpad\recon_check_work")
STATUS = f"{RES}/overnight_status.md"
STATE = f"{RES}/overnight_state_exp.json"
LOGD = f"{RES}/logs_overnight_exp"
CUTOFF = dt.datetime(2026, 10, 8, 7, 45)
RELEASE = f"{RES}/overnight_release_bgcorr.flag"
Q = f"{RES}/logs_expanded_PETPS_exp_queue.txt"
SHA0 = "7221dd9b4a305a7acf0c5043ce2d46dc86138f9385ddc92c80486f87cb7a335a"
NAMES = {"0": "Finish queued work: chain (z0 batch); Part C, Part B re-score, I90/I0, PET4; PE500/PP500 check",
         "1": "CNN-level track intervals for the z0 runs",
         "2": "Brightness arm G (--gain-range 0.3)",
         "3": "PP yield check (read-only)",
         "4": "PET vs PP, three folds (only if PP qualifies)",
         "5": "Arm N (--meannorm)",
         "6": "Raw-intensity build and batch (chain step 6, held)"}
SCRIPTS = {"1": ["scripts/overnight_step1.py"], "2": ["scripts/run_arm_exp.py", "armG"], "3": ["scripts/pp_yield_exp.py"],
           "4": ["scripts/pp_folds_exp.py"], "5": ["scripts/run_arm_exp.py", "armN"]}


def now():
    return dt.datetime.now()


def ts():
    return now().strftime("%Y-%m-%d %H:%M")


def rd(p):
    return open(p, encoding="utf-8", errors="ignore").read() if os.path.exists(p) else ""


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def avail_mb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("a", ctypes.c_ulong), ("b", ctypes.c_ulong), ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong),
                    ("e", ctypes.c_ulonglong), ("f", ctypes.c_ulonglong), ("g", ctypes.c_ulonglong),
                    ("h", ctypes.c_ulonglong), ("i", ctypes.c_ulonglong)]
    m = MS()
    m.a = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.d / 2 ** 20)


class Coordinator:
    def __init__(self):
        s = json.load(open(STATE)) if os.path.exists(STATE) else {}
        self.steps = s.get("steps") or {k: dict(status="waiting", note="", started=None, finished=None, attempts=0)
                                        for k in NAMES}
        self.events = s.get("events", [])
        self.procs = {}
        self.overlap = s.get("overlap", {})
        self.last_write = 0
        self.final_written = False

    # -------------------------------------------------------------------------------------------------- utilities
    def event(self, msg):
        self.events.append(f"{ts()}  {msg}")
        self.save()

    def save(self):
        json.dump(dict(steps=self.steps, events=self.events, overlap=self.overlap), open(STATE, "w"), indent=1)

    def set(self, k, status, note=None):
        s = self.steps[k]
        if status == "running" and s["status"] != "running":
            s["started"] = ts()
        if status in ("done", "skipped", "failed") and s["status"] not in ("done", "skipped", "failed"):
            s["finished"] = ts()
        if s["status"] != status or (note is not None and note != s["note"]):
            s["status"] = status
            if note is not None:
                s["note"] = note
            self.event(f"step {k} {status}" + (f": {note}" if note else ""))

    def launch(self, k):
        os.makedirs(LOGD, exist_ok=True)
        self.steps[k]["attempts"] += 1
        log = open(f"{LOGD}/overnight_step{k}_attempt{self.steps[k]['attempts']}.log", "w")
        self.procs[k] = subprocess.Popen([sys.executable] + SCRIPTS[k], stdout=log, stderr=subprocess.STDOUT)
        self.set(k, "running", f"attempt {self.steps[k]['attempts']}, pid {self.procs[k].pid}")

    def poll(self, k):
        """Returns None while running, else the exit code (one retry is done by the caller)."""
        p = self.procs.get(k)
        return None if p is None else p.poll()

    # -------------------------------------------------------------------------------------------------- chain state
    def chain(self):
        q = rd(Q)
        return dict(q=q, z0_done="=== DONE z0 rev" in q, complete="QUEUE COMPLETE" in q,
                    hold="HOLD" in rd(f"{RES}/expanded_PETPS_exp_build_bgcorr.log"),
                    proj=json.load(open(f"{RES}/expanded_PETPS_exp/part4_projection.json"))
                    if os.path.exists(f"{RES}/expanded_PETPS_exp/part4_projection.json") else None)

    def driver(self):
        s = rd(os.path.join(WORK, "followup_status.log"))
        out = rd(os.path.join(WORK, "followup_stdout.txt"))
        return dict(finished=bool(re.search(r" finished\s*$", s, re.M)), failed="Traceback" in out, text=s)

    # -------------------------------------------------------------------------------------------------- overlap rule
    def activity(self):
        """Intervals (start, end) in which lower-priority jobs ran: the Part C driver (from its log) and PP yield."""
        iv = []
        active_from = None
        for l in self.driver()["text"].splitlines():  # active except between 'waiting before' and 'resumed'
            if not re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ", l):
                continue
            t = dt.datetime.strptime(l[:19], "%Y-%m-%d %H:%M:%S")
            if " start '" in l or "resumed '" in l:
                active_from = active_from or t
            elif "waiting before '" in l or l.rstrip().endswith(" finished"):
                if active_from:
                    iv.append((active_from, t))
                active_from = None
        if active_from:
            iv.append((active_from, now()))
        st3 = self.steps["3"]
        if st3["started"]:
            a = dt.datetime.strptime(st3["started"], "%Y-%m-%d %H:%M")
            b = dt.datetime.strptime(st3["finished"], "%Y-%m-%d %H:%M") if st3["finished"] else now()
            iv.append((a, b))
        return iv

    def assess_overlap(self):
        q = rd(Q)
        runs = re.findall(r"^(\S+) exit=0 (\d+)s (\d\d-\d\d \d\d:\d\d:\d\d)$", q, re.M)
        proj = self.chain()["proj"]
        if not proj:
            return
        ratio_rev = proj["rev_train_ratio"]
        iv = self.activity()
        rows = []
        for name, secs, end in runs:
            p = f"{RES}/{name}/results.json"
            if not os.path.exists(p):
                continue
            ep = len(json.load(open(p))["history"])
            e = dt.datetime.strptime(f"2026-{end}", "%Y-%m-%d %H:%M:%S")
            b = e - dt.timedelta(seconds=int(secs))
            ov_s = sum(max(0.0, (min(e, y) - max(b, a)).total_seconds()) for a, y in iv)
            overl = ov_s > 60  # a run counts as overlapped if a lower-priority job ran for more than 60 s of it
            per = int(secs) / ep / (ratio_rev if "_rev_" in name else 1.0)
            rows.append(dict(run=name, epochs=ep, seconds=int(secs), sec_per_epoch_scaled=per, overlap=overl,
                             overlap_seconds=round(ov_s), start=str(b), end=str(e)))
        base = [r["sec_per_epoch_scaled"] for r in rows if not r["overlap"]]
        if not base:
            return
        b0 = float(np.median(base))
        ov = [r["sec_per_epoch_scaled"] / b0 for r in rows if r["overlap"]]
        ok = bool(ov) and max(ov) <= 1.20
        self.overlap = dict(baseline_sec_per_epoch=b0, ratios=[round(x, 3) for x in ov], ok=ok, n_overlap_runs=len(ov),
                            rows=rows, note=("no overlap run measured yet" if not ov else
                                             f"largest rise {100 * (max(ov) - 1):.1f}% over {len(ov)} overlapping runs"))
        json.dump(dict(ok=ok, ratios=self.overlap["ratios"], baseline_sec_per_epoch=b0, note=self.overlap["note"],
                       updated=ts()), open(f"{RES}/overnight_overlap_ok.json", "w"), indent=1)

    # -------------------------------------------------------------------------------------------------- headline numbers
    @staticmethod
    def arm_numbers(prefix_real, prefix_bg, seeds, manifest):
        out = {}
        for direction, sfx in (("original", ""), ("reverse", "_rev")):
            rs, bs = [], []
            for s in seeds[direction]:
                for lst, pre in ((rs, prefix_real), (bs, prefix_bg)):
                    p = f"{RES}/{pre}{sfx}_s{s}/results.json"
                    if os.path.exists(p):
                        lst.append(json.load(open(p)))
            if not rs or not bs:
                continue
            f = lambda L: [r["test_report"]["macro avg"]["f1-score"] for r in L]
            ms = lambda v: (st.mean(v), st.pstdev(v) if len(v) > 1 else float("nan"))
            real, pf = ms(f(rs)), ms(f(bs))
            gap, floor = real[0] - pf[0], max(real[1], pf[1])
            man = pd.read_csv(manifest.replace("_PETPS_exp", "_PETPS_rev_exp") if direction == "reverse" else manifest)
            classes = rs[0]["classes"]
            man = man[man["class"].isin(classes)]
            tr, te = man[man.split_video_holdout == "train"]["class"], man[man.split_video_holdout == "test"]["class"].values
            p_ = tr.value_counts(normalize=True).reindex(classes).values
            rng = np.random.default_rng(0)
            ch = float(np.mean([f1_score(te, rng.choice(classes, size=len(te), p=p_), labels=classes, average="macro",
                                         zero_division=0) for _ in range(1000)]))
            complete = len(rs) == len(bs) == len(seeds[direction])
            if not complete:
                T = "(incomplete)"
            elif gap > floor and real[0] > ch + 0.10:
                T = "T1"
            elif gap <= floor and pf[0] > ch + 0.10:
                T = "T2"
            elif real[0] <= ch + 0.10 and pf[0] <= ch + 0.10:
                T = "T3"
            else:
                T = "T4"
            rec = {kind: {c: st.mean([r["test_report"][c]["recall"] for r in L]) for c in classes} for kind, L in (("real", rs), ("particle-free", bs))}
            out[direction] = dict(real=real, pf=pf, gap=gap, floor=floor, chance=ch, T=T, rec=rec, n=(len(rs), len(bs)))
        return out

    def numbers_text(self):
        L = []
        c = self.chain()
        if c["proj"]:
            seeds = {"original": [int(s) for s in c["proj"]["original_seeds"].split()], "reverse": [0, 1, 2]}
            z = self.arm_numbers("real_tc_PETvsPS_z0_exp", "bg_tc_PETvsPS_z0_exp", seeds,
                                 "pipeline_output/manifest_recon_s0_z0_PETPS_exp.csv")
            for d, v in z.items():
                L.append(f"- z0 baseline, {d}: real {v['real'][0]:.3f}+/-{v['real'][1]:.3f}, particle-free {v['pf'][0]:.3f}+/-"
                         f"{v['pf'][1]:.3f}, gap {v['gap']:+.3f}, floor {v['floor']:.3f}, chance {v['chance']:.3f}, {v['T']} "
                         f"(runs {v['n'][0]}+{v['n'][1]}); recall real PET {v['rec']['real']['PET']:.3f} PS "
                         f"{v['rec']['real']['PS']:.3f}, particle-free PET {v['rec']['particle-free']['PET']:.3f} PS "
                         f"{v['rec']['particle-free']['PS']:.3f}")
            for arm in ("armG", "armN"):
                if os.path.exists(f"{RES}/overnight_arm_{arm}_exp.json"):
                    a = self.arm_numbers(f"real_tc_PETvsPS_z0_{arm}_exp", f"bg_tc_PETvsPS_z0_{arm}_exp",
                                         {"original": [0, 1, 2], "reverse": [0, 1, 2]},
                                         "pipeline_output/manifest_recon_s0_z0_PETPS_exp.csv")
                    for d, v in a.items():
                        L.append(f"- {arm}, {d}: real {v['real'][0]:.3f}+/-{v['real'][1]:.3f}, particle-free {v['pf'][0]:.3f}+/-"
                                 f"{v['pf'][1]:.3f}, gap {v['gap']:+.3f}, floor {v['floor']:.3f}, {v['T']} (runs {v['n'][0]}+{v['n'][1]}); "
                                 f"recall real PET {v['rec']['real']['PET']:.3f} PS {v['rec']['real']['PS']:.3f}")
                    k = re.search(r"^Reading: (\S+)", rd(f"{RES}/report_{arm}_exp.txt"), re.M)
                    if k:
                        L.append(f"- {arm} reading: {k.group(1)}")
        t = rd(f"{RES}/report_z0_trackCI_exp.txt")
        for line in t.splitlines():
            if line.strip().startswith("real ") and "[" in line:
                L.append(f"- z0 track intervals: {line.strip()}")
        y = json.load(open(f"{RES}/pp_yield_exp.json")) if os.path.exists(f"{RES}/pp_yield_exp.json") else None
        if y and "qualifies" in y:
            L.append("- PP yield: " + ", ".join(f"{k} {v['kept_after_15px']} patches / {v['tracks']} tracks / edge {v['edge_fraction']:.2f}"
                                               for k, v in y["out"].items()) + f" -> {'PP QUALIFIES' if y['qualifies'] else 'PP does not qualify'}")
        b2 = os.path.join(WORK, "partB2.json")
        if os.path.exists(b2):
            B = json.load(open(b2))
            fails = [f"{r['method']} at {r['dx_um']:.2f} um ({r['exact_round_trip_err_region']:.2e})" for r in B["rows"]
                     if r["disc"].startswith("50") and not r["new_gate_pass"]]
            L.append("- Part B under the new gate: " + ("PASS in every combination" if not fails else
                     "FAILS mechanically (round trip error not below 1e-5) for " + "; ".join(fails) +
                     "; no variant uses these combinations (V1 is unpadded at 6.9 um, V2 to V5 are at 3.45 um); recorded, "
                     "rule not rewritten, Part C continued as instructed"))
        return L

    # -------------------------------------------------------------------------------------------------- status file
    def write_status(self, final=False):
        c, drv = self.chain(), self.driver()
        running = [f"step {k} ({NAMES[k]})" for k, s in self.steps.items() if s["status"] == "running"]
        nxt = [f"step {k}" for k, s in self.steps.items() if s["status"] == "waiting"]
        nums = self.numbers_text()
        L = ["# Overnight status 2026-10-07/08", "", "## READ THIS FIRST", "",
             f"- Updated {ts()}{' (FINAL)' if final else ''}. No new step starts after 07:45; the step in progress is finished.",
             f"- Running: {', '.join(running) or 'nothing'}.", f"- Next: {', '.join(nxt) or 'nothing'}."]
        if c["hold"] and not os.path.exists(RELEASE):
            L.append("- The chain waits before the raw-intensity build (step 6 of the overnight order). To release it by hand, "
                     "create the empty file results/overnight_release_bgcorr.flag; the chain then builds and, if its own time "
                     "check passes, runs the intensity batch (ending before Fri 09:00).")
        if self.overlap:
            L.append(f"- Overlap rule (training run times may rise at most 20%): {self.overlap.get('note', '')}; "
                     f"overlap {'allowed' if self.overlap.get('ok') else 'not allowed'} for the PP yield job. The Part C driver "
                     "runs alongside training at idle priority as arranged before the overnight message; it was not paused "
                     "or stopped (an OS-level pause was refused by the session's permission check).")
        L += nums + ["", "## Steps", "", "| step | what | status | started | finished | note |", "|---|---|---|---|---|---|"]
        for k, s in self.steps.items():
            L.append(f"| {k} | {NAMES[k]} | {s['status']} | {s['started'] or ''} | {s['finished'] or ''} | {s['note']} |")
        L += ["", "## Chain", "", "```", *c["q"].strip().splitlines()[-12:], "```", "",
              f"Part C driver: {'finished' if drv['finished'] else ('FAILED, see its log' if drv['failed'] else 'running or waiting')}; "
              f"last lines:", "```", *drv["text"].strip().splitlines()[-6:], "```", "",
              f"train_compact_cnn.py sha256 at the start: {SHA0}; now: {sha('train_compact_cnn.py')}"
              f"{' (unchanged)' if sha('train_compact_cnn.py') == SHA0 else ' (CHANGED)'}",
              f"train_compact_cnn_bright.py sha256: {sha('train_compact_cnn_bright.py') if os.path.exists('train_compact_cnn_bright.py') else '-'}",
              f"Brightness flags unit test: {self.unit_test()}",
              f"Available RAM {avail_mb()} MB; free disk {shutil.disk_usage(D).free / 1e9:.0f} GB", "", "## Timeline", ""]
        L += [f"- {e}" for e in self.events[-80:]]
        if final:
            L += ["", "## FINAL STATUS BLOCK", "", "| step | status | note |", "|---|---|---|"]
            L += [f"| {k} {NAMES[k]} | {s['status']} | {s['note']} |" for k, s in self.steps.items()]
            L += ["", "Headline numbers:"] + (nums or ["- none"])
            L += ["", "New files: see the list in results/overnight_new_files_exp.txt.", "",
                  "First three things for Phila to look at:",
                  "1. results/report_z0_trackCI_exp.txt (z0 T readings with the seed floor, then the track intervals, per class).",
                  "2. results/report_armG_exp.txt (brightness arm G, K reading and its caveat).",
                  "3. results/report_recon_check.txt and results/recon_check/ (Part C sheets, Part B re-score, I90/I0)."]
        text = "\n".join(L)
        text = text.replace("\u2014", "-")
        open(STATUS, "w", encoding="utf-8").write(text + "\n")
        self.last_write = time.time()

    @staticmethod
    def unit_test():
        m = re.search(r"overall: (\w+)", rd(f"{RES}/test_bright_flags_exp.txt"))
        return m.group(1) if m else "not finished"

    def new_files(self):
        pats = ["_exp", "overnight", "recon_check", "train_compact_cnn_bright"]
        out = []
        for root in (".", "results", "scripts", "pipeline_output"):
            for f in sorted(os.listdir(root)):
                p = os.path.join(root, f)
                if any(x in f for x in pats) and os.path.getmtime(p) > dt.datetime(2026, 10, 7, 16, 0).timestamp():
                    out.append(p)
        open(f"{RES}/overnight_new_files_exp.txt", "w").write("\n".join(out) + "\n")

    # -------------------------------------------------------------------------------------------------- main loop
    def tick(self):
        c, drv = self.chain(), self.driver()
        self.assess_overlap()
        late = now() >= CUTOFF
        # step 0
        if self.steps["0"]["status"] == "waiting":
            self.set("0", "running")
        if c["z0_done"] and (drv["finished"] or drv["failed"]) and self.steps["0"]["status"] == "running":
            self.set("0", "done", "z0 batch finished; Part C driver " + ("finished" if drv["finished"] else "FAILED (see its log)"))
        # steps 1 to 5: launch / poll / retry once
        for k in ("1", "2", "3", "4", "5"):
            s = self.steps[k]
            if s["status"] == "running" and k in self.procs:
                rc = self.poll(k)
                if rc is None:
                    continue
                if rc == 0:
                    self.set(k, "done", s["note"].split(",")[0] + ", exit 0")
                elif s["attempts"] < 2 and not late:
                    self.event(f"step {k} exited with {rc}; retrying once")
                    self.launch(k)
                else:
                    self.set(k, "failed", f"exit {rc} after {s['attempts']} attempts; skipped and logged")
            elif s["status"] == "running" and k not in self.procs:  # coordinator restarted while it ran
                self.set(k, "failed", "coordinator restarted while the step ran; see its log")
        ready = {k: self.steps[k]["status"] in ("done", "skipped", "failed") for k in self.steps}
        if late:
            for k in ("1", "2", "3", "4", "5", "6"):
                if self.steps[k]["status"] == "waiting":
                    self.set(k, "skipped", "not started: 07:45 cutoff")
        else:
            if self.steps["1"]["status"] == "waiting" and c["z0_done"]:
                self.launch("1")
            if self.steps["2"]["status"] == "waiting" and ready["1"]:
                ut = rd(f"{RES}/test_bright_flags_exp.txt")
                if "overall: PASS" in ut:
                    self.launch("2")
                elif "overall: FAIL" in ut:
                    self.set("2", "skipped", "cannot run: unit test of train_compact_cnn_bright.py failed")
            if self.steps["3"]["status"] == "waiting" and (drv["finished"] or drv["failed"]):
                self.launch("3")
            if self.steps["4"]["status"] == "waiting" and ready["3"] and ready["2"]:
                y = json.load(open(f"{RES}/pp_yield_exp.json")) if os.path.exists(f"{RES}/pp_yield_exp.json") else {}
                if self.steps["3"]["status"] != "done" or "qualifies" not in y:
                    self.set("4", "skipped", "PP yield check did not finish")
                elif not y["qualifies"]:
                    self.set("4", "skipped", "PP does not qualify (yield table in results/report_PP_yield_exp.txt)")
                elif not os.path.exists(SCRIPTS["4"][0]):
                    self.set("4", "skipped", "script not ready")
                elif avail_mb() >= 3000 and shutil.disk_usage(D).free / 1e9 >= 20:
                    self.launch("4")
            if self.steps["5"]["status"] == "waiting" and ready["4"] and ready["2"]:
                if "overall: PASS" in rd(f"{RES}/test_bright_flags_exp.txt"):
                    self.launch("5")
                else:
                    self.set("5", "skipped", "cannot run: unit test not passed")
            if self.steps["6"]["status"] == "waiting" and all(ready[k] for k in ("1", "2", "3", "4", "5")):
                if avail_mb() >= 3000 and shutil.disk_usage(D).free / 1e9 >= 20:
                    open(RELEASE, "w").write(f"released by scripts/overnight_coordinator.py at {ts()}\n")
                    self.set("6", "running", "release flag written; the chain builds and runs the intensity batch")
        if self.steps["6"]["status"] == "running" and c["complete"]:
            self.set("6", "done", "chain reports QUEUE COMPLETE")
        busy = any(s["status"] == "running" for k, s in self.steps.items() if k != "0") or self.steps["0"]["status"] == "running"
        waiting = any(s["status"] == "waiting" for s in self.steps.values())
        if (late or not waiting) and not busy:
            self.new_files()
            self.write_status(final=True)
            self.final_written = True
            return False
        if time.time() - self.last_write > 120:
            self.write_status()
        return True

    def run(self):
        self.event("coordinator started")
        self.write_status()
        while True:
            try:
                if not self.tick():
                    self.event("final status block written; coordinator ends")
                    break
            except Exception as e:  # keep going; record the problem
                self.event(f"coordinator error: {type(e).__name__}: {e}")
            time.sleep(30)


if __name__ == "__main__":
    Coordinator().run()
