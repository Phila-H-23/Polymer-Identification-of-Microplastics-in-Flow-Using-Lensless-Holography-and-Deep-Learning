"""Waits until scripts/overnight_coordinator.py has written its final status block, then appends to
results/overnight_status.md what that block does not hold: step 4 (PET vs PP) headline numbers per fold, particle-free
recall of the arms, and the list of failed or retried runs. Read-only for everything else."""
import os
import re
import json
import time

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
RES = "results"


def rd(p):
    return open(p, encoding="utf-8", errors="ignore").read() if os.path.exists(p) else ""


def main():
    while "final status block written" not in rd(f"{RES}/overnight_state_exp.json"):
        time.sleep(120)
    time.sleep(10)
    L = ["", "## FINAL STATUS BLOCK, ADDENDUM (written after the coordinator's block)", ""]
    s = f"{RES}/PETvsPP_summary_exp.json"
    if os.path.exists(s):
        L.append("Step 4, PET vs PP on z0, per fold (pre-registered T reading with the seed floor; track intervals in "
                 "results/report_PETvsPP_exp.txt):")
        for fold, v in json.load(open(s)).items():
            rec = v.get("rec", {})
            r_ = rec.get("real", [])
            p_ = rec.get("particle-free", [])
            cl = v.get("classes", ["PET", "PP"])
            L.append(f"- held out {fold}: real {v['real'][0]:.3f}+/-{v['real'][1]:.3f}, particle-free {v['pf'][0]:.3f}+/-"
                     f"{v['pf'][1]:.3f}, gap {v['gap']:+.3f}, floor {v['floor']:.3f}, chance {v['chance']:.3f}, {v['T']}; "
                     f"track CI gap [{v['gap_ci'][1]:+.3f}, {v['gap_ci'][2]:+.3f}]; recall real "
                     + ", ".join(f"{c} {r_[i][0]:.3f}" for i, c in enumerate(cl) if i < len(r_)) + "; particle-free "
                     + ", ".join(f"{c} {p_[i][0]:.3f}" for i, c in enumerate(cl) if i < len(p_)))
        sz = re.findall(r"^  fold .*$", rd(f"{RES}/report_PETvsPP_exp.txt"), re.M)
        L += ["Size-only baseline per fold:"] + [f"- {x.strip()}" for x in sz]
    else:
        L.append("Step 4: no summary file (see its status in the table above and results/logs_overnight_exp/pp_folds_exp.log).")
    for arm in ("armG", "armN"):
        p = f"{RES}/{arm}_trackCI_summary_exp.json"
        if os.path.exists(p):
            for d, v in json.load(open(p)).items():
                pf = v.get("rec", {}).get("particle-free", [])
                cl = v.get("classes", ["PET", "PS"])
                L.append(f"- {arm} {d}: particle-free recall " + ", ".join(f"{c} {pf[i][0]:.3f}" for i, c in enumerate(cl) if i < len(pf)))
    fails = []
    for p in (f"{RES}/overnight_arm_armG_exp.json", f"{RES}/overnight_arm_armN_exp.json"):
        if os.path.exists(p):
            fails += json.load(open(p)).get("failed", [])
    rt = f"{RES}/overnight_z0_retries_exp.json"
    retries = json.load(open(rt)) if os.path.exists(rt) else {}
    logs = rd(f"{RES}/logs_overnight_exp/run_arm_exp.log") + rd(f"{RES}/logs_overnight_exp/pp_folds_exp.log")
    failed_lines = [l for l in logs.splitlines() if "FAILED" in l or "failed twice" in l]
    ev = [l for l in json.load(open(f"{RES}/overnight_state_exp.json"))["events"] if "fail" in l or "retry" in l or "error" in l]
    L += ["", "Failed or retried:", f"- runs skipped after a failed retry: {fails or 'none'}",
          f"- z0 runs retried in step 1: {retries or 'none'}",
          f"- run attempts that failed (logs): {failed_lines or 'none'}",
          f"- coordinator events: {ev or 'none'}"]
    text = "\n".join(L).replace("\u2014", "-")
    assert "significant" not in text.lower()
    with open(f"{RES}/overnight_status.md", "a", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
