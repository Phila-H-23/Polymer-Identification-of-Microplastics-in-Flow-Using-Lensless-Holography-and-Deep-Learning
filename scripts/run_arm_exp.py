"""Overnight arms G and N (z0, PET vs PS, expanded set) with train_compact_cnn_bright.py; the original training script
is not used or changed. Same splits, class-weight files, regime and command line as the z0 baseline
(scripts/run_expanded_PETPS_v2.sh: --classes PET,PS --epochs 150 --patience 30, other settings at their defaults),
plus the arm's flag:
  armG  --gain-range 0.3            (per-sample gain uniform in [0.7, 1.3] instead of +/-5%)
  armN  --meannorm                  (patch divided by its own mean after augmentation; +/-5% gain kept)
Original direction seeds 0 to 2, then reverse direction seeds 0 to 2; real then particle-free per seed. A failed run is
retried once into <out>_retry1, then skipped and logged. Afterwards: scripts/eval_track_ci_exp.py <arm>, then the K
readings -> results/report_<arm>_exp.txt.
usage: python scripts/run_arm_exp.py armG | armN"""
import os
import sys
import json
import time
import statistics as st
import subprocess

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
PO = "pipeline_output"
FLAGS = {"armG": ["--gain-range", "0.3"], "armN": ["--meannorm"]}
DESC = {"armG": "per-sample gain uniform in [0.7, 1.3] (--gain-range 0.3)",
        "armN": "per-patch mean normalisation (--meannorm), +/-5% gain kept"}
LOGD = "results/logs_overnight_exp"


def cmd(kind, direction, seed, out, arm):
    sfx = "_PETPS_rev_exp" if direction == "reverse" else "_PETPS_exp"
    if kind == "real":
        man, pdir, split = f"{PO}/manifest_recon_s0_z0{sfx}.csv", f"{PO}/patches_recon_s0_z0_PETPS_exp", "video_holdout"
    else:
        man, pdir, split = f"{PO}/manifest_background_recon_s0_z0{sfx}.csv", f"{PO}/patches_background_recon_s0_z0_PETPS_exp", "particle"
    cw = f"{PO}/class_weights_twoclass_PETvsPS{'_rev' if direction == 'reverse' else ''}_exp.json"
    return [sys.executable, "train_compact_cnn_bright.py", "--channels", "recon_s0", "--split", split, "--manifest", man,
            "--patch-dir", pdir, "--class-weights", cw, "--classes", "PET,PS", "--epochs", "150", "--patience", "30",
            "--seed", str(seed), *FLAGS[arm], "--out", out]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(f"{LOGD}/run_arm_exp.log", "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


def main(arm):
    os.makedirs(LOGD, exist_ok=True)
    cfg_path = f"results/overnight_arm_{arm}_exp.json"
    runs = {"original": {"real": {}, "particle-free": {}}, "reverse": {"real": {}, "particle-free": {}}}
    failed = []
    for direction in ("original", "reverse"):
        for seed in (0, 1, 2):
            for kind, pre in (("real", "real"), ("particle-free", "bg")):
                base = f"results/{pre}_tc_PETvsPS_z0_{arm}_exp{'_rev' if direction == 'reverse' else ''}_s{seed}"
                done = None
                for attempt, out in enumerate((base, base + "_retry1")):
                    if os.path.exists(f"{out}/results.json"):
                        done = out
                        break
                    t0 = time.time()
                    name = os.path.basename(out)
                    with open(f"{LOGD}/{name}.log", "w", encoding="utf-8") as fh:
                        rc = subprocess.run(cmd(kind, direction, seed, out, arm), stdout=fh, stderr=subprocess.STDOUT).returncode
                    ok = rc == 0 and os.path.exists(f"{out}/results.json")
                    log(f"{arm} {name} exit={rc} {time.time() - t0:.0f}s {'ok' if ok else 'FAILED'}")
                    if ok:
                        done = out
                        break
                if done:
                    runs[direction][kind][seed] = done
                else:
                    failed.append(base)
                    log(f"{arm} {base}: failed twice, skipped")
                json.dump(dict(arm=arm, runs=runs, failed=failed), open(cfg_path, "w"), indent=1)
    log(f"{arm} training finished; failed {failed or 'none'}")
    subprocess.run([sys.executable, "scripts/eval_track_ci_exp.py", arm], stdout=open(f"{LOGD}/eval_{arm}.log", "w"),
                   stderr=subprocess.STDOUT)
    k_report(arm)
    log(f"{arm} report written")


def k_report(arm):
    """K readings fixed in the overnight order. Margin per direction = larger real seed std (population std, as in
    every report) of the arm and of the z0 baseline over the seeds both have; comparison of means over those seeds."""
    S = json.load(open(f"results/{arm}_trackCI_summary_exp.json"))
    cfg = json.load(open(f"results/overnight_arm_{arm}_exp.json"))
    f1 = lambda d: json.load(open(f"{d}/results.json"))["test_report"]["macro avg"]["f1-score"]
    L = [f"ARM {arm[-1]} ({DESC[arm]}), z0, PET vs PS, expanded set, track-based validation, seeds 0 to 2 per direction",
         "Same splits, class-weight files (class_weights_twoclass_PETvsPS[_rev]_exp.json) and regime as the z0 baseline;",
         "real and particle-free with the same augmentation. Script train_compact_cnn_bright.py (flags off = original).",
         "Readings fixed before the result: margin per direction = larger real seed std of this arm and of the z0 baseline",
         "over the seeds both have; K1: arm real macro-F1 >= z0 real macro-F1 - margin in both directions AND the arm reads",
         "T1 in both; K2: the arm reads T2 or T3 in either direction; K3: otherwise."]
    ok_dirs, T_arm, rows = [], {}, []
    for direction in ("original", "reverse"):
        s = S.get(direction)
        if not s:
            L.append(f"{direction}: no complete result")
            continue
        T_arm[direction] = s["T"]
        seeds_arm = sorted(int(k) for k in cfg["runs"][direction]["real"])
        base_pref = f"results/real_tc_PETvsPS_z0_exp{'_rev' if direction == 'reverse' else ''}"
        shared = [k for k in seeds_arm if os.path.exists(f"{base_pref}_s{k}/results.json")]
        a = [f1(cfg["runs"][direction]["real"][str(k)] if str(k) in cfg["runs"][direction]["real"]
                else cfg["runs"][direction]["real"][k]) for k in shared]
        z = [f1(f"{base_pref}_s{k}") for k in shared]
        sa, sz = (st.pstdev(a) if len(a) > 1 else float("nan")), (st.pstdev(z) if len(z) > 1 else float("nan"))
        margin = max(sa, sz)
        ok = st.mean(a) >= st.mean(z) - margin
        ok_dirs.append(ok)
        rows.append(f"{direction}: seeds both have {shared}; arm real {st.mean(a):.4f} (std {sa:.4f}), z0 real {st.mean(z):.4f} "
                    f"(std {sz:.4f}), margin {margin:.4f}; arm >= z0 - margin: {ok}; arm T reading {s['T']} (real "
                    f"{s['real'][0]:.4f}, particle-free {s['pf'][0]:.4f}, gap {s['gap']:+.4f}, floor {s['floor']:.4f}, chance "
                    f"{s['chance']:.4f}); track CI real [{s['real_ci'][1]:.4f}, {s['real_ci'][2]:.4f}], particle-free "
                    f"[{s['pf_ci'][1]:.4f}, {s['pf_ci'][2]:.4f}], gap [{s['gap_ci'][1]:+.4f}, {s['gap_ci'][2]:+.4f}]")
    L += rows
    if len(T_arm) == 2 and len(ok_dirs) == 2:
        if all(ok_dirs) and all(t == "T1" for t in T_arm.values()):
            K = "K1"
        elif any(t in ("T2", "T3") for t in T_arm.values()):
            K = "K2"
        else:
            K = "K3"
    else:
        K = "(incomplete: a direction has no complete result)"
    L.append(f"Reading: {K}")
    if arm == "armG":
        L.append("Note: K1 means only that the z0 result survives a +/-30% per-sample gain; it does not rule out brightness as "
                 "a cue (a gain changes the patch mean and its contrast together and leaves the pattern of brightness inside "
                 "the patch untouched).")
    else:
        L.append("Note: K1 means only that the z0 result survives dividing each patch by its own mean; it does not rule out "
                 "brightness as a cue in other forms (the pattern of brightness inside the patch is kept).")
    L += ["", "Per-class values and track intervals (from scripts/eval_track_ci_exp.py):"]
    L += open(f"results/{arm}_trackCI_exp.txt", encoding="utf-8").read().splitlines()
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    open(f"results/report_{arm}_exp.txt", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
