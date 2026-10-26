"""Overnight step 1: z0 runs missing a results.json are retried once (same command line as
scripts/run_expanded_PETPS_v2.sh, output <dir>_retry1, recorded in results/overnight_z0_retries_exp.json), then
scripts/eval_track_ci_exp.py z0 (per-patch predictions from best.pt, macro-F1 check, track bootstrap)."""
import os
import sys
import json
import subprocess

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
PO = "pipeline_output"


def main():
    p = json.load(open("results/expanded_PETPS_exp/part4_projection.json"))
    seeds = {"original": [int(s) for s in p["original_seeds"].split()], "reverse": [0, 1, 2]}
    retries = {}
    rt = "results/overnight_z0_retries_exp.json"
    if os.path.exists(rt):
        retries = json.load(open(rt))
    for direction, ss in seeds.items():
        sfx = "_rev" if direction == "reverse" else ""
        for s in ss:
            for kind in ("real", "bg"):
                d = f"results/{kind}_tc_PETvsPS_z0_exp{sfx}_s{s}"
                if os.path.exists(f"{d}/results.json") or d in retries:
                    continue
                man = (f"{PO}/manifest_recon_s0_z0_PETPS{sfx}_exp.csv" if kind == "real"
                       else f"{PO}/manifest_background_recon_s0_z0_PETPS{sfx}_exp.csv")
                pdir = f"{PO}/patches_recon_s0_z0_PETPS_exp" if kind == "real" else f"{PO}/patches_background_recon_s0_z0_PETPS_exp"
                out = d + "_retry1"
                cmd = [sys.executable, "train_compact_cnn.py", "--channels", "recon_s0", "--split",
                       "video_holdout" if kind == "real" else "particle", "--manifest", man, "--patch-dir", pdir,
                       "--class-weights", f"{PO}/class_weights_twoclass_PETvsPS{sfx}_exp.json", "--classes", "PET,PS",
                       "--epochs", "150", "--patience", "30", "--seed", str(s), "--out", out]
                os.makedirs("results/logs_overnight_exp", exist_ok=True)
                with open(f"results/logs_overnight_exp/{os.path.basename(out)}.log", "w") as fh:
                    rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
                print(f"retry {d}: exit {rc}", flush=True)
                if rc == 0 and os.path.exists(f"{out}/results.json"):
                    retries[d] = out
                json.dump(retries, open(rt, "w"), indent=1)
    rc = subprocess.run([sys.executable, "scripts/eval_track_ci_exp.py", "z0"]).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
