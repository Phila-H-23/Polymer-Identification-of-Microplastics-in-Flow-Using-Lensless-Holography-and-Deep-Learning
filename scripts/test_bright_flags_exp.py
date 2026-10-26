"""Unit test of train_compact_cnn_bright.py (copy with --gain-range and --meannorm), writes
results/test_bright_flags_exp.txt. Read-only for every existing file; training outputs go to the session scratchpad.
  (a) both flags off: one fixed-seed epoch (seed 0, expanded z0 PET vs PS, original split) with train_compact_cnn.py
      and with the copy; best.pt tensors and every results.json field except 'args' must be identical
  (b) --gain-range 0.3: per-sample ratio (augmented patch mean / unaugmented patch mean) spreads over [0.7, 1.3]
      (expected std 0.3 / sqrt(3) = 0.173), against [0.95, 1.05] (std 0.029) without the flag
  (c) --meannorm: patch mean 1 within 1e-6 for training (augmented) and test (unaugmented) patches
Child training processes run at IDLE priority."""
import os
import sys
import json
import random
import hashlib
import subprocess
import importlib.util
import ctypes
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # IDLE
import numpy as np
import pandas as pd
import torch

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
WORK = (r"C:\Users\haany\AppData\Local\Temp\claude\c--Users-haany-OneDrive-research-project-dataset-10jan23"
        r"\075ab033-0168-4afa-a675-c38e4f73b481\scratchpad\unit_bright")
PO = "pipeline_output"
MAN, PDIR = f"{PO}/manifest_recon_s0_z0_PETPS_exp.csv", f"{PO}/patches_recon_s0_z0_PETPS_exp"
CW = f"{PO}/class_weights_twoclass_PETvsPS_exp.json"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(script, out):
    cmd = [sys.executable, script, "--channels", "recon_s0", "--split", "video_holdout", "--manifest", MAN,
           "--patch-dir", PDIR, "--class-weights", CW, "--classes", "PET,PS", "--epochs", "1", "--patience", "30",
           "--seed", "0", "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=0x40)
    return r.returncode, r.stdout[-400:], r.stderr[-400:]


def main():
    os.makedirs(WORK, exist_ok=True)
    L = ["UNIT TEST: train_compact_cnn_bright.py", f"sha256 train_compact_cnn.py        {sha('train_compact_cnn.py')}",
         f"sha256 train_compact_cnn_bright.py {sha('train_compact_cnn_bright.py')}"]
    # (a)
    oa, ob = os.path.join(WORK, "orig"), os.path.join(WORK, "copy")
    ra, rb = run("train_compact_cnn.py", oa), run("train_compact_cnn_bright.py", ob)
    ok_a = ra[0] == 0 and rb[0] == 0
    if ok_a:
        sa, sb = torch.load(f"{oa}/best.pt"), torch.load(f"{ob}/best.pt")
        same_w = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
        ja, jb = json.load(open(f"{oa}/results.json")), json.load(open(f"{ob}/results.json"))
        same_j = all(ja[k] == jb[k] for k in ja if k != "args")
        extra = sorted(set(jb["args"]) - set(ja["args"]))
        same_args = all(ja["args"][k] == jb["args"][k] for k in ja["args"] if k != "out")
        ok_a = same_w and same_j and same_args
        L.append(f"(a) flags off, 1 epoch, seed 0: best.pt identical {same_w}; results.json identical except args {same_j}; "
                 f"shared args identical {same_args}; extra args in the copy {extra}; test macro-F1 "
                 f"{ja['test_report']['macro avg']['f1-score']:.6f} vs {jb['test_report']['macro avg']['f1-score']:.6f} -> "
                 f"{'PASS' if ok_a else 'FAIL'}")
    else:
        L.append(f"(a) a run failed: {ra} {rb}")
    # (b), (c)
    spec = importlib.util.spec_from_file_location("tb", "train_compact_cnn_bright.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    man = pd.read_csv(MAN)
    man = man[man["class"].isin(["PET", "PS"])]
    c2i = {"PET": 0, "PS": 1}
    tr, te = man[man.split_video_holdout == "train"], man[man.split_video_holdout == "test"]
    idx = np.random.default_rng(0).choice(len(tr), 300, replace=False)
    plain = tb.PatchDataset(tr, PDIR, c2i, channels="recon_s0")
    base = np.array([float(plain[i][0].mean()) for i in idx])
    res = {}
    for g in (None, 0.3):
        random.seed(1)
        torch.manual_seed(1)
        ds = tb.PatchDataset(tr, PDIR, c2i, channels="recon_s0", augment=True, gain_range=g)
        res[g] = np.array([float(ds[i][0].mean()) for i in idx]) / base
    r0, r3 = res[None], res[0.3]
    ok_b = (r3.min() >= 0.69 and r3.max() <= 1.31 and 0.14 < r3.std() < 0.21 and r0.min() >= 0.94 and r0.max() <= 1.06)
    L.append(f"(b) ratio augmented / unaugmented patch mean over 300 training patches: default +/-5% gain: range "
             f"{r0.min():.3f} to {r0.max():.3f}, std {r0.std():.4f} (expected about 0.029); --gain-range 0.3: range "
             f"{r3.min():.3f} to {r3.max():.3f}, std {r3.std():.4f} (expected about 0.173) -> {'PASS' if ok_b else 'FAIL'}")
    devs = []
    for df, aug in ((tr, True), (te, False)):
        ds = tb.PatchDataset(df, PDIR, c2i, channels="recon_s0", augment=aug, meannorm=True)
        sel = np.random.default_rng(1).choice(len(df), 200, replace=False)
        devs.append(max(abs(float(ds[i][0].double().mean()) - 1.0) for i in sel))
    ok_c = max(devs) <= 1e-6
    mean, std = tb.compute_channel_stats(tr, PDIR, c2i, "recon_s0", meannorm=True)
    L.append(f"(c) --meannorm: largest |patch mean - 1| training (augmented) {devs[0]:.2e}, test {devs[1]:.2e} -> "
             f"{'PASS' if ok_c else 'FAIL'}; training channel stats with --meannorm: mean {float(mean[0]):.4f}, "
             f"std {float(std[0]):.4f}")
    L.append(f"overall: {'PASS' if ok_a and ok_b and ok_c else 'FAIL'}")
    open("results/test_bright_flags_exp.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
