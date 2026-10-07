"""
p_eval_checkpoints.py

Read-only re-evaluation of saved best.pt checkpoints (no training). For each arm,
real (video-holdout test) and background-only (its own test split), all 5 seeds:
rebuilds the run's inputs from the args stored in its results.json (manifest,
patch dir, channel mode, split), recomputes the training-split normalisation
stats, loads best.pt, predicts the test set. Reports seed-averaged per-class
precision / recall / F1, the seed-averaged 7x7 confusion matrix (mean counts),
and the fraction of NON-PET test patches predicted PET. Cross-checks every
recomputed confusion matrix against the one saved in results.json.
Writes results/p_report.txt.
"""
import os
import json
import importlib.util
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(DATA_DIR)
spec = importlib.util.spec_from_file_location("tcc", "train_compact_cnn.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ARMS = [("intensity_bgcorr_vnorm", "intbv_c849"), ("z0", "recons0z0_c849"), ("recon_s0", "recons0_r180"),
        ("recon_biref", "reconbiref_c849"), ("pol4_bgcorr_vnorm", "pol4bgcorrvnorm_c849"), ("I45", "i45bv_c849"),
        ("pol4_bgcorr", "pol4bgcorr_c849"), ("Stokes", "stokes_c849")]
IN_CH = {"intensity_bgcorr_vnorm": 1, "recon_s0": 1, "recon_biref": 3, "pol4_bgcorr_vnorm": 4,
         "i45_bgcorr_vnorm": 1, "pol4_bgcorr": 4, "stokes": 3}


def evaluate(run_dir):
    saved = json.load(open(os.path.join(run_dir, "results.json")))
    a = saved["args"]
    split_col = m.SPLIT_CONFIG[a["split"]]["split_column"]
    man = pd.read_csv(a["manifest"])
    classes = saved["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    train_df, test_df = man[man[split_col] == "train"], man[man[split_col] == "test"]
    mean, std = m.compute_channel_stats(train_df, a["patch_dir"], c2i, a["channels"])
    model = m.CompactHoloCNN(IN_CH[a["channels"]], len(classes), a["width"], a["dropout"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    model.eval()
    ds = m.PatchDataset(test_df, a["patch_dir"], c2i, channels=a["channels"], mean=mean, std=std)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=64, shuffle=False):
            preds += model(x).argmax(1).tolist()
            labels += y.tolist()
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(len(classes))), zero_division=0)
    matches = np.array_equal(cm, np.array(saved["confusion_matrix"]))
    return classes, cm, p, r, f, matches


def main():
    lines, mismatches = [], 0
    for name, tag in ARMS:
        for side, prefix in (("REAL", f"real_vh_{tag}"), ("BACKGROUND-ONLY", f"bg_{tag}")):
            cms, ps, rs, fs = [], [], [], []
            for s in range(5):
                classes, cm, p, r, f, ok = evaluate(f"results/{prefix}_s{s}")
                mismatches += not ok
                cms.append(cm); ps.append(p); rs.append(r); fs.append(f)
            cm_mean = np.mean(cms, axis=0)
            pet = classes.index("PET")
            nonpet = [i for i in range(len(classes)) if i != pet]
            frac = np.mean([cm[nonpet, pet].sum() / cm[nonpet].sum() for cm in cms])
            n_test = int(cms[0].sum())
            lines.append(f"\n{name} - {side} ({prefix}, 5 seeds, {n_test} test patches)")
            lines.append(f"  non-PET test patches predicted PET: {frac:.3f} "
                         f"({cm_mean[nonpet, pet].sum():.1f} of {cms[0][nonpet].sum()})")
            lines.append(f"  {'class':5s} {'precision':>9s} {'recall':>7s} {'F1':>6s}   confusion matrix, mean counts "
                         f"(rows true, cols predicted: {' '.join(classes)})")
            for i, c in enumerate(classes):
                row = " ".join(f"{v:6.1f}" for v in cm_mean[i])
                lines.append(f"  {c:5s} {np.mean(ps, 0)[i]:9.3f} {np.mean(rs, 0)[i]:7.3f} {np.mean(fs, 0)[i]:6.3f}   {row}")
            print(lines[-len(classes) - 3])
    lines.insert(0, f"Recomputed confusion matrices that differ from results.json: {mismatches} of {len(ARMS) * 10}")
    with open("results/p_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(lines[0])


if __name__ == "__main__":
    main()
