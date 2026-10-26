"""Item 46, Part 0 (light, idle priority, waits for 1.5 GB free RAM): how well the width-32 models fit their own
training data. For results/real_tc_PETvsPS_z0depthD_exp_s0 and _exp_rev_s0 (arm D) and the z0 expanded real runs
results/real_tc_PETvsPS_z0_exp_s0 and _exp_rev_s0: best.pt evaluated in eval mode on the TRAINING split patches,
unshifted (arm D: pipeline_output/patches_depth_z0_PETPS_exp/2.250, byte copies of the z0 patches), no augmentation,
channel statistics from the training split as in the runs. Writes results/item46_part0_trainfit.txt."""
import os
import sys
import json
import time

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import item46_chain as c46  # noqa: E402  (gate)

RUNS = [("arm D, width 32", "original", "results/real_tc_PETvsPS_z0depthD_exp_s0"),
        ("arm D, width 32", "reverse", "results/real_tc_PETvsPS_z0depthD_exp_rev_s0"),
        ("z0, width 32", "original", "results/real_tc_PETvsPS_z0_exp_s0"),
        ("z0, width 32", "reverse", "results/real_tc_PETvsPS_z0_exp_rev_s0")]
OUT = "results/item46_part0_trainfit.txt"


def main():
    c46.gate("Part 0")
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score, recall_score
    import eval_track_ci_exp as ev
    m = ev.module("train_compact_cnn.py")
    L = ["ITEM 46, PART 0: TRAINING-SET FIT OF THE WIDTH-32 MODELS (seed 0; best.pt, eval mode, training split, unshifted, no "
         "augmentation)", "", f"{'run':42s} {'arm':16s} {'direction':9s} {'train F1':>8s} {'recall PET':>10s} {'recall PS':>9s} "
         f"{'val F1':>7s} {'test F1':>7s}  n train"]
    f1s = []
    for arm, direction, d in RUNS:
        if not os.path.exists(f"{d}/best.pt"):
            L.append(f"{os.path.basename(d):42s} best.pt missing: skipped")
            continue
        saved = json.load(open(f"{d}/results.json"))
        a = saved["args"]
        pdir = os.path.join(a["patch_dir"], "2.250") if a.get("depth_aug", "none") != "none" else a["patch_dir"]
        man = pd.read_csv(a["manifest"])
        man = man[man["class"].isin(a["classes"].split(","))]
        classes = saved["classes"]
        c2i = {c: i for i, c in enumerate(classes)}
        tr = man[man[m.SPLIT_CONFIG[a["split"]]["split_column"]] == "train"].reset_index(drop=True)
        mean, std = m.compute_channel_stats(tr, pdir, c2i, a["channels"])
        net = m.CompactHoloCNN(1, len(classes), a["width"], a["dropout"])
        net.load_state_dict(torch.load(f"{d}/best.pt", map_location="cpu"))
        net.eval()
        ds = m.PatchDataset(tr, pdir, c2i, channels=a["channels"], mean=mean, std=std)
        pred = []
        with torch.no_grad():
            for x, _ in DataLoader(ds, batch_size=64, shuffle=False):
                pred += net(x).argmax(1).tolist()
        y = [c2i[c] for c in tr["class"]]
        f1 = f1_score(y, pred, average="macro")
        rec = recall_score(y, pred, average=None, labels=[0, 1])
        f1s.append(f1)
        L.append(f"{os.path.basename(d):42s} {arm:16s} {direction:9s} {f1:8.4f} {rec[0]:10.3f} {rec[1]:9.3f} "
                 f"{saved['best_val_macro_f1']:7.4f} {saved['test_report']['macro avg']['f1-score']:7.4f}  {len(tr)}")
    if len(f1s) < 4:
        R = f"incomplete ({len(f1s)} of 4 runs)"
    elif all(f >= 0.98 for f in f1s):
        R = "P0a"
    elif any(f <= 0.90 for f in f1s):
        R = "P0b"
    else:
        R = "P0c"
    L += ["", "Part 0 readings (results/item46_readings.md): P0a training macro-F1 at least 0.98 in all four runs; P0b at most 0.90 in",
          "any of the four; P0c otherwise.", f"Reading: {R}", f"Written {time.strftime('%Y-%m-%d %H:%M')}."]
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    if os.path.exists(OUT):
        raise FileExistsError(OUT)
    open(OUT, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
