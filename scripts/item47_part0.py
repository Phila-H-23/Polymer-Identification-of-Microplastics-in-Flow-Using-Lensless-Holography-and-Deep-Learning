"""Item 47, Part 0 (copy of the item 46 Part 0 code, generalised; no training): training-set macro-F1 of best.pt in eval
mode on the unshifted training patches (depth-aug runs: <patch_dir>/2.250, byte copies of the z0 patches), no
augmentation, channel statistics from the training split as in the runs. The network is built with
train_compact_cnn_deep.py and the run's own --width and --extra-convs (0 = the original model).
usage: python scripts/item47_part0.py w64 | deep [--out PATH]"""
import os
import sys
import json
import time
import importlib.util

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)

R = "results"
SETS = {"w64": [("Wn (width 64, none)", "original", f"{R}/real_tc_PETvsPS_z0w64_exp_s{k}", 0.907, "z0 original") for k in (0, 1, 2)]
        + [("Wd (width 64, all21)", "original", f"{R}/real_tc_PETvsPS_z0depthD_w64_exp_s{k}", 0.899, "D original") for k in (0, 1, 2)],
        "deep": [("Ln (18 layers, none)", "original", f"{R}/real_tc_PETvsPS_z0deep18_exp_s{k}", 0.907, "z0 original") for k in (0, 1, 2)]
        + [("Ln (18 layers, none)", "reverse", f"{R}/real_tc_PETvsPS_z0deep18_exp_rev_s{k}", 0.918, "z0 reverse") for k in (0, 1, 2)]
        + [("Ld (18 layers, all21)", "original", f"{R}/real_tc_PETvsPS_z0deep18depthD_exp_s{k}", 0.899, "D original") for k in (0, 1, 2)]
        + [("Ld (18 layers, all21)", "reverse", f"{R}/real_tc_PETvsPS_z0deep18depthD_exp_rev_s{k}", 0.950, "D reverse") for k in (0, 1, 2)]}
OUTS = {"w64": f"{R}/item47_part0_trainfit.txt", "deep": f"{R}/item47_part0_deep_trainfit.txt"}


def n0(f, c):
    return "N0a" if f >= 0.98 else ("N0b" if f > c + 0.03 else "N0c")


def main(mode, out):
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score, recall_score
    spec = importlib.util.spec_from_file_location("tcdeep", "train_compact_cnn_deep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    L = [f"ITEM 47, PART 0 ({mode}): TRAINING-SET FIT (best.pt, eval mode, training split, unshifted, no augmentation)", "",
         f"{'run':42s} {'network':21s} {'direction':9s} {'train F1':>8s} {'recall PET':>10s} {'recall PS':>9s} {'val F1':>7s} {'test F1':>7s} {'n train':>7s}"]
    got = {}
    for net, direction, d, cref, cname in SETS[mode]:
        if not os.path.exists(f"{d}/best.pt"):
            L.append(f"{os.path.basename(d):42s} {net:21s} {direction:9s} not run or best.pt missing: skipped")
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
        model = m.CompactHoloCNN(1, len(classes), a["width"], a["dropout"], extra_convs=a.get("extra_convs", 0))
        model.load_state_dict(torch.load(f"{d}/best.pt", map_location="cpu"))
        model.eval()
        ds = m.PatchDataset(tr, pdir, c2i, channels=a["channels"], mean=mean, std=std)
        pred = []
        with torch.no_grad():
            for x, _ in DataLoader(ds, batch_size=64, shuffle=False):
                pred += model(x).argmax(1).tolist()
        y = [c2i[c] for c in tr["class"]]
        f1 = f1_score(y, pred, average="macro")
        rec = recall_score(y, pred, average=None, labels=[0, 1])
        got.setdefault((net, direction, cref, cname), {})[int(d.rsplit("_s", 1)[1])] = f1
        L.append(f"{os.path.basename(d):42s} {net:21s} {direction:9s} {f1:8.4f} {rec[0]:10.3f} {rec[1]:9.3f} "
                 f"{saved['best_val_macro_f1']:7.4f} {saved['test_report']['macro avg']['f1-score']:7.4f} {len(tr):7d}")
    L += ["", "N0 readings (results/item47_readings.md; seed 0 against its seed-0 width-32 counterpart; descriptive, the 0.03 is arbitrary):"]
    for (net, direction, cref, cname), v in got.items():
        mean_f = sum(v.values()) / len(v)
        r = n0(v[0], cref) if 0 in v else "not available (seed 0 not run)"
        L.append(f"  {net:21s} {direction:9s} seed 0 {v.get(0, float('nan')):.4f} against {cname} {cref:.3f}: {r}; mean over seeds "
                 f"{sorted(v)} {mean_f:.4f}")
    L.append(f"Written {time.strftime('%Y-%m-%d %H:%M')}.")
    text = "\n".join(L)
    assert "\u2014" not in text and ("signif" + "icant") not in text.lower() and ("valid" + "ated") not in text.lower()
    if os.path.exists(out):
        raise FileExistsError(out)
    open(out, "w", encoding="utf-8").write(text + "\n")
    print("\n".join(l for l in L if l.startswith("  ") or l.startswith("real_tc")))


if __name__ == "__main__":
    mode = sys.argv[1]
    main(mode, sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else OUTS[mode])
