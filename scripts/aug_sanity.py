"""Sanity image for the surround augmentation -> results/aug_examples.png.
12 training patches (per domain z0 and intensity_bgcorr_vnorm: 3 particle + 3 background, mixed classes,
one clipped particle patch), each as original / BG-swapped / BG+TEX, common colour scale per row, using
the PatchDataset augmentation in train_compact_cnn.py. Also prints an edge-jump measure: |mean of the ring
r 10-12 px - mean of the ring r 12-14 px| around the disc centre, divided by the patch std."""
import os
import random
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
spec = importlib.util.spec_from_file_location("tcc", "train_compact_cnn.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PO = "pipeline_output"
C2I = {c: i for i, c in enumerate(["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"])}
DOMAINS = [("z0", "recon_s0", "manifest_recon_s0_z0.csv", "patches_recon_s0_z0",
            "manifest_background_recon_s0_z0.csv", "patches_background_recon_s0_z0"),
           ("intensity_bgcorr_vnorm", "intensity_bgcorr_vnorm", "manifest_bgcorrected_common849.csv", "patches_bgcorrected",
            "manifest_background_bgcorrected_common849.csv", "patches_background_bgcorrected")]


def pick(df, split_col, classes, clipped_first=False):
    tr = df[df[split_col] == "train"]
    out = []
    if clipped_first:
        c = tr[(tr.x < 32) | (tr.y < 32) | (tr.x + 32 > 1224) | (tr.y + 32 > 1024)]
        out.append(c.sample(1, random_state=1))
    for cl in classes[len(out):]:
        out.append(tr[tr["class"] == cl].sample(1, random_state=1))
    return pd.concat(out)


def show(x):
    return x.mean(axis=-1)  # z0: the single channel; intensity: mean of the 4 channels, as the model sees it


def edge_jump(img, cy, cx):
    yy, xx = np.mgrid[:64, :64]
    d = np.hypot(yy - cy, xx - cx)
    a, b = img[(d >= 10) & (d < 12)].mean(), img[(d >= 12) & (d < 14)].mean()
    return abs(a - b) / (img.std() + 1e-9)


rows = []
for dom, ch, rman, rdir, bman, bdir in DOMAINS:
    real, bg = pd.read_csv(os.path.join(PO, rman)), pd.read_csv(os.path.join(PO, bman))
    donors = bg[bg.split == "train"]
    for kind, df, split_col, pdir, classes, clip in (
            ("particle", real, "split_video_holdout", rdir, ["PE", "PET", "PS"] if dom == "z0" else ["PMMA", "PVC", "PP"], True),
            ("background", bg, "split", bdir, ["PET", "PVC", "PMMA"] if dom == "z0" else ["PE", "PS", "PVA"], False)):
        sel = pick(df, split_col, classes, clipped_first=clip)
        dss = {mode: m.PatchDataset(sel, os.path.join(PO, pdir), C2I, channels=ch, augment=True, aug_mode=mode,
                                    donor_df=donors, donor_dir=os.path.join(PO, bdir)) for mode in ("bg", "bgtex")}
        for i in range(len(sel)):
            r = dss["bg"].df.iloc[i]
            cy, cx = int(r.y) - max(0, int(r.y) - 32), int(r.x) - max(0, int(r.x) - 32)
            orig = dss["bg"]._load(i)[0]
            random.seed(100 + i); np.random.seed(100 + i)
            sw = dss["bg"]._surround_aug(orig.copy(), i)
            random.seed(100 + i); np.random.seed(100 + i)
            tx = dss["bgtex"]._surround_aug(orig.copy(), i)
            clipped = (r.x < 32) or (r.y < 32) or (r.x + 32 > 1224) or (r.y + 32 > 1024)
            rows.append(dict(label=f"{dom}\n{kind} {r['class']}\n{r.source_file[:-8]}{' clipped' if clipped else ''}",
                             imgs=[show(orig), show(sw), show(tx)], cy=cy, cx=cx))

fig, axes = plt.subplots(len(rows), 3, figsize=(7.2, 2.25 * len(rows)))
print(f"{'row':4s} {'patch':42s} edge jump / patch std: original, BG, BG+TEX")
for k, row in enumerate(rows):
    lo, hi = np.percentile(row["imgs"][0], [1, 99])
    jumps = [edge_jump(im, row["cy"], row["cx"]) for im in row["imgs"]]
    print(f"{k:<4d} {row['label'].replace(chr(10), ' | '):42s} {jumps[0]:.3f}, {jumps[1]:.3f}, {jumps[2]:.3f}")
    for j, (im, t) in enumerate(zip(row["imgs"], ["original", "BG-swapped", "BG+TEX"])):
        ax = axes[k, j]
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_title(t, fontsize=9)
        if j == 0:
            ax.set_ylabel(row["label"], fontsize=7, rotation=0, ha="right", va="center")
fig.suptitle("Surround augmentation on training patches (disc r 12 px at the particle position)", fontsize=9)
plt.tight_layout(rect=[0.06, 0, 1, 0.985])
fig.savefig("results/aug_examples.png", dpi=100)
print("Saved -> results/aug_examples.png")
