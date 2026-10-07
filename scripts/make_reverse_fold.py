"""Reverse-fold split on the common 849 patches (new files only).
PE, PMMA, PP, PS, PVA, PVC: train on the current TEST video, test on ALL patches
(train + val) of the current training video. PET: test on all PET1 patches, train
on PET2, PET3, PET4. Val is drawn as in the original split: about 17.5% of each
class's patches from the newly added training video (one patch per particle, so no
particle spans train and val), random_state=0; PET3/PET4 keep their existing
train/val labels and val is drawn from PET2 only. Background-only patches take the
split of the real patch of the same particle (source_file, frame, particle_id).
Balanced class weights n_train / (n_classes * n_c) from the reverse-fold training
counts -> pipeline_output/class_weights_rev.json, used for BOTH arms."""
import os
import json
import pandas as pd

PO = r"C:\Users\haany\OneDrive\research project\dataset 10jan23\pipeline_output"
KEY = ["source_file", "frame", "particle_id"]
VAL_FRAC = 0.175

real = pd.read_csv(os.path.join(PO, "manifest_bgcorrected_common849.csv"))
orig = real["split_video_holdout"]
new = pd.Series(index=real.index, dtype=object)
new[orig.isin(["train", "val"]) & (real["class"] != "PET")] = "test"
new[(real["class"] == "PET") & (real.source_file == "PET1_600.mat")] = "test"
keep = (real["class"] == "PET") & real.source_file.isin(["PET3_600.mat", "PET4_600.mat"])
new[keep] = orig[keep]
promote = orig.eq("test")  # current test videos (incl. PET2) become training videos
for c, g in real[promote].groupby("class"):
    n_val = max(1, int(round(VAL_FRAC * len(g))))
    val_idx = g.sample(n=n_val, random_state=0).index
    new[g.index] = "train"
    new[val_idx] = "val"
assert new.notna().all()
mapping = dict(zip(real.patch_file, new))
keymap = {tuple(k): s for k, s in zip(real[KEY].values, new)}

pairs = [("manifest_bgcorrected_common849.csv", "manifest_background_bgcorrected_common849.csv"),
         ("manifest_recon_s0.csv", "manifest_background_recon_s0.csv")]
for rname, bname in pairs:
    r = pd.read_csv(os.path.join(PO, rname))
    r["split_orig"] = r["split_video_holdout"]
    r["split_video_holdout"] = r.patch_file.map(mapping)
    b = pd.read_csv(os.path.join(PO, bname))
    b["split_orig"] = b["split"]
    b["split"] = [keymap[tuple(k)] for k in b[KEY].values]
    assert r.split_video_holdout.notna().all() and b.split.notna().all()
    r.to_csv(os.path.join(PO, rname.replace(".csv", "_rev.csv")), index=False)
    b.to_csv(os.path.join(PO, bname.replace(".csv", "_rev.csv")), index=False)

n = real.assign(s=new)[lambda d: d.s == "train"]["class"].value_counts()
w = {c: round(n.sum() / (len(n) * n[c]), 4) for c in sorted(n.index)}
json.dump(w, open(os.path.join(PO, "class_weights_rev.json"), "w"), indent=2)

t = real.assign(s=new, video=real.source_file.str.replace("_600.mat", ""))
print(pd.crosstab([t["class"], t.video], t.s).to_string())
print("class weights (reverse fold, both arms):", w)
