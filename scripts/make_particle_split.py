"""Particle-identity split on the common 849 patches (new files only).
Source: the earlier particle-identity split (column 'split' of
pipeline_output/manifest.csv, identical to manifest_backup_pre_video_holdout.csv),
restricted to the 849 common patches. Group = particle (video + particle_id); every
patch is a distinct particle. Background-only patches take the split of the real
patch of the same particle (source_file, frame, particle_id).
Writes:
  pipeline_output/split_particle_identity_849.csv  (patch id, class, video, particle id, split)
  pipeline_output/*_psplit.csv  (copies of the arm manifests with 'split' = this split;
                                 the original column kept as 'split_before_psplit')
  pipeline_output/class_weights_psplit.json  (n_train / (n_classes * n_c), this split)"""
import os
import json
import pandas as pd

PO = r"C:\Users\haany\OneDrive\research project\dataset 10jan23\pipeline_output"
KEY = ["source_file", "frame", "particle_id"]
FAMILIES = [("manifest_common849.csv", "manifest_background_common849.csv"),
            ("manifest_bgcorrected_common849.csv", "manifest_background_bgcorrected_common849.csv"),
            ("manifest_recon_s0.csv", "manifest_background_recon_s0.csv"),
            ("manifest_recon_s0_z0.csv", "manifest_background_recon_s0_z0.csv"),
            ("manifest_recon_prop.csv", "manifest_background_recon_prop.csv")]

base = pd.read_csv(os.path.join(PO, "manifest.csv"))
c849 = pd.read_csv(os.path.join(PO, "manifest_common849.csv"))
split = base[base.patch_file.isin(c849.patch_file)][["patch_file", "class", "source_file", "particle_id", "frame", "split"]]
assert len(split) == 849
groups = split.groupby(["source_file", "particle_id"])["split"].nunique()
assert (groups == 1).all(), "a particle appears in more than one split"
split.rename(columns={"patch_file": "patch_id", "source_file": "video"}).drop(columns="frame").to_csv(
    os.path.join(PO, "split_particle_identity_849.csv"), index=False)

by_file = dict(zip(split.patch_file, split.split))
by_key = {tuple(k): s for k, s in zip(split[KEY].values, split.split)}
for rname, bname in FAMILIES:
    r = pd.read_csv(os.path.join(PO, rname))
    b = pd.read_csv(os.path.join(PO, bname))
    r["split_before_psplit"], b["split_before_psplit"] = r["split"], b["split"]
    r["split"] = r.patch_file.map(by_file)
    b["split"] = [by_key[tuple(k)] for k in b[KEY].values]
    assert r.split.notna().all() and b.split.notna().all() and len(r) == len(b) == 849
    pair = r.merge(b, on=KEY, suffixes=("_r", "_b"))
    assert len(pair) == 849 and (pair.split_r == pair.split_b).all()
    r.to_csv(os.path.join(PO, rname.replace(".csv", "_psplit.csv")), index=False)
    b.to_csv(os.path.join(PO, bname.replace(".csv", "_psplit.csv")), index=False)

n = split[split.split == "train"]["class"].value_counts()
w = {c: round(float(n.sum() / (len(n) * n[c])), 4) for c in sorted(n.index)}
json.dump(w, open(os.path.join(PO, "class_weights_psplit.json"), "w"), indent=2)

t = pd.crosstab(split["class"], split.split)
t["videos_train"] = split[split.split == "train"].groupby("class").source_file.nunique()
t["videos_test"] = split[split.split == "test"].groupby("class").source_file.nunique()
print(t.to_string())
print("ASSERT no particle in more than one split: passed (", groups.size, "particles )")
print("ASSERT background patch split == paired real patch split, 5 manifest families: passed")
low = t.index[t["test"] < 5].tolist()
print("classes with fewer than 5 test patches:", low or "none")
print("class weights (this split, both classifiers):", w)
