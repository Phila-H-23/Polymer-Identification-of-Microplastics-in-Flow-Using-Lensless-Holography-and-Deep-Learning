"""PET session-count test on intensity_bgcorr_vnorm (common 849).
Test set and every non-PET class unchanged. PET train AND val (val drives
checkpoint selection) restricted per config:
  v1 / v3 / v4   : PET from that single training video only
  m1 / m3 / m4   : patch-matched controls - random subsample of PET pooled over all
                   3 training videos (stratified by video, random_state=0) with the
                   same PET train and val counts as v1 / v3 / v4
  all3           : all 3 training videos
  p13 / p14 / p34: 2-video pairs (optional)
Background-only manifests keep the background patches paired with the same
particles (source_file, frame, particle_id). Balanced class weights
n_train / (n_classes * n_c) from each config's own real training counts on the
video-holdout split, used for BOTH arms."""
import os
import json
import numpy as np
import pandas as pd

PO = r"C:\Users\haany\OneDrive\research project\dataset 10jan23\pipeline_output"
KEY = ["source_file", "frame", "particle_id"]
V = {"1": "PET1_600.mat", "3": "PET3_600.mat", "4": "PET4_600.mat"}

real = pd.read_csv(os.path.join(PO, "manifest_bgcorrected_common849.csv"))
bg = pd.read_csv(os.path.join(PO, "manifest_background_bgcorrected_common849.csv"))
pet = (real["class"] == "PET") & real["split_video_holdout"].isin(["train", "val"])
pet_rows = real[pet]


def stratified(pool, n, seed=0):
    """n rows from pool, allocated to videos in proportion to their counts (largest remainder)."""
    counts = pool["source_file"].value_counts()
    exact = counts / counts.sum() * n
    alloc = np.floor(exact).astype(int)
    for v in (exact - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[v] += 1
    return pd.concat([pool[pool.source_file == v].sample(n=int(k), random_state=seed) for v, k in alloc.items() if k])


def keep_for(cfg):
    if cfg.startswith("v"):
        return pet_rows[pet_rows.source_file == V[cfg[1]]]
    if cfg.startswith("p"):
        return pet_rows[pet_rows.source_file.isin([V[cfg[1]], V[cfg[2]]])]
    if cfg == "all3":
        return pet_rows
    ref = pet_rows[pet_rows.source_file == V[cfg[1]]]  # matched control m1/m3/m4
    parts = []
    for split in ("train", "val"):
        n = int((ref.split_video_holdout == split).sum())
        parts.append(stratified(pet_rows[pet_rows.split_video_holdout == split], n))
    return pd.concat(parts)


def main():
    for cfg in ["v1", "v3", "v4", "m1", "m3", "m4", "all3", "p13", "p14", "p34"]:
        keep = keep_for(cfg)
        drop = pet_rows[~pet_rows.patch_file.isin(keep.patch_file)]
        r = real[~real.patch_file.isin(drop.patch_file)]
        drop_keys = set(map(tuple, drop[KEY].values))
        b = bg[~bg[KEY].apply(tuple, axis=1).isin(drop_keys)]
        r.to_csv(os.path.join(PO, f"manifest_bgcorrected_common849_pet{cfg}.csv"), index=False)
        b.to_csv(os.path.join(PO, f"manifest_background_bgcorrected_common849_pet{cfg}.csv"), index=False)
        n = r[r.split_video_holdout == "train"]["class"].value_counts()
        w = {c: round(n.sum() / (len(n) * n[c]), 4) for c in sorted(n.index)}
        json.dump(w, open(os.path.join(PO, f"class_weights_pet{cfg}.json"), "w"), indent=2)
        kt, kv = keep[keep.split_video_holdout == "train"], keep[keep.split_video_holdout == "val"]
        bpet = b[(b["class"] == "PET")]
        print(f"{cfg:5s} PET train {len(kt):3d} {kt.source_file.str[:4].value_counts().to_dict()} | "
              f"val {len(kv):2d} {kv.source_file.str[:4].value_counts().to_dict()} | "
              f"bg PET train/val {int((bpet.split=='train').sum())}/{int((bpet.split=='val').sum())} | "
              f"PET test {int(((r['class']=='PET') & (r.split_video_holdout=='test')).sum())} | PET weight {w['PET']}")


if __name__ == "__main__":
    main()
