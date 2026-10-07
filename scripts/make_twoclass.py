"""Two-class test (PET vs one other polymer), Step 0. New files only:
  pipeline_output/class_weights_twoclass_<pair>.json        n_train / (2 * n_c), the pair's video-holdout training split
  pipeline_output/class_weights_twoclass_PETvsPS_rev.json    same construction on the reverse-fold training split
  pipeline_output/manifest_recon_s0_z0_rev.csv, manifest_background_recon_s0_z0_rev.csv
      the reverse-fold split of 2026-10-04 (copied patch by patch from manifest_bgcorrected_common849_rev.csv),
      applied to the z0 manifests, which did not exist in reverse-fold form
Prints the Step 0 tables."""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PO = r"C:\Users\haany\OneDrive\research project\dataset 10jan23\pipeline_output"
KEY = ["source_file", "frame", "particle_id"]
ARMS = {"intensity_bgcorr_vnorm": ("manifest_bgcorrected_common849.csv", "manifest_background_bgcorrected_common849.csv"),
        "z0": ("manifest_recon_s0_z0.csv", "manifest_background_recon_s0_z0.csv")}


def balanced(train_classes):
    n = train_classes.value_counts()
    return {c: round(float(n.sum() / (len(n) * n[c])), 4) for c in sorted(n.index)}


def chance(train_classes, test_classes, seed=0, draws=1000):
    labels = sorted(train_classes.unique())
    p = train_classes.value_counts(normalize=True).reindex(labels).values
    rng = np.random.default_rng(seed)
    s = [f1_score(test_classes, rng.choice(labels, size=len(test_classes), p=p), labels=labels,
                  average="macro", zero_division=0) for _ in range(draws)]
    return float(np.mean(s)), float(np.std(s))


def main():
    real = {a: pd.read_csv(os.path.join(PO, m[0])) for a, m in ARMS.items()}
    bg = {a: pd.read_csv(os.path.join(PO, m[1])) for a, m in ARMS.items()}
    r = real["intensity_bgcorr_vnorm"]
    for a in ARMS:  # both arms carry the same patches and split
        assert set(real[a].patch_file) == set(r.patch_file)
        assert (real[a].set_index("patch_file").split_video_holdout.sort_index()
                == r.set_index("patch_file").split_video_holdout.sort_index()).all()
        assert (bg[a].sort_values("patch_file").split.values == bg["intensity_bgcorr_vnorm"].sort_values("patch_file").split.values).all()

    # seven-class weights: same construction n_train / (7 * n_c), from manifest.csv (883) video-holdout train counts
    m883 = pd.read_csv(os.path.join(PO, "manifest.csv"))
    w7 = balanced(m883[m883.split_video_holdout == "train"]["class"])
    w7_file = json.load(open(os.path.join(PO, "class_weights_video_holdout.json")))
    print(f"seven-class weights file == n_train/(7*n_c) on the 883-patch video-holdout training split: "
          f"{all(abs(w7[c] - w7_file[c]) < 1e-4 for c in w7)}")

    totals = r["class"].value_counts()
    others = sorted([c for c in totals.index if c != "PET"], key=lambda c: -(totals["PET"] + totals[c]))
    order = ["PS"] + [c for c in others if c != "PS"]
    print("pair order (PET vs PS first, then descending total patches in the common-849 manifest):")
    print("  " + ", ".join(f"PETvs{c} ({totals['PET'] + totals[c]})" for c in order))

    for c in order:
        pair, keep = f"PETvs{c}", ["PET", c]
        rp = r[r["class"].isin(keep)]
        bp = bg["intensity_bgcorr_vnorm"][bg["intensity_bgcorr_vnorm"]["class"].isin(keep)]
        w = balanced(rp[rp.split_video_holdout == "train"]["class"])
        json.dump(w, open(os.path.join(PO, f"class_weights_twoclass_{pair}.json"), "w"), indent=2)
        ch = chance(rp[rp.split_video_holdout == "train"]["class"], rp[rp.split_video_holdout == "test"]["class"].values)
        vids = {s: sorted(rp[rp.split_video_holdout == s].source_file.str.replace("_600.mat", "").unique())
                for s in ("train", "val", "test")}
        cnt = pd.crosstab(rp["class"], rp.split_video_holdout)
        cntb = pd.crosstab(bp["class"], bp.split)
        print(f"\n{pair}: train {vids['train']} | val {vids['val']} | test {vids['test']} | PET test video PET2")
        print("  particle patches   " + "; ".join(f"{k}: train {cnt.loc[k,'train']}, val {cnt.loc[k,'val']}, test {cnt.loc[k,'test']}" for k in keep))
        print("  background patches " + "; ".join(f"{k}: train {cntb.loc[k,'train']}, val {cntb.loc[k,'val']}, test {cntb.loc[k,'test']}" for k in keep))
        print(f"  class weights {w}  -> class_weights_twoclass_{pair}.json")
        print(f"  chance macro-F1 {ch[0]:.4f} +/- {ch[1]:.4f}")

    # Part 2: reverse-fold split of 2026-10-04, restricted to PET and PS
    rev = pd.read_csv(os.path.join(PO, "manifest_bgcorrected_common849_rev.csv"))
    rev_bg = pd.read_csv(os.path.join(PO, "manifest_background_bgcorrected_common849_rev.csv"))
    rs0 = pd.read_csv(os.path.join(PO, "manifest_recon_s0_rev.csv"))
    assert (rev.set_index("patch_file").split_video_holdout.sort_index()
            == rs0.set_index("patch_file").split_video_holdout.sort_index()).all()
    by_file = dict(zip(rev.patch_file, rev.split_video_holdout))
    by_key = {tuple(k): s for k, s in zip(rev_bg[KEY].values, rev_bg.split)}
    z0r = pd.read_csv(os.path.join(PO, "manifest_recon_s0_z0.csv"))
    z0b = pd.read_csv(os.path.join(PO, "manifest_background_recon_s0_z0.csv"))
    z0r["split_orig"], z0b["split_orig"] = z0r.split_video_holdout, z0b.split
    z0r["split_video_holdout"] = z0r.patch_file.map(by_file)
    z0b["split"] = [by_key[tuple(k)] for k in z0b[KEY].values]
    assert z0r.split_video_holdout.notna().all() and z0b.split.notna().all()
    z0r.to_csv(os.path.join(PO, "manifest_recon_s0_z0_rev.csv"), index=False)
    z0b.to_csv(os.path.join(PO, "manifest_background_recon_s0_z0_rev.csv"), index=False)
    rp = rev[rev["class"].isin(["PET", "PS"])]
    w = balanced(rp[rp.split_video_holdout == "train"]["class"])
    json.dump(w, open(os.path.join(PO, "class_weights_twoclass_PETvsPS_rev.json"), "w"), indent=2)
    ch = chance(rp[rp.split_video_holdout == "train"]["class"], rp[rp.split_video_holdout == "test"]["class"].values)
    vids = {s: sorted(rp[rp.split_video_holdout == s].source_file.str.replace("_600.mat", "").unique()) for s in ("train", "val", "test")}
    cnt = pd.crosstab(rp["class"], rp.split_video_holdout)
    print(f"\nPart 2 reverse fold (2026-10-04 split, found in manifest_*_rev.csv; z0 copy written), PETvsPS: "
          f"train {vids['train']} | val {vids['val']} | test {vids['test']}")
    print("  particle patches " + "; ".join(f"{k}: train {cnt.loc[k,'train']}, val {cnt.loc[k,'val']}, test {cnt.loc[k,'test']}" for k in ["PET", "PS"]))
    print(f"  class weights {w} -> class_weights_twoclass_PETvsPS_rev.json; chance macro-F1 {ch[0]:.4f} +/- {ch[1]:.4f}")


if __name__ == "__main__":
    main()
