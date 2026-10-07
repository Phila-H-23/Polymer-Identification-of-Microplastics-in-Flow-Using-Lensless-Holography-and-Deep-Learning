"""Part 1 assembly for the expanded PET and PS set (new files only, suffix _exp).
Input: results/expanded_PETPS_exp/part1_detections.csv from scripts/extract_PETPS_exp.py --part1.
Writes:
  pipeline_output/manifest_PETPS_exp.csv             real patches; split_video_holdout = original direction,
                                                      split_rev = reverse fold; old_patch, split_old, split_rev_old
  pipeline_output/manifest_background_PETPS_exp.csv  paired background-only rows; split / split_rev copied from the
                                                      paired real patch (same source_file, frame, particle_id)
  pipeline_output/class_weights_twoclass_PETvsPS_exp.json, class_weights_twoclass_PETvsPS_rev_exp.json
                                                      n_train / (2 x n_c) from the new training splits
  results/expanded_PETPS_exp/part1_duplicates.csv     20 px duplicate check (report only, nothing dropped)
Rules:
  - frame 0 patches are left out (no preceding frames, so no background estimate; same as the common 849 set).
  - background-only rows: the 883 stored rows are copied for the old patches (positions unchanged); the new patches
    get positions from the same sampler (scripts/build_background_dataset.py: sample_background_center, Chebyshev
    distance >= 96 px from every real detection of that frame, box in bounds, numpy default_rng(42), rows in
    (source_file, frame, particle_id) order), run over the new rows only so the stored positions do not move.
  - splits (video roles unchanged): original direction PET train PET1, PET3, PET4 / test PET2, PS train PS2 / test
    PS1; reverse fold (scripts/make_reverse_fold.py, 2026-10-04) PET test PET1, train PET2, PET3, PET4 with PET3 and
    PET4 keeping their original-direction train / val labels, PS test PS2, train PS1.
    Validation: the stored rule draws 15/85 (about 17.6%) of each class's training-video patches at random, per
    patch (original direction), and 17.5% of the newly added training video (reverse fold, random_state 0). Here
    the same fractions are drawn as whole sampled frames (all patches of one frame in one split): frames shuffled
    with numpy default_rng(0), and the prefix whose patch count is closest to the target goes to validation.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
PO = os.path.join(D, "pipeline_output")
OUT = os.path.join(D, "results", "expanded_PETPS_exp")
sys.path.insert(0, os.path.join(D, "scripts"))
import build_background_dataset as bbd  # noqa: E402

KEY = ["source_file", "frame", "particle_id"]
VAL_ORIG, VAL_REV = 15 / 85, 0.175


def frame_val(df, frac, seed=0):
    """Whole frames to validation, count closest to frac * len(df)."""
    fr = df.groupby(["source_file", "frame"]).size().sort_index()
    order = np.random.default_rng(seed).permutation(len(fr))
    cum = np.cumsum(fr.values[order])
    target = frac * len(df)
    j = int(np.argmin(np.abs(cum - target)))
    chosen = set(fr.index[order[: j + 1]])
    return [(s, f) in chosen for s, f in zip(df.source_file, df.frame)]


def main():
    det = pd.read_csv(os.path.join(OUT, "part1_detections.csv"))
    det = det[det.frame != 0].reset_index(drop=True)
    c849 = pd.read_csv(os.path.join(PO, "manifest_bgcorrected_common849.csv"))
    c849r = pd.read_csv(os.path.join(PO, "manifest_bgcorrected_common849_rev.csv"))
    old849 = c849[c849["class"].isin(["PET", "PS"])]
    assert set(det[det.old_patch].patch_file) == set(old849.patch_file), "old patches differ from the common 849 set"
    det["split_old"] = det.patch_file.map(dict(zip(c849.patch_file, c849.split_video_holdout)))
    det["split_rev_old"] = det.patch_file.map(dict(zip(c849r.patch_file, c849r.split_video_holdout)))

    v = det.source_file.str.replace("_600.mat", "", regex=False)
    # original direction
    s = pd.Series("train", index=det.index)
    s[v.isin(["PET2", "PS1"])] = "test"
    for vids in (["PET1", "PET3", "PET4"], ["PS2"]):
        pool = det[v.isin(vids)]
        s[pool.index[frame_val(pool, VAL_ORIG)]] = "val"
    det["split_video_holdout"] = s
    # reverse fold
    r = pd.Series("train", index=det.index)
    r[v.isin(["PET1", "PS2"])] = "test"
    keep = v.isin(["PET3", "PET4"])
    r[keep] = s[keep]
    for vid in ("PET2", "PS1"):
        pool = det[v == vid]
        r[pool.index[frame_val(pool, VAL_REV)]] = "val"
    det["split_rev"] = r
    for col in ("split_video_holdout", "split_rev"):  # no frame in two splits
        assert (det.groupby(["source_file", "frame"])[col].nunique() == 1).all()

    # paired background-only rows
    bg_old = pd.read_csv(os.path.join(PO, "manifest_background.csv"))
    old_rows = det[det.old_patch][KEY].merge(bg_old, on=KEY, how="left")
    assert old_rows.patch_file.notna().all() and len(old_rows) == det.old_patch.sum()
    rng = np.random.default_rng(bbd.SEED)
    new_rows, n_fallback = [], 0
    nd = det[~det.old_patch].sort_values(KEY)
    for (sf, fr), fd in nd.groupby(["source_file", "frame"], sort=False):
        ex = list(zip(det[(det.source_file == sf) & (det.frame == fr)].x.astype(int),
                      det[(det.source_file == sf) & (det.frame == fr)].y.astype(int)))
        for _, row in fd.iterrows():
            cx, cy, ok = bbd.sample_background_center(rng, ex, 1224, 1024)
            n_fallback += int(not ok)
            new_rows.append(dict(patch_file=f"{sf.replace('.mat', '')}_bgexpparticle{int(row.particle_id) - 1000}"
                                            f"_frame{int(fr)}.npy",
                                 particle_id=int(row.particle_id), frame=int(fr), x=int(cx), y=int(cy),
                                 source_file=sf, **{"class": row["class"]}))
    bg = pd.concat([old_rows[["patch_file", "particle_id", "frame", "x", "y", "source_file", "class"]],
                    pd.DataFrame(new_rows)], ignore_index=True)
    k = det.set_index(KEY)
    bg["split"] = [k.loc[tuple(t), "split_video_holdout"] for t in bg[KEY].values]
    bg["split_rev"] = [k.loc[tuple(t), "split_rev"] for t in bg[KEY].values]
    bg["old_patch"] = [bool(k.loc[tuple(t), "old_patch"]) for t in bg[KEY].values]
    assert len(bg) == len(det) and bg.patch_file.is_unique and det.patch_file.is_unique

    cols = ["patch_file", "particle_id", "frame", "x", "y", "source_file", "class", "split_video_holdout", "split_rev",
            "old_patch", "split_old", "split_rev_old", "bbox_w_px", "bbox_h_px", "region_px_dilated",
            "region_px_predilation", "region_px_in_patch", "clipped"]
    det[cols].to_csv(os.path.join(PO, "manifest_PETPS_exp.csv"), index=False)
    bg.to_csv(os.path.join(PO, "manifest_background_PETPS_exp.csv"), index=False)

    for col, name in (("split_video_holdout", "class_weights_twoclass_PETvsPS_exp.json"),
                      ("split_rev", "class_weights_twoclass_PETvsPS_rev_exp.json")):
        n = det[det[col] == "train"]["class"].value_counts()
        w = {c: round(float(n.sum() / (2 * n[c])), 4) for c in sorted(n.index)}
        json.dump(w, open(os.path.join(PO, name), "w"), indent=2)
        print(name, w, dict(n))

    # duplicate check (report only): centroid within 20 px of a centroid in another sampled frame of the same video
    rows = []
    for sf, g in det.groupby("source_file"):
        xy, fr = g[["x", "y"]].values.astype(float), g.frame.values
        for i, (pf, old) in enumerate(zip(g.patch_file, g.old_patch)):
            d = np.hypot(*(xy - xy[i]).T)
            other = fr != fr[i]
            rows.append(dict(patch_file=pf, source_file=sf, old_patch=old,
                             min_dist_other_frame_px=float(d[other].min()) if other.any() else np.nan))
    dup = pd.DataFrame(rows)
    dup["within_20px"] = dup.min_dist_other_frame_px < 20
    dup.to_csv(os.path.join(OUT, "part1_duplicates.csv"), index=False)
    print(f"background fallback positions: {n_fallback} of {len(new_rows)} new rows")
    print(pd.crosstab([det["class"], v], [det.split_video_holdout]).to_string())
    print(pd.crosstab([det["class"], v], [det.split_rev]).to_string())


if __name__ == "__main__":
    main()
