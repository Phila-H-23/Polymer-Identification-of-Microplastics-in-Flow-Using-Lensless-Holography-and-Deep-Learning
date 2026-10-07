"""Addendum (a) and (b) for the expanded PET and PS set. Updates only the split and index files (new files of this
task): pipeline_output/manifest_PETPS_exp.csv, manifest_background_PETPS_exp.csv and the two _exp class-weight files.
(a) track_id: within each video, detections whose centroids lie < 20 px apart in two different sampled frames are
    linked; connected components are tracks. track_id_oldset: the same linking among the old (849-set) patches only.
    A particle-free row inherits the track of its paired real row (same source_file, frame, particle_id).
    collision_kept: old patch kept although the unchanged 15 px Step 7 order would have dropped it (from
    results/expanded_PETPS_exp/part1_collisions.csv, written by scripts/extract_PETPS_exp.py --collisions; left
    empty until that file exists).
(b) validation by track. Original direction, per class: the class's training-video patches (PET1, PET3, PET4 / PS2)
    are the pool; whole tracks, in numpy default_rng(0) order, go to validation, and the prefix whose patch count is
    closest to 15/85 of the pool is used (the fraction of the stored split). Reverse fold (stored 2026-10-04 rule):
    PET3 and PET4 keep their original-direction labels, validation is drawn from PET2 tracks (target 0.175 of PET2);
    PS validation from PS1 tracks (target 0.175 of PS1). Fallback per class and direction: if the largest track in
    the pool exceeds 15% of that class's training patches (train + val of the class in that direction), the class
    keeps the frame grouping. Test splits never change. The frame-grouped splits are kept as *_framegroup columns
    and class_weights_twoclass_PETvsPS[_rev]_framegroup_exp.json, and are the reference for the 'moved' counts.
"""
import os
import json
import numpy as np
import pandas as pd

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
PO = os.path.join(D, "pipeline_output")
OUT = os.path.join(D, "results", "expanded_PETPS_exp")
KEY = ["source_file", "frame", "particle_id"]
LINK_PX = 20


def tracks(df):
    """Connected components of the < 20 px cross-frame links, per video. Returns a Series of track ids."""
    out = pd.Series(index=df.index, dtype=object)
    for v, g in df.groupby("source_file"):
        xy, fr, idx = g[["x", "y"]].values.astype(float), g.frame.values, g.index.values
        parent = list(range(len(g)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(g)):
            d = np.hypot(*(xy[i + 1:] - xy[i]).T)
            for j in np.flatnonzero((d < LINK_PX) & (fr[i + 1:] != fr[i])) + i + 1:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
        roots = [find(i) for i in range(len(g))]
        order = {r: k for k, r in enumerate(sorted(set(roots), key=lambda r: (fr[r], r)))}
        tag = v.replace("_600.mat", "")
        out[idx] = [f"{tag}_t{order[r]}" for r in roots]
    return out


def track_val(pool, frac, seed=0):
    t = pool.groupby("track_id").size().sort_index()
    order = np.random.default_rng(seed).permutation(len(t))
    cum = np.cumsum(t.values[order])
    j = int(np.argmin(np.abs(cum - frac * len(pool))))
    chosen = set(t.index[order[: j + 1]])
    return pool.track_id.isin(chosen)


def main():
    m = pd.read_csv(os.path.join(PO, "manifest_PETPS_exp.csv"))
    b = pd.read_csv(os.path.join(PO, "manifest_background_PETPS_exp.csv"))
    if "split_video_holdout_framegroup" not in m.columns:
        m["split_video_holdout_framegroup"], m["split_rev_framegroup"] = m.split_video_holdout, m.split_rev
        for sfx in ("", "_rev"):
            src = os.path.join(PO, f"class_weights_twoclass_PETvsPS{sfx}_exp.json")
            json.dump(json.load(open(src)), open(src.replace("_exp.json", "_framegroup_exp.json"), "w"), indent=2)
    m["track_id"] = tracks(m)
    m["track_id_oldset"] = None
    old = m[m.old_patch.astype(bool)]
    m.loc[old.index, "track_id_oldset"] = tracks(old)
    col = os.path.join(OUT, "part1_collisions.csv")
    if os.path.exists(col):
        c = pd.read_csv(col)
        m["collision_kept"] = m.patch_file.map(dict(zip(c.patch_file, c.collision_kept))).where(m.old_patch.astype(bool), False)
    elif "collision_kept" not in m.columns:
        m["collision_kept"] = None

    v = m.source_file.str.replace("_600.mat", "", regex=False)
    fg, fgr = m.split_video_holdout_framegroup, m.split_rev_framegroup
    s, r = fg.copy(), fgr.copy()
    decisions = []

    def assign(target_split, mask_pool, frac, cls, direction, n_class_train):
        pool = m[mask_pool]
        largest = int(pool.groupby("track_id").size().max())
        fallback = largest > 0.15 * n_class_train
        d = dict(cls=cls, direction=direction, pool_videos=sorted(v[mask_pool].unique()), pool_patches=len(pool),
                 class_training_patches=int(n_class_train), n_tracks=int(pool.track_id.nunique()),
                 largest_track=largest, threshold_15pct=round(0.15 * n_class_train, 1),
                 target_val=round(frac * len(pool), 1), fallback_framegroup=bool(fallback))
        if not fallback:
            is_val = track_val(pool, frac)
            target_split[pool.index] = np.where(is_val, "val", "train")
            d["val_patches"] = int(is_val.sum())
        decisions.append(d)
        return fallback

    for cls, vids in (("PET", ["PET1", "PET3", "PET4"]), ("PS", ["PS2"])):
        pool = v.isin(vids)
        assign(s, pool, 15 / 85, cls, "original", pool.sum())
    r[v.isin(["PET3", "PET4"])] = s[v.isin(["PET3", "PET4"])]
    for cls, vid, train_vids in (("PET", "PET2", ["PET2", "PET3", "PET4"]), ("PS", "PS1", ["PS1"])):
        assign(r, v == vid, 0.175, cls, "reverse", v.isin(train_vids).sum())
    assert (s[fg == "test"] == "test").all() and (s[fg != "test"] != "test").all()
    assert (r[fgr == "test"] == "test").all() and (r[fgr != "test"] != "test").all()
    m["split_video_holdout"], m["split_rev"] = s, r

    k = m.set_index(KEY)
    for c in ("track_id", "split_video_holdout_framegroup", "split_rev_framegroup", "collision_kept"):
        b[c] = [k.loc[tuple(t), c] for t in b[KEY].values]
    b["split"] = [k.loc[tuple(t), "split_video_holdout"] for t in b[KEY].values]
    b["split_rev"] = [k.loc[tuple(t), "split_rev"] for t in b[KEY].values]
    m.to_csv(os.path.join(PO, "manifest_PETPS_exp.csv"), index=False)
    b.to_csv(os.path.join(PO, "manifest_background_PETPS_exp.csv"), index=False)

    weights = {}
    for col_, name in (("split_video_holdout", "class_weights_twoclass_PETvsPS_exp.json"),
                       ("split_rev", "class_weights_twoclass_PETvsPS_rev_exp.json")):
        n = m[m[col_] == "train"]["class"].value_counts()
        w = {c: round(float(n.sum() / (2 * n[c])), 4) for c in sorted(n.index)}
        json.dump(w, open(os.path.join(PO, name), "w"), indent=2)
        weights[name] = dict(weights=w, n_train={c: int(n[c]) for c in sorted(n.index)})

    moved = {}
    for dname, new, ref in (("original", "split_video_holdout", "split_video_holdout_framegroup"),
                            ("reverse", "split_rev", "split_rev_framegroup")):
        x = m[m[new] != m[ref]]
        moved[dname] = {f"{c} {a} to {bb}": int(n) for (c, a, bb), n in x.groupby(["class", ref, new]).size().items()}
        moved[dname + "_frames_spanning_train_and_val"] = int(
            (m[m[new] != "test"].groupby(["source_file", "frame"])[new].nunique() > 1).sum())
    json.dump(dict(decisions=decisions, moved=moved, weights=weights,
                   collisions_available=bool(os.path.exists(col))),
              open(os.path.join(OUT, "split_by_track.json"), "w"), indent=2)
    print(json.dumps(dict(decisions=decisions, moved=moved, weights=weights), indent=1))


if __name__ == "__main__":
    main()
