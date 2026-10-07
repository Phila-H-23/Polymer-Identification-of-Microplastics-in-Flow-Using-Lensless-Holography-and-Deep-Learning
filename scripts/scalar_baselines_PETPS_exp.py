"""Part 3: scalar baselines on the expanded PET and PS set, no CNN (new files only).
3a size only: log(region area before dilation), bbox width, bbox height (px), as in build_report_size_and_arms.py.
3b brightness only, from the z0 patch (I / median background, float32, as saved by the z0 build, before any
   normalisation): whole-patch mean, std, min, max (over the pixels of the saved crop, no padding), and the mean and
   the sum of (z0 - 1) over the detected region before dilation (the region's pixels inside the patch crop, from
   the detection code; mask saved by scripts/extract_PETPS_exp.py). Same features on the PAIRED particle-free patch
   (same source_file, frame, particle_id), with the region replaced by a central disc of the same pixel count as the
   paired real patch's region inside its crop (the N pixels closest to the patch centre (32, 32)).
3c size + brightness together, real patches only.
All: class-weighted logistic regression (L2, C=1.0, lbfgs, max_iter 1000), class weights n_train / (2 x n_c) from
the training rows used, features standardised with training-split mean / std, fitted on the training split only,
test macro-F1 with a 95% bootstrap interval (1000 resamples of test patches, numpy default_rng(0), percentile), and
beside it a cluster bootstrap interval (1000 resamples of whole test tracks, track_id from scripts/tracks_PETPS_exp.py,
numpy default_rng(0), percentile).
Both directions; all patches and unclipped patches (particle-free rows follow their paired real patch's clipped flag).
Writes results/expanded_PETPS_exp/part3_features.csv and part3_results.json.
Runs single-threaded (OMP / MKL / OPENBLAS threads = 1).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, recall_score  # noqa: E402

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
PO = os.path.join(D, "pipeline_output")
OUT = os.path.join(D, "results", "expanded_PETPS_exp")
KEY = ["source_file", "frame", "particle_id"]
CLS = ["PET", "PS"]
SIZE = ["log_area", "bbox_w_px", "bbox_h_px"]
BRIGHT = ["z0_mean", "z0_std", "z0_min", "z0_max", "region_mean_z0m1", "region_sum_z0m1"]


def disc(n, shape=(64, 64), c=(32, 32)):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    d = np.hypot(yy - c[0], xx - c[1]).ravel()
    m = np.zeros(d.size, bool)
    m[np.argsort(d, kind="stable")[:n]] = True
    return m.reshape(shape)


def bright(z, region):
    v = z - 1.0
    return dict(z0_mean=float(z.mean()), z0_std=float(z.std()), z0_min=float(z.min()), z0_max=float(z.max()),
                region_mean_z0m1=float(v[region].mean()) if region.any() else 0.0,
                region_sum_z0m1=float(v[region].sum()))


def features():
    m = pd.read_csv(os.path.join(PO, "manifest_PETPS_exp.csv"))
    b = pd.read_csv(os.path.join(PO, "manifest_background_PETPS_exp.csv"))
    masks = np.load(os.path.join(OUT, "part1_region_masks.npz"))
    rows = []
    bmap = b.set_index(KEY)
    for _, r in m.iterrows():
        z = np.load(os.path.join(PO, "patches_recon_s0_z0_PETPS_exp", r.patch_file))[..., 0].astype(np.float64)
        reg = masks[r.patch_file.replace(".npy", "")]
        assert reg.shape == z.shape and int(reg.sum()) == r.region_px_in_patch
        rec = dict(r[["patch_file", "source_file", "frame", "particle_id", "class", "split_video_holdout", "split_rev",
                      "clipped", "old_patch", "region_px_in_patch", "track_id"]])
        rec.update(kind="real", log_area=float(np.log(r.region_px_predilation)), bbox_w_px=r.bbox_w_px,
                   bbox_h_px=r.bbox_h_px, **bright(z, reg))
        rows.append(rec)
        br = bmap.loc[(r.source_file, r.frame, r.particle_id)]
        zb = np.load(os.path.join(PO, "patches_background_recon_s0_z0_PETPS_exp", br.patch_file))[..., 0].astype(np.float64)
        rec_b = dict(rec, kind="particle-free", patch_file=br.patch_file, **bright(zb, disc(int(r.region_px_in_patch), zb.shape)))
        assert br.split == r.split_video_holdout and br.split_rev == r.split_rev
        rows.append(rec_b)
    return pd.DataFrame(rows)


def fit(d, feats, split_col):
    tr, te = d[d[split_col] == "train"], d[d[split_col] == "test"]
    X = lambda x: x[feats].values.astype(float)
    mu, sd = X(tr).mean(0), X(tr).std(0)
    sd[sd == 0] = 1.0
    n = tr["class"].value_counts()
    w = {c: len(tr) / (2 * n[c]) for c in CLS}
    lr = LogisticRegression(class_weight=w, C=1.0, solver="lbfgs", max_iter=1000).fit((X(tr) - mu) / sd, tr["class"].values)
    yt, yp = te["class"].values, lr.predict((X(te) - mu) / sd)
    f1 = f1_score(yt, yp, labels=CLS, average="macro", zero_division=0)
    rec = recall_score(yt, yp, labels=CLS, average=None, zero_division=0)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        i = rng.integers(0, len(yt), len(yt))
        boot.append(f1_score(yt[i], yp[i], labels=CLS, average="macro", zero_division=0))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # cluster bootstrap (addendum c): resample whole tracks of the test videos, 1000 resamples, default_rng(0)
    tid = te.track_id.values
    ut = np.unique(tid)
    groups = [np.flatnonzero(tid == t) for t in ut]
    rng_t = np.random.default_rng(0)
    boot_t = []
    for _ in range(1000):
        i = np.concatenate([groups[p] for p in rng_t.integers(0, len(ut), len(ut))])
        boot_t.append(f1_score(yt[i], yp[i], labels=CLS, average="macro", zero_division=0))
    lo_t, hi_t = np.percentile(boot_t, [2.5, 97.5])
    cm = [[int(((yt == a) & (yp == p)).sum()) for p in CLS] for a in CLS]
    return dict(f1=float(f1), lo=float(lo), hi=float(hi), lo_track=float(lo_t), hi_track=float(hi_t),
                n_test_tracks=int(len(ut)), rec_PET=float(rec[0]), rec_PS=float(rec[1]),
                bacc=float(rec.mean()), n_train=len(tr), n_test=len(te), cm=cm, weights=w,
                coef=dict(zip(feats, map(float, lr.coef_[0]))), positive_class=str(lr.classes_[1]))


def main_size_only():
    """3a alone, from the manifest only (runs before the z0 patches exist); the full run repeats it identically."""
    m = pd.read_csv(os.path.join(PO, "manifest_PETPS_exp.csv"))
    m["log_area"] = np.log(m.region_px_predilation)
    res = []
    for direction, col in (("original", "split_video_holdout"), ("reverse", "split_rev")):
        for subset in ("all", "unclipped"):
            d = m if subset == "all" else m[~m.clipped.astype(bool)]
            res.append(dict(model="3a size only", kind="real", direction=direction, subset=subset, features=SIZE,
                            **fit(d, SIZE, col)))
            print("3a", direction, subset, round(res[-1]["f1"], 4), flush=True)
    json.dump(dict(results=res, n_region_empty_real=0, per_class={}, per_video={}, only_3a=True),
              open(os.path.join(OUT, "part3_results.json"), "w"), indent=1)


def main():
    f = features()
    f.to_csv(os.path.join(OUT, "part3_features.csv"), index=False)
    res = []
    models = [("3a size only", "real", SIZE), ("3b brightness only", "real", BRIGHT),
              ("3b brightness only", "particle-free", BRIGHT), ("3c size + brightness", "real", SIZE + BRIGHT)]
    for name, kind, feats in models:
        for direction, col in (("original", "split_video_holdout"), ("reverse", "split_rev")):
            for subset in ("all", "unclipped"):
                d = f[f.kind == kind]
                if subset == "unclipped":
                    d = d[~d.clipped.astype(bool)]
                res.append(dict(model=name, kind=kind, direction=direction, subset=subset, features=feats,
                                **fit(d, feats, col)))
                print(name, kind, direction, subset, round(res[-1]["f1"], 4), flush=True)
    stats = (f.groupby(["kind", "class"])[BRIGHT].agg(["mean", "std"]), f.groupby(["kind", "source_file"])[BRIGHT].agg(["mean", "std"]))
    json.dump(dict(results=res, n_region_empty_real=int((f[f.kind == "real"].region_px_in_patch == 0).sum()),
                   per_class={f"{k[0]}|{k[1]}": {f"{c[0]}|{c[1]}": float(v) for c, v in row.items()} for k, row in stats[0].iterrows()},
                   per_video={f"{k[0]}|{k[1]}": {f"{c[0]}|{c[1]}": float(v) for c, v in row.items()} for k, row in stats[1].iterrows()}),
              open(os.path.join(OUT, "part3_results.json"), "w"), indent=1)


if __name__ == "__main__":
    import sys
    main_size_only() if "--3a-only" in sys.argv else main()
