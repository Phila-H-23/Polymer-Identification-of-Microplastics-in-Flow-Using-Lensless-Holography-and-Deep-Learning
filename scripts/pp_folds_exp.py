"""Overnight step 4 (only if PP qualifies, after arm G): PET vs PP on z0 in three folds. New files only (suffix _exp).
1. PP patches: the kept detections of the PP yield check (results/pp_yield_detections_exp.csv: same extraction rules as
   the PET and PS expansion, up to 256 sampled frames per recording, frame 0 left out); paired particle-free rows (the
   stored rows of manifest_background.csv for old patches, the same sampler as scripts/build_background_dataset.py with
   numpy default_rng(42) for new ones); track ids (20 px linkage, scripts/tracks_PETPS_exp.tracks).
2. z0 for the PP frames with the existing build code unchanged (scripts/build_reconstructed_dataset.py, --z 0 --s0-only,
   suffix _z0_PP_exp); combined patch folders patches_recon_s0_z0_PETPP_exp and patches_background_recon_s0_z0_PETPP_exp
   hold copies of the PET z0 patches of the expanded set and the new PP patches.
3. Folds (order PP1, PP2, PP500FRAMES): fold k tests on PP recording k and trains on the other two PP recordings,
   validation by whole tracks (numpy default_rng(0) track order, prefix closest to 15% of the training PP patches);
   PET trains on PET1, PET3, PET4 and tests on PET2 with the track-based validation of the expanded set. Class weights
   n_train / (2 x n_c) from each fold's own training split (class_weights_twoclass_PETvsPP_fold<k>_exp.json).
4. Training: train_compact_cnn.py unchanged, z0 regime of the PET vs PS runs (--channels recon_s0 --classes PET,PP
   --epochs 150 --patience 30), real and particle-free, seeds 0 to 2; a failed run is retried once into <out>_retry1.
5. Per fold: chance (1000 draws), T reading with the seed floor, per-class recall, track intervals
   (scripts/eval_track_ci_exp.analyse), size-only baseline (Part 3a method) with its patch and track intervals; PET vs PS
   z0 numbers printed beside them without conclusions. -> results/report_PETvsPP_exp.txt
"""
import os
import sys
import json
import time
import shutil
import subprocess
import numpy as np
import pandas as pd

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import tracks_PETPS_exp as tr  # noqa: E402
import build_background_dataset as bbd  # noqa: E402

PO, RES = "pipeline_output", "results"
LOGD = f"{RES}/logs_overnight_exp"
KEY = ["source_file", "frame", "particle_id"]
FOLDS = [("PP1", "PP1_600.mat"), ("PP2", "PP2_600.mat"), ("PP500FRAMES", "PP500FRAMES.mat")]
M_PP, B_PP = f"{PO}/manifest_PP_exp.csv", f"{PO}/manifest_background_PP_exp.csv"
RDIR, BDIR = f"{PO}/patches_recon_s0_z0_PETPP_exp", f"{PO}/patches_background_recon_s0_z0_PETPP_exp"
STATE = f"{RES}/pp_folds_state_exp.json"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(f"{LOGD}/pp_folds_exp.log", "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


def assemble():
    det = pd.read_csv(f"{RES}/pp_yield_detections_exp.csv")
    det["track_id"] = tr.tracks(det)
    bg_old = pd.read_csv(f"{PO}/manifest_background.csv")
    old = det[det.old_patch][KEY].merge(bg_old, on=KEY, how="left")
    assert old.patch_file.notna().all()
    rng = np.random.default_rng(bbd.SEED)
    new_rows = []
    for (sf, fr), fd in det[~det.old_patch].sort_values(KEY).groupby(["source_file", "frame"], sort=False):
        ex_ = list(zip(det[(det.source_file == sf) & (det.frame == fr)].x.astype(int),
                       det[(det.source_file == sf) & (det.frame == fr)].y.astype(int)))
        for _, r in fd.iterrows():
            cx, cy, ok = bbd.sample_background_center(rng, ex_, 1224, 1024)
            new_rows.append(dict(patch_file=f"{sf[:-4]}_bgexpparticle{int(r.particle_id) - 1000}_frame{int(fr)}.npy",
                                 particle_id=int(r.particle_id), frame=int(fr), x=int(cx), y=int(cy), source_file=sf,
                                 **{"class": "PP"}))
    bg = pd.concat([old[["patch_file", "particle_id", "frame", "x", "y", "source_file", "class"]], pd.DataFrame(new_rows)],
                   ignore_index=True)
    k = det.set_index(KEY)
    bg["track_id"] = [k.loc[tuple(t), "track_id"] for t in bg[KEY].values]
    bg["old_patch"] = [bool(k.loc[tuple(t), "old_patch"]) for t in bg[KEY].values]
    # split columns needed by the build's sanity prints; the fold splits are added later
    det["split_video_holdout"], bg["split"] = "train", "train"
    det.to_csv(M_PP, index=False)
    bg.to_csv(B_PP, index=False)
    log(f"assembled {len(det)} PP patches, {det.track_id.nunique()} tracks; {len(new_rows)} new background positions")


def build_z0():
    import build_reconstructed_dataset as br
    br.REAL_MANIFEST_PATH, br.BG_MANIFEST_PATH = M_PP, B_PP
    sys.argv = ["build_reconstructed_dataset.py", "--z", "0", "--suffix", "_z0_PP_exp", "--s0-only"]
    br.main()


def combine():
    os.makedirs(RDIR, exist_ok=True)
    os.makedirs(BDIR, exist_ok=True)
    pet = pd.read_csv(f"{PO}/manifest_PETPS_exp.csv")
    pet = pet[pet["class"] == "PET"]
    petb = pd.read_csv(f"{PO}/manifest_background_PETPS_exp.csv")
    petb = petb[petb["class"] == "PET"]
    pp, ppb = pd.read_csv(M_PP), pd.read_csv(B_PP)
    for src, dst, files in ((f"{PO}/patches_recon_s0_z0_PETPS_exp", RDIR, pet.patch_file),
                            (f"{PO}/patches_background_recon_s0_z0_PETPS_exp", BDIR, petb.patch_file),
                            (f"{PO}/patches_recon_s0_z0_PP_exp", RDIR, pp.patch_file),
                            (f"{PO}/patches_background_recon_s0_z0_PP_exp", BDIR, ppb.patch_file)):
        for f in files:
            if not os.path.exists(os.path.join(dst, f)):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    return pet, petb, pp, ppb


def folds(pet, petb, pp, ppb):
    cols = ["patch_file", "particle_id", "frame", "x", "y", "source_file", "class", "track_id", "clipped", "bbox_w_px",
            "bbox_h_px", "region_px_predilation"]
    out = {}
    for k, (tag, fn) in enumerate(FOLDS, start=1):
        s = pd.Series("train", index=pp.index)
        s[pp.source_file == fn] = "test"
        pool = pp[pp.source_file != fn]
        s[pool.index[tr.track_val(pool, 0.15)]] = "val"
        r = pd.concat([pet[cols].assign(split_video_holdout=pet.split_video_holdout.values),
                       pp[cols].assign(split_video_holdout=s.values)], ignore_index=True)
        kk = r.set_index(KEY)
        b = pd.concat([petb[["patch_file", "particle_id", "frame", "x", "y", "source_file", "class", "track_id"]],
                       ppb[["patch_file", "particle_id", "frame", "x", "y", "source_file", "class", "track_id"]]],
                      ignore_index=True)
        b["split"] = [kk.loc[tuple(t), "split_video_holdout"] for t in b[KEY].values]
        rp, bp = f"{PO}/manifest_recon_s0_z0_PETPP_fold{k}_exp.csv", f"{PO}/manifest_background_recon_s0_z0_PETPP_fold{k}_exp.csv"
        r.to_csv(rp, index=False)
        b.to_csv(bp, index=False)
        n = r[r.split_video_holdout == "train"]["class"].value_counts()
        w = {c: round(float(n.sum() / (2 * n[c])), 4) for c in sorted(n.index)}
        cw = f"{PO}/class_weights_twoclass_PETvsPP_fold{k}_exp.json"
        json.dump(w, open(cw, "w"), indent=2)
        out[k] = dict(tag=tag, real=rp, bg=bp, cw=cw, weights=w, counts=pd.crosstab([r["class"], r.source_file],
                                                                                      r.split_video_holdout).to_dict())
        log(f"fold {k} ({tag} held out): weights {w}")
    return out


def train_all(F):
    runs = {}
    for k, f in F.items():
        runs[f["tag"]] = {"real": {}, "particle-free": {}}
        for seed in (0, 1, 2):
            for kind, pre, man, pdir, split in (("real", "real", f["real"], RDIR, "video_holdout"),
                                                ("particle-free", "bg", f["bg"], BDIR, "particle")):
                base = f"{RES}/{pre}_tc_PETvsPP_z0_fold{k}_exp_s{seed}"
                done = None
                for out in (base, base + "_retry1"):
                    if os.path.exists(f"{out}/results.json"):
                        done = out
                        break
                    cmd = [sys.executable, "train_compact_cnn.py", "--channels", "recon_s0", "--split", split, "--manifest",
                           man, "--patch-dir", pdir, "--class-weights", f["cw"], "--classes", "PET,PP", "--epochs", "150",
                           "--patience", "30", "--seed", str(seed), "--out", out]
                    t0 = time.time()
                    with open(f"{LOGD}/{os.path.basename(out)}.log", "w") as fh:
                        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
                    ok = rc == 0 and os.path.exists(f"{out}/results.json")
                    log(f"{os.path.basename(out)} exit={rc} {time.time() - t0:.0f}s {'ok' if ok else 'FAILED'}")
                    if ok:
                        done = out
                        break
                if done:
                    runs[f["tag"]][kind][seed] = done
                json.dump(dict(runs=runs), open(STATE, "w"), indent=1)
    return runs


def report(F, runs):
    import eval_track_ci_exp as ev
    import scalar_baselines_PETPS_exp as sb
    sb.CLS = ["PET", "PP"]
    L, summ, excl = ev.analyse("PET vs PP z0", runs, f"{RES}/PETvsPP_predictions_exp.csv")
    head = ["PET vs PP ON z0, THREE FOLDS (overnight step 4)",
            "Fold k holds out PP recording k as the test video and trains on the other two PP recordings (validation by whole",
            "tracks, 15% of the training PP patches); PET trains on PET1, PET3, PET4 and tests on PET2 (track-based validation",
            "of the expanded set). Real and particle-free, seeds 0 to 2, train_compact_cnn.py unchanged, z0 regime of the",
            "PET vs PS runs; class weights from each fold's own training split. Chance by 1000 random draws per fold.",
            "PP500FRAMES.mat is dated 29 Dec 2022 and has 500 frames (the other recordings: 10 Jan 2023, 600 frames), so the",
            "PP500FRAMES fold also tests a different recording session.",
            f"Runs excluded from the track analysis: {excl or 'none'}"]
    for k, f in F.items():
        head.append(f"fold {k} ({f['tag']} held out): class weights {f['weights']}")
    S = ["", "Size-only baseline per fold (Part 3a method: log region area, bbox width and height, logistic regression):"]
    for k, f in F.items():
        r = pd.read_csv(f["real"])
        r["log_area"] = np.log(r.region_px_predilation)
        x = sb.fit(r, sb.SIZE, "split_video_holdout")
        S.append(f"  fold {k} ({f['tag']}): macro-F1 {x['f1']:.4f}, patch CI [{x['lo']:.4f}, {x['hi']:.4f}], track CI "
                 f"[{x['lo_track']:.4f}, {x['hi_track']:.4f}]; recall PET {x['rec_PET']:.3f}, PP {x['rec_PS']:.3f}")
    side = ["", "Beside them, without conclusions: z0 PET vs PS (expanded set) from results/report_z0_trackCI_exp.txt"]
    z = f"{RES}/z0_trackCI_summary_exp.json"
    if os.path.exists(z):
        for d, v in json.load(open(z)).items():
            side.append(f"  PET vs PS {d}: real {v['real'][0]:.4f}+/-{v['real'][1]:.4f}, particle-free {v['pf'][0]:.4f}+/-"
                        f"{v['pf'][1]:.4f}, gap {v['gap']:+.4f}, floor {v['floor']:.4f}, {v['T']}")
    for fold, v in summ.items():
        side.append(f"  PET vs PP fold {fold}: real {v['real'][0]:.4f}+/-{v['real'][1]:.4f}, particle-free {v['pf'][0]:.4f}+/-"
                    f"{v['pf'][1]:.4f}, gap {v['gap']:+.4f}, floor {v['floor']:.4f}, {v['T']}")
    text = "\n".join(head + L + S + side)
    text = text.replace("recall PET", "recall PET")
    assert "\u2014" not in text and "significant" not in text.lower()
    open(f"{RES}/report_PETvsPP_exp.txt", "w", encoding="utf-8").write(text + "\n")
    json.dump(summ, open(f"{RES}/PETvsPP_summary_exp.json", "w"), indent=1, default=float)


def main():
    os.makedirs(LOGD, exist_ok=True)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if not os.path.exists(M_PP):
        assemble()
    if not os.path.exists(f"{PO}/manifest_recon_s0_z0_PP_exp.csv"):
        log("z0 build for the PP frames (existing build code, loads one whole video at a time)")
        build_z0()
    pet, petb, pp, ppb = combine()
    F = folds(pet, petb, pp, ppb)
    runs = train_all(F)
    report(F, runs)
    log("step 4 finished")


if __name__ == "__main__":
    main()
