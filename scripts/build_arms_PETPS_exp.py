"""Part 2: z0 and intensity_bgcorr_vnorm datasets for the expanded PET and PS set (new files only, suffix _exp).
Runs the EXISTING build code unchanged, with only its input manifests and output paths pointed at new files:
  z0      scripts/build_reconstructed_dataset.py main(), --z 0 --suffix _z0_PETPS_exp --s0-only
          (the existing z0 build was --z 0 --suffix _z0 --s0-only), inputs manifest_PETPS_exp.csv and
          manifest_background_PETPS_exp.csv
  bgcorr  scripts/build_bgcorrected_dataset.py main(), outputs patches_bgcorrected_PETPS_exp,
          patches_background_bgcorrected_PETPS_exp and their manifests
  post    checks that every old patch is byte-identical to the stored one, copies the split, track and collision
          columns from the current master manifests into the build's output manifests, writes the reverse-fold
          manifests (*_PETPS_rev_exp.csv: split_video_holdout / split := split_rev) and, for bgcorr, the new per-video
          normalisation file pipeline_output/video_background_stats_PETPS_exp.json (same formula as
          video_background_stats.json: float32 /255, mean and std over all pixels and 4 channels of that video's
          background-only bgcorrected patches; reproduced exactly on the stored file before use)
usage: python scripts/build_arms_PETPS_exp.py z0|bgcorr|post_z0|post_bgcorr
"""
import os
import sys
import json
import numpy as np
import pandas as pd

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
PO = os.path.join(D, "pipeline_output")
sys.path.insert(0, os.path.join(D, "scripts"))
REAL, BG = os.path.join(PO, "manifest_PETPS_exp.csv"), os.path.join(PO, "manifest_background_PETPS_exp.csv")
ARMS = {"z0": dict(real="manifest_recon_s0_z0_PETPS_exp.csv", bg="manifest_background_recon_s0_z0_PETPS_exp.csv",
                   rdir="patches_recon_s0_z0_PETPS_exp", bdir="patches_background_recon_s0_z0_PETPS_exp",
                   old_rdir="patches_recon_s0_z0", old_bdir="patches_background_recon_s0_z0"),
        "bgcorr": dict(real="manifest_bgcorrected_PETPS_exp.csv", bg="manifest_background_bgcorrected_PETPS_exp.csv",
                       rdir="patches_bgcorrected_PETPS_exp", bdir="patches_background_bgcorrected_PETPS_exp",
                       old_rdir="patches_bgcorrected", old_bdir="patches_background_bgcorrected")}


def build(arm):
    if arm == "z0":
        import build_reconstructed_dataset as br
        br.REAL_MANIFEST_PATH, br.BG_MANIFEST_PATH = REAL, BG
        sys.argv = ["build_reconstructed_dataset.py", "--z", "0", "--suffix", "_z0_PETPS_exp", "--s0-only"]
        br.main()
    else:
        # Overnight order of 2026-10-07/08: the raw-intensity build and batch are step 6 (lowest priority). The build
        # waits here until results/overnight_release_bgcorr.flag exists (created by scripts/overnight_coordinator.py
        # when steps 1 to 5 are finished or cannot run, before 07:45, with at least 3 GB RAM free, or by hand).
        import time
        release = os.path.join(D, "results", "overnight_release_bgcorr.flag")
        while not os.path.exists(release):
            print(f"HOLD {time.strftime('%Y-%m-%d %H:%M:%S')}: waiting for {release} "
                  f"(raw-intensity build is step 6 of the overnight order)", flush=True)
            time.sleep(300)
        print(f"released {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        import build_bgcorrected_dataset as bb
        bb.REAL_MANIFEST_PATH, bb.BG_MANIFEST_PATH = REAL, BG
        a = ARMS["bgcorr"]
        bb.OUT_REAL_PATCH_DIR, bb.OUT_BG_PATCH_DIR = os.path.join(PO, a["rdir"]), os.path.join(PO, a["bdir"])
        bb.OUT_REAL_MANIFEST, bb.OUT_BG_MANIFEST = os.path.join(PO, a["real"]), os.path.join(PO, a["bg"])
        bb.main()


def post(arm):
    a = ARMS[arm]
    r, b = pd.read_csv(os.path.join(PO, a["real"])), pd.read_csv(os.path.join(PO, a["bg"]))
    rm, bm = pd.read_csv(REAL), pd.read_csv(BG)
    assert set(r.patch_file) == set(rm.patch_file) and set(b.patch_file) == set(bm.patch_file), "build incomplete"
    same, diff = 0, []
    for df, new_dir, old_dir in ((r, a["rdir"], a["old_rdir"]), (b, a["bdir"], a["old_bdir"])):
        for f in df[df.old_patch.astype(bool)].patch_file:
            x, y = np.load(os.path.join(PO, new_dir, f)), np.load(os.path.join(PO, old_dir, f))
            if x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y):
                same += 1
            else:
                diff.append(f)
    print(f"{arm}: old patches byte-identical to the stored ones: {same}, different: {len(diff)} {diff[:5]}")
    # The build copies the manifest rows it read at its start. Split, track and collision columns are taken from the
    # current master manifests (track-based validation was set after the z0 build had started), and the build's
    # own output manifests are rewritten with them (index files only; no patch changes).
    key = ["source_file", "frame", "particle_id"]
    sync = ["split_video_holdout", "split_rev", "split_video_holdout_framegroup", "split_rev_framegroup", "track_id",
            "track_id_oldset", "collision_kept"]
    r = r[[c for c in r.columns if c not in sync]].merge(rm[["patch_file"] + sync], on="patch_file", how="left")
    bsync = ["split", "split_rev", "split_video_holdout_framegroup", "split_rev_framegroup", "track_id", "collision_kept"]
    b = b[[c for c in b.columns if c not in bsync]].merge(bm[key + bsync], on=key, how="left")
    assert r.split_video_holdout.notna().all() and b.split.notna().all() and len(r) == len(rm) and len(b) == len(bm)
    r.to_csv(os.path.join(PO, a["real"]), index=False)
    b.to_csv(os.path.join(PO, a["bg"]), index=False)
    r2, b2 = r.copy(), b.copy()
    r2["split_video_holdout"], b2["split"] = r2.split_rev, b2.split_rev
    r2.to_csv(os.path.join(PO, a["real"].replace("_PETPS_exp", "_PETPS_rev_exp")), index=False)
    b2.to_csv(os.path.join(PO, a["bg"].replace("_PETPS_exp", "_PETPS_rev_exp")), index=False)
    out = dict(arm=arm, old_identical=same, old_different=diff)
    if arm == "bgcorr":
        stats = {}
        for v, g in b.groupby("source_file"):
            x = np.stack([np.load(os.path.join(PO, a["bdir"], f)) for f in g.patch_file]).astype(np.float32) / 255.0
            stats[v] = dict(mean=float(x.mean()), std=float(x.std()), n_patches=int(len(g)), n_pixels=int(x[..., 0].size))
        json.dump(stats, open(os.path.join(PO, "video_background_stats_PETPS_exp.json"), "w"), indent=2)
        print(json.dumps(stats, indent=1))
    json.dump(out, open(os.path.join(D, "results", "expanded_PETPS_exp", f"part2_{arm}_check.json"), "w"), indent=2)


if __name__ == "__main__":
    step = sys.argv[1]
    if step in ("z0", "bgcorr"):
        build(step)
    else:
        post(step.replace("post_", ""))
