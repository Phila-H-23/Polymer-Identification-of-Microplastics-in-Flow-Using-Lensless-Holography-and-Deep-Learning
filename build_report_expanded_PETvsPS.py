"""Expanded PET and PS set: report -> results/report_expanded_PETvsPS.txt (refreshed after every step, every arm and
direction). Reads the outputs of scripts/extract_PETPS_exp.py, scripts/assemble_PETPS_exp.py,
scripts/build_arms_PETPS_exp.py, scripts/scalar_baselines_PETPS_exp.py and the Part 4 run folders."""
import os
import json
import hashlib
import statistics as st
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PO = "pipeline_output"
EXP = "results/expanded_PETPS_exp"
CLS = ["PET", "PS"]
MARGIN = 0.10
OLD = {"z0": {"original": (0.8735, 0.6279), "reverse": (0.7593, 0.5301)}, "size": {"original": 0.6283, "reverse": 0.5862}}
DIRS = (("original", "", "split_video_holdout"), ("reverse", "_rev", "split_rev"))
W = 110


def rd(p, **kw):
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


def js(p):
    return json.load(open(p)) if os.path.exists(p) else None


def txt(p):
    return open(p, encoding="utf-8").read().strip() if os.path.exists(p) else None


def ms(v):
    return (st.mean(v), st.pstdev(v)) if len(v) > 1 else ((v[0], float("nan")) if v else (float("nan"),) * 2)


def f(t):
    return f"{t[0]:.4f}+/-{t[1]:.4f}"


def title(s):
    return ["", "=" * W, s, "=" * W]


def chance(m, col):
    tr, te = m[m[col] == "train"]["class"], m[m[col] == "test"]["class"].values
    p = tr.value_counts(normalize=True).reindex(CLS).values
    rng = np.random.default_rng(0)
    s = [f1_score(te, rng.choice(CLS, size=len(te), p=p), labels=CLS, average="macro", zero_division=0) for _ in range(1000)]
    return float(np.mean(s)), float(np.std(s))


def t_reading(real, pf, gap, floor, ch):
    if gap > floor and real > ch + MARGIN:
        return "T1"
    if gap <= floor and pf > ch + MARGIN:
        return "T2"
    if real <= ch + MARGIN and pf <= ch + MARGIN:
        return "T3"
    return "T4"


def z_reading(s, r):
    return "1" if s >= r - 0.05 else ("2" if s <= r - 0.15 else "3")


def header(proj, status):
    s0 = txt("results/sha256_train_compact_cnn_start_exp.txt")
    s1 = hashlib.sha256(open("train_compact_cnn.py", "rb").read()).hexdigest()
    seeds = "not started"
    if proj:
        seeds = (f"z0 original direction seeds {proj['original_seeds'].replace(' ', ', ')}, reverse seeds 0, 1, 2; "
                 f"intensity_bgcorr_vnorm the same seeds if run")
    return [
        "EXPANDED PET AND PS PATCH SET FROM THE EXISTING RECORDINGS, THEN PET vs PS AGAIN",
        f"sha256 of train_compact_cnn.py at the start: {s0.split()[0] if s0 else '?'}",
        f"sha256 of train_compact_cnn.py at this report refresh: {s1}"
        + (" (unchanged)" if s0 and s0.split()[0] == s1 else " (CHANGED)"),
        "Extraction script and parameter: holo_pipeline_starter.py (it produced pipeline_output/manifest.csv), frame",
        "  count parameter N_KEPT_FRAMES = 18, sampled frames = unique(round(linspace(0, n_frames - 1, N_KEPT_FRAMES))).",
        "  That script appends to manifest.csv, so it was not run; scripts/extract_PETPS_exp.py takes its function",
        "  definitions and constants from it unchanged (parsed with ast) and copies its Step 6 and Step 7 loops. Same",
        "  thresholds (z 7.0, debris z 5.0, displacement 5 px, comparison offset 10 frames, recurrence 15 px, 500 to",
        "  60000 px regions), same cached reference pipeline_output/background_reference.npz, same 64 x 64 crop.",
        "  Expanded grid: unique(round(linspace(0, n_frames - 1, 17 x K + 1))), which contains the 18 original frames.",
        "  pipeline_output/manifest_recon_prop_rev.csv and manifest_background_recon_prop_rev.csv (previous task) were",
        "  written by an inline python command, not a saved script: they copy the 2026-10-04 reverse-fold column of",
        "  manifest_bgcorrected_common849_rev.csv into the recon_prop manifests, patch by patch.",
        "Background-only pairs: scripts/build_background_dataset.py rule (Chebyshev >= 96 px from every detection of the",
        "  frame, box in bounds, numpy default_rng(42)); stored positions kept for the old patches, the same sampler run",
        "  over the new patches only. Builds: scripts/build_reconstructed_dataset.py (--z 0 --s0-only) and",
        "  scripts/build_bgcorrected_dataset.py, unchanged, through scripts/build_arms_PETPS_exp.py (paths only).",
        "Training: same command line as scripts/run_size_arms.sh: train_compact_cnn.py --channels <arm> --split",
        "  video_holdout (real) / particle (particle-free) --class-weights <file> --classes PET,PS --epochs 150",
        "  --patience 30 --seed <s>, no --aug (as in every earlier z0 run); batch 32, AdamW lr 1e-3, weight decay 1e-4,",
        "  cosine annealing, CompactHoloCNN width 32, dropout 0.3, rotation 0/180, translation +/-4 px, gain +/-5%,",
        "  noise 0.01, CPU. Class weights class_weights_twoclass_PETvsPS_exp.json (original) and",
        "  class_weights_twoclass_PETvsPS_rev_exp.json (reverse), n_train / (2 x n_c), same file for real and control.",
        "  intensity_bgcorr_vnorm runs through scripts/train_vnorm_exp.py, which calls train_compact_cnn.main() with",
        "  VIDEO_BG_STATS_PATH set to pipeline_output/video_background_stats_PETPS_exp.json (file not edited).",
        "Splits: video roles unchanged in both directions; validation drawn by whole tracks (addendum b; first drawn by",
        "  whole frames in Part 1, kept as *_framegroup columns). Same split for real and particle-free runs.",
        f"Seeds actually run: {seeds}.",
        (f"Timing: first finished run {proj['first_run_seconds'] / 60:.1f} min; projected z0 batch with original seeds "
         f"0-4: {proj['projected_z0_batch_hours_5_seeds']:.2f} h"
         + ("; above 5 h, so the original direction uses seeds 0 to 2 for both arms."
            if proj["original_seeds"] == "0 1 2" else "; within 5 h, so seeds 0 to 4 are used.")) if proj else
        "Timing: no run finished yet.",
        f"intensity_bgcorr_vnorm arm: {status or 'not decided yet (runs last, after the z0 batch and Part 3)'}",
        "Readings fixed before any result (task text): T1 to T4 (margin 0.10 over chance, floor = larger seed std);",
        "  Z1 to Z3 (size-only S vs z0 real R); Y1 to Y3 (scalar model F vs R); H-hold / H-partial / H-gone.",
    ]


def part0():
    v, fr = rd(f"{EXP}/part0_videos.csv"), rd(f"{EXP}/part0_frames.csv")
    L = title("PART 0  ORIGINAL EXTRACTION, PET AND PS VIDEOS (read-only)")
    if v is None:
        return L + ["not run yet"]
    L += ["a) sampling rule: frames = unique(round(linspace(0, n_frames - 1, 18))); one comparison frame 10 frames later",
          "   (10 earlier near the end) per sampled frame for the debris check. Filter counts summed over the 18 frames:",
          "   regions = connected regions after thresholding (z > 7) and 10 px dilation; size filter keeps 500 to 60000 px;",
          "   debris: still elevated (comparison-frame z > 5 in the box) or moved < 5 px; recurrence: within 15 px of an",
          "   earlier kept detection (any frame). Edge clipping drops nothing: clipped patches are kept and zero padded",
          "   at training time. Frame 0 patches are dropped from the common set (no preceding frames, no background).",
          f"{'video':6s} {'frames':>6s} {'regions':>7s} {'<500px':>6s} {'>60000':>6s} {'detect':>6s} {'deb z':>6s} "
          f"{'deb <5px':>8s} {'recur':>5s} {'kept':>5s} {'clipped':>7s} {'frame0':>6s} {'in 849':>6s}"]
    for _, r in v.iterrows():
        g = fr[fr.video == r.video]
        f0 = int(g[g.frame == 0].n_final.sum())
        L.append(f"{r.video[:-8]:6s} {r.n_frames:>6d} {g.n_regions.sum():>7d} {g.n_too_small.sum():>6d} "
                 f"{g.n_too_large.sum():>6d} {g.n_detections.sum():>6d} {g.n_debris_still_elevated.sum():>6d} "
                 f"{g.n_debris_low_displacement.sum():>8d} {g.n_recurrence_dropped.sum():>5d} {g.n_final.sum():>5d} "
                 f"{g.n_clipped.sum():>7d} {f0:>6d} {r.common849_rows:>6d}")
    L.append("sampled frames and patches kept per sampled frame (frame:patches):")
    for _, r in v.iterrows():
        g = fr[fr.video == r.video]
        L.append(f"  {r.video[:-8]:5s} " + " ".join(f"{a}:{b}" for a, b in zip(g.frame, g.n_final)))
    L += ["b) reproduction check against pipeline_output/manifest.csv (the 883-row source of the common 849 set used",
          "   by the two-class and size-and-arms runs):",
          f"{'video':6s} {'manifest rows':>13s} {'reproduced':>10s} {'common 849':>10s} {'reproduced excl. frame 0':>24s} "
          f"{'max centroid dist':>17s} {'identical frame, id, x, y':>25s}"]
    for _, r in v.iterrows():
        L.append(f"{r.video[:-8]:6s} {r.manifest_rows:>13d} {r.reproduced_rows:>10d} {r.common849_rows:>10d} "
                 f"{r.reproduced_rows_excl_frame0:>24d} {r.max_centroid_dist_px:>14.1f} px {str(r.identical_frame_id_x_y):>25s}")
    ok = bool(v.identical_frame_id_x_y.all() and (v.rows_not_within_1px == 0).all())
    L.append(f"reproduces the stored rows exactly: {'yes' if ok else 'NO - stop'}")
    return L


def counts_table(old, oldr, new):
    L = [f"{'class':5s} {'video':6s} {'direction':9s} {'old train':>9s} {'old val':>7s} {'old test':>8s} "
         f"{'new train':>9s} {'new val':>7s} {'new test':>8s}"]
    for dname, sfx, col in DIRS:
        o = oldr if sfx else old
        for c in CLS:
            for vid in sorted(new[new["class"] == c].source_file.unique()):
                a = o[o.source_file == vid].split_video_holdout.value_counts()
                b = new[new.source_file == vid][col].value_counts()
                L.append(f"{c:5s} {vid[:-8]:6s} {dname:9s} {a.get('train', 0):>9d} {a.get('val', 0):>7d} {a.get('test', 0):>8d} "
                         f"{b.get('train', 0):>9d} {b.get('val', 0):>7d} {b.get('test', 0):>8d}")
            a = o[o["class"] == c].split_video_holdout.value_counts()
            b = new[new["class"] == c][col].value_counts()
            L.append(f"{c:5s} {'all':6s} {dname:9s} {a.get('train', 0):>9d} {a.get('val', 0):>7d} {a.get('test', 0):>8d} "
                     f"{b.get('train', 0):>9d} {b.get('val', 0):>7d} {b.get('test', 0):>8d}")
    return L


def part1():
    pv, m = rd(f"{EXP}/part1_videos.csv"), rd(f"{PO}/manifest_PETPS_exp.csv")
    L = title("PART 1  EXPANDED SET (manifest_PETPS_exp.csv, manifest_background_PETPS_exp.csv)")
    if pv is None or m is None:
        return L + ["not run yet"]
    choice = txt(f"{EXP}/part1_choice.txt")
    if choice:
        L += choice.split("\n")
    L += [f"{'video':6s} {'K':>2s} {'n frames':>8s} {'grid':>5s} {'old kept':>8s} {'old lost if Step 7 unchanged':>28s} "
          f"{'Step 7 unchanged total':>22s} {'final':>6s} {'final excl. frame 0':>19s}"]
    for _, r in pv.iterrows():
        L.append(f"{r.video[:-8]:6s} {r.k:>2d} {r.n_frames:>8d} {r.n_grid_frames:>5d} {r.old_kept:>8d} "
                 f"{r.old_lost_if_step7_unchanged:>28d} {r.unchanged_step7_total:>22d} {r.final_total:>6d} {r.final_excl_frame0:>19d}")
    pf = rd(f"{EXP}/part1_frames.csv")
    if pf is not None:
        L.append("patches per sampled frame, new frames vs original frames (mean, frame 0 excluded):")
        for vid, g in pf[pf.frame != 0].groupby("video"):
            L.append(f"  {vid[:-8]:5s} original frames {g[g.is_original_frame].n_final.mean():.2f}, "
                     f"new frames {g[~g.is_original_frame].n_final.mean():.2f}; recurrence drops in new frames "
                     f"{g[~g.is_original_frame].n_recurrence_dropped.sum()}, debris drops in new frames "
                     f"{(g[~g.is_original_frame].n_debris_still_elevated + g[~g.is_original_frame].n_debris_low_displacement).sum()}")
    old = pd.read_csv(f"{PO}/manifest_bgcorrected_common849.csv")
    oldr = pd.read_csv(f"{PO}/manifest_bgcorrected_common849_rev.csv")
    old, oldr = old[old["class"].isin(CLS)], oldr[oldr["class"].isin(CLS)]
    L += ["", "Patches per video, class, split and direction (old = common 849 set with its stored splits; new = expanded set",
          "with validation by track, see the addendum section):"]
    L += counts_table(old, oldr, m)
    L.append(f"totals: old PET {int((old['class'] == 'PET').sum())}, PS {int((old['class'] == 'PS').sum())}; "
             f"new PET {int((m['class'] == 'PET').sum())}, PS {int((m['class'] == 'PS').sum())} "
             f"(old patches inside the new set: {int(m.old_patch.sum())})")
    ch = m[m.old_patch]
    a_, b_ = ("split_video_holdout_framegroup", "split_rev_framegroup") if "split_rev_framegroup" in m else \
        ("split_video_holdout", "split_rev")
    moved = ch[ch[a_] != ch.split_old]
    movedr = ch[ch[b_] != ch.split_rev_old]
    tr = lambda d, a, b: ", ".join(f"{x} to {y} {n}" for (x, y), n in d.groupby([a, b]).size().items())
    L.append(f"old patches whose split changed when validation was first drawn by whole frames (Part 1 rule): original "
             f"{len(moved)} ({tr(moved, 'split_old', a_)}), reverse {len(movedr)} ({tr(movedr, 'split_rev_old', b_)}); "
             f"test membership unchanged. The splits in the table above and used for training are the track-based ones "
             f"(addendum section).")
    s1 = rd("results/recon_fidelity/s1_detection_stats.csv")
    if s1 is not None:
        s1 = s1[s1.kind == "real"].set_index("patch_file")
        o = ch.set_index("patch_file")
        same = all((o[c] == s1.loc[o.index, c]).all() for c in ("bbox_w_px", "bbox_h_px", "region_px_predilation"))
        L.append(f"size features of the old patches equal the S1 values (results/recon_fidelity/s1_detection_stats.csv): {same}")
    L += ["", "Size distribution, region area before dilation in px (1 px = 6.9 x 6.9 um), new set (old set in brackets):",
          f"{'group':10s} {'n':>11s} {'median':>13s} {'P10':>11s} {'P90':>11s}"]
    groups = [(vid[:-8], m.source_file == vid, old.source_file == vid) for vid in sorted(m.source_file.unique())]
    groups += [(c, m["class"] == c, old["class"] == c) for c in CLS]
    s1all = rd("results/recon_fidelity/s1_detection_stats.csv")
    s1map = dict(zip(s1all.patch_file, s1all.region_px_predilation)) if s1all is not None else {}
    for name, a, b in groups:
        x = m[a].region_px_predilation.values
        y = np.array([s1map.get(p, np.nan) for p in old[b].patch_file])
        q = lambda z, p: np.nanpercentile(z, p) if len(z) else np.nan
        L.append(f"{name:10s} {len(x):>5d} ({len(y):>3d}) {np.median(x):>6.0f} ({np.nanmedian(y):>4.0f}) "
                 f"{q(x, 10):>5.0f} ({q(y, 10):>3.0f}) {q(x, 90):>5.0f} ({q(y, 90):>4.0f})")
    dup = rd(f"{EXP}/part1_duplicates.csv")
    if dup is not None:
        L += ["", "Duplicate check (report only, nothing dropped): share of patches whose centroid lies within 20 px of a",
              "centroid in another sampled frame of the same video (the recurrence check already removes pairs < 15 px):"]
        for vid, g in dup.groupby("source_file"):
            nn, oo = g[~g.old_patch], g[g.old_patch]
            L.append(f"  {vid[:-8]:5s} new-frame patches {nn.within_20px.mean():.1%} ({int(nn.within_20px.sum())} of {len(nn)}); "
                     f"old patches, for reference {oo.within_20px.mean():.1%} ({int(oo.within_20px.sum())} of {len(oo)})")
        nn = dup[~dup.old_patch]
        L.append(f"  all   new-frame patches {nn.within_20px.mean():.1%} ({int(nn.within_20px.sum())} of {len(nn)})")
    return L


def part2():
    L = title("PART 2  DATASETS (z0 and intensity_bgcorr_vnorm), CLASS WEIGHTS")
    for arm, name in (("z0", "z0 (patches_recon_s0_z0_PETPS_exp)"), ("bgcorr", "bgcorrected (patches_bgcorrected_PETPS_exp)")):
        c = js(f"{EXP}/part2_{arm}_check.json")
        L.append(f"{name}: " + (f"built; old patches byte-identical to the stored ones: {c['old_identical']}, "
                                f"different: {len(c['old_different'])}" if c else "not built yet"))
    for nm in ("class_weights_twoclass_PETvsPS_exp.json", "class_weights_twoclass_PETvsPS_rev_exp.json"):
        w = js(f"{PO}/{nm}")
        L.append(f"{nm}: {w if w else 'not written yet'}")
    s = js(f"{PO}/video_background_stats_PETPS_exp.json")
    if s:
        L.append("per-video normalisation (video_background_stats_PETPS_exp.json; stored file values in brackets):")
        o = js(f"{PO}/video_background_stats.json")
        for v, d in s.items():
            L.append(f"  {v[:-8]:5s} mean {d['mean']:.4f} ({o[v]['mean']:.4f}), std {d['std']:.4f} ({o[v]['std']:.4f}), "
                     f"{d['n_patches']} background patches ({o[v]['n_patches']})")
    return L


def tracks_section():
    m = rd(f"{PO}/manifest_PETPS_exp.csv")
    sp = js(f"{EXP}/split_by_track.json")
    L = title("ADDENDUM: TRACKS, VALIDATION BY TRACK, CLUSTER BOOTSTRAP")
    if m is None or "track_id" not in m.columns or not sp:
        return L + ["not run yet"]
    L += ["a) Tracks (scripts/tracks_PETPS_exp.py): within each video, detections whose centroids lie < 20 px apart in two",
          "   different sampled frames are linked, with no limit on the frame gap; connected components are tracks.",
          "   Old set: the same linking among the 561 old patches only, with their stored splits. New set: all patches, with",
          "   the track-based splits below. A particle-free patch carries the track of its paired real patch.",
          f"{'set':4s} {'class':5s} {'video':6s} {'split':5s} {'patches':>7s} {'tracks':>6s} {'median':>6s} {'P90':>5s} {'largest':>7s}"]
    old = pd.read_csv(f"{PO}/manifest_bgcorrected_common849.csv")[["patch_file", "split_video_holdout"]]
    o = m[m.track_id_oldset.notna()].drop(columns=["split_video_holdout"]).merge(old, on="patch_file")
    for name, d, tcol in (("old", o, "track_id_oldset"), ("new", m, "track_id")):
        rows = [(c, v, s, g) for (c, v, s), g in d.groupby(["class", "source_file", "split_video_holdout"])]
        rows += [(c, "all", s, g) for (c, s), g in d.groupby(["class", "split_video_holdout"])]
        rows += [(c, "all", "all", g) for c, g in d.groupby("class")]
        for c, v, s, g in rows:
            sz = g.groupby(tcol).size()
            L.append(f"{name:4s} {c:5s} {v.replace('_600.mat', ''):6s} {s:5s} {len(g):>7d} {len(sz):>6d} {np.median(sz):>6.0f} "
                     f"{np.percentile(sz, 90):>5.0f} {sz.max():>7d}")
    if m.collision_kept.notna().any():
        ck = m[m.old_patch.astype(bool)].collision_kept.astype(str).str.lower().eq("true")
        L.append(f"collision_kept = true (old patch kept although the unchanged 15 px order would drop it): {int(ck.sum())} "
                 f"of {int(m.old_patch.sum())} old patches in the set (frame 0 excluded); per video "
                 + ", ".join(f"{v[:-8]} {int(n)}" for v, n in m[m.old_patch.astype(bool)].assign(ck=ck.values)
                             .groupby("source_file").ck.sum().items()))
    else:
        L.append("collision_kept: not filled yet (needs the Step 6 rerun after the z0 build)")
    L += ["b) Validation by track: whole tracks in numpy default_rng(0) order go to validation; the prefix whose patch",
          "   count is closest to the target is used (original 15/85 of the class's training-video patches, reverse 0.175",
          "   of the newly added training video, the fractions of the stored splits). Test splits unchanged. Fallback to",
          "   frame grouping if the largest track in the pool exceeds 15% of the class's training patches.",
          f"{'class':5s} {'direction':9s} {'pool':>5s} {'tracks':>6s} {'largest':>7s} {'15% limit':>9s} {'target':>6s} {'val':>4s}  rule used"]
    for d in sp["decisions"]:
        L.append(f"{d['cls']:5s} {d['direction']:9s} {d['pool_patches']:>5d} {d['n_tracks']:>6d} {d['largest_track']:>7d} "
                 f"{d['threshold_15pct']:>9.1f} {d['target_val']:>6.1f} {d.get('val_patches', 0):>4d}  "
                 + ("frame grouping kept (fallback)" if d["fallback_framegroup"] else "tracks"))
    for dname in ("original", "reverse"):
        mv = sp["moved"][dname]
        L.append(f"moved against the frame-grouped assignment, {dname}: {sum(mv.values())} patches ("
                 + ", ".join(f"{k} {n}" for k, n in mv.items()) + f"); sampled frames now holding both train and val "
                 f"patches: {sp['moved'][dname + '_frames_spanning_train_and_val']}")
    for nm, w in sp["weights"].items():
        L.append(f"{nm}: {w['weights']} from training counts {w['n_train']} (frame-grouped version kept as "
                 f"{nm.replace('_exp.json', '_framegroup_exp.json')})")
    L += ["c) Cluster bootstrap: Part 3 gives both intervals side by side ('patch CI' and 'track CI'). CNN runs:",
          "   train_compact_cnn.py does not save per-patch test predictions (results.json holds the classification report,",
          "   confusion matrix and history; best.pt holds the weights), so no track-level bootstrap is reported for the",
          "   CNN arms. Predictions could be regenerated from each run's best.pt by a separate evaluation script without",
          "   changing train_compact_cnn.py; that was not run.",
          "d) The pre-registered T, Z and Y readings are unchanged and use the seed-based floor and the patch-level values;",
          "   the track intervals are an additional column only."]
    return L


def load_runs(prefix, seeds):
    out = {}
    for s in seeds:
        p = f"results/{prefix}_s{s}/results.json"
        if os.path.exists(p):
            out[s] = json.load(open(p))
    return out


def part4(m, proj):
    """Returns lines and a dict of summary values per (arm, direction)."""
    L = title("PART 4  PET vs PS CNN RUNS ON THE EXPANDED SET (macro-F1 mean +/- population std over seeds)")
    summ, bad, failed = {}, [], []
    seeds_orig = [int(x) for x in proj["original_seeds"].split()] if proj else [0, 1, 2, 3, 4]
    for arm in ("z0", "intensity_bgcorr_vnorm"):
        for dname, sfx, col in DIRS:
            seeds = seeds_orig if not sfx else [0, 1, 2]
            cw = f"{PO}/class_weights_twoclass_PETvsPS{sfx}_exp.json"
            rr = load_runs(f"real_tc_PETvsPS_{arm}_exp{sfx}", seeds)
            br = load_runs(f"bg_tc_PETvsPS_{arm}_exp{sfx}", seeds)
            for runs in (rr, br):
                bad += [r["args"]["out"] for r in runs.values()
                        if r["args"].get("class_weights") != cw or r["args"].get("classes") != "PET,PS"]
            for s in seeds:
                for kind, runs in (("real", rr), ("bg", br)):
                    d = f"results/{kind}_tc_PETvsPS_{arm}_exp{sfx}_s{s}"
                    if os.path.isdir(d) and s not in runs:
                        failed.append(d)
            if not rr and not br:
                continue
            ch = chance(m, col)
            L += ["", f"{arm}, {dname} direction (real n={len(rr)}, particle-free n={len(br)}; chance {f(ch)})",
                  f"{'seed':>4s} {'real F1':>8s} {'real rec PET':>12s} {'real rec PS':>11s} {'p-free F1':>9s} "
                  f"{'p-free rec PET':>14s} {'p-free rec PS':>13s}"]
            g = lambda r, k: r["test_report"][k]
            for s in seeds:
                a, b = rr.get(s), br.get(s)
                fa = lambda r, c, k: f"{g(r, c)[k]:.4f}" if r else "   -  "
                L.append(f"{s:>4d} {fa(a, 'macro avg', 'f1-score'):>8s} {fa(a, 'PET', 'recall'):>12s} {fa(a, 'PS', 'recall'):>11s} "
                         f"{fa(b, 'macro avg', 'f1-score'):>9s} {fa(b, 'PET', 'recall'):>14s} {fa(b, 'PS', 'recall'):>13s}")
            if not rr or not br:
                continue
            val = lambda runs, c, k: ms([g(r, c)[k] for r in runs.values()])
            real, pf = val(rr, "macro avg", "f1-score"), val(br, "macro avg", "f1-score")
            gap, floor = real[0] - pf[0], max(real[1], pf[1])
            complete = len(rr) == len(br) == len(seeds)
            T = t_reading(real[0], pf[0], gap, floor, ch[0]) if complete else "(incomplete)"
            summ[(arm, dname)] = dict(real=real, pf=pf, gap=gap, floor=floor, chance=ch[0], T=T, n=(len(rr), len(br)))
            L += [f"mean real {f(real)} | particle-free {f(pf)} | gap {gap:+.4f} | floor {floor:.4f} "
                  f"{'(gap above floor)' if gap > floor else '(gap within floor)'} | chance {ch[0]:.4f} | reading {T}",
                  f"per-class recall, mean +/- std: real PET {f(val(rr, 'PET', 'recall'))}, PS {f(val(rr, 'PS', 'recall'))}; "
                  f"particle-free PET {f(val(br, 'PET', 'recall'))}, PS {f(val(br, 'PS', 'recall'))}",
                  f"per-class F1, mean: real PET {val(rr, 'PET', 'f1-score')[0]:.4f}, PS {val(rr, 'PS', 'f1-score')[0]:.4f}; "
                  f"particle-free PET {val(br, 'PET', 'f1-score')[0]:.4f}, PS {val(br, 'PS', 'f1-score')[0]:.4f}",
                  f"balanced accuracy: real {np.mean([(g(r, 'PET')['recall'] + g(r, 'PS')['recall']) / 2 for r in rr.values()]):.4f}, "
                  f"particle-free {np.mean([(g(r, 'PET')['recall'] + g(r, 'PS')['recall']) / 2 for r in br.values()]):.4f}",
                  "mean confusion matrix over seeds (rows true class, columns predicted: PET PS)"]
            for name, runs in (("real", rr), ("particle-free", br)):
                cm = np.mean([np.array(r["confusion_matrix"], float) for r in runs.values()], axis=0)
                L.append(f"  {name:13s} true PET: {cm[0, 0]:7.1f} {cm[0, 1]:7.1f} | true PS: {cm[1, 0]:7.1f} {cm[1, 1]:7.1f}")
    return L, summ, bad, failed


def part3(R):
    res = js(f"{EXP}/part3_results.json")
    L = title("PART 3  SCALAR BASELINES ON THE EXPANDED SET (no CNN)")
    if not res:
        return L + ["not run yet"], {}
    L += ["3a size only: log(region area before dilation), bbox width, bbox height. 3b brightness only, from the z0",
          "patch (I / median background, float32, before any normalisation): whole-patch mean, std, min, max, and mean",
          "and sum of (z0 - 1) over the detected region before dilation (its pixels inside the crop); on particle-free",
          "patches the region is a central disc with the same pixel count as the paired real region. 3c size +",
          "brightness, real only. Logistic regression L2, C=1.0, lbfgs, class weights n_train / (2 x n_c), standardised",
          "on the training split, fitted on the training split only; 95% bootstrap interval, 1000 resamples of test",
          "patches, numpy default_rng(0). Particle-free 'unclipped' rows drop the pairs whose real patch is clipped.",
          f"Detected region with no pixel inside the crop (region mean set to 0): {res['n_region_empty_real']} real patches.",
          "Readings (fixed in the task): R = z0 real CNN macro-F1, same direction (all test patches). Y1 F >= R - 0.05;",
          "Y2 F <= R - 0.15; Y3 between. Z1 to Z3 for 3a use the same cut points. Readings are left open while R is missing.",
          "Intervals: 'patch CI' resamples test patches (pre-registered); 'track CI' resamples whole test tracks",
          "(addendum c, an additional column; readings do not use it).",
          f"{'model':22s} {'patches':13s} {'direction':9s} {'subset':9s} {'n tr/te':>9s} {'tracks':>6s}  {'F':>6s} "
          f"{'patch CI':>16s} {'track CI':>16s}  {'rec PET':>7s} {'rec PS':>6s} {'bacc':>6s}  {'R':>6s} {'F - R':>7s} reading"]
    out = {}
    for r in res["results"]:
        Rv = R.get(r["direction"])
        rdg = "open (R not finished)"
        if Rv is not None:
            y = z_reading(r["f1"], Rv)
            rdg = (f"Z{y} / " if r["model"].startswith("3a") else "") + f"Y{y}"
        out[(r["model"], r["kind"], r["direction"], r["subset"])] = (r["f1"], rdg)
        tci = f"[{r['lo_track']:.3f}, {r['hi_track']:.3f}]" if "lo_track" in r else "-"
        L.append(f"{r['model']:22s} {r['kind']:13s} {r['direction']:9s} {r['subset']:9s} {r['n_train']:>4d}/{r['n_test']:<4d} "
                 f"{r.get('n_test_tracks', 0):>6d}  {r['f1']:.4f} [{r['lo']:.3f}, {r['hi']:.3f}] {tci:>16s}  "
                 f"{r['rec_PET']:7.3f} {r['rec_PS']:6.3f} {r['bacc']:6.3f}  "
                 + (f"{Rv:6.4f} {r['f1'] - Rv:+7.4f} " if Rv is not None else f"{'-':>6s} {'-':>7s} ") + rdg)
    L += ["", "Brightness features, mean +/- std per class and per video (z0 units; region features over the detected region",
          "for real patches and over the matched central disc for particle-free patches):"]
    feats = ["z0_mean", "z0_std", "z0_min", "z0_max", "region_mean_z0m1", "region_sum_z0m1"]
    L.append(f"{'patches':13s} {'group':6s} " + " ".join(f"{x:>19s}" for x in feats))
    for key in ("per_class", "per_video"):
        for g, d in res[key].items():
            kind, grp = g.split("|")
            L.append(f"{kind:13s} {grp.replace('_600.mat', ''):6s} " + " ".join(
                f"{d[x + '|mean']:>9.4f}+/-{d[x + '|std']:<7.4f}" if abs(d[x + '|mean']) < 1000 else
                f"{d[x + '|mean']:>9.1f}+/-{d[x + '|std']:<7.1f}" for x in feats))
    return L, out


def main():
    m = rd(f"{PO}/manifest_PETPS_exp.csv")
    proj = js(f"{EXP}/part4_projection.json")
    status = txt(f"{EXP}/part4_intensity_status.txt")
    lines = header(proj, status) + part0() + part1() + part2()
    summ, bad, failed = {}, [], []
    if m is not None:
        L4, summ, bad, failed = part4(m, proj)
    else:
        L4 = title("PART 4  PET vs PS CNN RUNS") + ["not started"]
    R = {d: summ[("z0", d)]["real"][0] for _, d in [(None, "original"), (None, "reverse")]
         if ("z0", d) in summ and summ[("z0", d)]["T"] != "(incomplete)"}
    L3, p3 = part3(R)
    lines += L3 + L4 + tracks_section()

    lines += title("READINGS AND COMPARISON WITH THE 849-PATCH RESULTS")
    lines.append(f"{'arm':24s} {'direction':9s} {'real':>15s} {'particle-free':>15s} {'gap':>7s} {'floor':>6s} {'chance':>6s}  T")
    for (arm, d), s in summ.items():
        lines.append(f"{arm:24s} {d:9s} {f(s['real'])} {f(s['pf'])} {s['gap']:+.4f} {s['floor']:.4f} {s['chance']:.4f}  {s['T']}")
    zr = {}
    for d in ("original", "reverse"):
        S = p3.get(("3a size only", "real", d, "all"))
        if S and d in R:
            zr[d] = "Z" + z_reading(S[0], R[d])
            lines.append(f"Z reading, {d}: S = {S[0]:.4f} (size only, all patches), R = {R[d]:.4f} -> {zr[d]}")
    if len(zr) == 2 and all(("z0", d) in summ for d in ("original", "reverse")):
        t = [summ[("z0", d)]["T"] for d in ("original", "reverse")]
        z = list(zr.values())
        if t == ["T1", "T1"] and z == ["Z2", "Z2"]:
            h = "H-hold"
        elif t.count("T1") == 1 or "Z3" in z:
            h = "H-partial"
        elif all(x in ("T2", "T3") for x in t):
            h = "H-gone"
        else:
            h = "none of H-hold, H-partial, H-gone applies"
        lines.append(f"Summary reading: z0 T = {t[0]} (original), {t[1]} (reverse); size only {z[0]}, {z[1]} -> {h}")
    else:
        lines.append("Summary reading (H): open until the z0 batch is finished.")
    lines += ["", "Old (849 patches) vs new (expanded) values:",
              f"{'quantity':28s} {'old original':>12s} {'new original':>12s} {'old reverse':>11s} {'new reverse':>11s}"]
    nv = lambda k, d: (f"{summ[('z0', d)][k][0]:.4f}" if ("z0", d) in summ else "-")
    lines.append(f"{'z0 real macro-F1':28s} {OLD['z0']['original'][0]:>12.4f} {nv('real', 'original'):>12s} "
                 f"{OLD['z0']['reverse'][0]:>11.4f} {nv('real', 'reverse'):>11s}")
    lines.append(f"{'z0 particle-free macro-F1':28s} {OLD['z0']['original'][1]:>12.4f} {nv('pf', 'original'):>12s} "
                 f"{OLD['z0']['reverse'][1]:>11.4f} {nv('pf', 'reverse'):>11s}")
    sv = lambda d: (f"{p3[('3a size only', 'real', d, 'all')][0]:.4f}" if ("3a size only", "real", d, "all") in p3 else "-")
    lines.append(f"{'size only macro-F1 (all)':28s} {OLD['size']['original']:>12.4f} {sv('original'):>12s} "
                 f"{OLD['size']['reverse']:>11.4f} {sv('reverse'):>11s}")
    lines.append("Old chance: 0.4997 original, 0.3895 reverse (old test sets); new chance is in the Part 4 blocks.")

    failed_txt = "results/logs_expanded_PETPS_exp/failed.txt"
    logged = open(failed_txt).read().split() if os.path.exists(failed_txt) else []
    lines += ["", f"class-weights file and --classes PET,PS recorded as expected in every finished run: "
                  f"{'yes' if not bad else 'NO: ' + str(bad)}",
              f"failed or incomplete runs: {failed or 'none'}" + (f"; logged failures: {logged}" if logged else "")]
    for name, label in (("observations.txt", "MEASURED OBSERVATIONS"), ("interpretation.txt", "Interpretation (not verified)"),
                        ("summary.txt", "FIVE-LINE SUMMARY")):
        t = txt(f"{EXP}/{name}")
        lines += title(label) + ([t] if t else ["(written when the runs are finished)"])
    text = "\n".join(lines)
    assert "—" not in text and "significant" not in text.lower()
    open("results/report_expanded_PETvsPS.txt", "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
