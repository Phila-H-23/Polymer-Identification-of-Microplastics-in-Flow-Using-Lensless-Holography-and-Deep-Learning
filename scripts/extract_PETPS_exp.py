"""Expanded PET and PS patch set from the EXISTING recordings. New files only (suffix _exp).

Extraction code: holo_pipeline_starter.py (the script that produced pipeline_output/manifest.csv; frame-count
parameter N_KEPT_FRAMES = 18, frames = unique(round(linspace(0, n_frames - 1, N_KEPT_FRAMES)))). That script runs
top to bottom and appends to manifest.csv, so it is not run here. Instead its function definitions (demosaic,
local_variance, normalized_local_variance, compute_z_field, detect_particles) and constants (SEARCH_RANGE,
COMPARISON_OFFSET, MIN_DISPLACEMENT_PX, DEBRIS_CHECK_Z_THRESH, RECURRENCE_DISTANCE_PX, CROP_SIZE, N_KEPT_FRAMES,
CACHE_VERSION) are parsed out of holo_pipeline_starter.py with ast and executed unchanged, and the Step 6 (per-frame
detection + debris check) and Step 7 (cross-frame recurrence check) loop bodies are copied from it (lines 425-601).
Same cached detection reference pipeline_output/background_reference.npz (read only).

  --part0         re-run on the 18 originally sampled frames of every PET and PS video, compare with manifest.csv,
                  write results/expanded_PETPS_exp/part0_*.csv
  --part1 [--k-pet 2 3] [--k-ps 5 6]   denser grid unique(round(linspace(0, n - 1, 17 * K + 1))), which contains
                  the 18 original frames (every K-th grid point). Step 6 runs once on the union of the candidate
                  grids; Step 7 per K; per class the K whose total (frame 0 excluded) is closest to the target
                  (PET 700, PS 1100) is kept. Writes the detections, their size features and pre-dilation region
                  masks (cropped like the patch) to results/expanded_PETPS_exp/
"""
import ast
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage
import trackpy as tp

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
PO = os.path.join(D, "pipeline_output")
OUT = os.path.join(D, "results", "expanded_PETPS_exp")
sys.path.insert(0, os.path.join(D, "scripts"))
import reconstruction_validate as rv  # noqa: E402

VIDEOS = ["PET1_600.mat", "PET2_600.mat", "PET3_600.mat", "PET4_600.mat", "PS1_600.mat", "PS2_600.mat"]
H_DEM, W_DEM = 1024, 1224


def load_holo():
    tree = ast.parse(open(os.path.join(D, "holo_pipeline_starter.py"), encoding="utf-8").read())
    fn = {"demosaic", "local_variance", "normalized_local_variance", "compute_z_field", "detect_particles"}
    cn = {"N_KEPT_FRAMES", "SEARCH_RANGE", "COMPARISON_OFFSET", "MIN_DISPLACEMENT_PX", "DEBRIS_CHECK_Z_THRESH",
          "RECURRENCE_DISTANCE_PX", "CROP_SIZE", "CACHE_VERSION"}
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in fn)
             or (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in cn)]
    ns = {"np": np, "pd": pd, "ndimage": ndimage}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "holo_pipeline_starter.py", "exec"), ns)
    assert fn | cn <= set(ns), sorted((fn | cn) - set(ns))
    return ns


H = load_holo()


def grid(n_frames, k):
    return np.unique(np.linspace(0, n_frames - 1, min(17 * k + 1, n_frames)).round().astype(int))


def region_info(frame, bg_mean, bg_std, var_window=9, smooth=9, z_thresh=7.0, dilate_px=10, min_pixels=500,
                max_pixels=60000):
    """Measurement only (does not decide anything): the same steps as detect_particles up to labelling, plus the
    pre-dilation pixel count per region and the counts needed for the Part 0 filter table."""
    z = H["compute_z_field"](frame, bg_mean, bg_std, var_window, smooth)
    mask = z > z_thresh
    mask_dil = ndimage.binary_dilation(mask, iterations=dilate_px)
    labeled, n = ndimage.label(mask_dil)
    sizes = ndimage.sum(mask_dil, labeled, index=range(1, n + 1))
    pre = ndimage.sum(mask, labeled, index=range(1, n + 1))
    regions = {}
    for lab, (size, p, sl) in enumerate(zip(sizes, pre, ndimage.find_objects(labeled)), start=1):
        if sl is None or size < min_pixels or size > max_pixels:
            continue
        ys, xs = sl
        regions[((xs.start + xs.stop) / 2, (ys.start + ys.stop) / 2, xs.stop - xs.start, ys.stop - ys.start)] = \
            (lab, int(size), int(p))
    counts = dict(n_regions=int(n), n_too_small=int((sizes < min_pixels).sum()),
                  n_too_large=int((sizes > max_pixels).sum()))
    return regions, counts, labeled, mask


def crop(a, cx, cy, half=32):
    y0, y1 = max(0, cy - half), min(a.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(a.shape[1], cx + half)
    return a[y0:y1, x0:x1]


def step6(get_sum, total_frames, kept_frame_indices, bg_lv_mean, bg_lv_std):
    """holo_pipeline_starter.py Step 6 (lines 425-538), SAVE_REJECTED_FOR_REVIEW branches removed, with counters
    and measurements added. Decisions are made exactly as there."""
    detect_particles, compute_z_field = H["detect_particles"], H["compute_z_field"]
    SEARCH_RANGE, COMPARISON_OFFSET = H["SEARCH_RANGE"], H["COMPARISON_OFFSET"]
    MIN_DISPLACEMENT_PX, DEBRIS_CHECK_Z_THRESH = H["MIN_DISPLACEMENT_PX"], H["DEBRIS_CHECK_Z_THRESH"]
    final_rows, frame_stats, meas = [], [], {}
    for kept_idx in kept_frame_indices:
        kept_idx = int(kept_idx)
        frame = get_sum(kept_idx)
        dets = detect_particles(frame, bg_lv_mean, bg_lv_std)
        regions, counts, labeled, mask = region_info(frame, bg_lv_mean, bg_lv_std)
        st = dict(frame=kept_idx, **counts, n_detections=len(dets), n_debris_still_elevated=0,
                  n_debris_low_displacement=0, n_kept_step6=0)
        assert len(dets) == len(regions)
        for _, r in dets.iterrows():  # measurements for every detection of this frame
            lab, size, p = regions[(r["x"], r["y"], r["w"], r["h"])]
            assert size == r["pixel_count"]
            m = crop((labeled == lab) & mask, int(r["x"]), int(r["y"]))
            meas[(kept_idx, r["x"], r["y"])] = dict(bbox_w_px=r["w"], bbox_h_px=r["h"], region_px_dilated=size,
                                                    region_px_predilation=p, region_px_in_patch=int(m.sum()),
                                                    mask=m.copy())
        if len(dets) == 0:
            frame_stats.append(st)
            continue
        dets["frame"] = 0

        comp_idx = kept_idx + COMPARISON_OFFSET
        if comp_idx >= total_frames:
            comp_idx = kept_idx - COMPARISON_OFFSET
        comp_idx = int(max(0, min(total_frames - 1, comp_idx)))
        comp_frame = get_sum(comp_idx)
        comp_dets = detect_particles(comp_frame, bg_lv_mean, bg_lv_std)
        comp_z = compute_z_field(comp_frame, bg_lv_mean, bg_lv_std)
        st["comparison_frame"] = comp_idx

        def _comp_still_elevated(row):
            y0 = max(0, int(row["y"] - row["h"] / 2))
            y1 = min(comp_z.shape[0], int(row["y"] + row["h"] / 2))
            x0 = max(0, int(row["x"] - row["w"] / 2))
            x1 = min(comp_z.shape[1], int(row["x"] + row["w"] / 2))
            region = comp_z[y0:y1, x0:x1]
            return region.max() if region.size else -np.inf

        if len(comp_dets) == 0:
            dets["frame"] = kept_idx
            dets["displacement"] = np.nan
            z_vals = dets.apply(_comp_still_elevated, axis=1)
            still_present = z_vals > DEBRIS_CHECK_Z_THRESH
            st["n_debris_still_elevated"] += int(still_present.sum())
            st["n_kept_step6"] += int((~still_present).sum())
            final_rows.append(dets[~still_present])
            frame_stats.append(st)
            continue

        comp_dets["frame"] = 1
        pair = pd.concat([dets, comp_dets], ignore_index=True)
        linked = tp.link(pair, search_range=SEARCH_RANGE, memory=0)

        kept_this_frame = []
        for particle_id, group in linked.groupby("particle"):
            row_kept = group[group["frame"] == 0]
            if len(row_kept) == 0:
                continue
            row_kept = row_kept.iloc[0].copy()
            z_val = _comp_still_elevated(row_kept)
            if z_val > DEBRIS_CHECK_Z_THRESH:
                st["n_debris_still_elevated"] += 1
                continue
            if len(group) == 1:
                row_kept["displacement"] = np.nan
            else:
                row_comp = group[group["frame"] == 1].iloc[0]
                displacement = np.hypot(row_kept["x"] - row_comp["x"], row_kept["y"] - row_comp["y"])
                row_kept["displacement"] = displacement
                if displacement < MIN_DISPLACEMENT_PX:
                    st["n_debris_low_displacement"] += 1
                    continue
            row_kept["frame"] = kept_idx
            kept_this_frame.append(row_kept)
            st["n_kept_step6"] += 1

        if kept_this_frame:
            final_rows.append(pd.DataFrame(kept_this_frame))
        frame_stats.append(st)
    return pd.concat(final_rows, ignore_index=True).reset_index(drop=True), pd.DataFrame(frame_stats), meas


def step7(selected_df, priority_xy=None):
    """holo_pipeline_starter.py Step 7 (lines 575-594). priority_xy=None: unchanged. Otherwise kept_xy starts with
    the given (x, y) list (the detections kept by the original 18-frame run) and those rows are kept, so the
    original patches stay a subset; every other row is dropped if it lies within RECURRENCE_DISTANCE_PX of any
    kept detection. Same distance rule and same end state (no two kept detections within 15 px)."""
    R = H["RECURRENCE_DISTANCE_PX"]
    selected_df = selected_df.sort_values("frame").reset_index(drop=True)
    if priority_xy is None:
        kept_xy, keep_mask = [], []
        for _, row in selected_df.iterrows():
            is_recurrence = any(np.hypot(row["x"] - kx, row["y"] - ky) < R for kx, ky in kept_xy)
            keep_mask.append(not is_recurrence)
            if not is_recurrence:
                kept_xy.append((row["x"], row["y"]))
    else:
        pri = set(priority_xy)
        kept_xy = list(priority_xy)
        keep_mask = []
        for _, row in selected_df.iterrows():
            key = (row["frame"], row["x"], row["y"])
            if key in pri:
                keep_mask.append(True)
                continue
            is_recurrence = any(np.hypot(row["x"] - kx, row["y"] - ky) < R for _, kx, ky in kept_xy)
            keep_mask.append(not is_recurrence)
            if not is_recurrence:
                kept_xy.append(key)
    dropped = selected_df[[not k for k in keep_mask]].copy()
    selected_df = selected_df[keep_mask].reset_index(drop=True)
    selected_df["particle_id"] = selected_df.index
    return selected_df, dropped


def load_video(video):
    block, mode = rv.load_video_block(os.path.join(D, video))
    n = block.shape[-1] if mode == "scipy" else block.shape[0]
    cache = {}

    def get_sum(i):  # small cache (most recent 24 frames) so dense grids do not fill memory
        if i not in cache:
            if len(cache) >= 24:
                del cache[next(iter(cache))]
            cache[i] = np.sum(np.stack(H["demosaic"](rv.get_raw_frame(block, mode, int(i))), axis=0).astype(np.int64),
                              axis=0)
        return cache[i]
    return block, n, get_sum


def run_video(video, frames, bg_mean, bg_std):
    block, n, get_sum = load_video(video)
    sel, fstats, meas = step6(get_sum, n, frames, bg_mean, bg_std)
    del block
    return n, sel, fstats, meas


def ref():
    r = np.load(os.path.join(PO, "background_reference.npz"))
    assert str(r["version"]) == H["CACHE_VERSION"]
    return r["bg_lv_mean"], r["bg_lv_std"]


def clipped(x, y):
    return (y - 32 < 0) or (y + 32 > H_DEM) or (x - 32 < 0) or (x + 32 > W_DEM)


def part0():
    os.makedirs(OUT, exist_ok=True)
    bg_mean, bg_std = ref()
    man = pd.read_csv(os.path.join(PO, "manifest.csv"))
    c849 = pd.read_csv(os.path.join(PO, "manifest_bgcorrected_common849.csv"))
    rows_f, rows_v, rows_d = [], [], []
    for video in VIDEOS:
        block, n, get_sum = load_video(video)
        frames = np.unique(np.linspace(0, n - 1, min(H["N_KEPT_FRAMES"], n)).round().astype(int))
        sel, fst, meas = step6(get_sum, n, frames, bg_mean, bg_std)
        del block
        final, dropped = step7(sel)
        fst["video"] = video
        fst["n_recurrence_dropped"] = fst.frame.map(dropped.frame.value_counts()).fillna(0).astype(int)
        fst["n_final"] = fst.frame.map(final.frame.value_counts()).fillna(0).astype(int)
        final["x_int"], final["y_int"] = final.x.astype(int), final.y.astype(int)
        fst["n_clipped"] = fst.frame.map(final[[clipped(x, y) for x, y in zip(final.x_int, final.y_int)]]
                                         .frame.value_counts()).fillna(0).astype(int)
        rows_f.append(fst)
        m = man[man.source_file == video].sort_values(["frame", "particle_id"])
        f = final.sort_values(["frame", "particle_id"])
        same_ids = (len(m) == len(f)) and (m[["frame", "particle_id", "x", "y"]].values
                                           == f[["frame", "particle_id", "x_int", "y_int"]].values).all()
        maxd, unmatched = 0.0, 0  # nearest reproduced centroid in the same frame, one-to-one by frame
        for fr, g in m.groupby("frame"):
            h = f[f.frame == fr]
            for _, r in g.iterrows():
                if len(h) == 0:
                    unmatched += 1
                    continue
                d = np.hypot(h.x_int - r.x, h.y_int - r.y).min()
                maxd = max(maxd, d)
                unmatched += int(d > 1)
        c = c849[c849.source_file == video]
        rows_v.append(dict(video=video, n_frames=n, sampled_frames=" ".join(map(str, frames)),
                           manifest_rows=len(m), reproduced_rows=len(f), common849_rows=len(c),
                           reproduced_rows_excl_frame0=int((f.frame != 0).sum()),
                           max_centroid_dist_px=maxd, rows_not_within_1px=unmatched,
                           identical_frame_id_x_y=bool(same_ids)))
        final["video"] = video
        rows_d.append(final.drop(columns=[c for c in final.columns if c not in
                                          ("video", "frame", "particle_id", "x_int", "y_int", "x", "y", "w", "h")]))
        print(rows_v[-1], flush=True)
    pd.concat(rows_f).to_csv(os.path.join(OUT, "part0_frames.csv"), index=False)
    pd.DataFrame(rows_v).to_csv(os.path.join(OUT, "part0_videos.csv"), index=False)
    pd.concat(rows_d).to_csv(os.path.join(OUT, "part0_reproduced_detections.csv"), index=False)


def part1(k_pet, k_ps, target_pet=700, target_ps=1100):
    """Step 6 once on the union of the candidate grids; Step 7 per candidate K; per class the K whose total
    (frame 0 excluded) is closest to the target is kept."""
    bg_mean, bg_std = ref()
    man = pd.read_csv(os.path.join(PO, "manifest.csv"))
    key = lambda d: set(zip(d.frame, d.x, d.y))
    per_video, cand = {}, []
    for video in VIDEOS:
        ks = k_pet if video.startswith("PET") else k_ps
        block, n, get_sum = load_video(video)
        old_frames = np.unique(np.linspace(0, n - 1, min(H["N_KEPT_FRAMES"], n)).round().astype(int))
        union = sorted(set().union(*[set(grid(n, k)) for k in ks]))
        sel_all, fst_all, meas = step6(get_sum, n, union, bg_mean, bg_std)
        del block
        old_final, _ = step7(sel_all[sel_all.frame.isin(old_frames)])  # the original 18-frame run (as in Part 0)
        m = man[man.source_file == video].sort_values(["frame", "particle_id"])
        o = old_final.sort_values(["frame", "particle_id"])
        assert (m[["frame", "particle_id", "x", "y"]].values == np.c_[o.frame, o.particle_id, o.x.astype(int),
                                                                       o.y.astype(int)]).all()
        old_keys = key(old_final)
        per_video[video] = dict(n=n, old_frames=old_frames, old_final=old_final, meas=meas, k={})
        for k in ks:
            frames = grid(n, k)
            assert set(old_frames) <= set(frames)
            sel = sel_all[sel_all.frame.isin(frames)]
            unchanged, _ = step7(sel)
            final, dropped = step7(sel, priority_xy=sorted(old_keys))
            assert old_keys <= key(final)
            fst = fst_all[fst_all.frame.isin(frames)].copy()
            per_video[video]["k"][k] = (frames, final, dropped, fst)
            cand.append(dict(video=video, k=k, n_frames=n, n_grid_frames=len(frames), n_old_frames=len(old_frames),
                             old_kept=len(old_keys), old_lost_if_step7_unchanged=len(old_keys - key(unchanged)),
                             unchanged_step7_total=len(unchanged), final_total=len(final),
                             final_excl_frame0=int((final.frame != 0).sum())))
            print(cand[-1], flush=True)
    cand = pd.DataFrame(cand)
    cand["class"] = np.where(cand.video.str.startswith("PET"), "PET", "PS")
    tot = cand.groupby(["class", "k"]).final_excl_frame0.sum()
    chosen = {c: int((tot[c] - (target_pet if c == "PET" else target_ps)).abs().idxmin()) for c in ("PET", "PS")}
    os.makedirs(OUT, exist_ok=True)
    cand.to_csv(os.path.join(OUT, "part1_candidates.csv"), index=False)
    rows, rows_f, masks, summary = [], [], {}, []
    for video, pv in per_video.items():
        c = "PET" if video.startswith("PET") else "PS"
        k = chosen[c]
        frames, final, dropped, fst = pv["k"][k]
        old_final, meas = pv["old_final"], pv["meas"]
        fst["video"] = video
        fst["is_original_frame"] = fst.frame.isin(pv["old_frames"])
        fst["n_recurrence_dropped"] = fst.frame.map(dropped.frame.value_counts()).fillna(0).astype(int)
        fst["n_final"] = fst.frame.map(final.frame.value_counts()).fillna(0).astype(int)
        rows_f.append(fst)
        oid = {(f, x, y): pid for f, x, y, pid in zip(old_final.frame, old_final.x, old_final.y, old_final.particle_id)}
        tag = video.replace(".mat", "")
        new_i = 0
        for _, r in final.iterrows():
            kf = (r.frame, r.x, r.y)
            mm = meas[kf]
            if kf in oid:
                pid, pf, old = int(oid[kf]), f"{tag}_particle{int(oid[kf])}_frame{int(r.frame)}.npy", True
            else:
                pid, pf, old = 1000 + new_i, f"{tag}_expparticle{new_i}_frame{int(r.frame)}.npy", False
                new_i += 1
            rows.append(dict(patch_file=pf, particle_id=pid, frame=int(r.frame), x=int(r.x), y=int(r.y),
                             source_file=video, **{"class": c}, old_patch=old, bbox_w_px=mm["bbox_w_px"],
                             bbox_h_px=mm["bbox_h_px"], region_px_dilated=mm["region_px_dilated"],
                             region_px_predilation=mm["region_px_predilation"],
                             region_px_in_patch=mm["region_px_in_patch"], clipped=clipped(int(r.x), int(r.y))))
            masks[pf] = mm["mask"]
        summary.append(cand[(cand.video == video) & (cand.k == k)].iloc[0].to_dict())
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "part1_detections.csv"), index=False)
    pd.concat(rows_f).to_csv(os.path.join(OUT, "part1_frames.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "part1_videos.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "part1_region_masks.npz"), **{k.replace(".npy", ""): v for k, v in masks.items()})
    print("totals per class and K (frame 0 excluded):", tot.to_dict(), "chosen:", chosen)


def collisions():
    """Addendum: which old patches the UNCHANGED Step 7 order would drop on the chosen grid (collision_kept). Reruns
    Step 6 on the chosen grid (part1_videos.csv) of every video; also saves the Step 6 survivors."""
    bg_mean, bg_std = ref()
    chosen = pd.read_csv(os.path.join(OUT, "part1_videos.csv")).set_index("video")
    key = lambda d: set(zip(d.frame, d.x, d.y))
    rows, surv = [], []
    for video in VIDEOS:
        k = int(chosen.loc[video, "k"])
        block, n, get_sum = load_video(video)
        old_frames = np.unique(np.linspace(0, n - 1, min(H["N_KEPT_FRAMES"], n)).round().astype(int))
        frames = grid(n, k)
        sel, _, _ = step6(get_sum, n, frames, bg_mean, bg_std)
        del block
        old_final, _ = step7(sel[sel.frame.isin(old_frames)])
        unchanged, _ = step7(sel)
        lost = key(old_final) - key(unchanged)
        assert len(lost) == int(chosen.loc[video, "old_lost_if_step7_unchanged"])
        tag = video.replace(".mat", "")
        for f, x, y, pid in zip(old_final.frame, old_final.x, old_final.y, old_final.particle_id):
            if int(f) != 0:
                rows.append(dict(patch_file=f"{tag}_particle{int(pid)}_frame{int(f)}.npy", source_file=video,
                                 collision_kept=(f, x, y) in lost))
        s = sel[["frame", "x", "y", "w", "h", "pixel_count"]].copy()
        s["video"] = video
        surv.append(s)
        print(video, "K", k, "old lost if Step 7 unchanged:", len(lost), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "part1_collisions.csv"), index=False)
    pd.concat(surv).to_csv(os.path.join(OUT, "part1_step6_survivors_chosen_grid.csv"), index=False)


if __name__ == "__main__":
    tp.quiet()
    ap = argparse.ArgumentParser()
    ap.add_argument("--part0", action="store_true")
    ap.add_argument("--part1", action="store_true")
    ap.add_argument("--collisions", action="store_true")
    ap.add_argument("--k-pet", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--k-ps", type=int, nargs="+", default=[5, 6])
    a = ap.parse_args()
    if a.part0:
        part0()
    if a.part1:
        part1(a.k_pet, a.k_ps)
    if a.collisions:
        collisions()
