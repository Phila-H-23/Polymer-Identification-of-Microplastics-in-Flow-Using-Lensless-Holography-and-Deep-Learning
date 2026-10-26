"""Overnight step 3: PP yield check (read-only, detection only, no patches saved). For PP1_600.mat, PP2_600.mat and
PP500FRAMES.mat, with the extraction rules of the PET and PS expansion (scripts/extract_PETPS_exp.py: the detection
functions and constants of holo_pipeline_starter.py, Step 6 debris check, Step 7 15 px recurrence rule with the old
18-frame patches kept where they collide, frame 0 left out), grid unique(round(linspace(0, n - 1, 17 x 15 + 1))) =
up to 256 sampled frames per recording. Frames are read in column bands into a disk-backed stack.
Reports per recording: detections kept after the 15 px rule, tracks (20 px linkage across sampled frames), frame-edge
fraction. Rule fixed before the result: PP QUALIFIES if each of the three recordings gives at least 50 tracks.
The kept detections (with size features) are saved for step 4: results/pp_yield_detections_exp.csv.
Runs in background mode; waits while the chain is in a heavy step, and while results/overnight_overlap_ok.json says
overlap with training is not allowed (written by scripts/overnight_coordinator.py)."""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import recon_check_followup as fu  # noqa: E402  (idle priority, single thread; read_pass, gate)
import extract_PETPS_exp as ex  # noqa: E402
import tracks_PETPS_exp as tr  # noqa: E402

D = fu.D
OVERLAP = os.path.join(D, "results", "overnight_overlap_ok.json")
VIDS = [("PP1", "PP1_600.mat"), ("PP2", "PP2_600.mat"), ("PP500FRAMES", "PP500FRAMES.mat")]
K = 15


def training_running():
    """True if a training log was written to in the last 60 s (chain z0 runs, overnight arms, PET vs PP folds)."""
    now = time.time()
    for d in ("logs_expanded_PETPS_exp", "logs_overnight_exp"):
        p = os.path.join(D, "results", d)
        if os.path.isdir(p):
            for f in os.listdir(p):
                if f.endswith(".log") and now - os.path.getmtime(os.path.join(p, f)) < 60 and (f.startswith("real_") or f.startswith("bg_")):
                    return True
    return False


def overlap_wait(what):
    t0, said = time.time(), False
    while training_running():
        ok = json.load(open(OVERLAP)).get("ok", False) if os.path.exists(OVERLAP) else False
        if ok:
            break
        if not said:
            fu.log(f"pp_yield: waiting before '{what}': training running and overlap not allowed (results/overnight_overlap_ok.json)")
            said = True
        time.sleep(60)
    if said:
        fu.log(f"pp_yield: resumed '{what}' after {(time.time() - t0) / 60:.1f} min")


class BandStack:
    """Frames stored band after band (column bands of 64 px, all wanted frames per band), written with plain file
    writes so nothing large is mapped while writing; frame j is reassembled from its pieces when read."""

    def __init__(self, path, n, widths):
        self.mm = np.memmap(path, np.uint8, "r")
        self.n, self.widths = n, widths
        self.off = np.cumsum([0] + [n * fu.H_RAW * w for w in widths])

    def __getitem__(self, j):
        return np.concatenate([np.asarray(self.mm[o + j * fu.H_RAW * w: o + (j + 1) * fu.H_RAW * w]).reshape(fu.H_RAW, w)
                               for o, w in zip(self.off[:-1], self.widths)], axis=1)


def read_bands(tag, path, frames):
    import h5py
    out = os.path.join(fu.WORK, f"bands_pp_{tag}.u8")
    widths, t0 = [], time.time()
    with h5py.File(path, "r") as h, open(out, "wb") as fo:
        ds = h[[k for k in h.keys() if not k.startswith("#")][0]]
        for c0 in range(0, fu.W_RAW, 64):
            overlap_wait(f"read {tag}")
            fu.gate(200, f"pp_yield band read {tag}")
            c1 = min(fu.W_RAW, c0 + 64)
            band = ds[:, 0, c0:c1, :]  # (N, w, H); every chunk holds all frames of 1 column x 107 rows
            fo.write(np.ascontiguousarray(band[frames].transpose(0, 2, 1)).tobytes())
            widths.append(c1 - c0)
            del band
    fu.log(f"pp_yield read {tag}: {len(frames)} frames in {time.time() - t0:.0f} s")
    return out, widths, time.time() - t0


def main():
    fu.background_mode()
    bgm, bgs = ex.ref()
    man = pd.read_csv(os.path.join(fu.PO, "manifest.csv"))
    out, rows = {}, []
    for tag, fn in VIDS:
        t0 = time.time()
        path = os.path.join(D, fn)
        N = fu.n_frames(path)
        frames = ex.grid(N, K)
        comps = sorted(set(min(N - 1, f + 10) if f + 10 < N else max(0, f - 10) for f in frames))
        allf = sorted(set(frames) | set(comps))
        overlap_wait(f"read {tag}")
        bpath, widths, secs = read_bands(tag, path, allf)
        st = BandStack(bpath, len(allf), widths)
        paths = {"all": bpath}
        pos = {f: j for j, f in enumerate(allf)}
        cache = {}

        def get_sum(i):
            if i not in cache:
                if len(cache) >= 24:
                    del cache[next(iter(cache))]
                cache[i] = np.sum(np.stack(ex.H["demosaic"](np.asarray(st[pos[i]])), axis=0).astype(np.int64), axis=0)
            return cache[i]
        sel_parts, fst_parts, meas = [], [], {}
        for j in range(0, len(frames), 16):  # in chunks so the overlap and chain gates are checked often
            overlap_wait(f"detect {tag}")
            fu.gate(300, f"pp_yield detect {tag}")
            chunk = frames[j:j + 16]
            try:
                s, f_, m = ex.step6(get_sum, N, chunk, bgm, bgs)
                sel_parts.append(s)
                fst_parts.append(f_)
                meas.update(m)
            except ValueError:  # Step 6 raises when no detection of the chunk survives: redo frame by frame
                for f1 in chunk:
                    try:
                        s, f_, m = ex.step6(get_sum, N, [f1], bgm, bgs)
                        sel_parts.append(s)
                        fst_parts.append(f_)
                        meas.update(m)
                    except ValueError:  # nothing survived in this frame: all detections dropped by the debris check
                        regions, counts, _, _ = ex.region_info(get_sum(f1), bgm, bgs)
                        fst_parts.append(pd.DataFrame([dict(frame=f1, n_detections=len(regions),
                                                            n_debris_still_elevated=len(regions),
                                                            n_debris_low_displacement=0, n_kept_step6=0)]))
        sel = (pd.concat(sel_parts, ignore_index=True) if sel_parts else
               pd.DataFrame(columns=["frame", "x", "y", "w", "h", "pixel_count", "particle_id"]))
        old = man[man.source_file == fn]
        old_frames = np.unique(np.linspace(0, N - 1, min(ex.H["N_KEPT_FRAMES"], N)).round().astype(int)) if len(old) else []
        key = lambda d: set(zip(d.frame, d.x, d.y))
        reproduced = None
        if len(old):
            old_final, _ = ex.step7(sel[sel.frame.isin(old_frames)])
            o = old_final.sort_values(["frame", "particle_id"])
            m_ = old.sort_values(["frame", "particle_id"])
            reproduced = bool(len(o) == len(m_) and (m_[["frame", "particle_id", "x", "y"]].values ==
                                                     np.c_[o.frame, o.particle_id, o.x.astype(int), o.y.astype(int)]).all())
            unchanged, _ = ex.step7(sel)
            final, _ = ex.step7(sel, priority_xy=sorted(key(old_final)))
            lost = len(key(old_final) - key(unchanged))
            oid = {(f, x, y): int(p) for f, x, y, p in zip(old_final.frame, old_final.x, old_final.y, old_final.particle_id)}
        else:
            final, _ = ex.step7(sel)
            lost, oid = 0, {}
        final = final[final.frame != 0].reset_index(drop=True)
        new_i = 0
        for _, r in final.iterrows():
            kf = (r.frame, r.x, r.y)
            mm = meas[kf]
            if kf in oid:
                pid, pf, is_old = oid[kf], f"{fn[:-4]}_particle{oid[kf]}_frame{int(r.frame)}.npy", True
            else:
                pid, pf, is_old = 1000 + new_i, f"{fn[:-4]}_expparticle{new_i}_frame{int(r.frame)}.npy", False
                new_i += 1
            rows.append(dict(patch_file=pf, particle_id=pid, frame=int(r.frame), x=int(r.x), y=int(r.y), source_file=fn,
                             **{"class": "PP"}, old_patch=is_old, bbox_w_px=mm["bbox_w_px"], bbox_h_px=mm["bbox_h_px"],
                             region_px_dilated=mm["region_px_dilated"], region_px_predilation=mm["region_px_predilation"],
                             region_px_in_patch=mm["region_px_in_patch"], clipped=ex.clipped(int(r.x), int(r.y))))
        kept = pd.DataFrame([r for r in rows if r["source_file"] == fn])
        n_tracks = int(tr.tracks(kept).nunique()) if len(kept) else 0
        fst = pd.concat(fst_parts, ignore_index=True)
        out[tag] = dict(file=fn, n_frames=int(N), n_sampled=len(frames), detections=int(fst.n_detections.sum()),
                        debris_dropped=int((fst.n_debris_still_elevated + fst.n_debris_low_displacement).sum()),
                        step6_kept=int(len(sel)), kept_after_15px=int(len(kept)), old_patches=int(len(old) - (old.frame == 0).sum()),
                        old_lost_if_step7_unchanged=int(lost), old_rows_reproduced=reproduced, tracks=n_tracks,
                        edge_fraction=float(kept.clipped.mean()) if len(kept) else float("nan"),
                        read_seconds=secs, seconds=time.time() - t0)
        del st, cache, get_sum
        try:
            os.remove(paths["all"])
        except OSError:  # still mapped somewhere: leave the scratch file
            pass
        fu.log(f"pp_yield {tag}: {out[tag]}")
        json.dump(out, open(os.path.join(D, "results", "pp_yield_exp.json"), "w"), indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(D, "results", "pp_yield_detections_exp.csv"), index=False)
    qual = all(out[t]["tracks"] >= 50 for t, _ in VIDS)
    L = ["PP YIELD CHECK (overnight step 3; read-only, detection only, no patches saved)",
         "Rules of the PET and PS expansion: holo_pipeline_starter.py detection and thresholds, Step 6 debris check,",
         "Step 7 15 px recurrence rule (old 18-frame patches kept where they collide), frame 0 left out; up to 256",
         "sampled frames per recording (grid unique(round(linspace(0, n - 1, 256)))). Tracks: 20 px linkage of",
         "centroids in different sampled frames, connected components. Frame edge: the 64 x 64 crop leaves the frame.",
         f"{'recording':12s} {'frames':>6s} {'sampled':>7s} {'detections':>10s} {'debris drop':>11s} {'after 15 px':>11s} "
         f"{'tracks':>6s} {'edge frac':>9s} {'old kept':>8s} {'old lost if unchanged':>21s} {'old rows reproduced':>19s}"]
    for t, _ in VIDS:
        o = out[t]
        L.append(f"{t:12s} {o['n_frames']:>6d} {o['n_sampled']:>7d} {o['detections']:>10d} {o['debris_dropped']:>11d} "
                 f"{o['kept_after_15px']:>11d} {o['tracks']:>6d} {o['edge_fraction']:>9.3f} {o['old_patches']:>8d} "
                 f"{o['old_lost_if_step7_unchanged']:>21d} {str(o['old_rows_reproduced']):>19s}")
    L.append(f"Rule fixed before the result: PP QUALIFIES if each recording gives at least 50 tracks -> "
             f"{'PP QUALIFIES' if qual else 'PP DOES NOT QUALIFY'}")
    L.append("PP500FRAMES.mat: MATLAB 7.3 file created 29 Dec 2022 (the other recordings 10 Jan 2023), 500 frames.")
    open(os.path.join(D, "results", "report_PP_yield_exp.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(dict(out=out, qualifies=qual), open(os.path.join(D, "results", "pp_yield_exp.json"), "w"), indent=1)
    fu.log("pp_yield finished: " + ("PP QUALIFIES" if qual else "PP does not qualify"))


if __name__ == "__main__":
    main()
