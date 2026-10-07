"""
build_reconstructed_dataset.py

Builds the reconstructed patch data for BOTH real and background-only
patches at a caller-specified propagation distance z. Writes ONLY new
files - never touches pipeline_output/patches/, manifest.csv,
pipeline_output/patches_background/, manifest_background.csv,
patches_bgcorrected/, or existing results/.

Mirrors build_bgcorrected_dataset.py: one bulk contiguous read per video;
the expensive per-frame step (per-channel rolling-median background + full-
frame angular-spectrum propagation) is done ONCE per unique (video, frame)
and reused for every real and background patch sharing that frame.

Per (video, frame f):
  1. demosaic the raw frame -> I0, I45, I90, I135
  2. per-channel background I_c_bg(f): rolling MEDIAN over up to 90
     preceding raw frames (median-then-demosaic-once, uint8 median)
  3. field per channel = sqrt(I_c / I_c_bg(f)), zero phase, propagated by
     -z (FULL FRAME, never per-patch)
  4. outputs per patch (64x64 crop, half=32):
     - recon_s0 (1ch): mean of the four propagated intensities, the SAME
       combination the raw "intensity" mode uses
     - recon_prop (6ch): [I0p, I45p, I90p, I135p, A_bg, B_bg], from which
       recon_biref can later be built offline without re-reading video

Per video, A_bg/B_bg in the SAME domain as A/B (propagated, per-channel-
divided), computed ONCE per video rather than per frame to cut cost:
  - 30 frames evenly spaced over the whole video
  - per-channel video background = median of those 30 raw frames
  - each of the 30 frames divided by it per channel, propagated by -z,
    A(i)=(I0p-I90p)/(I0p+I90p), B(i)=(I135p-I45p)/(I45p+I135p)
  - A_bg, B_bg = per-pixel median over the 30 maps
  TRADE-OFF: a per-video map cannot follow drift within the video (the
  per-frame rolling version could). Measured, not assumed: the median of
  the first 15 vs last 15 frames (temporal drift) is compared with the
  median of odd vs even frames (same sample size, no temporal separation =
  noise baseline), at every patch position -> results/recon_abbg_drift.csv.
  Also note the target frames are divided by their ROLLING background
  I_c_bg(f) while the A_bg frames use the per-video background; the two
  differ only by background drift, which the same diagnostic bounds.

FRAME-0 EDGE CASE: no preceding frames -> no rolling background ->
amplitude sqrt(I/I_bg) undefined. Those patches are SKIPPED and counted.

IDEMPOTENT: each output has its own manifest; patches already listed are
skipped on a re-run.

Usage:
  python build_reconstructed_dataset.py --z 2.25
  python build_reconstructed_dataset.py --z 2.25 --max-groups 2    (dry run)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import reconstruction as recon
import reconstruction_validate as rv

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
PO = os.path.join(DATA_DIR, "pipeline_output")
REAL_MANIFEST_PATH = os.path.join(PO, "manifest.csv")
BG_MANIFEST_PATH = os.path.join(PO, "manifest_background.csv")


def output_paths(name, suffix):
    """(real_dir, real_manifest, bg_dir, bg_manifest) for output 's0' or 'prop'."""
    return (f"patches_recon_{name}{suffix}", f"manifest_recon_{name}{suffix}.csv",
            f"patches_background_recon_{name}{suffix}", f"manifest_background_recon_{name}{suffix}.csv")


HALF = 32
BG_WINDOW = rv.BG_WINDOW
N_AB_FRAMES = 30


def crop_patch(full, cx, cy, half=HALF):
    h, w = full.shape[:2]
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    return full[y0:y1, x0:x1, ...]


def load_existing(manifest_path):
    if not os.path.exists(manifest_path):
        return set(), []
    df = pd.read_csv(manifest_path)
    return set(df["patch_file"]), df.to_dict("records")


def n_frames_of(block, mode):
    return block.shape[-1] if mode == "scipy" else block.shape[0]


def propagate_channels(I_list, Ibg_list, H):
    out = []
    for I, Ibg in zip(I_list, Ibg_list):
        amp = np.sqrt(np.clip(I / (Ibg + 1e-6), 0, None)).astype(np.complex128)
        out.append(np.abs(recon.angular_spectrum_propagate(amp, None, H=H)) ** 2)
    return out


def a_b_maps(I0p, I45p, I90p, I135p):
    with np.errstate(invalid="ignore", divide="ignore"):
        D0, D45 = I0p + I90p, I45p + I135p
        A = np.where(D0 > 0, (I0p - I90p) / D0, np.nan)
        B = np.where(D45 > 0, (I135p - I45p) / D45, np.nan)
    return A.astype(np.float32), B.astype(np.float32)


def video_ab_background(block, mode, H):
    n = n_frames_of(block, mode)
    idx = np.unique(np.linspace(0, n - 1, N_AB_FRAMES).round().astype(int))
    raw_stack = np.stack([rv.get_raw_frame(block, mode, i) for i in idx], axis=0)
    Ibg_video = recon.demosaic(np.median(raw_stack, axis=0))
    del raw_stack
    A_list, B_list = [], []
    for i in idx:
        I = recon.demosaic(rv.get_raw_frame(block, mode, int(i)).astype(np.float64))
        A, B = a_b_maps(*propagate_channels(I, Ibg_video, H))
        A_list.append(A)
        B_list.append(B)
    A_st, B_st = np.stack(A_list), np.stack(B_list)
    k = len(idx) // 2
    maps = dict(
        A_bg=np.nanmedian(A_st, axis=0), B_bg=np.nanmedian(B_st, axis=0),
        dA_time=np.nanmedian(A_st[k:], axis=0) - np.nanmedian(A_st[:k], axis=0),
        dB_time=np.nanmedian(B_st[k:], axis=0) - np.nanmedian(B_st[:k], axis=0),
        dA_noise=np.nanmedian(A_st[1::2], axis=0) - np.nanmedian(A_st[0::2], axis=0),
        dB_noise=np.nanmedian(B_st[1::2], axis=0) - np.nanmedian(B_st[0::2], axis=0),
    )
    return maps, idx


def eta_equiv(p):
    return np.arcsin(np.clip(p, 0, 1)) / (np.pi / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=recon.DEFAULT_Z_M * 1e3, help="propagation distance in mm")
    ap.add_argument("--max-groups", type=int, default=None, help="dry run: stop after this many (video, frame) groups")
    ap.add_argument("--suffix", default="", help="appended to every output dir/manifest name, e.g. _z0")
    ap.add_argument("--s0-only", action="store_true",
                    help="write only recon_s0 (skips the per-video A_bg/B_bg maps and recon_prop)")
    args = ap.parse_args()
    z = args.z * 1e-3
    print(f"z = {args.z:.3f} mm, suffix '{args.suffix}', s0-only {args.s0_only}", flush=True)
    OUTPUTS = {name: output_paths(name, args.suffix) for name in (("s0",) if args.s0_only else ("s0", "prop"))}
    drift_csv = os.path.join(DATA_DIR, "results", f"recon_abbg_drift{args.suffix}.csv")

    real_manifest = pd.read_csv(REAL_MANIFEST_PATH)
    bg_manifest = pd.read_csv(BG_MANIFEST_PATH)

    state = {}
    for name, (rdir, rman, bdir, bman) in OUTPUTS.items():
        for d in (rdir, bdir):
            os.makedirs(os.path.join(PO, d), exist_ok=True)
        state[name] = dict(real=load_existing(os.path.join(PO, rman)), bg=load_existing(os.path.join(PO, bman)))

    drift_rows = []
    n_skip_real = n_skip_bg = n_groups = 0
    H = None
    stop = False

    for source_file in sorted(real_manifest["source_file"].unique()):
        if stop:
            break
        real_v = real_manifest[real_manifest["source_file"] == source_file]
        bg_v = bg_manifest[bg_manifest["source_file"] == source_file]
        needed = sorted(set(real_v["frame"]) | set(bg_v["frame"]))

        print(f"\n{source_file}: loading (bulk contiguous read)...", flush=True)
        block, mode = rv.load_video_block(os.path.join(DATA_DIR, source_file))
        if H is None:
            shape = recon.demosaic(rv.get_raw_frame(block, mode, 0))[0].shape
            H = recon.angular_spectrum_transfer(shape, -z)

        if "prop" in OUTPUTS:
            ab, ab_idx = video_ab_background(block, mode, H)
            p_time = np.hypot(ab["dA_time"], ab["dB_time"])
            p_noise = np.hypot(ab["dA_noise"], ab["dB_noise"])
            print(f"  per-video A_bg/B_bg from {len(ab_idx)} frames: mean A_bg={np.nanmean(ab['A_bg']):+.4f}, "
                  f"mean B_bg={np.nanmean(ab['B_bg']):+.4f}; drift |dAB| first-vs-second half "
                  f"{np.nanmean(p_time):.4f} vs odd-vs-even noise {np.nanmean(p_noise):.4f}", flush=True)
        print(f"  {len(needed)} unique frames ({len(real_v)} real, {len(bg_v)} background patches)", flush=True)

        for frame_idx in needed:
            if args.max_groups is not None and n_groups >= args.max_groups:
                stop = True
                break
            frame_idx = int(frame_idx)
            real_f = real_v[real_v["frame"] == frame_idx]
            bg_f = bg_v[bg_v["frame"] == frame_idx]

            Ibg = rv.channel_background_median(block, mode, frame_idx, window=BG_WINDOW)
            if Ibg is None:
                n_skip_real += len(real_f)
                n_skip_bg += len(bg_f)
                print(f"  frame {frame_idx}: SKIPPED (no preceding frames) - "
                      f"{len(real_f)} real + {len(bg_f)} background dropped", flush=True)
                n_groups += 1
                continue

            I = recon.demosaic(rv.get_raw_frame(block, mode, frame_idx).astype(np.float64))
            I0p, I45p, I90p, I135p = propagate_channels(I, Ibg, H)
            fulls = {"s0": ((I0p + I45p + I90p + I135p) / 4.0).astype(np.float32)[..., None]}
            if "prop" in OUTPUTS:
                fulls["prop"] = np.stack([I0p, I45p, I90p, I135p, ab["A_bg"], ab["B_bg"]], axis=-1).astype(np.float32)

            for kind, rows in (("real", real_f), ("bg", bg_f)):
                for _, row in rows.iterrows():
                    pf, x, y = row["patch_file"], int(row["x"]), int(row["y"])
                    for name, full in fulls.items():
                        existing, out_rows = state[name][kind]
                        if pf in existing:
                            continue
                        out_dir = OUTPUTS[name][0] if kind == "real" else OUTPUTS[name][2]
                        np.save(os.path.join(PO, out_dir, pf), crop_patch(full, x, y))
                        out_rows.append(row.to_dict())
                        existing.add(pf)
                    if "prop" in OUTPUTS:
                        drift_rows.append(dict(
                            source_file=source_file, patch_file=pf, kind=kind, cls=row["class"], frame=frame_idx,
                            drift_p_time=float(np.nanmean(crop_patch(p_time, x, y))),
                            drift_p_noise=float(np.nanmean(crop_patch(p_noise, x, y))),
                            drift_eta_equiv_time=float(np.nanmean(eta_equiv(crop_patch(p_time, x, y)))),
                        ))

            print(f"  frame {frame_idx}: built {len(real_f)} real + {len(bg_f)} background patches", flush=True)
            n_groups += 1

        del block

    for name, (rdir, rman, bdir, bman) in OUTPUTS.items():
        pd.DataFrame(state[name]["real"][1]).to_csv(os.path.join(PO, rman), index=False)
        pd.DataFrame(state[name]["bg"][1]).to_csv(os.path.join(PO, bman), index=False)
    if "prop" in OUTPUTS:
        pd.DataFrame(drift_rows).to_csv(drift_csv, index=False)

    print("\n=== Summary ===")
    print(f"groups processed: {n_groups}")
    for name in OUTPUTS:
        print(f"{name}: {len(state[name]['real'][1])} real, {len(state[name]['bg'][1])} background patches")
    print(f"frame-0 edge case skipped: {n_skip_real} real, {n_skip_bg} background patches")
    if "prop" in OUTPUTS:
        print(f"drift diagnostics -> {drift_csv}")


if __name__ == "__main__":
    main()
