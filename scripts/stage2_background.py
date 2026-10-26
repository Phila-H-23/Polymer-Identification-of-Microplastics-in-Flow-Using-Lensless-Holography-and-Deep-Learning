"""Stage 2 (item 41): how much of the background is subtracted? Read-only: no dataset, no training, new files only
(results/stage2_frame_NN.png, results/stage2_spectra.png, results/report_stage2_background.txt; working files in the
session scratchpad). The report is assembled in two steps: `python scripts/stage2_background.py` computes everything,
writes the sheets and the numbers (scratchpad); `python scripts/stage2_background.py report "<notes file>"` writes
results/report_stage2_background.txt from the numbers plus the notes on the sheets.

Frames (seeded rule, numpy default_rng(0), fixed before looking): one frame from each of PET1, PET2, PS1, PS2 plus two
more from two videos drawn without replacement; one of the six slots (drawn) gets a frame in 45..89 (preceding
window shorter than 90 frames), the others a frame in 90..554 (full 90 preceding frames); every frame has 45 frames
after it for the centred window. 4 of the 6 frames (drawn) get a sheet.
Region (same for every frame and method): channel rows 0..1023 (the full height: the channel images are 1024 rows,
so 1024 rows cannot avoid the top and bottom edge) x channel columns 96..1119 (mosaic columns 192..2239, the central
1024 columns of 1224 to within 4.5 px). Mask: every detection of the frame by the pipeline's own detect_particles
(holo_pipeline_starter.py via scripts/extract_PETPS_exp.py, same background_reference.npz) as its bounding box plus
32 channel px (220 um) on every side.
Channels: strided demosaic as in the pipeline (I0 [1::2,1::2], I45 [0::2,1::2], I90 [0::2,0::2], I135 [1::2,0::2]).
Noise floor: per-pixel temporal std (ddof 0) over the up to 90 preceding frames of frame / per-pixel median of those
frames (the z0 background), median over the unmasked region pixels; for the four-channel mean the same with the
four-channel mean of frame / median per frame.
Methods (relative residuals): (a) I / median of up to 90 preceding frames - 1 (current z0); (b) Fraser's: I - B + <B>
with B the per-pixel mean of the up to 90 preceding frames and <B> its mean over the whole frame, divided by <B>, - 1,
i.e. (I - B) / <B> (per channel <B> of that channel; for the four-channel mean the whole-mosaic <B>); (c) I / median
of frames f-45..f-1 and f+1..f+45 - 1; (d) (a) with the background multiplied by g = median(I) / median(background)
over the unmasked region, per frame and channel; (e) (c) with the gain of (d). Four-channel mean residual = mean over
channels of the per-channel ratio (the z0 recon_s0 value), - 1; for (b) mean over channels of (I - B), / <B>.
Readings fixed in the task: B1 (a) within 1.3 x the floor for the four-channel mean in every frame; B2 one and the
same alternative lowers the residual std of the four-channel mean by at least 15% relative to (a) in at least 5 of
the 6 frames; B3 otherwise (residual above 1.3 x the floor in at least one frame and no alternative reaches B2); the
strict wording (above 1.3 x in every frame and no alternative lowers it by 15% in any frame) is printed beside it.
"""
import os
import sys
import json
import time

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.NORMAL)  # stages have priority over the background jobs
import numpy as np  # noqa: E402
import h5py  # noqa: E402
import item41_status as st  # noqa: E402

WORK = (r"C:\Users\haany\AppData\Local\Temp\claude\c--Users-haany-OneDrive-research-project-dataset-10jan23"
        r"\075ab033-0168-4afa-a675-c38e4f73b481\scratchpad\item41_stage2")
os.makedirs(WORK, exist_ok=True)
KEY = "stage2"
H_RAW, W_RAW = 2048, 2448
MC0, MC1 = 192, 2240  # region, mosaic columns
CC0, CC1 = MC0 // 2, MC1 // 2  # region, channel columns 96..1119
SITES = {"I0": (1, 1), "I45": (0, 1), "I90": (0, 0), "I135": (1, 0)}
MARGIN = 32  # channel px = 220.8 um
METHODS = ["a", "b", "c", "d", "e"]
MNAME = {"a": "(a) current: / median of up to 90 preceding", "b": "(b) Fraser's: - mean of up to 90 preceding, + its mean",
         "c": "(c) / median of centred 45 + 45 (current excluded)", "d": "(d) (a) with per-frame per-channel gain",
         "e": "(e) (c) with per-frame per-channel gain"}
BANDS = [(0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.75)]


def choose_frames():
    rng = np.random.default_rng(0)
    videos = ["PET1", "PET2", "PS1", "PS2"]
    extra = sorted(int(i) for i in rng.choice(4, 2, replace=False))
    slots = videos + [videos[i] for i in extra]
    short = int(rng.integers(6))
    frames = []
    for k, v in enumerate(slots):
        while True:
            f = int(rng.integers(45, 90)) if k == short else int(rng.integers(90, 555))
            if (v, f) not in frames:
                break
        frames.append((v, f))
    sheet = sorted(int(i) for i in rng.choice(6, 4, replace=False))
    return frames, sheet


def compute(video, f):
    """Stats of one frame: region arrays at mosaic resolution and the full current frame for detection."""
    path = os.path.join(D, f"{video}_600.mat")
    lo, hi = max(0, f - 90), f + 46
    n_pre = f - lo
    cur = n_pre
    cen = list(range(n_pre - 45, n_pre)) + list(range(n_pre + 1, n_pre + 46))
    raw = np.empty((H_RAW, W_RAW), np.uint8)
    W = MC1 - MC0
    med_p, mean_p, med_c = (np.empty((H_RAW, W), np.float32) for _ in range(3))
    fl_mosaic = np.empty((H_RAW, W), np.float32)
    fl_mean = np.empty((H_RAW // 2, W // 2), np.float32)
    bsum = {k: 0.0 for k in SITES}
    bcnt = {k: 0 for k in SITES}
    t_stats = 0.0
    with h5py.File(path, "r") as h:
        ds = h[[k for k in h.keys() if not k.startswith("#")][0]]
        assert ds.shape[0] >= hi, (video, ds.shape)
        for c0 in range(0, W_RAW, 64):
            c1 = min(W_RAW, c0 + 64)
            band = np.ascontiguousarray(ds[lo:hi, 0, c0:c1, :].transpose(0, 2, 1))  # (n, 2048, w)
            raw[:, c0:c1] = band[cur]
            t = time.time()
            P = band[:n_pre]
            mp = P.mean(axis=0, dtype=np.float64)
            for k, (r, c) in SITES.items():
                v = mp[r::2, c::2]
                bsum[k] += float(v.sum())
                bcnt[k] += v.size
            if MC0 <= c0 < MC1:
                j0, j1 = c0 - MC0, c1 - MC0
                mdp = np.median(P, axis=0).astype(np.float32)
                med_p[:, j0:j1] = mdp
                mean_p[:, j0:j1] = mp
                med_c[:, j0:j1] = np.median(band[cen], axis=0)
                r = P.astype(np.float32) / np.maximum(mdp, 1.0)  # a zero median (dead pixel) would divide by 0
                fl_mosaic[:, j0:j1] = r.std(axis=0)
                m = (r[:, 0::2, 0::2] + r[:, 0::2, 1::2] + r[:, 1::2, 0::2] + r[:, 1::2, 1::2]) / 4.0
                fl_mean[:, j0 // 2:j1 // 2] = m.std(axis=0)
                del r, m, mdp
            t_stats += time.time() - t
            del band, P
    return dict(raw=raw, med_p=med_p, mean_p=mean_p, med_c=med_c, fl_mosaic=fl_mosaic, fl_mean=fl_mean,
                B_mean={k: bsum[k] / bcnt[k] for k in SITES}, B_mean_all=sum(bsum.values()) / sum(bcnt.values()),
                n_pre=n_pre, n_cen=len(cen), t_stats=t_stats)


def detections(raw):
    import extract_PETPS_exp as ex
    bg_mean, bg_std = ex.ref()
    frame = np.sum(np.stack(ex.H["demosaic"](raw), axis=0).astype(np.int64), axis=0)
    dets = ex.H["detect_particles"](frame, bg_mean, bg_std)
    keep = np.ones((H_RAW // 2, W_RAW // 2), bool)
    boxes = []
    for _, r in dets.iterrows():
        x0, x1 = int(np.floor(r["x"] - r["w"] / 2)) - MARGIN, int(np.ceil(r["x"] + r["w"] / 2)) + MARGIN
        y0, y1 = int(np.floor(r["y"] - r["h"] / 2)) - MARGIN, int(np.ceil(r["y"] + r["h"] / 2)) + MARGIN
        keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
        boxes.append([float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])])
    return keep[:, CC0:CC1], boxes


def channel(a, k, mosaic_cols_offset=True):
    r, c = SITES[k]
    return a[r::2, c::2]


def residuals(S, keep):
    """Relative residual images (channel resolution, region) per method, per channel and four-channel mean."""
    I = {k: channel(S["raw"][:, MC0:MC1].astype(np.float32), k) for k in SITES}
    mp = {k: np.maximum(channel(S["med_p"], k), 1.0) for k in SITES}  # guard against a zero median (dead pixel)
    mn = {k: channel(S["mean_p"], k) for k in SITES}
    mc = {k: np.maximum(channel(S["med_c"], k), 1.0) for k in SITES}
    gains = {}
    R = {m: {} for m in METHODS}
    for k in SITES:
        gd = float(np.median(I[k][keep]) / np.median(mp[k][keep]))
        ge = float(np.median(I[k][keep]) / np.median(mc[k][keep]))
        gains[k] = (gd, ge)
        R["a"][k] = I[k] / mp[k] - 1
        R["b"][k] = (I[k] - mn[k]) / S["B_mean"][k]
        R["c"][k] = I[k] / mc[k] - 1
        R["d"][k] = I[k] / (gd * mp[k]) - 1
        R["e"][k] = I[k] / (ge * mc[k]) - 1
    for m in METHODS:
        if m == "b":
            R[m]["mean"] = sum(I[k] - mn[k] for k in SITES) / 4.0 / S["B_mean_all"]
        else:
            R[m]["mean"] = sum(R[m][k] + 1 for k in SITES) / 4.0 - 1
    raw_rel = sum(I[k] / np.median(I[k][keep]) for k in SITES) / 4.0 - 1
    return R, raw_rel, gains


def spectrum(img, keep):
    v = np.where(keep, img - img[keep].mean(), 0.0)
    P = np.abs(np.fft.fft2(v)) ** 2 / keep.sum()
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    fr = np.hypot(fy, fx)
    nb = img.shape[0] // 2
    idx = np.minimum((fr * img.shape[0]).round().astype(int), nb)
    prof = np.bincount(idx.ravel(), P.ravel(), minlength=nb + 1) / np.maximum(np.bincount(idx.ravel(), minlength=nb + 1), 1)
    nz = fr > 0
    low = float(P[nz & (fr < 0.02)].sum() / P[nz].sum())
    bands = [float(P[nz & (fr >= a) & (fr < b)].sum() / P[nz].sum()) for a, b in BANDS]
    return np.arange(nb + 1) / img.shape[0], prof, low, bands


def jobs_in_run():
    s = st.load()
    return any(s.get(k, {}).get("in_run") for k in ("j1", "j2", "j3"))


def main():
    st.update(KEY, status="running", started=time.strftime("%Y-%m-%d %H:%M"), pid=os.getpid())
    frames, sheet = choose_frames()
    st.update(KEY, note="frames (seed 0): " + ", ".join(f"{i + 1:02d} {v} f{f}" for i, (v, f) in enumerate(frames))
              + f"; sheets for {', '.join(f'{i + 1:02d}' for i in sheet)}")
    out, units = [], []
    for i, (v, f) in enumerate(frames):
        busy0 = jobs_in_run()
        t0 = time.time()
        S = compute(v, f)
        t_read = time.time() - t0 - S["t_stats"]
        keep, boxes = detections(S["raw"])
        R, raw_rel, gains = residuals(S, keep)
        floor = {k: float(np.median(channel(S["fl_mosaic"], k)[keep])) for k in SITES}
        floor["mean"] = float(np.median(S["fl_mean"][keep]))
        res = {m: {k: float(R[m][k][keep].std()) for k in list(SITES) + ["mean"]} for m in METHODS}
        rec = dict(n=i + 1, video=v, frame=f, n_pre=S["n_pre"], n_cen=S["n_cen"], n_detections=len(boxes), boxes=boxes,
                   unmasked_fraction=float(keep.mean()), floor=floor, res=res, gains=gains, B_mean=S["B_mean"],
                   B_mean_all=S["B_mean_all"])
        np.savez_compressed(os.path.join(WORK, f"frame_{i + 1:02d}.npz"), keep=keep, raw_rel=raw_rel.astype(np.float32),
                            **{f"{m}_{k}": R[m][k].astype(np.float32) for m in METHODS for k in list(SITES) + ["mean"]})
        json.dump(rec, open(os.path.join(WORK, f"frame_{i + 1:02d}.json"), "w"), indent=1)
        out.append(rec)
        del S, R
        t_all = time.time() - t0
        units.append(dict(n=i + 1, t_compute=round((t_all - t_read) * 100.0 / (rec["n_pre"] + rec["n_cen"]), 1),
                          t_read=round(t_read, 1), job_in_run=busy0 or jobs_in_run(), unit="s per 100 window frames"))
        st.update(KEY, unit_times=units)
    best = min("bcde", key=lambda m: np.mean([r["res"][m]["mean"] for r in out]))
    spec = {}
    for r in out:
        a = np.load(os.path.join(WORK, f"frame_{r['n']:02d}.npz"))
        spec[r["n"]] = {m: spectrum(a[f"{m}_mean"].astype(np.float64), a["keep"]) for m in ("a", best)}
    files = sheets(out, sheet, best, spec)
    numbers(frames, sheet, out, best, spec)
    st.update(KEY, status="computed; notes on the sheets pending", files=files, best=best)


def sheets(out, sheet, best, spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    for i in sheet:
        r = out[i]
        a = np.load(os.path.join(WORK, f"frame_{r['n']:02d}.npz"))
        keep = a["keep"]
        fig = plt.figure(figsize=(20, 15.5), dpi=110)
        gs = fig.add_gridspec(3, 4)
        z = (slice(384, 640), slice(384, 640))
        top = [("raw region: four-channel mean / its median - 1", a["raw_rel"]), (MNAME["a"], a["a_mean"]),
               (f"best alternative {MNAME[best]}", a[f"{best}_mean"])]
        for j, (t, img) in enumerate(top):
            for row, sl in ((0, (slice(None), slice(None))), (1, z)):
                ax = fig.add_subplot(gs[row, j])
                ax.imshow(img[sl], cmap="gray", vmin=-0.05, vmax=0.05, interpolation="nearest")
                ax.contour((~keep[sl]).astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
                ax.set_title(("" if row == 0 else "centre 256 x 256 px: ") + t, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
        ax = fig.add_subplot(gs[0, 3])
        ax.imshow(keep, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"unmasked (white) {keep.mean():.1%}; {r['n_detections']} detections in the frame, box + 220 um", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, 3])
        for m, col in (("a", "k"), (best, "r")):
            fq, prof, low, _ = spec[r["n"]][m]
            ax.loglog(fq[1:], prof[1:], color=col, lw=1, label=f"({m}) power below 0.02 cyc/px {low:.1%}")
        ax.axvline(0.02, color="gray", ls=":")
        ax.set_xlabel("cycles per channel pixel")
        ax.set_title("azimuthally averaged power spectrum, four-channel mean residual", fontsize=8)
        ax.legend(fontsize=7)
        for j, k in enumerate(SITES):
            ax = fig.add_subplot(gs[2, j])
            ax.imshow(a[f"a_{k}"][z], cmap="gray", vmin=-0.05, vmax=0.05, interpolation="nearest")
            ax.contour((~keep[z]).astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
            ax.set_title(f"(a) residual, channel {k}, centre 256 x 256 px (std {r['res']['a'][k]:.4f}, floor {r['floor'][k]:.4f})",
                         fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Stage 2, frame {r['n']:02d}: {r['video']} frame {r['frame']} (preceding window {r['n_pre']} frames); region "
                     f"channel rows 0-1023, columns 96-1119 (6.9 um px); every image at -5% to +5%; cyan = masked "
                     f"(detection box + 220 um)", fontsize=10)
        fn = f"results/stage2_frame_{r['n']:02d}.png"
        if os.path.exists(fn):
            raise FileExistsError(fn)
        fig.savefig(fn)
        plt.close(fig)
        files.append(fn)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=100)
    for ax, r in zip(axes.ravel(), out):
        for m, col in (("a", "k"), (best, "r")):
            fq, prof, low, _ = spec[r["n"]][m]
            ax.loglog(fq[1:], prof[1:], color=col, lw=1, label=f"({m}) below 0.02: {low:.1%}")
        ax.axvline(0.02, color="gray", ls=":")
        ax.set_title(f"{r['n']:02d} {r['video']} f{r['frame']}", fontsize=9)
        ax.set_xlabel("cycles per channel pixel")
        ax.legend(fontsize=7)
    fig.suptitle(f"Stage 2: power spectra of the masked four-channel mean residual, (a) and best alternative ({best})")
    fig.tight_layout()
    fn = "results/stage2_spectra.png"
    if os.path.exists(fn):
        raise FileExistsError(fn)
    fig.savefig(fn)
    plt.close(fig)
    return files + [fn]


def numbers(frames, sheet, out, best, spec):
    L = ["STAGE 2: HOW MUCH OF THE BACKGROUND IS SUBTRACTED? (PET vs PS videos only; read-only)", ""]
    L += [x.rstrip() for x in __doc__.strip().split("\n")[6:]]
    L += ["", "FRAMES: " + "; ".join(f"{r['n']:02d} {r['video']} frame {r['frame']} (preceding window {r['n_pre']}, "
                                     f"{r['n_detections']} detections, unmasked {r['unmasked_fraction']:.1%})" for r in out),
          f"Sheets: frames {', '.join(f'{i + 1:02d}' for i in sheet)} -> results/stage2_frame_NN.png; spectra of all six: "
          "results/stage2_spectra.png", ""]
    cols = list(SITES) + ["mean"]
    L.append("NOISE FLOOR (median over unmasked pixels of the per-pixel temporal std / per-pixel median, preceding window):")
    L.append(f"{'frame':14s} " + " ".join(f"{c:>8s}" for c in cols))
    for r in out:
        L.append(f"{r['n']:02d} {r['video']:4s} f{r['frame']:<5d} " + " ".join(f"{r['floor'][c]:8.4f}" for c in cols))
    L += ["", "RESIDUAL SPATIAL STD (unmasked region) and, in brackets, its ratio to the floor:"]
    for m in METHODS:
        L.append(MNAME[m])
        for r in out:
            L.append(f"  {r['n']:02d} {r['video']:4s} f{r['frame']:<5d} " + " ".join(
                f"{r['res'][m][c]:.4f} ({r['res'][m][c] / r['floor'][c]:4.2f})" for c in cols))
    L += ["", "CHANGE OF THE FOUR-CHANNEL MEAN RESIDUAL STD AGAINST (a) (negative = lower):",
          f"{'frame':14s} " + " ".join(f"{m:>8s}" for m in "bcde")]
    for r in out:
        L.append(f"{r['n']:02d} {r['video']:4s} f{r['frame']:<5d} " + " ".join(
            f"{r['res'][m]['mean'] / r['res']['a']['mean'] - 1:+8.1%}" for m in "bcde"))
    L += ["", "Gains of (d) and (e) per channel (median of frame / median of background over the unmasked region):"]
    for r in out:
        L.append(f"  {r['n']:02d}: " + ", ".join(f"{k} {g[0]:.4f} / {g[1]:.4f}" for k, g in r["gains"].items()))
    L += ["", f"BEST ALTERNATIVE (lowest four-channel mean residual std averaged over the six frames): {best}, {MNAME[best]}",
          "POWER SPECTRUM of the masked four-channel mean residual (masked pixels set to 0 after removing the unmasked mean,",
          "no window, |FFT|^2 / unmasked pixel count, DC excluded); share of the power per radial frequency band (cycles",
          "per channel pixel; the last band reaches the corners at 0.71):",
          f"{'frame':14s} {'method':6s} {'< 0.02':>7s} " + " ".join(f"{a:>4.2f}-{b:<4.2f}" for a, b in BANDS)]
    for r in out:
        for m in ("a", best):
            _, _, low, bands = spec[r["n"]][m]
            L.append(f"{r['n']:02d} {r['video']:4s} f{r['frame']:<5d} ({m})    {low:7.1%} " + " ".join(f"{b:9.1%}" for b in bands))
    ratio_a = [r["res"]["a"]["mean"] / r["floor"]["mean"] for r in out]
    b1 = all(x <= 1.3 for x in ratio_a)
    count = {m: sum(r["res"][m]["mean"] <= 0.85 * r["res"]["a"]["mean"] for r in out) for m in "bcde"}
    b2 = any(c >= 5 for c in count.values())
    b3 = (not b1) and (not b2)
    strict = all(x > 1.3 for x in ratio_a) and all(r["res"][m]["mean"] > 0.85 * r["res"]["a"]["mean"] for r in out for m in "bcde")
    L += ["", "READINGS (fixed in the task; four-channel mean):",
          f"(a) residual / floor per frame: {', '.join(f'{x:.2f}' for x in ratio_a)}",
          f"B1 (a) within 1.3 x the floor in every frame: {b1}",
          f"frames where an alternative lowers the residual std by at least 15%: {', '.join(f'({m}) {c}' for m, c in count.items())}",
          f"B2 one alternative does so in at least 5 of 6 frames: {b2}",
          f"B3 (neither B1 nor B2): {b3}; strict wording (above 1.3 x in every frame and no alternative lowers it by 15% in any "
          f"frame): {strict}"]
    reading = " and ".join(x for x, ok in (("B1", b1), ("B2", b2), ("B3", b3)) if ok)
    L.append(f"Reading: {reading}")
    json.dump(dict(lines=L, reading=reading, best=best), open(os.path.join(WORK, "numbers.json"), "w"), indent=1)
    st.update(KEY, reading=reading)


def final_report(notes_path):
    n = json.load(open(os.path.join(WORK, "numbers.json")))
    notes = open(notes_path, encoding="utf-8").read().strip().splitlines()
    text = "\n".join(n["lines"] + ["", "WHAT REMAINS ON THE SHEETS (looked at by eye; numbers above are the readings):"] + notes)
    assert "\u2014" not in text and "significant" not in text.lower()
    out = "results/report_stage2_background.txt"
    if os.path.exists(out):
        raise FileExistsError(out)
    open(out, "w", encoding="utf-8").write(text + "\n")
    s = st.load().get(KEY, {})
    st.update(KEY, status="done", finished=time.strftime("%Y-%m-%d %H:%M"), files=s.get("files", []) + [out])


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "report":
        final_report(sys.argv[2])
    else:
        main()
