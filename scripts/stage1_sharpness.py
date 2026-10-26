"""Stage 1 (item 41): reconstruct 10 PET and PS particles and look at what could make them sharper. Read-only: no
dataset, no training, new files only (results/stage1_particle_NN.png, results/stage1_overview.png,
results/report_stage1_sharpness.txt; working files in the session scratchpad).

Particles (rule fixed before looking): from the 561 PET and PS patches of the common 849 set (the as-built recon_s0
PET vs PS runs), z0 = the stored patches_recon_s0_z0 patch, detected region = the pre-dilation region mask of the
detection (results/expanded_PETPS_exp/part1_region_masks.npz; the same detection as the original build). Per class:
the 2 patches with the lowest minimum of z0 inside the region, then the 2 whose minimum is nearest the class median,
at most 2 per video (otherwise the next in order; ties broken by patch name); then PET3 #35 (f211) and PS2 #80 (f352).
Variants at z = 2.25 mm: A as built (the stored recon_s0 patch, V1); B = V3 (per-channel bilinear demosaic to
3.45 um, division by the per-pixel median of up to 90 preceding raw frames, reflect padding to 2x); C = V2 (mosaic at
3.45 um, I - B + mean(B) with B the per-pixel mean of up to 90 preceding frames and mean(B) its mean over the whole
mosaic, reflect padding to 2x, four channels split by stride, mean of the four). B and C are propagated on a
1024 x 1024 px crop (3.45 um) around the particle, reflect padded to 2048 x 2048; checked once against the whole
frame. D: autofocus on B and C, z = 1.50 to 3.00 mm in 25 um steps.
"""
import os
import sys
import json
import time
from collections import Counter

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import recon_check as rc  # noqa: E402  (helpers; its import tries idle priority, set to normal below)
import recon_check_followup as fu  # noqa: E402
import prio41  # noqa: E402
prio41.set_priority(prio41.NORMAL)  # stages have priority over the background jobs
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import h5py  # noqa: E402
import scipy.fft as sfft  # noqa: E402
from scipy import ndimage  # noqa: E402
from reconstruction_zsweep2 import laplacian_variance_amp, tamura_coefficient  # noqa: E402
import item41_status as st  # noqa: E402

PO, RES = "pipeline_output", "results"
# Corrected run. The first run (PID 118000, started 10:54, scratch folder item41_stage1) took the real part of the
# propagated field instead of its modulus (region() cast complex to float); it could not be stopped (permission
# denied), so this run keeps its own scratch folder and state key and never writes over that run's files.
WORK = os.path.join(os.path.dirname(rc.WORK), "item41_stage1_fixed")
os.makedirs(WORK, exist_ok=True)
KEY = "stage1c"
FIRST_RUN_PID = 118000
Z_AF = np.round(np.arange(1.500, 3.0001, 0.025), 3)  # mm, 61 values
Z0 = 2.25
H_RAW, W_RAW, CROP = 2048, 2448, 1024
EXTRA = ["PET3_600_particle35_frame211.npy", "PS2_600_particle80_frame352.npy"]


# ------------------------------------------------------------------------------------------------ selection
def select():
    m = pd.read_csv(f"{PO}/manifest_recon_s0_z0.csv")
    m = m[m["class"].isin(["PET", "PS"])]
    assert len(m) == 561
    masks = np.load("results/expanded_PETPS_exp/part1_region_masks.npz")
    big = pd.read_csv(f"{PO}/manifest_PETPS_exp.csv").set_index("patch_file")
    rows = []
    for _, r in m.iterrows():
        z = np.load(f"{PO}/patches_recon_s0_z0/{r.patch_file}")[..., 0]
        mk = masks[r.patch_file[:-4]]
        assert mk.shape == z.shape
        A = float(big.loc[r.patch_file, "region_px_predilation"])
        rows.append(dict(patch_file=r.patch_file, cls=r["class"], video=r.source_file.replace("_600.mat", ""), frame=int(r.frame),
                         x=int(r.x), y=int(r.y), min_z0=float(z[mk].min()) if mk.any() else np.nan, region_px=A,
                         eq_diam_um=float(2 * np.sqrt(A / np.pi) * 6.9), clipped=bool(big.loc[r.patch_file, "clipped"])))
    df = pd.DataFrame(rows).sort_values("patch_file").reset_index(drop=True)
    chosen, med = [], {}
    for cls in ("PET", "PS"):
        d = df[(df.cls == cls) & df.min_z0.notna()]
        per_video = Counter()

        def take(order, k, why):
            n = 0
            for _, r in order.iterrows():
                if n == k:
                    break
                if r.patch_file in [c["patch_file"] for c in chosen] or per_video[r.video] >= 2:
                    continue
                chosen.append({**r.to_dict(), "why": why})
                per_video[r.video] += 1
                n += 1
        take(d.sort_values(["min_z0", "patch_file"]), 2, "darkest")
        med[cls] = float(d.min_z0.median())
        take(d.assign(dist=(d.min_z0 - med[cls]).abs()).sort_values(["dist", "patch_file"]), 2, "nearest the class median")
    for pf in EXTRA:
        if pf not in [c["patch_file"] for c in chosen]:
            chosen.append({**df[df.patch_file == pf].iloc[0].to_dict(), "why": "added (earlier check)"})
    for i, c in enumerate(chosen, start=1):
        c["n"] = i
    return chosen, med


# ------------------------------------------------------------------------------------------------ frames
def window_stats(video, f):
    """Current raw frame, per-pixel mean and median of the up to 90 preceding raw frames (mosaic, float32).
    HDF5: read in 64-column bands of the needed frames (each chunk holds all frames of 1 column x 107 rows), no
    whole video in memory. MAT v5 (PET4): one streamed zlib pass, frames written to a scratch file."""
    path = os.path.join(D, f"{video}_600.mat")
    lo = max(0, f - 90)
    raw = np.empty((H_RAW, W_RAW), np.float32)
    bmean, bmed = np.empty_like(raw), np.empty_like(raw)
    if open(path, "rb").read(116).decode("latin-1").startswith("MATLAB 7.3"):
        with h5py.File(path, "r") as h:
            ds = h[[k for k in h.keys() if not k.startswith("#")][0]]
            for c0 in range(0, W_RAW, 64):
                c1 = min(W_RAW, c0 + 64)
                band = ds[lo:f + 1, 0, c0:c1, :]  # (n + 1, w, H)
                raw[:, c0:c1] = band[-1].T
                bmean[:, c0:c1] = band[:-1].mean(axis=0, dtype=np.float64).T
                bmed[:, c0:c1] = np.median(band[:-1], axis=0).T
                del band
    else:
        tmp = os.path.join(WORK, f"frames_{video}_{f}.u8")
        with open(tmp, "wb") as fo:
            fu.mat5_stream(path, list(range(lo, f + 1)), lambda k, fr: fo.write(np.ascontiguousarray(fr).tobytes()))
        mm = np.memmap(tmp, np.uint8, "r", shape=(f + 1 - lo, H_RAW, W_RAW))
        raw[:] = mm[-1]
        for r0 in range(0, H_RAW, 256):
            blk = np.asarray(mm[:-1, r0:r0 + 256])
            bmean[r0:r0 + 256] = blk.mean(axis=0, dtype=np.float64)
            bmed[r0:r0 + 256] = np.median(blk, axis=0)
        del mm
        try:
            os.remove(tmp)
        except OSError:
            pass
    return raw, bmean, bmed, f - lo


def full_H(shape, z_mm, dx):
    return np.concatenate([rc.transfer_rows(shape, -z_mm * 1e-3, dx, r, min(shape[0], r + 256)) for r in range(0, shape[0], 256)])


def region(a, yc, xc, half=64):
    out = np.full((2 * half, 2 * half), np.nan, np.complex128 if np.iscomplexobj(a) else np.float64)
    y0, y1, x0, x1 = max(0, yc - half), min(a.shape[0], yc + half), max(0, xc - half), min(a.shape[1], xc + half)
    out[y0 - (yc - half):y1 - (yc - half), x0 - (xc - half):x1 - (xc - half)] = a[y0:y1, x0:x1]
    return out


def mean4(reg):
    ch = [reg[r::2, c::2] for (r, c) in rc.SITES.values()]
    return (ch[0] + ch[1] + ch[2] + ch[3]) / 4.0


def clean(img):
    img = img[:, np.isfinite(img).all(axis=0)]
    return img[np.isfinite(img).all(axis=1)]


def crit(intensity):
    """Criteria on the amplitude sqrt(I) of the region (in-frame pixels only)."""
    a = np.sqrt(np.clip(clean(intensity), 0, None))
    g = np.hypot(ndimage.sobel(a, axis=1, mode="reflect"), ndimage.sobel(a, axis=0, mode="reflect"))
    return dict(tog=float(np.sqrt(g.std() / g.mean())), lap=float(laplacian_variance_amp(a)), tam=float(tamura_coefficient(a)))


# ------------------------------------------------------------------------------------------------ one particle
def process(p, verify=False):
    t0 = time.time()
    raw, bmean, bmed, nbg = window_stats(p["video"], p["frame"])
    t_read = time.time() - t0
    t1 = time.time()
    mB = float(bmean.mean())
    Y, X = 2 * p["y"], 2 * p["x"]
    r0, c0 = int(np.clip(Y - CROP // 2, 0, H_RAW - CROP)), int(np.clip(X - CROP // 2, 0, W_RAW - CROP))
    yc, xc = Y - r0, X - c0
    cr = lambda a: a[r0:r0 + CROP, c0:c0 + CROP]
    FB = []
    for (r, cc) in rc.SITES.values():
        Ic, Bc = rc.bilinear(cr(raw), r, cc), rc.bilinear(cr(bmed), r, cc)
        FB.append(rc.spectrum(np.sqrt(np.clip(Ic / (Bc + 1e-6), 0, None)).astype(np.float32), True))
    FC = rc.spectrum(np.sqrt(np.maximum(cr(raw) - cr(bmean) + mB, 0)).astype(np.float32), True)
    pad = CROP // 2
    Bimg, Cimg, crits = {}, {}, {"B": [], "C": []}
    for z in Z_AF:
        Hz = full_H(FC.shape, z, rc.DX3)
        acc = np.zeros((128, 128))
        for F in FB:
            u = sfft.ifft2(F * Hz, workers=1)[pad:pad + CROP, pad:pad + CROP]
            acc += np.abs(region(u, yc, xc)) ** 2 / 4.0
            del u
        u = sfft.ifft2(FC * Hz, workers=1)[pad:pad + CROP, pad:pad + CROP]
        regC = np.abs(region(u, yc, xc)) ** 2 / mB
        del u, Hz
        Bimg[z], Cimg[z] = acc.astype(np.float32), mean4(regC).astype(np.float32)
        if abs(z - Z0) < 1e-9:
            Cimg["fine_225"] = regC.astype(np.float32)
        crits["B"].append(crit(acc))
        crits["C"].append(crit(Cimg[z]))
    del FB, FC
    t_comp = time.time() - t1
    out = dict(sel=p, n_bg_frames=nbg, crop_origin=[r0, c0], mean_B=mB, t_read=t_read, t_compute=t_comp, crits=crits)
    if verify:  # whole frame at z = 2.25 mm against the crop, central 128 x 128 px (3.45 um)
        tv = time.time()
        accW = np.zeros((128, 128))
        for (r, cc) in rc.SITES.values():
            Ic, Bc = rc.bilinear(raw, r, cc), rc.bilinear(bmed, r, cc)
            F = rc.spectrum(np.sqrt(np.clip(Ic / (Bc + 1e-6), 0, None)).astype(np.float32), True)
            u = rc.propagate_from(F, -Z0 * 1e-3, rc.DX3, (H_RAW, W_RAW), True)
            accW += np.abs(region(u, Y, X)) ** 2 / 4.0
            del F, u, Ic, Bc
        F = rc.spectrum(np.sqrt(np.maximum(raw - bmean + mB, 0)).astype(np.float32), True)
        u = rc.propagate_from(F, -Z0 * 1e-3, rc.DX3, (H_RAW, W_RAW), True)
        regCW = np.abs(region(u, Y, X)) ** 2 / mB
        del F, u
        ok = np.isfinite(accW) & np.isfinite(Bimg[Z0])
        out["verify"] = dict(B_max_abs_diff=float(np.abs(accW - Bimg[Z0])[ok].max()),
                             C_max_abs_diff=float(np.abs(regCW - Cimg["fine_225"])[np.isfinite(regCW)].max()),
                             B_value_range=[float(np.nanmin(accW)), float(np.nanmax(accW))], seconds=time.time() - tv)
        json.dump(out["verify"], open(os.path.join(WORK, "crop_check.json"), "w"), indent=1)
        if max(out["verify"]["B_max_abs_diff"], out["verify"]["C_max_abs_diff"]) > 1e-3:
            st.update(KEY, status="stopped", note=f"crop check failed: {out['verify']}")
            raise RuntimeError(f"crop check failed: {out['verify']}")
    A = np.load(f"{PO}/patches_recon_s0/{p['patch_file']}")[..., 0]
    np.savez_compressed(os.path.join(WORK, f"particle_{p['n']:02d}.npz"), A=rc.place(A, p["x"], p["y"], 32),
                        **{f"B_{z:.3f}": Bimg[z] for z in Z_AF}, **{f"C_{z:.3f}": Cimg[z] for z in Z_AF})
    json.dump(out, open(os.path.join(WORK, f"particle_{p['n']:02d}.json"), "w"), indent=1, default=float)
    return out


# ------------------------------------------------------------------------------------------------ sheets and report
def best(c):
    v = [x["tog"] for x in c]
    i = int(np.argmax(v))
    return Z_AF[i], v[i], v[int(np.argmin(np.abs(Z_AF - Z0)))]


def sheets(chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gray = matplotlib.colormaps["gray"].copy()
    gray.set_bad("#9ecae1")
    ext = [-220.8, 220.8, 220.8, -220.8]
    files, ov = [], []
    for p in chosen:
        n = p["n"]
        a = np.load(os.path.join(WORK, f"particle_{n:02d}.npz"))
        o = json.load(open(os.path.join(WORK, f"particle_{n:02d}.json")))
        zb, zc = best(o["crits"]["B"])[0], best(o["crits"]["C"])[0]
        panels = [("A as built (V1), z 2.25 mm", a["A"]), ("B (V3), z 2.25 mm", a[f"B_{Z0:.3f}"]),
                  ("C (V2), z 2.25 mm", a[f"C_{Z0:.3f}"]), (f"B at its best z {zb:.3f} mm", a[f"B_{zb:.3f}"]),
                  (f"C at its best z {zc:.3f} mm", a[f"C_{zc:.3f}"])]
        fig = plt.figure(figsize=(17, 11.5), dpi=100)
        gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.8])
        for j, (t, img) in enumerate(panels):
            for row, (lo, hi, lab) in enumerate(((0.5, 1.5, "display 0.5 to 1.5"), (None, None, "1st to 99th percentile"))):
                ax = fig.add_subplot(gs[row, j])
                if lo is None:
                    lo, hi = np.nanpercentile(img, [1, 99])
                ax.imshow(img, cmap=gray, vmin=lo, vmax=hi, interpolation="nearest", extent=ext)
                ax.plot([-200, -150], [200, 200], color="yellow", lw=3)
                ax.text(-200, 185, "50 um", color="yellow", fontsize=8)
                ax.set_title(f"{t}\n{lab}" + ("" if row == 0 else f" ({lo:.3f} to {hi:.3f})"), fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
        for j, v in enumerate(("B", "C")):
            ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
            tog = [x["tog"] for x in o["crits"][v]]
            ax.plot(Z_AF, tog, "-o", ms=2.5)
            zbest = best(o["crits"][v])[0]
            ax.axvline(zbest, color="red", ls="--", label=f"best z {zbest:.3f} mm")
            ax.axvline(Z0, color="gray", ls=":", label="2.25 mm")
            ax.set_xlabel("z (mm)")
            ax.set_ylabel("Tamura of gradient")
            ax.set_title(f"{v}: edge sparsity (Tamura of the gradient magnitude of the amplitude)", fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle(f"Stage 1, particle {n:02d}: {p['patch_file'][:-4]} ({p['cls']}, {p['video']}, frame {p['frame']}, centre "
                     f"x={p['x']} y={p['y']} px, {p['why']}{', frame edge' if p['clipped'] else ''}); every panel 441.6 um "
                     f"square; blue = outside the sensor", fontsize=10)
        fn = new_file(f"results/stage1_particle_{n:02d}{SFX}.png")
        fig.savefig(fn)
        plt.close(fig)
        files.append(fn)
        ov.append((p, a["A"], a[f"B_{zb:.3f}"], zb))
    fig, axes = plt.subplots(len(ov), 4, figsize=(9.5, 2.25 * len(ov)), dpi=100)
    for i, (p, A, Bb, zb) in enumerate(ov):
        for j, (img, t, fixed) in enumerate(((A, "A, 2.25 mm", True), (Bb, f"B, best z {zb:.3f}", True),
                                             (A, "A, 2.25 mm", False), (Bb, f"B, best z {zb:.3f}", False))):
            ax = axes[i, j]
            lo, hi = (0.5, 1.5) if fixed else np.nanpercentile(img, [1, 99])
            ax.imshow(img, cmap=gray, vmin=lo, vmax=hi, interpolation="nearest", extent=ext)
            ax.plot([-200, -150], [200, 200], color="yellow", lw=2)
            ax.set_title(f"{p['n']:02d} {p['cls']} {p['video']}: {t}\n" + ("0.5 to 1.5" if fixed else "1st to 99th pct"), fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Stage 1 overview: every particle as built (A, 2.25 mm) and as B at its own best z; scale bar 50 um", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fn = new_file(f"results/stage1_overview{SFX}.png")
    fig.savefig(fn)
    plt.close(fig)
    return files + [fn]


def report(chosen, med, results):
    L = ["STAGE 1: RECONSTRUCT A FEW PARTICLES AND SEE WHAT COULD MAKE THEM SHARPER (PET vs PS only; read-only)", ""]
    if SFX:
        L += ["CORRECTED RUN. A first run (started 10:54) squared the real part of the propagated field instead of its",
              "modulus, so its files results/stage1_particle_NN.png, results/stage1_overview.png and",
              "results/report_stage1_sharpness.txt are wrong and should be ignored. That run could not be stopped and its",
              "files were not overwritten (new files only); the correct results are the files ending in _corrected.", ""]
    L += ["PARTICLES (rule fixed before looking): 561 PET and PS patches of the common 849 set; minimum of the stored z0 patch",
         "inside the pre-dilation detected region; per class the 2 darkest and the 2 nearest the class median, at most 2 per",
         f"video, ties by patch name; then PET3 #35 (f211) and PS2 #80 (f352). Class medians of the minimum: PET {med['PET']:.4f}, "
         f"PS {med['PS']:.4f}. Equivalent diameter = 2 sqrt(region area / pi) x 6.9 um (pre-dilation region; a detection",
         "size, not a particle size).",
         f"{'NN':>2s} {'patch id':32s} {'class':5s} {'video':5s} {'frame':>5s} {'x':>5s} {'y':>5s} {'min z0':>7s} {'eq diam um':>10s} "
         f"{'edge':5s} why"]
    for p in chosen:
        L.append(f"{p['n']:>2d} {p['patch_file'][:-4]:32s} {p['cls']:5s} {p['video']:5s} {p['frame']:>5d} {p['x']:>5d} {p['y']:>5d} "
                 f"{p['min_z0']:7.4f} {p['eq_diam_um']:10.1f} {'yes' if p['clipped'] else 'no':5s} {p['why']}")
    edge = [p for p in chosen if p["clipped"]]
    L.append(f"Frame-edge patches among them: {', '.join(p['patch_file'][:-4] for p in edge) if edge else 'none'}.")
    v = next((r["verify"] for r in results if "verify" in r), None)
    if v:
        L += ["", f"CROP CHECK (particle {next(r['sel']['n'] for r in results if 'verify' in r):02d}): B and C at z = 2.25 mm "
                  f"propagated from the whole frame (reflect padded to 4096 x 4896) against the 1024 x 1024 crop (reflect padded "
                  f"to 2048 x 2048); largest absolute difference on the central 128 x 128 px (3.45 um), relative intensity units: "
                  f"B {v['B_max_abs_diff']:.2e}, C {v['C_max_abs_diff']:.2e} (limit 1e-3: "
                  f"{'met' if max(v['B_max_abs_diff'], v['C_max_abs_diff']) <= 1e-3 else 'NOT met'})."]
    L += ["", "WHAT WAS COMPUTED (D, autofocus): z = 1.50 to 3.00 mm in 25 um steps (61 depths), propagation by -z as in",
          "every earlier variant. B: the mean of the four propagated channel intensities on a 128 x 128 px region (3.45 um,",
          "441.6 um square) centred on the detection centre. C: its output image, the mean of the four stride-split channels",
          "(64 x 64 px at 6.9 um, the same 441.6 um square; C has no 3.45 um output of its own). Criterion (edge sparsity, our",
          "reading of Zhang et al. 2017): amplitude a = sqrt(intensity); gradient magnitude g = sqrt(sobel_x(a)^2 + sobel_y(a)^2)",
          "(scipy.ndimage.sobel, reflect edges); criterion = sqrt(std(g) / mean(g)) over the region, larger = sharper. Also, as",
          "before (scripts/reconstruction_zsweep2.py): Laplacian variance of a, and its 'Tamura' std(a) / sqrt(mean(a)) (note:",
          "this earlier function is not the same formula as Zhang's sqrt(std / mean)). Pixels outside the sensor are left out.",
          "rel diff = (criterion at best z - criterion at 2.25 mm) / criterion at 2.25 mm. Lap and Tam: value at the edge",
          "sparsity best z, value at 2.25 mm, and the z where each of them is largest.",
          f"{'NN':>2s} {'var':3s} {'best z':>6s} {'interior':8s} {'crit best':>9s} {'crit 2.25':>9s} {'rel diff':>8s} "
          f"{'Lap@best':>10s} {'Lap@2.25':>10s} {'Lap max z':>9s} {'Tam@best':>9s} {'Tam@2.25':>9s} {'Tam max z':>9s}"]
    counts = {"B": 0, "C": 0}
    i225 = int(np.argmin(np.abs(Z_AF - Z0)))
    for r in results:
        for var in ("B", "C"):
            c = r["crits"][var]
            zb, vb, v225 = best(c)
            ib = int(np.argmin(np.abs(Z_AF - zb)))
            interior = zb not in (Z_AF[0], Z_AF[-1])
            rel = (vb - v225) / v225
            zl = Z_AF[int(np.argmax([x["lap"] for x in c]))]
            zt = Z_AF[int(np.argmax([x["tam"] for x in c]))]
            counts[var] += int(interior and rel >= 0.10)
            L.append(f"{r['sel']['n']:>2d} {var:3s} {zb:6.3f} {'yes' if interior else 'no':8s} {vb:9.4f} {v225:9.4f} {rel:8.1%} "
                     f"{c[ib]['lap']:10.3e} {c[i225]['lap']:10.3e} {zl:9.3f} {c[ib]['tam']:9.4f} {c[i225]['tam']:9.4f} {zt:9.3f}")
    n = len(results)
    L += ["", "READINGS (fixed beforehand; numbers only, Phila judges the images): P1 = the autofocus has an interior optimum",
          "AND the criterion changes by at least 10% between the best z and 2.25 mm, for at least 6 of the 10 particles",
          "(both conditions counted per particle); P2 otherwise."]
    readings = {}
    for var in ("B", "C"):
        readings[var] = "P1" if counts[var] >= 6 else "P2"
        L.append(f"{var}: {counts[var]} of {n} particles meet both conditions -> {readings[var]}")
    t = [r["t_compute"] for r in results]
    L += ["", f"Timing: reading {sum(r['t_read'] for r in results) / 60:.1f} min in total, computing "
              f"{sum(t) / 60:.1f} min (per particle {min(t):.0f} to {max(t):.0f} s)."]
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    open(new_file(f"results/report_stage1_sharpness{SFX}.txt"), "w", encoding="utf-8").write(text + "\n")
    return readings


def jobs_in_run():
    s = st.load()
    return any(s.get(k, {}).get("in_run") for k in ("j1", "j2", "j3"))


def alive(pid):
    """Read-only query: is process pid still running?"""
    import ctypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = ctypes.c_void_p
    k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    h = k.OpenProcess(0x1000, 0, pid)
    if not h:
        return False
    code = ctypes.c_uint32()
    k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return code.value == 259


def requested(n_particles):
    return ([f"results/stage1_particle_{n:02d}.png" for n in range(1, n_particles + 1)]
            + ["results/stage1_overview.png", "results/report_stage1_sharpness.txt"])


def main():
    st.update(KEY, status="running", started=time.strftime("%Y-%m-%d %H:%M"))
    chosen, med = select()
    json.dump(dict(chosen=chosen, med=med), open(os.path.join(WORK, "selection.json"), "w"), indent=1, default=float)
    st.update(KEY, note="particles: " + ", ".join(f"{c['n']:02d} {c['patch_file'][:-4]}" for c in chosen))
    results, units = [], []
    vi = next(i for i, c in enumerate(chosen) if not c["clipped"])
    for i, p in enumerate(chosen):
        busy0 = jobs_in_run()
        r = process(p, verify=(i == vi))
        results.append(r)
        units.append(dict(n=p["n"], t_compute=round(r["t_compute"], 1), t_read=round(r["t_read"], 1),
                          job_in_run=busy0 or jobs_in_run()))
        st.update(KEY, unit_times=units)
    global SFX
    req = requested(len(chosen))
    SFX = "" if not alive(FIRST_RUN_PID) and not any(os.path.exists(p) for p in req) else "_corrected"
    files = sheets(chosen)
    readings = report(chosen, med, results)
    st.update(KEY, status="done", finished=time.strftime("%Y-%m-%d %H:%M"),
              reading=f"B {readings['B']}, C {readings['C']}", files=files + [f"results/report_stage1_sharpness{SFX}.txt"],
              names="requested names" if SFX == "" else "requested names with _corrected (first run alive or its files present)")


SFX = "_corrected"


def new_file(path):
    """New files only: refuse to write over anything."""
    if os.path.exists(path):
        raise FileExistsError(path)
    return path


if __name__ == "__main__":
    main()
