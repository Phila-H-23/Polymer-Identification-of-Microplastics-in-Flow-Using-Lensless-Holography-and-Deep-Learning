"""Item 44, step 1: depth-shifted z0 patches (no training; idle priority; new files only).
Per expanded-set real PET and PS patch and per polarisation channel: ratio r from
pipeline_output/patches_recon_z0pol4_PETPS_exp (stored x 255, so r = stored / 255), amplitude sqrt(r), zero phase,
reflect padding to 2x, FFT, then the combined propagation by z_k - 2.25 mm (the reconstruction to the object plane
by the code's convention, -2.25 mm, followed by the forward propagation by +z_k; recon_check.transfer_rows, 405 nm /
1.33, 6.9 um, complex128), inverse FFT, crop back, intensity |u|^2; the CNN input is the mean of the four channels'
intensities, one channel, float32, the stored shape (frame-edge patches keep theirs). Depths 2.000, 2.025, ..., 2.500
mm. Written to pipeline_output/patches_depth_z0_PETPS_exp/<z>/<patch_file>; the 2.250 folder holds a byte copy of
the stored z0 patch (pipeline_output/patches_recon_s0_z0_PETPS_exp), so unshifted inputs are exactly the z0 runs'.
Checks: (i) the computed z = 2.25 mm output against the stored z0 patch; (ii) the explicit chain on the padded grid
(FFT, x H(-2.25 mm), inverse FFT = object-plane field; FFT, x H(+z_k), inverse FFT, crop, |u|^2) against the combined
form on the first 20 patches in manifest order at every depth; (iii) finite outputs and min, max, mean per depth.
Sheet results/depth_shift_sheet.png. Results in results/logs_item44/depth_build.json."""
import os
import sys
import json
import time
import shutil

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import recon_check as rc  # noqa: E402  (transfer_rows, DX6)
prio41.set_priority(prio41.IDLE)

PO = "pipeline_output"
SRC = f"{PO}/patches_recon_z0pol4_PETPS_exp"
REF = f"{PO}/patches_recon_s0_z0_PETPS_exp"
OUT = f"{PO}/patches_depth_z0_PETPS_exp"
MAN = f"{PO}/manifest_recon_s0_z0_PETPS_exp.csv"
DEPTHS = [round(2.0 + 0.025 * k, 3) for k in range(21)]
Z0 = 2.25
_H = {}


def H(shape, dz_mm):
    k = (shape, round(dz_mm, 4))
    if k not in _H:
        _H[k] = rc.transfer_rows(shape, dz_mm * 1e-3, rc.DX6, 0, shape[0], dtype=np.complex128)
    return _H[k]


def spectrum(a):
    h, w = a.shape
    return np.fft.fft2(np.pad(a, ((h // 2, h // 2), (w // 2, w // 2)), mode="reflect").astype(np.complex128)), (h, w)


def crop(u, hw):
    h, w = hw
    return u[h // 2:h // 2 + h, w // 2:w // 2 + w]


def shifted(r4, z):
    """Combined form: mean over channels of |propagate(sqrt(r), z - 2.25 mm)|^2."""
    acc = 0.0
    for c in range(4):
        F, hw = spectrum(np.sqrt(np.clip(r4[..., c], 0, None)))
        acc = acc + np.abs(crop(np.fft.ifft2(F * H(F.shape, z - Z0)), hw)) ** 2
    return acc / 4.0


def explicit(r4, z):
    acc = 0.0
    for c in range(4):
        F, hw = spectrum(np.sqrt(np.clip(r4[..., c], 0, None)))
        obj = np.fft.ifft2(F * H(F.shape, -Z0))  # object plane (code's convention), on the padded grid
        acc = acc + np.abs(crop(np.fft.ifft2(np.fft.fft2(obj) * H(F.shape, z)), hw)) ** 2
    return acc / 4.0


def main():
    t0 = time.time()
    m = pd.read_csv(MAN)
    for z in DEPTHS:
        os.makedirs(f"{OUT}/{z:.3f}", exist_ok=False)
    chk1, chk2 = [], {z: 0.0 for z in DEPTHS}
    stats = {z: dict(mn=np.inf, mx=-np.inf, s=0.0, n=0, nonfinite=0) for z in DEPTHS}
    for i, p in enumerate(m.patch_file):
        r4 = np.load(f"{SRC}/{p}").astype(np.float64) / 255.0
        ref = np.load(f"{REF}/{p}")
        for z in DEPTHS:
            out = shifted(r4, z)
            st = stats[z]
            st["nonfinite"] += int((~np.isfinite(out)).sum())
            st["mn"], st["mx"] = min(st["mn"], float(out.min())), max(st["mx"], float(out.max()))
            st["s"] += float(out.sum())
            st["n"] += out.size
            if z == Z0:
                d = np.abs(out - ref[..., 0].astype(np.float64))
                k = np.unravel_index(np.argmax(d), d.shape)
                chk1.append((float(d.max()), float(d[k] / max(abs(ref[..., 0][k]), 1e-12)), float(ref[..., 0][k]), p))
                shutil.copyfile(f"{REF}/{p}", f"{OUT}/{z:.3f}/{p}")
            else:
                np.save(f"{OUT}/{z:.3f}/{p}", out.astype(np.float32)[..., None])
            if i < 20:
                chk2[z] = max(chk2[z], float(np.max(np.abs(explicit(r4, z) - out))))
    zr = [s for s in m.patch_file]
    ref_stats = dict(mn=min(float(np.load(f"{REF}/{p}").min()) for p in zr), mx=max(float(np.load(f"{REF}/{p}").max()) for p in zr))
    over = [x for x in chk1 if x[0] > 1e-6]
    res = dict(seconds=round(time.time() - t0, 1),
               check_i=dict(max_abs=max(x[0] for x in chk1), n_over_1e6=len(over), over=over, max_rel=max(x[1] for x in chk1)),
               check_ii={f"{z:.3f}": v for z, v in chk2.items()}, check_ii_max=max(chk2.values()),
               check_iii={f"{z:.3f}": dict(min=v["mn"], max=v["mx"], mean=v["s"] / v["n"], nonfinite=v["nonfinite"]) for z, v in stats.items()},
               ref_min=ref_stats["mn"], ref_max=ref_stats["mx"])
    os.makedirs("results/logs_item44", exist_ok=True)
    json.dump(res, open("results/logs_item44/depth_build.json", "w"), indent=1)
    sheet(m)
    print(json.dumps({k: v for k, v in res.items() if k != "check_iii"}, indent=1))


def sheet(m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for c in ("PET", "PS"):
        sub = m[m["class"] == c].reset_index(drop=True)
        rows += [(c, k, sub.patch_file[k]) for k in (0, 100, 200)]
    cols = [("unshifted (2.250 mm)", 2.25), ("2.000 mm", 2.0), ("2.125 mm", 2.125), ("2.375 mm", 2.375), ("2.500 mm", 2.5)]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(12, 2.6 * len(rows)), dpi=90)
    for i, (c, k, p) in enumerate(rows):
        imgs = [np.load(f"{OUT}/{z:.3f}/{p}")[..., 0] for _, z in cols]
        lo, hi = min(float(a.min()) for a in imgs), max(float(a.max()) for a in imgs)
        for j, ((t, z), a) in enumerate(zip(cols, imgs)):
            ax = axes[i, j]
            ax.imshow(a, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(f"{c} #{k + 1} {p[:-4]}\n{t} (row range {lo:.2f} to {hi:.2f})" if j == 0 else t, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Depth shift of the z0 input (mean of the four channels' intensities); 1st, 101st and 201st PET and PS real patch "
                 "in manifest order; the same grey scale (row min to max) within a row", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fn = "results/depth_shift_sheet.png"
    if os.path.exists(fn):
        raise FileExistsError(fn)
    fig.savefig(fn)


if __name__ == "__main__":
    main()
