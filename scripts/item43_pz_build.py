"""Item 43, arm Pz: per-channel z0 patches for every expanded-set real PET and PS patch (idle priority, new files only).
Per patch, exactly the z0 build (scripts/build_reconstructed_dataset.py --z 0 --s0-only) without the final mean:
raw frame f and the per-pixel median of the up to 90 preceding raw frames on the mosaic ([f-90, f), np.median of the
uint8 stack), strided demosaic (I0 [1::2,1::2], I45 [0::2,1::2], I90 [0::2,0::2], I135 [1::2,0::2]), per channel
I_c / (median_c + 1e-6) (the z0 build's z = 0 propagation is an identity), float32, the same crop (64 x 64 at the
detection centre, half 32, clipped at the frame edge), channel order I0, I45, I90, I135 (the order of the pol4
patches). Only the 128 x 128 mosaic window of each patch is read and medianed: each video is read once in 64-column
bands (HDF5) or streamed once (PET4, MAT v5).
STORAGE: the training script divides every 4-channel patch by 255 (its uint8 convention for pol4 data), so the ratios
are stored multiplied by 255; after the loader's division the network input is the per-channel z0 ratio itself.
Check: mean of the four channels / 255 against the stored z0 patch (patches_recon_s0_z0_PETPS_exp), max |difference|.
usage: python scripts/item43_pz_build.py [VIDEO ...]   (default: all six)"""
import os
import sys
import json
import time

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import h5py  # noqa: E402
import recon_check_followup as fu  # noqa: E402  (mat5_stream)
prio41.set_priority(prio41.IDLE)

PO = "pipeline_output"
OUT = f"{PO}/patches_recon_z0pol4_PETPS_exp"
REF = f"{PO}/patches_recon_s0_z0_PETPS_exp"
MAN = f"{PO}/manifest_recon_s0_z0_PETPS_exp.csv"
SCRATCH = r"C:\Users\haany\AppData\Local\Temp\claude\c--Users-haany-OneDrive-research-project-dataset-10jan23\075ab033-0168-4afa-a675-c38e4f73b481\scratchpad"
H_RAW, W_RAW, H_DEM, W_DEM = 2048, 2448, 1024, 1224
SITES = [(1, 1), (0, 1), (0, 0), (1, 0)]  # I0, I45, I90, I135 (recon.demosaic / holo_pipeline_starter order)
VIDEOS = ["PET1_600.mat", "PET2_600.mat", "PET3_600.mat", "PET4_600.mat", "PS1_600.mat", "PS2_600.mat"]


def window(x, y):
    """Mosaic window of the 64 x 64 (6.9 um) crop at (x, y), clipped at the frame edge, and its 6.9 um shape."""
    ya, yb = max(0, y - 32), min(H_DEM, y + 32)
    xa, xb = max(0, x - 32), min(W_DEM, x + 32)
    return 2 * ya, 2 * yb, 2 * xa, 2 * xb


def finish(raw_w, med_w):
    """raw and median mosaic windows (float64) -> (h, w, 4) float32 per-channel ratios."""
    ch = [raw_w[r::2, c::2] / (med_w[r::2, c::2] + 1e-6) for r, c in SITES]
    return np.stack(ch, axis=-1)


def build_video(src, rows):
    path = os.path.join(D, src)
    fmax = int(rows.frame.max())
    win = {}
    for r in rows.itertuples(index=False):
        Y0, Y1, X0, X1 = window(int(r.x), int(r.y))
        win[r.patch_file] = dict(f=int(r.frame), Y0=Y0, Y1=Y1, X0=X0, X1=X1, raw=np.zeros((Y1 - Y0, X1 - X0)),
                                 med=np.zeros((Y1 - Y0, X1 - X0)))
    if open(path, "rb").read(116).decode("latin-1").startswith("MATLAB 7.3"):
        with h5py.File(path, "r") as h:
            ds = h[[k for k in h.keys() if not k.startswith("#")][0]]
            for c0 in range(0, W_RAW, 64):
                c1 = min(W_RAW, c0 + 64)
                hit = [(p, w) for p, w in win.items() if w["X0"] < c1 and w["X1"] > c0]
                if not hit:
                    continue
                band = ds[0:fmax + 1, 0, c0:c1, :]  # (n, cols, rows)
                for p, w in hit:
                    a, b = max(c0, w["X0"]), min(c1, w["X1"])
                    sl = band[:, a - c0:b - c0, w["Y0"]:w["Y1"]]
                    w["raw"][:, a - w["X0"]:b - w["X0"]] = sl[w["f"]].T
                    w["med"][:, a - w["X0"]:b - w["X0"]] = np.median(sl[max(0, w["f"] - 90):w["f"]], axis=0).T
                del band
    else:
        lo = max(0, int(rows.frame.min()) - 90)
        tmp = os.path.join(SCRATCH, f"item43_frames_{src[:-4]}.u8")
        with open(tmp, "wb") as fo:
            fu.mat5_stream(path, list(range(lo, fmax + 1)), lambda k, a: fo.write(np.ascontiguousarray(a).tobytes()))
        mm = np.memmap(tmp, np.uint8, "r", shape=(fmax + 1 - lo, H_RAW, W_RAW))
        for p, w in win.items():
            st = np.array(mm[max(0, w["f"] - 90) - lo:w["f"] + 1 - lo, w["Y0"]:w["Y1"], w["X0"]:w["X1"]])
            w["raw"][:] = st[-1]
            w["med"][:] = np.median(st[:-1], axis=0)
        mm._mmap.close()
        del mm
        try:
            os.remove(tmp)
        except OSError:
            pass
    worst = 0.0
    for p, w in win.items():
        ratios = finish(w["raw"], w["med"]).astype(np.float32)
        ref = np.load(os.path.join(REF, p))[..., 0]
        assert ref.shape == ratios.shape[:2], (p, ref.shape, ratios.shape)
        stored = (ratios * np.float32(255.0)).astype(np.float32)
        back = stored.astype(np.float32) / 255.0  # what the loader gives the network
        worst = max(worst, float(np.max(np.abs(back.astype(np.float64).mean(axis=-1) - ref.astype(np.float64)))))
        dst = os.path.join(OUT, p)
        if os.path.exists(dst):
            raise FileExistsError(dst)
        with open(dst + ".part", "wb") as fh:
            np.save(fh, stored)
        os.replace(dst + ".part", dst)
    return worst


def main(videos):
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_csv(MAN)
    out = {}
    for src in videos:
        rows = m[(m.source_file == src) & ~m.patch_file.isin(os.listdir(OUT))]
        if rows.empty:
            continue
        t0 = time.time()
        worst = build_video(src, rows)
        out[src] = dict(patches=len(rows), frames=int(rows.frame.nunique()), seconds=round(time.time() - t0, 1), max_abs_diff=worst)
        print(time.strftime("%H:%M:%S"), src, out[src], flush=True)
    log = f"results/logs_item43/pz_build.json"
    os.makedirs("results/logs_item43", exist_ok=True)
    old = json.load(open(log)) if os.path.exists(log) else {}
    old.update(out)
    json.dump(old, open(log, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or VIDEOS)
