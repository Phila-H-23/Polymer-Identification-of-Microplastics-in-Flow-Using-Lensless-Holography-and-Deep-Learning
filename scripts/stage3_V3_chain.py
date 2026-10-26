"""Item 41 stage 3 chain (detached, idle priority; this process and every child). New files only.
STEP 1: rebuild recon_s0 as V3 for PET vs PS on the old 561 patches and their 561 paired particle-free patches
(same patch ids, frames and positions as the as-built recon_s0 PET vs PS runs). V3 exactly as the corrected Stage 1
variant B (scripts/stage1_sharpness.py process()): per-channel bilinear demosaic to 3.45 um (recon_check.bilinear),
division by the per-pixel median of the up to 90 preceding raw frames (+1e-6), sqrt, zero phase, reflect padding to 2x,
angular spectrum propagation by -2.25 mm (recon_check.transfer_rows, the code's convention), on a 1024 x 1024 px
crop at 3.45 um around the particle (crop placed inside the frame; at a frame edge the crop's own reflect padding is
the frame's), mean of the four propagated channel intensities |u|^2 on the 128 x 128 px region centred on (2x, 2y),
then 2 x 2 block means -> 64 x 64 px at 6.9 um covering exactly the as-built patch (frame-edge patches: the in-frame
part, the same shape as the stored as-built patch). Frames are read once per video (HDF5: 64-column bands of all
needed frames; PET4 MAT v5: one streamed zlib pass to a scratch file); medians as np.median over [f-90, f).
Checks (a) to (d); any failure stops the chain before training.
STEP 2: 16 runs of train_compact_cnn.py (unchanged) with the as-built recon_s0 PET vs PS settings, evaluation with
scripts/eval_track_ci_exp.py analyse() (unchanged), report results/report_recon_V3_PETvsPS.txt.
usage: python scripts/stage3_V3_chain.py            (the chain)
       python scripts/stage3_V3_chain.py test-build  (dry test into the scratchpad)"""
import os
import sys
import json
import time
import shutil
import ctypes
import statistics
import subprocess
import traceback

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import h5py  # noqa: E402
import scipy.fft as sfft  # noqa: E402
import recon_check as rc  # noqa: E402  (bilinear, spectrum, transfer_rows, SITES, DX3)
import recon_check_followup as fu  # noqa: E402  (mat5_stream)
import item41_stage3_status as st  # noqa: E402
prio41.set_priority(prio41.IDLE)  # again, after the imports

PO = "pipeline_output"
H_RAW, W_RAW, CROP = 2048, 2448, 1024
H_DEM, W_DEM = 1024, 1224
Z_MM = 2.25
VIDEOS = ["PET1_600.mat", "PET2_600.mat", "PET3_600.mat", "PET4_600.mat", "PS1_600.mat", "PS2_600.mat"]
KEY = ["source_file", "frame", "particle_id"]
SCRATCH = os.path.dirname(rc.WORK)
WORK = os.path.join(SCRATCH, "item41_stage3")
STAGE1 = os.path.join(SCRATCH, "item41_stage1_fixed")
LOGD = "results/logs_item41_stage3"
IDLE_FLAG = 0x40
CFG = dict(out_real=f"{PO}/patches_recon_V3_PETPS", out_bg=f"{PO}/patches_background_recon_V3_PETPS",
           man={("real", "original"): (f"{PO}/manifest_recon_s0.csv", f"{PO}/manifest_recon_V3_PETPS.csv"),
                ("real", "reverse"): (f"{PO}/manifest_recon_s0_rev.csv", f"{PO}/manifest_recon_V3_PETPS_rev.csv"),
                ("bg", "original"): (f"{PO}/manifest_background_recon_s0.csv", f"{PO}/manifest_background_recon_V3_PETPS.csv"),
                ("bg", "reverse"): (f"{PO}/manifest_background_recon_s0_rev.csv",
                                    f"{PO}/manifest_background_recon_V3_PETPS_rev.csv")},
           checks_json="results/recon_V3_checks_PETvsPS.json", sheet="results/recon_V3_dataset_sheet.png",
           edge_sheet="results/recon_V3_edge_check.png", runs_json="results/recon_V3_runs_PETvsPS.json",
           pred_csv="results/recon_V3_predictions_PETvsPS.csv", trackci_json="results/recon_V3_trackCI_PETvsPS.json",
           report="results/report_recon_V3_PETvsPS.txt", test=False)
STRIPE_ABS, STRIPE_REL = 0.005, 3.0  # check (c), fixed before the build


def log(msg):
    os.makedirs(LOGD, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(f"{LOGD}/chain{'_test' if CFG['test'] else ''}.log", "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


def status(key, **kw):
    if not CFG["test"]:
        st.update(key, **kw)


def avail_mb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 20


def gate(key, label):
    """Wait until 1.5 GB of RAM and 20 GB of disk are free (checked every minute)."""
    t0, t_log, waited = time.time(), 0.0, False
    while True:
        a, free = avail_mb(), shutil.disk_usage(D).free / 2 ** 30
        why = ([f"available RAM {a:.0f} MB < 1500 MB"] if a < 1500 else []) + ([f"disk {free:.0f} GB < 20 GB"] if free < 20 else [])
        if not why:
            if waited:
                log(f"gate passed before {label} after {time.time() - t0:.0f} s")
            return
        if time.time() - t_log > 1800:  # log the first wait, then every 30 minutes
            log(f"waiting before {label}: {'; '.join(why)}")
            status(key, note=f"waiting before {label} since {time.strftime('%H:%M', time.localtime(t0))}: {'; '.join(why)}")
            t_log = time.time()
        waited = True
        time.sleep(60)


# ----------------------------------------------------------------------------------------------- patch table
def patch_table():
    r = pd.read_csv(CFG["man"][("real", "original")][0])
    r = r[r["class"].isin(["PET", "PS"])].copy()
    b = pd.read_csv(CFG["man"][("bg", "original")][0])
    b = b[b["class"].isin(["PET", "PS"])].copy()
    assert len(r) == 561 and len(b) == 561 and r["class"].value_counts().to_dict() == {"PET": 342, "PS": 219}
    e = pd.read_csv(f"{PO}/manifest_PETPS_exp.csv")
    tmap = dict(zip(e.patch_file, e.track_id_oldset))
    r["track_id"] = r.patch_file.map(tmap)
    assert r.track_id.notna().all(), "old-set track id missing"
    b = b.merge(r[KEY + ["track_id"]], on=KEY, how="left")
    assert b.track_id.notna().all() and len(b) == 561, "particle-free pairing"
    return r, b


# ----------------------------------------------------------------------------------------------- frames
def frames_for_video(src, frames):
    """Raw frame (uint8) and 2 x median of the up to 90 preceding raw frames (uint16, exact) for every needed frame."""
    path = os.path.join(D, src)
    fr = sorted(set(int(f) for f in frames))
    raws = {f: np.empty((H_RAW, W_RAW), np.uint8) for f in fr}
    med2 = {f: np.empty((H_RAW, W_RAW), np.uint16) for f in fr}
    if open(path, "rb").read(116).decode("latin-1").startswith("MATLAB 7.3"):
        with h5py.File(path, "r") as h:
            ds = h[[k for k in h.keys() if not k.startswith("#")][0]]
            fmax = max(fr)
            for c0 in range(0, W_RAW, 64):
                c1 = min(W_RAW, c0 + 64)
                band = ds[0:fmax + 1, 0, c0:c1, :]  # (n, w, 2048)
                for f in fr:
                    raws[f][:, c0:c1] = band[f].T
                    med2[f][:, c0:c1] = np.round(2.0 * np.median(band[max(0, f - 90):f], axis=0)).astype(np.uint16).T
                del band
    else:  # MAT v5 with zlib (PET4): one streamed pass of the needed frame range to a scratch file
        lo_all, fmax = max(0, min(fr) - 90), max(fr)
        os.makedirs(WORK, exist_ok=True)
        tmp = os.path.join(WORK, f"frames_{src[:-4]}.u8")
        with open(tmp, "wb") as fo:
            fu.mat5_stream(path, list(range(lo_all, fmax + 1)), lambda k, a: fo.write(np.ascontiguousarray(a).tobytes()))
        mm = np.memmap(tmp, np.uint8, "r", shape=(fmax + 1 - lo_all, H_RAW, W_RAW))
        for f in fr:
            raws[f][:] = mm[f - lo_all]
            lo = max(0, f - 90)
            for r0 in range(0, H_RAW, 128):
                blk = np.array(mm[lo - lo_all:f - lo_all, r0:r0 + 128])  # a copy, so no view keeps the file open
                med2[f][r0:r0 + 128] = np.round(2.0 * np.median(blk, axis=0)).astype(np.uint16)
                del blk
        mm._mmap.close()
        del mm
        import gc
        gc.collect()
        try:
            os.remove(tmp)  # scratch copy of the frames, made by this script
        except OSError as e:
            log(f"scratch file {tmp} not removed ({e}); it is left in the scratchpad")
    return raws, med2


_HZ = []


def h225():
    if not _HZ:
        shape = (2 * CROP, 2 * CROP)
        _HZ.append(np.concatenate([rc.transfer_rows(shape, -Z_MM * 1e-3, rc.DX3, r, min(shape[0], r + 256))
                                   for r in range(0, shape[0], 256)]))
    return _HZ[0]


def v3_patch(rawf, bmedf, x, y):
    """V3 at 2.25 mm for the patch centred at (x, y) (6.9 um px). Returns (64 x 64 or in-frame part at 6.9 um,
    in-frame part of the 128 x 128 region at 3.45 um)."""
    Y, X = 2 * y, 2 * x
    r0, c0 = int(np.clip(Y - CROP // 2, 0, H_RAW - CROP)), int(np.clip(X - CROP // 2, 0, W_RAW - CROP))
    ya, yb, xa, xb = max(0, Y - 64), min(H_RAW, Y + 64), max(0, X - 64), min(W_RAW, X + 64)
    craw, cbg = rawf[r0:r0 + CROP, c0:c0 + CROP], bmedf[r0:r0 + CROP, c0:c0 + CROP]
    acc = np.zeros((yb - ya, xb - xa))
    hz, pad = h225(), CROP // 2
    for (r, cc) in rc.SITES.values():  # channel order 0, 45, 90, 135 as in Stage 1
        Ic, Bc = rc.bilinear(craw, r, cc), rc.bilinear(cbg, r, cc)
        F = rc.spectrum(np.sqrt(np.clip(Ic / (Bc + 1e-6), 0, None)).astype(np.float32), True)
        u = sfft.ifft2(F * hz, workers=1)[pad:pad + CROP, pad:pad + CROP]
        acc += np.abs(u[ya - r0:yb - r0, xa - c0:xb - c0].astype(np.complex128)) ** 2 / 4.0  # modulus squared
        del u, F, Ic, Bc
    return acc.reshape((yb - ya) // 2, 2, (xb - xa) // 2, 2).mean(axis=(1, 3)), acc


def save_new(arr, out_dir, name):
    path = os.path.join(out_dir, name)
    if os.path.exists(path):
        raise FileExistsError(path)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def build(r, b, videos=VIDEOS, only=None):
    """Build every patch not yet present (a restart continues where it stopped). only: optional set of patch
    files (test mode)."""
    os.makedirs(CFG["out_real"], exist_ok=True)
    os.makedirs(CFG["out_bg"], exist_ok=True)
    allp = pd.concat([r.assign(out=CFG["out_real"], src_dir=f"{PO}/patches_recon_s0"),
                      b.assign(out=CFG["out_bg"], src_dir=f"{PO}/patches_background_recon_s0")])
    if only is not None:
        allp = allp[allp.patch_file.isin(only)]
    n_done = 0
    for src in videos:
        sub = allp[allp.source_file == src]
        todo = sub[[not os.path.exists(os.path.join(o, p)) for o, p in zip(sub.out, sub.patch_file)]]
        if todo.empty:
            continue
        gate("build", f"frames of {src}")
        t0 = time.time()
        raws, med2 = frames_for_video(src, todo.frame.unique())
        t_read = time.time() - t0
        t1 = time.time()
        for f, g in todo.groupby("frame"):
            rawf, bmedf = raws.pop(int(f)).astype(np.float32), med2.pop(int(f)).astype(np.float32) / 2.0
            for row in g.itertuples(index=False):
                patch, _ = v3_patch(rawf, bmedf, int(row.x), int(row.y))
                v1_shape = np.load(os.path.join(row.src_dir, row.patch_file), mmap_mode="r").shape
                assert patch.shape == v1_shape[:2], (row.patch_file, patch.shape, v1_shape)
                save_new(patch.astype(np.float32)[..., None], row.out, row.patch_file)
            del rawf, bmedf
        n_done += len(todo)
        log(f"build {src}: {len(todo)} patches ({todo.frame.nunique()} frames), frames and medians {t_read:.0f} s, "
            f"propagation {time.time() - t1:.0f} s ({(time.time() - t1) / len(todo):.1f} s per patch)")
        status("build", progress=f"{src} done ({n_done} patches in this session)")
    return n_done


def write_manifests(r, b):
    files = []
    for (kind, direction), (src, dst) in CFG["man"].items():
        m = pd.read_csv(src)
        m = m[m["class"].isin(["PET", "PS"])].copy()
        tr = (r if kind == "real" else b).set_index("patch_file").track_id
        m["track_id"] = m.patch_file.map(tr)
        assert m.track_id.notna().all() and len(m) == 561
        if os.path.exists(dst):  # written by an earlier session of this chain: keep it if identical, else stop
            old = pd.read_csv(dst)
            assert old.patch_file.tolist() == m.patch_file.tolist() and old.columns.tolist() == m.columns.tolist(), dst
        else:
            m.to_csv(dst, index=False)
        files.append(dst)
    return files


# ----------------------------------------------------------------------------------------------- checks
def load_patch(d, name):
    return np.load(os.path.join(d, name))[..., 0].astype(np.float64)


def check_a(r, b):
    out = {}
    for kind, df, d in (("real", r, CFG["out_real"]), ("particle-free", b, CFG["out_bg"])):
        n_px = n_out = n_nonfinite = 0
        lo, hi = np.inf, -np.inf
        for p in df.patch_file:
            a = load_patch(d, p)
            fin = np.isfinite(a)
            n_nonfinite += int((~fin).sum())
            v = a[fin]
            n_px += v.size
            n_out += int(((v < 0.5) | (v > 1.5)).sum())
            lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
        out[kind] = dict(patches=len(df), pixels=n_px, nonfinite=n_nonfinite, frac_outside_0p5_1p5=n_out / n_px, min=lo, max=hi)
    out["pass"] = all(out[k]["nonfinite"] == 0 for k in ("real", "particle-free"))
    return out


def block_in_frame(ref128, x, y):
    """In-frame part of a 128 x 128 (3.45 um) region centred on (2x, 2y), as 2 x 2 block means."""
    Y, X = 2 * y, 2 * x
    ya, yb, xa, xb = max(0, Y - 64), min(H_RAW, Y + 64), max(0, X - 64), min(W_RAW, X + 64)
    a = ref128[ya - (Y - 64):yb - (Y - 64), xa - (X - 64):xb - (X - 64)].astype(np.float64)
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3))


def check_b(r):
    rows = []
    sel1 = json.load(open(os.path.join(STAGE1, "selection.json")))["chosen"]
    for p in sel1:
        ref = np.load(os.path.join(STAGE1, f"particle_{p['n']:02d}.npz"))[f"B_{Z_MM:.3f}"]
        mine = load_patch(CFG["out_real"], p["patch_file"])
        d = float(np.max(np.abs(block_in_frame(ref, int(p["x"]), int(p["y"])) - mine)))
        rows.append(dict(set="Stage 1 (crop)", id=f"particle {p['n']:02d}", patch_file=p["patch_file"], max_abs_diff=d))
    selc = json.load(open(os.path.join(rc.WORK, "selection.json")))["selection"]
    for s in selc:
        ref = np.load(os.path.join(rc.WORK, f"partC_{s['i']}.npz"))[f"V3_{Z_MM:.3f}"]
        mine = load_patch(CFG["out_real"], s["patch_file"])
        d = float(np.max(np.abs(block_in_frame(ref, int(s["x"]), int(s["y"])) - mine)))
        rows.append(dict(set="earlier check script (whole frame)", id=f"check patch {s['i']}", patch_file=s["patch_file"],
                         max_abs_diff=d, clipped=bool(s["clipped"])))
    return dict(rows=rows, worst=max(x["max_abs_diff"] for x in rows), pass_=all(x["max_abs_diff"] <= 1e-6 for x in rows))


def edges_near(x, y):
    """Frame edges whose distance to the 64 px patch window is at most 32 px (220 um)."""
    e = []
    if x - 32 <= 32:
        e.append("left")
    if W_DEM - (x + 32) <= 32:
        e.append("right")
    if y - 32 <= 32:
        e.append("top")
    if H_DEM - (y + 32) <= 32:
        e.append("bottom")
    return e


def alt(p):
    p = p - p.mean()
    return abs(float(np.mean(p * (-1.0) ** np.arange(len(p)))))


def stripe(a, edge):
    """Column (or row) alternation in the 8 columns (rows) of the stored patch nearest the frame edge, and the same in
    the 8 at the opposite side. The as-built V1 edge stripe is this alternation (Stage 1 particle 02: 0.029 against
    0.0007)."""
    prof = a.mean(axis=0) if edge in ("left", "right") else a.mean(axis=1)
    near, far = (prof[:8], prof[-8:]) if edge in ("left", "top") else (prof[-8:], prof[:8])
    return alt(near), alt(far)


def check_c(r, b):
    rows = []
    for kind, df, d_new, d_old in (("real", r, CFG["out_real"], f"{PO}/patches_recon_s0"),
                                   ("particle-free", b, CFG["out_bg"], f"{PO}/patches_background_recon_s0")):
        for row in df.itertuples(index=False):
            for e in edges_near(int(row.x), int(row.y)):
                v3, v1 = load_patch(d_new, row.patch_file), load_patch(d_old, row.patch_file)
                n3, f3 = stripe(v3, e)
                n1, f1 = stripe(v1, e)
                rows.append(dict(kind=kind, patch_file=row.patch_file, x=int(row.x), y=int(row.y), edge=e,
                                 v3_near=n3, v3_far=f3, v3_stripe=bool(n3 > STRIPE_ABS and n3 > STRIPE_REL * f3),
                                 v1_near=n1, v1_far=f1, v1_stripe=bool(n1 > STRIPE_ABS and n1 > STRIPE_REL * f1)))
    p02 = [x for x in rows if x["patch_file"] == "PET1_600_particle9_frame176.npy" and x["kind"] == "real"]
    n3 = sum(x["v3_stripe"] for x in rows)
    return dict(rows=rows, n_cases=len(rows), n_patches=len(set((x["kind"], x["patch_file"]) for x in rows)),
                v3_stripes=n3, v1_stripes=sum(x["v1_stripe"] for x in rows), particle02=p02,
                pass_=(n3 == 0 and len(p02) > 0 and not any(x["v3_stripe"] for x in p02)))


def sheets(r, b, c_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gray = matplotlib.colormaps["gray"].copy()
    gray.set_bad("#9ecae1")
    ext = [-220.8, 220.8, 220.8, -220.8]
    sel1 = json.load(open(os.path.join(STAGE1, "selection.json")))["chosen"]
    pf_of = {"PET1_600_particle21_frame317.npy": 1, "PET2_600_particle143_frame599.npy": 3,
             "PS1_600_particle58_frame317.npy": 5, "PS2_600_particle115_frame599.npy": 7}
    bmap = b.set_index(KEY)
    rows = []
    for p in sel1:
        rows.append((f"{p['n']:02d} {p['cls']} {p['patch_file'][:-4]}", int(p["x"]), int(p["y"]),
                     load_patch(f"{PO}/patches_recon_s0", p["patch_file"]), load_patch(CFG["out_real"], p["patch_file"])))
    for p in sel1:
        if p["patch_file"] in pf_of:
            q = bmap.loc[(p["patch_file"].split("_")[0] + "_600.mat", int(p["frame"]), int(p["patch_file"].split("particle")[1].split("_")[0]))]
            rows.append((f"particle-free pair of {p['n']:02d}\n{q.patch_file[:-4]}", int(q.x), int(q.y),
                         load_patch(f"{PO}/patches_background_recon_s0", q.patch_file), load_patch(CFG["out_bg"], q.patch_file)))
    fig, axes = plt.subplots(len(rows), 4, figsize=(10, 2.45 * len(rows)), dpi=90)
    for i, (t, x, y, v1, v3) in enumerate(rows):
        for j, (img, lab, fixed) in enumerate(((v1, "as-built V1", True), (v3, "V3", True), (v1, "as-built V1", False), (v3, "V3", False))):
            ax = axes[i, j]
            canvas = rc.place(img, x, y, 32)
            lo, hi = (0.5, 1.5) if fixed else np.nanpercentile(canvas, [1, 99])
            ax.imshow(canvas, cmap=gray, vmin=lo, vmax=hi, interpolation="nearest", extent=ext)
            ax.plot([-200, -150], [200, 200], color="yellow", lw=2)
            ax.set_title(f"{t}\n{lab}, " + ("0.5 to 1.5" if fixed else f"1st-99th pct ({lo:.2f} to {hi:.2f})"), fontsize=6.5)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("V3 dataset (stored 64 x 64 patches, 6.9 um, 441.6 um square; scale bar 50 um) beside the as-built V1 patches:\n"
                 "the 10 Stage 1 particles, then the particle-free pairs of particles 01, 03, 05 and 07", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    if os.path.exists(CFG["sheet"]):
        raise FileExistsError(CFG["sheet"])
    fig.savefig(CFG["sheet"])
    plt.close(fig)
    # edge check sheet: particle 02, then the 7 other edge cases with the largest V1 stripe value
    cases = sorted(c_res["rows"], key=lambda x: -x["v1_near"])
    pick = [x for x in cases if x["patch_file"] == "PET1_600_particle9_frame176.npy"][:1]
    pick += [x for x in cases if x not in pick][:8 - len(pick)]
    fig, axes = plt.subplots(len(pick), 2, figsize=(5.4, 2.6 * len(pick)), dpi=90)
    for i, x in enumerate(pick):
        d_old = f"{PO}/patches_recon_s0" if x["kind"] == "real" else f"{PO}/patches_background_recon_s0"
        d_new = CFG["out_real"] if x["kind"] == "real" else CFG["out_bg"]
        for j, (d, lab, nv) in enumerate(((d_old, "as-built V1", x["v1_near"]), (d_new, "V3", x["v3_near"]))):
            ax = axes[i, j]
            ax.imshow(rc.place(load_patch(d, x["patch_file"]), x["x"], x["y"], 32), cmap=gray, vmin=0.5, vmax=1.5,
                      interpolation="nearest", extent=ext)
            ax.set_title(f"{x['kind']} {x['patch_file'][:-4]}\n{lab}, {x['edge']} edge, alternation {nv:.4f}", fontsize=6.5)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Edge stripe check (display 0.5 to 1.5)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    if os.path.exists(CFG["edge_sheet"]):
        raise FileExistsError(CFG["edge_sheet"])
    fig.savefig(CFG["edge_sheet"])
    plt.close(fig)
    return [CFG["sheet"], CFG["edge_sheet"]]


# ----------------------------------------------------------------------------------------------- training
def asbuilt_dir(kind, direction, seed):
    return f"results/{'real' if kind == 'real' else 'bg'}_tc_PETvsPS_recon_s0{'_rev' if direction == 'reverse' else ''}_s{seed}"


def cmd(kind, direction, seed, out):
    a = json.load(open(f"{asbuilt_dir(kind, direction, seed)}/results.json"))["args"]
    man = CFG["man"][("real" if kind == "real" else "bg", direction)][1]
    pdir = CFG["out_real"] if kind == "real" else CFG["out_bg"]
    c = [sys.executable, "train_compact_cnn.py", "--channels", a["channels"], "--split", a["split"], "--manifest", man,
         "--patch-dir", pdir, "--class-weights", a["class_weights"], "--classes", a["classes"], "--epochs", str(a["epochs"]),
         "--patience", str(a["patience"]), "--batch-size", str(a["batch_size"]), "--lr", str(a["lr"]), "--weight-decay",
         str(a["weight_decay"]), "--width", str(a["width"]), "--dropout", str(a["dropout"]), "--seed", str(seed), "--out", out]
    assert a["aug"] is None and not a["patch_norm"] and a["donor_manifest"] is None
    return c


def train_all():
    runs = {"original": {"real": {}, "particle-free": {}}, "reverse": {"real": {}, "particle-free": {}}}
    failed, timing = [], []
    for direction, seeds in (("original", range(5)), ("reverse", range(3))):
        for seed in seeds:
            for kind in ("real", "particle-free"):
                base = f"results/{'real' if kind == 'real' else 'bg'}_tc_PETvsPS_recon_V3{'_rev' if direction == 'reverse' else ''}_s{seed}"
                done = None
                for out in (base, base + "_retry1"):
                    if os.path.exists(f"{out}/results.json"):
                        done = out
                        break
                    if os.path.exists(out):
                        log(f"{out} exists without results.json; not reused (new files only)")
                        continue
                    gate("train", os.path.basename(out))
                    status("train", note=f"running {os.path.basename(out)}")
                    t0 = time.time()
                    with open(f"{LOGD}/{os.path.basename(out)}.log", "w", encoding="utf-8") as fh:
                        rc_ = subprocess.run(cmd(kind, direction, seed, out), stdout=fh, stderr=subprocess.STDOUT,
                                             creationflags=IDLE_FLAG).returncode
                    ok = rc_ == 0 and os.path.exists(f"{out}/results.json")
                    same = None
                    if ok:
                        a_new = json.load(open(f"{out}/results.json"))["args"]
                        a_old = json.load(open(f"{asbuilt_dir(kind, direction, seed)}/results.json"))["args"]
                        same = all(a_new.get(k) == v for k, v in a_old.items() if k not in ("manifest", "patch_dir", "out"))
                    timing.append(dict(run=os.path.basename(out), seconds=round(time.time() - t0), exit=rc_, ok=ok, settings_same=same))
                    log(f"{os.path.basename(out)} exit={rc_} {time.time() - t0:.0f}s {'ok' if ok else 'FAILED'}; settings as built: {same}")
                    if ok:
                        done = out
                        break
                if done:
                    runs[direction][kind][seed] = done
                else:
                    failed.append(base)
                    log(f"{base}: failed twice, skipped")
                json.dump(dict(runs=runs, failed=failed, timing=timing), open(CFG["runs_json"], "w"), indent=1)
                status("train", progress=f"{sum(len(v) for d in runs.values() for v in d.values())} of 16 runs done; failed {failed or 'none'}")
    return runs, failed, timing


# ----------------------------------------------------------------------------------------------- report
def f1_of(d):
    return json.load(open(f"{d}/results.json"))["test_report"]["macro avg"]["f1-score"]


def compare(a, b):
    """a, b: {seed: f1}. Means over the shared seeds, margin = larger population std of the two over those seeds."""
    s = sorted(set(a) & set(b))
    av, bv = [a[k] for k in s], [b[k] for k in s]
    sa = statistics.pstdev(av) if len(av) > 1 else float("nan")
    sb = statistics.pstdev(bv) if len(bv) > 1 else float("nan")
    return dict(seeds=s, a=statistics.mean(av), b=statistics.mean(bv), sa=sa, sb=sb, margin=max(sa, sb),
                diff=statistics.mean(av) - statistics.mean(bv))


def reading(cmps, kind):
    """kind 'R' (against z0) or 'M' (against as-built recon_s0)."""
    d = [c["diff"] for c in cmps.values()]
    m = [c["margin"] for c in cmps.values()]
    within = all(abs(x) <= y for x, y in zip(d, m))
    above = all(x > y for x, y in zip(d, m))
    below = all(x < -y for x, y in zip(d, m))
    if kind == "R":
        return "R1" if above else "R2" if within else "R3" if below else "R4"
    return "M1" if within else "M2" if above else "M3" if below else "M4"


def report(runs, failed, timing, checks):
    import eval_track_ci_exp as ev
    for p in (CFG["pred_csv"], CFG["trackci_json"], CFG["report"]):
        if os.path.exists(p):
            raise FileExistsError(p)
    L_ev, summ, excl = ev.analyse("recon_V3", runs, CFG["pred_csv"])
    json.dump(summ, open(CFG["trackci_json"], "w"), indent=1, default=float)
    new = {(d, k): {int(s): f1_of(p) for s, p in runs[d][k].items()} for d in runs for k in runs[d]}
    ref = {}
    for arm, pref in (("z0", "z0"), ("asbuilt", "recon_s0")):
        for d, seeds in (("original", range(5)), ("reverse", range(3))):
            for k, pk in (("real", "real"), ("particle-free", "bg")):
                ref[(arm, d, k)] = {s: f1_of(f"results/{pk}_tc_PETvsPS_{pref}{'_rev' if d == 'reverse' else ''}_s{s}") for s in seeds}
    cz = {d: compare(new[(d, "real")], ref[("z0", d, "real")]) for d in ("original", "reverse")}
    ca = {d: compare(new[(d, "real")], ref[("asbuilt", d, "real")]) for d in ("original", "reverse")}
    R, M = reading(cz, "R"), reading(ca, "M")
    ms = lambda v: (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else float("nan"))
    a_ck, b_ck, c_ck = checks["a"], checks["b"], checks["c"]
    L = ["V3 REBUILD OF recon_s0, PET vs PS, OLD 561-PATCH SET (item 41, stage 3)", "",
         "BUILD. " + " ".join(x.strip() for x in __doc__.split("STEP 1:")[1].split("Checks (a)")[0].split("\n")),
         f"Real patches: the 561 PET and PS rows of {CFG['man'][('real', 'original')][0]} (342 PET, 219 PS); particle-free:",
         f"the 561 paired rows of {CFG['man'][('bg', 'original')][0]} (same frame, own position). New patch folders",
         f"{CFG['out_real']} and {CFG['out_bg']}; new manifests (as-built rows and splits, plus track_id = the old-set track",
         "id track_id_oldset of pipeline_output/manifest_PETPS_exp.csv; a particle-free patch carries its pair's):",
         "  " + ", ".join(v[1] for v in CFG["man"].values()), "",
         "CHECKS: (a), (b) and (d) passed; (c) FAILED by the rule fixed before the build, which was not rewritten or loosened.",
         "Going ahead with the training on the dataset as built was Phila's decision after seeing the check (c) result.",
         f"(a) every patch finite (non-finite pixels: real {a_ck['real']['nonfinite']}, particle-free {a_ck['particle-free']['nonfinite']}); "
         f"pixels outside 0.5 to 1.5: real {a_ck['real']['frac_outside_0p5_1p5']:.4%}, particle-free "
         f"{a_ck['particle-free']['frac_outside_0p5_1p5']:.4%} (ranges {a_ck['real']['min']:.3f} to {a_ck['real']['max']:.3f} and "
         f"{a_ck['particle-free']['min']:.3f} to {a_ck['particle-free']['max']:.3f})",
         f"(b) stored V3 patch against the corrected Stage 1 output (10 particles, crop) and the earlier check script output (8 "
         f"patches, whole frame), both after the same 2 x 2 block means: largest difference {b_ck['worst']:.2e} (limit 1e-6)"]
    L += [f"      {x['id']:15s} {x['patch_file'][:-4]:34s} {x['max_abs_diff']:.2e}  {x['set']}" for x in b_ck["rows"]]
    L += [f"(c) FAILED. Edge stripes: {c_ck['n_patches']} patches within 220 um of a frame edge ({c_ck['n_cases']} patch-edge cases, real and "
          f"particle-free), stripe = alternation of the 8 column (row) means nearest the edge > {STRIPE_ABS} and > {STRIPE_REL} x the "
          f"same at the opposite side of the patch (rule fixed before the build): V3 {c_ck['v3_stripes']} cases, as-built V1 "
          f"{c_ck['v1_stripes']} cases; Stage 1 particle 02: " + "; ".join(
              f"{x['edge']} edge V3 {x['v3_near']:.4f} (far {x['v3_far']:.4f}), V1 {x['v1_near']:.4f} (far {x['v1_far']:.4f})"
              for x in c_ck["particle02"]) + f"; sheet {CFG['edge_sheet']}; the 7 flagged patches: results/recon_V3_flagged_ids.txt "
          "and results/recon_V3_edge_flagged.png",
          f"(d) contact sheet {CFG['sheet']}", ""]
    L += ["TRAINING. train_compact_cnn.py (unchanged), settings read from the as-built recon_s0 PET vs PS run folders",
          "results/{real,bg}_tc_PETvsPS_recon_s0[_rev]_s<seed>/results.json and passed explicitly: --channels recon_s0, --split",
          "video_holdout (real) / particle (particle-free), --class-weights pipeline_output/class_weights_twoclass_PETvsPS[_rev].json,",
          "--classes PET,PS, --epochs 150, --patience 30, --batch-size 32, --lr 0.001, --weight-decay 0.0001, --width 32,",
          "--dropout 0.3, no surround augmentation, no patch norm (the script's own training augmentation as in those runs);",
          "only --manifest and --patch-dir point to the V3 build. Original seeds 0 to 4, reverse seeds 0 to 2.",
          f"Every run's recorded settings equal the as-built run's apart from manifest, patch_dir and out: "
          f"{all(t['settings_same'] for t in timing if t['ok'])}. Runs skipped after a failed retry: {failed or 'none'}. Runs excluded "
          f"by the 4-decimal check of the evaluation: {excl or 'none'}.", "",
          "SUMMARY (macro-F1, mean +/- population std over seeds; floor = larger seed std; chance = mean macro-F1 of 1000 random",
          "draws with the training class proportions; T as in the earlier items):",
          f"{'arm':26s} {'direction':9s} {'real':>15s} {'particle-free':>15s} {'gap':>8s} {'floor':>7s} {'chance':>7s}  T"]
    for d in ("original", "reverse"):
        s = summ[d]
        L.append(f"{'V3 rebuilt recon_s0':26s} {d:9s} {s['real'][0]:.4f}+/-{s['real'][1]:.4f} {s['pf'][0]:.4f}+/-{s['pf'][1]:.4f} "
                 f"{s['gap']:+.4f} {s['floor']:.4f} {s['chance']:.4f}  {s['T']}")
    earlier = {("z0", "original"): "0.8735+/-0.0217 0.6279+/-0.0323 +0.2455 0.0323 0.4997  T1",
               ("z0", "reverse"): "0.7593+/-0.0078 0.5301+/-0.0331 +0.2292 0.0331 0.3895  T1",
               ("asbuilt", "original"): "0.8679+/-0.0229 0.6252+/-0.0058 +0.2427 0.0229 0.4997  T1",
               ("asbuilt", "reverse"): "0.7288+/-0.0893 0.5276+/-0.0255 +0.2013 0.0893 0.3895  T1"}
    for arm, lab in (("z0", "z0, same set"), ("asbuilt", "as-built recon_s0 (V1)")):
        for d in ("original", "reverse"):
            rr, pp = ms(list(ref[(arm, d, "real")].values())), ms(list(ref[(arm, d, "particle-free")].values()))
            L.append(f"{lab:26s} {d:9s} {rr[0]:.4f}+/-{rr[1]:.4f} {pp[0]:.4f}+/-{pp[1]:.4f} {rr[0] - pp[0]:+.4f} {max(rr[1], pp[1]):.4f}"
                     f"         (earlier report, results/report_size_and_arms.txt: {earlier[(arm, d)]})")
    L += ["", "READINGS (fixed beforehand; margin per direction = larger real seed std of the two compared arms over the seeds",
          "they share; means over those seeds):"]
    for d in ("original", "reverse"):
        z, a = cz[d], ca[d]
        L.append(f"{d:9s} vs z0: V3 {z['a']:.4f} - z0 {z['b']:.4f} = {z['diff']:+.4f}, margin {z['margin']:.4f} (seeds {z['seeds']}); "
                 f"vs as-built: V3 {a['a']:.4f} - V1 {a['b']:.4f} = {a['diff']:+.4f}, margin {a['margin']:.4f}")
    L += [f"R reading (against z0): {R}   (R1 above by more than the margin in both directions, R2 within in both, R3 below by "
          f"more than the margin in both, R4 none of these)",
          f"M reading (against the as-built recon_s0): {M}   (M1 within in both, M2 above by more than the margin in both, M3 below "
          f"by more than the margin in both, M4 none of these)",
          f"T reading of the rebuilt arm: original {summ['original']['T']}, reverse {summ['reverse']['T']}", "",
          "PER-CLASS RECALL AND PRECISION, track-resampled 95% intervals (scripts/eval_track_ci_exp.py analyse(): 1000 resamples",
          "of whole test tracks, numpy default_rng(0), the same resampled tracks for the real and the particle-free classifier;",
          f"tracks = old-set track ids). Predictions of every run: {CFG['pred_csv']} (run, kind, direction, seed, patch id, track",
          "id, true class, predicted class)."] + L_ev
    # extra tables asked for after the check (c) result: unclipped test patches only, and only the 7 flagged patches dropped
    import unclipped_tracks as ut
    pred = pd.read_csv(CFG["pred_csv"])
    flagged = [x for x in c_ck["rows"] if x["v3_stripe"]]
    f_real = {x["patch_file"] for x in flagged if x["kind"] == "real"}
    f_pf = {x["patch_file"] for x in flagged if x["kind"] == "particle-free"}
    full, red, drops_u, red7, drops7, clip_info = {}, {}, {}, {}, {}, []
    for d in ("original", "reverse"):
        m, bm = pd.read_csv(CFG["man"][("real", d)][1]), pd.read_csv(CFG["man"][("bg", d)][1])
        clip_real = set(m.patch_file[[ut.clipped(x, y) for x, y in zip(m.x, m.y)]])
        stored_clip = {p for p in m.patch_file if np.load(os.path.join(CFG["out_real"], p), mmap_mode="r").shape[:2] != (64, 64)}
        pairs = bm.merge(m[KEY + ["patch_file"]], on=KEY, suffixes=("", "_real"))
        assert len(pairs) == len(bm)
        clip_pf = set(pairs.patch_file[pairs.patch_file_real.isin(clip_real)])
        train = m[m.split_video_holdout == "train"]["class"].values
        te = m[m.split_video_holdout == "test"]
        full[d] = ut.analyse(pred, d, train, te["class"].values)
        red[d] = ut.analyse(pred, d, train, te[~te.patch_file.isin(clip_real)]["class"].values, clip_real, clip_pf)
        drops_u[d] = ut.dropped_counts(pred, d, clip_real, clip_pf)
        red7[d] = ut.analyse(pred, d, train, te[~te.patch_file.isin(f_real)]["class"].values, f_real, f_pf)
        drops7[d] = ut.dropped_counts(pred, d, f_real, f_pf)
        clip_info.append(f"{d}: clipped real patches {len(clip_real)} (same as the patches stored smaller than 64 x 64: "
                         f"{clip_real == stored_clip}), of them in the test split {len(clip_real & set(te.patch_file))}")
    U = "U1" if all(red[d]["U_cond"] for d in red) else "U2"
    L += ["", "UNCLIPPED TEST PATCHES ONLY (as for z0 in results/report_z0_unclipped_exp.txt; from the saved predictions above with",
          "scripts/unclipped_tracks.py, nothing retrained): every real test patch clipped at a frame edge (Part 3b definition) is",
          "dropped together with its paired particle-free test patch (same source_file, frame, particle_id); the track bootstrap",
          "resamples the tracks left in the two reduced sets, the same tracks for both; chance from 1000 random label draws on the",
          "reduced real test labels.", "  " + "; ".join(clip_info)]
    L += ut.table("recon_V3", full, red, drops_u, "unclipped only")
    L += [f"U reading (the rule fixed for z0: U1 if gap > floor and real > chance + 0.10 in both directions): {U}", "",
          "ONLY THE 7 FLAGGED PATCHES DROPPED (results/recon_V3_flagged_ids.txt): exactly those patches are removed wherever they",
          "are test patches (6 real, 1 particle-free); their pairs are kept.", "  flagged real test patches: " + "; ".join(
              f"{d} {len(f_real & set(pred[(pred.direction == d) & (pred.kind == 'real')].patch_id))}" for d in ("original", "reverse"))
          + "; flagged particle-free test patches: " + "; ".join(
              f"{d} {len(f_pf & set(pred[(pred.direction == d) & (pred.kind == 'particle-free')].patch_id))}" for d in ("original", "reverse"))]
    L += ut.table("recon_V3", full, red7, drops7, "7 flagged dropped")
    L += ["Check: the full-set rows above recompute the main summary from the saved predictions with the same procedure; equal "
          "to 4 decimals: " + ", ".join(f"{d} {ut.same_as_saved(full[d], d, path=CFG['trackci_json'])}" for d in ("original", "reverse"))]
    L += ["", "Run times: " + ", ".join(f"{t['run']} {t['seconds']} s" for t in timing)]
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    open(CFG["report"], "w", encoding="utf-8").write(text + "\n")
    rows = []
    for d in ("original", "reverse"):
        s = summ[d]
        rows.append(f"| {d} | {s['real'][0]:.4f} +/- {s['real'][1]:.4f} | {s['pf'][0]:.4f} +/- {s['pf'][1]:.4f} | {s['gap']:+.4f} | "
                    f"{s['floor']:.4f} | {s['chance']:.4f} | {s['T']} | {cz[d]['diff']:+.4f} (margin {cz[d]['margin']:.4f}) | "
                    f"{ca[d]['diff']:+.4f} (margin {ca[d]['margin']:.4f}) |")
    table = ["| direction | V3 real | V3 particle-free | gap | floor | chance | T | V3 - z0 real | V3 - as-built real |",
             "|---|---|---|---|---|---|---|---|---|"] + rows + [
        "", f"R reading (against z0): **{R}**. M reading (against the as-built recon_s0): **{M}**.",
        f"Unclipped test patches only (V3): real {red['original']['real']['f1'][0]:.4f} / {red['reverse']['real']['f1'][0]:.4f}, "
        f"particle-free {red['original']['particle-free']['f1'][0]:.4f} / {red['reverse']['particle-free']['f1'][0]:.4f} "
        f"(original / reverse), U reading **{U}**. Only the 7 flagged patches dropped: real {red7['original']['real']['f1'][0]:.4f} / "
        f"{red7['reverse']['real']['f1'][0]:.4f}, particle-free {red7['original']['particle-free']['f1'][0]:.4f} / "
        f"{red7['reverse']['particle-free']['f1'][0]:.4f}."]
    summary = [
        f"V3 dataset built: 561 real and 561 particle-free patches; checks (a), (b), (d) passed (largest difference to the Stage 1 and "
        f"check-script outputs {b_ck['worst']:.1e}); check (c) FAILED by the rule fixed beforehand ({c_ck['v3_stripes']} of "
        f"{c_ck['n_cases']} edge cases flagged in V3, {c_ck['v1_stripes']} in V1); going ahead was Phila's decision after seeing it.",
        f"Original direction: V3 real {summ['original']['real'][0]:.4f}, particle-free {summ['original']['pf'][0]:.4f}, gap "
        f"{summ['original']['gap']:+.4f}, {summ['original']['T']} (z0 0.8735 / 0.6279; as-built 0.8679 / 0.6252).",
        f"Reverse: V3 real {summ['reverse']['real'][0]:.4f}, particle-free {summ['reverse']['pf'][0]:.4f}, gap "
        f"{summ['reverse']['gap']:+.4f}, {summ['reverse']['T']} (z0 0.7593 / 0.5301; as-built 0.7288 / 0.5276).",
        f"R reading {R}: V3 minus z0 real {cz['original']['diff']:+.4f} (margin {cz['original']['margin']:.4f}) original, "
        f"{cz['reverse']['diff']:+.4f} (margin {cz['reverse']['margin']:.4f}) reverse.",
        f"M reading {M}: V3 minus as-built real {ca['original']['diff']:+.4f} (margin {ca['original']['margin']:.4f}) original, "
        f"{ca['reverse']['diff']:+.4f} (margin {ca['reverse']['margin']:.4f}) reverse. Full report {CFG['report']}."]
    return R, M, summ, summary, table


# ----------------------------------------------------------------------------------------------- main
def main():
    os.makedirs(LOGD, exist_ok=True)
    log(f"chain started, pid {os.getpid()}, idle priority")
    r, b = patch_table()
    s0 = st.load()
    if s0.get("checks", {}).get("status") == "done" and os.path.exists(CFG["checks_json"]):  # restart after the checks
        log("build and checks already done in an earlier session: going on to training")
        checks = json.load(open(CFG["checks_json"]))
        return after_checks(checks)
    status("build", status="running", started=time.strftime("%Y-%m-%d %H:%M"), pid=os.getpid())
    try:
        t0 = time.time()
        build(r, b)
        files = write_manifests(r, b)
        n_real = sum(os.path.exists(os.path.join(CFG["out_real"], p)) for p in r.patch_file)
        n_bg = sum(os.path.exists(os.path.join(CFG["out_bg"], p)) for p in b.patch_file)
        assert n_real == 561 and n_bg == 561, (n_real, n_bg)
        status("build", status="done", finished=time.strftime("%Y-%m-%d %H:%M"),
               summary=f"561 real + 561 particle-free patches in {(time.time() - t0) / 60:.0f} min (this session)",
               files=[CFG["out_real"] + "/", CFG["out_bg"] + "/"] + files + ["scripts/stage3_V3_chain.py", "scripts/item41_stage3_status.py"])
    except Exception:
        log("BUILD FAILED\n" + traceback.format_exc())
        status("build", status="failed (chain stopped)", note=traceback.format_exc()[-600:])
        return
    status("checks", status="running", started=time.strftime("%Y-%m-%d %H:%M"))
    try:
        gate("checks", "checks")
        ca_, cb_, cc_ = check_a(r, b), check_b(r), check_c(r, b)
        sheet_files = sheets(r, b, cc_)
        checks = dict(a=ca_, b=cb_, c=cc_)
        json.dump(checks, open(CFG["checks_json"], "w"), indent=1, default=float)
        ok = ca_["pass"] and cb_["pass_"] and cc_["pass_"]
        summ = (f"(a) {'pass' if ca_['pass'] else 'FAIL'}: non-finite real {ca_['real']['nonfinite']}, particle-free "
                f"{ca_['particle-free']['nonfinite']}; outside 0.5 to 1.5 real {ca_['real']['frac_outside_0p5_1p5']:.4%}, particle-free "
                f"{ca_['particle-free']['frac_outside_0p5_1p5']:.4%}. (b) {'pass' if cb_['pass_'] else 'FAIL'}: largest difference "
                f"{cb_['worst']:.2e} (limit 1e-6). (c) {'pass' if cc_['pass_'] else 'FAIL'}: V3 stripe cases {cc_['v3_stripes']} of "
                f"{cc_['n_cases']} (as-built V1 {cc_['v1_stripes']}). (d) sheets written.")
        status("checks", status="done" if ok else "FAILED (chain stopped before training; waiting for Phila)",
               finished=time.strftime("%Y-%m-%d %H:%M"), summary=summ, files=[CFG["checks_json"]] + sheet_files)
        log("checks: " + summ)
        if not ok:
            return
    except Exception:
        log("CHECKS FAILED TO RUN\n" + traceback.format_exc())
        status("checks", status="failed to run (chain stopped)", note=traceback.format_exc()[-600:])
        return
    after_checks(checks)


def after_checks(checks):
    status("train", status="running", started=time.strftime("%Y-%m-%d %H:%M"), pid=os.getpid())
    runs, failed, timing = train_all()
    status("train", status="done", finished=time.strftime("%Y-%m-%d %H:%M"), files=[CFG["runs_json"]],
           summary=f"{sum(len(v) for d in runs.values() for v in d.values())} of 16 runs; failed {failed or 'none'}")
    status("report", status="running", started=time.strftime("%Y-%m-%d %H:%M"))
    try:
        gate("report", "evaluation")
        R, M, summ, summary, table = report(runs, failed, timing, checks)
        st.top(summary, table)
        status("report", status="done", finished=time.strftime("%Y-%m-%d %H:%M"), reading=f"{R}, {M}; T original "
               f"{summ['original']['T']}, reverse {summ['reverse']['T']}",
               files=[CFG["report"], CFG["pred_csv"], CFG["trackci_json"]])
        log(f"report written: {R}, {M}")
    except Exception:
        log("REPORT FAILED\n" + traceback.format_exc())
        status("report", status="failed", note=traceback.format_exc()[-600:])


def step2():
    """Phila's decision on 2026-10-08 evening, after seeing check (c) fail (rule unchanged): run Step 2 on the dataset
    as built. Trains the 16 runs (train_all), then runs scripts/stage3_V3_report.py (written while the runs train)."""
    log("STEP 2 started by Phila's decision after check (c) FAILED by the rule fixed beforehand (rule unchanged)")
    status("train", status="running", started=time.strftime("%Y-%m-%d %H:%M"), pid=os.getpid())
    runs, failed, timing = train_all()
    status("train", status="done", finished=time.strftime("%Y-%m-%d %H:%M"), files=[CFG["runs_json"]],
           summary=f"{sum(len(v) for d in runs.values() for v in d.values())} of 16 runs; failed {failed or 'none'}")
    for attempt in (1, 2):
        gate("report", "evaluation and report")
        status("report", status="running", started=time.strftime("%Y-%m-%d %H:%M"))
        with open(f"{LOGD}/report_attempt{attempt}.log", "w", encoding="utf-8") as fh:
            rc_ = subprocess.run([sys.executable, "scripts/stage3_V3_report.py"], stdout=fh, stderr=subprocess.STDOUT,
                                 creationflags=IDLE_FLAG).returncode
        log(f"report script attempt {attempt} exit {rc_}")
        if rc_ == 0:
            return
    status("report", status="failed twice (see results/logs_item41_stage3/report_attempt*.log)")


def test_build():
    """Dry test into the scratchpad: build 4 real patches (Stage 1 particle 09 = check patch 2, PET3; check patch 3,
    PET4 frame edge; Stage 1 particle 02, PET1, near the right edge) and one particle-free patch, then checks (b)
    and (c) on them and the sheets with as-built V1 copies standing in for missing V3 patches."""
    T = os.path.join(SCRATCH, "item41_stage3_test")
    CFG.update(out_real=os.path.join(T, "real"), out_bg=os.path.join(T, "bg"), sheet=os.path.join(T, "sheet.png"),
               edge_sheet=os.path.join(T, "edge.png"), test=True)
    r, b = patch_table()
    only = {"PET3_600_particle35_frame211.npy", "PET4_600_particle33_frame68.npy", "PET1_600_particle9_frame176.npy"}
    pair = b.merge(r[r.patch_file == "PET1_600_particle9_frame176.npy"][KEY], on=KEY).patch_file.iloc[0]
    t0 = time.time()
    build(r, b, videos=["PET3_600.mat", "PET4_600.mat", "PET1_600.mat"], only=only | {pair})
    log(f"test build {time.time() - t0:.0f} s")
    for name, ref_set in (("PET3_600_particle35_frame211.npy", "both"), ("PET4_600_particle33_frame68.npy", "check"),
                          ("PET1_600_particle9_frame176.npy", "stage1")):
        row = r[r.patch_file == name].iloc[0]
        mine = load_patch(CFG["out_real"], name)
        out = [name, mine.shape]
        if ref_set in ("both", "stage1"):
            p = [q for q in json.load(open(os.path.join(STAGE1, "selection.json")))["chosen"] if q["patch_file"] == name][0]
            ref = np.load(os.path.join(STAGE1, f"particle_{p['n']:02d}.npz"))["B_2.250"]
            out.append(f"vs Stage 1 {np.max(np.abs(block_in_frame(ref, int(row.x), int(row.y)) - mine)):.2e}")
        if ref_set in ("both", "check"):
            s = [q for q in json.load(open(os.path.join(rc.WORK, "selection.json")))["selection"] if q["patch_file"] == name][0]
            ref = np.load(os.path.join(rc.WORK, f"partC_{s['i']}.npz"))["V3_2.250"]
            out.append(f"vs check script {np.max(np.abs(block_in_frame(ref, int(row.x), int(row.y)) - mine)):.2e}")
        log("test: " + " ".join(str(x) for x in out))
    # check (c) and sheets: copy-free stand-in, read V1 where V3 is missing
    for d_new, d_old, df in ((CFG["out_real"], f"{PO}/patches_recon_s0", r), (CFG["out_bg"], f"{PO}/patches_background_recon_s0", b)):
        for p in df.patch_file:
            if not os.path.exists(os.path.join(d_new, p)):
                shutil.copyfile(os.path.join(d_old, p), os.path.join(d_new, p))
    cc_ = check_c(r, b)
    log(f"test check (c) on V1 stand-ins plus 4 V3 patches: cases {cc_['n_cases']} in {cc_['n_patches']} patches, V3-folder stripes "
        f"{cc_['v3_stripes']}, V1 stripes {cc_['v1_stripes']}, particle 02 {cc_['particle02']}")
    log(f"test check (a): {check_a(r, b)}")
    log(f"test sheets: {sheets(r, b, cc_)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test-build":
        test_build()
    elif len(sys.argv) > 1 and sys.argv[1] == "step2":
        step2()
    else:
        main()
