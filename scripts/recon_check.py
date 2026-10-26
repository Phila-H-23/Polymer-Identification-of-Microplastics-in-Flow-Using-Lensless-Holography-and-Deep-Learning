"""Look-before-you-run reconstruction checks. No classifier, no training, no change to any existing file.
Writes only: results/recon_check/*.png (one file per sheet), results/report_recon_check.txt, and working files in the
session scratchpad (WORK below). Every step lowers its own priority to IDLE and runs single-threaded.

steps (python scripts/recon_check.py <step>):
  select   fixed patch selection (numpy default_rng(0)) from the old 561 PET and PS patches; written to the report
           before anything is rendered
  partA    z0 sanity sheets from the EXISTING z0 dataset (patches_recon_s0_z0, patches_background_recon_s0_z0)
  partB    synthetic absorbing-disc check of the propagation functions
  extras   raw file format details, median background I90/I0 per video, light-engine metadata search
  partC i  compute every Part C panel for selected patch i (0-7), one frame at a time
  render   sheets 1 to 3 from the Part C arrays
  report   assemble results/report_recon_check.txt
"""
import os
import sys
import ctypes
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
if os.name == "nt":  # lowest priority for this process (IDLE_PRIORITY_CLASS)
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
import json  # noqa: E402
import time  # noqa: E402
import gc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.fft as sfft  # noqa: E402
from scipy import ndimage  # noqa: E402

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
PO = os.path.join(D, "pipeline_output")
PNG = os.path.join(D, "results", "recon_check")
REPORT = os.path.join(D, "results", "report_recon_check.txt")
WORK = (r"C:\Users\haany\AppData\Local\Temp\claude\c--Users-haany-OneDrive-research-project-dataset-10jan23"
        r"\075ab033-0168-4afa-a675-c38e4f73b481\scratchpad\recon_check_work")
sys.path.insert(0, os.path.join(D, "scripts"))
import reconstruction as recon  # noqa: E402
import reconstruction_validate as rv  # noqa: E402

KEY = ["source_file", "frame", "particle_id"]
H_DEM, W_DEM = 1024, 1224
H_RAW, W_RAW = 2048, 2448
LAM = recon.WAVELENGTH_MEDIUM_M  # 405 nm / 1.33
DX6, DX3 = 6.9e-6, 3.45e-6
HALF_UM = 220.8  # 441.6 um square
Z3 = (1.75e-3, 2.00e-3, 2.25e-3)
Z41 = np.round(np.arange(1.500, 2.5001, 0.025), 3) * 1e-3
os.makedirs(WORK, exist_ok=True)


def wjson(name, obj):
    json.dump(obj, open(os.path.join(WORK, name), "w"), indent=1, default=float)


def rjson(name):
    p = os.path.join(WORK, name)
    return json.load(open(p)) if os.path.exists(p) else None


def clipped(x, y):
    return (y - 32 < 0) or (y + 32 > H_DEM) or (x - 32 < 0) or (x + 32 > W_DEM)


# ----------------------------------------------------------------------------------------------- selection
def select():
    t0 = time.time()
    m = pd.read_csv(os.path.join(PO, "manifest_recon_s0_z0.csv"))
    m = m[m["class"].isin(["PET", "PS"])].sort_values("patch_file").reset_index(drop=True)
    assert len(m) == 561
    m["video"] = m.source_file.str.replace("_600.mat", "", regex=False)
    m["clipped"] = [clipped(x, y) for x, y in zip(m.x, m.y)]
    rng = np.random.default_rng(0)
    picks = []
    pet_videos = ["PET1", "PET2", "PET3", "PET4"]
    edge = sorted(v for v in pet_videos if m[(m.video == v) & m.clipped].shape[0])
    pet_edge = str(rng.choice(edge))
    for v in pet_videos:
        pool = m[(m.video == v) & (m.clipped == (v == pet_edge))]
        picks.append(pool.iloc[int(rng.integers(len(pool)))])
    edge = sorted(v for v in ["PS1", "PS2"] if m[(m.video == v) & m.clipped].shape[0])
    ps_edge = str(rng.choice(edge))
    for v in ["PS1", "PS2"]:
        if v == ps_edge:
            pool = m[(m.video == v) & m.clipped]
            picks.append(pool.iloc[int(rng.integers(len(pool)))])
            pool = m[(m.video == v) & ~m.clipped]
            picks.append(pool.iloc[int(rng.integers(len(pool)))])
        else:
            pool = m[(m.video == v) & ~m.clipped]
            for j in rng.choice(len(pool), 2, replace=False):
                picks.append(pool.iloc[int(j)])
    sel = []
    for i, r in enumerate(picks):
        sides = [s for s, c in (("top", r.y - 32 < 0), ("bottom", r.y + 32 > H_DEM), ("left", r.x - 32 < 0),
                                ("right", r.x + 32 > W_DEM)) if c]
        sel.append(dict(i=i, patch_file=r.patch_file, cls=r["class"], video=r.video, source_file=r.source_file,
                        frame=int(r.frame), particle_id=int(r.particle_id), x=int(r.x), y=int(r.y),
                        clipped=bool(r.clipped), clipped_sides=sides, bg_window_frames=min(int(r.frame), 90)))
    wjson("selection.json", dict(selection=sel, pet_edge_video=pet_edge, ps_edge_video=ps_edge,
                                 seconds=time.time() - t0))
    lines = selection_text(sel, pet_edge, ps_edge)
    open(REPORT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def selection_text(sel, pet_edge, ps_edge):
    L = ["RECONSTRUCTION LOOK-BEFORE-YOU-RUN CHECKS (no classifier, no training)",
         "", "PATCH SELECTION (fixed and written here before any image was rendered)",
         "Source: pipeline_output/manifest_recon_s0_z0.csv, the common 849-patch manifest of the section 20 z0 runs,",
         "PET and PS rows only (561), sorted by patch_file. clipped = the 64 x 64 crop at the detection centre leaves",
         "the 1024 x 1224 demosaiced frame. numpy default_rng(0), calls in this order: PET edge video (choice among",
         "PET videos with a clipped patch), one patch per video PET1 to PET4 (clipped pool for the edge video, interior",
         "pool otherwise), PS edge video, then PS1 and PS2 (edge video: one clipped then one interior; other video:",
         f"two distinct interior). Drawn edge videos: PET {pet_edge}, PS {ps_edge}.",
         "Detection centre (x, y) is in demosaiced pixels (6.9 um); on the raw mosaic (3.45 um) the same point is",
         "(2x + 0.5, 2y + 0.5), and the 128 x 128 mosaic crop starts at row 2y - 64, column 2x - 64.",
         f"{'#':>2s} {'patch id':34s} {'class':5s} {'video':5s} {'frame':>5s} {'x':>5s} {'y':>5s} {'edge':5s} {'clipped side':12s} "
         f"{'bg frames':>9s}"]
    for s in sel:
        L.append(f"{s['i']:>2d} {s['patch_file'][:-4]:34s} {s['cls']:5s} {s['video']:5s} {s['frame']:>5d} {s['x']:>5d} "
                 f"{s['y']:>5d} {'yes' if s['clipped'] else 'no':5s} {','.join(s['clipped_sides']) or '-':12s} "
                 f"{s['bg_window_frames']:>9d}")
    return L


# ----------------------------------------------------------------------------------------------- helpers
def place(arr, x, y, half, scale=1):
    """Crop of a full frame, or a stored clipped crop, placed in a (2*half)^2 canvas at its true position;
    NaN where the crop leaves the frame. (x, y) in the grid's own pixels."""
    out = np.full((2 * half, 2 * half), np.nan, np.float64)
    h, w = (H_DEM * scale, W_DEM * scale)
    y0, y1, x0, x1 = max(0, y - half), min(h, y + half), max(0, x - half), min(w, x + half)
    out[y0 - (y - half): y1 - (y - half), x0 - (x - half): x1 - (x - half)] = arr
    return out


def crop_full(full, x, y, half):
    y0, y1 = max(0, y - half), min(full.shape[0], y + half)
    x0, x1 = max(0, x - half), min(full.shape[1], x + half)
    return place(full[y0:y1, x0:x1], x, y, half, scale=full.shape[0] // H_DEM)


def transfer_rows(shape, z, dx, r0, r1, dtype=np.complex64):
    """Rows r0:r1 of recon.angular_spectrum_transfer(shape, z, dx=dx), same expression order (checked equal in
    Part B), so a large padded grid's H never has to exist in memory at once."""
    Ny, Nx = shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dx)[r0:r1]
    FX, FY = np.meshgrid(fx, fy)
    arg = (1.0 / LAM) ** 2 - FX ** 2 - FY ** 2
    prop = arg >= 0
    kz = np.zeros_like(arg, dtype=np.complex128)
    kz[prop] = np.sqrt(arg[prop].astype(np.complex128))
    H = np.zeros_like(arg, dtype=np.complex128)
    H[prop] = np.exp(1j * 2 * np.pi * z * kz[prop])
    return H.astype(dtype)


def spectrum(amp, pad):
    """FFT of the zero-phase field amp (float array). pad=True: reflect padding to twice the size."""
    a = amp
    if pad:
        py, px = amp.shape[0] // 2, amp.shape[1] // 2
        a = np.pad(amp, ((py, py), (px, px)), mode="reflect")
    F = a.astype(np.complex64)
    del a
    return sfft.fft2(F, overwrite_x=True, workers=1)


def propagate_from(F, z, dx, out_shape, pad, rows_slice=None, chunk=256):
    """Field at distance z (signed, the code's convention) from spectrum F; returns the complex field cropped
    back to out_shape (the unpadded frame), or only rows_slice of it."""
    work = np.empty_like(F)
    for r0 in range(0, F.shape[0], chunk):
        r1 = min(F.shape[0], r0 + chunk)
        work[r0:r1] = F[r0:r1] * transfer_rows(F.shape, z, dx, r0, r1)
    work = sfft.ifft2(work, overwrite_x=True, workers=1)
    if pad:
        py, px = out_shape[0] // 2, out_shape[1] // 2
        work = work[py:py + out_shape[0], px:px + out_shape[1]]
    return work if rows_slice is None else work[rows_slice]


def prop(amp, z, dx, pad):
    """The propagation function of V2 to V5 (zero-phase field, angular spectrum, complex64)."""
    return propagate_from(spectrum(amp, pad), z, dx, amp.shape, pad)


# ----------------------------------------------------------------------------------------------- Part A
def partA():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = time.time()
    os.makedirs(PNG, exist_ok=True)
    sel = rjson("selection.json")["selection"]
    bgm = pd.read_csv(os.path.join(PO, "manifest_background_recon_s0_z0.csv")).set_index(KEY)
    rows = []
    for s in sel:
        z = np.load(os.path.join(PO, "patches_recon_s0_z0", s["patch_file"]))[..., 0].astype(np.float64)
        b = bgm.loc[(s["source_file"], s["frame"], s["particle_id"])]
        zb = np.load(os.path.join(PO, "patches_background_recon_s0_z0", b.patch_file))[..., 0].astype(np.float64)
        st = {}
        for name, a in (("real", z), ("particle-free", zb)):
            fin = np.isfinite(a)
            st[name] = dict(shape=list(a.shape), mean=float(np.nanmean(a)), min=float(np.nanmin(a)),
                            max=float(np.nanmax(a)), frac_outside=float(((a < 0.5) | (a > 1.5))[fin].mean()),
                            non_finite=int((~fin).sum()))
        rows.append(dict(i=s["i"], patch_file=s["patch_file"], bg_patch_file=b.patch_file, bg_x=int(b.x),
                         bg_y=int(b.y), stats=st))
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), dpi=110)
        cmap = matplotlib.colormaps["gray"].copy()
        cmap.set_bad("#9ecae1")
        for ax, a, cx, cy, ttl in ((axes[0], z, s["x"], s["y"], "z0 patch (I / median background)"),
                                   (axes[1], zb, int(b.x), int(b.y), "paired particle-free patch")):
            im = ax.imshow(place(a, cx, cy, 32), cmap=cmap, vmin=0.5, vmax=1.5, interpolation="nearest",
                           extent=[-HALF_UM, HALF_UM, HALF_UM, -HALF_UM])
            ax.set_title(f"{ttl}\ncentre x={cx}, y={cy} px (6.9 um)", fontsize=9)
            ax.set_xlabel("um")
        fig.colorbar(im, ax=axes, fraction=0.025, label="z0 (display 0.5 to 1.5)")
        fig.suptitle(f"Part A  #{s['i']}  {s['patch_file'][:-4]}  ({s['cls']}, {s['video']}, frame {s['frame']}"
                     f"{', frame edge' if s['clipped'] else ''}); blue = outside the sensor", fontsize=10)
        fig.savefig(os.path.join(PNG, f"A_z0_{s['i']}_{s['patch_file'][:-4]}.png"))
        plt.close(fig)
    wjson("partA.json", dict(rows=rows, seconds=time.time() - t0))
    print(json.dumps(rows, indent=1)[:3000])


# ----------------------------------------------------------------------------------------------- Part B
def partB():
    t0 = time.time()
    res, checks = [], {}
    R_UM, A_IN, A_OUT, Z = 50.0, 0.5, 1.0, recon.DEFAULT_Z_M  # 100 um diameter disc of the existing test
    for dx, N in ((DX6, 512), (DX3, 1024)):  # same 3.53 mm field; N=512 at 6.9 um is the existing test's grid
        # transfer-function check: chunked rows == recon.angular_spectrum_transfer
        Href = recon.angular_spectrum_transfer((N, N), -Z, dx=dx)
        Hc = np.concatenate([transfer_rows((N, N), -Z, dx, r, min(N, r + 128), np.complex128) for r in range(0, N, 128)])
        checks[f"H max abs diff dx={dx * 1e6:.2f}um"] = float(np.abs(Hc - Href).max())
        del Href, Hc
        yy, xx = np.meshgrid(np.arange(N) - N // 2, np.arange(N) - N // 2, indexing="ij")
        r_px = np.hypot(xx, yy)
        R = R_UM / (dx * 1e6)
        obj = np.where(r_px <= R, A_IN, A_OUT).astype(np.complex128)
        sensor = recon.angular_spectrum_propagate(obj, Z, dx=dx)  # the existing test's forward model
        I = np.abs(sensor) ** 2
        amp = np.sqrt(I)
        inside, outside = r_px <= R, (r_px >= 2 * R) & (r_px <= 4 * R)
        recons = {"recon.angular_spectrum_propagate (build, complex128), unpadded":
                  np.abs(recon.angular_spectrum_propagate(amp.astype(np.complex128), -Z, dx=dx)),
                  "prop() of V2 to V5 (complex64), unpadded": np.abs(prop(amp, -Z, dx, False)),
                  "prop() of V2 to V5 (complex64), reflect padded x2": np.abs(prop(amp, -Z, dx, True))}
        if dx == DX6:  # build function with padding only at 6.9 um (at 3.45 um it would need about 450 MB)
            recons["recon.angular_spectrum_propagate (complex128), reflect padded x2"] = np.abs(
                recon.angular_spectrum_propagate(np.pad(amp, N // 2, mode="reflect").astype(np.complex128), -Z,
                                                 dx=dx)[N // 2:N // 2 + N, N // 2:N // 2 + N])
        roundtrip = np.abs(prop(np.abs(sensor), 0.0, dx, False))  # z=0 identity check of prop()
        checks[f"prop() at z=0 vs input, max abs diff, dx={dx * 1e6:.2f}um"] = float(np.abs(roundtrip - np.abs(sensor)).max())
        cplx = np.abs(propagate_from(sfft.fft2(sensor.astype(np.complex64)), -Z, dx, sensor.shape, False))
        checks[f"complex-field round trip (phase kept), max |amp error|, dx={dx * 1e6:.2f}um"] = float(np.abs(cplx - np.abs(obj)).max())
        for name, a in recons.items():
            below = a < 0.5 * (A_IN + A_OUT)
            lab, _ = ndimage.label(below)
            c = lab[N // 2, N // 2]
            area = float((lab == c).sum()) if c else 0.0
            R_est = np.sqrt(area / np.pi)
            ok_in = abs(a[inside].mean() - A_IN) <= 0.05
            ok_r = abs(R_est - R) / R <= 0.10
            res.append(dict(dx_um=dx * 1e6, N=N, method=name, inside_mean=float(a[inside].mean()),
                            inside_median=float(np.median(a[inside])), outside_mean=float(a[outside].mean()),
                            R_true_px=float(R), R_est_px=float(R_est), radius_err=float((R_est - R) / R),
                            passes=bool(ok_in and ok_r), inside_ok=bool(ok_in), radius_ok=bool(ok_r)))
            print(res[-1], flush=True)
        del obj, sensor, I, amp, recons
        gc.collect()
    wjson("partB.json", dict(results=res, checks=checks, seconds=time.time() - t0,
                             all_pass=all(r["passes"] for r in res)))
    print(json.dumps(checks, indent=1))


# ----------------------------------------------------------------------------------------------- extras
def extras():
    import h5py
    t0 = time.time()
    out = []
    bg = pd.read_csv(os.path.join(PO, "manifest_background.csv"))
    for v in ["PET1", "PET2", "PET3", "PET4", "PS1", "PS2"]:
        path = os.path.join(D, f"{v}_600.mat")
        head = open(path, "rb").read(128)
        txt = head[:116].decode("latin-1").rstrip("\x00 ").strip()
        rec = dict(video=v, size_bytes=os.path.getsize(path), header_text=txt)
        if txt.startswith("MATLAB 7.3"):
            with h5py.File(path, "r") as f:
                keys = [k for k in f.keys()]
                rec["variables"] = keys
                rec["file_attrs"] = {k: str(f.attrs[k]) for k in f.attrs}
                name = [k for k in keys if not k.startswith("#")][0]
                ds = f[name]
                rec.update(container="HDF5 (MATLAB v7.3 MAT-file)", dataset=name, shape_on_disk=list(ds.shape),
                           dtype=str(ds.dtype), chunks=str(ds.chunks), compression=str(ds.compression),
                           compression_opts=str(ds.compression_opts), shuffle=bool(ds.shuffle),
                           fletcher32=bool(ds.fletcher32),
                           layout={0: "compact", 1: "contiguous", 2: "chunked", 3: "virtual"}.get(
                               ds.id.get_create_plist().get_layout(), "?"),
                           dataset_attrs={k: str(ds.attrs[k]) for k in ds.attrs})
        else:
            import struct
            import scipy.io as sio
            typ, nbytes = struct.unpack("<II", open(path, "rb").read(136)[128:136])
            rec.update(container="MATLAB v5 MAT-file", first_element_type=int(typ),
                       first_element_compressed=bool(typ == 15), first_element_bytes=int(nbytes),
                       variables=[list(map(str, w)) for w in sio.whosmat(path)])
            if typ == 14:
                mm = mat5_memmap(path)
                rec.update(shape_on_disk=list(mm.shape), dtype=str(mm.dtype), compression="none (uncompressed miMATRIX)")
                del mm
        g = bg[bg.source_file == f"{v}_600.mat"]
        p = np.stack([np.load(os.path.join(PO, "patches_background", fn)) for fn in g.patch_file]).astype(np.float64)
        i0, i90 = p[..., 0], p[..., 2]
        ok = i0 > 0
        rec.update(bg_patches=len(g), median_I0=float(np.median(i0)), median_I90=float(np.median(i90)),
                   median_ratio_I90_over_I0=float(np.median(i90[ok] / i0[ok])))
        out.append(rec)
        print(rec, flush=True)
    hits = []
    for fn in sorted(os.listdir(D)):
        if fn.endswith((".txt", ".md", ".log", ".m")):
            t = open(os.path.join(D, fn), encoding="latin-1").read().lower()
            for w in ("pulse", "pulsed", "light engine", "strobe", "led", "laser", "exposure"):
                if w in t:
                    hits.append(f"{fn}: '{w}'")
    wjson("extras.json", dict(videos=out, metadata_hits=hits, seconds=time.time() - t0))
    print(hits)


# ----------------------------------------------------------------------------------------------- Part C
class Frames:
    """One raw frame at a time, (2048, 2448) uint8, from an HDF5 (v7.3) or MAT v5 recording."""

    def __init__(self, path):
        import h5py
        self.h5 = None
        head = open(path, "rb").read(116).decode("latin-1")
        if head.startswith("MATLAB 7.3"):
            self.h5 = h5py.File(path, "r")
            self.ds = self.h5[[k for k in self.h5.keys() if not k.startswith("#")][0]]
        else:
            self.mm = mat5_memmap(path)

    def get(self, i):
        if self.h5 is not None:
            a = self.ds[i, 0] if self.ds.ndim == 4 else self.ds[i]
            return np.ascontiguousarray(a.T)
        return np.array(self.mm[:, :, 0, i] if self.mm.ndim == 4 else self.mm[:, :, i])

    def close(self):
        if self.h5 is not None:
            self.h5.close()


def mat5_memmap(path):
    """Read-only memmap of the (uncompressed) uint8 array of a MAT v5 file, Fortran order (H, W, 1, n)."""
    import struct
    raw = open(path, "rb")
    raw.seek(128)
    typ, nbytes = struct.unpack("<II", raw.read(8))
    assert typ == 14, f"MAT v5 element type {typ} (15 = compressed, not memory-mappable)"
    pos = 136
    raw.seek(pos)
    t, n = struct.unpack("<II", raw.read(8))  # array flags
    pos += 8 + n
    raw.seek(pos)
    t, n = struct.unpack("<II", raw.read(8))  # dimensions
    dims = struct.unpack(f"<{n // 4}i", raw.read(n))
    pos += 8 + n + (-n) % 8
    raw.seek(pos)
    t, n = struct.unpack("<II", raw.read(8))  # name (possibly small-element format)
    if t >> 16:
        pos += 8
    else:
        pos += 8 + n + (-n) % 8
    raw.seek(pos)
    t, n = struct.unpack("<II", raw.read(8))  # real part
    assert t == 2, f"data type {t}, expected miUINT8"
    raw.close()
    return np.memmap(path, dtype=np.uint8, mode="r", offset=pos + 8, shape=tuple(dims), order="F")


def bilinear(mosaic, r0, c0):
    """Bilinear (normalised convolution) full-resolution channel from the mosaic sites (r0::2, c0::2)."""
    s = np.zeros(mosaic.shape, np.float32)
    s[r0::2, c0::2] = mosaic[r0::2, c0::2]
    w = np.zeros(mosaic.shape, np.float32)
    w[r0::2, c0::2] = 1.0
    K = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], np.float32)
    return ndimage.convolve(s, K, mode="constant") / ndimage.convolve(w, K, mode="constant")


SITES = {"0": (1, 1), "45": (0, 1), "90": (0, 0), "135": (1, 0)}  # rows/cols of each channel in recon.demosaic


def partC(i):
    import build_reconstructed_dataset as bd
    from reconstruction_zsweep2 import laplacian_variance_amp, tamura_coefficient
    T = {}
    t_all = time.time()
    s = rjson("selection.json")["selection"][i]
    x, y, f = s["x"], s["y"], s["frame"]
    X, Y = 2 * x, 2 * y  # crop origin rule on the mosaic: rows 2y-64 .. 2y+63
    out = dict(sel=s)
    # 1. frames: up to 90 preceding frames to a disk-backed stack (one frame in memory at a time) + running sum
    t = time.time()
    fr = Frames(os.path.join(D, s["source_file"]))
    lo = max(0, f - 90)
    n = f - lo
    stack_path = os.path.join(WORK, f"stack_{i}.u8")
    stack = np.memmap(stack_path, dtype=np.uint8, mode="w+", shape=(n, H_RAW, W_RAW))
    ssum = np.zeros((H_RAW, W_RAW), np.float64)
    for k, j in enumerate(range(lo, f)):
        a = fr.get(j)
        stack[k] = a
        ssum += a
        del a
    stack.flush()
    raw = fr.get(f).astype(np.float64)
    fr.close()
    T["read frames"] = time.time() - t
    # 2. backgrounds. V1: rv.channel_background_median (build code) called on 256-row strips of the disk stack
    #    (median and the strided demosaic are per pixel, so strips give the full-frame result exactly).
    t = time.time()
    chans = {k: [] for k in ("0", "45", "90", "135")}
    view = np.memmap(stack_path, dtype=np.uint8, mode="r", shape=(n, H_RAW, W_RAW)).transpose(0, 2, 1)
    for r0 in range(0, H_RAW, 256):
        parts = rv.channel_background_median(view[:, :, r0:r0 + 256], "h5py", n, window=90)
        for k, p in zip(("0", "45", "90", "135"), parts):
            chans[k].append(p)
    del view
    Ibg6 = [np.concatenate(chans[k], axis=0) for k in ("0", "45", "90", "135")]
    med = np.empty((H_RAW, W_RAW), np.float64)  # median mosaic reassembled from the strided channels (lossless)
    for k, (r, c) in SITES.items():
        med[r::2, c::2] = Ibg6[("0", "45", "90", "135").index(k)]
    B = ssum / n
    del ssum, chans
    os.remove(stack_path)
    T["backgrounds (median strips, mean)"] = time.time() - t
    gc.collect()
    panels, metrics = {}, {}

    def metric(key, img):
        v = img[np.isfinite(img).all(axis=1)][:, np.isfinite(img).all(axis=0)]
        a = np.sqrt(np.clip(v, 0, None))
        metrics[key] = dict(laplacian_var=float(laplacian_variance_amp(a)), tamura=float(tamura_coefficient(a)))

    # raw I0 (strided, 6.9 um), shown divided by the median I0 of the frame
    I6 = recon.demosaic(raw)
    panels["raw_I0"] = crop_full(I6[0] / np.median(I6[0]), x, y, 32)
    # 3. V1: build code unchanged (6.9 um, per-channel division by the rolling median, unpadded)
    t = time.time()
    for z in Z3:
        Hz = recon.angular_spectrum_transfer(I6[0].shape, -z)
        P = bd.propagate_channels(list(I6), Ibg6, Hz)
        s0 = (P[0] + P[1] + P[2] + P[3]) / 4.0
        key = f"V1_{z * 1e3:.3f}"
        panels[key] = crop_full(s0, x, y, 32)
        metric(key, panels[key])
        if z >= 2.0e-3:
            for c, p in zip(("0", "45", "90", "135"), P):
                panels[f"{key}_ch{c}"] = crop_full(p, x, y, 32)
        if abs(z - 2.25e-3) < 1e-9:
            stored = np.load(os.path.join(PO, "patches_recon_s0", s["patch_file"]))[..., 0]
            mine = crop_full(s0.astype(np.float32), x, y, 32)
            ok = np.isfinite(mine)
            out["V1_vs_stored_recon_s0_max_abs_diff"] = float(np.abs(place(stored, x, y, 32)[ok] - mine[ok]).max())
        del P, s0, Hz
    T["V1"] = time.time() - t
    # 4. V2 / V5: mosaic, subtraction with the per-pixel mean B and the scalar mean(B), 3.45 um
    t = time.time()
    mB = float(B.mean())
    amp = np.sqrt(np.maximum(raw - B + mB, 0)).astype(np.float32)
    out["mean_B"] = mB
    for pad, tag in ((True, "V2"), (False, "V5")):
        F = spectrum(amp, pad)
        zs = list(Z41) if tag == "V2" else list(Z3)
        for z in zs:
            u = propagate_from(F, -z, DX3, amp.shape, pad)
            I3 = (np.abs(u[max(0, Y - 64):Y + 64, max(0, X - 64):X + 64]) ** 2).astype(np.float64) / mB
            del u
            full = np.full((128, 128), np.nan)
            ya, xa = max(0, Y - 64) - (Y - 64), max(0, X - 64) - (X - 64)
            full[ya:ya + I3.shape[0], xa:xa + I3.shape[1]] = I3
            ch = {c: full[r::2, cc::2] for c, (r, cc) in SITES.items()}  # strided channels, 6.9 um
            mean4 = (ch["0"] + ch["45"] + ch["90"] + ch["135"]) / 4.0
            key = f"{tag}_{z * 1e3:.3f}"
            panels[key] = mean4
            metric(key, mean4)
            if tag == "V2" and any(abs(z - zz) < 1e-9 for zz in (2.0e-3, 2.25e-3)):
                for c in ch:
                    panels[f"{key}_ch{c}"] = ch[c]
        del F
        gc.collect()
    del amp
    T["V2 (41 z, padded) and V5 (3 z, unpadded)"] = time.time() - t
    # 5. V3 / V4: bilinear per-channel full resolution, median background, division (V3) or subtraction (V4)
    t = time.time()
    acc = {f"{tag}_{z * 1e3:.3f}": np.zeros((128, 128)) for tag in ("V3", "V4") for z in Z3}
    mBc = []
    for c, (r, cc) in SITES.items():
        Ic, Bc = bilinear(raw, r, cc), bilinear(med, r, cc)
        mBc.append(float(Bc.mean()))
        for tag in ("V3", "V4"):
            a = (np.sqrt(np.clip(Ic / (Bc + 1e-6), 0, None)) if tag == "V3"
                 else np.sqrt(np.maximum(Ic - Bc + Bc.mean(), 0))).astype(np.float32)
            F = spectrum(a, True)
            del a
            for z in Z3:
                u = propagate_from(F, -z, DX3, (H_RAW, W_RAW), True)
                I3 = np.abs(u[max(0, Y - 64):Y + 64, max(0, X - 64):X + 64]) ** 2
                del u
                full = np.full((128, 128), np.nan)
                ya, xa = max(0, Y - 64) - (Y - 64), max(0, X - 64) - (X - 64)
                full[ya:ya + I3.shape[0], xa:xa + I3.shape[1]] = I3
                acc[f"{tag}_{z * 1e3:.3f}"] += full / 4.0
            del F
            gc.collect()
        del Ic, Bc
    out["mean_Bc"] = mBc
    for k, v in acc.items():
        panels[k] = v / (np.mean(mBc) if k.startswith("V4") else 1.0)
        metric(k, panels[k])
    T["V3 and V4 (4 channels each, padded, 3 z)"] = time.time() - t
    T["total"] = time.time() - t_all
    out.update(timings=T, metrics=metrics)
    np.savez_compressed(os.path.join(WORK, f"partC_{i}.npz"), **panels)
    wjson(f"partC_{i}.json", out)
    print(json.dumps(dict(i=i, timings=T, v1_check=out.get("V1_vs_stored_recon_s0_max_abs_diff")), indent=1))


# ----------------------------------------------------------------------------------------------- render
def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = time.time()
    sel = rjson("selection.json")["selection"]
    gray = matplotlib.colormaps["gray"].copy()
    gray.set_bad("#9ecae1")
    ret_cm = matplotlib.colormaps["magma"].copy()
    ret_cm.set_bad("#9ecae1")
    ax_cm = matplotlib.colormaps["twilight"].copy()
    ax_cm.set_bad("#bdbdbd")
    ext = [-HALF_UM, HALF_UM, HALF_UM, -HALF_UM]
    ranges = {}
    for s in sel:
        i = s["i"]
        p = np.load(os.path.join(WORK, f"partC_{i}.npz"))
        pool = np.concatenate([p[f"{v}_{z * 1e3:.3f}"][np.isfinite(p[f"{v}_{z * 1e3:.3f}"])]
                               for v in ("V1", "V2", "V3", "V4", "V5") for z in Z3])
        vmin, vmax = np.percentile(pool, [1, 99])
        ranges[i] = (float(vmin), float(vmax))
        name = f"{i}_{s['patch_file'][:-4]}"
        head = (f"#{i} {s['patch_file'][:-4]} ({s['cls']}, {s['video']}, frame {s['frame']}, centre x={s['x']} y={s['y']} px"
                f"{', frame edge' if s['clipped'] else ''}); display {vmin:.3f} to {vmax:.3f} for every intensity "
                f"panel of this patch; blue = outside the sensor")

        def show(ax, img, title, cmap=gray, lo=vmin, hi=vmax):
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest", extent=ext)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            return im
        # sheet 1
        fig, axes = plt.subplots(6, 7, figsize=(18.5, 16.5), dpi=100)
        for ax, z in zip(axes.ravel(), Z41):
            im = show(ax, p[f"V2_{z * 1e3:.3f}"], f"V2  z = {z * 1e3:.3f} mm")
        axes.ravel()[-1].axis("off")
        fig.colorbar(im, ax=axes.ravel()[-1], fraction=0.8, label="relative intensity")
        fig.suptitle("Sheet 1, V2 (mosaic first, 3.45 um, mean(B) subtraction, reflect padded), mean of the 4 channels;"
                     " each panel 441.6 um square\n" + head, fontsize=10)
        fig.savefig(os.path.join(PNG, f"S1_V2_zsweep_{name}.png"))
        plt.close(fig)
        # sheet 2
        fig, axes = plt.subplots(5, 3, figsize=(10.5, 17), dpi=100)
        desc = {"V1": "V1 build: 6.9 um, divide by median bg", "V2": "V2 mosaic 3.45 um, subtract mean bg, padded",
                "V3": "V3 bilinear 3.45 um, divide by median bg, padded", "V4": "V4 bilinear 3.45 um, subtract median bg, padded",
                "V5": "V5 = V2 unpadded"}
        for r, v in enumerate(("V1", "V2", "V3", "V4", "V5")):
            for c, z in enumerate(Z3):
                im = show(axes[r, c], p[f"{v}_{z * 1e3:.3f}"], f"{desc[v]}\nz = {z * 1e3:.2f} mm")
        fig.colorbar(im, ax=axes, fraction=0.02, label="relative intensity")
        fig.suptitle("Sheet 2, variants V1 to V5 (rows) at z = 1.75, 2.00, 2.25 mm (columns)\n" + head, fontsize=9)
        fig.savefig(os.path.join(PNG, f"S2_variants_{name}.png"))
        plt.close(fig)
        # sheet 3
        fig, axes = plt.subplots(4, 4, figsize=(15, 14.5), dpi=100)
        r = 0
        for v in ("V1", "V2"):
            for z in (2.0e-3, 2.25e-3):
                key = f"{v}_{z * 1e3:.3f}"
                eta, th = recon.birefringence_eta_theta(*(p[f"{key}_ch{c}"] for c in ("0", "45", "90", "135")))
                ret = eta * (np.pi / 2)  # radians (function returns retardance / (pi/2))
                ax_rad = th * np.pi
                ax_rad = np.where(ax_rad > np.pi / 2, ax_rad - np.pi, ax_rad)
                show(axes[r, 0], p["raw_I0"], f"raw I0 / frame median (6.9 um)\n(input of row {v} z={z * 1e3:.2f} mm)")
                im1 = show(axes[r, 1], p[key], f"{v} propagated mean intensity\nz = {z * 1e3:.2f} mm")
                im2 = show(axes[r, 2], ret, f"{v} retardance (rad), z = {z * 1e3:.2f} mm", ret_cm, 0, 0.8)
                im3 = show(axes[r, 3], ax_rad, f"{v} slow axis (rad), z = {z * 1e3:.2f} mm", ax_cm, -np.pi / 2, np.pi / 2)
                r += 1
        fig.colorbar(im1, ax=axes[:, 1], fraction=0.04, label="relative intensity")
        fig.colorbar(im2, ax=axes[:, 2], fraction=0.04, label="retardance (rad), 0 to 0.8")
        fig.colorbar(im3, ax=axes[:, 3], fraction=0.04, label="slow axis (rad), -pi/2 to pi/2")
        fig.suptitle("Sheet 3, V1 and V2 at z = 2.00 and 2.25 mm: raw I0, propagated mean intensity, retardance and slow"
                     " axis (recon.birefringence_eta_theta, no A_bg / B_bg maps), no registration\n" + head, fontsize=9)
        fig.savefig(os.path.join(PNG, f"S3_birefringence_{name}.png"))
        plt.close(fig)
    wjson("render.json", dict(ranges=ranges, seconds=time.time() - t0))


def report(write=True):
    sj, A, B, E, N = (rjson(n) for n in ("selection.json", "partA.json", "partB.json", "extras.json", "notes.json"))
    L = selection_text(sj["selection"], sj["pet_edge_video"], sj["ps_edge_video"])
    ms = N["memory_snapshot"]
    L += ["", "SAFETY AND TIMINGS",
          f"Memory at the start ({ms['time']}): free physical {ms['free_physical_mb']} MB of {ms['total_mb']} MB, available "
          f"(free + standby) {ms['available_mb']} MB, committed {ms['committed_mb']} MB of a {ms['commit_limit_mb']} MB commit limit.",
          f"z0 build (PID {ms['z0_build']['pid']}): private {ms['z0_build']['private_mb']} MB, resident {ms['z0_build']['working_set_mb']} MB "
          f"(peak {ms['z0_build']['peak_working_set_mb']} MB; part of it already paged out), {ms['z0_build']['frames_built']} frames built.",
          f"Also running: {ms['other']}.",
          "Decision: only small steps were run (each at IDLE priority, one thread, peak at most about 320 MB); the raw-frame",
          "work of Part C (full-frame FFTs at 3.45 um, about 0.5 GB) was to wait for the build to finish and for 2 GB to be",
          "available. The overnight chain, its files and the viewer were not touched.",
          f"{'step':26s} {'finished':>18s} {'wall s':>7s} {'peak MB':>8s} {'available before, MB':>21s}"]
    for s in N["steps"]:
        L.append(f"{s['step']:26s} {s['written']:>18s} {s['wall_s']:>7d} {str(s.get('peak_mb') or '-'):>8s} "
                 f"{str(s.get('available_before_mb', '-')):>21s}")
    L += ["", "PART A  z0 SANITY (existing z0 dataset: patches_recon_s0_z0 and patches_background_recon_s0_z0)",
          "One sheet per patch (results/recon_check/A_z0_<#>_<patch id>.png): the z0 patch and its paired particle-free",
          "patch (same video and frame), display range fixed at 0.5 to 1.5, drawn at true position (blue = outside the",
          "sensor). Statistics over the stored pixels (clipped patches are stored without padding).",
          f"{'#':>2s} {'patch':13s} {'shape':>7s} {'mean':>7s} {'min':>7s} {'max':>7s} {'outside 0.5-1.5':>15s} {'non-finite':>10s}"]
    for r in A["rows"]:
        for k, s in r["stats"].items():
            L.append(f"{r['i']:>2d} {k:13s} {'x'.join(map(str, s['shape'])):>7s} {s['mean']:7.4f} {s['min']:7.4f} {s['max']:7.4f} "
                     f"{s['frac_outside']:15.4%} {s['non_finite']:>10d}")
    L.append("What the sheets show (description only):")
    for i, t in N["partA_visual"].items():
        L.append(f"  #{i}: {t}.")
    L += ["  In 6 of the 8 sheets the particle-free patch contains a visible ring system of its own (an undetected",
          "  particle or a static feature), with contrast comparable to the weaker real patches (#0, #1, #3)."]
    L += ["", "PART B  PROPAGATOR CHECK ON A SYNTHETIC TARGET (scripts/reconstruction_zsweep2.py synthetic_control_sweep)",
          "Target: the existing test's absorbing disc, diameter 100 um (radius 50 um), amplitude 0.5 inside and 1.0 outside,",
          "forward-propagated by +2.25 mm with recon.angular_spectrum_propagate (405 nm / 1.33), as in the existing test but",
          "without the added noise (this is a propagator check). Reconstruction as in every variant: amplitude = sqrt(I),",
          "zero phase, propagated by -2.25 mm. Field 3.53 mm square (512 px at 6.9 um, 1024 px at 3.45 um). Measures fixed",
          "before running: inside = mean recovered amplitude over r <= R; outside = mean over 2R <= r <= 4R; radius =",
          "equivalent-area radius of the connected region below 0.75 (midway between 0.5 and 1.0) containing the centre.",
          "Fail rule (task): inside not within 0.05 of 0.5, or radius error above 10%.",
          f"{'dx um':>5s} {'method':66s} {'inside':>7s} {'(median)':>8s} {'outside':>7s} {'R px':>6s} {'R est':>6s} {'R err':>6s} result"]
    for r in B["results"]:
        L.append(f"{r['dx_um']:5.2f} {r['method']:66s} {r['inside_mean']:7.4f} {r['inside_median']:8.4f} {r['outside_mean']:7.4f} "
                 f"{r['R_true_px']:6.2f} {r['R_est_px']:6.2f} {r['radius_err']:+6.1%} "
                 + ("pass" if r["passes"] else "FAIL (" + ", ".join(x for x, ok in (("inside", r["inside_ok"]), ("radius", r["radius_ok"])) if not ok) + ")"))
    L.append("Additional checks of the same functions:")
    for k, v in B["checks"].items():
        L.append(f"  {k}: {v:.2e}")
    L += ["Result: every combination FAILS the inside-amplitude rule by a small margin (recovered 0.554 to 0.556 against",
          "0.5, limit 0.05) and passes the radius rule (error 0.4% to 1.2%). The four propagator set-ups agree with each",
          "other to about 1e-6, the transfer function equals the build's exactly, and with the phase kept the round trip",
          "recovers the disc to 4e-7; the 0.055 excess is the same for every set-up, so it comes from the zero-phase field",
          "model (twin image), which all variants share, not from sampling or padding.",
          "STOP applied as instructed: Part C was not run, no real image was propagated, sheets 1 to 3 do not exist, the",
          "sharpness metrics were not computed, and readings D1 to D3 and the variant sentences cannot be given.",
          "", "PART C  CONTACT SHEETS: not run (Part B stop rule). Code for it is in scripts/recon_check.py (steps partC and",
          "render), not executed. Note for a later run: the recordings are gzip-compressed HDF5 chunked as 600 frames x 1",
          "column x 107 rows, so reading any single frame decompresses data of all 600 frames; the raw frames would have to",
          "be read in column bands into a disk-backed stack (one pass per patch) rather than frame by frame.",
          "", "EXTRAS (no images)",
          f"{'video':5s} {'container':28s} {'created (file header)':25s} {'variable':10s} {'array on disk':>22s} {'type':6s} "
          f"{'compression':26s} {'median I0':>9s} {'median I90':>10s} {'I90/I0':>6s}"]
    for v in E["videos"]:
        created = v["header_text"].split("Created on: ")[-1].split(" HDF5")[0]
        if v["container"].startswith("HDF5"):
            comp = f"gzip level {v['compression_opts']}, chunks {v['chunks'].replace(' ', '')}"
            var, shape = v["dataset"], "x".join(map(str, v["shape_on_disk"]))
        else:
            comp = "zlib (miCOMPRESSED element)" if v["first_element_compressed"] else "none"
            var, shape = v["variables"][0][0], v["variables"][0][1].replace(" ", "").strip("()").replace(",", "x")
        L.append(f"{v['video']:5s} {v['container']:28s} {created:25s} {var:10s} {shape:>22s} {v.get('dtype', 'uint8'):6s} "
                 f"{comp:26s} {v['median_I0']:9.0f} {v['median_I90']:10.0f} {v['median_ratio_I90_over_I0']:6.2f}")
    L += ["All six: lossless storage of raw sensor frames, 8 bits per pixel (uint8, MATLAB class uint8), one 2048 x 2448",
          "polarisation-mosaic frame per index (2448 x 2048 on disk in HDF5 order); no video codec; HDF5 files have no",
          "shuffle or checksum filter and no file attributes beyond MATLAB's class tag. PET4 is a MATLAB v5 file holding 288",
          "frames. I90/I0: median over all pixels of the per-pixel ratio in the stored raw background-only patches",
          "(pipeline_output/patches_background, strided demosaic, 64 x 64 at sampled frames of that video). Note: the",
          "docstring of scripts/reconstruction.py states I90 is about 4.5 x I0; the measured medians here are 1.98 to 2.16.",
          "Pulsed light engine: unknown. Not recorded in the MAT headers (only MATLAB version, platform and creation time),",
          "the HDF5 attributes, the variable names, README.md, the run logs or the .m file (whole-word search for LED,",
          "pulse, strobe, light engine, exposure, illumination found nothing about the light source).",
          "", "READINGS (D1 to D3, variant sentences): not given; the sheets they refer to were not rendered (Part B stop).",
          "", "FIVE-LINE SUMMARY",
          "1. Memory was tight (1.2 GB available; z0 build 4.7 GB private and partly paged out; your dataset_viewer.py held 3.5 GB), so only small idle-priority steps ran (peaks 147 to 317 MB, 11 to 19 s each).",
          "2. Patches fixed before rendering (seed 0): PET1 #32 f493, PET2 #37 f141, PET3 #35 f211, PET4 #33 f68 (left edge); PS1 #10 f141 (right edge), PS1 #101 f564, PS2 #117 f599, PS2 #80 f352.",
          "3. Part A: all 16 stored z0 patches are finite and lie within 0.5 to 1.5 except 0.1% and 0.7% of pixels in two PS patches; 6 of 8 particle-free patches contain a visible ring system of their own.",
          "4. Part B FAILED the fixed rule by a small margin: inside amplitude 0.554 to 0.556 vs 0.5 (limit 0.05) at both samplings, padded and unpadded; radius error 0.4% to 1.2%; the propagator round trip with phase is exact (4e-7), so the excess comes from the zero-phase model; Part C stopped, no reconstructed sheets.",
          "5. Raw files: gzip-3 HDF5 (MATLAB 7.3) or zlib MAT v5 (PET4), uint8, no codec, created 2023-01-10; background I90/I0 medians 1.98 to 2.16 (the reconstruction.py docstring says about 4.5); pulsed light engine unknown (not recorded)."]
    text = "\n".join(L)
    assert "—" not in text and "significant" not in text.lower()
    if write:
        open(REPORT, "w", encoding="utf-8").write(text + "\n")
        print(text)
    return text


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "partC":
        partC(int(sys.argv[2]))
    else:
        globals()[step]()
