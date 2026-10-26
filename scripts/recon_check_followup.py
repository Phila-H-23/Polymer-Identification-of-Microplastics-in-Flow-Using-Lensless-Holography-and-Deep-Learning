"""Follow-up to the look-before-you-run checks (no classifier, no training). New files only: results/recon_check/*.png,
results/report_recon_check.txt and working files in the session scratchpad. Runs in Windows background mode (idle CPU
priority, low I/O and memory priority), one thread.

Order (one process, python scripts/recon_check_followup.py all):
  partB2    Part B under the changed gate (exact-phase round trip < 1e-5 and radius error <= 10% in every combination),
            plus the 2026-09-30 disc (radius 20 px = 138 um) through the same code
  reads     one band-read pass per PET / PS recording (HDF5: column bands of all needed frames; MAT v5 zlib: one
            streamed pass): the Part C window of every selected patch (up to 90 preceding frames + the frame) and 30
            evenly spaced frames, into disk-backed stacks
  partC     every Part C panel for the 8 fixed patches from those stacks, then sheets 1 to 3 (recon_check.render)
  ratios    I90 / I0 reconciliation per recording, PET4 vs PET3, frames_bg_30.mat
  newvids   PE500FRAMES and PP500FRAMES: metadata, 20 evenly spaced frames, detections, I90 / I0, background levels
Before every heavy step and inside every loop it waits while the z0 build or the chain's collisions step is running,
while the chain's first (timed) z0 run is running, during the chain's bgcorrected build, or when available RAM is
below 2 GB plus the step's own need. It never signals, stops or delays the chain.
"""
import os
import re
import sys
import json
import time
import gc
import zlib
import struct
import ctypes

sys.path.insert(0, os.path.dirname(__file__))
import recon_check as rc  # noqa: E402  (sets idle priority and one thread at import)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.fft as sfft  # noqa: E402
from scipy import ndimage  # noqa: E402

D, PO, WORK = rc.D, rc.PO, rc.WORK
recon, rv = rc.recon, rc.rv
H_RAW, W_RAW = rc.H_RAW, rc.W_RAW
RES = os.path.join(D, "results")
CHAIN_Q = os.path.join(RES, "logs_expanded_PETPS_exp_queue.txt")
BUILD_LOG = os.path.join(RES, "expanded_PETPS_exp_build_z0.log")
FIRST_RUN_LOG = os.path.join(RES, "logs_expanded_PETPS_exp", "real_tc_PETvsPS_z0_exp_s0.log")
PROJ = os.path.join(RES, "expanded_PETPS_exp", "part4_projection.json")
STATUS = os.path.join(WORK, "followup_status.log")
WAITS = {}


# ---------------------------------------------------------------------------------------------------- resources
class _MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def avail_mb():
    m = _MS()
    m.dwLength = ctypes.sizeof(_MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.ullAvailPhys / 2 ** 20)


def background_mode():
    h = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.kernel32.SetPriorityClass(h, 0x40)  # IDLE_PRIORITY_CLASS
    ctypes.windll.kernel32.SetPriorityClass(h, 0x00100000)  # PROCESS_MODE_BACKGROUND_BEGIN (low I/O and memory priority)


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(STATUS, "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


def chain_state():
    rd = lambda p: open(p, encoding="utf-8", errors="ignore").read() if os.path.exists(p) else ""
    b, q = rd(BUILD_LOG), rd(CHAIN_Q)
    if "=== Summary" not in b and "Traceback" not in b:
        return "z0 build running"
    if "collisions done" not in q and "collisions FAILED" not in q and "FAILED" not in q:
        return "chain collisions step (video loads) running or about to start"
    if os.path.exists(FIRST_RUN_LOG) and not os.path.exists(PROJ):
        return "chain's first z0 run (timed, decides the seeds) running"
    if "=== bgcorrected build" in q and not re.search(r"bgcorrected build (checked|FAILED)", q):
        blog = rd(os.path.join(RES, "expanded_PETPS_exp_build_bgcorr.log"))
        if "HOLD" not in blog or "released" in blog:  # on hold (overnight step 6) = not running
            return "chain's bgcorrected build running"
    return None


def gate(need_mb, what):
    """Waits while the chain is in a step listed in chain_state(), or while starting this step would leave the
    chain less than 1 GB of available RAM (the 2 GB rule of the task applied while the z0 build ran, and nothing
    runs before the build has finished)."""
    t0, said = time.time(), None
    while True:
        st, a = chain_state(), avail_mb()
        if st is None and a >= 1000 + need_mb:
            break
        msg = st or f"available RAM {a} MB < 1000 + {need_mb} MB"
        if msg != said:
            log(f"waiting before '{what}': {msg}")
            said = msg
        time.sleep(30)
    w = time.time() - t0
    if w > 1:
        WAITS[what] = WAITS.get(what, 0) + w
        log(f"resumed '{what}' after {w / 60:.1f} min (available {avail_mb()} MB)")


# ---------------------------------------------------------------------------------------------------- Part B
def partB2():
    t0 = time.time()
    Z, A_IN, A_OUT = recon.DEFAULT_Z_M, 0.5, 1.0
    rows = []
    discs = [("50 um radius (this task)", 50.0), ("138 um radius (2026-09-30 run, 20 px at 6.9 um)", 138.0)]
    for dname, R_UM in discs:
        for dx, N in ((rc.DX6, 512), (rc.DX3, 1024)):
            if R_UM > 100 and dx == rc.DX3:
                continue
            yy, xx = np.meshgrid(np.arange(N) - N // 2, np.arange(N) - N // 2, indexing="ij")
            r_px = np.hypot(xx, yy)
            R = R_UM / (dx * 1e6)
            obj = np.where(r_px <= R, A_IN, A_OUT).astype(np.complex128)
            sensor = recon.angular_spectrum_propagate(obj, Z, dx=dx)
            amp0 = np.abs(sensor)  # zero-phase field
            inside, outside, region = r_px <= R, (r_px >= 2 * R) & (r_px <= 4 * R), r_px <= 4 * R

            def methods(field):
                out = {"build recon.angular_spectrum_propagate, complex128, unpadded":
                       recon.angular_spectrum_propagate(np.asarray(field, np.complex128), -Z, dx=dx),
                       "prop() of V2 to V5, complex64, unpadded":
                       rc.propagate_from(rc.spectrum(field, False), -Z, dx, field.shape, False),
                       "prop() of V2 to V5, complex64, reflect padded x2":
                       rc.propagate_from(rc.spectrum(field, True), -Z, dx, field.shape, True)}
                if dx == rc.DX6:
                    p = np.pad(np.asarray(field, np.complex128), N // 2, mode="reflect")
                    out["build recon.angular_spectrum_propagate, complex128, reflect padded x2"] = \
                        recon.angular_spectrum_propagate(p, -Z, dx=dx)[N // 2:N // 2 + N, N // 2:N // 2 + N]
                return out
            zero = {k: np.abs(v) for k, v in methods(amp0).items()}
            exact = {k: np.abs(v) for k, v in methods(sensor).items()}
            for k in zero:
                a = zero[k]
                lab, _ = ndimage.label(a < 0.75)
                c = lab[N // 2, N // 2]
                R_est = np.sqrt(float((lab == c).sum()) / np.pi) if c else 0.0
                e = exact[k]
                labe, _ = ndimage.label(e < 0.75)
                ce = labe[N // 2, N // 2]
                R_exact = np.sqrt(float((labe == ce).sum()) / np.pi) if ce else 0.0
                err_region = float(np.abs(e - np.abs(obj))[region].max())
                rows.append(dict(disc=dname, dx_um=dx * 1e6, N=N, method=k, R_true_px=float(R),
                                 inside_zero_phase=float(a[inside].mean()), outside_zero_phase=float(a[outside].mean()),
                                 R_est_zero_phase=float(R_est), radius_err_zero_phase=float((R_est - R) / R),
                                 exact_round_trip_err_region=err_region,
                                 exact_round_trip_err_full=float(np.abs(e - np.abs(obj)).max()),
                                 R_est_exact=float(R_exact), radius_err_exact=float((R_exact - R) / R),
                                 old_gate_pass=bool(abs(a[inside].mean() - A_IN) <= 0.05 and abs(R_est - R) / R <= 0.10),
                                 new_gate_pass=bool(err_region < 1e-5 and abs(R_est - R) / R <= 0.10)))
                log(f"partB2 {dname} dx={dx * 1e6:.2f} {k}: inside {rows[-1]['inside_zero_phase']:.4f}, "
                    f"exact err {err_region:.2e}, R err {rows[-1]['radius_err_zero_phase']:+.3f}")
            del obj, sensor, amp0, zero, exact
            gc.collect()
    this = [r for r in rows if r["disc"].startswith("50")]
    rc.wjson("partB2.json", dict(rows=rows, seconds=time.time() - t0,
                                 new_gate_all_pass=all(r["new_gate_pass"] for r in this),
                                 old_gate_all_pass=all(r["old_gate_pass"] for r in this)))


# ---------------------------------------------------------------------------------------------------- reading
def mat5_stream(path, wanted, sink):
    """One streamed zlib pass through a compressed MAT v5 file; sink(frame_index, (H, W) uint8) for every wanted
    frame, in order. Frame k occupies bytes [k*H*W, (k+1)*H*W) of the uint8 array (Fortran order)."""
    wanted = sorted(set(wanted))
    f = open(path, "rb")
    f.seek(128)
    typ, nbytes = struct.unpack("<II", f.read(8))
    assert typ == 15, typ
    d = zlib.decompressobj()
    head, data0, dims = b"", None, None
    pos = 0  # bytes of array data consumed
    cur, cur_k = None, None
    wi = 0
    left = nbytes
    while left > 0 and wi < len(wanted):
        comp = f.read(min(1 << 22, left))
        left -= len(comp)
        out = d.decompress(comp)
        if data0 is None:
            head += out
            if len(head) < 256:
                continue
            # miMATRIX tag, flags, dims, name, real part tag
            p = 8
            t, n = struct.unpack("<II", head[p:p + 8]); p += 8 + n
            t, n = struct.unpack("<II", head[p:p + 8]); dims = struct.unpack(f"<{n // 4}i", head[p + 8:p + 8 + n])
            p += 8 + n + (-n) % 8
            t, n = struct.unpack("<II", head[p:p + 8])
            p += 8 if t >> 16 else 8 + n + (-n) % 8
            t, n = struct.unpack("<II", head[p:p + 8])
            assert t == 2, t
            data0 = p + 8
            out = head[data0:]
        HW = dims[0] * dims[1]
        mv = memoryview(out)
        i = 0
        while i < len(mv) and wi < len(wanted):
            k = (pos + i) // HW
            off = (pos + i) % HW
            take = min(HW - off, len(mv) - i)
            if k == wanted[wi]:
                if cur is None:
                    cur, cur_k = np.empty(HW, np.uint8), k
                cur[off:off + take] = np.frombuffer(mv[i:i + take], np.uint8)
                if off + take == HW:
                    sink(k, cur.reshape(dims[1], dims[0]).T.copy())
                    cur = None
                    wi += 1
            i += take
        pos += len(mv)
    f.close()
    return dims


def read_pass(tag, path, sets):
    """sets: {name: sorted frame list}. Returns {name: memmap path}; stacks are (n, 2048, 2448) uint8 on disk."""
    t0 = time.time()
    allf = sorted(set().union(*sets.values()))
    idx = {f: j for j, f in enumerate(allf)}
    paths = {name: os.path.join(WORK, f"stack_{tag}_{name}.u8") for name in sets}
    mms = {name: np.memmap(paths[name], np.uint8, "w+", shape=(len(fr), H_RAW, W_RAW)) for name, fr in sets.items()}
    where = {name: {f: k for k, f in enumerate(fr)} for name, fr in sets.items()}
    head = open(path, "rb").read(116).decode("latin-1")
    if head.startswith("MATLAB 7.3"):
        import h5py
        with h5py.File(path, "r") as h:
            ds = h[[k for k in h.keys() if not k.startswith("#")][0]]
            for c0 in range(0, W_RAW, 64):  # chunks hold all frames of 1 column x 107 rows: full frame range per band
                gate(200, f"band read {tag}")
                c1 = min(W_RAW, c0 + 64)
                band = ds[:, 0, c0:c1, :]  # (N, w, H), each chunk decompressed once
                for name, fr in sets.items():
                    for f, k in where[name].items():
                        mms[name][k, :, c0:c1] = band[f].T
                del band
    else:
        gate(150, f"zlib stream {tag}")

        def sink(k, frame):
            for name in sets:
                if k in where[name]:
                    mms[name][where[name][k]] = frame
        mat5_stream(path, allf, sink)
    for m in mms.values():
        m.flush()
    del mms
    log(f"read pass {tag}: {len(allf)} frames in {time.time() - t0:.0f} s")
    return paths, time.time() - t0


def stack(path, n):
    return np.memmap(path, np.uint8, "r", shape=(n, H_RAW, W_RAW))


def median_mosaic(path, n, rows=256):
    """Per-pixel median over the n frames of a disk stack, computed in row strips."""
    s = stack(path, n)
    out = np.empty((H_RAW, W_RAW), np.float64)
    for r0 in range(0, H_RAW, rows):
        out[r0:r0 + rows] = np.median(np.asarray(s[:, r0:r0 + rows, :]), axis=0)
    del s
    return out


# ---------------------------------------------------------------------------------------------------- ratios
def chan_stats(mos, label):
    """(a) ratio of channel medians, (b) median of the per-pixel ratio, Stokes q as in to_stokes / q_ref."""
    I0, I45, I90, I135 = (np.asarray(c, np.float64) for c in recon.demosaic(mos))
    ok = I0 > 0
    S = (I0 + I45 + I90 + I135) / 4.0 / 255.0
    q = (I0 - I90) / 255.0 / (S + 0.05)
    return dict(label=label, med_I0=float(np.median(I0)), med_I45=float(np.median(I45)), med_I90=float(np.median(I90)),
                med_I135=float(np.median(I135)), ratio_of_medians=float(np.median(I90) / np.median(I0)),
                median_of_ratio=float(np.median(I90[ok] / I0[ok])), mean_I0=float(I0.mean()), mean_I90=float(I90.mean()),
                ratio_of_means=float(I90.mean() / I0.mean()), q_stokes_median=float(np.median(q)),
                q_stokes_mean=float(q.mean()))


def level_stats(path, n, med):
    """Per channel: mean and spatial std of the median background, temporal std of the per-frame channel means."""
    s = stack(path, n)
    per_frame = np.array([[float(np.asarray(c, np.float64).mean()) for c in recon.demosaic(np.asarray(s[k]))]
                          for k in range(n)])
    del s
    out = {}
    for j, (name, c) in enumerate(zip(("I0", "I45", "I90", "I135"), recon.demosaic(med))):
        out[name] = dict(mean=float(c.mean()), spatial_std=float(c.std()), temporal_std_of_frame_means=float(per_frame[:, j].std()),
                         p5=float(np.percentile(c, 5)), p95=float(np.percentile(c, 95)))
    return out


# ---------------------------------------------------------------------------------------------------- Part C
def partC_from_stack(i, win_path, n_win):
    import build_reconstructed_dataset as bd
    from reconstruction_zsweep2 import laplacian_variance_amp, tamura_coefficient
    T, t_all = {}, time.time()
    s = rc.rjson("selection.json")["selection"][i]
    x, y = s["x"], s["y"]
    X, Y = 2 * x, 2 * y
    n = n_win - 1  # preceding frames; the last stack frame is the patch's own frame
    out = dict(sel=s, n_bg_frames=n)
    gate(400, f"Part C #{i} backgrounds")
    t = time.time()
    st = stack(win_path, n_win)
    view = st[:n].transpose(0, 2, 1)
    chans = {k: [] for k in ("0", "45", "90", "135")}
    for r0 in range(0, H_RAW, 256):  # build code, called unchanged on row strips of the disk stack
        for k, p in zip(("0", "45", "90", "135"), rv.channel_background_median(view[:, :, r0:r0 + 256], "h5py", n, window=90)):
            chans[k].append(p)
    Ibg6 = [np.concatenate(chans[k], axis=0) for k in ("0", "45", "90", "135")]
    med = np.empty((H_RAW, W_RAW), np.float64)
    for k, (r, c) in rc.SITES.items():
        med[r::2, c::2] = Ibg6[("0", "45", "90", "135").index(k)]
    B = np.zeros((H_RAW, W_RAW), np.float64)
    for r0 in range(0, H_RAW, 256):
        B[r0:r0 + 256] = np.asarray(st[:n, r0:r0 + 256, :], np.float64).mean(axis=0)
    raw = np.asarray(st[n], np.float64)
    del st, view, chans
    T["backgrounds"] = time.time() - t
    # item 3 (c): channel ratios of this frame raw, after the bgcorrected build's correction, after the z0 division
    corr = np.clip(raw - B + B.mean(), 0, 255).astype(np.uint8)  # build_bgcorrected_dataset.correct_frame
    I6 = recon.demosaic(raw)
    z0 = [I / (b + 1e-6) for I, b in zip(I6, Ibg6)]
    zm = np.empty((H_RAW, W_RAW))
    for k, (r, c) in rc.SITES.items():
        zm[r::2, c::2] = z0[("0", "45", "90", "135").index(k)]
    out["ratios_frame"] = dict(raw=rc_chan(raw), bgcorrected=rc_chan(corr.astype(np.float64)), z0_divided=rc_chan(zm, scale=False))
    del corr, z0, zm
    panels, metrics = {}, {}

    def metric(key, img):
        v = img[np.isfinite(img).all(axis=1)][:, np.isfinite(img).all(axis=0)]
        a = np.sqrt(np.clip(v, 0, None))
        metrics[key] = dict(laplacian_var=float(laplacian_variance_amp(a)), tamura=float(tamura_coefficient(a)))

    panels["raw_I0"] = rc.crop_full(I6[0] / np.median(I6[0]), x, y, 32)
    # V1: build code unchanged
    t = time.time()
    for z in rc.Z3:
        gate(300, f"Part C #{i} V1")
        Hz = recon.angular_spectrum_transfer(I6[0].shape, -z)
        P = bd.propagate_channels(list(I6), Ibg6, Hz)
        s0 = (P[0] + P[1] + P[2] + P[3]) / 4.0
        key = f"V1_{z * 1e3:.3f}"
        panels[key] = rc.crop_full(s0, x, y, 32)
        metric(key, panels[key])
        if z >= 2.0e-3:
            for c, p in zip(("0", "45", "90", "135"), P):
                panels[f"{key}_ch{c}"] = rc.crop_full(p, x, y, 32)
        if abs(z - 2.25e-3) < 1e-9:
            stored = np.load(os.path.join(PO, "patches_recon_s0", s["patch_file"]))[..., 0]
            mine = rc.crop_full(s0.astype(np.float32), x, y, 32)
            ok = np.isfinite(mine)
            out["V1_vs_stored_recon_s0_max_abs_diff"] = float(np.abs(rc.place(stored, x, y, 32)[ok] - mine[ok]).max())
        del P, s0, Hz
    T["V1"] = time.time() - t
    # V2 / V5
    t = time.time()
    mB = float(B.mean())
    amp = np.sqrt(np.maximum(raw - B + mB, 0)).astype(np.float32)
    out["mean_B"] = mB
    for pad, tag in ((True, "V2"), (False, "V5")):
        gate(700, f"Part C #{i} {tag}")
        F = rc.spectrum(amp, pad)
        for z in (list(rc.Z41) if tag == "V2" else list(rc.Z3)):
            gate(700, f"Part C #{i} {tag}")
            u = rc.propagate_from(F, -z, rc.DX3, amp.shape, pad)
            I3 = (np.abs(u[max(0, Y - 64):Y + 64, max(0, X - 64):X + 64]) ** 2).astype(np.float64) / mB
            del u
            full = np.full((128, 128), np.nan)
            ya, xa = max(0, Y - 64) - (Y - 64), max(0, X - 64) - (X - 64)
            full[ya:ya + I3.shape[0], xa:xa + I3.shape[1]] = I3
            ch = {c: full[r::2, cc::2] for c, (r, cc) in rc.SITES.items()}
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
    T["V2 (41 z) and V5 (3 z)"] = time.time() - t
    # V3 / V4
    t = time.time()
    acc = {f"{tag}_{z * 1e3:.3f}": np.zeros((128, 128)) for tag in ("V3", "V4") for z in rc.Z3}
    mBc = []
    for c, (r, cc) in rc.SITES.items():
        Ic, Bc = rc.bilinear(raw, r, cc), rc.bilinear(med, r, cc)
        mBc.append(float(Bc.mean()))
        for tag in ("V3", "V4"):
            gate(700, f"Part C #{i} {tag}")
            a = (np.sqrt(np.clip(Ic / (Bc + 1e-6), 0, None)) if tag == "V3"
                 else np.sqrt(np.maximum(Ic - Bc + Bc.mean(), 0))).astype(np.float32)
            F = rc.spectrum(a, True)
            del a
            for z in rc.Z3:
                gate(700, f"Part C #{i} {tag}")
                u = rc.propagate_from(F, -z, rc.DX3, (H_RAW, W_RAW), True)
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
    T["V3 and V4"] = time.time() - t
    T["total"] = time.time() - t_all
    out.update(timings=T, metrics=metrics)
    np.savez_compressed(os.path.join(WORK, f"partC_{i}.npz"), **panels)
    rc.wjson(f"partC_{i}.json", out)
    log(f"Part C #{i} done in {T['total'] / 60:.1f} min; V1 vs stored recon_s0 max diff "
        f"{out.get('V1_vs_stored_recon_s0_max_abs_diff')}")
    del raw, B, med, Ibg6, I6
    gc.collect()


def rc_chan(mos, scale=True):
    I0, I45, I90, I135 = (np.asarray(c, np.float64) for c in recon.demosaic(mos))
    ok = I0 > 0
    return dict(med_I0=float(np.median(I0)), med_I90=float(np.median(I90)), ratio_of_medians=float(np.median(I90) / np.median(I0)),
                median_of_ratio=float(np.median(I90[ok] / I0[ok])))


# ---------------------------------------------------------------------------------------------------- driver
def recordings():
    sel = rc.rjson("selection.json")["selection"]
    out = {}
    for v in ("PET1", "PET2", "PET3", "PET4", "PS1", "PS2"):
        out[v] = [s for s in sel if s["video"] == v]
    return out


def n_frames(path):
    head = open(path, "rb").read(116).decode("latin-1")
    if head.startswith("MATLAB 7.3"):
        import h5py
        with h5py.File(path, "r") as h:
            return h[[k for k in h.keys() if not k.startswith("#")][0]].shape[0]
    import scipy.io as sio
    return int(sio.whosmat(path)[0][1][-1])


def run_reads_partC_ratios():
    ratios = rc.rjson("ratios.json") or {}
    for v, sel in recordings().items():
        if v in ratios and all(os.path.exists(os.path.join(WORK, f"partC_{s['i']}.npz")) for s in sel):
            continue
        path = os.path.join(D, f"{v}_600.mat")
        N = n_frames(path)
        even = sorted(set(np.round(np.linspace(0, N - 1, 30)).astype(int).tolist()))
        sets = {"even30": even}
        for s in sel:
            sets[f"win{s['i']}"] = list(range(max(0, s["frame"] - 90), s["frame"] + 1))
        paths, secs = read_pass(v, path, sets)
        gate(300, f"ratios {v}")
        med = median_mosaic(paths["even30"], len(even))
        ratios[v] = dict(n_frames=N, frames_used=even, read_seconds=secs, raw=chan_stats(med, "raw uint8, median of 30 frames"),
                         levels=level_stats(paths["even30"], len(even), med))
        del med
        os.remove(paths["even30"])
        rc.wjson("ratios.json", ratios)
        for s in sel:
            partC_from_stack(s["i"], paths[f"win{s['i']}"], len(sets[f"win{s['i']}"]))
            os.remove(paths[f"win{s['i']}"])
        gc.collect()
    # frames_bg_30.mat, the source of q_ref
    if "frames_bg_30" not in ratios:
        import scipy.io as sio
        gate(400, "frames_bg_30")
        t = time.time()
        m = sio.loadmat(os.path.join(D, "frames_bg_30.mat"))
        a = m[[k for k in m if not k.startswith("__")][0]]
        a = a[:, :, 0, :] if a.ndim == 4 else a
        mean_mos = a.mean(axis=-1, dtype=np.float64)
        med_mos = np.median(a, axis=-1)
        I0, I45, I90, I135 = (c / 255.0 for c in recon.demosaic(mean_mos))
        S = (I0 + I45 + I90 + I135) / 4.0
        q_ref = np.clip((I0 - I90) / (S + 0.05), -1, 1)
        q_ref = np.where(S < 0.02, 0.0, q_ref)
        ratios["frames_bg_30"] = dict(n_frames=int(a.shape[-1]), raw=chan_stats(med_mos, "raw uint8, median of 30 frames"),
                                      q_ref_mean_as_stokes_diagnostic_v2=float(q_ref.mean()), seconds=time.time() - t)
        del m, a, mean_mos, med_mos
        rc.wjson("ratios.json", ratios)


def newvids():
    import h5py
    import extract_PETPS_exp as ex
    out = rc.rjson("newvids.json") or {}
    bgm, bgs = ex.ref()
    for name in ("PE500FRAMES", "PP500FRAMES"):
        if name in out:
            continue
        path = os.path.join(D, f"{name}.mat")
        t0 = time.time()
        head = open(path, "rb").read(116).decode("latin-1").rstrip("\x00 ").strip()
        rec = dict(file=f"{name}.mat", size_bytes=os.path.getsize(path), header_text=head,
                   file_modified=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))))
        with h5py.File(path, "r") as h:
            k = [k for k in h.keys() if not k.startswith("#")][0]
            ds = h[k]
            rec.update(container="HDF5 (MATLAB v7.3 MAT-file)", variable=k, shape_on_disk=list(ds.shape), dtype=str(ds.dtype),
                       chunks=str(ds.chunks), compression=f"{ds.compression} {ds.compression_opts}",
                       matlab_class=str(ds.attrs.get("MATLAB_class")))
            N = ds.shape[0]
        even = sorted(set(np.round(np.linspace(0, N - 1, 20)).astype(int).tolist()))
        comp = sorted(set(min(N - 1, f + 10) if f + 10 < N else max(0, f - 10) for f in even))
        paths, secs = read_pass(name, path, {"even20": even, "cmp": comp})
        st_e, st_c = stack(paths["even20"], len(even)), stack(paths["cmp"], len(comp))
        pos = {f: ("e", j) for j, f in enumerate(even)}
        pos.update({f: ("c", j) for j, f in enumerate(comp) if f not in pos})

        def get_sum(i):
            src, j = pos[i]
            fr = np.asarray(st_e[j] if src == "e" else st_c[j])
            return np.sum(np.stack(ex.H["demosaic"](fr), axis=0).astype(np.int64), axis=0)
        gate(400, f"detections {name}")
        det = []
        for f in even:  # existing detector and thresholds (detect_particles defaults), counts per frame
            regions, counts, _, _ = ex.region_info(get_sum(f), bgm, bgs)
            det.append(dict(frame=f, n_regions=counts["n_regions"], n_detections=len(regions)))
        try:  # holo_pipeline_starter Step 6 debris check on the same frames (comparison frame +/-10)
            _, fst, _ = ex.step6(get_sum, N, even, bgm, bgs)
            kept = dict(zip(fst.frame, fst.n_kept_step6))
        except ValueError:  # Step 6 raises when no detection survives in any frame
            kept = {f: 0 for f in even}
        for d in det:
            d["n_kept_after_debris_check"] = int(kept.get(d["frame"], 0))
        rec["detections"] = det
        med = median_mosaic(paths["even20"], len(even))
        rec.update(n_frames=N, frame_size=[H_RAW, W_RAW], frames_used=even, read_seconds=secs,
                   raw=chan_stats(med, "raw uint8, median of 20 frames"), levels=level_stats(paths["even20"], len(even), med),
                   seconds=time.time() - t0)
        del st_e, st_c, med
        for p in paths.values():
            os.remove(p)
        out[name] = rec
        rc.wjson("newvids.json", out)
        log(f"{name}: done in {rec['seconds'] / 60:.1f} min")


def main(step):
    background_mode()
    log(f"start '{step}' (pid {os.getpid()}, available {avail_mb()} MB)")
    if step in ("partB2", "all"):
        gate(400, "Part B under the new gate")
        partB2()
    if step in ("reads", "all"):
        run_reads_partC_ratios()
        gate(300, "render")
        rc.render()
        log("sheets rendered")
    if step in ("newvids", "all"):
        newvids()
    rc.wjson("followup_waits.json", WAITS)
    log("finished")


if __name__ == "__main__":
    main(sys.argv[1])
