"""
reconstruction_validate.py

Deliverables (a), (b), (c) from the reconstruction-pathway task. Read-only
with respect to the existing dataset - writes only new files under
results/. Does NOT build the full reconstructed patch dataset and does
NOT train anything; that's explicitly deferred pending review of this
validation.

(a) Synthetic validation: propagate a known absorbing disc forward by z,
    then back-propagate the resulting (zero-phase-assumed) field and
    check the disc is recovered.
(b) z-sweep 2.0-2.5mm on 5 real particles (one per class: PE, PET, PMMA,
    PS, PVC), Laplacian-variance sharpness metric, full-frame propagation
    then patch-crop per the task's explicit instruction.
(c) One figure: raw patch / propagated intensity / eta / theta for one
    PET, one PE, one PVC particle.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import reconstruction as recon

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
MANIFEST_PATH = os.path.join(DATA_DIR, "pipeline_output", "manifest.csv")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

Z_DEFAULT = recon.DEFAULT_Z_M
Z_SWEEP = np.arange(2.0e-3, 2.501e-3, 0.05e-3)  # 2.0..2.5mm step 0.05mm
BG_WINDOW = 90
HALF = 32  # patch crop half-width, matches every other patch-cropping step in this project

FIVE_PARTICLES = {
    "PET": dict(source_file="PET1_600.mat", frame=106, x=753, y=251),
    "PE":  dict(source_file="PE1_600.mat",  frame=106, x=697, y=360),
    "PVC": dict(source_file="PVC1_600.mat", frame=106, x=1210, y=365),
    "PMMA": dict(source_file="PMMA1_600.mat", frame=141, x=1112, y=237),
    "PS":  dict(source_file="PS1_600.mat",  frame=106, x=1164, y=302),
}
EXAMPLE_CLASSES = ["PET", "PE", "PVC"]


# ---------------------------------------------------------------------------
# (a) Synthetic disc validation
# ---------------------------------------------------------------------------
def synthetic_disc_validation():
    print("=== (a) Synthetic disc validation ===")
    N = 512
    dx = recon.PIXEL_PITCH_M
    yy, xx = np.meshgrid(np.arange(N) - N // 2, np.arange(N) - N // 2, indexing="ij")
    r_px = np.hypot(xx, yy)
    disc_radius_px = 20
    disc_radius_um = disc_radius_px * dx * 1e6

    # True object: pure absorbing disc, amplitude 0.5 inside / 1.0 outside, zero phase.
    amplitude_true = np.where(r_px <= disc_radius_px, 0.5, 1.0)
    field_object = amplitude_true.astype(np.complex128)

    z = Z_DEFAULT
    field_sensor = recon.angular_spectrum_propagate(field_object, z)
    I_sensor = np.abs(field_sensor) ** 2
    I_bg = 1.0  # background amplitude 1 -> intensity 1, matches the true background region exactly

    # Reconstruction recipe exactly as specified: amplitude=sqrt(I/I_bg), zero phase.
    amplitude_recon_at_sensor = np.sqrt(I_sensor / I_bg)
    field_recon_at_sensor = amplitude_recon_at_sensor.astype(np.complex128)  # zero phase
    field_back = recon.angular_spectrum_propagate(field_recon_at_sensor, -z)
    amplitude_back = np.abs(field_back)

    # Quantitative check: does a disc-shaped low-amplitude region reappear
    # at the right place/size?
    inside = r_px <= disc_radius_px
    outside = (r_px > disc_radius_px * 1.5) & (r_px < N // 2 - 10)
    mean_inside = amplitude_back[inside].mean()
    mean_outside = amplitude_back[outside].mean()
    contrast = (mean_outside - mean_inside) / mean_outside

    # Recovered radius: distance from center where amplitude crosses the midpoint
    # between mean_inside and mean_outside, azimuthally averaged.
    r_int = r_px.astype(int)
    radial_profile = ndimage.mean(amplitude_back, labels=r_int, index=np.arange(0, N // 2))
    midpoint = (mean_inside + mean_outside) / 2
    below = np.where(radial_profile <= midpoint)[0]
    recovered_radius_px = below.max() if len(below) else np.nan
    radius_error_pct = abs(recovered_radius_px - disc_radius_px) / disc_radius_px * 100

    print(f"wavelength (medium) = {recon.WAVELENGTH_MEDIUM_M*1e9:.2f} nm, "
          f"pixel pitch = {dx*1e6:.2f} um, z = {z*1e3:.3f} mm, grid {N}x{N}")
    print(f"true disc radius: {disc_radius_px} px ({disc_radius_um:.1f} um)")
    print(f"mean amplitude inside disc (recovered): {mean_inside:.4f} (true=0.5)")
    print(f"mean amplitude outside disc (recovered): {mean_outside:.4f} (true=1.0)")
    print(f"contrast (outside-inside)/outside: {contrast:.4f}")
    print(f"recovered disc radius: {recovered_radius_px} px  (error {radius_error_pct:.1f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(amplitude_true, cmap="gray", vmin=0.4, vmax=1.05)
    axes[0].set_title("true object amplitude")
    axes[1].imshow(np.sqrt(I_sensor), cmap="gray")
    axes[1].set_title(f"|field| at sensor (z=+{z*1e3:.2f}mm)")
    axes[2].imshow(amplitude_back, cmap="gray", vmin=0.4, vmax=1.05)
    axes[2].set_title(f"back-propagated amplitude (z=-{z*1e3:.2f}mm)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "recon_synthetic_validation.png"), dpi=110)
    plt.close(fig)
    print("Saved -> results/recon_synthetic_validation.png")

    ok = radius_error_pct < 25 and contrast > 0.2
    print(f"\nVerdict: {'PASS' if ok else 'FAIL/MARGINAL'} - disc "
          f"{'clearly recovered' if ok else 'not cleanly recovered'} "
          f"(contrast={contrast:.3f}, radius error={radius_error_pct:.1f}%)")
    return dict(contrast=contrast, radius_error_pct=radius_error_pct, ok=ok)


# ---------------------------------------------------------------------------
# Shared video-loading helpers (mirroring scripts/build_bgcorrected_dataset.py)
# ---------------------------------------------------------------------------
def load_video_block(path):
    try:
        mat = sio.loadmat(path)
        var_name = [k for k in mat.keys() if not k.startswith("__")][0]
        return mat[var_name], "scipy"
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            var_name = [k for k in f.keys() if not k.startswith("#")][0]
            return f[var_name][:], "h5py"


def get_raw_frame(block, mode, idx):
    if mode == "scipy":
        return block[:, :, 0, idx] if block.ndim == 4 else block[:, :, idx]
    else:
        return block[idx, 0].T if block.ndim == 4 else block[idx].T


def channel_background_median(block, mode, frame_idx, window=BG_WINDOW):
    """Median of the RAW MOSAIC frame over [frame_idx-window, frame_idx),
    demosaiced ONCE afterward - equivalent to (and much cheaper than)
    demosaicing each frame then medianing per channel, since demosaic()
    is a fixed per-pixel selection and median commutes with that
    selection exactly like it does for the mean (see
    build_bgcorrected_dataset.py's docstring for the mean case)."""
    lo, hi = max(0, frame_idx - window), frame_idx
    if hi <= lo:
        return None
    # Median taken on the native uint8 stack: identical result to a float64 median
    # (verified), at 1/8 the memory - the float64 stack forced paging on a 16GB machine.
    frames = np.stack([get_raw_frame(block, mode, i) for i in range(lo, hi)], axis=0)
    median_mosaic = np.median(frames, axis=0)
    return recon.demosaic(median_mosaic)  # (I0_bg, I45_bg, I90_bg, I135_bg)


def ab_background_median(block, mode, frame_idx, window=BG_WINDOW):
    """Median of PER-FRAME A=(I0-I90)/(I0+I90), B=(I135-I45)/(I45+I135)
    over the window - does NOT commute with a raw-frame median (A, B are
    nonlinear in the raw channels), so each frame is demosaiced and its
    own A/B computed before medianing across frames."""
    lo, hi = max(0, frame_idx - window), frame_idx
    if hi <= lo:
        return None, None
    A_list, B_list = [], []
    for i in range(lo, hi):
        raw = get_raw_frame(block, mode, i).astype(np.float64)
        I0, I45, I90, I135 = recon.demosaic(raw)
        with np.errstate(invalid="ignore", divide="ignore"):
            D0 = I0 + I90
            D45 = I45 + I135
            A = np.where(D0 > 0, (I0 - I90) / D0, np.nan)
            B = np.where(D45 > 0, (I135 - I45) / D45, np.nan)
        A_list.append(A)
        B_list.append(B)
    return np.nanmedian(A_list, axis=0), np.nanmedian(B_list, axis=0)


def crop(arr, cx, cy, half=HALF):
    h, w = arr.shape[:2]
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    return arr[y0:y1, x0:x1]


def laplacian_variance(img):
    lap = ndimage.laplace(img.astype(np.float64))
    return lap.var()


# ---------------------------------------------------------------------------
# (b) z-sweep on 5 real particles
# ---------------------------------------------------------------------------
def z_sweep():
    print("\n=== (b) z-sweep, 2.0-2.5mm, 5 real particles ===")
    rows = []
    video_cache = {}
    for cls, spec in FIVE_PARTICLES.items():
        video = spec["source_file"]
        if video not in video_cache:
            print(f"  loading {video} (bulk contiguous read)...")
            video_cache[video] = load_video_block(os.path.join(DATA_DIR, video))
        block, mode = video_cache[video]

        frame_idx, cx, cy = spec["frame"], spec["x"], spec["y"]
        raw = get_raw_frame(block, mode, frame_idx).astype(np.float64)
        I0, I45, I90, I135 = recon.demosaic(raw)
        bg = channel_background_median(block, mode, frame_idx)
        I0_bg, I45_bg, I90_bg, I135_bg = bg

        # recon_s0 basis: field per channel = sqrt(I_c / I_c_bg), zero phase;
        # combine I0+I90 (as specified for recon_s0) at the FIELD level via
        # their intensities after back-propagation (propagate each channel
        # separately, per the task).
        best_z, best_score, scores = None, -np.inf, []
        for z in Z_SWEEP:
            field0 = np.sqrt(np.clip(I0 / (I0_bg + 1e-6), 0, None)).astype(np.complex128)
            field90 = np.sqrt(np.clip(I90 / (I90_bg + 1e-6), 0, None)).astype(np.complex128)
            back0 = recon.angular_spectrum_propagate(field0, -z)
            back90 = recon.angular_spectrum_propagate(field90, -z)
            intensity_s0 = np.abs(back0) ** 2 + np.abs(back90) ** 2
            patch = crop(intensity_s0, cx, cy)
            score = laplacian_variance(patch)
            scores.append(score)
            if score > best_score:
                best_score, best_z = score, z
        rows.append(dict(cls=cls, video=video, frame=frame_idx,
                          best_z_mm=best_z * 1e3, best_score=best_score,
                          scores=scores))
        print(f"  {cls:5s} ({video}, frame {frame_idx}): best z = {best_z*1e3:.2f} mm "
              f"(sharpness {best_score:.4g})")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "scores"} for r in rows])
    df.to_csv(os.path.join(RESULTS_DIR, "recon_zsweep.csv"), index=False)
    print(f"Saved -> results/recon_zsweep.csv")
    print(f"\nmean best-focus z across 5 particles: {df['best_z_mm'].mean():.3f} mm "
          f"(std {df['best_z_mm'].std():.3f} mm), vs assumed 2.25mm")

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        ax.plot(Z_SWEEP * 1e3, r["scores"], marker="o", label=r["cls"])
    ax.axvline(2.25, color="gray", linestyle="--", label="assumed z=2.25mm")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Laplacian-variance sharpness (a.u.)")
    ax.set_title("Focus sweep, 5 real particles (recon_s0 basis)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "recon_zsweep.png"), dpi=110)
    plt.close(fig)
    print("Saved -> results/recon_zsweep.png")
    return df, video_cache


# ---------------------------------------------------------------------------
# (c) example figure: raw / propagated intensity / eta / theta, PET/PE/PVC
# ---------------------------------------------------------------------------
def example_figure(video_cache):
    print("\n=== (c) Example figure: PET / PE / PVC ===")
    fig, axes = plt.subplots(len(EXAMPLE_CLASSES), 4, figsize=(14, 3.3 * len(EXAMPLE_CLASSES)))
    z = Z_DEFAULT

    for row_idx, cls in enumerate(EXAMPLE_CLASSES):
        spec = FIVE_PARTICLES[cls]
        video = spec["source_file"]
        block, mode = video_cache[video]
        frame_idx, cx, cy = spec["frame"], spec["x"], spec["y"]

        raw = get_raw_frame(block, mode, frame_idx).astype(np.float64)
        I0, I45, I90, I135 = recon.demosaic(raw)
        I0_bg, I45_bg, I90_bg, I135_bg = channel_background_median(block, mode, frame_idx)
        A_bg, B_bg = ab_background_median(block, mode, frame_idx)

        fields_bg = {}
        for name, I, Ibg in [("I0", I0, I0_bg), ("I45", I45, I45_bg), ("I90", I90, I90_bg), ("I135", I135, I135_bg)]:
            amp = np.sqrt(np.clip(I / (Ibg + 1e-6), 0, None))
            fields_bg[name] = recon.angular_spectrum_propagate(amp.astype(np.complex128), -z)

        I0p = np.abs(fields_bg["I0"]) ** 2
        I45p = np.abs(fields_bg["I45"]) ** 2
        I90p = np.abs(fields_bg["I90"]) ** 2
        I135p = np.abs(fields_bg["I135"]) ** 2

        eta_full, theta_full = recon.birefringence_eta_theta(I0p, I45p, I90p, I135p, A_bg=A_bg, B_bg=B_bg)

        raw_patch = crop(I0, cx, cy)  # single representative raw channel for display
        intensity_patch = crop(I0p + I90p, cx, cy)
        eta_patch = crop(eta_full, cx, cy)
        theta_patch = crop(theta_full, cx, cy)

        axes[row_idx, 0].imshow(raw_patch, cmap="gray")
        axes[row_idx, 1].imshow(intensity_patch, cmap="gray")
        axes[row_idx, 2].imshow(eta_patch, cmap="viridis", vmin=0, vmax=1)
        axes[row_idx, 3].imshow(theta_patch, cmap="twilight", vmin=0, vmax=1)
        for c, title in enumerate(["raw I0 patch", "propagated intensity (I0+I90)", "eta", "theta"]):
            axes[row_idx, c].axis("off")
            if row_idx == 0:
                axes[row_idx, c].set_title(title, fontsize=10)
        axes[row_idx, 0].text(-0.15, 0.5, cls, transform=axes[row_idx, 0].transAxes,
                               fontsize=11, fontweight="bold", ha="right", va="center")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "recon_example_pet_pe_pvc.png"), dpi=110)
    plt.close(fig)
    print("Saved -> results/recon_example_pet_pe_pvc.png")


def main():
    synth_result = synthetic_disc_validation()
    zsweep_df, video_cache = z_sweep()
    example_figure(video_cache)

    print("\n=== Summary ===")
    print(f"(a) synthetic disc: {'PASS' if synth_result['ok'] else 'FAIL/MARGINAL'}")
    print(f"(b) z-sweep mean best focus: {zsweep_df['best_z_mm'].mean():.3f} mm "
          f"(assumed 2.25mm)")
    print("(c) example figure saved")


if __name__ == "__main__":
    main()
