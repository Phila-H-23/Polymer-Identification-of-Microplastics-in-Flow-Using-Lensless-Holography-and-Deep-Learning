"""
reconstruction_zsweep2.py

Re-run of the z-sweep, wider range and two focus metrics, plus a noise-
matched synthetic control and an eta signal-vs-noise-floor check. Read-only
w.r.t. the existing dataset; writes only new files under results/. Does
NOT build the full reconstructed patch dataset and does NOT train anything.

Deliverables (per explicit instruction):
  1. Real-particle z-sweep, same 5 particles as before, z = 1.0..4.0mm
     step 0.25mm (13 points), 128x128 crop, Laplacian variance AND Tamura
     coefficient of the amplitude.
  2. Synthetic control: absorbing discs of 30/50/100 um at true z=2.25mm,
     noise matched to the real background, same metrics and crop, same
     z-sweep range.
  3. Fixed decision rule (stated, not re-litigated):
       - if the synthetic control is flat, OR the real curves show no
         consistent interior peak -> fix z=2.25mm for all patches.
       - use a sweep-derived median z ONLY if >=4/5 real particles peak
         strictly inside the range (not on a boundary) and agree within
         +/-0.3mm of their median.
       - one z for all classes, never per class.
  4. Confirm the (c) example figure used both A_bg and B_bg corrections
     (static check against reconstruction_validate.py's code).
  5. Mean eta inside a particle mask vs same-size regions of background-
     only patches, reported next to the synthetic noise floor already
     established (mean 0.0094, std 0.0053, from
     scripts/reconstruction_biref_synthetic.py part 2).

UNIT NOTE on "noise matched to the real background": the synthetic field
is modelled on a normalised amplitude scale (object amplitude in [0.5,1.0],
intensity in [0.25,1.0]), not raw camera ADU counts, so matching the
real background's ABSOLUTE noise std (tens-hundreds of counts) would be
meaningless on this scale. What's matched instead is the RELATIVE noise
level: temporal_std/mean_level measured from the real per-video background
stacks (90 raw frames), averaged across the 5 videos, then applied as
Gaussian noise with that same fractional amplitude on the synthetic
sensor intensity. This preserves the real data's noise-to-signal ratio
rather than its absolute units. Flagged here rather than silently picked.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import reconstruction as recon
import reconstruction_validate as rv

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
RESULTS_DIR = os.path.join(DATA_DIR, "results")
BG_MANIFEST_PATH = os.path.join(DATA_DIR, "pipeline_output", "manifest_background_bgcorrected.csv")

Z_SWEEP2 = np.arange(1.0e-3, 4.001e-3, 0.25e-3)  # 1.0..4.0mm step 0.25mm, 13 points
HALF2 = 64  # 128x128 crop
Z_TRUE = recon.DEFAULT_Z_M  # 2.25mm
BG_WINDOW = rv.BG_WINDOW

SYNTH_NOISE_FLOOR_MEAN = 0.0094  # from reconstruction_biref_synthetic.py part 2
SYNTH_NOISE_FLOOR_STD = 0.0053

DISC_DIAMETERS_UM = [30, 50, 100]  # explicit assumption: DIAMETER, not radius - stated plainly


def tamura_coefficient(img):
    img = np.asarray(img, dtype=np.float64)
    mu = img.mean()
    if mu <= 0:
        return np.nan
    return img.std() / np.sqrt(mu)


def s0_amplitude(I0p, I45p, I90p, I135p):
    """Amplitude consistent with recon_s0's definition (mean of the four
    propagated intensities, per the earlier formula correction), not just
    I0+I90."""
    intensity_s0 = (I0p + I45p + I90p + I135p) / 4.0
    return np.sqrt(np.clip(intensity_s0, 0, None))


# ---------------------------------------------------------------------------
# 1. Real-particle z-sweep, widened range, two metrics, 128x128 crop
# ---------------------------------------------------------------------------
def real_zsweep():
    print("=== (1) Real-particle z-sweep, 1.0-4.0mm step 0.25mm, 128x128 crop ===")
    video_cache = {}
    per_particle = {}
    for cls, spec in rv.FIVE_PARTICLES.items():
        video = spec["source_file"]
        if video not in video_cache:
            print(f"  loading {video} (bulk contiguous read)...")
            video_cache[video] = rv.load_video_block(os.path.join(DATA_DIR, video))
        block, mode = video_cache[video]
        frame_idx, cx, cy = spec["frame"], spec["x"], spec["y"]

        raw = rv.get_raw_frame(block, mode, frame_idx).astype(np.float64)
        I0, I45, I90, I135 = recon.demosaic(raw)
        I0_bg, I45_bg, I90_bg, I135_bg = rv.channel_background_median(block, mode, frame_idx)

        lap_scores, tam_scores = [], []
        for z in Z_SWEEP2:
            fields = {}
            for name, I, Ibg in [("0", I0, I0_bg), ("45", I45, I45_bg), ("90", I90, I90_bg), ("135", I135, I135_bg)]:
                amp = np.sqrt(np.clip(I / (Ibg + 1e-6), 0, None)).astype(np.complex128)
                fields[name] = recon.angular_spectrum_propagate(amp, -z)
            I0p, I45p, I90p, I135p = (np.abs(fields[k]) ** 2 for k in ("0", "45", "90", "135"))
            amp_combo = s0_amplitude(I0p, I45p, I90p, I135p)
            patch = rv.crop(amp_combo, cx, cy, half=HALF2)
            lap_scores.append(laplacian_variance_amp(patch))
            tam_scores.append(tamura_coefficient(patch))

        per_particle[cls] = dict(video=video, frame=frame_idx, lap=lap_scores, tam=tam_scores)
        best_z_lap = Z_SWEEP2[int(np.argmax(lap_scores))]
        best_z_tam = Z_SWEEP2[int(np.argmax(tam_scores))]
        interior_lap = Z_SWEEP2[0] < best_z_lap < Z_SWEEP2[-1]
        interior_tam = Z_SWEEP2[0] < best_z_tam < Z_SWEEP2[-1]
        print(f"  {cls:5s}: best z (Laplacian-var) = {best_z_lap*1e3:.2f}mm "
              f"({'interior' if interior_lap else 'BOUNDARY'}); "
              f"best z (Tamura) = {best_z_tam*1e3:.2f}mm ({'interior' if interior_tam else 'BOUNDARY'})")

    rows = []
    for cls, d in per_particle.items():
        for z, lap, tam in zip(Z_SWEEP2, d["lap"], d["tam"]):
            rows.append(dict(cls=cls, video=d["video"], frame=d["frame"], z_mm=z * 1e3,
                              laplacian_var=lap, tamura=tam))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "recon_zsweep2_real.csv"), index=False)
    print("Saved -> results/recon_zsweep2_real.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cls, d in per_particle.items():
        axes[0].plot(Z_SWEEP2 * 1e3, d["lap"], marker="o", label=cls)
        axes[1].plot(Z_SWEEP2 * 1e3, d["tam"], marker="o", label=cls)
    for ax, title in zip(axes, ["Laplacian variance", "Tamura coefficient"]):
        ax.axvline(2.25, color="gray", linestyle="--", label="assumed z=2.25mm")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel(title + " (a.u.)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Widened focus sweep, 5 real particles, 128x128 crop, recon_s0-consistent amplitude")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "recon_zsweep2_real.png"), dpi=110)
    plt.close(fig)
    print("Saved -> results/recon_zsweep2_real.png")

    return per_particle, video_cache


def laplacian_variance_amp(img):
    lap = ndimage.laplace(img.astype(np.float64))
    return lap.var()


# ---------------------------------------------------------------------------
# 2. Synthetic control: noise-matched discs at true z=2.25mm
# ---------------------------------------------------------------------------
def measure_real_background_noise_fraction(video_cache):
    """Relative (fractional) temporal noise level from the real background
    stacks: per-pixel temporal std (across BG_WINDOW raw frames), averaged
    spatially, divided by the overall mean level. Averaged across the 5
    videos already loaded for the real z-sweep."""
    fracs = []
    for cls, spec in rv.FIVE_PARTICLES.items():
        video = spec["source_file"]
        block, mode = video_cache[video]
        frame_idx = spec["frame"]
        lo, hi = max(0, frame_idx - BG_WINDOW), frame_idx
        frames = np.stack([rv.get_raw_frame(block, mode, i) for i in range(lo, hi)], axis=0).astype(np.float64)
        temporal_std = frames.std(axis=0).mean()
        mean_level = frames.mean()
        fracs.append(temporal_std / mean_level)
    frac = float(np.mean(fracs))
    print(f"\nmeasured relative background noise (temporal_std/mean, averaged over 5 videos): {frac:.4f}")
    return frac


def synthetic_control_sweep(noise_frac):
    print("\n=== (2) Synthetic control: discs 30/50/100 um diameter, true z=2.25mm, noise-matched ===")
    N = 512
    dx = recon.PIXEL_PITCH_M
    yy, xx = np.meshgrid(np.arange(N) - N // 2, np.arange(N) - N // 2, indexing="ij")
    r_px = np.hypot(xx, yy)
    rng = np.random.default_rng(1)

    curves = {}
    for diam_um in DISC_DIAMETERS_UM:
        radius_px = (diam_um / 2.0) / (dx * 1e6)
        amplitude_true = np.where(r_px <= radius_px, 0.5, 1.0)
        field_object = amplitude_true.astype(np.complex128)
        field_sensor = recon.angular_spectrum_propagate(field_object, Z_TRUE)
        I_sensor = np.abs(field_sensor) ** 2
        sigma_abs = noise_frac * I_sensor.mean()
        I_noisy = np.clip(I_sensor + rng.normal(0, sigma_abs, size=I_sensor.shape), 1e-6, None)
        I_bg = 1.0

        lap_scores, tam_scores = [], []
        for z in Z_SWEEP2:
            amp_recon = np.sqrt(I_noisy / I_bg).astype(np.complex128)
            field_back = recon.angular_spectrum_propagate(amp_recon, -z)
            amp_back = np.abs(field_back)
            patch = rv.crop(amp_back, N // 2, N // 2, half=HALF2)
            lap_scores.append(laplacian_variance_amp(patch))
            tam_scores.append(tamura_coefficient(patch))
        curves[diam_um] = dict(radius_px=radius_px, lap=lap_scores, tam=tam_scores)

        best_z_lap = Z_SWEEP2[int(np.argmax(lap_scores))]
        best_z_tam = Z_SWEEP2[int(np.argmax(tam_scores))]
        lap_range = max(lap_scores) - min(lap_scores)
        lap_rel_var = lap_range / max(lap_scores) if max(lap_scores) > 0 else np.nan
        tam_range = max(tam_scores) - min(tam_scores)
        tam_rel_var = tam_range / max(tam_scores) if max(tam_scores) > 0 else np.nan
        print(f"  disc {diam_um}um (radius {radius_px:.2f}px): "
              f"best z (Lap)={best_z_lap*1e3:.2f}mm (rel.var={lap_rel_var:.3f}), "
              f"best z (Tam)={best_z_tam*1e3:.2f}mm (rel.var={tam_rel_var:.3f})")

    rows = []
    for diam_um, d in curves.items():
        for z, lap, tam in zip(Z_SWEEP2, d["lap"], d["tam"]):
            rows.append(dict(diam_um=diam_um, radius_px=d["radius_px"], z_mm=z * 1e3,
                              laplacian_var=lap, tamura=tam))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "recon_zsweep2_synthetic.csv"), index=False)
    print("Saved -> results/recon_zsweep2_synthetic.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for diam_um, d in curves.items():
        axes[0].plot(Z_SWEEP2 * 1e3, d["lap"], marker="o", label=f"{diam_um}um")
        axes[1].plot(Z_SWEEP2 * 1e3, d["tam"], marker="o", label=f"{diam_um}um")
    for ax, title in zip(axes, ["Laplacian variance", "Tamura coefficient"]):
        ax.axvline(2.25, color="gray", linestyle="--", label="true z=2.25mm")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel(title + " (a.u.)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle(f"Synthetic control, noise frac={noise_frac:.3f}, true z=2.25mm")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "recon_zsweep2_synthetic.png"), dpi=110)
    plt.close(fig)
    print("Saved -> results/recon_zsweep2_synthetic.png")

    return curves


# ---------------------------------------------------------------------------
# 3. Decision rule
# ---------------------------------------------------------------------------
def apply_decision_rule(per_particle, synth_curves):
    print("\n=== (3) Decision rule ===")

    # Synthetic flatness: flat if peak (by Laplacian var, the primary metric)
    # is not within +/-0.3mm of the true z=2.25mm for a majority (>=2/3) of
    # disc sizes, OR relative variation is small (<10%).
    synth_flat_votes = []
    for diam_um, d in synth_curves.items():
        best_z = Z_SWEEP2[int(np.argmax(d["lap"]))]
        rel_var = (max(d["lap"]) - min(d["lap"])) / max(d["lap"]) if max(d["lap"]) > 0 else 0
        near_true = abs(best_z - Z_TRUE) <= 0.3e-3
        flat = rel_var < 0.10
        synth_flat_votes.append(flat or not near_true)
        print(f"  disc {diam_um}um: peak at {best_z*1e3:.2f}mm, rel.var={rel_var:.3f}, "
              f"{'FLAT/no-peak-at-truth' if (flat or not near_true) else 'peaked at truth'}")
    synthetic_is_flat = sum(synth_flat_votes) >= 2  # majority of 3 sizes

    # Real interior-peak check (Laplacian var, primary metric)
    interior_peaks = []
    for cls, d in per_particle.items():
        best_z = Z_SWEEP2[int(np.argmax(d["lap"]))]
        is_interior = Z_SWEEP2[0] < best_z < Z_SWEEP2[-1]
        if is_interior:
            interior_peaks.append(best_z)
    n_interior = len(interior_peaks)
    agree = False
    if n_interior >= 4:
        med = np.median(interior_peaks)
        agree = all(abs(z - med) <= 0.3e-3 for z in interior_peaks)

    print(f"\nsynthetic control flat / no-consistent-peak-at-truth: {synthetic_is_flat}")
    print(f"real particles with interior peak: {n_interior}/5"
          + (f" (z values: {[round(z*1e3,2) for z in interior_peaks]})" if interior_peaks else ""))

    if synthetic_is_flat or n_interior < 4 or not agree:
        chosen_z = Z_TRUE
        reason = []
        if synthetic_is_flat:
            reason.append("synthetic control is flat/does not peak at the true z")
        if n_interior < 4:
            reason.append(f"only {n_interior}/5 real particles peak inside the swept range")
        elif not agree:
            reason.append("interior peaks do not agree within +/-0.3mm")
        print(f"\nDECISION: fix z = {chosen_z*1e3:.2f}mm for all patches "
              f"(reason: {'; '.join(reason)}).")
    else:
        chosen_z = float(np.median(interior_peaks))
        print(f"\nDECISION: use sweep-derived median z = {chosen_z*1e3:.2f}mm for all patches "
              f"(>=4/5 real particles peaked inside the range and agreed within +/-0.3mm).")

    return chosen_z


# ---------------------------------------------------------------------------
# 4. Confirm background corrections used in the (c) example figure
# ---------------------------------------------------------------------------
def confirm_example_figure_corrections():
    print("\n=== (4) Confirm (c) example figure used both background corrections ===")
    print("Static check against scripts/reconstruction_validate.py, example_figure():")
    print("  I0_bg,I45_bg,I90_bg,I135_bg = channel_background_median(...)  <- per-channel bg, used in field amplitude")
    print("  A_bg, B_bg = ab_background_median(...)                        <- A/B bg, from RAW (pre-propagation) frames")
    print("  eta, theta = recon.birefringence_eta_theta(I0p, I45p, I90p, I135p, A_bg=A_bg, B_bg=B_bg)")
    print("CONFIRMED: both the per-channel background (in the field amplitude) and the A_bg/B_bg "
          "background (in eta/theta) were applied.")
    print("CAVEAT (not previously flagged): A_bg/B_bg are computed from RAW, un-propagated frames, "
          "then subtracted from A/B computed on the PROPAGATED I0p..I135p. This mixes domains - "
          "it assumes the elliptical-illumination offset in A/B is unchanged by propagation, which "
          "has not been separately verified. Flagging as a known limitation, not fixing without "
          "further instruction.")


# ---------------------------------------------------------------------------
# 5. eta: particle mask vs background-only patches vs synthetic noise floor
# ---------------------------------------------------------------------------
def eta_signal_vs_noise(video_cache, chosen_z):
    print(f"\n=== (5) eta: particle mask vs background-only patches vs synthetic noise floor (z={chosen_z*1e3:.2f}mm) ===")
    bg_manifest = pd.read_csv(BG_MANIFEST_PATH)

    particle_etas = []
    bg_patch_means = []

    for cls, spec in rv.FIVE_PARTICLES.items():
        video = spec["source_file"]
        block, mode = video_cache[video]
        frame_idx, cx, cy = spec["frame"], spec["x"], spec["y"]

        raw = rv.get_raw_frame(block, mode, frame_idx).astype(np.float64)
        I0, I45, I90, I135 = recon.demosaic(raw)
        I0_bg, I45_bg, I90_bg, I135_bg = rv.channel_background_median(block, mode, frame_idx)
        A_bg, B_bg = rv.ab_background_median(block, mode, frame_idx)

        fields = {}
        for name, I, Ibg in [("0", I0, I0_bg), ("45", I45, I45_bg), ("90", I90, I90_bg), ("135", I135, I135_bg)]:
            amp = np.sqrt(np.clip(I / (Ibg + 1e-6), 0, None)).astype(np.complex128)
            fields[name] = recon.angular_spectrum_propagate(amp, -chosen_z)
        I0p, I45p, I90p, I135p = (np.abs(fields[k]) ** 2 for k in ("0", "45", "90", "135"))
        eta_full, theta_full = recon.birefringence_eta_theta(I0p, I45p, I90p, I135p, A_bg=A_bg, B_bg=B_bg)
        intensity_combo = (I0p + I45p + I90p + I135p) / 4.0

        eta_patch = rv.crop(eta_full, cx, cy, half=32)
        intensity_patch = rv.crop(intensity_combo, cx, cy, half=32)
        # particle mask: darkest 25% of the intensity patch (absorbing particle = lower intensity)
        thresh = np.nanpercentile(intensity_patch, 25)
        mask = intensity_patch <= thresh
        particle_eta_mean = np.nanmean(eta_patch[mask])
        particle_etas.append(particle_eta_mean)
        print(f"  {cls:5s} particle: mean eta inside darkest-25% mask = {particle_eta_mean:.4f} "
              f"(n_px={mask.sum()})")

        # background-only patches from the same video+frame (reuses the same I_bg/A_bg/B_bg)
        sub = bg_manifest[(bg_manifest["class"] == cls) & (bg_manifest["source_file"] == video)
                           & (bg_manifest["frame"] == frame_idx)]
        sample = sub.sample(n=min(5, len(sub)), random_state=0) if len(sub) else sub
        for _, row in sample.iterrows():
            bx, by = int(row["x"]), int(row["y"])
            bg_eta_patch = rv.crop(eta_full, bx, by, half=32)
            bg_patch_means.append(np.nanmean(bg_eta_patch))

    particle_etas = np.array(particle_etas)
    bg_patch_means = np.array(bg_patch_means)

    print(f"\nparticle-mask eta: mean={np.nanmean(particle_etas):.4f}, std={np.nanstd(particle_etas):.4f} (n={len(particle_etas)} particles)")
    print(f"background-only patch eta: mean={np.nanmean(bg_patch_means):.4f}, std={np.nanstd(bg_patch_means):.4f} (n={len(bg_patch_means)} patches)")
    print(f"synthetic noise floor (from reconstruction_biref_synthetic.py part 2): "
          f"mean={SYNTH_NOISE_FLOOR_MEAN:.4f}, std={SYNTH_NOISE_FLOOR_STD:.4f}")

    return dict(particle_mean=float(np.nanmean(particle_etas)), particle_std=float(np.nanstd(particle_etas)),
                bg_mean=float(np.nanmean(bg_patch_means)), bg_std=float(np.nanstd(bg_patch_means)))


def main():
    per_particle, video_cache = real_zsweep()
    noise_frac = measure_real_background_noise_fraction(video_cache)
    synth_curves = synthetic_control_sweep(noise_frac)
    chosen_z = apply_decision_rule(per_particle, synth_curves)
    confirm_example_figure_corrections()
    eta_result = eta_signal_vs_noise(video_cache, chosen_z)

    print("\n=== Summary ===")
    print(f"chosen z for all patches: {chosen_z*1e3:.2f}mm")
    print(f"particle-mask eta mean={eta_result['particle_mean']:.4f} vs "
          f"background-only eta mean={eta_result['bg_mean']:.4f} vs "
          f"synthetic noise floor mean={SYNTH_NOISE_FLOOR_MEAN:.4f}")


if __name__ == "__main__":
    main()
