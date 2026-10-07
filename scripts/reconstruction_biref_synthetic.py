"""
reconstruction_biref_synthetic.py

Synthetic birefringence-retrieval check. No real data. Read-only w.r.t.
the existing dataset; writes only results/recon_biref_synthetic.csv and
results/recon_biref_synthetic_bgcheck.png. No training.

Part 1 - forward-model round trip:
  For known retardance delta (0.1, 0.3, 0.6 rad) and known slow-axis
  azimuth phi (0, 30, 60, 120 deg), generate the four analyzer intensities
  from the Shribak circular-illumination model:
      A = sin(2*phi) * sin(delta)
      B = cos(2*phi) * sin(delta)
  inverted to intensities assuming a flat, non-elliptical illumination
  level D (no I90~4.5x I0 imbalance here - that's the separate imbalance
  modelled in Part 2):
      I0  = D * (1 + A) / 2,   I90  = D * (1 - A) / 2
      I135 = D * (1 + B) / 2,  I45  = D * (1 - B) / 2
  Then run reconstruction.birefringence_eta_theta on these (no A_bg/B_bg -
  this is the ideal, non-elliptical case) and compare recovered eta/theta
  against the known truth (eta_true = delta/(pi/2), theta_true = phi/pi
  after wrapping phi into [0, pi)).

Part 2 - background-offset check:
  Simulate a stack of background-only frames (no birefringent object,
  delta=0 everywhere) carrying the documented elliptical-illumination
  imbalance (I90 ~= 4.5x I0; I45/I135 left balanced - the 4.5x figure is
  the one explicitly documented for I0 vs I90, extending it to I45/I135
  without evidence would be an unsupported assumption) plus Poisson-like
  photon noise. Compute A_bg/B_bg as the per-pixel MEDIAN of A/B over that
  stack (mirrors the rolling-median background used by the real pathway),
  then evaluate eta on a held-out background frame from the same
  distribution, with and without the A_bg/B_bg subtraction.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import reconstruction as recon

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
RESULTS_DIR = os.path.join(DATA_DIR, "results")

DELTAS = [0.1, 0.3, 0.6]  # rad
PHIS_DEG = [0, 30, 60, 120]
D_LEVEL = 1000.0  # arbitrary flat illumination level (counts); A,B are ratios so scale-free

N_BG_FRAMES = 90
FRAME_SHAPE = (64, 64)
I0_LEVEL = 500.0
I90_OVER_I0 = 4.5  # documented elliptical-illumination imbalance
NOISE_SIGMA_FRAC = 0.02  # ~2% of local mean, stand-in for photon/read noise
RNG_SEED = 0


def forward_model(delta, phi_rad):
    A = np.sin(2 * phi_rad) * np.sin(delta)
    B = np.cos(2 * phi_rad) * np.sin(delta)
    I0 = D_LEVEL * (1 + A) / 2
    I90 = D_LEVEL * (1 - A) / 2
    I135 = D_LEVEL * (1 + B) / 2
    I45 = D_LEVEL * (1 - B) / 2
    return I0, I45, I90, I135


def part1_forward_round_trip():
    rows = []
    for delta in DELTAS:
        for phi_deg in PHIS_DEG:
            phi_rad = np.deg2rad(phi_deg)
            I0, I45, I90, I135 = forward_model(delta, phi_rad)
            I0 = np.full((4, 4), I0)
            I45 = np.full((4, 4), I45)
            I90 = np.full((4, 4), I90)
            I135 = np.full((4, 4), I135)
            eta, theta = recon.birefringence_eta_theta(I0, I45, I90, I135)
            eta_rec = float(np.nanmean(eta))
            theta_rec = float(np.nanmean(theta))

            eta_true = delta / (np.pi / 2)
            phi_wrapped = np.mod(phi_rad, np.pi)
            theta_true = phi_wrapped / np.pi

            rows.append(dict(
                delta_true=delta, phi_deg_true=phi_deg,
                eta_true=eta_true, eta_recovered=eta_rec, eta_error=eta_rec - eta_true,
                theta_true=theta_true, theta_recovered=theta_rec,
                theta_error_raw=theta_rec - theta_true,
            ))
    df = pd.DataFrame(rows)
    max_eta_err = df["eta_error"].abs().max()
    max_theta_err = df["theta_error_raw"].abs().max()
    print("=== Part 1: forward-model round trip (12 delta/phi combinations) ===")
    print(df.to_string(index=False))
    print(f"\nmax |eta_error| = {max_eta_err:.6f}")
    print(f"max |theta_error_raw| (no offset correction applied) = {max_theta_err:.6f}")
    if max_theta_err > 1e-6:
        print("NOTE: nonzero theta offset detected - reporting as found, not silently corrected.")
    else:
        print("NOTE: theta recovered with no measurable sign/axis offset relative to phi as defined "
              "by the forward model above (A=sin(2phi)sin(delta), B=cos(2phi)sin(delta)).")
    out_path = os.path.join(RESULTS_DIR, "recon_biref_synthetic.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return df


def simulate_bg_frame(rng):
    I0 = rng.normal(I0_LEVEL, I0_LEVEL * NOISE_SIGMA_FRAC, size=FRAME_SHAPE)
    I90 = rng.normal(I0_LEVEL * I90_OVER_I0, I0_LEVEL * I90_OVER_I0 * NOISE_SIGMA_FRAC, size=FRAME_SHAPE)
    I45 = rng.normal(I0_LEVEL, I0_LEVEL * NOISE_SIGMA_FRAC, size=FRAME_SHAPE)
    I135 = rng.normal(I0_LEVEL, I0_LEVEL * NOISE_SIGMA_FRAC, size=FRAME_SHAPE)
    return I0, I45, I90, I135


def part2_background_offset_check():
    rng = np.random.default_rng(RNG_SEED)
    A_stack, B_stack = [], []
    for _ in range(N_BG_FRAMES):
        I0, I45, I90, I135 = simulate_bg_frame(rng)
        D0 = I0 + I90
        D45 = I45 + I135
        A_stack.append((I0 - I90) / D0)
        B_stack.append((I135 - I45) / D45)
    A_bg = np.median(np.stack(A_stack, axis=0), axis=0)
    B_bg = np.median(np.stack(B_stack, axis=0), axis=0)

    I0, I45, I90, I135 = simulate_bg_frame(rng)  # held-out frame, same distribution

    eta_uncorrected, theta_uncorrected = recon.birefringence_eta_theta(I0, I45, I90, I135)
    eta_corrected, theta_corrected = recon.birefringence_eta_theta(I0, I45, I90, I135, A_bg=A_bg, B_bg=B_bg)

    A_offset_expected = (I0_LEVEL - I0_LEVEL * I90_OVER_I0) / (I0_LEVEL + I0_LEVEL * I90_OVER_I0)

    print("\n=== Part 2: background-offset check (simulated I90 ~= 4.5x I0 illumination) ===")
    print(f"expected steady-state A offset from imbalance alone: {A_offset_expected:.4f}")
    print(f"mean A_bg (median over {N_BG_FRAMES} simulated background frames): {np.mean(A_bg):.4f}")
    print(f"mean B_bg: {np.mean(B_bg):.4f}")
    print(f"eta WITHOUT A_bg/B_bg subtraction: mean={np.nanmean(eta_uncorrected):.4f}, "
          f"std={np.nanstd(eta_uncorrected):.4f}")
    print(f"eta WITH A_bg/B_bg subtraction:    mean={np.nanmean(eta_corrected):.4f}, "
          f"std={np.nanstd(eta_corrected):.4f}")
    if abs(np.nanmean(eta_corrected)) < 0.05:
        print("CONFIRMED: eta ~= 0 on a background-only frame after A_bg/B_bg subtraction.")
    else:
        print("NOT confirmed: eta after subtraction is not close to 0 - reporting as found.")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im0 = axes[0].imshow(eta_uncorrected, cmap="viridis", vmin=0)
    axes[0].set_title(f"eta, no bg subtraction\nmean={np.nanmean(eta_uncorrected):.3f}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(eta_corrected, cmap="viridis", vmin=0)
    axes[1].set_title(f"eta, with A_bg/B_bg subtraction\nmean={np.nanmean(eta_corrected):.3f}")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle("Synthetic background-only frame: eta before/after illumination-offset correction")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "recon_biref_synthetic_bgcheck.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved -> {out_path}")

    return dict(eta_uncorrected_mean=float(np.nanmean(eta_uncorrected)),
                eta_corrected_mean=float(np.nanmean(eta_corrected)))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    part1_forward_round_trip()
    part2_background_offset_check()


if __name__ == "__main__":
    main()
