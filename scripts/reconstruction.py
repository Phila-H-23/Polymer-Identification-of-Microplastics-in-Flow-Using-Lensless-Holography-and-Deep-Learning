"""
reconstruction.py

Core angular-spectrum propagation and birefringence-retrieval functions
for the reconstruction pathway. No training here - this module is
imported by the validation script and (later, on request) by the
dataset-building step and train_compact_cnn.py's new --channels modes.

PHYSICAL PARAMETERS (as given directly by the user, not assumed):
  wavelength_vacuum = 405 nm
  medium index n = 1.33  -> wavelength used in the propagation kernel is
                             the MEDIUM wavelength, wavelength_vacuum / n,
                             since propagation happens through the sample
                             medium, not vacuum.
  pixel pitch = 6.9 um, stated explicitly for the DEMOSAICED (1224x1024)
                          channels, not the raw mosaic - used as-is, not
                          doubled.
  z ~= 2.25 mm (approximate; the z-sweep in reconstruction_validate.py
                exists specifically to pin this down better)

ANGULAR SPECTRUM METHOD:
  U(x,y,z) = IFFT2{ FFT2{U(x,y,0)} * H(fx,fy,z) }
  H(fx,fy,z) = exp(i*2*pi*z*kz),  kz = sqrt(1/wavelength^2 - fx^2 - fy^2)

  Evanescent band (fx^2+fy^2 > 1/wavelength^2): kz is purely imaginary.
  Confirmed directly (not assumed) that numpy's csqrt principal branch
  gives +j*sqrt(|arg|) for negative-real input, so for z>0 this correctly
  DECAYS (exp(-2*pi*z*|kz|)) - physically right for forward propagation.
  For z<0 (back-propagation, what this module is actually used for) the
  same term EXPLODES (exp(+2*pi*|z|*|kz|)), which is the well-known
  ill-posedness of inverting evanescent decay: that spatial-frequency
  band lost real information going forward, and no back-propagation
  recovers it - naive numerical inversion, fed noisy recorded intensity,
  would instead amplify sensor noise in that band without bound. Standard
  practice (and what this implementation does): HARD-ZERO the transfer
  function for any (fx,fy) with fx^2+fy^2 >= 1/wavelength^2, in both
  propagation directions, rather than let it decay/blow up. This is a
  plain angular-spectrum propagator (not the bandlimited/Matsushima
  variant) with that one evanescent-suppression safeguard - deviating
  further (e.g. adding the bandlimiting anti-aliasing mask) was not
  requested and isn't applied here.
"""
import numpy as np

WAVELENGTH_VACUUM_M = 405e-9
MEDIUM_INDEX = 1.33
WAVELENGTH_MEDIUM_M = WAVELENGTH_VACUUM_M / MEDIUM_INDEX
PIXEL_PITCH_M = 6.9e-6
DEFAULT_Z_M = 2.25e-3


def angular_spectrum_transfer(shape, z, wavelength=WAVELENGTH_MEDIUM_M,
                               dx=PIXEL_PITCH_M, dy=None):
    """Transfer function H(fx,fy,z) for a (Ny, Nx) grid, evanescent band
    hard-zeroed - see module docstring."""
    if dy is None:
        dy = dx
    Ny, Nx = shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)  # (Ny, Nx), matches field layout

    arg = (1.0 / wavelength) ** 2 - FX ** 2 - FY ** 2
    propagating = arg >= 0
    kz = np.zeros_like(arg, dtype=np.complex128)
    kz[propagating] = np.sqrt(arg[propagating].astype(np.complex128))
    H = np.zeros_like(arg, dtype=np.complex128)
    H[propagating] = np.exp(1j * 2 * np.pi * z * kz[propagating])
    return H


def angular_spectrum_propagate(field, z, wavelength=WAVELENGTH_MEDIUM_M,
                                dx=PIXEL_PITCH_M, dy=None, H=None):
    """field: (Ny, Nx) complex array at z=0. Returns the complex field
    propagated to distance z (z>0 forward/away from source, z<0 back-
    propagation toward the source). Pass a precomputed H (from
    angular_spectrum_transfer with the same shape/z) to skip rebuilding it."""
    field = np.asarray(field, dtype=np.complex128)
    if H is None:
        H = angular_spectrum_transfer(field.shape, z, wavelength, dx, dy)
    return np.fft.ifft2(np.fft.fft2(field) * H)


def rolling_median_background(frame_stack_getter, frame_idx, window=90):
    """frame_stack_getter(i) -> (H, W) array for raw frame i (any single
    quantity - a raw channel, or a derived quantity like A/B). Returns
    the per-pixel MEDIAN over frames [max(0, frame_idx-window), frame_idx)
    - note this is a MEDIAN, not the MEAN used by the existing pol4_bgcorr
    pathway (scripts/build_bgcorrected_dataset.py). The user's spec for
    this reconstruction pathway explicitly says "rolling median" for both
    the per-channel background (step 2) and the A/B background (step 3);
    flagged in the report rather than silently treated as identical to
    the older mean-based pol4_bgcorr background, which it is not.

    Returns (background, was_computed): was_computed=False for frame_idx
    0 (no preceding frames - same edge case as pol4_bgcorr, handled the
    same documented way: caller falls back to not correcting)."""
    lo, hi = max(0, frame_idx - window), frame_idx
    if hi <= lo:
        return None, False
    frames = np.stack([frame_stack_getter(i) for i in range(lo, hi)], axis=0)
    return np.median(frames, axis=0), True


def demosaic(snapshot):
    I_0 = snapshot[1::2, 1::2]
    I_45 = snapshot[0::2, 1::2]
    I_90 = snapshot[0::2, 0::2]
    I_135 = snapshot[1::2, 0::2]
    return I_0, I_45, I_90, I_135


def birefringence_eta_theta(I0, I45, I90, I135, A_bg=None, B_bg=None):
    """Direct port of birefringence_eta_theta.m (read verbatim from the
    project's copy of that file, not re-derived from the user's paraphrase
    - the .m file has one extra NaN condition the paraphrase omitted:
    theta is ALSO set NaN at p==0, i.e. wherever retardance is exactly
    zero, since the slow-axis azimuth is undefined there; eta is not
    affected by that condition, only theta).

    A_bg, B_bg: optional per-pixel background arrays (same shape as the
    inputs) subtracted from A and B BEFORE computing p/eta/theta - this
    step is NOT in Fraser's .m function; it's the addition specified for
    this pathway, sourced to Shribak & Oldenbourg 2003 Sec. 3.C, needed
    because the illumination is elliptical (I90 ~4.5x I0) so A/B have a
    nonzero background offset even with no sample birefringence present.

    Returns (eta, theta), both normalised to [0,1] (theta wrapped mod pi
    first) with NaN where invalid, exactly matching the .m function's
    normalisation.
    """
    I0 = np.asarray(I0, dtype=np.float64)
    I45 = np.asarray(I45, dtype=np.float64)
    I90 = np.asarray(I90, dtype=np.float64)
    I135 = np.asarray(I135, dtype=np.float64)

    D0 = I0 + I90
    D45 = I45 + I135
    with np.errstate(invalid="ignore"):
        D0 = np.where(D0 <= 0, np.nan, D0)
        D45 = np.where(D45 <= 0, np.nan, D45)

        A = (I0 - I90) / D0
        B = (I135 - I45) / D45

        if A_bg is not None:
            A = A - A_bg
        if B_bg is not None:
            B = B - B_bg

        p = np.hypot(A, B)
        valid = np.isfinite(p) & (p <= 1)
        p = np.where(valid, p, np.nan)

        delta = np.arcsin(p)
        theta = np.mod(0.5 * np.arctan2(A, B), np.pi)
        theta = np.where(valid & (p != 0), theta, np.nan)

        eta = delta / (np.pi / 2)
        theta = theta / np.pi

    return eta, theta
