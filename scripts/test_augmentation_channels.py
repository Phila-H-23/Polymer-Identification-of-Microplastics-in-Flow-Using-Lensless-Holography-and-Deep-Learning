"""
test_augmentation_channels.py

Unit test of the rotation augmentation for every channel mode. Translation,
gain jitter and noise are disabled so only rotation + channel handling show.
For each mode and each rotation the augmentation can produce (0 and 180 deg):
  - prints the effective model input: source of each output channel and its
    orientation (from a marker pixel)
  - asserts: only k in {0, 2} is reachable; the k=0 input equals the
    un-augmented input; the k=2 input equals the k=0 input rotated by 180 deg
    (channel content unchanged, purely spatial).
Synthetic labelled patch: I0=40, I45=80, I90=120, I135=160 (/255), marker at
(row 4, col 12) in every channel; recon modes get their own float patches.
Pre-fix behaviour (odd quarter-turns with channel swaps) is recorded in
results/test_augmentation_channels_before_fix.txt.
"""
import os
import sys
import importlib.util
import tempfile
import numpy as np
import pandas as pd
import torch

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
spec = importlib.util.spec_from_file_location("tcc", os.path.join(DATA_DIR, "train_compact_cnn.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
os.chdir(DATA_DIR)  # module uses relative paths for vnorm / flat-field references

LEVELS = np.array([40, 80, 120, 160], dtype=np.float64)
SRC = ["I0", "I45", "I90", "I135"]
MARK = (4, 12)
VIDEO = "PE1_600.mat"
C2I = {c: i for i, c in enumerate(["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"])}
MODES_4CH = ["intensity", "intensity_ff", "intensity_bgcorr_vnorm", "i0_bgcorr_vnorm", "i45_bgcorr_vnorm",
             "i0i90_bgcorr_vnorm", "pol4", "pol4_ff", "pol4_bgcorr", "pol4_bgcorr_vnorm", "stokes",
             "stokes_bgcorr_vnorm"]
def prop_pixel(delta, phi_deg):
    """Shribak circular-illumination intensities for retardance delta and azimuth phi, A_bg=B_bg=0."""
    phi = np.deg2rad(phi_deg)
    A, B = np.sin(2 * phi) * np.sin(delta), np.cos(2 * phi) * np.sin(delta)
    return np.array([(1 + A) / 2, (1 - B) / 2, (1 - A) / 2, (1 + B) / 2, 0.0, 0.0], np.float32)


class _NoJitter:
    @staticmethod
    def randint(a, b):
        return 0

    @staticmethod
    def random():
        return 0.5


def model_input(df, pdir, mode, k=None):
    ds = m.PatchDataset(df, pdir, C2I, channels=mode, augment=k is not None)
    if k is not None:
        ds._rotation_k = lambda: k
    real_random, real_randn = m.random, torch.randn_like
    m.random, torch.randn_like = _NoJitter, torch.zeros_like
    try:
        return ds[0][0].numpy()
    finally:
        m.random, torch.randn_like = real_random, real_randn


def orientation(ch):
    pos = np.unravel_index(np.argmax(np.abs(ch - np.median(ch))), ch.shape)
    ref = np.zeros((64, 64))
    ref[MARK] = 1
    for k in range(4):
        if np.unravel_index(np.argmax(np.rot90(ref, k)), ref.shape) == pos:
            return f"{90 * k}"
    return "?"


def source_label(mode, j, x0):
    """Source of output channel j, read from its background level at k=0."""
    if mode.startswith("intensity") or mode == "recon_s0":
        return "mean4" if mode != "recon_s0" else "S0"
    if mode.startswith("stokes"):
        return ["S", "q", "u"][j]
    if mode == "recon_biref":
        return ["eta", "eta*cos", "eta*sin"][j]
    if mode in ("i0_bgcorr_vnorm", "i45_bgcorr_vnorm", "i0i90_bgcorr_vnorm"):
        idx = {"i0_bgcorr_vnorm": [0], "i45_bgcorr_vnorm": [1], "i0i90_bgcorr_vnorm": [0, 2]}[mode]
        return SRC[idx[j]]
    return SRC[j]


def check(mode, df, pdir):
    reach = sorted({m.PatchDataset(pd.DataFrame(), ".", C2I, channels=mode)._rotation_k() for _ in range(400)})
    x_none = model_input(df, pdir, mode)
    x0 = model_input(df, pdir, mode, 0)
    x2 = model_input(df, pdir, mode, 2)
    ok_reach = reach == [0, 2]
    ok_k0 = np.allclose(x0, x_none)
    ok_k2 = np.allclose(x2, np.rot90(x0, 2, axes=(1, 2)), atol=1e-6)
    # Level check: k=2 channel j has the same background level as k=0 channel j (no swap)
    ok_levels = all(np.isclose(np.median(x2[j]), np.median(x0[j]), atol=1e-5) for j in range(x0.shape[0]))
    cells = []
    for x in (x0, x2):
        cells.append(", ".join(f"{source_label(mode, j, x0)}={np.median(x[j]):+.3f}@{orientation(x[j])}"
                               for j in range(x.shape[0])))
    ok = ok_reach and ok_k0 and ok_k2 and ok_levels
    print(f"{mode:22s} {cells[0]:52s} {cells[1]:52s} {str(reach):7s} {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        p = np.broadcast_to(LEVELS.astype(np.uint8), (64, 64, 4)).copy()
        p[MARK] = 250
        np.save(os.path.join(tmp, "p.npy"), p)
        df = pd.DataFrame([{"patch_file": "p.npy", "class": "PE", "x": 500, "y": 500, "source_file": VIDEO}])
        print(f"{'mode':22s} {'k=0 (0 deg): channel=level@orientation':52s} {'k=2 (180 deg)':52s} {'reach':7s} result")
        for mode in MODES_4CH:
            results.append(check(mode, df, tmp))
        r = np.full((64, 64, 1), 1.0, np.float32)
        r[MARK] = 2.0
        np.save(os.path.join(tmp, "r.npy"), r)
        results.append(check("recon_s0", df.assign(patch_file="r.npy"), tmp))

        # recon_biref loads 6-ch [I0p, I45p, I90p, I135p, A_bg, B_bg] and converts at load time.
        prop = np.broadcast_to(prop_pixel(0.3, 30), (64, 64, 6)).copy()
        prop[MARK] = prop_pixel(0.6, 120)
        np.save(os.path.join(tmp, "b.npy"), prop)
        dfb = df.assign(patch_file="b.npy")
        results.append(check("recon_biref", dfb, tmp))
        x0 = model_input(dfb, tmp, "recon_biref", 0)
        eta, th = 0.3 / (np.pi / 2), 30 / 180
        expect = [eta, eta * np.cos(2 * np.pi * th), eta * np.sin(2 * np.pi * th)]
        ok_vals = np.allclose([np.median(x0[j]) for j in range(3)], expect, atol=1e-5)
        print(f"{'recon_biref values':22s} got {[round(float(np.median(x0[j])), 4) for j in range(3)]} "
              f"expected {[round(v, 4) for v in expect]} {'PASS' if ok_vals else 'FAIL'}")
        results.append(ok_vals)
    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} modes PASS")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
