"""Chapter 4 dataset figures, PET vs PS, expanded set (1832 patches, 777 PET and 1055 PS). Read-only on existing files.
Figure 1 results/figures/fig_dataset_examples: 2 x 6 stored z0 patches, common grey limits.
Figure 2 results/figures/fig_channels (PET2) and fig_channels_ps (PS1): raw counts and background-divided ratios per analyser
         channel for one particle, with automatic checks.
(Figure 3, the raw frame with boxed detections, is not in this script: the task text was cut off.)
Sources: pipeline_output/manifest_recon_s0_z0_PETPS_exp.csv (track ids, clipped flag, detected-region areas), stored z0 patches
pipeline_output/patches_recon_s0_z0_PETPS_exp, stored per-channel z0 patches pipeline_output/patches_recon_z0pol4_PETPS_exp
(ratios x 255), raw recordings PET2_600.mat and PS1_600.mat (MATLAB 7.3). The per-channel background is recomputed exactly as
scripts/item43_pz_build.py built the stored patches (per-pixel median of up to 90 preceding raw frames on the mosaic, strided
demosaic I0 [1::2,1::2], I45 [0::2,1::2], I90 [0::2,0::2], I135 [1::2,0::2], I_c / (median_c + 1e-6)); no background image is stored.
Order: selection (tables and stored patches only) -> results/figures/fig_dataset_selection.txt -> rendering -> provenance file.
Pixel sizes: demosaiced 6.9 um (z0 patches are 64 x 64 px = 441.6 um), raw sensor 3.45 um."""
import os
import sys
import time

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import h5py  # noqa: E402

PO, FD = "pipeline_output", "results/figures"
MAN = f"{PO}/manifest_recon_s0_z0_PETPS_exp.csv"
Z0 = f"{PO}/patches_recon_s0_z0_PETPS_exp"
POL4 = f"{PO}/patches_recon_z0pol4_PETPS_exp"
SEL, PROV = f"{FD}/fig_dataset_selection.txt", f"{FD}/fig_dataset_provenance.txt"
H_DEM, W_DEM = 1024, 1224
SITES = [(1, 1), (0, 1), (0, 0), (1, 0)]
UM = 6.9
CH = ["I0", "I45", "I90", "I135"]
AREA = "region_px_predilation"  # detected-region area before dilation, pixels at 6.9 um (as in the S1 table)


def clean(t):
    assert "\u2014" not in t and ("signif" + "icant") not in t.lower() and ("valid" + "ated") not in t.lower()
    return t


def pct(v, x):
    v = np.asarray(v)
    return 100.0 * ((v < x).sum() + 0.5 * (v == x).sum()) / len(v)


def rec(f):
    return f.split("_")[0]


def select_fig1(m):
    L = ["FIGURE 1 selection (rules fixed before any image was looked at)",
         f"Typical = unclipped patch of the recording whose detected-region area ({AREA}) is nearest the median of that class's",
         "unclipped patches in that recording; ties by lowest manifest row; extra slots take the next nearest from a different track.",
         "Faint = unclipped patch of the class with the highest minimum z0 (least contrast). Clipped = the clipped patch of the class",
         f"whose {AREA} is nearest the median of the class's clipped patches (ties by lowest manifest row). A forced patch takes the free slot of its",
         "own recording (main slot first, then extra); if the recording has none free it takes an extra slot, whose recording changes."]
    rows = {}
    for c, slots in (("PET", ["PET1", "PET2", "PET3", "PET4", "x", "x"]), ("PS", ["PS1", "PS1", "PS1", "PS2", "PS2", "PS2"])):
        mc = m[m["class"] == c]
        mu = mc[~mc.clipped]
        tr = mc.groupby("recording").track_id.nunique().sort_values(ascending=False, kind="mergesort")
        if c == "PET":
            ex = list(tr.index[:2])
            L.append(f"{c}: tracks per recording {tr.to_dict()}; extra slots: {ex}")
            slots = ["PET1", "PET2", "PET3", "PET4", ex[0], ex[1]]
        faint = mu.sort_values(["minz", "row"], ascending=[False, True], kind="mergesort").iloc[0]
        cl = mc[mc.clipped]
        cl = cl.assign(d=(cl[AREA] - cl[AREA].median()).abs()).sort_values(["d", "row"], kind="mergesort").iloc[0]
        assign, why = [None] * 6, [""] * 6
        for kind, p in (("faint", faint), ("clipped", cl)):
            free = [i for i in range(6) if assign[i] is None and slots[i] == p.recording]
            if not free:
                free = [i for i in range(4 if c == "PET" else 0, 6) if assign[i] is None]
                if c == "PS":
                    free = [i for i in range(6) if assign[i] is None]
                slots[free[0]] = p.recording
            assign[free[0]], why[free[0]] = p, f"forced: {kind}"
        used_tracks = {p.track_id for p in assign if p is not None}
        for i in range(6):
            if assign[i] is not None:
                continue
            r = slots[i]
            pool = mu[mu.recording == r]
            med = pool[AREA].median()
            cand = pool.assign(d=(pool[AREA] - med).abs()).sort_values(["d", "row"], kind="mergesort")
            cand = cand[~cand.track_id.isin(used_tracks)]
            p = cand.iloc[0]
            assign[i], why[i] = p, f"typical (recording median area {med:g} px)"
            used_tracks.add(p.track_id)
        for i, p in enumerate(assign):
            L.append(f"  {c} panel {i + 1} [{why[i]}]: {p.recording}, frame {p.frame}, particle {p.particle_id}, track {p.track_id}, x {p.x}, y {p.y}, "
                     f"file {p.patch_file}; min z0 {p.minz:.4f} (percentile {pct(mc.minz, p.minz):.1f} in {c}), area {p[AREA]:g} px "
                     f"(percentile {pct(mc[AREA], p[AREA]):.1f}), clipped {bool(p.clipped)}, stored shape {p.shape_}")
        rows[c] = sorted(zip(assign, why), key=lambda t: (t[0].recording, t[0].row))
    return rows, L


def select_fig2(m):
    L = ["", "FIGURE 2 selection: unclipped patch of the named recording whose detected-region area is nearest the median of the",
         f"class's unclipped patches (all recordings), ties by lowest manifest row ({AREA})."]
    out = {}
    for c, r in (("PET", "PET2"), ("PS", "PS1")):
        mu = m[(m["class"] == c) & ~m.clipped]
        med = mu[AREA].median()
        pool = mu[mu.recording == r]
        p = pool.assign(d=(pool[AREA] - med).abs()).sort_values(["d", "row"], kind="mergesort").iloc[0]
        out[c] = p
        L.append(f"  {c} ({r}): class median area {med:g} px; chosen {p.recording}, frame {p.frame}, particle {p.particle_id}, track {p.track_id}, "
                 f"x {p.x}, y {p.y}, area {p[AREA]:g} px, file {p.patch_file}, min z0 {p.minz:.4f}")
    return out, L


def recover(p):
    """Raw mosaic window (uint8 counts) and background (median of up to 90 preceding raw frames) of the patch's 64 x 64 crop."""
    x, y, f = int(p.x), int(p.y), int(p.frame)
    ya, yb, xa, xb = max(0, y - 32), min(H_DEM, y + 32), max(0, x - 32), min(W_DEM, x + 32)
    Y0, Y1, X0, X1 = 2 * ya, 2 * yb, 2 * xa, 2 * xb
    with h5py.File(p.source_file, "r") as h:
        ds = h[[k for k in h.keys() if not k.startswith("#")][0]]
        sl = ds[max(0, f - 90):f + 1, 0, X0:X1, Y0:Y1]  # (frames, cols, rows)
    return sl[-1].T.astype(np.float64), np.median(sl[:-1], axis=0).T


def chans(w):
    return [w[r::2, c::2] for r, c in SITES]


def pad64(a, fill):
    out = np.full((64, 64), fill, np.float64)
    pad = np.ones((64, 64), bool)
    out[:a.shape[0], :a.shape[1]] = a
    pad[:a.shape[0], :a.shape[1]] = False
    return out, pad


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
                         "pdf.fonttype": 42, "image.composite_image": False})
    return plt


def scalebar(ax, n=64):
    import matplotlib.patheffects as pe
    px = 100.0 / UM
    ax.plot([3, 3 + px], [n - 4, n - 4], color="white", lw=1.6, solid_capstyle="butt")
    ax.text(3 + px / 2, n - 6, "100 \u03bcm", color="white", fontsize=8, ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])


def fig1(rows, lim, path):
    plt = setup_mpl()
    W, left, gap = 6.5, 0.42, 0.04
    s = (W - left - 5 * gap - 0.03) / 6
    top, under, rgap, bottom = 0.04, 0.17, 0.05, 0.26
    H = top + 2 * (s + under) + rgap + bottom
    fig = plt.figure(figsize=(W, H), facecolor="white")
    for i, c in enumerate(("PET", "PS")):
        y0 = H - top - (i + 1) * s - i * (under + rgap)
        for j, (p, why, img, pad) in enumerate(rows[c]):
            ax = fig.add_axes([(left + j * (s + gap)) / W, y0 / H, s / W, s / H])
            g = np.clip((img - lim[0]) / (lim[1] - lim[0]), 0, 1)
            ax.imshow(g, cmap="gray", vmin=0, vmax=1, interpolation="none")
            if pad.any():
                import matplotlib.patches as mp
                r1 = int(np.where(~pad.all(axis=1))[0].max()) + 1
                c1 = int(np.where(~pad.all(axis=0))[0].max()) + 1
                if c1 < 64:
                    ax.add_patch(mp.Rectangle((c1 - 0.5, -0.5), 64 - c1, 64, fill=False, ec="0.5", lw=0.8))
                if r1 < 64:
                    ax.add_patch(mp.Rectangle((-0.5, r1 - 0.5), 64, 64 - r1, fill=False, ec="0.5", lw=0.8))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            ax.text(0.5, -0.03, p.recording, transform=ax.transAxes, ha="center", va="top", fontsize=8)
            if i == 0 and j == 0:
                scalebar(ax)
            if j == 0:
                ax.text(-0.07, 0.5, c, transform=ax.transAxes, ha="right", va="center", fontsize=8, fontweight="bold")
    fig.text(left / W, 0.05 / H, f"grey limits (1st to 99th percentile, 12 patches): {lim[0]:.3f} to {lim[1]:.3f}",
             fontsize=8, ha="left", va="bottom")
    os.makedirs(FD, exist_ok=True)
    fig.savefig(path + ".pdf", facecolor="white")
    fig.savefig(path + ".png", dpi=300, facecolor="white")
    plt.close(fig)


def fig2(raw5, rat5, bgmed, lims, path):
    plt = setup_mpl()
    W, left, gap = 6.5, 0.36, 0.04
    s = (W - left - 4 * gap - 0.03) / 5
    top, ttl, under, rgap, bottom = 0.04, 0.2, 0.40, 0.04, 0.02
    H = top + ttl + 2 * (s + under) + rgap + bottom
    fig = plt.figure(figsize=(W, H), facecolor="white")
    titles = ["I0", "I45", "I90", "I135", "average"]
    for i, (imgs, lim, lab) in enumerate(((raw5, lims[0], "raw counts"), (rat5, lims[1], "divided by background"))):
        y0 = H - top - ttl - (i + 1) * s - i * (under + rgap)
        for j in range(5):
            ax = fig.add_axes([(left + j * (s + gap)) / W, y0 / H, s / W, s / H])
            ax.imshow(np.clip((imgs[j] - lim[0]) / (lim[1] - lim[0]), 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if i == 0:
                ax.text(0.5, 1.03, titles[j], transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
                ax.text(0.5, -0.03, f"bg median {bgmed[j]:.1f}", transform=ax.transAxes, ha="center", va="top", fontsize=8)
            else:
                ax.text(0.5, -0.03, "z0 (mean of four)" if j == 4 else "", transform=ax.transAxes, ha="center", va="top", fontsize=8)
            if i == 0 and j == 0:
                scalebar(ax)
            if j == 0:
                ax.text(-0.07, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=8, fontweight="bold", rotation=90)
        fig.text(left / W, (y0 - 0.23) / H, f"{lab}: one shared grey scale {lim[0]:.2f} to {lim[1]:.2f}", fontsize=8, ha="left", va="top")
    fig.savefig(path + ".pdf", facecolor="white")
    fig.savefig(path + ".png", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    t0 = time.time()
    redraw = "--redraw" in sys.argv  # layout fixes only: re-renders the figures; the selection must be unchanged; text files kept
    for p in (SEL, PROV):
        if os.path.exists(p) and not redraw:
            raise FileExistsError(p)
    m = pd.read_csv(MAN)
    assert len(m) == 1832 and (m["class"] == "PET").sum() == 777 and (m["class"] == "PS").sum() == 1055
    m["row"] = np.arange(len(m))
    m["recording"] = m.source_file.str.split("_").str[0]
    m["clipped"] = m.clipped.astype(bool)
    st = [np.load(f"{Z0}/{f}")[..., 0] for f in m.patch_file]
    m["minz"] = [float(a.min()) for a in st]
    m["shape_"] = [f"{a.shape[0]} x {a.shape[1]}" for a in st]
    assert ((m.shape_ != "64 x 64") == m.clipped).all()
    rows1, L1 = select_fig1(m)
    sel2, L2 = select_fig2(m)
    os.makedirs(FD, exist_ok=True)
    if redraw:
        assert "\n".join(L1 + L2) in open(SEL, encoding="utf-8").read(), "selection changed"
    else:
        open(SEL, "w", encoding="utf-8").write(clean("\n".join(L1 + L2 + ["", "FIGURE 3: not selected (the task text was cut off after 'Show the').",
                                                                         f"Written {time.strftime('%Y-%m-%d %H:%M')} before any rendering."])) + "\n")
    print("\n".join(L1 + L2))
    # Figure 1 render
    full = {}
    for c in rows1:
        full[c] = []
        for p, why in rows1[c]:
            a = np.load(f"{Z0}/{p.patch_file}")[..., 0].astype(np.float64)
            img, pad = pad64(a, float(np.median(a)))
            full[c].append((p, why, img, pad))
    pool = np.concatenate([a[~pad].ravel() for c in full for p, w, a, pad in full[c]] if False else
                          [img[~pad].ravel() for c in full for p, w, img, pad in full[c]])
    lim = (float(np.percentile(pool, 1)), float(np.percentile(pool, 99)))
    fig1(full, lim, f"{FD}/fig_dataset_examples")
    LP = ["PROVENANCE (figures from scripts/make_dataset_figures.py)", "",
          "Expanded set: pipeline_output/manifest_recon_s0_z0_PETPS_exp.csv, 1832 patches (777 PET, 1055 PS); stored z0 patches",
          f"{Z0} (float32, 64 x 64 at 6.9 um = 441.6 um wide; demosaiced pixel 6.9 um, raw sensor pixel 3.45 um; pixel size used: 6.9 um).",
          "", "FIGURE 1 (fig_dataset_examples): 2 x 6 stored z0 patches, one common grey scale. Limits = 1st and 99th percentile of all stored",
          f"pixels pooled over the 12 patches (pad excluded): {lim[0]:.5f} to {lim[1]:.5f}. Scale bar 100 um = {100 / UM:.2f} px on the first panel.",
          "Clipped patches (stored smaller than 64 x 64) are placed top-left in a 64 x 64 canvas (as in the training code), the pad is filled with the",
          "patch's own median value, and the pad edge is outlined in mid-grey. Pixels clipped by the limits per panel (low, high):"]
    for c in full:
        for k, (p, why, img, pad) in enumerate(full[c]):
            v = img[~pad]
            LP.append(f"  {c} panel {k + 1} {p.recording} frame {p.frame} particle {p.particle_id} [{why}]: {100 * (v < lim[0]).mean():.2f}% low, "
                      f"{100 * (v > lim[1]).mean():.2f}% high; stored shape {p.shape_}")
    # Figure 2 render with checks
    LC = ["", "FIGURE 2 (fig_channels, fig_channels_ps): raw crops and per-channel ratios of one particle, recovered from the recording."]
    for c, name in (("PET", "fig_channels"), ("PS", "fig_channels_ps")):
        p = sel2[c]
        raw_w, med_w = recover(p)
        rc, bc = chans(raw_w), chans(med_w)
        ratio = [r / (b + 1e-6) for r, b in zip(rc, bc)]
        stored4 = np.load(f"{POL4}/{p.patch_file}").astype(np.float64) / 255.0
        z0 = np.load(f"{Z0}/{p.patch_file}")[..., 0].astype(np.float64)
        d1 = float(np.abs(stored4.mean(axis=-1) - z0).max())
        d2 = float(max(np.abs(ratio[k] - stored4[..., k]).max() for k in range(4)))
        shapes_ok = all(r.shape == (64, 64) for r in rc)
        pa = lambda ok: "PASS" if ok else "FAIL"
        LC += [f"  {c} {p.recording} frame {p.frame} particle {p.particle_id} ({p.patch_file}): raw crop shape {rc[0].shape}, uint8 range "
               f"{min(r.min() for r in rc):.0f} to {max(r.max() for r in rc):.0f}",
               f"    CHECK mean of the four stored per-channel z0 patches (/255) equals the stored z0 patch to 1e-6: max abs difference {d1:.3g} -> {pa(d1 <= 1e-6)}",
               f"    CHECK raw crops / (recomputed background + 1e-6) reproduce the stored per-channel ratio patches to 1e-4: max abs difference {d2:.3g} -> {pa(d2 <= 1e-4)}",
               "    (the background is recomputed from the recording by the item 43 procedure; no background image is stored)"]
        raw5 = rc + [np.mean(rc, axis=0)]
        rat5 = [stored4[..., k] for k in range(4)] + [z0]
        bgm = [float(np.median(b)) for b in bc] + [float(np.median(np.mean(bc, axis=0)))]
        lim_raw = (float(min(a.min() for a in raw5)), float(max(a.max() for a in raw5)))
        pr = np.concatenate([a.ravel() for a in rat5])
        lim_rat = (float(np.percentile(pr, 1)), float(np.percentile(pr, 99)))
        fig2(raw5, rat5, bgm, (lim_raw, lim_rat), f"{FD}/{name}")
        LC += [f"    raw-counts row: one shared limit = min and max of the five panels: {lim_raw[0]:.1f} to {lim_raw[1]:.1f}; background medians "
               + ", ".join(f"{n} {v:.1f}" for n, v in zip(CH + ["mean"], bgm)),
               f"    ratio row: one shared limit = 1st and 99th percentile pooled over the five: {lim_rat[0]:.4f} to {lim_rat[1]:.4f}"]
    LC += ["", "FIGURE 3 (fig_raw_frame, fig_raw_frame_ps): not made; the task text was cut off after 'Show the'.",
           f"Written {time.strftime('%Y-%m-%d %H:%M')} ({time.time() - t0:.0f} s)."]
    if not redraw:
        open(PROV, "w", encoding="utf-8").write(clean("\n".join(LP + LC)) + "\n")
    print("\n".join(LP + LC))


if __name__ == "__main__":
    main()
