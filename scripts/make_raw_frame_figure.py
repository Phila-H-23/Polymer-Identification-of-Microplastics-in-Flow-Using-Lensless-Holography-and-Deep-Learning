"""Figure 3: one raw frame with the kept detection boxes, PET2 (fig_raw_frame) and PS1 (fig_raw_frame_ps). Read-only on existing files.
Frame pick (fixed before rendering, written to results/figures/fig_raw_frame_selection.txt): among the frames used for the expanded
set (results/expanded_PETPS_exp/part1_frames.csv, rows of the recording) the frame whose number of kept detections (rows of
pipeline_output/manifest_recon_s0_z0_PETPS_exp.csv in that frame, 0 if none) is nearest the recording's median; ties by lowest frame.
Image: four-channel mean of the strided-demosaiced raw frame (1224 x 1024 at 6.9 um; I0 [1::2,1::2], I45 [0::2,1::2], I90 [0::2,0::2],
I135 [1::2,0::2]), grey limits 1st to 99th percentile of the frame. Boxes: bbox_w_px x bbox_h_px centred on (x, y) from the manifest.
Rejected detections: the pipeline saved only their counts (part1_frames.csv), not their positions, so they are not drawn.
One 64 x 64 patch outline on the kept detection with the lowest manifest row whose patch is unclipped. Scale bar 500 um = 72.46 px."""
import os
import sys
import time

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import h5py  # noqa: E402

FD = "results/figures"
SEL, PROV = f"{FD}/fig_raw_frame_selection.txt", f"{FD}/fig_raw_frame_provenance.txt"
SITES = [(1, 1), (0, 1), (0, 0), (1, 0)]
UM = 6.9


def clean(t):
    assert "\u2014" not in t and ("signif" + "icant") not in t.lower() and ("valid" + "ated") not in t.lower()
    return t


def pick(m, pf, rec):
    fr = pf[pf.video == rec].frame.astype(int).tolist()
    cnt = m[m.source_file == rec].groupby("frame").size().reindex(fr, fill_value=0)
    med = float(cnt.median())
    d = pd.DataFrame(dict(frame=cnt.index, n=cnt.values)).assign(d=lambda t: (t.n - med).abs()).sort_values(["d", "frame"], kind="mergesort")
    return int(d.iloc[0].frame), med, len(fr)


def load_frame(rec, f):
    with h5py.File(rec, "r") as h:
        ds = h[[k for k in h.keys() if not k.startswith("#")][0]]
        raw = ds[f, 0, :, :].T.astype(np.float64)
    assert raw.shape == (2048, 2448), raw.shape
    return np.mean([raw[r::2, c::2] for r, c in SITES], axis=0)


def render(img, lim, rows, pr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    import matplotlib.patheffects as pe
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "pdf.fonttype": 42})
    W = 6.5
    H = W * img.shape[0] / img.shape[1] + 0.3
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0.0, 0.3 / H, 1.0, (H - 0.3) / H])
    ax.imshow(np.clip((img - lim[0]) / (lim[1] - lim[0]), 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="none")
    for r in rows.itertuples(index=False):
        ax.add_patch(mp.Rectangle((r.x - r.bbox_w_px / 2, r.y - r.bbox_h_px / 2), r.bbox_w_px, r.bbox_h_px, fill=False, ec="#00e5ff", lw=0.6))
    ax.add_patch(mp.Rectangle((pr.x - 32, pr.y - 32), 64, 64, fill=False, ec="#ffd400", lw=0.8))
    px = 500.0 / UM
    ax.plot([30, 30 + px], [img.shape[0] - 30] * 2, color="white", lw=2.0, solid_capstyle="butt")
    ax.text(30 + px / 2, img.shape[0] - 38, "500 \u03bcm", color="white", fontsize=8, ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    h1 = mp.Patch(fill=False, ec="#00e5ff", lw=0.8, label="kept detection (detection box)")
    h2 = mp.Patch(fill=False, ec="#ffd400", lw=0.8, label="64 x 64 patch cut")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(path + ".pdf", facecolor="white")
    fig.savefig(path + ".png", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    redraw = "--redraw" in sys.argv  # layout fix only: re-render; the selection must be unchanged; text files kept
    for p in (SEL, PROV):
        if os.path.exists(p) and not redraw:
            raise FileExistsError(p)
    m = pd.read_csv("pipeline_output/manifest_recon_s0_z0_PETPS_exp.csv")
    m["row"] = np.arange(len(m))
    pf = pd.read_csv("results/expanded_PETPS_exp/part1_frames.csv")
    L = ["FIGURE 3 selection (fixed before rendering): the frame used for the expanded set whose number of kept detections is nearest the",
         "recording's median (frames with no kept detection count as 0); ties by lowest frame. Patch outline on the kept detection of that frame",
         "with the lowest manifest row whose patch is unclipped."]
    todo = []
    for rec, name in (("PET2_600.mat", "fig_raw_frame"), ("PS1_600.mat", "fig_raw_frame_ps")):
        f, med, nfr = pick(m, pf, rec)
        rows = m[(m.source_file == rec) & (m.frame == f)]
        pr = rows[~rows.clipped.astype(bool)].sort_values("row").iloc[0]
        st = pf[(pf.video == rec) & (pf.frame == f)].iloc[0]
        L.append(f"  {rec}: {nfr} frames used, median kept count {med:g}; chosen frame {f} with {len(rows)} kept detections; patch outline on "
                 f"particle {pr.particle_id} ({pr.patch_file}, x {pr.x}, y {pr.y})")
        todo.append((rec, name, f, med, nfr, rows, pr, st))
    os.makedirs(FD, exist_ok=True)
    if redraw:
        assert "\n".join(L) in open(SEL, encoding="utf-8").read(), "selection changed"
    else:
        open(SEL, "w", encoding="utf-8").write(clean("\n".join(L + ["", f"Written {time.strftime('%Y-%m-%d %H:%M')} before any rendering."])) + "\n")
    print("\n".join(L))
    LP = ["PROVENANCE (scripts/make_raw_frame_figure.py)", "",
          "Image: four-channel mean of the demosaiced raw frame, 1224 x 1024 px at 6.9 um (raw sensor pixel 3.45 um); scale bar 500 um = 72.46 px.",
          "Boxes: bbox_w_px x bbox_h_px centred on (x, y) of pipeline_output/manifest_recon_s0_z0_PETPS_exp.csv, one colour (cyan), thin lines.",
          "Rejected detections: NOT drawn. The pipeline kept only per-frame counts of rejected detections (results/expanded_PETPS_exp/part1_frames.csv);",
          "their positions were not saved (the SAVE_REJECTED branches were removed in scripts/extract_PETPS_exp.py), so they cannot be drawn without rerunning detection.",
          "Patch outline (yellow) on one kept detection shows the 64 x 64 cut at 6.9 um (441.6 um)."]
    for rec, name, f, med, nfr, rows, pr, st in todo:
        img = load_frame(rec, f)
        lim = (float(np.percentile(img, 1)), float(np.percentile(img, 99)))
        render(img, lim, rows, pr, f"{FD}/{name}")
        rej = int(st.n_detections - st.n_final)
        LP += ["", f"{name}: {rec}, frame {f}. Grey limits (1st to 99th percentile of the frame): {lim[0]:.2f} to {lim[1]:.2f} counts.",
               f"  Kept detections drawn: {len(rows)} (frame table n_final {int(st.n_final)}). Rejected detections (not drawn): {rej}",
               f"  = {int(st.n_debris_still_elevated)} still elevated at the comparison frame + {int(st.n_debris_low_displacement)} low displacement + "
               f"{int(st.n_recurrence_dropped)} dropped by the recurrence check, of {int(st.n_detections)} detections ({int(st.n_regions)} regions, "
               f"{int(st.n_too_small)} too small, {int(st.n_too_large)} too large). Frames used for this recording: {nfr}, median kept count {med:g}.",
               f"  Patch outline on particle {pr.particle_id} (x {pr.x}, y {pr.y}); clipped patches in this frame: {int(rows.clipped.astype(bool).sum())}."]
    LP.append(f"Written {time.strftime('%Y-%m-%d %H:%M')}.")
    if not redraw:
        open(PROV, "w", encoding="utf-8").write(clean("\n".join(LP)) + "\n")
    print("\n".join(LP))


if __name__ == "__main__":
    main()
