"""Figure: what propagation does to the input (companion to Table 4.9). No training.
Same patch in all three columns, from the original 561-patch PET vs PS set (342 PET, 219 PS):
  (1) z0, unpropagated: pipeline_output/patches_recon_s0_z0, manifest pipeline_output/manifest_recon_s0_z0.csv (the
      "z0, same set" arm of item 41 stage 3, built by scripts/build_reconstructed_dataset.py --z 0 --suffix _z0 --s0-only)
  (2) reconstruction as built (V1, 2.25 mm): pipeline_output/patches_recon_s0, manifest pipeline_output/manifest_recon_s0.csv
  (3) reconstruction rebuilt (V3): pipeline_output/patches_recon_V3_PETPS, manifest pipeline_output/manifest_recon_V3_PETPS.csv
Order of work: Step 1 assertion (same recording, frame, x, y for every patch in all three manifests); Step 2 selection
from tables only (results/recon_fidelity/s1_detection_stats.csv: clipped flag and detected region area before dilation),
written to results/figures/fig_recon_inputs_selection.txt before any patch pixel is loaded; Step 3 grey range per column =
0.5th to 99.5th percentile of all pixels of that variant over the 561 patches; Step 5 numbers; then the figures.
Writes results/figures/fig_recon_inputs.pdf / .png (600 dpi), fig_recon_inputs_backup.png, fig_recon_inputs_selection.txt
and fig_recon_inputs_provenance.txt (new files only)."""
import os
import sys
import time

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PO, FD = "pipeline_output", "results/figures"
VAR = [("z0", "z0, unpropagated", f"{PO}/patches_recon_s0_z0", f"{PO}/manifest_recon_s0_z0.csv"),
       ("V1", "Reconstruction, as built (V1)", f"{PO}/patches_recon_s0", f"{PO}/manifest_recon_s0.csv"),
       ("V3", "Reconstruction, rebuilt (V3)", f"{PO}/patches_recon_V3_PETPS", f"{PO}/manifest_recon_V3_PETPS.csv")]
EXP = (f"{PO}/patches_recon_s0_z0_PETPS_exp", f"{PO}/manifest_recon_s0_z0_PETPS_exp.csv")
S1 = "results/recon_fidelity/s1_detection_stats.csv"
SEL, PROV, FIG = f"{FD}/fig_recon_inputs_selection.txt", f"{FD}/fig_recon_inputs_provenance.txt", f"{FD}/fig_recon_inputs"
CLASSES = ("PET", "PS")
KEYS = ["source_file", "frame", "x", "y", "particle_id"]


def clean(text):
    assert "\u2014" not in text and ("signif" + "icant") not in text.lower() and ("valid" + "ated") not in text.lower()
    return text


def load(pdir, f):
    a = np.load(os.path.join(pdir, f))
    assert a.ndim == 3 and a.shape[2] == 1, a.shape
    return a[..., 0].astype(np.float64)


def prank(v, x):
    v = np.asarray(v)
    return 100.0 * ((v < x).sum() + 0.5 * (v == x).sum()) / len(v)


def draw(rows, ranges, path, labels, pdf=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                         "image.composite_image": False, "pdf.fonttype": 42})
    W = 16 / 2.54
    left, gap, right = 0.42, 0.06, 0.04
    s = (W - left - right - 2 * gap) / 3
    top, rgap, under = 0.24, 0.06, 0.22
    n = len(rows)
    H = top + n * s + (n - 1) * rgap + under
    fig = plt.figure(figsize=(W, H), facecolor="white")
    for i, row in enumerate(rows):
        y0 = H - top - (i + 1) * s - i * rgap
        for j, (v, title, _, _) in enumerate(VAR):
            lo, hi = ranges[v]
            g = np.clip((row[v] - lo) / (hi - lo), 0, 1)
            ax = fig.add_axes([(left + j * (s + gap)) / W, y0 / H, s / W, s / H])
            ax.imshow(np.repeat(g[..., None], 3, axis=2), interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if i == 0:
                ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)
            if i == n - 1:
                ax.text(0.5, -0.03, f"grey: {lo:.3f} to {hi:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=7)
            if j == 0:
                ax.text(-0.05, 0.5, labels[i], transform=ax.transAxes, ha="right", va="center", fontsize=8, fontweight="bold",
                        rotation=90 if len(labels[i]) > 3 else 0)
    os.makedirs(FD, exist_ok=True)
    fig.savefig(path + ".png", dpi=600, facecolor="white")
    if pdf:
        fig.savefig(path + ".pdf", facecolor="white")
    plt.close(fig)


def main():
    t0 = time.time()
    for p in (SEL, PROV, FIG + ".png", FIG + ".pdf", FIG + "_backup.png"):
        if os.path.exists(p):
            raise FileExistsError(p)
    # STEP 1: the three manifests, restricted to the 561 PET and PS patches, must agree on recording, frame, x, y.
    man = {}
    for v, _, pdir, mf in VAR:
        m = pd.read_csv(mf)
        man[v] = m[m["class"].isin(CLASSES)].reset_index().rename(columns={"index": "manifest_row"})
    base = man["V1"]
    assert len(base) == 561 and (base["class"] == "PET").sum() == 342 and (base["class"] == "PS").sum() == 219
    L1 = ["STEP 1: sources (561 PET vs PS patches, 342 PET, 219 PS; patch files matched by name)"]
    for v, title, pdir, mf in VAR:
        L1.append(f"  {v:3s} {title:31s} folder {pdir}, manifest {mf}")
    bad = 0
    for v in ("z0", "V3"):
        j = base.merge(man[v], on="patch_file", suffixes=("", "_o"))
        assert len(j) == 561, (v, len(j))
        diff = np.zeros(len(j), bool)
        for k in KEYS + ["class"]:
            diff |= (j[k].astype(str) != j[k + "_o"].astype(str)).values
        bad += int(diff.sum())
        L1.append(f"  {v} against V1: {len(j)} patches matched, rows with a different recording, frame, x, y, particle id or class: {int(diff.sum())}")
    if bad:
        print("\n".join(L1))
        raise SystemExit("assertion failed: the three manifests differ; stopped")
    L1.append("  Assertion: PASS (same recording, frame, x, y and particle id in all three manifests for all 561 patches)")
    # STEP 2: selection from tables only.
    s1 = pd.read_csv(S1)
    s1 = s1[s1.kind == "real"].set_index("patch_file")
    t = base.join(s1[["clipped", "region_px_predilation"]], on="patch_file")
    assert t.region_px_predilation.notna().all()
    L2 = ["STEP 2: selection (fixed rule, from tables only; written before any patch pixel was loaded or any image rendered)",
          f"Rule: per class, over the unclipped patches of the 561 set (clipped flag of {S1}, kind real), the patch whose",
          "detected region area before dilation (region_px_predilation, pixels at 6.9 um, same table) is nearest the class median;",
          "ties broken by the lowest row of the V1 manifest (pipeline_output/manifest_recon_s0.csv). The next two nearest are backups."]
    pick = {}
    for c in CLASSES:
        tc = t[t["class"] == c]
        u = tc[~tc.clipped.astype(bool)].copy()
        med = float(u.region_px_predilation.median())
        u["dist"] = (u.region_px_predilation - med).abs()
        u = u.sort_values(["dist", "manifest_row"], kind="mergesort")
        pick[c] = u.head(3)
        L2.append(f"{c}: {len(tc)} patches, {int(tc.clipped.astype(bool).sum())} clipped excluded, {len(u)} unclipped; median area "
                  f"{med:g} px")
        for r, (_, x) in zip(("chosen  ", "backup 1", "backup 2"), pick[c].iterrows()):
            L2.append(f"  {r}: {x.source_file}, frame {x.frame}, particle {x.particle_id}, x {x.x}, y {x.y}, area {x.region_px_predilation:g} px "
                      f"(median {med:g}), split {x.split}, split_video_holdout {x.split_video_holdout}, file {x.patch_file}, "
                      f"manifest row {x.manifest_row}")
    os.makedirs(FD, exist_ok=True)
    open(SEL, "w", encoding="utf-8").write(clean("\n".join(L1 + [""] + L2 + ["", f"Written {time.strftime('%Y-%m-%d %H:%M')}."])) + "\n")
    print("\n".join(L1 + L2))
    # STEP 3: grey range per column from all pixels of the variant over the 561 patches.
    pix = {v: [] for v, *_ in VAR}
    img = {v: {} for v, *_ in VAR}
    for f in base.patch_file:
        for v, _, pdir, _ in VAR:
            a = load(pdir, f)
            pix[v].append(a.ravel())
            img[v][f] = a
    ranges = {v: (float(np.percentile(np.concatenate(pix[v]), 0.5)), float(np.percentile(np.concatenate(pix[v]), 99.5))) for v in pix}
    L3 = ["STEP 3: grey range per column = 0.5th to 99.5th percentile of all pixels of the variant over the 561 patches "
          "(the same for both rows of a column):"] + [f"  {v}: {lo:.4f} to {hi:.4f}" for v, (lo, hi) in ranges.items()]
    ep, em = EXP
    me = pd.read_csv(em)
    common = sorted(set(base.patch_file) & set(me.patch_file))
    dz = max(float(np.abs(load(ep, f) - img["z0"][f]).max()) for f in common)
    L3.append(f"  (information) the expanded z0 folder {ep} holds {len(common)} of the 561 patch files; largest absolute pixel "
              f"difference against the 561-set z0 patches: {dz:.3g}")
    # STEP 5: information only.
    L5 = ["STEP 5 (information only; the selection is not changed): z0 raw std and the range of its 64 column means, each with",
          "its percentile among the class's unclipped patches; Pearson correlation of the three panels of the patch."]
    for c in CLASSES:
        tc = t[(t["class"] == c) & ~t.clipped.astype(bool)]
        sd = np.array([img["z0"][f].std() for f in tc.patch_file])
        cr = np.array([np.ptp(img["z0"][f].mean(axis=0)) for f in tc.patch_file])
        for r, (_, x) in zip(("chosen  ", "backup 1", "backup 2"), pick[c].iterrows()):
            z, a1, a3 = (img[v][x.patch_file].ravel() for v in ("z0", "V1", "V3"))
            s_, c_ = img["z0"][x.patch_file].std(), np.ptp(img["z0"][x.patch_file].mean(axis=0))
            L5.append(f"  {c} {r}: z0 std {s_:.5f} (percentile {prank(sd, s_):.1f}), column-mean range {c_:.5f} (percentile "
                      f"{prank(cr, c_):.1f}); r(z0, V1) {np.corrcoef(z, a1)[0, 1]:.3f}, r(z0, V3) {np.corrcoef(z, a3)[0, 1]:.3f}, "
                      f"r(V1, V3) {np.corrcoef(a1, a3)[0, 1]:.3f}")
    print("\n".join(L3 + L5))
    # STEP 4: render.
    rows = lambda k: [{v: img[v][pick[c].iloc[k].patch_file] for v, *_ in VAR} for c in CLASSES]
    draw(rows(0), ranges, FIG, list(CLASSES))
    draw(rows(1) + rows(2), ranges, FIG + "_backup", ["PET, backup 1", "PS, backup 1", "PET, backup 2", "PS, backup 2"], pdf=False)
    LP = ["PROVENANCE: fig_recon_inputs (.pdf, .png) and fig_recon_inputs_backup.png",
          "Patch set: the original 561-patch PET vs PS set (342 PET, 219 PS), the set the Table 4.9 comparison ran on.",
          "Variants (columns, in the order of Table 4.9):"]
    LP += [f"  {title}: folder {pdir}, manifest {mf}" for v, title, pdir, mf in VAR]
    LP += ["  z0 = unpropagated, background-divided, four analyser channels averaged; V1 = the stored recon_s0 patches as built",
           "  (2.25 mm); V3 = rebuilt (bilinear demosaic of each channel, reflect padding, propagation).", "Rows:"]
    for c in CLASSES:
        for r, (_, x) in zip(("main figure", "backup 1", "backup 2"), pick[c].iterrows()):
            LP.append(f"  {c} {r}: recording {x.source_file}, frame {x.frame}, particle id {x.particle_id}, x {x.x}, y {x.y}, "
                      f"file {x.patch_file}")
    LP += ["", *L2[1:4], "", *L3, "", *L5, "", f"Written {time.strftime('%Y-%m-%d %H:%M')} ({time.time() - t0:.0f} s)."]
    open(PROV, "w", encoding="utf-8").write(clean("\n".join(LP)) + "\n")


if __name__ == "__main__":
    main()
