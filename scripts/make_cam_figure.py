"""CAM figure for the report (no training): z0 expanded set, PET vs PS, original direction (test recordings PET2, PS1).
CAM code reused unchanged from scripts/j3_cam_z0_exp.py (cam_for_true_class: block4 features, last linear layer's
weights of the TRUE class, bilinear upsampling 8 x 8 -> 64 x 64; then ReLU and sum-to-1 as in its run_cams(); box from
its box_mask(); particle-free patches carry the box of their paired real patch). Checkpoints as in J3: real patches
with results/real_tc_PETvsPS_z0_exp_s0, particle-free patches with results/bg_tc_PETvsPS_z0_exp_s0 (the
particle-free-trained checkpoint). Inputs are the evaluation tensors of the training script's loader (PatchDataset,
training-split channel statistics, no augmentation), built as in run_cams().
Steps: check (a) class means of results/cam_z0_shares_exp.csv; selection rule from the CSV, manifests and predictions
only (no image looked at); check (b) recomputed shares of the chosen panels; then the figure.
Writes results/cam_figure_selection.txt, results/cam_figure_panels.csv, results/figures/fig_cam_pet_ps.pdf / .png
(600 dpi) and results/figures/fig_cam_pet_ps_backup.png.
Revision 1 (2026-10-21, asked by Phila): the panels show the stored raw z0 ratio patches on one grey scale (1st to 99th
percentile of their pooled pixels); CAMs are still computed on each checkpoint's standardised inputs; turbo colour map;
colour maximum = 99th percentile of the pooled pixels of the four overlays, rounded up, values above it saturate; the
panel table gains each checkpoint's mean and std. A rerun keeps the selection: it stops if the chosen files, shares or
predictions differ from the existing results/cam_figure_panels.csv.
usage: python scripts/make_cam_figure.py"""
import io
import os
import sys
import json
import time
import math

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import j3_cam_z0_exp as j3  # noqa: E402  (CAM code, box code, runs, manifests: unchanged)
import eval_track_ci_exp as ev  # noqa: E402

SEL, PANELS = "results/cam_figure_selection.txt", "results/cam_figure_panels.csv"
FIG = "results/figures/fig_cam_pet_ps"
STATED = {("real", "PET"): 0.5168, ("particle-free", "PET"): 0.3344, ("real", "PS"): 0.8053, ("particle-free", "PS"): 0.3104}
CLASSES = ("PET", "PS")


def setup(run_dir):
    """The dataset and model of j3.run_cams() for one checkpoint (same lines, kept per patch)."""
    saved = json.load(open(os.path.join(run_dir, "results.json")))
    a = saved["args"]
    m = ev.module("train_compact_cnn.py")
    split_col = m.SPLIT_CONFIG[a["split"]]["split_column"]
    man = pd.read_csv(a["manifest"])
    man = man[man["class"].isin(a["classes"].split(","))]
    classes = saved["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    train_df, test_df = man[man[split_col] == "train"], man[man[split_col] == "test"].reset_index(drop=True)
    mean, std = m.compute_channel_stats(train_df, a["patch_dir"], c2i, a["channels"], patch_norm=a.get("patch_norm", False))
    model = m.CompactHoloCNN(1, len(classes), a["width"], a["dropout"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    model.eval()
    ds = m.PatchDataset(test_df, a["patch_dir"], c2i, channels=a["channels"], mean=mean, std=std,
                        patch_norm=a.get("patch_norm", False))
    return dict(run=run_dir, pdir=a["patch_dir"], test=test_df, ds=ds, model=model, classes=classes, mean=float(mean.flatten()[0]),
                std=float(std.flatten()[0]), idx={f: i for i, f in enumerate(test_df.patch_file)})


def raw_box(row):
    """Box bounds in the canvas before clipping, by the formula of j3.box_mask()."""
    x, y, w, h = int(row.x), int(row.y), float(row.bbox_w_px), float(row.bbox_h_px)
    ox, oy = max(0, x - 32), max(0, y - 32)
    c0, c1 = int(np.floor(x - w / 2 - ox + 0.5)), int(np.floor(x + w / 2 - ox + 0.5))
    r0, r1 = int(np.floor(y - h / 2 - oy + 0.5)), int(np.floor(y + h / 2 - oy + 0.5))
    return r0, r1, c0, c1


def panel(S, patch_file, box):
    """Loader tensor, CAM (ReLU, sum 1), prediction and probability of one test patch under checkpoint S."""
    x, label = S["ds"][S["idx"][patch_file]]
    raw = np.load(os.path.join(S["pdir"], patch_file))  # stored float32 z0 ratio patch, before standardisation
    assert raw.shape == (64, 64, 1), raw.shape  # one channel, stored as 64 x 64 x 1
    raw = raw[..., 0]
    dev = float(np.abs((raw - S["mean"]) / S["std"] - x[0].numpy()).max())  # loader tensor = (raw - mean) / std
    cam, pred, grid = j3.cam_for_true_class(S["model"], x[None], label)
    c = np.maximum(cam, 0.0)
    s = c.sum()
    with torch.no_grad():
        prob = torch.softmax(S["model"](x[None]), 1)[0]
    return dict(x=x[0].numpy(), raw=raw, dev=dev, u=c / s * 4096 if s > 0 else None, share=float(c[box].sum() / s) if s > 0 else float("nan"),
                area=float(box.mean()), pred=S["classes"][pred], prob=float(prob[pred]), label=S["classes"][label], grid=grid)


def prank(v, x):
    v = np.asarray(v)
    return 100.0 * ((v < x).sum() + 0.5 * (v == x).sum()) / len(v)


def select(L):
    csv = pd.read_csv(j3.CSV)
    o = csv[csv.direction == "original"]
    ok_a = True
    L.append("STEP 2 (a): original-direction class means of results/cam_z0_shares_exp.csv (patches with a defined share):")
    for (kind, c), want in STATED.items():
        got = o[(o.kind == kind) & (o.cls == c) & o.share_inside.notna()].share_inside.mean()
        ok = round(got, 4) == want
        ok_a &= ok
        L.append(f"  {c} {kind:13s} {got:.4f} (table {want:.4f}) {'PASS' if ok else 'FAIL, difference ' + format(got - want, '+.6f')}")
    L.append(f"  (a) {'PASS' if ok_a else 'FAIL'}")
    real_run, pf_run = j3.RUNS["original"]["real"], j3.RUNS["original"]["particle-free"]
    rman = pd.read_csv(j3.REAL_MAN["original"]).set_index("patch_file")
    pf_args = json.load(open(os.path.join(pf_run, "results.json")))["args"]
    pman = pd.read_csv(pf_args["manifest"]).set_index("patch_file")
    real_pdir = json.load(open(os.path.join(real_run, "results.json")))["args"]["patch_dir"]
    R = o[o.kind == "real"].copy()
    P = o[o.kind == "particle-free"].copy()
    P["key"] = [tuple(pman.loc[f, k] for k in j3.KEY) for f in P.patch_file]
    pf_by_key = {k: r for k, r in zip(P.key, P.itertuples(index=False))}
    chosen, report = {}, {}
    L += ["", "STEP 3: selection (fixed rule, applied to the CSV, the manifests and the predictions; no image looked at)",
          "Candidates: original-direction real test patches that (i) the real seed-0 checkpoint classifies correctly, (ii) are",
          "not clipped at the frame edge (manifest 'clipped' False and a full 64 x 64 patch, CSV real_patch_at_edge False),",
          "(iii) have the whole detection box inside the patch (box bounds of j3.box_mask() before clipping within 0 to 64),",
          "(iv) have a paired particle-free test patch from the same frame (same source_file, frame, particle_id) and",
          "(v) have a defined CAM share for both (not all non-positive after ReLU).",
          "Distance = sqrt(((s_r - median_r)/IQR_r)^2 + ((s_p - median_p)/IQR_p)^2) over the candidates; the smallest is chosen,",
          "the next two are backups. Percentile = 100 x (candidates below + half the ties) / candidates."]
    for c in CLASSES:
        Rc = R[R.cls == c]
        steps = [("real test patches of the class", len(Rc))]
        Rc = Rc[Rc.pred == c]
        steps.append(("(i) correctly classified", len(Rc)))
        Rc = Rc[(~Rc.real_patch_at_edge.astype(bool)) & (~rman.loc[Rc.patch_file, "clipped"].astype(bool).values)]
        steps.append(("(ii) not clipped at the frame edge", len(Rc)))
        rows = []
        for r in Rc.itertuples(index=False):
            m = rman.loc[r.patch_file]
            r0, r1, c0, c1 = raw_box(m)
            if not (r0 >= 0 and c0 >= 0 and r1 <= 64 and c1 <= 64):
                continue
            key = (m.source_file, m.frame, m.particle_id)
            rows.append((r, key, (r0, r1, c0, c1)))
        steps.append(("(iii) whole box inside the patch", len(rows)))
        rows = [(r, k, b) for r, k, b in rows if k in pf_by_key]
        steps.append(("(iv) paired particle-free test patch", len(rows)))
        rows = [(r, k, b) for r, k, b in rows if not np.isnan(r.share_inside) and not np.isnan(pf_by_key[k].share_inside)]
        steps.append(("(v) defined share for both", len(rows)))
        sr = np.array([r.share_inside for r, k, b in rows])
        sp = np.array([pf_by_key[k].share_inside for r, k, b in rows])
        mr, mp = np.median(sr), np.median(sp)
        ir, ip = np.percentile(sr, 75) - np.percentile(sr, 25), np.percentile(sp, 75) - np.percentile(sp, 25)
        dist = np.sqrt(((sr - mr) / ir) ** 2 + ((sp - mp) / ip) ** 2)
        order = sorted(range(len(rows)), key=lambda i: (dist[i], rows[i][0].patch_file))
        picks = []
        for rank, i in enumerate(order[:3]):
            r, k, b = rows[i]
            pf = pf_by_key[k]
            picks.append(dict(rank=rank, real=r.patch_file, pf=pf.patch_file, key=k, box=b, s_r=float(sr[i]), s_p=float(sp[i]),
                              area=float(r.area_share), dist=float(dist[i]), pr=prank(sr, sr[i]), pp=prank(sp, sp[i]),
                              pf_pred=pf.pred))
        chosen[c] = picks
        report[c] = dict(n=len(rows), mr=mr, mp=mp, ir=ir, ip=ip, frac=float((sr > sp).mean()),
                         cands=[(r.patch_file, pf_by_key[k].patch_file) for r, k, b in rows])  # kept for the revision 2 diagnostic
        L += ["", f"{c}: " + "; ".join(f"{s} {n}" for s, n in steps)]
        L.append(f"  candidates {len(rows)}; median s_r {mr:.4f} (IQR {ir:.4f}), median s_p {mp:.4f} (IQR {ip:.4f}); "
                 f"fraction of candidate pairs with s_r above s_p {report[c]['frac']:.3f}")
        for p in picks:
            L.append(f"  {'chosen ' if p['rank'] == 0 else 'backup ' + str(p['rank'])}: real {p['real']}, particle-free {p['pf']} "
                     f"({p['key'][0]}, frame {p['key'][1]}, particle {p['key'][2]}); s_r {p['s_r']:.4f} (percentile {p['pr']:.1f}), "
                     f"s_p {p['s_p']:.4f} (percentile {p['pp']:.1f}), box area share {p['area']:.4f}, distance {p['dist']:.3f}"
                     + ("" if p["s_r"] > p["s_p"] else "; s_r is NOT above s_p (kept, as fixed)"))
    return ok_a, chosen, report, rman, pman, real_pdir


def nice_ceil(v):
    step = 1.0 if v >= 5 else (0.5 if v >= 1 else 0.1)
    return math.ceil(v / step - 1e-9) * step


def draw(sets, grey, vmax, path, row_labels, dpi=600, pdf=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import cm, colors
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                         "image.composite_image": False, "pdf.fonttype": 42})
    cmap = plt.get_cmap("turbo")
    W = 16 / 2.54
    left, gap, s = 0.32, 0.05, None
    cb_w, cb_pad, cb_lab = 0.11, 0.10, 0.50
    s = (W - left - 3 * gap - cb_pad - cb_w - cb_lab) / 4
    top, under, rgap, bottom = 0.34, 0.27, 0.05, 0.03
    nrow = len(sets)
    H = top + nrow * (s + under) + (nrow - 1) * rgap + bottom
    fig = plt.figure(figsize=(W, H), facecolor="white")
    titles = ["Real patch", "Real + CAM", "Particle-free patch,\nsame frame", "Particle-free + CAM"]
    for i, row in enumerate(sets):
        gr = grey[i]  # revision 2: one grey range per row
        g = lambda x: np.repeat(np.clip((x - gr[0]) / (gr[1] - gr[0]), 0, 1)[..., None], 3, axis=2)
        y0 = H - top - (i + 1) * s - i * (under + rgap)
        for j, (kind, over) in enumerate((("real", False), ("real", True), ("pf", False), ("pf", True))):
            p = row[kind]
            ax = fig.add_axes([(left + j * (s + gap)) / W, y0 / H, s / W, s / H])
            img = g(p["raw"])
            if over:
                img = 0.5 * img + 0.5 * cmap(np.clip(p["u"] / vmax, 0, 1))[..., :3]
            ax.imshow(img, interpolation="none")
            r0, r1, c0, c1 = row["box"]
            ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, edgecolor="cyan", linewidth=1.2))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            txt = (f"in box {p['share']:.2f}\n(box {p['area']:.2f} of patch)" if over else
                   f"predicted {p['pred']} (p {p['prob']:.2f})")
            ax.text(0.5, -0.04, txt, transform=ax.transAxes, ha="center", va="top", fontsize=7, linespacing=1.15)
            if i == 0:
                ax.text(0.5, 1.03, titles[j], transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, linespacing=1.1)
            if j == 0:
                ax.text(-0.06, 0.5, row_labels[i], transform=ax.transAxes, ha="right", va="center", fontsize=8,
                        fontweight="bold", rotation=90 if len(row_labels[i]) > 3 else 0)
    yb = H - top - nrow * s - (nrow - 1) * (under + rgap)
    cax = fig.add_axes([(left + 4 * s + 3 * gap + cb_pad) / W, yb / H, cb_w / W, (H - top - yb) / H])
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, vmax), cmap=cmap), cax=cax)
    step = 0.5 if vmax <= 2.5 else (1.0 if vmax <= 6 else 2.0 if vmax <= 12 else 5.0)
    ticks = [t for t in np.arange(0, vmax + 1e-9, step) if vmax - t > 0.3 * step] + [vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}" for t in ticks[:-1]] + [f">= {vmax:g}"])
    cb.set_label("CAM mass per pixel (1 = uniform)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".png", dpi=dpi, facecolor="white")
    if pdf:
        fig.savefig(path + ".pdf", facecolor="white")
    plt.close(fig)


def main():
    redraw = "--redraw" in sys.argv
    t0 = time.time()
    L = ["CAM FIGURE: z0 expanded set, PET vs PS, original direction (test recordings PET2 and PS1)",
         "Checkpoints as in J3 (scripts/j3_cam_z0_exp.py RUNS): real patches results/real_tc_PETvsPS_z0_exp_s0; particle-free",
         "patches results/bg_tc_PETvsPS_z0_exp_s0, the particle-free-trained checkpoint, on its own test patches; CAM target =",
         "true class of the recording, block4, ReLU, sum to 1, 8 x 8 upsampled to 64 x 64, box of the real patch on both.", ""]
    ok_a, chosen, report, rman, pman, real_pdir = select(L)
    if not ok_a:
        print("\n".join(L))
        raise SystemExit("check (a) failed: stopped before the figure")
    Sr, Sp = setup(j3.RUNS["original"]["real"]), setup(j3.RUNS["original"]["particle-free"])
    sets, rowsCSV, ok_b = {}, [], True
    L += ["", "STEP 2 (b): shares recomputed from the checkpoints for the chosen pairs and backups, against the CSV (4 decimals):"]
    for c in CLASSES:
        for p in chosen[c]:
            mrow = rman.loc[p["real"]].copy()
            mrow["patch_file"] = p["real"]
            bm, edge = j3.box_mask(mrow, real_pdir)
            rr, pp = panel(Sr, p["real"], bm), panel(Sp, p["pf"], bm)
            br = rr["share"], pp["share"]
            ok = round(br[0], 4) == round(p["s_r"], 4) and round(br[1], 4) == round(p["s_p"], 4) and rr["pred"] == c \
                and pp["pred"] == p["pf_pred"] and rr["label"] == c and pp["label"] == c
            ok_b &= ok  # revision 2: every pair drawn (chosen and both backups) must pass
            L.append(f"  {c} {'chosen ' if p['rank'] == 0 else 'backup ' + str(p['rank'])}: real {br[0]:.4f} (CSV {p['s_r']:.4f}), "
                     f"particle-free {br[1]:.4f} (CSV {p['s_p']:.4f}); predictions real {rr['pred']} p {rr['prob']:.3f}, particle-free "
                     f"{pp['pred']} p {pp['prob']:.3f} (CSV {p['pf_pred']}) -> {'PASS' if ok else 'FAIL'}")
            rows_ = np.where(bm.any(1))[0]
            cols_ = np.where(bm.any(0))[0]
            box = (int(rows_[0]), int(rows_[-1]) + 1, int(cols_[0]), int(cols_[-1]) + 1)
            assert box == p["box"], (box, p["box"])
            sets.setdefault(p["rank"], []).append(dict(real=rr, pf=pp, box=box, cls=c))
            for kind, f, q, man_, S_ in (("real", p["real"], rr, rman, Sr), ("particle-free", p["pf"], pp, pman, Sp)):
                ck = S_["run"]
                m = man_.loc[f]
                for ptype in ("patch", "patch + CAM"):
                    rowsCSV.append(dict(sheet="main" if p["rank"] == 0 else f"backup {p['rank']}", row=c,
                                        column=("Real" if kind == "real" else "Particle-free") + (" + CAM" if ptype != "patch" else " patch"),
                                        file=f, kind=kind, checkpoint=ck, **{"class": c}, recording=m.source_file, frame=int(m.frame),
                                        particle_id=int(m.particle_id), x=int(m.x), y=int(m.y), predicted_class=q["pred"],
                                        probability=round(q["prob"], 4),
                                        box=f"rows {box[0]} to {box[1] - 1}, columns {box[2]} to {box[3] - 1} (pixels of the 64 x 64 canvas)",
                                        cam_share=round(q["share"], 4), area_share=round(q["area"], 4),
                                        input_mean=S_["mean"], input_std=S_["std"], raw_mean=round(float(q["raw"].mean()), 6),
                                        raw_std=round(float(q["raw"].std()), 6), _q=q, _over=ptype != "patch"))
    L.append(f"  (b) {'PASS' if ok_b else 'FAIL'} (every pair drawn: the chosen pairs and both backups)")
    if not ok_b:
        print("\n".join(L))
        raise SystemExit("check (b) failed: stopped before the figure")
    main_set = [s for s in sets[0]]
    # Revision 1 (2026-10-21): raw z0 patches for display, turbo, colour maximum = 99th percentile of the pooled overlays.
    # Revision 2 (2026-10-21): one grey range per row = 0.1st to 99.9th percentile of the pooled raw pixels of the row's
    # real and particle-free patch; alternative figures with the PS backup pairs in row 2; same colour maximum.
    for r_ in sets.values():
        for s in r_:
            pix = np.r_[s["real"]["raw"].ravel(), s["pf"]["raw"].ravel()]
            s["grey"] = (float(np.percentile(pix, 0.1)), float(np.percentile(pix, 99.9)))
            for k in ("real", "pf"):
                x = s[k]["raw"]
                s[k]["clip"] = (float((x < s["grey"][0]).mean()), float((x > s["grey"][1]).mean()))
    upool = np.concatenate([s[k]["u"].ravel() for s in main_set for k in ("real", "pf")])
    u99, umax = float(np.percentile(upool, 99)), float(upool.max())
    vmax = nice_ceil(u99)
    assert vmax == 14, vmax  # the colour maximum of revision 1, kept for the alternatives
    bk = [s for r in (1, 2) for s in sets[r]]
    alts = {1: [sets[0][0], sets[1][1]], 2: [sets[0][0], sets[2][1]]}
    dev = max(s[k]["dev"] for s in main_set + bk for k in ("real", "pf"))
    lab2 = (("real", "real"), ("pf", "particle-free"))
    sat = lambda rows_: ", ".join(f"{s['cls']} {lab} {int((s[k]['u'] > vmax).sum())}" for s in rows_ for k, lab in lab2)
    clip = lambda s, k: f"{100 * s[k]['clip'][0]:.2f}% low, {100 * s[k]['clip'][1]:.2f}% high"
    L += ["", "FIGURE (revision 2: grey range per row; revision 1: raw z0 patches, turbo, colour maximum from the 99th percentile)",
          f"Patches shown: the stored float32 z0 ratio patches (before standardisation); the loader tensor equals (raw - mean) / std "
          f"to {dev:.1e} at most. Grey display range per row: 0.1st to 99.9th percentile of the pooled raw pixels of the row's real "
          "and particle-free patch (display only).",
          f"Standardisation each checkpoint applies (training-split statistics): real checkpoint mean {Sr['mean']:.5f}, std "
          f"{Sr['std']:.5f}; particle-free checkpoint mean {Sp['mean']:.5f}, std {Sp['std']:.5f}. CAMs are computed on these inputs.",
          "Main figure rows (grey range; raw mean and std; pixels clipped at either end):"]
    for s in main_set:
        L.append(f"  {s['cls']} row: grey {s['grey'][0]:.4f} to {s['grey'][1]:.4f}")
        for k, lab in lab2:
            L.append(f"    {lab:13s} mean {s[k]['raw'].mean():.5f}, std {s[k]['raw'].std():.5f}, clipped {clip(s, k)}")
    L.append("Backup pairs (rows of the backup sheet; PS backup 1 and 2 are also row 2 of fig_cam_pet_ps_alt1 and _alt2):")
    for s, nm in zip(bk, ["PET backup 1", "PS backup 1", "PET backup 2", "PS backup 2"]):
        L.append(f"  {nm}: grey {s['grey'][0]:.4f} to {s['grey'][1]:.4f}; real clipped {clip(s, 'real')}; particle-free clipped "
                 f"{clip(s, 'pf')}")
    L.append(f"CAM colour scale: turbo, linear, 0 to {vmax:g} times uniform attention (99th percentile of the pooled overlay pixels "
             f"of the main figure {u99:.3f}, rounded up; largest value {umax:.3f}); values above {vmax:g} saturate; overlays blended at "
             "alpha 0.5; the same maximum on every sheet.")
    L.append("Saturated pixels per overlay (of 4096): main " + sat(main_set) + "; alt1 " + sat(alts[1]) + "; alt2 " + sat(alts[2])
             + "; backup sheet " + sat(bk) + ".")
    rawf = lambda pdir, f: np.load(os.path.join(pdir, f))[..., 0]
    L += ["", "DIAGNOSTIC (revision 2, information only; the selection is not changed): raw-patch std of the real patch and of its",
          "particle-free pair, and the range of the real patch's column-mean profile (maximum minus minimum of the 64 column means),",
          "each with its percentile among the selection candidates of the class (particle-free among the candidates' pairs)."]
    for c in CLASSES:
        cands = report[c]["cands"]
        rs = np.array([rawf(Sr["pdir"], a).std() for a, b in cands])
        ps = np.array([rawf(Sp["pdir"], b).std() for a, b in cands])
        cr = np.array([np.ptp(rawf(Sr["pdir"], a).mean(axis=0)) for a, b in cands])
        idx = {a: i for i, (a, b) in enumerate(cands)}
        L.append(f"  {c} candidates ({len(cands)}): median real std {np.median(rs):.5f}, median particle-free std {np.median(ps):.5f}, "
                 f"median real column-mean range {np.median(cr):.5f}")
        for p in chosen[c]:
            i = idx[p["real"]]
            assert cands[i][1] == p["pf"]
            L.append(f"  {c} {'chosen ' if p['rank'] == 0 else 'backup ' + str(p['rank'])}: real std {rs[i]:.5f} (percentile "
                     f"{prank(rs, rs[i]):.1f}), particle-free std {ps[i]:.5f} (percentile {prank(ps, ps[i]):.1f}), real column-mean "
                     f"range {cr[i]:.5f} (percentile {prank(cr, cr[i]):.1f})")
    L.append(f"Time {time.time() - t0:.0f} s. Written {time.strftime('%Y-%m-%d %H:%M')}.")
    text = "\n".join(L)
    assert "\u2014" not in text and ("signif" + "icant") not in text.lower() and ("valid" + "ated") not in text.lower()
    rank_of = {"main": 0, "backup 1": 1, "backup 2": 2}
    for r in rowsCSV:
        q, over = r.pop("_q"), r.pop("_over")
        r["saturated_pixels"] = int((q["u"] > vmax).sum()) if over else ""
        r["cam_colour_max"] = vmax if over else ""
        s = sets[rank_of[r["sheet"]]][CLASSES.index(r["row"])]
        k = "real" if r["kind"] == "real" else "pf"
        r["grey_min"], r["grey_max"] = round(s["grey"][0], 6), round(s["grey"][1], 6)
        r["clipped_low"], r["clipped_high"] = round(s[k]["clip"][0], 6), round(s[k]["clip"][1], 6)
    dfp = pd.DataFrame(rowsCSV)
    if os.path.exists(PANELS):  # the selection must not change
        old = pd.read_csv(PANELS)
        cols = ["sheet", "row", "column", "file", "cam_share", "area_share", "predicted_class"]
        assert old[cols].astype(str).values.tolist() == pd.read_csv(io.StringIO(dfp.to_csv(index=False)))[cols].astype(str).values.tolist(), \
            "selection or shares changed"
    open(SEL, "w", encoding="utf-8").write(text + "\n")
    dfp.to_csv(PANELS, index=False)
    draw(main_set, [s["grey"] for s in main_set], vmax, FIG, list(CLASSES))
    for a, rows_ in alts.items():
        draw(rows_, [s["grey"] for s in rows_], vmax, f"{FIG}_alt{a}", list(CLASSES))
    draw(bk, [s["grey"] for s in bk], vmax, FIG + "_backup", ["PET, backup 1", "PS, backup 1", "PET, backup 2", "PS, backup 2"], pdf=False)
    print(text)


if __name__ == "__main__":
    main()
