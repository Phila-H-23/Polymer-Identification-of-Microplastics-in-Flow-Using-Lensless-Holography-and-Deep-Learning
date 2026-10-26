"""Item 41 J3: class activation maps (CAM) of the seed-0 z0 checkpoints, original and reverse direction, real and
particle-free classifiers (read-only; no training). CAM code of the earlier CAM review (scripts/cam_control_pet.py,
cam_for_class: stem, block1 to block4, einsum of the last linear layer's weights for the class with the block4
features, bilinear upsampling to 64 x 64, align_corners False) with the target = the TRUE class; its min-max scaling is
replaced, as asked, by ReLU (after the upsampling) and normalisation to sum 1. Test inputs are rebuilt exactly as in
scripts/eval_track_ci_exp.py evaluate() (training-split channel statistics), and the predictions from the same forward
pass are checked against the run's results.json macro-F1 (4 decimals).
Box: the detection's bounding box (bbox_w_px x bbox_h_px of the dilated detected region, centred on the detection
centre), in the patch canvas (frame-edge patches sit top-left in the 64 x 64 canvas, as in the training code), pixel
centres inside the box; particle-free patches use the box of their paired real patch (same source_file, frame,
particle_id). Share inside = CAM mass inside the box; area share = box pixels / 4096.
Track intervals: 1000 resamples of whole test tracks (numpy default_rng(0)); a particle-free patch carries the track
of its paired real patch.
Readings fixed in the task: C1 real minus particle-free share inside >= 0.10 for both classes in both directions;
C2 the real share exceeds the area share but that difference is below 0.10; C3 otherwise. 'That difference' is read
as worded (real share minus area share), required in every class and direction; the reading with 'that difference'
= real minus particle-free is printed beside it.
Writes results/report_cam_z0_exp.txt and results/cam_z0_shares_exp.csv (new files only)."""
import os
import sys
import json
import time

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
import eval_track_ci_exp as ev  # noqa: E402
import item41_status as st  # noqa: E402

RUNS = {"original": {"real": "results/real_tc_PETvsPS_z0_exp_s0", "particle-free": "results/bg_tc_PETvsPS_z0_exp_s0"},
        "reverse": {"real": "results/real_tc_PETvsPS_z0_exp_rev_s0", "particle-free": "results/bg_tc_PETvsPS_z0_exp_rev_s0"}}
REAL_MAN = {"original": "pipeline_output/manifest_recon_s0_z0_PETPS_exp.csv",
            "reverse": "pipeline_output/manifest_recon_s0_z0_PETPS_rev_exp.csv"}
OUT, CSV = "results/report_cam_z0_exp.txt", "results/cam_z0_shares_exp.csv"
KEY = ["source_file", "frame", "particle_id"]
H_DEM, W_DEM, B = 1024, 1224, 1000


def cam_for_true_class(model, x, cls_idx):
    """scripts/cam_control_pet.py cam_for_class up to the upsampling (unchanged), returning the signed map."""
    with torch.no_grad():
        f = model.stem(x)
        for blk in (model.block1, model.block2, model.block3, model.block4):
            f = blk(f)
        cam = torch.einsum("k,khw->hw", model.head[-1].weight[cls_idx], f[0])
        grid = tuple(cam.shape)
        cam = F.interpolate(cam[None, None], size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
        logits = model.head[-1](f.mean(dim=(2, 3)))
        return cam.numpy(), int(logits.argmax(1).item()), grid


def box_mask(row, patch_dir):
    """Detection bounding box in the 64 x 64 canvas of the real patch (pixel centres inside the box)."""
    x, y, w, h = int(row.x), int(row.y), float(row.bbox_w_px), float(row.bbox_h_px)
    ox, oy = max(0, x - 32), max(0, y - 32)
    ph, pw = np.load(os.path.join(patch_dir, row.patch_file), mmap_mode="r").shape[:2]
    c0, c1 = int(np.floor(x - w / 2 - ox + 0.5)), int(np.floor(x + w / 2 - ox + 0.5))
    r0, r1 = int(np.floor(y - h / 2 - oy + 0.5)), int(np.floor(y + h / 2 - oy + 0.5))
    m = np.zeros((64, 64), bool)
    m[max(0, r0):min(ph, 64, r1), max(0, c0):min(pw, 64, c1)] = True
    return m, (ph, pw) != (64, 64)


def run_cams(run_dir, boxes):
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
    rows, preds, grid = [], [], None
    for i in range(len(ds)):
        x, label = ds[i]
        cam, pred, grid = cam_for_true_class(model, x[None], label)
        preds.append(classes[pred])
        r = test_df.iloc[i]
        bm, edge = boxes[tuple(r[k] for k in KEY)]
        c = np.maximum(cam, 0.0)
        s = c.sum()
        rows.append(dict(patch_file=r.patch_file, track_id=r.track_id, cls=r["class"], pred=classes[pred],
                         share_inside=float(c[bm].sum() / s) if s > 0 else np.nan, area_share=float(bm.mean()),
                         cam_zero=bool(s <= 0), real_patch_at_edge=edge))
    f1 = f1_score(test_df["class"], preds, labels=classes, average="macro", zero_division=0)
    return pd.DataFrame(rows), float(f1), float(saved["test_report"]["macro avg"]["f1-score"]), grid, classes


def main():
    t0 = time.time()
    if os.path.exists(OUT):  # J3 already ran in a gap; the runner's call after J2 then does nothing
        st.update("j3", status="done", in_run=False, note="ran earlier in a gap; not rerun")
        return
    st.update("j3", status="running", started=time.strftime("%Y-%m-%d %H:%M"), pid=os.getpid(), in_run=True)
    for p in (OUT, CSV):
        if os.path.exists(p):
            raise FileExistsError(p)
    L = ["J3: CAM SHARE INSIDE THE DETECTED BOX, seed-0 z0 checkpoints (z0, PET vs PS, expanded set, track-based validation)"]
    L += [x.strip() for x in __doc__.split("\n")[1:16]]
    all_rows, summary, checks = [], {}, []
    for direction, kinds in RUNS.items():
        rm = pd.read_csv(REAL_MAN[direction])
        pdir = json.load(open(os.path.join(kinds["real"], "results.json")))["args"]["patch_dir"]
        boxes = {}
        for r in rm.itertuples(index=False):
            boxes[(r.source_file, r.frame, r.particle_id)] = box_mask(r, pdir)
        res = {}
        for kind, rd in kinds.items():
            df, f1, f1s, grid, classes = run_cams(rd, boxes)
            checks.append(f"{os.path.basename(rd)}: macro-F1 from this forward pass {f1:.4f}, results.json {f1s:.4f} -> "
                          f"{'same' if round(f1, 4) == round(f1s, 4) else 'DIFFERENT'}; CAM grid {grid[0]} x {grid[1]} (block4) "
                          f"upsampled to 64 x 64; patches {len(df)}, CAM all <= 0 after ReLU: {int(df.cam_zero.sum())}, real box "
                          f"from a frame-edge patch: {int(df.real_patch_at_edge.sum())}")
            res[kind] = df.assign(kind=kind, direction=direction)
            all_rows.append(res[kind])
        tracks = sorted(set(res["real"].track_id) | set(res["particle-free"].track_id))
        ti = {t: i for i, t in enumerate(tracks)}
        rng = np.random.default_rng(0)
        W = np.stack([np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks)) for _ in range(B)])
        summary[direction] = {}
        for c in classes:
            boot, pt = {}, {}
            for kind in ("real", "particle-free"):
                d = res[kind][(res[kind].cls == c) & res[kind].share_inside.notna()]
                S, N, A = np.zeros(len(tracks)), np.zeros(len(tracks)), np.zeros(len(tracks))
                for t, v, a in zip(d.track_id, d.share_inside, d.area_share):
                    S[ti[t]] += v
                    N[ti[t]] += 1
                    A[ti[t]] += a
                pt[kind] = (S.sum() / N.sum(), A.sum() / N.sum(), int(N.sum()))
                boot[kind] = (W @ S) / np.maximum(W @ N, 1e-12)
                boot[kind + "_area"] = (W @ A) / np.maximum(W @ N, 1e-12)
            diff = boot["real"] - boot["particle-free"]
            ex = boot["real"] - boot["real_area"]
            q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            summary[direction][c] = dict(
                real=pt["real"][0], real_ci=q(boot["real"]), pf=pt["particle-free"][0], pf_ci=q(boot["particle-free"]),
                area_real=pt["real"][1], area_pf=pt["particle-free"][1], n_real=pt["real"][2], n_pf=pt["particle-free"][2],
                diff=pt["real"][0] - pt["particle-free"][0], diff_ci=q(diff), excess=pt["real"][0] - pt["real"][1],
                excess_ci=q(ex))
    pd.concat(all_rows).to_csv(CSV, index=False)
    L += ["", "Checks:"] + ["  " + x for x in checks]
    L += ["", "Mean share of CAM mass inside the box (point value over all test patches of the class; [track 95% interval]):",
          f"{'direction':9s} {'class':5s} {'n real':>6s} {'real share':>23s} {'n pf':>5s} {'particle-free share':>23s} "
          f"{'area share':>10s} {'real - pf':>25s} {'real - area':>25s}"]
    for d, cs in summary.items():
        for c, v in cs.items():
            L.append(f"{d:9s} {c:5s} {v['n_real']:6d} {v['real']:.4f} [{v['real_ci'][0]:.4f}, {v['real_ci'][1]:.4f}] {v['n_pf']:5d} "
                     f"{v['pf']:.4f} [{v['pf_ci'][0]:.4f}, {v['pf_ci'][1]:.4f}] {v['area_real']:10.4f} {v['diff']:+.4f} "
                     f"[{v['diff_ci'][0]:+.4f}, {v['diff_ci'][1]:+.4f}] {v['excess']:+.4f} [{v['excess_ci'][0]:+.4f}, "
                     f"{v['excess_ci'][1]:+.4f}]")
    L.append("(area share is the same for real and particle-free patches, since a particle-free patch uses its pair's box)")
    cells = [v for cs in summary.values() for v in cs.values()]
    c1 = all(v["diff"] >= 0.10 for v in cells)
    c2_worded = (not c1) and all(0 < v["excess"] < 0.10 for v in cells)
    c2_alt = (not c1) and all(v["excess"] > 0 and v["diff"] < 0.10 for v in cells)
    reading = "C1" if c1 else ("C2" if c2_worded else "C3")
    alt = "C1" if c1 else ("C2" if c2_alt else "C3")
    L += ["", "Readings (fixed in the task; point values, every class in both directions):",
          f"C1 (real - particle-free >= 0.10 in all four class-direction cells): {c1}",
          f"C2 as worded (real share above the area share by more than 0 and less than 0.10 in all four cells): {c2_worded}",
          f"Reading: {reading}",
          f"(With 'that difference' read as real minus particle-free instead, i.e. real share above the area share and real -"
          f" particle-free below 0.10 in all four cells: {alt}.)",
          "", f"Per-patch values: {CSV}. Time {time.time() - t0:.0f} s."]
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    open(OUT, "w", encoding="utf-8").write(text + "\n")
    st.update("j3", status="done", finished=time.strftime("%Y-%m-%d %H:%M"), in_run=False,
              reading=f"{reading} (as worded); {alt} if 'that difference' means real minus particle-free",
              files=[OUT, CSV, "scripts/j3_cam_z0_exp.py"])
    print(text)


if __name__ == "__main__":
    main()
