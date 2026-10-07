"""Size-only baselines (Part A) and PET vs PS on four more input arms (Part B) -> results/report_size_and_arms.txt.
Part A is read-only (logistic regression on S1 size features, no CNN). Part B reads finished run folders.
z0 and intensity_bgcorr_vnorm numbers are copied from results/report_twoclass.txt (not rerun, not recomputed).
Refreshed after Part A and after every Part B arm."""
import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from build_report_twoclass import chance, load, side, reading

PO = "pipeline_output"
S1 = "results/recon_fidelity/s1_detection_stats.csv"
OTHERS = ["PS", "PE", "PMMA", "PVC", "PP", "PVA"]
FEATS = ["log region area", "bbox width", "bbox height"]
ARMS = ["recon_s0", "recon_biref", "pol4_bgcorr_vnorm", "i45_bgcorr_vnorm"]
SEEDS = {False: 5, True: 3}
UM2_PER_PX = 6.9 ** 2

HEADER = """SIZE-ONLY BASELINES AND PET vs PS ON FOUR MORE INPUT ARMS
Part A (read-only, no CNN). Features per patch from the S1 detection stats ({s1}):
  log(region area, px, before dilation), bounding-box width and height (px). Class-weighted logistic regression
  (sklearn LogisticRegression, L2, C=1.0, lbfgs), features standardised with training-split mean/std, class weights
  n_train / (2 x n_c) from the training rows used. Fitted on the training split only (no hyperparameter is selected,
  so validation patches are not used). Split: the stored two-class video-holdout assignment (reverse fold: the
  2026-10-04 assignment). Test macro-F1 with a bootstrap 95% CI (1000 resamples of test patches with replacement,
  numpy default_rng(0), percentile interval). Run on all patches and on unclipped patches only (clipped patches
  removed from train, validation and test).
  Readings (pre-registered; S = size-only macro-F1, R = z0 real macro-F1 for the same pair and split, from
  results/report_twoclass.txt): Z1 S >= R - 0.05; Z2 S <= R - 0.15; Z3 between. Primary: PET vs PS, original
  direction, all patches. For unclipped rows R is still the z0 value on all test patches.
Part B: PET vs PS, real and particle-free, regime and split of the two-class test (results/report_twoclass.txt):
  common 849 patches; 150 epochs, patience 30 on validation macro-F1 (best-validation checkpoint); batch 32;
  AdamW lr 1e-3, weight decay 1e-4, cosine annealing; CompactHoloCNN width 32, dropout 0.3, 2 outputs
  (--classes PET,PS); rotation 0/180, translation +/-4 px, gain +/-5%, noise 0.01; CPU. Original direction seeds 0-4
  with pipeline_output/class_weights_twoclass_PETvsPS.json; reverse fold seeds 0-2 with
  pipeline_output/class_weights_twoclass_PETvsPS_rev.json; the same file for the real and the particle-free classifier.
  T readings (margin 0.10 over chance; floor = larger seed std of macro-F1): T1 gap > floor AND real > chance + 0.10;
  T2 gap <= floor AND particle-free > chance + 0.10; T3 real <= chance + 0.10 AND particle-free <= chance + 0.10; T4 none.
  B readings against z0 (original 0.874, reverse 0.759); margin per direction = larger real seed std of arm and z0:
  B1 arm real above z0 by more than the margin in both directions AND arm gap > arm floor in both directions;
  B2 arm real below z0 by more than the margin in both directions; B3 otherwise.
""".format(s1=S1)


# ------------------------------------------------------------------ Part A
def size_table():
    s = pd.read_csv(S1)
    s = s[s.kind == "real"].copy()
    s["log_area"] = np.log(s.region_px_predilation)
    out = {}
    for rev in (False, True):
        m = pd.read_csv(f"{PO}/manifest_bgcorrected_common849{'_rev' if rev else ''}.csv")
        split = dict(zip(m.patch_file, m.split_video_holdout))
        d = s.copy()
        if not rev:
            assert (d.patch_file.map(split) == d.split_video_holdout).all()
        d["split_used"] = d.patch_file.map(split)
        assert d.split_used.notna().all() and len(d) == 849
        out[rev] = d
    return out


def size_only(d, cls, unclipped):
    d = d[d["class"].isin(cls)]
    if unclipped:
        d = d[~d.clipped.astype(bool)]
    tr, te = d[d.split_used == "train"], d[d.split_used == "test"]
    X = lambda x: x[["log_area", "bbox_w_px", "bbox_h_px"]].values.astype(float)
    mu, sd = X(tr).mean(0), X(tr).std(0)
    n = tr["class"].value_counts()
    w = {c: len(tr) / (2 * n[c]) for c in cls}
    lr = LogisticRegression(class_weight=w, max_iter=1000).fit((X(tr) - mu) / sd, tr["class"].values)
    yt, yp = te["class"].values, lr.predict((X(te) - mu) / sd)
    f1 = f1_score(yt, yp, labels=cls, average="macro", zero_division=0)
    rec = recall_score(yt, yp, labels=cls, average=None, zero_division=0)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        i = rng.integers(0, len(yt), len(yt))
        boot.append(f1_score(yt[i], yp[i], labels=cls, average="macro", zero_division=0))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(f1=f1, lo=lo, hi=hi, rec=dict(zip(cls, rec)), bacc=float(rec.mean()), w=w,
                n=(len(tr), len(te)), ntest={c: int((yt == c).sum()) for c in cls},
                coef=dict(zip(FEATS, lr.coef_[0])), pos=lr.classes_[1])


def z_reading(S, R):
    if S >= R - 0.05:
        return "Z1"
    if S <= R - 0.15:
        return "Z2"
    return "Z3"


def part_a(z0):
    tabs = size_table()
    L = ["", "=" * 110, "PART A1  SIZE-ONLY BASELINE (logistic regression on log area, bbox width, bbox height)", "=" * 110,
         f"{'pair':10s} {'direction':9s} {'patches':9s} {'n tr/te':>8s}  {'size-only macro-F1 [95% CI]':>28s}  "
         f"{'PET rec':>7s} {'other rec':>9s} {'bacc':>6s}  {'z0 R':>6s} {'S - R':>7s}  reading"]
    weights_ok = []
    for o in OTHERS + ["PS_rev"]:
        rev = o == "PS_rev"
        other = "PS" if rev else o
        cls, pair = ["PET", other], f"PETvs{other}"
        R = z0[(pair, rev)]["real"][0]
        for unc in (False, True):
            r = size_only(tabs[rev], cls, unc)
            if not unc:
                f = json.load(open(f"{PO}/class_weights_twoclass_{pair}{'_rev' if rev else ''}.json"))
                weights_ok.append(all(abs(f[c] - r["w"][c]) < 1e-4 for c in cls))
            L.append(f"{pair:10s} {'reverse' if rev else 'original':9s} {'unclipped' if unc else 'all':9s} "
                     f"{r['n'][0]:>3d}/{r['n'][1]:<4d}  {r['f1']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]       "
                     f"{r['rec']['PET']:7.3f} {r['rec'][other]:9.3f} {r['bacc']:6.3f}  {R:6.4f} {r['f1'] - R:+7.4f}  "
                     f"{z_reading(r['f1'], R)}" + ("  <- primary" if pair == "PETvsPS" and not rev and not unc else ""))
    L += [f"class weights for 'all' rows equal pipeline_output/class_weights_twoclass_<pair>[_rev].json: {all(weights_ok)}",
          "PET rec / other rec = test recall of PET / of the other class; bacc = balanced accuracy (mean of the two recalls).",
          "Z readings (pre-registered): Z1 S >= R - 0.05; Z2 S <= R - 0.15; Z3 between. Unclipped rows compare with R on all patches."]

    # A2: size distributions
    L += ["", "=" * 110, "PART A2  SIZE DISTRIBUTION: region area before dilation, px (1 px = 6.9 x 6.9 um = 47.6 um^2)", "=" * 110,
          f"{'class':6s} {'split':14s} {'n':>4s} {'median px':>9s} {'P10 px':>7s} {'P90 px':>7s}   {'median um^2':>11s}"]
    d0 = tabs[False]
    rows = [(c, sp, d0[(d0["class"] == c) & (d0.split_used == sp)]) for c in sorted(d0["class"].unique())
            for sp in ("train", "val", "test")]
    rows += [(c, f"{sp} (reverse)", tabs[True][(tabs[True]["class"] == c) & (tabs[True].split_used == sp)])
             for c in ("PET", "PS") for sp in ("train", "val", "test")]
    for c, sp, x in rows:
        a = x.region_px_predilation.values
        L.append(f"{c:6s} {sp:14s} {len(a):>4d} {np.median(a):9.0f} {np.percentile(a, 10):7.0f} {np.percentile(a, 90):7.0f}   "
                 f"{np.median(a) * UM2_PER_PX:11.0f}")
    L += ["", "Overlap of test sizes with the other class's training size range (P10-P90 of region area, all patches):",
          f"{'pair':10s} {'direction':9s}  {'PET train P10-P90':>17s}  {'other test inside':>17s}  "
          f"{'other train P10-P90':>19s}  {'PET test inside':>15s}"]
    for o in OTHERS + ["PS_rev"]:
        rev = o == "PS_rev"
        other = "PS" if rev else o
        d = tabs[rev]
        a = lambda c, sp: d[(d["class"] == c) & (d.split_used == sp)].region_px_predilation.values
        p_lo, p_hi = np.percentile(a("PET", "train"), [10, 90])
        o_lo, o_hi = np.percentile(a(other, "train"), [10, 90])
        o_in = np.mean((a(other, "test") >= p_lo) & (a(other, "test") <= p_hi))
        p_in = np.mean((a("PET", "test") >= o_lo) & (a("PET", "test") <= o_hi))
        L.append(f"PETvs{other:5s} {'reverse' if rev else 'original':9s}  {p_lo:7.0f} - {p_hi:<7.0f}  {o_in:17.1%}  "
                 f"{o_lo:9.0f} - {o_hi:<7.0f}  {p_in:15.1%}")
    return L


A3 = """
==============================================================================================================
PART A3  PREPROCESSING: intensity_bgcorr_vnorm versus z0 (from the code; PET vs PS uses both exactly as below)
==============================================================================================================
Sources: scripts/build_bgcorrected_dataset.py (intensity_bgcorr_vnorm patches), scripts/build_reconstructed_dataset.py
  --z 0 --suffix _z0 --s0-only with scripts/reconstruction.py and scripts/reconstruction_validate.py (z0 patches),
  train_compact_cnn.py (PatchDataset._load, apply_vnorm, __getitem__, _augment, compute_channel_stats).

step                    | intensity_bgcorr_vnorm                         | z0                                              | could it change the central particle region?
------------------------+------------------------------------------------+-------------------------------------------------+---------------------------------------------
1 demosaic              | strided 2x2 pixel selection (I0,I45,I90,I135), | same strided selection (reconstruction.demosaic)| no - identical selection; it commutes with
                        | AFTER background correction of the raw mosaic  | BEFORE division (background median is taken on  | both background estimates (per-pixel), so
                        |                                                | the raw mosaic, then demosaiced)                | order does not matter
2 background estimate   | per-pixel MEAN of up to 90 preceding raw frames| per-pixel MEDIAN of up to 90 preceding raw      | yes - a particle (or other particles) that
                        | (float64)                                      | frames (uint8 stack)                            | sat at the same pixels in earlier frames
                        |                                                |                                                 | enters the mean in proportion to its dwell
                        |                                                |                                                 | time; the median ignores it below half the
                        |                                                |                                                 | window
3 background application| additive: raw - bg_mean + mean(bg_mean), on    | divisive: I_c / (I_c_bg + 1e-6) per channel,    | yes - subtraction keeps the particle's
                        | the raw mosaic                                 | after demosaic (clip at 0 has no effect)        | contrast scaled by the local illumination;
                        |                                                |                                                 | division expresses it relative to the local
                        |                                                |                                                 | background (removes multiplicative
                        |                                                |                                                 | illumination and dust-shadow modulation)
4 clipping and storage  | clipped to 0-255 and cast to uint8 (truncation)| float32, no upper clip, no quantisation         | yes, where the particle's fringes or core
                        | before saving; /255 at load                    | (values about 1, background ratio)              | reach 0 or 255 after correction; quantised
                        |                                                |                                                 | to integer steps everywhere
5 per-video             | apply_vnorm: (x - mean_v) / std_v per video,   | none                                            | values yes (one offset and scale per video,
  normalisation         | stats pooled over the 4 channels of that       |                                                 | applied to every pixel); spatial pattern
                        | video's background-only patches                |                                                 | within a patch no
                        | (video_background_stats.json)                  |                                                 |
6 square root           | none                                           | amplitude sqrt(I/I_bg), later squared (|.|^2):  | no - cancels (to float rounding)
                        |                                                | cancels                                         |
7 phase / propagation   | none                                           | zero phase, angular-spectrum propagation with   | no - at z=0 H = 1 on every frequency of the
                        |                                                | z=0: H = exp(i 2 pi z kz) = 1 (6.9 um pitch:    | grid, so z0 = I/I_bg exactly (to FFT
                        |                                                | all frequencies propagating, none zeroed)       | rounding)
8 channel combination   | mean of the 4 channels in __getitem__, AFTER   | mean of the 4 propagated channels at build      | no for the signal (same mean); the training
                        | augmentation (per-channel noise is averaged:   | time, BEFORE augmentation (noise added to the   | noise differs, see 12
                        | effective noise std 0.005)                     | single channel: std 0.01)                       |
9 input channels        | 1                                              | 1                                               | no
10 patch size /         | 64x64, crop half=32 at the detection centre,   | same crop formula (HALF=32), no resampling      | no
   resampling           | no resampling                                  |                                                 |
11 zero padding of      | patch vnormed first, then placed on a zero     | zero canvas = ratio 0, far below the background | no - padding lies outside the frame; it
   clipped patches      | canvas: 0 = that video's background mean       | level of about 1 (a hard dark border)           | changes the border of the 118 clipped
   (118 of 849)         | (no visible border)                            |                                                 | patches, not the particle
12 training-time        | per-channel train-split mean/std (no           | same procedure; PET vs PS train stats mean      | no for the pattern (one global affine map);
   normalisation and    | augmentation); PET vs PS train stats mean      | 0.9711, std 0.1688 (std includes the zero       | but the augmentation acts at different
   augmentation scale   | 0.0038, std 0.6633. Augmentation is applied    | padding). Gain +/-5% on values about 1 shifts   | relative strength on the particle during
                        | BEFORE this normalisation: gain +/-5% on values| the whole patch by up to about 0.05 (0.29 train | training (noise 0.01/0.1688 = 0.059 of the
                        | centred near 0 changes contrast only; noise    | std); noise 0.01 = 0.059 train std              | train std vs 0.0075)
                        | 0.005/0.6633 = 0.0075 train std                |                                                 |
13 frames used          | common 849 patches, frame 0 excluded           | same 849 patches                                | no

Steps that differ in a way that could change the central particle region: 2 (mean vs median background), 3 (subtract vs
divide), 4 (0-255 clipping and uint8 quantisation), 5 (per-video normalisation, values only) and 12 (augmentation
strength relative to the input scale). More than one step differs.

PART C: not built and not run. Its condition (exactly one step differs in a way that matters, switchable with an
existing option or a small additive flag) is not met: five steps differ, and steps 2-4 sit in the dataset
build scripts, so a swap would need a new patch build, not a training flag.
"""


# ------------------------------------------------------------------ Part B
def copied_twoclass():
    """Summary rows of results/report_twoclass.txt for PET vs PS, z0 and intensity_bgcorr_vnorm, both directions."""
    pat = re.compile(r"^(PETvs\w+)\s+(z0|intensity_bgcorr_vnorm)\s+(\[Part 2, reverse fold\])?\s*"
                     r"([\d.]+)\+/-([\d.]+) ([\d.]+)\+/-([\d.]+) ([+-][\d.]+) ([\d.]+) ([\d.]+)\s+([\d.]+) ([\d.]+)\s+(T\d)\s*$")
    out = {}
    for line in open("results/report_twoclass.txt"):
        m = pat.match(line.rstrip("\n"))
        if not m:
            continue
        g = m.groups()
        v = [float(x) for x in g[3:12]]
        out[(g[0], g[1], bool(g[2]))] = dict(real=(v[0], v[1]), pf=(v[2], v[3]), gap=v[4], floor=v[5], chance=v[6],
                                             bacc_r=v[7], bacc_pf=v[8], T=g[12])
    return out


def b_reading(arm, z0):
    if any(arm.get(rev) is None for rev in (False, True)):
        return "(incomplete)", ""
    above, below, gap_ok, det = [], [], [], []
    for rev in (False, True):
        a, z = arm[rev], z0[rev]
        margin = max(a["real"][1], z["real"][1])
        d = a["real"][0] - z["real"][0]
        above.append(d > margin)
        below.append(-d > margin)
        gap_ok.append(a["gap"] > a["floor"])
        det.append(f"{'reverse' if rev else 'original'}: real - z0 {d:+.4f} (margin {margin:.4f}), "
                   f"gap {a['gap']:+.4f} {'>' if a['gap'] > a['floor'] else '<='} floor {a['floor']:.4f}")
    if all(above) and all(gap_ok):
        return "B1", "; ".join(det)
    if all(below):
        return "B2", "; ".join(det)
    return "B3", "; ".join(det)


def part_b(tc):
    f = lambda t: f"{t[0]:.4f}+/-{t[1]:.4f}"
    ch = {rev: chance(["PET", "PS"], rev) for rev in (False, True)}
    z0 = {rev: tc[("PETvsPS", "z0", rev)] for rev in (False, True)}
    res, blocks, failures, cw_bad, done = {}, [], [], [], 0
    for arm in ARMS:
        res[arm] = {}
        for rev in (False, True):
            sfx = "_rev" if rev else ""
            cw = f"{PO}/class_weights_twoclass_PETvsPS{sfx}.json"
            rr, rm = load(f"real_tc_PETvsPS_{arm}{sfx}", SEEDS[rev])
            br, bm = load(f"bg_tc_PETvsPS_{arm}{sfx}", SEEDS[rev])
            failures += rm + bm
            done += len(rr) + len(br)
            cw_bad += [r["args"]["out"] for r in rr + br
                       if r["args"].get("class_weights") != cw or r["args"].get("classes") != "PET,PS"
                       or r["args"].get("channels") != arm]
            if not rr or not br:
                res[arm][rev] = None
                continue
            real, pf = side(rr, ["PET", "PS"]), side(br, ["PET", "PS"])
            gap, floor = real["f1"][0] - pf["f1"][0], max(real["f1"][1], pf["f1"][1])
            complete = real["n"] == pf["n"] == SEEDS[rev]
            T = reading(real["f1"][0], pf["f1"][0], gap, floor, ch[rev][0]) if complete else "(incomplete)"
            res[arm][rev] = dict(real=real["f1"], pf=pf["f1"], gap=gap, floor=floor, chance=ch[rev][0],
                                 bacc_r=real["bacc"][0], bacc_pf=pf["bacc"][0], T=T, complete=complete)
            b = ["", "=" * 110, f"PETvsPS  {arm}  {'reverse fold' if rev else 'original direction'}  "
                                f"(real n={real['n']}, particle-free n={pf['n']})", "=" * 110,
                 f"macro-F1: real {f(real['f1'])} | particle-free {f(pf['f1'])} | gap {gap:+.4f} floor {floor:.4f} "
                 f"{'ABOVE' if gap > floor else 'within'} | chance {f(ch[rev])} | reading {T}",
                 f"balanced accuracy: real {f(real['bacc'])} | particle-free {f(pf['bacc'])}",
                 "PET precision / recall / F1: real " + " / ".join(f(real['pet'][k]) for k in ('precision', 'recall', 'f1-score'))
                 + " | particle-free " + " / ".join(f(pf['pet'][k]) for k in ('precision', 'recall', 'f1-score')),
                 f"{'class':6s} {'n test':>6s} {'real recall':>15s} {'p-free recall':>15s} {'gap':>8s} {'floor':>6s} clears?"]
            for c in real["classes"]:
                r, p = real["rec"][c], pf["rec"][c]
                g, fl = r[0] - p[0], max(r[1], p[1])
                b.append(f"{c:6s} {real['support'][c]:>6d} {r[0]:.3f}+/-{r[1]:.3f} {p[0]:.3f}+/-{p[1]:.3f} {g:>+8.3f} "
                         f"{fl:>6.3f}  {'YES' if g > fl else 'no'}")
            cl = real["classes"]
            b.append(f"mean confusion matrix over seeds (rows true, columns predicted: {' '.join(cl)})")
            for name, s_ in (("real", real), ("particle-free", pf)):
                b.append(f"  {name:13s} " + " | ".join(f"{cl[i]}: " + " ".join(f"{v:6.1f}" for v in s_["cm"][i])
                                                       for i in range(len(cl))))
            blocks += b

    L = ["", "=" * 110, "PART B  SUMMARY, PET vs PS (macro-F1 mean +/- std over seeds; balanced accuracy = mean of the two recalls)",
         "=" * 110,
         f"{'arm':24s} {'direction':9s}  {'real':>15s} {'particle-free':>15s} {'gap':>7s} {'floor':>6s} {'chance':>6s}  "
         f"{'bacc R':>6s} {'bacc PF':>7s}  T     B"]
    row = lambda name, rev, d, B: (f"{name:24s} {'reverse' if rev else 'original':9s}  {f(d['real'])} {f(d['pf'])} "
                                   f"{d['gap']:+.4f} {d['floor']:.4f} {d['chance']:.4f}  {d['bacc_r']:.4f} {d['bacc_pf']:.4f}  "
                                   f"{d['T']:5s} {B}")
    for name in ("z0", "intensity_bgcorr_vnorm"):
        for rev in (False, True):
            L.append(row(name, rev, tc[("PETvsPS", name, rev)], "(copied from report_twoclass.txt)"))
    details = []
    for arm in ARMS:
        B, det = b_reading(res[arm], z0) if all(res[arm][r] and res[arm][r]["complete"] for r in (False, True)) \
            else ("(incomplete)", "")
        for rev in (False, True):
            d = res[arm][rev]
            L.append(row(arm, rev, d, B) if d else f"{arm:24s} {'reverse' if rev else 'original':9s}  not run yet")
        if det:
            details.append(f"  {arm}: {B}  ({det})")
    L += ["B reading detail (z0 real: original 0.8735+/-0.0217, reverse 0.7593+/-0.0078):"] + (details or ["  none complete yet"])
    L += blocks
    return L, done, failures, cw_bad, res


def main():
    tc = copied_twoclass()
    z0 = {(f"PETvs{o}", False): tc[(f"PETvs{o}", "z0", False)] for o in OTHERS}
    z0[("PETvsPS", True)] = tc[("PETvsPS", "z0", True)]
    lines = [HEADER] + part_a(z0) + A3.split("\n")
    bl, done, failures, cw_bad, res = part_b(tc)
    lines += bl
    not_run = [f"{a} {'reverse' if r else 'original'}" for a in ARMS for r in (False, True) if res[a][r] is None]
    failed_txt = "results/logs_size_arms/failed.txt"
    logged = open(failed_txt).read().split() if os.path.exists(failed_txt) else []
    lines += ["", "=" * 110, f"runs finished: {done} of 64 (4 arms x (5 + 5 original + 3 + 3 reverse))",
              "class-weights file (pipeline_output/class_weights_twoclass_PETvsPS[_rev].json), --classes PET,PS and "
              f"--channels recorded as expected in every finished run: {'yes' if not cw_bad else 'NO: ' + str(cw_bad)}",
              f"failed / incomplete runs: {failures or 'none'}" + (f"; logged failures: {logged}" if logged else ""),
              f"arm-directions not run: {not_run or 'none'}"]
    text = "\n".join(lines)
    open("results/report_size_and_arms.txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
