"""J1 (item 41): single-feature breakdown of the Part 3b brightness scalars (read-only, no training, new files only).
Reuses scripts/scalar_baselines_PETPS_exp.py unchanged: its fit() (class-weighted logistic regression, L2, C=1.0,
lbfgs, class weights n_train / (2 x n_c), standardised on the training split, fitted on the training split only,
1000 patch and 1000 track bootstrap resamples with numpy default_rng(0)) and the saved Part 3b features
(results/expanded_PETPS_exp/part3_features.csv: patch mean, std, min, max of z0, and mean and sum of z0 - 1 over the
detected region; particle-free rows use a central disc with the same pixel count as the paired real region).
Step 1 reproduces the all-six result on real patches (0.8016 original, 0.7852 reverse, all patches, 4 decimals) and
stops if it does not. Then each feature alone, drop-one, and three groups, on real and particle-free patches, both
directions, all patches. Y readings against R = 0.9052 (original) and 0.8477 (reverse), fixed in the task.
Writes results/report_brightness_features_exp.txt and results/brightness_features_exp.json. Idle priority."""
import os
import sys
import json
import ctypes
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import scalar_baselines_PETPS_exp as sb  # noqa: E402

R = {"original": 0.9052, "reverse": 0.8477}
COL = {"original": "split_video_holdout", "reverse": "split_rev"}
NAMES = {"z0_mean": "patch mean", "z0_std": "patch std", "z0_min": "patch min", "z0_max": "patch max",
         "region_mean_z0m1": "region mean of z0-1", "region_sum_z0m1": "region sum of z0-1"}
OUT = "results/report_brightness_features_exp.txt"


def y_reading(F, r):
    return "Y1" if F >= r - 0.05 else ("Y2" if F <= r - 0.15 else "Y3")


def main():
    f = pd.read_csv("results/expanded_PETPS_exp/part3_features.csv")
    real = f[f.kind == "real"]
    L = ["J1: SINGLE-FEATURE BREAKDOWN OF THE PART 3b BRIGHTNESS SCALARS (z0, PET vs PS, expanded set, track-based validation)",
         "Code and features of Part 3b reused unchanged (scripts/scalar_baselines_PETPS_exp.py fit(); "
         "results/expanded_PETPS_exp/part3_features.csv). All patches. Track CI: 1000 resamples of whole test tracks.",
         "Y readings (fixed in the task): R = 0.9052 original, 0.8477 reverse; Y1 F >= R - 0.05, Y2 F <= R - 0.15, Y3 between."]
    rep = {}
    for d in ("original", "reverse"):
        x = sb.fit(real, sb.BRIGHT, COL[d])
        rep[d] = round(x["f1"], 4)
    target = {"original": 0.8016, "reverse": 0.7852}
    ok = all(rep[d] == target[d] for d in rep)
    L.append(f"Reproduction of the all-six fit on real patches: original {rep['original']:.4f} (target 0.8016), reverse "
             f"{rep['reverse']:.4f} (target 0.7852) -> {'reproduced' if ok else 'NOT reproduced: job stopped'}")
    if not ok:
        open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
        return
    sets = [("all six", list(sb.BRIGHT))]
    sets += [(f"alone: {NAMES[c]}", [c]) for c in sb.BRIGHT]
    sets += [(f"drop: {NAMES[c]}", [x for x in sb.BRIGHT if x != c]) for c in sb.BRIGHT]
    sets += [("group level {patch mean}", ["z0_mean"]), ("group contrast {std, min, max}", ["z0_std", "z0_min", "z0_max"]),
             ("group deficit {region mean, region sum}", ["region_mean_z0m1", "region_sum_z0m1"])]
    rows = []
    L += ["", f"{'feature set':42s} {'patches':13s} {'direction':9s} {'macro-F1':>8s} {'track 95% CI':>17s} {'rec PET':>7s} "
              f"{'rec PS':>6s}  {'R':>6s} reading"]
    for name, feats in sets:
        for kind in ("real", "particle-free"):
            for d in ("original", "reverse"):
                x = sb.fit(f[f.kind == kind], feats, COL[d])
                y = y_reading(x["f1"], R[d])
                rows.append(dict(set=name, features=feats, kind=kind, direction=d, f1=x["f1"], lo_track=x["lo_track"],
                                 hi_track=x["hi_track"], rec_PET=x["rec_PET"], rec_PS=x["rec_PS"], Y=y))
                L.append(f"{name:42s} {kind:13s} {d:9s} {x['f1']:8.4f} [{x['lo_track']:.4f}, {x['hi_track']:.4f}] "
                         f"{x['rec_PET']:7.3f} {x['rec_PS']:6.3f}  {R[d]:.4f} {y}")
    L += ["", "Feature values on the real patches, mean +/- std per class (training and test split of each direction):",
          f"{'feature':22s} {'direction':9s} {'split':5s} {'PET':>21s} {'PS':>21s}"]
    for c in sb.BRIGHT:
        for d in ("original", "reverse"):
            for s in ("train", "test"):
                g = real[real[COL[d]] == s]
                v = {k: g[g["class"] == k][c] for k in ("PET", "PS")}
                L.append(f"{NAMES[c]:22s} {d:9s} {s:5s} " + " ".join(f"{v[k].mean():>10.5f}+/-{v[k].std():<10.5f}" for k in ("PET", "PS")))
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    open(OUT, "w", encoding="utf-8").write(text + "\n")
    json.dump(dict(reproduced=rep, rows=rows), open("results/brightness_features_exp.json", "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
