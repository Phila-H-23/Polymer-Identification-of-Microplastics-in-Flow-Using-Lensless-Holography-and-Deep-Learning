"""Item 46 analysis (run automatically by scripts/item46_waiter.py when the item 46 chain exits; idle priority).
Reuses the item 44 evaluation (scripts/item44_chain.py evaluate() and stats(): best.pt on the unshifted test patches,
4-decimal check, 1000 resamples of whole test tracks with default_rng(0), mean over seeds per resample). Wn runs are
evaluated with scripts/eval_track_ci_exp.py evaluate() (flat z0 patch folder). Width-32 references: arm D from
results/depth_aug_predictions_exp.csv, z0 from results/z0_predictions_exp.csv (real, seeds 0 to 2).
Writes results/report_width_depth_exp.txt and results/width_depth_predictions_exp.csv."""
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
import item44_chain as c44  # noqa: E402
import eval_track_ci_exp as ev  # noqa: E402
prio41.set_priority(prio41.IDLE)

RES = "results"
REPORT, PRED = f"{RES}/report_width_depth_exp.txt", f"{RES}/width_depth_predictions_exp.csv"
B = 1000


def evaluate(arm, d):
    if arm == "Wd":
        df, f1, f1s = c44.evaluate(d)
    else:
        df, f1, f1s, _ = ev.evaluate(d)
    return df.rename(columns={"class": "true_class", "pred": "predicted_class"}), f1, f1s


def cmp(A, B_, W, tracks):
    """Paired comparison of arm A minus arm B over their shared seeds."""
    s = sorted(set(A.seed) & set(B_.seed))
    a, b = c44.stats(A[A.seed.isin(s)], tracks, W), c44.stats(B_[B_.seed.isin(s)], tracks, W)
    d = a["boot"] - b["boot"]
    return dict(seeds=s, a=a["f1"], b=b["f1"], diff=a["f1"][0] - b["f1"][0], margin=max(a["f1"][1], b["f1"][1]),
                ci=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))))


def main(runs_path=f"{RES}/logs_item46/runs.json"):
    rj = json.load(open(runs_path))
    runs, timing, dirs = rj["runs"], rj["timing"], rj["dirs"]
    d32 = pd.read_csv(f"{RES}/depth_aug_predictions_exp.csv")
    z32 = pd.read_csv(f"{RES}/z0_predictions_exp.csv")
    z32 = z32[(z32.kind == "real") & z32.seed.isin([0, 1, 2])].rename(columns={"patch_id": "patch_file"})
    allp, excl, res = [], [], {}
    for direction in ("original", "reverse"):
        arms = {}
        for arm in ("Wn", "Wd"):
            P = []
            for s, d in sorted(runs.get(arm, {}).get(direction, {}).items(), key=lambda x: int(x[0])):
                df, f1, f1s = evaluate(arm, d)
                if round(f1, 4) != round(f1s, 4):
                    excl.append(f"{d} (recomputed {f1:.4f} vs saved {f1s:.4f})")
                    continue
                P.append(df.assign(seed=int(s), run=os.path.basename(d), arm=arm, direction=direction))
            if P:
                arms[arm] = pd.concat(P)
                allp.append(arms[arm])
        if not arms:
            continue
        arms["D"] = d32[(d32.arm == "D") & (d32.direction == direction)]
        arms["z0"] = z32[z32.direction == direction]
        tracks = sorted(set(arms["z0"].track_id))
        rng = np.random.default_rng(0)
        W = np.stack([np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks)) for _ in range(B)])
        st = {k: c44.stats(v, tracks, W) for k, v in arms.items() if len(v)}
        c = {}
        if "Wd" in arms:
            c["Wd - D"] = cmp(arms["Wd"], arms["D"], W, tracks)
        if "Wd" in arms and "Wn" in arms:
            c["Wd - Wn"] = cmp(arms["Wd"], arms["Wn"], W, tracks)
        if "Wn" in arms:
            c["Wn - z0"] = cmp(arms["Wn"], arms["z0"], W, tracks)
        res[direction] = dict(stats=st, cmp=c, seeds={k: sorted(set(v.seed)) for k, v in arms.items()})
    if allp:
        pd.concat(allp)[["run", "arm", "direction", "seed", "patch_file", "track_id", "true_class", "predicted_class"]].to_csv(PRED, index=False)

    def holds(direction, key):
        c = res.get(direction, {}).get("cmp", {})
        if key == "X1":
            return "Wd - D" in c and "Wd - Wn" in c and c["Wd - D"]["diff"] > c["Wd - D"]["margin"] and c["Wd - Wn"]["diff"] > c["Wd - Wn"]["margin"]
        if key == "X2":
            return "Wn - z0" in c and c["Wn - z0"]["diff"] > c["Wn - z0"]["margin"]
        if key == "X3":
            return ("Wd - D" in c and "Wn - z0" in c and abs(c["Wd - D"]["diff"]) <= c["Wd - D"]["margin"]
                    and abs(c["Wn - z0"]["diff"]) <= c["Wn - z0"]["margin"])
        if key == "X4":
            return "Wd - D" in c and c["Wd - D"]["diff"] < -c["Wd - D"]["margin"]
    X = next((k for k in ("X1", "X2", "X3", "X4") if all(holds(d, k) for d in ("original", "reverse"))), "X5")
    Xorig = next((k for k in ("X1", "X2", "X3", "X4") if holds("original", k)), "X5")
    p0 = "not available"
    if os.path.exists(f"{RES}/item46_part0_trainfit.txt"):
        p0 = [l for l in open(f"{RES}/item46_part0_trainfit.txt", encoding="utf-8") if l.startswith("Reading:")][-1].strip()
    L = ["ITEM 46: WIDER CNN (WIDTH 64) WITH AND WITHOUT DEPTH AUGMENTATION, z0 EXPANDED SET, PET vs PS, REAL RUNS ONLY", "",
         f"READINGS (results/item46_readings.md): Part 1 {X}. Part 0 {p0} (results/item46_part0_trainfit.txt).",
         f"Directions run: {', '.join(dirs)}." + ("" if set(dirs) == {"original", "reverse"} else
         " X1 to X4 require both directions, so as written only X5 can hold; on the original direction alone the first "
         f"condition that holds is {Xorig} (descriptive, not the pre-registered reading)."),
         "No particle-free runs, so there is no T reading. Nothing here shows that the CNN identifies polymers.",
         "Margin for each comparison = the larger real seed std (population std) of the two arms over their shared seeds.", ""]
    for direction, r in res.items():
        L.append(f"{direction.upper()} DIRECTION (seeds: " + ", ".join(f"{k} {v}" for k, v in r["seeds"].items()) + ")")
        L.append(f"  {'arm':14s} {'real macro-F1':>16s} {'track CI':>17s}  recall PET / PS [CI]                 precision PET / PS [CI]")
        for k, lab in (("Wn", "Wn (w64, none)"), ("Wd", "Wd (w64, all21)"), ("D", "D (w32, all21)"), ("z0", "z0 (w32, none)")):
            if k not in r["stats"]:
                continue
            x = r["stats"][k]
            L.append(f"  {lab:14s} {x['f1'][0]:.4f}+/-{x['f1'][1]:.4f} [{x['ci'][0]:.4f}, {x['ci'][1]:.4f}]  "
                     f"{x['rec'][0][0][0]:.3f} [{x['rec'][0][1][0]:.3f}, {x['rec'][0][1][1]:.3f}] / {x['rec'][1][0][0]:.3f} "
                     f"[{x['rec'][1][1][0]:.3f}, {x['rec'][1][1][1]:.3f}]   {x['prec'][0][0][0]:.3f} [{x['prec'][0][1][0]:.3f}, "
                     f"{x['prec'][0][1][1]:.3f}] / {x['prec'][1][0][0]:.3f} [{x['prec'][1][1][0]:.3f}, {x['prec'][1][1][1]:.3f}]")
        for k, c in r["cmp"].items():
            v = "above" if c["diff"] > c["margin"] else ("below" if c["diff"] < -c["margin"] else "within")
            L.append(f"  {k:8s} {c['diff']:+.4f} (margin {c['margin']:.4f}, {v}; paired track CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}], seeds {c['seeds']})")
        L.append("")
    L += [f"Runs excluded by the 4-decimal check: {excl or 'none'}. Test predictions: {PRED}.",
          "Run times: " + ", ".join(f"{t['run']} {t['seconds']} s ({t['epochs']} epochs){'' if t['ok'] else ' FAILED'}" for t in timing),
          f"Written {time.strftime('%Y-%m-%d %H:%M')}."]
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    if os.path.exists(REPORT):
        raise FileExistsError(REPORT)
    open(REPORT, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
