"""Item 43, step 1 (evaluation only, from the saved per-patch predictions; nothing retrained): table and figure of all
completed z0 (PET video, PS video) test pairs, original and reverse included.
usage: python scripts/item43_pairs_table.py <csv out> <png out> <report section title>"""
import os
import sys
import json

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import unclipped_tracks as ut  # noqa: E402

PO, RES = "pipeline_output", "results"
CL = ["PET", "PS"]
B = 1000


def t_reading(real, pf, gap, floor, ch, margin=0.10):  # scripts/eval_track_ci_exp.py, unchanged
    if gap > floor and real > ch + margin:
        return "T1"
    if gap <= floor and pf > ch + margin:
        return "T2"
    if real <= ch + margin and pf <= ch + margin:
        return "T3"
    return "T4"


def pairs():
    out = [dict(pair="PET2+PS1", role="original", pred=f"{RES}/z0_predictions_exp.csv", direction="original",
                man=f"{PO}/manifest_recon_s0_z0_PETPS_exp.csv"),
           dict(pair="PET1+PS2", role="reverse", pred=f"{RES}/z0_predictions_exp.csv", direction="reverse",
                man=f"{PO}/manifest_recon_s0_z0_PETPS_rev_exp.csv")]
    for pet in ("PET1", "PET2", "PET3", "PET4"):
        for ps in ("PS1", "PS2"):
            p = f"{RES}/z0_pair_{pet}_{ps}_predictions_exp.csv"
            if os.path.exists(p):
                out.append(dict(pair=f"{pet}+{ps}", role="new", pred=p, direction="pair",
                                man=f"{PO}/manifest_recon_s0_z0_PETPS_pair_{pet}_{ps}_exp.csv"))
    return out


def one(spec):
    P = pd.read_csv(spec["pred"])
    P = P[P.direction == spec["direction"]]
    m = pd.read_csv(spec["man"])
    seeds = sorted(set(P.seed))
    tracks = sorted(set(P.track_id))
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks)) for _ in range(B)])
    r = {}
    for kind in ("real", "particle-free"):
        f1, rec, prec, bf = [], [], [], []
        for s in seeds:
            C = ut.per_track_counts(P[(P.kind == kind) & (P.seed == s)], tracks, CL)
            f, rc, pc = ut.metrics(C.sum(0))
            f1.append(float(f))
            rec.append(rc)
            prec.append(pc)
            bf.append(ut.metrics(np.einsum("bt,tij->bij", W, C))[0])
        r[kind] = dict(f1=ut.ms(f1), rec=[ut.ms([float(x[i]) for x in rec]) for i in range(2)],
                       prec=[ut.ms([float(x[i]) for x in prec]) for i in range(2)], boot=np.mean(bf, axis=0),
                       bacc=ut.ms([float(np.mean(x)) for x in rec]))
    te = m[m.split_video_holdout == "test"]
    ch = ut.chance(m[m.split_video_holdout == "train"]["class"].values, te["class"].values, CL)
    gap = r["real"]["f1"][0] - r["particle-free"]["f1"][0]
    floor = max(r["real"]["f1"][1], r["particle-free"]["f1"][1])
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    gb = r["real"]["boot"] - r["particle-free"]["boot"]
    n = te["class"].value_counts()
    vids = sorted(te.source_file.str.replace("_600.mat", "", regex=False).unique())
    return dict(pair=spec["pair"], role=spec["role"], test_videos="+".join(vids), n_test_PET=int(n.get("PET", 0)),
                n_test_PS=int(n.get("PS", 0)), PET_share=round(n.get("PET", 0) / len(te), 4), seeds=len(seeds),
                real_f1=round(r["real"]["f1"][0], 4), real_f1_std=round(r["real"]["f1"][1], 4),
                real_ci_lo=round(q(r["real"]["boot"])[0], 4), real_ci_hi=round(q(r["real"]["boot"])[1], 4),
                pf_f1=round(r["particle-free"]["f1"][0], 4), pf_f1_std=round(r["particle-free"]["f1"][1], 4),
                pf_ci_lo=round(q(r["particle-free"]["boot"])[0], 4), pf_ci_hi=round(q(r["particle-free"]["boot"])[1], 4),
                gap=round(gap, 4), gap_ci_lo=round(q(gb)[0], 4), gap_ci_hi=round(q(gb)[1], 4), floor=round(floor, 4),
                chance=round(ch, 4), T=t_reading(r["real"]["f1"][0], r["particle-free"]["f1"][0], gap, floor, ch),
                real_recall_PET=round(r["real"]["rec"][0][0], 4), real_recall_PS=round(r["real"]["rec"][1][0], 4),
                real_precision_PET=round(r["real"]["prec"][0][0], 4), real_precision_PS=round(r["real"]["prec"][1][0], 4),
                real_balanced_accuracy=round(r["real"]["bacc"][0], 4), n_tracks=len(tracks))


def main(csv_out, png_out, title):
    for p in (csv_out, png_out):
        if os.path.exists(p):
            raise FileExistsError(p)
    rows = [one(s) for s in pairs()]
    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    x = np.arange(len(df))
    for off, k, col, lab in ((-0.12, "real", "tab:blue", "real"), (0.12, "pf", "tab:orange", "particle-free")):
        y = df[f"{k}_f1"].values
        lo, hi = y - df[f"{k}_ci_lo"].values, df[f"{k}_ci_hi"].values - y
        ax.errorbar(x + off, y, yerr=[lo, hi], fmt="o", color=col, capsize=3, label=f"{lab} macro-F1 (track 95% CI)")
    for i, r in df.iterrows():
        if r.role != "new":
            ax.axvspan(i - 0.4, i + 0.4, color="0.9", zorder=0)
            ax.text(i, 0.36, r.role, ha="center", fontsize=8)
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"test {p}\n({a} PET / {b} PS)" for p, a, b in zip(df.pair, df.n_test_PET, df.n_test_PS)], fontsize=8)
    ax.set_ylabel("macro-F1 (mean over seeds)")
    ax.set_ylim(0.33, 1.0)
    ax.set_title("z0, expanded set, PET vs PS: one point per (PET video, PS video) test pair; grey = original and reverse.\n"
                 "The pairs share videos, so they are not independent samples.", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(png_out)
    cols = ["pair", "role", "n_test_PET", "n_test_PS", "PET_share", "seeds", "real_f1", "real_f1_std", "real_ci_lo", "real_ci_hi",
            "pf_f1", "gap", "gap_ci_lo", "gap_ci_hi", "floor", "chance", "T", "real_recall_PET", "real_recall_PS",
            "real_precision_PET", "real_precision_PS", "real_balanced_accuracy"]
    L = ["", title, f"Files: {csv_out} (all columns), {png_out}. From the saved per-patch predictions (results/z0_predictions_exp.csv",
         "and results/z0_pair_<PET>_<PS>_predictions_exp.csv); tracks resampled 1000 times (default_rng(0)); chance from 1000",
         "random label draws with each pair's training class proportions; floor = larger seed std; T as before.",
         "The pairs share videos (every video is the test video of some pairs and a training video of the others), so they are",
         "not independent samples: no mean or standard error over pairs is given as if they were.",
         f"{'test pair':10s} {'role':8s} {'PET/PS test':>11s} {'PET %':>5s} {'real macro-F1 [track CI]':>32s} {'p-free':>6s} "
         f"{'gap [track CI]':>25s} {'floor':>6s} {'chance':>6s} T  rec PET/PS  prec PET/PS  bal.acc"]
    for _, r in df.iterrows():
        L.append(f"{r.pair:10s} {r.role:8s} {r.n_test_PET:5d}/{r.n_test_PS:<5d} {100 * r.PET_share:5.1f} {r.real_f1:.4f}+/-{r.real_f1_std:.4f} "
                 f"[{r.real_ci_lo:.4f}, {r.real_ci_hi:.4f}] {r.pf_f1:.4f} {r.gap:+.4f} [{r.gap_ci_lo:+.4f}, {r.gap_ci_hi:+.4f}] "
                 f"{r.floor:.4f} {r.chance:.4f} {r['T']} {r.real_recall_PET:.3f}/{r.real_recall_PS:.3f} "
                 f"{r.real_precision_PET:.3f}/{r.real_precision_PS:.3f}  {r.real_balanced_accuracy:.4f}")
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    with open(f"{RES}/report_z0_pol4_exp.txt", "a", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    return df


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
