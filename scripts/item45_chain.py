"""Item 45 chain (detached, idle priority; one run at a time; 1.5 GB RAM wait; retry once): arm Pz, exactly as in
item 43 (scripts/item43_chain.py pz_cmd, run; --channels pol4_bgcorr on pipeline_output/patches_recon_z0pol4_PETPS_exp,
every other setting of the z0 expanded real runs, each pair with its z0 manifest and class weights). Real only, seeds 0
to 2, pairs PET2+PS1 (original), PET1+PS2 (reverse), PET1+PS1, PET2+PS2. The two finished original-pair runs (seeds 0
and 1) are reused (their recorded settings equal this command); original seed 2 runs fresh as ..._orig_s2_fresh, the
interrupted ..._orig_s2 is left in place. Evaluation from best.pt (eval_track_ci_exp.evaluate(), 4-decimal check),
readings Pz1 to Pz4 (item 43), per-class recall and precision with track intervals and the paired track bootstrap of
the difference (scripts/item44_chain.py stats(): 1000 resamples of whole test tracks, default_rng(0), mean over seeds
per resample). Writes results/report_pol4_exp.txt and the summary at the top of results/item45_status.md."""
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
import item45_status as st  # noqa: E402
import item43_chain as c43  # noqa: E402
import item44_chain as c44  # noqa: E402
prio41.set_priority(prio41.IDLE)

RES = "results"
REPORT = f"{RES}/report_pol4_exp.txt"
LOGD = f"{RES}/logs_item45"
c43.st, c43.LOGD = st, LOGD  # item 43 runner, logging to this item
B = 1000


def main():
    os.makedirs(LOGD, exist_ok=True)
    c43.log(f"item 45 chain started, pid {os.getpid()}, idle priority")
    st.update("train", status="running", started=time.strftime("%Y-%m-%d %H:%M"), pid=os.getpid())
    runs, timing = {}, []
    for name, man, cw, zpref, zpred, zdir, tag in c43.PAIRS:
        for s in (0, 1, 2):
            out = f"{RES}/real_tc_PETvsPS_z0pol4_exp_{tag}_s{s}"
            if tag == "orig" and s == 2:
                out += "_fresh"
            t0 = time.time()
            reused = os.path.exists(f"{out}/results.json")
            d = c43.run("train", c43.pz_cmd(man, cw, s, out), out)
            timing.append(dict(run=os.path.basename(out), seconds=0 if reused else round(time.time() - t0), reused=reused, ok=bool(d)))
            if d:
                runs.setdefault(tag, {})[s] = d
            json.dump(dict(runs=runs, timing=timing), open(f"{LOGD}/runs.json", "w"), indent=1)
            st.update("train", progress=f"{sum(len(v) for v in runs.values())} of 12 run results available (2 reused)")
    st.update("train", status="done", finished=time.strftime("%Y-%m-%d %H:%M"))
    c43.gate("report", "evaluation")
    report(runs, timing)
    c43.log("finished")


def report(runs, timing):
    import eval_track_ci_exp as ev
    res, excl, preds = [], [], []
    for name, man, cw, zpref, zpred, zdir, tag in c43.PAIRS:
        P = []
        for s, d in sorted(runs.get(tag, {}).items()):
            df, f1, f1s, _ = ev.evaluate(d)
            if round(f1, 4) != round(f1s, 4):
                excl.append(f"{d} (recomputed {f1:.4f} vs saved {f1s:.4f})")
                continue
            P.append(df.assign(seed=s, run=os.path.basename(d)))
        if len(P) < 2:
            continue
        P = pd.concat(P).rename(columns={"class": "true_class", "pred": "predicted_class"})
        preds.append(P.assign(pair=name)[["run", "pair", "seed", "patch_file", "track_id", "true_class", "predicted_class"]])
        Z = pd.read_csv(zpred)
        Z = Z[(Z.direction == zdir) & (Z.kind == "real")].rename(columns={"patch_id": "patch_file"})
        seeds = sorted(set(P.seed) & set(Z.seed))
        P, Z = P[P.seed.isin(seeds)], Z[Z.seed.isin(seeds)]
        assert set(P.patch_file) == set(Z.patch_file)
        tracks = sorted(set(Z.track_id))
        rng = np.random.default_rng(0)
        W = np.stack([np.bincount(rng.integers(0, len(tracks), len(tracks)), minlength=len(tracks)) for _ in range(B)])
        a, z = c44.stats(P, tracks, W), c44.stats(Z, tracks, W)
        margin, diff = max(a["f1"][1], z["f1"][1]), a["f1"][0] - z["f1"][0]
        db = a["boot"] - z["boot"]
        res.append(dict(pair=name, seeds=seeds, a=a, z=z, margin=margin, diff=diff,
                        verdict="above" if diff > margin else ("below" if diff < -margin else "within"),
                        ci=(float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))), n=len(set(Z.patch_file)), nt=len(tracks)))
    out = f"{RES}/pol4_z0_predictions_exp.csv"
    if preds and not os.path.exists(out):
        pd.concat(preds).to_csv(out, index=False)
    ab, be = sum(r["verdict"] == "above" for r in res), sum(r["verdict"] == "below" for r in res)
    R = (f"incomplete ({len(res)} of 4 pairs)" if len(res) < 4 else "Pz1" if ab >= 2 and be == 0 else
         "Pz2" if all(r["verdict"] == "within" for r in res) else "Pz3" if be >= 2 and ab == 0 else "Pz4")
    L = ["POLARISATION UNDER z0 (ARM Pz), PET vs PS (item 45; continues the item 43 arm)", "",
         f"READING: {R} (above in {ab}, below in {be}, within in {sum(r['verdict'] == 'within' for r in res)} of {len(res)} pairs).",
         "  Fixed in item 43: per pair, Pz real macro-F1 against z0 real over the shared seeds; margin = the larger real seed std",
         "  (population std) of the two; above if Pz > z0 + margin, below if Pz < z0 - margin, else within. Pz1 above in >= 2",
         "  pairs and below in none; Pz2 within in every pair; Pz3 below in >= 2 pairs and above in none; Pz4 none of these.",
         "Real runs only, so there is no T reading for this arm.", "",
         "SETUP: per-channel z0 patches (pipeline_output/patches_recon_z0pol4_PETPS_exp; channels I0, I45, I90, I135, each the",
         "frame divided by that channel's per-pixel median of up to 90 preceding frames, stored x 255 because the script divides",
         "4-channel patches by 255), --channels pol4_bgcorr, every other setting of the z0 expanded real runs, each pair's z0",
         "manifest and class weights. Reused: results/real_tc_PETvsPS_z0pol4_exp_orig_s0 and _orig_s1 (item 43; recorded settings",
         "equal to this command). Original seed 2 ran fresh as results/real_tc_PETvsPS_z0pol4_exp_orig_s2_fresh; the interrupted",
         "item 43 run results/real_tc_PETvsPS_z0pol4_exp_orig_s2 (best.pt only) is left in place and not used.",
         f"Runs excluded by the 4-decimal check: {excl or 'none'}. Test predictions: {out}.", "",
         f"{'test pair':22s} {'seeds':9s} {'Pz real [track CI]':>32s} {'z0 real':>15s} {'Pz - z0':>8s} {'margin':>7s} {'':6s} paired track CI"]
    for r in res:
        L.append(f"{r['pair']:22s} {str(r['seeds']):9s} {r['a']['f1'][0]:.4f}+/-{r['a']['f1'][1]:.4f} [{r['a']['ci'][0]:.4f}, {r['a']['ci'][1]:.4f}] "
                 f"{r['z']['f1'][0]:.4f}+/-{r['z']['f1'][1]:.4f} {r['diff']:+.4f} {r['margin']:.4f} {r['verdict']:6s} "
                 f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  ({r['n']} test patches, {r['nt']} tracks)")
    L.append("Per-class recall and precision, real (mean +/- seed std [track 95% CI]):")
    for r in res:
        for lab, x in (("Pz", r["a"]), ("z0", r["z"])):
            L.append(f"  {r['pair']:22s} {lab}: recall PET {x['rec'][0][0][0]:.3f}+/-{x['rec'][0][0][1]:.3f} [{x['rec'][0][1][0]:.3f}, "
                     f"{x['rec'][0][1][1]:.3f}]  PS {x['rec'][1][0][0]:.3f}+/-{x['rec'][1][0][1]:.3f} [{x['rec'][1][1][0]:.3f}, {x['rec'][1][1][1]:.3f}]"
                     f" | precision PET {x['prec'][0][0][0]:.3f}+/-{x['prec'][0][0][1]:.3f} [{x['prec'][0][1][0]:.3f}, {x['prec'][0][1][1]:.3f}]"
                     f"  PS {x['prec'][1][0][0]:.3f}+/-{x['prec'][1][0][1]:.3f} [{x['prec'][1][1][0]:.3f}, {x['prec'][1][1][1]:.3f}]")
    L += ["", "Run times: " + ", ".join(f"{t['run']} {'reused' if t['reused'] else str(t['seconds']) + ' s'}{'' if t['ok'] else ' FAILED'}" for t in timing),
          "Deviation carried over from item 43: the per-channel check matched the stored z0 patches to float32 precision (one bright",
          "pixel 1.43e-6 absolute, relative 4.5e-8)."]
    text = "\n".join(L)
    assert "\u2014" not in text and "significant" not in text.lower()
    if os.path.exists(REPORT):
        raise FileExistsError(REPORT)
    open(REPORT, "w", encoding="utf-8").write(text + "\n")
    st.top([f"Reading {R}: above in {ab}, below in {be} of {len(res)} pairs (real only, no T reading)."] +
           [f"{r['pair']}: Pz {r['a']['f1'][0]:.4f} vs z0 {r['z']['f1'][0]:.4f} ({r['diff']:+.4f}, margin {r['margin']:.4f}, {r['verdict']}; "
            f"paired CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}])" for r in res][:4], [f"Report {REPORT}."])
    st.update("report", status="done", finished=time.strftime("%Y-%m-%d %H:%M"), reading=R, files=[REPORT, out, "scripts/item45_chain.py"])
    c43.log(f"report written: {R}")


if __name__ == "__main__":
    main()
