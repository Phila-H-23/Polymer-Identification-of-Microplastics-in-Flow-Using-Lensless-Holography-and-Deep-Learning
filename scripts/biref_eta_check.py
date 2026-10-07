"""
biref_eta_check.py

The one fix attempt for the A_bg/B_bg domain mix, evaluated offline from the
saved recon_prop patches (6 channels: I0p, I45p, I90p, I135p, A_bg, B_bg -
see build_reconstructed_dataset.py). A_bg/B_bg are per-video maps from 30
propagated, per-channel-divided frames, i.e. the same domain as A/B.
Read-only; writes results/biref_eta_check*.csv/json. No training.

Reports:
  (a) the SAME 5 particles / 13 background-only patches as the earlier check
      (reconstruction_zsweep2.py section 5: darkest-25% s0 mask for the
      particle, whole patch for background-only; bg sample n<=5 per class,
      random_state=0) - directly comparable to the old 0.2537 / 0.2410
  (b) every patch: real particle-mask eta vs background-only eta, per class
  (c) within-video drift of A_bg/B_bg from results/recon_abbg_drift.csv
Pass condition (fixed before the first run): background-only mean eta
<= 0.0094 + 2*0.0053 = 0.020.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import reconstruction as recon
import reconstruction_validate as rv

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
PO = os.path.join(DATA_DIR, "pipeline_output")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PASS_THRESHOLD = 0.0094 + 2 * 0.0053
OLD = dict(particle=0.2537, background=0.2410)


def patch_eta(prop, correct=True):
    I0p, I45p, I90p, I135p, A_bg, B_bg = (prop[..., c] for c in range(6))
    if correct:
        eta, _ = recon.birefringence_eta_theta(I0p, I45p, I90p, I135p, A_bg=A_bg, B_bg=B_bg)
    else:
        eta, _ = recon.birefringence_eta_theta(I0p, I45p, I90p, I135p)
    return eta, (I0p + I45p + I90p + I135p) / 4.0


def summarise(row, kind, prop):
    eta, s0 = patch_eta(prop)
    eta_nc, _ = patch_eta(prop, correct=False)
    mask = s0 <= np.nanpercentile(s0, 25) if kind == "real" else np.ones(eta.shape, bool)
    return dict(patch_file=row["patch_file"], cls=row["class"], source_file=row["source_file"],
                frame=row["frame"], kind=kind, eta=float(np.nanmean(eta[mask])),
                eta_uncorrected=float(np.nanmean(eta_nc[mask])), nan_frac=float(np.isnan(eta[mask]).mean()))


def stats_line(s):
    return f"mean={s.mean():.4f} std={s.std(ddof=0):.4f} n={len(s)}"


def main():
    real = pd.read_csv(os.path.join(PO, "manifest_recon_prop.csv"))
    bg = pd.read_csv(os.path.join(PO, "manifest_background_recon_prop.csv"))
    rows = []
    for kind, df, pdir in (("real", real, "patches_recon_prop"), ("background", bg, "patches_background_recon_prop")):
        for _, row in df.iterrows():
            rows.append(summarise(row, kind, np.load(os.path.join(PO, pdir, row["patch_file"]))))
    allp = pd.DataFrame(rows)
    allp.to_csv(os.path.join(RESULTS_DIR, "biref_eta_check_all.csv"), index=False)

    # (a) same evaluation set as the earlier check
    bg_bgc = pd.read_csv(os.path.join(PO, "manifest_background_bgcorrected.csv"))
    part_files, bg_files = [], []
    for cls, sp in rv.FIVE_PARTICLES.items():
        hit = real[(real["source_file"] == sp["source_file"]) & (real["frame"] == sp["frame"])
                   & (real["x"] == sp["x"]) & (real["y"] == sp["y"])]
        part_files += list(hit["patch_file"])
        sub = bg_bgc[(bg_bgc["class"] == cls) & (bg_bgc["source_file"] == sp["source_file"])
                     & (bg_bgc["frame"] == sp["frame"])]
        bg_files += list(sub.sample(n=min(5, len(sub)), random_state=0)["patch_file"])
    a_part = allp[(allp["kind"] == "real") & allp["patch_file"].isin(part_files)]["eta"]
    a_bg = allp[(allp["kind"] == "background") & allp["patch_file"].isin(bg_files)]["eta"]

    print("=== (a) same 5 particles / background patches as the earlier check ===")
    print(f"particle-mask eta:   {stats_line(a_part)}   [old raw-domain correction: {OLD['particle']}]")
    print(f"background-only eta: {stats_line(a_bg)}   [old raw-domain correction: {OLD['background']}]")

    # (b) every patch
    b_part = allp[allp["kind"] == "real"]["eta"]
    b_bg = allp[allp["kind"] == "background"]["eta"]
    print("\n=== (b) all patches ===")
    print(f"real particle-mask eta: {stats_line(b_part)}")
    print(f"background-only eta:    {stats_line(b_bg)}")
    print(f"without A_bg/B_bg:      real {allp[allp.kind=='real'].eta_uncorrected.mean():.4f}, "
          f"background {allp[allp.kind=='background'].eta_uncorrected.mean():.4f}")
    print(f"\n{'class':6s} {'real eta':>18s} {'bg eta':>18s} {'gap':>8s}")
    for cls in sorted(allp["cls"].unique()):
        r = allp[(allp.kind == "real") & (allp.cls == cls)]["eta"]
        b = allp[(allp.kind == "background") & (allp.cls == cls)]["eta"]
        print(f"{cls:6s} {r.mean():.4f} +/- {r.std(ddof=0):.4f} {b.mean():.4f} +/- {b.std(ddof=0):.4f} "
              f"{r.mean() - b.mean():>+8.4f}")

    verdict = {}
    for label, p, b in (("same_set", a_part, a_bg), ("all_patches", b_part, b_bg)):
        gap, floor = p.mean() - b.mean(), max(p.std(ddof=0), b.std(ddof=0))
        verdict[label] = dict(particle_mean=p.mean(), bg_mean=b.mean(), gap=gap, larger_std=floor,
                              passed=bool(b.mean() <= PASS_THRESHOLD), particle_exceeds=bool(gap > floor))
        print(f"\n[{label}] PASS (bg eta <= {PASS_THRESHOLD:.4f}): {verdict[label]['passed']} "
              f"(bg mean {b.mean():.4f}); particle - bg = {gap:+.4f} vs larger std {floor:.4f} -> "
              f"{'particle EXCEEDS background' if gap > floor else 'particle does NOT exceed background'}")

    # (c) drift
    drift = pd.read_csv(os.path.join(RESULTS_DIR, "recon_abbg_drift.csv"))
    dv = drift.groupby("source_file")[["drift_p_time", "drift_p_noise", "drift_eta_equiv_time"]].mean()
    excess = np.sqrt(np.clip(dv["drift_p_time"] ** 2 - dv["drift_p_noise"] ** 2, 0, None))
    dv["excess_drift_eta_equiv"] = np.arcsin(np.clip(excess, 0, 1)) / (np.pi / 2)
    print("\n=== (c) within-video drift of A_bg/B_bg (first vs second half, vs odd/even noise) ===")
    print(dv.round(4).to_string())
    print(f"median excess-drift eta-equivalent across videos: {dv['excess_drift_eta_equiv'].median():.4f} "
          f"(max {dv['excess_drift_eta_equiv'].max():.4f}) vs threshold {PASS_THRESHOLD:.4f}")

    with open(os.path.join(RESULTS_DIR, "biref_eta_check_summary.json"), "w") as f:
        json.dump(dict(verdict=verdict, pass_threshold=PASS_THRESHOLD,
                       drift_excess_eta_median=float(dv["excess_drift_eta_equiv"].median()),
                       drift_excess_eta_max=float(dv["excess_drift_eta_equiv"].max())), f, indent=2, default=float)


if __name__ == "__main__":
    main()
