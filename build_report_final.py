"""Final report for the session-confound control task.

Two separate, differently-designed checks - kept in separate tables since
they use different splits and answer different questions:

TABLE A - video-holdout background control (matches the split used by
--split video_holdout on the real data): background-only pol4/intensity,
vs real pol4/intensity with and without --patch-norm (all under the
particle/default split for the real-data rows, per the original task
wording - "real pol4" etc reuses the v3 grid's particle-split runs).

TABLE B - particle-identity-matched session-leakage check: background
patches split with a plain stratified 70/15/15 by class (no video
logic at all - pipeline_output/manifest_background_particleident.csv),
sized to match the real particle-identity split (618/133/132) as closely
as possible (got 618/132/133), compared directly against the real
pol4/intensity numbers from the SAME particle-identity split
(results/compact_particle_pol4_s0-2, results/compact_particle_intensity_s0-2
in summary_v3.csv) - the directly comparable pair, since both sides use
the same split methodology.
"""
import json
import statistics as stats
from pathlib import Path

CLASSES = ["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"]
CHANCE = 1 / len(CLASSES)  # ~0.1429, matches the ~0.14 given in the task


def load(out_dir):
    p = Path(out_dir) / "results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    tr = d["test_report"]
    return {
        "test_macro_f1": tr["macro avg"]["f1-score"],
        "test_accuracy": tr["accuracy"],
        "recall": {c: tr.get(c, {}).get("recall", float("nan")) for c in CLASSES},
    }


def group_stats(out_dirs):
    runs = [load(d) for d in out_dirs]
    missing = [d for d, r in zip(out_dirs, runs) if r is None]
    runs = [r for r in runs if r is not None]
    f1s = [r["test_macro_f1"] for r in runs]
    mean_f1 = stats.mean(f1s) if f1s else float("nan")
    std_f1 = stats.pstdev(f1s) if len(f1s) > 1 else 0.0
    recall_means = {c: stats.mean([r["recall"][c] for r in runs]) for c in CLASSES} if runs else {}
    return dict(n=len(runs), missing=missing, mean_f1=mean_f1, std_f1=std_f1,
                f1s=f1s, recall_means=recall_means)


def print_table(title, rows):
    print(f"\n=== {title} ===")
    header = f"{'row':32s} {'n':3s} {'mean_test_f1':>13s} {'std':>7s}  " + "  ".join(f"{c:>6s}" for c in CLASSES)
    print(header)
    for name, out_dirs in rows:
        g = group_stats(out_dirs)
        if g["missing"]:
            print(f"{name:32s}  INCOMPLETE - missing: {g['missing']}")
            continue
        recall_str = "  ".join(f"{g['recall_means'][c]:.3f}" for c in CLASSES)
        print(f"{name:32s} {g['n']:<3d} {g['mean_f1']:>13.4f} {g['std_f1']:>7.4f}  {recall_str}")
    return {name: group_stats(out_dirs) for name, out_dirs in rows}


def main():
    table_a_rows = [
        ("background-only pol4 (video_holdout split)", [f"results/bg_pol4_s{s}" for s in range(5)]),
        ("background-only intensity (video_holdout split)", [f"results/bg_intensity_s{s}" for s in range(5)]),
        ("real pol4 (video_holdout split)", [f"results/real_vh_pol4_s{s}" for s in range(5)]),
        ("real pol4 + patch-norm (video_holdout split)", [f"results/real_vh_pol4_patchnorm_s{s}" for s in range(5)]),
        ("real intensity (video_holdout split)", [f"results/real_vh_intensity_s{s}" for s in range(5)]),
        ("real intensity + patch-norm (video_holdout split)", [f"results/real_vh_intensity_patchnorm_s{s}" for s in range(5)]),
    ]
    a = print_table("TABLE A - session-confound control (video-holdout background split)", table_a_rows)

    table_b_rows = [
        ("background pol4 (particle-identity split)", [f"results/bg_particleident_pol4_s{s}" for s in range(3)]),
        ("background intensity (particle-identity split)", [f"results/bg_particleident_intensity_s{s}" for s in range(3)]),
        ("real pol4 (particle-identity split)", [f"results/compact_particle_pol4_s{s}" for s in range(3)]),
        ("real intensity (particle-identity split)", [f"results/compact_particle_intensity_s{s}" for s in range(3)]),
    ]
    b = print_table("TABLE B - particle-identity-matched session-leakage check", table_b_rows)

    print("\n=== Plain answers ===")
    if not a["background-only pol4 (video_holdout split)"]["missing"] and not a["background-only intensity (video_holdout split)"]["missing"]:
        bg_pol4 = a["background-only pol4 (video_holdout split)"]["mean_f1"]
        bg_int = a["background-only intensity (video_holdout split)"]["mean_f1"]
        print(f"(a) Background-only, video_holdout split: pol4 mean={bg_pol4:.4f} "
              f"({bg_pol4/CHANCE:.2f}x chance {CHANCE:.4f}), intensity mean={bg_int:.4f} "
              f"({bg_int/CHANCE:.2f}x chance)")

    if not b["background pol4 (particle-identity split)"]["missing"] and not b["background intensity (particle-identity split)"]["missing"]:
        bgp_pol4 = b["background pol4 (particle-identity split)"]["mean_f1"]
        bgp_int = b["background intensity (particle-identity split)"]["mean_f1"]
        print(f"    Background-only, particle-identity split: pol4 mean={bgp_pol4:.4f} "
              f"({bgp_pol4/CHANCE:.2f}x chance), intensity mean={bgp_int:.4f} ({bgp_int/CHANCE:.2f}x chance)")

    # Correctly-matched comparison: background-only vs real, BOTH under video_holdout.
    real_vh_pol4 = a["real pol4 (video_holdout split)"]["mean_f1"]
    real_vh_int = a["real intensity (video_holdout split)"]["mean_f1"]
    if not any(x != x for x in [bg_pol4, bg_int, real_vh_pol4, real_vh_int]):
        print(f"(a2) Correctly-matched (both video_holdout, n=5): background-only pol4 {bg_pol4:.4f} "
              f"recovers {bg_pol4/real_vh_pol4*100:.1f}% of real pol4's {real_vh_pol4:.4f}; "
              f"background-only intensity {bg_int:.4f} recovers {bg_int/real_vh_int*100:.1f}% "
              f"of real intensity's {real_vh_int:.4f}")

    # patch-norm cost: now directly comparable, all four rows on video_holdout.
    real_pol4_pn = a["real pol4 + patch-norm (video_holdout split)"]["mean_f1"]
    real_int_pn = a["real intensity + patch-norm (video_holdout split)"]["mean_f1"]
    if not any(x != x for x in [real_vh_pol4, real_pol4_pn, real_vh_int, real_int_pn]):
        print(f"(b) --patch-norm cost (video_holdout split, n=5 both sides): "
              f"pol4 {real_vh_pol4:.4f} -> {real_pol4_pn:.4f} "
              f"(delta={real_pol4_pn-real_vh_pol4:+.4f}); "
              f"intensity {real_vh_int:.4f} -> {real_int_pn:.4f} "
              f"(delta={real_int_pn-real_vh_int:+.4f})")


if __name__ == "__main__":
    main()
