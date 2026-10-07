"""Staged channel ablation, Rung 1: intensity vs I0 (single channel),
both on bgcorr+vnorm-corrected data, compact CNN, split_video_holdout,
n=5 seeds. Same report format as the closed raw-input investigation for
direct comparability.
"""
import json
import statistics as stats
from pathlib import Path

CLASSES = ["PE", "PET", "PMMA", "PP", "PS", "PVA", "PVC"]
DUSTY = ["PE", "PMMA", "PVC"]
CLEAN = ["PET", "PP", "PS", "PVA"]


def load(out_dir):
    p = Path(out_dir) / "results.json"
    if not p.exists():
        return None
    return json.load(open(p))


def group_stats(out_dirs):
    ds = [load(d) for d in out_dirs]
    missing = [d for d, x in zip(out_dirs, ds) if x is None]
    ds = [x for x in ds if x is not None]
    f1s = [d["test_report"]["macro avg"]["f1-score"] for d in ds]
    mean_f1 = stats.mean(f1s) if f1s else float("nan")
    std_f1 = stats.pstdev(f1s) if len(f1s) > 1 else 0.0
    recalls = {c: [d["test_report"].get(c, {}).get("recall", float("nan")) for d in ds] for c in CLASSES}
    mean_recalls = {c: stats.mean(recalls[c]) for c in CLASSES}
    return dict(n=len(ds), missing=missing, mean_f1=mean_f1, std_f1=std_f1, f1s=f1s, mean_recalls=mean_recalls)


def report_variant(name, real_prefix, bg_prefix):
    real = group_stats([f"results/{real_prefix}_s{s}" for s in range(5)])
    bg = group_stats([f"results/{bg_prefix}_s{s}" for s in range(5)])

    print(f"\n{'='*70}\n{name}\n{'='*70}")
    for side, g in [("real", real), ("background-only", bg)]:
        if g["missing"]:
            print(f"{side:20s} INCOMPLETE - missing: {g['missing']}")
        else:
            print(f"{side:20s} n={g['n']} mean={g['mean_f1']:.4f} std={g['std_f1']:.4f}  {[round(v,4) for v in g['f1s']]}")

    if real["missing"] or bg["missing"]:
        return None

    gap = real["mean_f1"] - bg["mean_f1"]
    larger_std = max(real["std_f1"], bg["std_f1"])
    exceeds = abs(gap) > larger_std
    print(f"\ngap (real - background-only) = {gap:+.4f}, larger std = {larger_std:.4f}  "
          f"[{'EXCEEDS larger std -> gap likely real' if exceeds else 'within larger std -> noise'}]")

    print(f"\n{'class':6s} {'real':>8s} {'background':>11s} {'gap':>8s}")
    for c in CLASSES:
        r, b = real["mean_recalls"][c], bg["mean_recalls"][c]
        print(f"{c:6s} {r:>8.4f} {b:>11.4f} {r-b:>+8.4f}")

    dusty_gap = stats.mean([real["mean_recalls"][c] - bg["mean_recalls"][c] for c in DUSTY])
    clean_gap = stats.mean([real["mean_recalls"][c] - bg["mean_recalls"][c] for c in CLEAN])
    print(f"\nMean gap, PE/PMMA/PVC (dust-affected): {dusty_gap:+.4f}")
    print(f"Mean gap, PET/PP/PS/PVA (clean):        {clean_gap:+.4f}")

    if exceeds and abs(dusty_gap) > 2 * abs(clean_gap) + 0.02:
        verdict = "CONCENTRATED in PE/PMMA/PVC -> same confound as before, not new signal"
    elif exceeds:
        verdict = "gap exceeds noise AND roughly uniform across classes -> PROMISING, genuinely new"
    else:
        verdict = "gap does not exceed noise -> null/ambiguous"
    print(f"\nVerdict: {verdict}")

    return dict(gap=gap, exceeds=exceeds, dusty_gap=dusty_gap, clean_gap=clean_gap, verdict=verdict,
                real_mean=real["mean_f1"], bg_mean=bg["mean_f1"])


def main():
    results = {}
    results["intensity"] = report_variant("INTENSITY (bgcorr+vnorm)", "real_vh_intbv", "bg_intbv")
    results["i0"] = report_variant("I0 ONLY (bgcorr+vnorm)", "real_vh_i0bv", "bg_i0bv")
    results["i0i90"] = report_variant("I0+I90 (bgcorr+vnorm) - RUNG 2 CONFIRMATION", "real_vh_i0i90bv", "bg_i0i90bv")

    print(f"\n{'='*70}\nRUNG 1 SUMMARY / ESCALATION DECISION\n{'='*70}")
    for name, r in results.items():
        if r is None:
            print(f"{name}: INCOMPLETE")
            continue
        print(f"{name}: real={r['real_mean']:.4f} bg={r['bg_mean']:.4f} gap={r['gap']:+.4f} -> {r['verdict']}")

    if all(r is not None for r in results.values()):
        any_promising = any(r["exceeds"] and "PROMISING" in r["verdict"] for r in results.values())
        if any_promising:
            print("\n>>> At least one rung-1 variant is promising (gap exceeds noise, roughly uniform). "
                  "Per the escalation rule: proceed to rung 2 (I0+I90) as confirmation, then stop.")
        else:
            print("\n>>> Rung 1 is null/ambiguous or confound-concentrated for both variants. "
                  "Per the escalation rule: proceed to rung 2 (I0+I90) as confirmation, then stop regardless of outcome.")


if __name__ == "__main__":
    main()
