"""Part 2: reverse-fold test. Per arm and class: real vs background-only recall,
gap, floor (larger of the two seed stds), clears; PET precision/recall/F1; macro-F1;
test patches per class. Original-split numbers next to the reverse ones: original
real runs (5 seeds) vs the matched-weight background-only runs (*_cwvh, 5 seeds), so
both directions compare arms trained with identical class weights."""
import statistics as stats
from build_report_variants import load_all, CLASSES

ARMS = [("intensity_bgcorr_vnorm", "intbv_c849", "intbv_rev"),
        ("pol4_bgcorr_vnorm", "pol4bgcorrvnorm_c849", "pol4bgcorrvnorm_rev"),
        ("recon_s0", "recons0_r180", "recons0_rev"),
        ("pol4_bgcorr", "pol4bgcorr_c849", "pol4bgcorr_rev")]


def ms(v):
    return (stats.mean(v), stats.pstdev(v)) if len(v) > 1 else ((v[0], float("nan")) if v else (float("nan"),) * 2)


def side(runs):
    g = lambda c, k: ms([r["test_report"][c][k] for r in runs])
    return dict(n=len(runs), rec={c: g(c, "recall") for c in CLASSES},
                pet={k: g("PET", k) for k in ("precision", "recall", "f1-score")},
                f1=ms([r["test_report"]["macro avg"]["f1-score"] for r in runs]),
                support={c: int(runs[0]["test_report"][c]["support"]) for c in CLASSES} if runs else {})


def compare(real, bg):
    out = {}
    for c in CLASSES:
        gap, floor = real["rec"][c][0] - bg["rec"][c][0], max(real["rec"][c][1], bg["rec"][c][1])
        out[c] = (gap, floor, gap > floor)
    return out


def main():
    lines, clears_rev, bg_f1_rev = [], {}, {}
    for name, orig, rev in ARMS:
        o_r, o_b = side(load_all(f"real_vh_{orig}")), side(load_all(f"bg_{orig}_cwvh"))
        r_r, r_b = side(load_all(f"real_vh_{rev}")), side(load_all(f"bg_{rev}"))
        if r_r["n"] == 0 or r_b["n"] == 0:
            lines.append(f"\n{name}: reverse-fold runs missing (real n={r_r['n']}, bg n={r_b['n']})")
            continue
        co, cr = compare(o_r, o_b), compare(r_r, r_b)
        clears_rev[name] = [c for c in CLASSES if cr[c][2]]
        bg_f1_rev[name] = r_b["f1"][0]
        f = lambda t: f"{t[0]:.3f}+/-{t[1]:.3f}"
        lines += ["", "=" * 118,
                  f"{name}   original: real n={o_r['n']}, bg (cwvh) n={o_b['n']} | reverse: real n={r_r['n']}, bg n={r_b['n']}",
                  "=" * 118,
                  f"macro-F1  original: real {f(o_r['f1'])}  bg {f(o_b['f1'])}  |  reverse: real {f(r_r['f1'])}  bg {f(r_b['f1'])}",
                  f"PET P/R/F1 original: real {' / '.join(f(o_r['pet'][k]) for k in ('precision', 'recall', 'f1-score'))}"
                  f"  bg {' / '.join(f(o_b['pet'][k]) for k in ('precision', 'recall', 'f1-score'))}",
                  f"PET P/R/F1 reverse:  real {' / '.join(f(r_r['pet'][k]) for k in ('precision', 'recall', 'f1-score'))}"
                  f"  bg {' / '.join(f(r_b['pet'][k]) for k in ('precision', 'recall', 'f1-score'))}",
                  f"{'':6s} {'ORIGINAL SPLIT':^58s} | {'REVERSE FOLD':^58s}",
                  f"{'class':6s} {'n':>4s} {'real':>13s} {'bg':>13s} {'gap':>7s} {'floor':>6s} {'clr':>4s} | "
                  f"{'n':>4s} {'real':>13s} {'bg':>13s} {'gap':>7s} {'floor':>6s} {'clr':>4s}"]
        for c in CLASSES:
            lines.append(f"{c:6s} {o_r['support'][c]:>4d} {f(o_r['rec'][c]):>13s} {f(o_b['rec'][c]):>13s} "
                         f"{co[c][0]:>+7.3f} {co[c][1]:>6.3f} {'YES' if co[c][2] else 'no':>4s} | "
                         f"{r_r['support'][c]:>4d} {f(r_r['rec'][c]):>13s} {f(r_b['rec'][c]):>13s} "
                         f"{cr[c][0]:>+7.3f} {cr[c][1]:>6.3f} {'YES' if cr[c][2] else 'no':>4s}")
        lines.append(f"classes clearing the floor: original {[c for c in CLASSES if co[c][2]] or 'none'} | "
                     f"reverse {clears_rev[name] or 'none'}")

    lines += ["", "Pre-registered readings (reverse fold):"]
    if len(clears_rev) == len(ARMS):
        n_pet = sum("PET" in v for v in clears_rev.values())
        others = {c: sum(c in v for v in clears_rev.values()) for c in CLASSES if c != "PET"}
        r1 = n_pet >= 3 and all(n < 2 for n in others.values())
        r2 = bg_f1_rev.get("pol4_bgcorr", float("nan")) >= 0.35
        lines.append(f"(R1) PET clears in >= 3 of 4 arms and no other class clears in >= 2 arms: "
                     f"{'APPLIES' if r1 else 'does not apply'} (PET clears in {n_pet}; other classes "
                     f"{ {c: n for c, n in others.items() if n} or 'none'})")
        lines.append(f"(R2) pol4_bgcorr background-only macro-F1 >= 0.35 in the reverse direction: "
                     f"{'APPLIES' if r2 else 'does not apply'} ({bg_f1_rev.get('pol4_bgcorr', float('nan')):.3f})")
        if not (r1 or r2):
            lines.append("Neither applies: reported as is.")
    else:
        lines.append("not evaluated: reverse-fold runs incomplete")
    text = "\n".join(lines)
    open("results/report_reverse_fold.txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
