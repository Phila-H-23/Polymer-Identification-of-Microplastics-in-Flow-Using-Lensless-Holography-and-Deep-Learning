"""R: background-only arms re-run with the video-holdout class weights the real
runs use (*_cwvh). Per arm: old vs new background-only per-class recall, real
recall (unchanged runs), and the new gap and noise floor per class; plus macro-F1."""
import statistics as stats
from build_report_variants import load_all, CLASSES

ARMS = [("intensity_bgcorr_vnorm", "intbv_c849"), ("z0", "recons0z0_c849"), ("recon_s0", "recons0_r180"),
        ("recon_biref", "reconbiref_c849"), ("pol4_bgcorr_vnorm", "pol4bgcorrvnorm_c849"),
        ("I45", "i45bv_c849"), ("Stokes", "stokes_c849")]


def side(runs):
    sd = lambda v: stats.pstdev(v) if len(v) > 1 else float("nan")
    rec = {c: [r["test_report"][c]["recall"] for r in runs] for c in CLASSES}
    f1 = [r["test_report"]["macro avg"]["f1-score"] for r in runs]
    return dict(n=len(runs), rec={c: (stats.mean(v), sd(v)) for c, v in rec.items()} if runs else {},
                f1=(stats.mean(f1), sd(f1)) if runs else (float("nan"), float("nan")))


for name, tag in ARMS:
    real, old, new = side(load_all(f"real_vh_{tag}")), side(load_all(f"bg_{tag}")), side(load_all(f"bg_{tag}_cwvh"))
    print(f"\n{'=' * 92}\n{name}  (real n={real['n']}, old bg n={old['n']}, new bg n={new['n']})\n{'=' * 92}")
    if new["n"] == 0:
        print("new background-only runs not finished yet")
        continue
    g, fl = real["f1"][0] - new["f1"][0], max(real["f1"][1], new["f1"][1])
    print(f"macro-F1: real {real['f1'][0]:.4f}+/-{real['f1'][1]:.4f} | old bg {old['f1'][0]:.4f}+/-{old['f1'][1]:.4f} "
          f"| new bg {new['f1'][0]:.4f}+/-{new['f1'][1]:.4f} | new gap {g:+.4f} floor {fl:.4f} "
          f"{'EXCEEDS' if abs(g) > fl else 'within'}")
    print(f"{'class':6s} {'real':>15s} {'old bg':>15s} {'new bg':>15s} {'new gap':>8s} {'floor':>6s} clears?")
    clears = []
    for c in CLASSES:
        r, o, n = real["rec"][c], old["rec"][c], new["rec"][c]
        gp, f = r[0] - n[0], max(r[1], n[1])
        clears += [c] if gp > f else []
        print(f"{c:6s} {r[0]:.3f}+/-{r[1]:.3f} {o[0]:.3f}+/-{o[1]:.3f} {n[0]:.3f}+/-{n[1]:.3f} {gp:>+8.3f} {f:>6.3f}  "
              f"{'YES' if gp > f else 'no'}")
    print(f"classes clearing the floor with the new background-only arm: {clears or 'none'}")
