"""R2: background-only arms R did not cover, re-run with the video-holdout class
weights the real runs use (*_cwvh). Same format as build_report_reweighted_bg.py."""
import statistics as stats
from build_report_variants import load_all, CLASSES

ARMS = [("stokes_bgcorr_vnorm", "stokesbv_c849"), ("pol4_bgcorr", "pol4bgcorr_c849"), ("pol4", "pol4_c849"),
        ("pol4_ff", "pol4ff_c849"), ("I0", "i0bv_c849"), ("I0+I90", "i0i90bv_c849")]


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
