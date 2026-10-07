"""E: PET session-count test (intensity_bgcorr_vnorm, common 849, 3 seeds).
Per config: PET recall and precision (real), PET recall (background-only), gap
real - background-only recall, noise floor (larger std), and macro-F1."""
import statistics as stats
from build_report_variants import load_all

CONFIGS = [("v1", "PET1 only", 36), ("m1", "matched to PET1, 3 videos", 36),
           ("v3", "PET3 only", 72), ("m3", "matched to PET3, 3 videos", 72),
           ("v4", "PET4 only", 55), ("m4", "matched to PET4, 3 videos", 55),
           ("all3", "all 3 videos", 163),
           ("p13", "PET1+PET3", 108), ("p14", "PET1+PET4", 91), ("p34", "PET3+PET4", 127)]


def ms(v):
    return (stats.mean(v), stats.pstdev(v)) if len(v) > 1 else ((v[0], float("nan")) if v else (float("nan"),) * 2)


print(f"{'config':28s} {'PET tr':>6s} {'n':>5s} {'real PET recall':>16s} {'real PET prec':>15s} "
      f"{'bg PET recall':>15s} {'gap':>7s} {'floor':>6s} {'clears':>6s} {'real mF1':>9s} {'bg mF1':>7s}")
for cfg, label, ntr in CONFIGS:
    real, bg = load_all(f"real_vh_intbv_pet{cfg}"), load_all(f"bg_intbv_pet{cfg}")
    if not real and not bg:
        continue
    rr = ms([r["test_report"]["PET"]["recall"] for r in real])
    rp = ms([r["test_report"]["PET"]["precision"] for r in real])
    br = ms([r["test_report"]["PET"]["recall"] for r in bg])
    rf = ms([r["test_report"]["macro avg"]["f1-score"] for r in real])
    bf = ms([r["test_report"]["macro avg"]["f1-score"] for r in bg])
    gap, floor = rr[0] - br[0], max(rr[1], br[1])
    print(f"{label:28s} {ntr:>6d} {len(real)}/{len(bg):<3d} {rr[0]:.3f}+/-{rr[1]:.3f}  {rp[0]:.3f}+/-{rp[1]:.3f}  "
          f"{br[0]:.3f}+/-{br[1]:.3f} {gap:>+7.3f} {floor:>6.3f} {'YES' if gap > floor else 'no':>6s} "
          f"{rf[0]:>9.3f} {bf[0]:>7.3f}")
