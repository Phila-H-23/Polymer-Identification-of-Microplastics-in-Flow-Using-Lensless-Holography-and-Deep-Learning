#!/usr/bin/env bash
# Overnight queue v6. Part 1 first, then Part 2. New run folders only.
#  1a  background-only arms R did not cover, video-holdout weights (the file the real
#      runs used), 5 seeds, *_cwvh
#  1b  item-P evaluation on all *_cwvh background-only checkpoints + stokes_bgcorr_vnorm real
#  2   reverse fold: 4 arms x 3 seeds x (real, background-only), *_rev, weights from the
#      reverse-fold training counts (pipeline_output/class_weights_rev.json) for both arms
cd "/c/Users/haany/OneDrive/research project/dataset 10jan23" || exit 1
PO=pipeline_output
CWVH=$PO/class_weights_video_holdout.json
BGB=($PO/manifest_background_bgcorrected_common849.csv $PO/patches_background_bgcorrected)
RAWB=($PO/manifest_background_common849.csv $PO/patches_background)

rbg() {  # channels tag bg_manifest bg_dir
  echo "=== START 1a $2 $(date '+%m-%d %H:%M:%S')"
  ARMS="bg" bash scripts/run_seeds.sh "$1" "$2_cwvh" - - "$3" "$4" "$CWVH"
}
rev() {  # channels tag real_manifest real_dir bg_manifest bg_dir
  echo "=== START 2 $2 $(date '+%m-%d %H:%M:%S')"
  SEEDS="0 1 2" bash scripts/run_seeds.sh "$1" "$2" "$3" "$4" "$5" "$6" $PO/class_weights_rev.json
}

echo "queue v6 started $(date '+%m-%d %H:%M:%S')"
rbg stokes_bgcorr_vnorm stokesbv_c849 "${BGB[@]}"
rbg pol4_bgcorr pol4bgcorr_c849 "${BGB[@]}"
rbg pol4 pol4_c849 "${RAWB[@]}"
rbg pol4_ff pol4ff_c849 "${RAWB[@]}"
rbg i0_bgcorr_vnorm i0bv_c849 "${BGB[@]}"
rbg i0i90_bgcorr_vnorm i0i90bv_c849 "${BGB[@]}"
python build_report_R2.py > results/report_R2_reweighted_bg.txt 2>&1
echo "R2 report exit=$? $(date '+%m-%d %H:%M:%S')"

echo "=== START 1b $(date '+%m-%d %H:%M:%S')"
python -u scripts/p_eval_cwvh.py > results/p_eval_cwvh.log 2>&1
echo "1b exit=$? $(date '+%m-%d %H:%M:%S')"

BGCR=($PO/manifest_bgcorrected_common849_rev.csv $PO/patches_bgcorrected
      $PO/manifest_background_bgcorrected_common849_rev.csv $PO/patches_background_bgcorrected)
rev intensity_bgcorr_vnorm intbv_rev "${BGCR[@]}"
rev pol4_bgcorr_vnorm pol4bgcorrvnorm_rev "${BGCR[@]}"
rev recon_s0 recons0_rev $PO/manifest_recon_s0_rev.csv $PO/patches_recon_s0 \
    $PO/manifest_background_recon_s0_rev.csv $PO/patches_background_recon_s0
rev pol4_bgcorr pol4bgcorr_rev "${BGCR[@]}"
python build_report_reverse_fold.py > results/report_reverse_fold.log 2>&1
echo "reverse-fold report exit=$? $(date '+%m-%d %H:%M:%S')"
echo "QUEUE COMPLETE $(date '+%m-%d %H:%M:%S')"
