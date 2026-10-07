#!/usr/bin/env bash
# Queue v5, order as requested after D (and P, already done):
#   S1 (running separately; waited for) -> S2 -> S3 -> E-revised -> R -> E pairs (optional)
# If S2 meets pre-registered reading (c), the queue stops after S2 (no rebuild, no training).
cd "/c/Users/haany/OneDrive/research project/dataset 10jan23" || exit 1
PO=pipeline_output
RF=results/recon_fidelity
CWVH=$PO/class_weights_video_holdout.json

alive() {  # $1 = command-line pattern of a python job
  powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }) { exit 0 } else { exit 1 }"
}
pet() {
  echo "=== START E $1 $(date '+%m-%d %H:%M:%S')"
  SEEDS="0 1 2" bash scripts/run_seeds.sh intensity_bgcorr_vnorm intbv_pet$1 \
    $PO/manifest_bgcorrected_common849_pet$1.csv $PO/patches_bgcorrected \
    $PO/manifest_background_bgcorrected_common849_pet$1.csv $PO/patches_background_bgcorrected \
    $PO/class_weights_pet$1.json
}
rbg() {  # channels tag bg_manifest bg_dir
  echo "=== START R $2 $(date '+%m-%d %H:%M:%S')"
  ARMS="bg" bash scripts/run_seeds.sh "$1" "$2_cwvh" - - "$3" "$4" "$CWVH"
}

echo "queue v5 waiting for S1 $(date '+%m-%d %H:%M:%S')"
while alive s1_detection_stats; do sleep 30; done
echo "S1 finished $(date '+%m-%d %H:%M:%S')"

echo "=== START S2 $(date '+%m-%d %H:%M:%S')"
python -u $RF/s2_registration_order.py > $RF/s2.log 2>&1
echo "S2 exit=$? $(date '+%m-%d %H:%M:%S')"
if grep -q "^(c).*: MET" $RF/s2_verdict.txt 2>/dev/null; then
  echo "S2 reading (c) MET - stopping queue for review $(date '+%m-%d %H:%M:%S')"
  exit 0
fi

echo "=== START S3 $(date '+%m-%d %H:%M:%S')"
python -u $RF/s3_scalar_baselines.py > $RF/s3.log 2>&1
echo "S3 exit=$? $(date '+%m-%d %H:%M:%S')"

for v in 1 3 4; do pet v$v; pet m$v; done
pet all3

BGB=($PO/manifest_background_bgcorrected_common849.csv $PO/patches_background_bgcorrected)
rbg intensity_bgcorr_vnorm intbv_c849 "${BGB[@]}"
rbg recon_s0 recons0z0_c849 $PO/manifest_background_recon_s0_z0.csv $PO/patches_background_recon_s0_z0
rbg recon_s0 recons0_r180 $PO/manifest_background_recon_s0.csv $PO/patches_background_recon_s0
rbg recon_biref reconbiref_c849 $PO/manifest_background_recon_prop.csv $PO/patches_background_recon_prop
rbg pol4_bgcorr_vnorm pol4bgcorrvnorm_c849 "${BGB[@]}"
rbg i45_bgcorr_vnorm i45bv_c849 "${BGB[@]}"
rbg stokes stokes_c849 $PO/manifest_background_common849.csv $PO/patches_background

for p in p13 p14 p34; do pet $p; done
echo "QUEUE COMPLETE $(date '+%m-%d %H:%M:%S')"
