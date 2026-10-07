#!/usr/bin/env bash
# Unattended training queue with the corrected (0/180 deg) augmentation.
# Waits for the reconstructed build to finish, writes patch counts, runs the
# biref eta check, then trains in priority order. Each variant: 5 seeds x
# (real video-holdout, background-only), epochs 150, patience 30, via
# run_real_bg_seeds.sh; every run writes its own results.json when it ends and
# completed runs are skipped on a restart. CAM is NOT run here.
cd "/c/Users/haany/OneDrive/research project/dataset 10jan23" || exit 1
export PYTHONUNBUFFERED=1
PO=pipeline_output
RAW=($PO/manifest.csv $PO/patches $PO/manifest_background.csv $PO/patches_background)
BGC=($PO/manifest_bgcorrected.csv $PO/patches_bgcorrected $PO/manifest_background_bgcorrected.csv $PO/patches_background_bgcorrected)
RS0=($PO/manifest_recon_s0.csv $PO/patches_recon_s0 $PO/manifest_background_recon_s0.csv $PO/patches_background_recon_s0)

build_alive() {
  powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'build_reconstructed_dataset' }) { exit 0 } else { exit 1 }"
}

echo "queue waiting for build $(date '+%m-%d %H:%M:%S')"
while build_alive; do sleep 30; done
if grep -q "=== Summary ===" build_reconstructed.log; then
  echo "build finished OK $(date '+%m-%d %H:%M:%S')"
  python scripts/recon_patch_counts.py > results/recon_patch_counts.txt 2>&1
  python scripts/biref_eta_check.py > results/biref_eta_check.log 2>&1
  echo "counts + eta check written (exit $?) $(date '+%m-%d %H:%M:%S')"
  HAVE_RECON=1
else
  echo "build did NOT finish cleanly - skipping recon_s0 $(date '+%m-%d %H:%M:%S')"
  HAVE_RECON=0
fi

run() { echo "=== START $2 ($1) $(date '+%m-%d %H:%M:%S')"; bash scripts/run_real_bg_seeds.sh "$@"; }

# (a)
[ "$HAVE_RECON" = 1 ] && run recon_s0 recons0_r180 "${RS0[@]}"
run intensity_bgcorr_vnorm intbv_r180 "${BGC[@]}"
# (b)
run pol4_bgcorr_vnorm pol4bgcorrvnorm_r180 "${BGC[@]}"
# (c)
run i0_bgcorr_vnorm i0bv_r180 "${BGC[@]}"
# (d)
run i45_bgcorr_vnorm i45bv_r180 "${BGC[@]}"
# (e) if time
run pol4 pol4_r180 "${RAW[@]}"
run i0i90_bgcorr_vnorm i0i90bv_r180 "${BGC[@]}"
run pol4_ff pol4ff_r180 "${RAW[@]}"
run pol4_bgcorr pol4bgcorr_r180 "${BGC[@]}"
run stokes stokes_r180 "${RAW[@]}"
echo "QUEUE COMPLETE $(date '+%m-%d %H:%M:%S')"
