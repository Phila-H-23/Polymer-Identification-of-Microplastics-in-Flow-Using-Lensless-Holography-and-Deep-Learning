#!/usr/bin/env bash
# Training queue v2: every non-reconstructed arm on the common 849 patches
# (*_common849.csv), corrected 0/180 deg augmentation, z=0 no-propagation control
# and recon_biref inserted after group (d). Each variant: 5 seeds x (real
# video-holdout, background-only), epochs 150, patience 30 (run_real_bg_seeds.sh);
# finished runs (results.json present) are skipped. CAM is NOT run here.
cd "/c/Users/haany/OneDrive/research project/dataset 10jan23" || exit 1
export PYTHONUNBUFFERED=1
PO=pipeline_output
RAW=($PO/manifest_common849.csv $PO/patches $PO/manifest_background_common849.csv $PO/patches_background)
BGC=($PO/manifest_bgcorrected_common849.csv $PO/patches_bgcorrected
     $PO/manifest_background_bgcorrected_common849.csv $PO/patches_background_bgcorrected)
RS0=($PO/manifest_recon_s0.csv $PO/patches_recon_s0 $PO/manifest_background_recon_s0.csv $PO/patches_background_recon_s0)
RZ0=($PO/manifest_recon_s0_z0.csv $PO/patches_recon_s0_z0 $PO/manifest_background_recon_s0_z0.csv $PO/patches_background_recon_s0_z0)
RPR=($PO/manifest_recon_prop.csv $PO/patches_recon_prop $PO/manifest_background_recon_prop.csv $PO/patches_background_recon_prop)

run() { echo "=== START $2 ($1) $(date '+%m-%d %H:%M:%S')"; bash scripts/run_real_bg_seeds.sh "$@"; }

echo "queue v2 started $(date '+%m-%d %H:%M:%S')"
# (a)
run recon_s0 recons0_r180 "${RS0[@]}"
run intensity_bgcorr_vnorm intbv_c849 "${BGC[@]}"
# (b)
run pol4_bgcorr_vnorm pol4bgcorrvnorm_c849 "${BGC[@]}"
# (c)
run i0_bgcorr_vnorm i0bv_c849 "${BGC[@]}"
# (d)
run i45_bgcorr_vnorm i45bv_c849 "${BGC[@]}"
# z = 0 no-propagation control: same rolling-median division, same patches, identity propagator
if ! grep -q "=== Summary ===" build_reconstructed_z0.log 2>/dev/null; then
  echo "=== BUILD z=0 control $(date '+%m-%d %H:%M:%S')"
  python -u scripts/build_reconstructed_dataset.py --z 0 --suffix _z0 --s0-only > build_reconstructed_z0.log 2>&1
  echo "z=0 build exit=$? $(date '+%m-%d %H:%M:%S')"
fi
if grep -q "=== Summary ===" build_reconstructed_z0.log; then
  run recon_s0 recons0z0_c849 "${RZ0[@]}"
else
  echo "z=0 build did NOT finish cleanly - skipping its training $(date '+%m-%d %H:%M:%S')"
fi
# recon_biref: eta/theta computed at load time from the saved propagated intensities + A_bg/B_bg
run recon_biref reconbiref_c849 "${RPR[@]}"
# (e)
run pol4 pol4_c849 "${RAW[@]}"
run i0i90_bgcorr_vnorm i0i90bv_c849 "${BGC[@]}"
run pol4_ff pol4ff_c849 "${RAW[@]}"
run pol4_bgcorr pol4bgcorr_c849 "${BGC[@]}"
run stokes stokes_c849 "${RAW[@]}"
echo "QUEUE COMPLETE $(date '+%m-%d %H:%M:%S')"
