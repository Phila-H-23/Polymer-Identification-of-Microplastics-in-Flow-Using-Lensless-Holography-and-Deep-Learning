#!/usr/bin/env bash
# Surround-augmentation runs (60): {z0, intensity_bgcorr_vnorm} x {BG, BG+TEX, INTERIOR} x 5 seeds x
# (real, particle-free). Regime of the corrected queue (run_queue_r180.sh -> run_real_bg_seeds.sh): epochs 150,
# patience 30, other hyperparameters at train_compact_cnn.py defaults; video-holdout split; both classifiers use
# pipeline_output/class_weights_video_holdout.json. Donors: the arm's own background manifest (training rows).
# BG / BG+TEX: training patches only. INTERIOR: input representation on train, val and test (both classifiers).
# Order (user's change 2): let the particle-split queue finish its FIRST arm (intensity_bgcorr_vnorm, 10 runs), stop
# it there (its runner is killed once the arm's last run has started; that run completes), keep its results, refresh
# its report; then the 60 augmentation runs; then the remaining particle-split arms (scripts/run_psplit_rest.sh).
# Before the first augmentation run, with nothing training, the tested train_compact_cnn.py (INTERIOR on every split)
# is copied in. Failed runs are logged and skipped.
cd "/c/Users/haany/OneDrive/research project/dataset 10jan23" || exit 1
export PYTHONUNBUFFERED=1
PO=pipeline_output
CW=$PO/class_weights_video_holdout.json
LOG=results/logs_aug
STAGED="C:/Users/haany/AppData/Local/Temp/claude/c--Users-haany-OneDrive-research-project-dataset-10jan23/075ab033-0168-4afa-a675-c38e4f73b481/scratchpad/train_compact_cnn_aug.py"
mkdir -p "$LOG"

alive() {  # process name, command-line pattern
  powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='$1'\" | Where-Object { \$_.CommandLine -match '$2' }) { exit 0 } else { exit 1 }"
}

echo "aug queue started $(date '+%m-%d %H:%M:%S')"
if alive bash.exe run_psplit.sh; then
  echo "particle-split queue running: waiting for the last run of its first arm (bg_intbv_s4_psplit) to start"
  until alive python.exe bg_intbv_s4_psplit || [ -f results/bg_intbv_s4_psplit/results.json ] || ! alive bash.exe run_psplit.sh; do
    sleep 30
  done
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -match 'run_psplit.sh' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
  echo "particle-split runner stopped after its first arm started its last run $(date '+%m-%d %H:%M:%S')"
  while alive python.exe _psplit; do sleep 30; done
  for d in results/*_pol4bgcorrvnorm_s*_psplit results/*_pol4bgcorr_s*_psplit results/*_pol4_s*_psplit; do
    [ -e "$d" ] && echo "WARNING: unexpected particle-split folder $d"
  done
  python build_report_particle_split.py > /dev/null 2>> "$LOG/psplit_report_errors.txt"
  echo "particle-split report refreshed $(date '+%m-%d %H:%M:%S')"
fi
while alive python.exe train_compact_cnn; do echo "another training job is running - waiting"; sleep 60; done
cp "$STAGED" train_compact_cnn.py && cmp -s "$STAGED" train_compact_cnn.py \
  && echo "train_compact_cnn.py updated (INTERIOR on every split) $(date '+%m-%d %H:%M:%S')" \
  || { echo "ERROR: could not install the updated train_compact_cnn.py - stopping"; exit 1; }
echo "existing unit test: $(python scripts/test_augmentation_channels.py 2>&1 | tail -n 1)"
START=$(date +%s)

train() {  # split manifest patch_dir out channels aug donor_manifest donor_dir
  if [ -f "results/$4/results.json" ]; then echo "$4 already done, skipped"; return; fi
  python train_compact_cnn.py --channels "$5" --split "$1" --manifest "$2" --patch-dir "$3" --class-weights "$CW" \
    --aug "$6" --donor-manifest "$7" --donor-patch-dir "$8" --epochs 150 --patience 30 --seed "$s" \
    --out "results/$4" > "$LOG/$4.log" 2>&1
  rc=$?
  echo "$4 exit=$rc $(date '+%m-%d %H:%M:%S')"
  [ "$rc" -ne 0 ] && echo "$4 exit=$rc" >> "$LOG/failed.txt"
}
cfg() {  # channels tag aug real_manifest real_dir bg_manifest bg_dir
  echo "=== START $2 $3 $(date '+%m-%d %H:%M:%S')"
  for s in 0 1 2 3 4; do
    train video_holdout "$4" "$5" "real_vh_$2_s${s}_$3" "$1" "$3" "$6" "$7"
    train particle "$6" "$7" "bg_$2_s${s}_$3" "$1" "$3" "$6" "$7"
  done
  python build_report_aug.py > /dev/null 2>> "$LOG/report_errors.txt"
  echo "=== DONE $2 $3, report updated $(date '+%m-%d %H:%M:%S')"
}
Z0=(recon_s0 recons0z0_c849)
Z0M=($PO/manifest_recon_s0_z0.csv $PO/patches_recon_s0_z0 $PO/manifest_background_recon_s0_z0.csv $PO/patches_background_recon_s0_z0)
IN=(intensity_bgcorr_vnorm intbv_c849)
INM=($PO/manifest_bgcorrected_common849.csv $PO/patches_bgcorrected $PO/manifest_background_bgcorrected_common849.csv $PO/patches_background_bgcorrected)
cfg "${Z0[@]}" bg "${Z0M[@]}"
cfg "${IN[@]}" bg "${INM[@]}"
cfg "${Z0[@]}" bgtex "${Z0M[@]}"
cfg "${IN[@]}" bgtex "${INM[@]}"
cfg "${Z0[@]}" interior "${Z0M[@]}"
cfg "${IN[@]}" interior "${INM[@]}"
echo "AUGMENTATION RUNS COMPLETE $(date '+%m-%d %H:%M:%S'), training time $(( ($(date +%s) - START) / 60 )) min"
echo "=== re-queued particle-split arms (pol4_bgcorr_vnorm, pol4_bgcorr, pol4) $(date '+%m-%d %H:%M:%S')"
bash scripts/run_psplit_rest.sh >> results/queue_psplit.log 2>&1
echo "QUEUE COMPLETE $(date '+%m-%d %H:%M:%S')"
