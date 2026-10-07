#!/usr/bin/env bash
# Particle-identity split comparison: 7 arms x 5 seeds x (real, particle-free) = 70 runs.
# Regime copied from the corrected queue (run_queue_r180.sh -> run_real_bg_seeds.sh): epochs 150,
# patience 30, all other hyperparameters at train_compact_cnn.py defaults; rotations 0/180 are the
# current code. Both classifiers: --split particle (column 'split' of the *_psplit manifests) and
# --class-weights pipeline_output/class_weights_psplit.json. Folders results/{real,bg}_<tag>_s<seed>_psplit.
# A failed run is logged to results/logs_psplit/failed.txt and the queue continues. The report is
# regenerated after each arm. Finished runs are skipped on a restart.
cd "/c/Users/haany/OneDrive/research project/dataset 10jan23" || exit 1
export PYTHONUNBUFFERED=1
PO=pipeline_output
CW=$PO/class_weights_psplit.json
LOG=results/logs_psplit
mkdir -p "$LOG"
START=$(date +%s)

train() {  # channels manifest patch_dir out
  if [ -f "results/$4/results.json" ]; then echo "$4 already done, skipped"; return; fi
  python train_compact_cnn.py --channels "$1" --split particle --manifest "$2" --patch-dir "$3" \
    --class-weights "$CW" --epochs 150 --patience 30 --seed "$s" --out "results/$4" > "$LOG/$4.log" 2>&1
  rc=$?
  echo "$4 exit=$rc $(date '+%m-%d %H:%M:%S')"
  [ "$rc" -ne 0 ] && echo "$4 exit=$rc" >> "$LOG/failed.txt"
}
arm() {  # channels tag real_manifest real_dir bg_manifest bg_dir
  echo "=== START $2 ($1) $(date '+%m-%d %H:%M:%S')"
  for s in 0 1 2 3 4; do
    train "$1" "$3" "$4" "real_$2_s${s}_psplit"
    train "$1" "$5" "$6" "bg_$2_s${s}_psplit"
  done
  python build_report_particle_split.py > /dev/null 2>> "$LOG/report_errors.txt"
  echo "=== DONE $2, report updated $(date '+%m-%d %H:%M:%S')"
}

BGC=($PO/manifest_bgcorrected_common849_psplit.csv $PO/patches_bgcorrected
     $PO/manifest_background_bgcorrected_common849_psplit.csv $PO/patches_background_bgcorrected)
RAW=($PO/manifest_common849_psplit.csv $PO/patches $PO/manifest_background_common849_psplit.csv $PO/patches_background)
arm intensity_bgcorr_vnorm intbv "${BGC[@]}"
arm pol4_bgcorr_vnorm pol4bgcorrvnorm "${BGC[@]}"
arm pol4_bgcorr pol4bgcorr "${BGC[@]}"
arm pol4 pol4 "${RAW[@]}"
arm recon_s0 recons0z0 $PO/manifest_recon_s0_z0_psplit.csv $PO/patches_recon_s0_z0 \
    $PO/manifest_background_recon_s0_z0_psplit.csv $PO/patches_background_recon_s0_z0
arm recon_s0 recons0 $PO/manifest_recon_s0_psplit.csv $PO/patches_recon_s0 \
    $PO/manifest_background_recon_s0_psplit.csv $PO/patches_background_recon_s0
arm recon_biref reconbiref $PO/manifest_recon_prop_psplit.csv $PO/patches_recon_prop \
    $PO/manifest_background_recon_prop_psplit.csv $PO/patches_background_recon_prop
echo "QUEUE COMPLETE $(date '+%m-%d %H:%M:%S'), total $(( ($(date +%s) - START) / 60 )) min"
