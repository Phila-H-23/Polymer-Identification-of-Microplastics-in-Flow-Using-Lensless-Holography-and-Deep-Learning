#!/usr/bin/env bash
# Part B of the size-and-arms task: PET vs PS, real and particle-free, on four more input arms, in order
# recon_s0 (z=2.25 mm), recon_biref, pol4_bgcorr_vnorm, i45_bgcorr_vnorm. Same regime, split, --classes and class-weights
# files as the two-class test (scripts/run_twoclass.sh). Per arm: original direction seeds 0-4 with
# class_weights_twoclass_PETvsPS.json, then the reverse fold (2026-10-04 split) seeds 0-2 with _rev.json.
# Deadline guard: an arm starts only if its 16 runs x MIN_PER_RUN fit before 2026-10-08 12:00.
# Failed runs are logged and skipped; finished runs are skipped on a restart. Report refreshed after every arm.
# usage: MIN_PER_RUN=<minutes> bash scripts/run_size_arms.sh
cd "/c/Users/haany/OneDrive/research project/dataset 10jan23" || exit 1
export PYTHONUNBUFFERED=1
PO=pipeline_output
LOG=results/logs_size_arms
mkdir -p "$LOG"
DEADLINE=$(date -d "2026-10-08 12:00" +%s)
MIN_PER_RUN=${MIN_PER_RUN:-8}

fits() {  # number of runs
  local need=$(( $1 * MIN_PER_RUN * 60 ))
  [ $(( $(date +%s) + need )) -le "$DEADLINE" ]
}
arm_vars() {  # arm [rev]
  local sfx=${2:+_rev}
  CH=$1
  case $1 in
    recon_s0)    RM=$PO/manifest_recon_s0$sfx.csv; RD=$PO/patches_recon_s0
                 BM=$PO/manifest_background_recon_s0$sfx.csv; BD=$PO/patches_background_recon_s0 ;;
    recon_biref) RM=$PO/manifest_recon_prop$sfx.csv; RD=$PO/patches_recon_prop
                 BM=$PO/manifest_background_recon_prop$sfx.csv; BD=$PO/patches_background_recon_prop ;;
    *)           RM=$PO/manifest_bgcorrected_common849$sfx.csv; RD=$PO/patches_bgcorrected
                 BM=$PO/manifest_background_bgcorrected_common849$sfx.csv; BD=$PO/patches_background_bgcorrected ;;
  esac
}
train() {  # split manifest patch_dir out classes class_weights
  local split=$1 man=$2 pdir=$3 out=$4 cls=$5 cw=$6
  if [ -f "results/$out/results.json" ]; then echo "$out already done, skipped"; return; fi
  python train_compact_cnn.py --channels "$CH" --split "$split" --manifest "$man" --patch-dir "$pdir" \
    --class-weights "$cw" --classes "$cls" --epochs 150 --patience 30 --seed "$s" \
    --out "results/$out" > "$LOG/$out.log" 2>&1
  local rc=$?
  echo "$out exit=$rc $(date '+%m-%d %H:%M:%S')"
  [ "$rc" -ne 0 ] && echo "$out exit=$rc" >> "$LOG/failed.txt"
}

echo "size-and-arms queue started $(date '+%m-%d %H:%M:%S'), MIN_PER_RUN=$MIN_PER_RUN"
for arm in recon_s0 recon_biref pol4_bgcorr_vnorm i45_bgcorr_vnorm; do
  if ! fits 16; then echo "NOT RUN (deadline): $arm"; continue; fi
  echo "=== START $arm $(date '+%m-%d %H:%M:%S')"
  arm_vars $arm
  for s in 0 1 2 3 4; do
    train video_holdout "$RM" "$RD" "real_tc_PETvsPS_${arm}_s$s" "PET,PS" "$PO/class_weights_twoclass_PETvsPS.json"
    train particle "$BM" "$BD" "bg_tc_PETvsPS_${arm}_s$s" "PET,PS" "$PO/class_weights_twoclass_PETvsPS.json"
  done
  arm_vars $arm rev
  for s in 0 1 2; do
    train video_holdout "$RM" "$RD" "real_tc_PETvsPS_${arm}_rev_s$s" "PET,PS" "$PO/class_weights_twoclass_PETvsPS_rev.json"
    train particle "$BM" "$BD" "bg_tc_PETvsPS_${arm}_rev_s$s" "PET,PS" "$PO/class_weights_twoclass_PETvsPS_rev.json"
  done
  python build_report_size_and_arms.py > /dev/null 2>> "$LOG/report_errors.txt"
  echo "=== DONE $arm, report updated $(date '+%m-%d %H:%M:%S')"
done
echo "QUEUE COMPLETE $(date '+%m-%d %H:%M:%S')"
