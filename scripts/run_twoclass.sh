#!/usr/bin/env bash
# Two-class test: PET vs one other polymer, real and particle-free, video-holdout split, corrected regime
# (epochs 150, patience 30, other hyperparameters at train_compact_cnn.py defaults, rotations 0/180).
# Part 1: 6 pairs x {intensity_bgcorr_vnorm, z0} x 5 seeds x (real, particle-free); order PS, PE, PMMA, PVC, PP, PVA.
# Part 2: PET vs PS reverse fold (2026-10-04 split), both arms, 3 seeds. Part 3 (optional): PET vs PS, z0,
# --aug interior, 5 seeds. Class weights: pipeline_output/class_weights_twoclass_<pair>.json for both classifiers.
# Deadline guard: a pair / part starts only if (its runs x MIN_PER_RUN) fits before 2026-10-09 12:00.
# Failed runs are logged and skipped; finished runs are skipped on a restart. Report refreshed after every pair.
# usage: MIN_PER_RUN=<minutes> bash scripts/run_twoclass.sh
cd "/c/Users/haany/OneDrive/research project/dataset 10jan23" || exit 1
export PYTHONUNBUFFERED=1
PO=pipeline_output
LOG=results/logs_twoclass
mkdir -p "$LOG"
DEADLINE=$(date -d "2026-10-09 12:00" +%s)
MIN_PER_RUN=${MIN_PER_RUN:-10}

fits() {  # number of runs
  local need=$(( $1 * MIN_PER_RUN * 60 ))
  [ $(( $(date +%s) + need )) -le "$DEADLINE" ]
}
arm_vars() {  # arm [rev]
  local sfx=${2:+_rev}
  if [ "$1" = z0 ]; then
    CH=recon_s0; RM=$PO/manifest_recon_s0_z0$sfx.csv; RD=$PO/patches_recon_s0_z0
    BM=$PO/manifest_background_recon_s0_z0$sfx.csv; BD=$PO/patches_background_recon_s0_z0
  else
    CH=intensity_bgcorr_vnorm; RM=$PO/manifest_bgcorrected_common849$sfx.csv; RD=$PO/patches_bgcorrected
    BM=$PO/manifest_background_bgcorrected_common849$sfx.csv; BD=$PO/patches_background_bgcorrected
  fi
}
train() {  # split manifest patch_dir out classes class_weights [extra args...]
  local split=$1 man=$2 pdir=$3 out=$4 cls=$5 cw=$6; shift 6
  if [ -f "results/$out/results.json" ]; then echo "$out already done, skipped"; return; fi
  python train_compact_cnn.py --channels "$CH" --split "$split" --manifest "$man" --patch-dir "$pdir" \
    --class-weights "$cw" --classes "$cls" "$@" --epochs 150 --patience 30 --seed "$s" \
    --out "results/$out" > "$LOG/$out.log" 2>&1
  local rc=$?
  echo "$out exit=$rc $(date '+%m-%d %H:%M:%S')"
  [ "$rc" -ne 0 ] && echo "$out exit=$rc" >> "$LOG/failed.txt"
}

echo "two-class queue started $(date '+%m-%d %H:%M:%S'), MIN_PER_RUN=$MIN_PER_RUN"
for other in PS PE PMMA PVC PP PVA; do
  pair=PETvs$other
  if ! fits 20; then echo "NOT RUN (deadline): $pair"; continue; fi
  echo "=== START Part 1 $pair $(date '+%m-%d %H:%M:%S')"
  for arm in intensity_bgcorr_vnorm z0; do
    arm_vars $arm
    for s in 0 1 2 3 4; do
      train video_holdout "$RM" "$RD" "real_tc_${pair}_${arm}_s$s" "PET,$other" "$PO/class_weights_twoclass_$pair.json"
      train particle "$BM" "$BD" "bg_tc_${pair}_${arm}_s$s" "PET,$other" "$PO/class_weights_twoclass_$pair.json"
    done
  done
  python build_report_twoclass.py > /dev/null 2>> "$LOG/report_errors.txt"
  echo "=== DONE Part 1 $pair, report updated $(date '+%m-%d %H:%M:%S')"
done

if fits 12; then
  echo "=== START Part 2 PETvsPS reverse fold $(date '+%m-%d %H:%M:%S')"
  for arm in intensity_bgcorr_vnorm z0; do
    arm_vars $arm rev
    for s in 0 1 2; do
      train video_holdout "$RM" "$RD" "real_tc_PETvsPS_${arm}_rev_s$s" "PET,PS" "$PO/class_weights_twoclass_PETvsPS_rev.json"
      train particle "$BM" "$BD" "bg_tc_PETvsPS_${arm}_rev_s$s" "PET,PS" "$PO/class_weights_twoclass_PETvsPS_rev.json"
    done
  done
  python build_report_twoclass.py > /dev/null 2>> "$LOG/report_errors.txt"
  echo "=== DONE Part 2, report updated $(date '+%m-%d %H:%M:%S')"
else
  echo "NOT RUN (deadline): Part 2"
fi

if fits 10; then
  echo "=== START Part 3 PETvsPS z0 interior $(date '+%m-%d %H:%M:%S')"
  arm_vars z0
  for s in 0 1 2 3 4; do
    train video_holdout "$RM" "$RD" "real_tc_PETvsPS_z0_interior_s$s" "PET,PS" "$PO/class_weights_twoclass_PETvsPS.json" \
      --aug interior --donor-manifest "$BM" --donor-patch-dir "$BD"
    train particle "$BM" "$BD" "bg_tc_PETvsPS_z0_interior_s$s" "PET,PS" "$PO/class_weights_twoclass_PETvsPS.json" \
      --aug interior --donor-manifest "$BM" --donor-patch-dir "$BD"
  done
  python build_report_twoclass.py > /dev/null 2>> "$LOG/report_errors.txt"
  echo "=== DONE Part 3, report updated $(date '+%m-%d %H:%M:%S')"
else
  echo "NOT RUN (deadline or optional): Part 3"
fi
echo "QUEUE COMPLETE $(date '+%m-%d %H:%M:%S')"
