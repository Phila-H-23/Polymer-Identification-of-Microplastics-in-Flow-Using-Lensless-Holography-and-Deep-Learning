#!/usr/bin/env bash
# Real (video-holdout) vs background-only, 5 seeds each, same hyperparameters
# as the rung-1/rung-2 runs (epochs 150, patience 30).
# usage: run_real_bg_seeds.sh CHANNELS TAG REAL_MANIFEST REAL_DIR BG_MANIFEST BG_DIR [CLASS_WEIGHTS]
# CLASS_WEIGHTS (optional) is passed to BOTH arms; omitted = each split's default file.
# outputs: results/real_vh_${TAG}_s{0..4}, results/bg_${TAG}_s{0..4}
cd "/c/Users/haany/OneDrive/research project/dataset 10jan23" || exit 1
export PYTHONUNBUFFERED=1
ch=$1; tag=$2; rman=$3; rdir=$4; bman=$5; bdir=$6; cw=${7:-}
mkdir -p "results/logs_$tag"
train() {  # split manifest patch_dir out log
  if [ -f "results/$4/results.json" ]; then echo "$4 already done, skipped"; return; fi
  python train_compact_cnn.py --channels "$ch" --split "$1" --manifest "$2" --patch-dir "$3" \
    ${cw:+--class-weights "$cw"} --epochs 150 --patience 30 --seed "$s" --out "results/$4" > "results/logs_$tag/$5" 2>&1
  echo "$4 exit=$? $(date '+%m-%d %H:%M:%S')"
}
for s in 0 1 2 3 4; do
  train video_holdout "$rman" "$rdir" "real_vh_${tag}_s$s" "real_s$s.log"
  train particle "$bman" "$bdir" "bg_${tag}_s$s" "bg_s$s.log"
done
echo "$tag ALL_DONE $(date +%H:%M:%S)"
