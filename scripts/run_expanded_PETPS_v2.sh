#!/usr/bin/env bash
# Expanded PET and PS set, everything after Part 1, unattended (v2: adds the addendum steps for tracks,
# collision_kept and validation by track before post_z0; replaces run_expanded_PETPS.sh, which was stopped
# while it was still waiting for the z0 build):
#   1. wait for the running z0 build (scripts/build_arms_PETPS_exp.py z0), then post_z0 (old patches byte-identical
#      check, reverse-fold manifests)
#   2. Part 3 in full (3a, 3b, 3c; scripts/scalar_baselines_PETPS_exp.py, single-threaded; no training is running)
#   3. Part 4 z0: original direction then reverse fold (seeds 0-2); the first finished run is timed and the z0 batch
#      projected; if the projection exceeds 5 h the original direction uses seeds 0-2 for both arms, else 0-4
#   4. bgcorrected build + post_bgcorr (new per-video normalisation file); builds and training never overlap (memory)
#   5. Part 4 intensity_bgcorr_vnorm (optional, last): only if Part 3 is complete and the projected end is before
#      2026-10-10 09:00
# Training command line as scripts/run_size_arms.sh: --classes PET,PS --epochs 150 --patience 30, other settings at
# train_compact_cnn.py defaults, no --aug (as in every earlier z0 run). Class weights
# class_weights_twoclass_PETvsPS_exp.json (original) / _rev_exp.json (reverse), same file for real and particle-free.
# Failed runs are logged and skipped; finished runs are skipped on a restart. Report refreshed after every step.
cd "/c/Users/haany/OneDrive/research project/dataset 10jan23" || exit 1
export PYTHONUNBUFFERED=1
PO=pipeline_output
LOG=results/logs_expanded_PETPS_exp
EXP=results/expanded_PETPS_exp
mkdir -p "$LOG"
LATEST_END=$(date -d "2026-10-10 09:00" +%s)
ts() { date '+%m-%d %H:%M:%S'; }
report() { python build_report_expanded_PETvsPS.py > /dev/null 2>> "$LOG/report_errors.txt"; }

arm_vars() {  # arm [rev]
  local sfx=_PETPS${2:+_rev}_exp
  if [ "$1" = z0 ]; then
    CH=recon_s0; PY=train_compact_cnn.py
    RM=$PO/manifest_recon_s0_z0$sfx.csv; RD=$PO/patches_recon_s0_z0_PETPS_exp
    BM=$PO/manifest_background_recon_s0_z0$sfx.csv; BD=$PO/patches_background_recon_s0_z0_PETPS_exp
  else
    CH=intensity_bgcorr_vnorm; PY=scripts/train_vnorm_exp.py
    RM=$PO/manifest_bgcorrected$sfx.csv; RD=$PO/patches_bgcorrected_PETPS_exp
    BM=$PO/manifest_background_bgcorrected$sfx.csv; BD=$PO/patches_background_bgcorrected_PETPS_exp
  fi
  CW=$PO/class_weights_twoclass_PETvsPS${2:+_rev}_exp.json
}
train() {  # split manifest patch_dir out
  local split=$1 man=$2 pdir=$3 out=$4
  if [ -f "results/$out/results.json" ]; then echo "$out already done, skipped"; return; fi
  local t0=$(date +%s)
  python $PY --channels "$CH" --split "$split" --manifest "$man" --patch-dir "$pdir" \
    --class-weights "$CW" --classes "PET,PS" --epochs 150 --patience 30 --seed "$s" \
    --out "results/$out" > "$LOG/$out.log" 2>&1
  local rc=$?
  LAST_SECONDS=$(( $(date +%s) - t0 ))
  echo "$out exit=$rc ${LAST_SECONDS}s $(ts)"
  [ "$rc" -ne 0 ] && echo "$out exit=$rc" >> "$LOG/failed.txt"
}
batch() {  # arm direction(orig|rev) seeds...
  local arm=$1 dir=$2; shift 2
  local rsfx=""; [ "$dir" = rev ] && rsfx=_rev
  arm_vars $arm ${rsfx:+rev}
  echo "=== START $arm $dir seeds $* $(ts)"
  for s in "$@"; do
    train video_holdout "$RM" "$RD" "real_tc_PETvsPS_${arm}_exp${rsfx}_s$s"
    train particle "$BM" "$BD" "bg_tc_PETvsPS_${arm}_exp${rsfx}_s$s"
  done
  report
  echo "=== DONE $arm $dir, report updated $(ts)"
}

echo "expanded PET vs PS chain started $(ts)"
# 1. z0 build
until grep -q -E "=== Summary|Traceback" results/expanded_PETPS_exp_build_z0.log; do sleep 60; done
if grep -q Traceback results/expanded_PETPS_exp_build_z0.log; then echo "z0 build FAILED $(ts)"; exit 1; fi
sleep 90  # let the build process write its manifests and exit before anything else loads data
# addendum (a): collision_kept needs the Step 6 survivors of the chosen grid (video reloads, no training running)
python scripts/extract_PETPS_exp.py --collisions > "$LOG/collisions.log" 2>&1 && echo "collisions done $(ts)" || echo "collisions FAILED $(ts)"
# addendum (a), (b): tracks, collision_kept, validation by track, class weights (deterministic; same splits as before)
python scripts/tracks_PETPS_exp.py > "$LOG/tracks.log" 2>&1 || { echo "tracks FAILED $(ts)"; exit 1; }
python scripts/build_arms_PETPS_exp.py post_z0 > "$LOG/post_z0.log" 2>&1 || { echo "post_z0 FAILED $(ts)"; exit 1; }
echo "z0 build checked: $(head -1 "$LOG/post_z0.log") $(ts)"
report
# 2. Part 3 in full
python scripts/scalar_baselines_PETPS_exp.py > "$EXP/part3.log" 2>&1 && echo "Part 3 done $(ts)" || echo "Part 3 FAILED $(ts)"
report
# 3. z0, first run timed
s=0; arm_vars z0
train video_holdout "$RM" "$RD" "real_tc_PETvsPS_z0_exp_s0"
[ -f "$EXP/part4_projection.json" ] || python - "$LAST_SECONDS" <<'EOF'
import json, sys, pandas as pd
t = float(sys.argv[1]); m = pd.read_csv("pipeline_output/manifest_PETPS_exp.csv")
r = (m.split_rev == "train").sum() / (m.split_video_holdout == "train").sum()
z0 = t * (10 + 6 * r) / 3600
seeds = "0 1 2" if z0 > 5 else "0 1 2 3 4"
used = t * ((6 if z0 > 5 else 10) + 6 * r) / 3600
json.dump(dict(first_run="real_tc_PETvsPS_z0_exp_s0", first_run_seconds=t, rev_train_ratio=float(r),
               projected_z0_batch_hours_5_seeds=z0, original_seeds=seeds, projected_z0_batch_hours_used=used,
               projected_intensity_batch_hours=used),
          open("results/expanded_PETPS_exp/part4_projection.json", "w"), indent=2)
print(f"first run {t / 60:.1f} min; projected z0 batch with 5 original seeds {z0:.2f} h -> original seeds {seeds}")
EOF
report
ORIG_SEEDS=$(python -c "import json; print(json.load(open('$EXP/part4_projection.json'))['original_seeds'])")
INT_H=$(python -c "import json; print(int(json.load(open('$EXP/part4_projection.json'))['projected_intensity_batch_hours']) + 2)")
batch z0 orig $ORIG_SEEDS
batch z0 rev 0 1 2
# 4. bgcorrected build
if [ ! -f "$EXP/part2_bgcorr_check.json" ]; then
  echo "=== bgcorrected build $(ts)"
  python scripts/build_arms_PETPS_exp.py bgcorr > results/expanded_PETPS_exp_build_bgcorr.log 2>&1 \
    && python scripts/build_arms_PETPS_exp.py post_bgcorr > "$LOG/post_bgcorr.log" 2>&1 \
    && echo "bgcorrected build checked: $(head -1 "$LOG/post_bgcorr.log") $(ts)" || echo "bgcorrected build FAILED $(ts)"
  report
fi
# 5. intensity_bgcorr_vnorm, optional and last
P3_DONE=$(python -c "import json; print(0 if json.load(open('$EXP/part3_results.json')).get('only_3a') else 1)" 2>/dev/null)
if [ "$P3_DONE" != 1 ]; then
  echo "SKIPPED: Part 3 not finished when the z0 batch ended" | tee "$EXP/part4_intensity_status.txt"
elif [ ! -f "$EXP/part2_bgcorr_check.json" ]; then
  echo "SKIPPED: bgcorrected build failed" | tee "$EXP/part4_intensity_status.txt"
elif [ $(( $(date +%s) + INT_H * 3600 )) -gt "$LATEST_END" ]; then
  echo "SKIPPED: projected end after 2026-10-10 09:00" | tee "$EXP/part4_intensity_status.txt"
else
  echo "RUN: started $(date '+%Y-%m-%d %H:%M'), after the z0 batch and Part 3 were finished" | tee "$EXP/part4_intensity_status.txt"
  batch intensity_bgcorr_vnorm orig $ORIG_SEEDS
  batch intensity_bgcorr_vnorm rev 0 1 2
fi
report
echo "QUEUE COMPLETE $(ts)"
