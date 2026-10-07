"""Runs train_compact_cnn.py's main() unchanged (same arguments, passed straight through), with one module constant
pointed elsewhere: VIDEO_BG_STATS_PATH = pipeline_output/video_background_stats_PETPS_exp.json (the per-video
normalisation statistics recomputed on the expanded set's background-only patches). train_compact_cnn.py itself is
not modified. After training, records the stats file used (path and sha256) in <out>/vnorm_stats_used.json.
usage: python scripts/train_vnorm_exp.py <train_compact_cnn.py arguments>"""
import os
import sys
import json
import hashlib

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
STATS = "pipeline_output/video_background_stats_PETPS_exp.json"
os.chdir(D)
sys.path.insert(0, D)
import train_compact_cnn as t  # noqa: E402

t.VIDEO_BG_STATS_PATH = STATS
t.main()
out = sys.argv[sys.argv.index("--out") + 1]
json.dump(dict(video_bg_stats_path=STATS, sha256=hashlib.sha256(open(STATS, "rb").read()).hexdigest()),
          open(os.path.join(out, "vnorm_stats_used.json"), "w"), indent=2)
