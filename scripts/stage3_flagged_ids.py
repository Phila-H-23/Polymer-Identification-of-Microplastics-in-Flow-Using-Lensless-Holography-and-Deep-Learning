"""Item 41 stage 3: the 7 patches flagged by check (c) (rule fixed before the build, unchanged) ->
results/recon_V3_flagged_ids.txt (read-only, new file)."""
import os
import json
import pandas as pd

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
OUT = "results/recon_V3_flagged_ids.txt"
c = json.load(open("results/recon_V3_checks_PETvsPS.json"))["c"]
man = {("real", d): pd.read_csv(f"pipeline_output/manifest_recon_V3_PETPS{s}.csv") for d, s in (("original", ""), ("reverse", "_rev"))}
man |= {("particle-free", d): pd.read_csv(f"pipeline_output/manifest_background_recon_V3_PETPS{s}.csv")
        for d, s in (("original", ""), ("reverse", "_rev"))}
col = {"real": "split_video_holdout", "particle-free": "split"}
L = ["PATCHES FLAGGED BY CHECK (c) IN THE V3 DATASET (rule fixed before the build and not changed: in a patch within 220 um",
     "of a frame edge, the alternation of the 8 column (or row) means nearest that edge, |mean((p - mean p) * (-1)^k)|, is above",
     "0.005 and above 3 x the same quantity for the 8 columns (rows) at the opposite side of the patch). Check (c) FAILED: 7 of",
     "228 patch-edge cases flagged in V3 (17 in the as-built V1); going ahead with Step 2 was Phila's decision after seeing it.",
     "Split = the patch's split in the original / reverse direction (only 'test' patches enter the test metrics).", "",
     f"{'patch id':36s} {'class':5s} {'kind':13s} {'video':5s} {'frame':>5s} {'x':>5s} {'y':>5s}  flag that fired"]
for x in c["rows"]:
    if not x["v3_stripe"]:
        continue
    m = man[(x["kind"], "original")].set_index("patch_file").loc[x["patch_file"]]
    sp = [man[(x["kind"], d)].set_index("patch_file").loc[x["patch_file"], col[x["kind"]]] for d in ("original", "reverse")]
    L.append(f"{x['patch_file'][:-4]:36s} {m['class']:5s} {x['kind']:13s} {m['source_file'][:-8]:5s} {int(m['frame']):5d} {int(m['x']):5d} "
             f"{int(m['y']):5d}  {x['edge']} edge: V3 alternation {x['v3_near']:.4f} > 0.005 and > 3 x {x['v3_far']:.4f} (far side); "
             f"as-built V1 {x['v1_near']:.4f} / {x['v1_far']:.4f} ({'also' if x['v1_stripe'] else 'not'} flagged in V1); split "
             f"{sp[0]} / {sp[1]}")
L += ["", "Images: results/recon_V3_edge_flagged.png (V1 beside V3 and the edge profiles)."]
text = "\n".join(L)
assert "\u2014" not in text and "significant" not in text.lower()
if os.path.exists(OUT):
    raise FileExistsError(OUT)
open(OUT, "w", encoding="utf-8").write(text + "\n")
print(text)
