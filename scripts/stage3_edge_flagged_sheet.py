"""Item 41 stage 3, report aid after check (c) failed (read-only; new file results/recon_V3_edge_flagged.png):
the 7 patch-edge cases flagged as stripes in V3, each as-built V1 beside V3 (display 0.5 to 1.5), and the column (row)
mean profile of the 8 columns (rows) nearest the frame edge, the quantity check (c) tests."""
import os
import sys
import json

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import recon_check as rc  # noqa: E402

OUT = "results/recon_V3_edge_flagged.png"
c = json.load(open("results/recon_V3_checks_PETvsPS.json"))["c"]
rows = [x for x in c["rows"] if x["v3_stripe"]]
gray = matplotlib.colormaps["gray"].copy()
gray.set_bad("#9ecae1")
ext = [-220.8, 220.8, 220.8, -220.8]
fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.0 * len(rows)), dpi=90)
for i, x in enumerate(rows):
    d_old = "pipeline_output/patches_recon_s0" if x["kind"] == "real" else "pipeline_output/patches_background_recon_s0"
    d_new = "pipeline_output/patches_recon_V3_PETPS" if x["kind"] == "real" else "pipeline_output/patches_background_recon_V3_PETPS"
    prof = {}
    for j, (d, lab) in enumerate(((d_old, "as-built V1"), (d_new, "V3"))):
        a = np.load(os.path.join(d, x["patch_file"]))[..., 0].astype(np.float64)
        ax = axes[i, j]
        ax.imshow(rc.place(a, x["x"], x["y"], 32), cmap=gray, vmin=0.5, vmax=1.5, interpolation="nearest", extent=ext)
        ax.set_title(f"{x['kind']} {x['patch_file'][:-4]} (x={x['x']}, y={x['y']})\n{lab}, {x['edge']} edge, alternation "
                     f"{x['v1_near' if j == 0 else 'v3_near']:.4f} (far side {x['v1_far' if j == 0 else 'v3_far']:.4f})", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        p = a.mean(axis=0) if x["edge"] in ("left", "right") else a.mean(axis=1)
        prof[lab] = p[:12] if x["edge"] in ("left", "top") else p[::-1][:12]
    ax = axes[i, 2]
    for lab, col in (("as-built V1", "k"), ("V3", "r")):
        ax.plot(np.arange(12), prof[lab], "-o", ms=3, color=col, label=lab)
    ax.axvspan(-0.5, 7.5, color="0.9")
    ax.set_xlabel(f"column (row) of the stored patch counted from the {x['edge']} frame edge")
    ax.set_ylabel("mean along the edge")
    ax.set_title("mean profile, first 12 columns (rows) from the edge; grey = the 8 tested", fontsize=7)
    ax.legend(fontsize=7)
fig.suptitle("Check (c): the 7 patch-edge cases flagged in V3 (rule: alternation of the 8 means nearest the edge > 0.005 "
             "and > 3 x the far side)", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.985])
if os.path.exists(OUT):
    raise FileExistsError(OUT)
fig.savefig(OUT)
print(OUT)
