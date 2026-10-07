"""Per-class patch counts of the reconstructed datasets vs the originals (frame-0
patches are dropped by the build: no rolling background exists for them)."""
import os
import pandas as pd

PO = r"C:\Users\haany\OneDrive\research project\dataset 10jan23\pipeline_output"

orig_real = pd.read_csv(os.path.join(PO, "manifest.csv"))
orig_bg = pd.read_csv(os.path.join(PO, "manifest_background.csv"))
s0_real = pd.read_csv(os.path.join(PO, "manifest_recon_s0.csv"))
s0_bg = pd.read_csv(os.path.join(PO, "manifest_background_recon_s0.csv"))
prop_real = pd.read_csv(os.path.join(PO, "manifest_recon_prop.csv"))
prop_bg = pd.read_csv(os.path.join(PO, "manifest_background_recon_prop.csv"))

t = pd.DataFrame({
    "real_orig": orig_real["class"].value_counts(),
    "real_recon": s0_real["class"].value_counts(),
    "bg_orig": orig_bg["class"].value_counts(),
    "bg_recon": s0_bg["class"].value_counts(),
}).fillna(0).astype(int)
t["real_dropped"] = t["real_orig"] - t["real_recon"]
t["bg_dropped"] = t["bg_orig"] - t["bg_recon"]
t.loc["TOTAL"] = t.sum()
print("recon_s0 per-class patch counts (recon_prop identical: "
      f"{set(prop_real.patch_file) == set(s0_real.patch_file) and set(prop_bg.patch_file) == set(s0_bg.patch_file)})")
print(t.to_string())

print("\nsplit breakdown (real: split_video_holdout, background-only: split)")
print(pd.crosstab(s0_real["class"], s0_real["split_video_holdout"], margins=True).to_string())
print()
print(pd.crosstab(s0_bg["class"], s0_bg["split"], margins=True).to_string())
