"""
gradcam_recon.py

Exact CAM (same method as scripts/gradcam_pet.py: block4 feature map
weighted by the final Linear layer, no gradient approximation) on the seed-0
recon checkpoints, real and background-only classifiers, for the classes
given on the command line. Read-only, no retraining.

Usage: python scripts/gradcam_recon.py --variant recon_s0_z0 --classes PE PVC
"""
import os
import argparse
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
RESULTS_DIR = os.path.join(DATA_DIR, "results")
N_EXAMPLES = 8

# Queue v2 checkpoints (seed 0). Manifests must match training so the
# normalisation stats (computed on the train split) are the same.
BGC = ("manifest_bgcorrected_common849.csv", "patches_bgcorrected")
BGC_BG = ("manifest_background_bgcorrected_common849.csv", "patches_background_bgcorrected")
RAW = ("manifest_common849.csv", "patches")
RAW_BG = ("manifest_background_common849.csv", "patches_background")
CONFIG = {
    # variant: channels, in_ch, display channel indices (None = 1-ch float patch), real, bg
    "recon_s0": dict(channels="recon_s0", in_ch=1, disp=None,
                     real=("manifest_recon_s0.csv", "patches_recon_s0", "real_vh_recons0_r180_s0"),
                     bg=("manifest_background_recon_s0.csv", "patches_background_recon_s0", "bg_recons0_r180_s0")),
    "recon_s0_z0": dict(channels="recon_s0", in_ch=1, disp=None,
                        real=("manifest_recon_s0_z0.csv", "patches_recon_s0_z0", "real_vh_recons0z0_c849_s0"),
                        bg=("manifest_background_recon_s0_z0.csv", "patches_background_recon_s0_z0",
                            "bg_recons0z0_c849_s0")),
    "recon_biref": dict(channels="recon_biref", in_ch=3, disp="prop",
                        real=("manifest_recon_prop.csv", "patches_recon_prop", "real_vh_reconbiref_c849_s0"),
                        bg=("manifest_background_recon_prop.csv", "patches_background_recon_prop",
                            "bg_reconbiref_c849_s0")),
    "intensity_bgcorr_vnorm": dict(channels="intensity_bgcorr_vnorm", in_ch=1, disp=[0, 1, 2, 3],
                                   real=(*BGC, "real_vh_intbv_c849_s0"), bg=(*BGC_BG, "bg_intbv_c849_s0")),
    "pol4_bgcorr_vnorm": dict(channels="pol4_bgcorr_vnorm", in_ch=4, disp=[0, 1, 2, 3],
                              real=(*BGC, "real_vh_pol4bgcorrvnorm_c849_s0"),
                              bg=(*BGC_BG, "bg_pol4bgcorrvnorm_c849_s0")),
    "i0": dict(channels="i0_bgcorr_vnorm", in_ch=1, disp=[0],
               real=(*BGC, "real_vh_i0bv_c849_s0"), bg=(*BGC_BG, "bg_i0bv_c849_s0")),
    "i45": dict(channels="i45_bgcorr_vnorm", in_ch=1, disp=[1],
                real=(*BGC, "real_vh_i45bv_c849_s0"), bg=(*BGC_BG, "bg_i45bv_c849_s0")),
    "i0i90": dict(channels="i0i90_bgcorr_vnorm", in_ch=2, disp=[0, 2],
                  real=(*BGC, "real_vh_i0i90bv_c849_s0"), bg=(*BGC_BG, "bg_i0i90bv_c849_s0")),
    "pol4": dict(channels="pol4", in_ch=4, disp=[0, 1, 2, 3],
                 real=(*RAW, "real_vh_pol4_c849_s0"), bg=(*RAW_BG, "bg_pol4_c849_s0")),
    "pol4_ff": dict(channels="pol4_ff", in_ch=4, disp=[0, 1, 2, 3],
                    real=(*RAW, "real_vh_pol4ff_c849_s0"), bg=(*RAW_BG, "bg_pol4ff_c849_s0")),
    "pol4_bgcorr": dict(channels="pol4_bgcorr", in_ch=4, disp=[0, 1, 2, 3],
                        real=(*BGC, "real_vh_pol4bgcorr_c849_s0"), bg=(*BGC_BG, "bg_pol4bgcorr_c849_s0")),
    "stokes": dict(channels="stokes", in_ch=3, disp=[0, 1, 2, 3],
                   real=(*RAW, "real_vh_stokes_c849_s0"), bg=(*RAW_BG, "bg_stokes_c849_s0")),
}

spec = importlib.util.spec_from_file_location("gradcam_pet", os.path.join(DATA_DIR, "scripts", "gradcam_pet.py"))
gp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gp)
m = gp.m


def display_image(patch_dir, patch_file, disp):
    """disp: list of uint8 channel indices to average, None = 1-ch float patch,
    "prop" = mean of the four propagated intensities of a 6-ch recon_prop patch."""
    if isinstance(disp, list):
        return gp.raw_display_image(patch_dir, patch_file, disp)
    patch = np.load(os.path.join(patch_dir, patch_file)).astype(np.float32)
    canvas = np.zeros((64, 64), dtype=np.float32)
    h, w = patch.shape[:2]
    canvas[:h, :w] = patch[:64, :64, :4].mean(axis=-1) if disp == "prop" else patch[:64, :64, 0]
    return canvas


def correct_examples(model, mean, std, test_df, patch_dir, class_to_idx, channels, cls):
    out = []
    for _, row in test_df[test_df["class"] == cls].iterrows():
        if len(out) >= N_EXAMPLES:
            break
        ds = m.PatchDataset(pd.DataFrame([row]), patch_dir, class_to_idx, channels=channels,
                             augment=False, mean=mean, std=std)
        x, _ = ds[0]
        pred, cam = gp.compute_cam(model, x.unsqueeze(0))
        if pred == class_to_idx[cls]:
            out.append((row["patch_file"], cam))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(CONFIG), required=True)
    ap.add_argument("--classes", nargs="+", required=True)
    args = ap.parse_args()
    cfg = CONFIG[args.variant]
    channels = cfg["channels"]

    sides = {}
    for side, split_col in (("real", "split_video_holdout"), ("bg", "split")):
        man, pdir, ckpt = cfg[side]
        man_path = os.path.join(DATA_DIR, "pipeline_output", man)
        patch_dir = os.path.join(DATA_DIR, "pipeline_output", pdir)
        classes = sorted(pd.read_csv(man_path)["class"].unique())
        model, mean, std, test_df, c2i = gp.load_model_and_stats(
            man_path, split_col, patch_dir, os.path.join(RESULTS_DIR, ckpt), classes, channels, cfg["in_ch"])
        sides[side] = (model, mean, std, test_df, c2i, patch_dir)

    for cls in args.classes:
        rows = []
        for side, label in (("real", "real"), ("bg", "background-only")):
            model, mean, std, test_df, c2i, patch_dir = sides[side]
            items = correct_examples(model, mean, std, test_df, patch_dir, c2i, channels, cls)
            print(f"{cls} {label}: {len(items)} correctly-classified test examples")
            rows.append((label, items, patch_dir))

        fig, axes = plt.subplots(2, N_EXAMPLES, figsize=(3 * N_EXAMPLES, 6))
        for r, (label, items, patch_dir) in enumerate(rows):
            for j in range(N_EXAMPLES):
                ax = axes[r, j]
                if j < len(items):
                    pf, cam = items[j]
                    ax.imshow(display_image(patch_dir, pf, cfg["disp"]), cmap="gray")
                    ax.imshow(cam, cmap="jet", alpha=0.45)
                ax.axis("off")
            fig.text(0.005, 1 - (r + 0.5) / 2, label, fontsize=11, fontweight="bold", va="center")
        fig.suptitle(f"CAM on correctly-classified {cls} patches - {args.variant} (seed 0)", fontsize=13)
        plt.tight_layout(rect=[0.05, 0, 1, 0.93])
        out = os.path.join(RESULTS_DIR, f"gradcam_{args.variant}_{cls}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
