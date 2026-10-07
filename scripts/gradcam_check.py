"""
gradcam_check.py

Read-only visual check, no new training. Uses the already-saved
pol4_bgcorr_vnorm checkpoints (seed 0, real and background-only) to see
where each classifier's attention concentrates.

CAM, not approximate Grad-CAM: CompactHoloCNN ends in
AdaptiveAvgPool2d(1) -> Flatten -> Dropout -> Linear directly (Dropout is
identity in eval mode), which is exactly the architecture the original
Zhou et al. CAM method targets - so CAM_c(h,w) = sum_k W_fc[c,k] *
feat_k(h,w) is computed exactly, no gradient approximation needed, using
the last conv feature map (block4 output, 8x8x128) and the final Linear
layer's weights.

Base image for the overlay: mean of the 4 RAW pol4_bgcorr channels
(pre-vnorm) - matches the visualization convention used in every earlier
gallery/visual-check image in this project, so it's directly comparable
by eye. The CAM itself is computed on the ACTUAL model input (vnorm +
dataset channel mean/std normalised), loaded via the same PatchDataset
class the training scripts use, so the heatmap reflects what the network
actually saw, just displayed on a more readable background.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
RESULTS_DIR = os.path.join(DATA_DIR, "results")
REAL_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_bgcorrected.csv")
REAL_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_bgcorrected")
BG_MANIFEST = os.path.join(DATA_DIR, "pipeline_output", "manifest_background_bgcorrected.csv")
BG_PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches_background_bgcorrected")
REAL_CKPT_DIR = os.path.join(RESULTS_DIR, "real_vh_pol4bgcorrvnorm_s0")
BG_CKPT_DIR = os.path.join(RESULTS_DIR, "bg_pol4bgcorrvnorm_s0")

spec = importlib.util.spec_from_file_location("train_compact_cnn", os.path.join(DATA_DIR, "train_compact_cnn.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def load_model_and_stats(manifest_path, split_col, patch_dir, ckpt_dir, classes):
    manifest = pd.read_csv(manifest_path)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_df = manifest[manifest[split_col] == "train"]
    test_df = manifest[manifest[split_col] == "test"]
    mean, std = m.compute_channel_stats(train_df, patch_dir, class_to_idx, "pol4_bgcorr_vnorm")

    model = m.CompactHoloCNN(in_ch=4, n_classes=len(classes), width=32, dropout=0.3)
    state = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, mean, std, test_df, class_to_idx


def compute_cam(model, x):
    """x: (1, 4, 64, 64) normalised tensor. Returns (predicted_class_idx, cam_64x64)."""
    with torch.no_grad():
        feat = model.stem(x)
        feat = model.block1(feat)
        feat = model.block2(feat)
        feat = model.block3(feat)
        feat = model.block4(feat)  # (1, 128, 8, 8)
        pooled = feat.mean(dim=(2, 3))  # matches AdaptiveAvgPool2d(1)+Flatten
        logits = model.head[-1](pooled)  # skip AvgPool/Flatten/Dropout (already pooled, eval-mode dropout=identity)
        pred = int(logits.argmax(dim=1).item())

        fc_weight = model.head[-1].weight  # (n_classes, 128)
        cam = torch.einsum("k,khw->hw", fc_weight[pred], feat[0])  # (8, 8)
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return pred, cam.numpy()


def raw_display_image(patch_dir, patch_file):
    patch = np.load(os.path.join(patch_dir, patch_file)).astype(np.float32)
    h, w = patch.shape[:2]
    canvas = np.zeros((64, 64, 4), dtype=np.float32)
    canvas[:h, :w, :] = patch[:64, :64, :]
    return canvas.mean(axis=-1)


def find_correct_examples(model, mean, std, test_df, patch_dir, class_to_idx, classes, target_classes, n_per_class=3):
    """Returns dict: class -> list of (patch_file, cam, display_img)"""
    out = {c: [] for c in target_classes}
    for cls in target_classes:
        sub = test_df[test_df["class"] == cls]
        true_idx = class_to_idx[cls]
        for _, row in sub.iterrows():
            if len(out[cls]) >= n_per_class:
                break
            ds = m.PatchDataset(pd.DataFrame([row]), patch_dir, class_to_idx, channels="pol4_bgcorr_vnorm",
                                 augment=False, mean=mean, std=std)
            x, label = ds[0]
            pred, cam = compute_cam(model, x.unsqueeze(0))
            if pred == true_idx:
                disp = raw_display_image(patch_dir, row["patch_file"])
                out[cls].append((row["patch_file"], cam, disp))
    return out


def save_grid(examples_dict, title, out_path, n_per_class=3):
    classes = list(examples_dict.keys())
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(3 * n_per_class, 3 * len(classes)))
    if len(classes) == 1:
        axes = axes.reshape(1, -1)
    for ci, cls in enumerate(classes):
        items = examples_dict[cls]
        for j in range(n_per_class):
            ax = axes[ci, j]
            if j < len(items):
                patch_file, cam, disp = items[j]
                ax.imshow(disp, cmap="gray")
                ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.axis("off")
        fig.text(0.01, 1 - (ci + 0.5) / len(classes), cls, fontsize=11, fontweight="bold", va="center")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0.05, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    classes = sorted(pd.read_csv(BG_MANIFEST)["class"].unique())
    print("classes:", classes)

    print("\nChecking checkpoints exist...")
    for ckpt_dir in [REAL_CKPT_DIR, BG_CKPT_DIR]:
        p = os.path.join(ckpt_dir, "best.pt")
        if not os.path.exists(p):
            print(f"MISSING: {p} - stopping, not retraining.")
            return
        print(f"  found: {p}")

    # ------------------------------------------------------------------
    # Background-only classifier
    # ------------------------------------------------------------------
    print("\n=== Loading background-only classifier (bg_pol4bgcorrvnorm_s0) ===")
    bg_model, bg_mean, bg_std, bg_test, bg_c2i = load_model_and_stats(
        BG_MANIFEST, "split", BG_PATCH_DIR, BG_CKPT_DIR, classes)

    print("Finding correctly-classified PE/PMMA/PVC background-only patches...")
    dusty_examples = find_correct_examples(bg_model, bg_mean, bg_std, bg_test, BG_PATCH_DIR, bg_c2i, classes,
                                            ["PE", "PMMA", "PVC"])
    for c, v in dusty_examples.items():
        print(f"  {c}: {len(v)} correct examples found")
    save_grid(dusty_examples, "Background-only classifier - CAM on PE/PMMA/PVC (dust-flagged classes)",
              os.path.join(RESULTS_DIR, "gradcam_bg_dusty.png"))

    print("\nFinding correctly-classified PET/PP/PS background-only patches...")
    clean_examples = find_correct_examples(bg_model, bg_mean, bg_std, bg_test, BG_PATCH_DIR, bg_c2i, classes,
                                            ["PET", "PP", "PS"])
    for c, v in clean_examples.items():
        print(f"  {c}: {len(v)} correct examples found")
    save_grid(clean_examples, "Background-only classifier - CAM on PET/PP/PS (clean classes)",
              os.path.join(RESULTS_DIR, "gradcam_bg_clean.png"))

    # ------------------------------------------------------------------
    # Real classifier
    # ------------------------------------------------------------------
    print("\n=== Loading real classifier (real_vh_pol4bgcorrvnorm_s0) ===")
    real_model, real_mean, real_std, real_test, real_c2i = load_model_and_stats(
        REAL_MANIFEST, "split_video_holdout", REAL_PATCH_DIR, REAL_CKPT_DIR, classes)

    print("Finding correctly-classified PE/PVC real patches...")
    real_dusty = find_correct_examples(real_model, real_mean, real_std, real_test, REAL_PATCH_DIR, real_c2i, classes,
                                        ["PE", "PVC"])
    for c, v in real_dusty.items():
        print(f"  {c}: {len(v)} correct examples found")

    print("Finding correctly-classified PET/PS real patches...")
    real_clean = find_correct_examples(real_model, real_mean, real_std, real_test, REAL_PATCH_DIR, real_c2i, classes,
                                        ["PET", "PS"])
    for c, v in real_clean.items():
        print(f"  {c}: {len(v)} correct examples found")

    real_combined = {**real_dusty, **real_clean}
    save_grid(real_combined, "Real classifier - CAM on PE/PVC vs PET/PS",
              os.path.join(RESULTS_DIR, "gradcam_real.png"))

    print("\nDone. Saved: gradcam_bg_dusty.png, gradcam_bg_clean.png, gradcam_real.png")


if __name__ == "__main__":
    main()
