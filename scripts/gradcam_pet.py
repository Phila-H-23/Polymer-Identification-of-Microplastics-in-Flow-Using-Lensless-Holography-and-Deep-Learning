"""
gradcam_pet.py

Last diagnostic in the channel-ablation thread. Read-only, no retraining.
Exact CAM (same method as scripts/gradcam_check.py: CompactHoloCNN ends
in AdaptiveAvgPool2d(1) -> Flatten -> Dropout -> Linear directly, so
CAM_c(h,w) = sum_k W_fc[c,k] * feat_k(h,w) on the block4 feature map is
exact, no gradient approximation) on the already-trained I0-only and
I0+I90 checkpoints (seed 0), class PET only, real and background-only
classifiers.
"""
import os
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

N_EXAMPLES = 8

spec = importlib.util.spec_from_file_location("train_compact_cnn", os.path.join(DATA_DIR, "train_compact_cnn.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def load_model_and_stats(manifest_path, split_col, patch_dir, ckpt_dir, classes, channels, in_ch):
    manifest = pd.read_csv(manifest_path)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_df = manifest[manifest[split_col] == "train"]
    test_df = manifest[manifest[split_col] == "test"]
    mean, std = m.compute_channel_stats(train_df, patch_dir, class_to_idx, channels)

    model = m.CompactHoloCNN(in_ch=in_ch, n_classes=len(classes), width=32, dropout=0.3)
    state = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, mean, std, test_df, class_to_idx


def compute_cam(model, x):
    with torch.no_grad():
        feat = model.stem(x)
        feat = model.block1(feat)
        feat = model.block2(feat)
        feat = model.block3(feat)
        feat = model.block4(feat)
        pooled = feat.mean(dim=(2, 3))
        logits = model.head[-1](pooled)
        pred = int(logits.argmax(dim=1).item())

        fc_weight = model.head[-1].weight
        cam = torch.einsum("k,khw->hw", fc_weight[pred], feat[0])
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return pred, cam.numpy()


def raw_display_image(patch_dir, patch_file, channel_indices):
    patch = np.load(os.path.join(patch_dir, patch_file)).astype(np.float32)
    h, w = patch.shape[:2]
    canvas = np.zeros((64, 64, 4), dtype=np.float32)
    canvas[:h, :w, :] = patch[:64, :64, :]
    return canvas[:, :, channel_indices].mean(axis=-1)


def find_correct_pet(model, mean, std, test_df, patch_dir, class_to_idx, channels, n=N_EXAMPLES):
    true_idx = class_to_idx["PET"]
    sub = test_df[test_df["class"] == "PET"]
    out = []
    for _, row in sub.iterrows():
        if len(out) >= n:
            break
        ds = m.PatchDataset(pd.DataFrame([row]), patch_dir, class_to_idx, channels=channels,
                             augment=False, mean=mean, std=std)
        x, label = ds[0]
        pred, cam = compute_cam(model, x.unsqueeze(0))
        if pred == true_idx:
            out.append((row["patch_file"], cam))
    return out


def save_row_grid(real_items, bg_items, channel_indices, title, out_path, n=N_EXAMPLES):
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for row_idx, (label, items, patch_dir) in enumerate([
        ("real", real_items, REAL_PATCH_DIR), ("background-only", bg_items, BG_PATCH_DIR)
    ]):
        for j in range(n):
            ax = axes[row_idx, j]
            if j < len(items):
                patch_file, cam = items[j]
                disp = raw_display_image(patch_dir, patch_file, channel_indices)
                ax.imshow(disp, cmap="gray")
                ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.axis("off")
        fig.text(0.005, 1 - (row_idx + 0.5) / 2, label, fontsize=11, fontweight="bold", va="center")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0.05, 0, 1, 0.93])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def run_variant(name, channels, in_ch, channel_indices, real_ckpt, bg_ckpt, classes):
    print(f"\n=== {name} ===")
    real_model, real_mean, real_std, real_test, real_c2i = load_model_and_stats(
        REAL_MANIFEST, "split_video_holdout", REAL_PATCH_DIR, real_ckpt, classes, channels, in_ch)
    bg_model, bg_mean, bg_std, bg_test, bg_c2i = load_model_and_stats(
        BG_MANIFEST, "split", BG_PATCH_DIR, bg_ckpt, classes, channels, in_ch)

    real_items = find_correct_pet(real_model, real_mean, real_std, real_test, REAL_PATCH_DIR, real_c2i, channels)
    bg_items = find_correct_pet(bg_model, bg_mean, bg_std, bg_test, BG_PATCH_DIR, bg_c2i, channels)
    print(f"  real correct PET examples found: {len(real_items)}")
    print(f"  background-only correct PET examples found: {len(bg_items)}")

    out_path = os.path.join(RESULTS_DIR, f"gradcam_pet_{name}.png")
    save_row_grid(real_items, bg_items, channel_indices, f"CAM on correctly-classified PET patches - {name}", out_path)
    return real_items, bg_items


def main():
    classes = sorted(pd.read_csv(BG_MANIFEST)["class"].unique())
    print("classes:", classes)

    run_variant("i0", "i0_bgcorr_vnorm", 1, [0],
                os.path.join(RESULTS_DIR, "real_vh_i0bv_s0"), os.path.join(RESULTS_DIR, "bg_i0bv_s0"), classes)
    run_variant("i0i90", "i0i90_bgcorr_vnorm", 2, [0, 2],
                os.path.join(RESULTS_DIR, "real_vh_i0i90bv_s0"), os.path.join(RESULTS_DIR, "bg_i0i90bv_s0"), classes)


if __name__ == "__main__":
    main()
