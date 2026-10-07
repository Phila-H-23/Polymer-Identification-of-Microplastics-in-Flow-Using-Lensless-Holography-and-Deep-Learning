"""
Baseline classifier - pretrained ResNet18 adapted for 4-channel polarisation patches
=====================================================================================

WHAT THIS DOES: trains the generic transfer-learned baseline (Table 2.2 of the
literature review). Its job is to prove the pipeline produces something
learnable and to give the compact domain-tuned CNN a number to beat. It is
NOT the architecture the thesis argues for.

Changes from the first version:
  1. The new 4-channel conv1 is initialised from the mean of the pretrained
     RGB filters (copied into all four channels) rather than random weights.
  2. The whole network is fine-tuned by default (--freeze restores the
     frozen feature-extractor variant, i.e. the Kim et al. setup).
  3. Polarisation-aware augmentation: a 90 degree rotation of the patch
     permutes the channels (I0<->I90, I45<->I135) per Montandon & Nicolls
     Eq. 13. Rotating pixels without permuting channels would teach the
     network that polarisation is noise.
  4. Model selection and reporting use macro-F1 and per-class recall, not
     accuracy or loss, because of the 12x class imbalance.
  5. --channels intensity trains the same network on the mean of the four
     channels: the intensity-only control (Lopera et al.) that shows
     whether polarisation, not just fringe structure, carries the signal.

Usage:
  python train_baseline_resnet.py                                  # particle split, 4 channels
  python train_baseline_resnet.py --split video_holdout
  python train_baseline_resnet.py --channels intensity
  python train_baseline_resnet.py --freeze                         # frozen-backbone variant
  python train_baseline_resnet.py --seed 1

Expects, in the same folder:
  - pipeline_output/manifest.csv (with 'split' AND 'split_video_holdout' columns)
  - pipeline_output/class_weights.json               (particle mode)
  - pipeline_output/class_weights_video_holdout.json (video_holdout mode)
  - pipeline_output/patches/*.npy   (H, W, 4) uint8 arrays, channel order I0/I45/I90/I135

CHANNEL ORDER: the rotation permutation assumes [I0, I45, I90, I135]. If
holo_pipeline_starter.py stores them differently, edit ROT90_PERM.
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MANIFEST_PATH = "pipeline_output/manifest.csv"
PATCH_DIR = "pipeline_output/patches"
CROP_SIZE = 64  # must match what holo_pipeline_starter.py actually cropped
PATCH_NORM_BORDER = 8  # outer border width used by --patch-norm, matches scripts/stokes_diagnostic.py's BORDER_PX
FLATFIELD_REF_PATH = "pipeline_output/flatfield_ref.npy"  # built by scripts/build_flatfield_ref.py
FLATFIELD_HALF = 32  # matches CROP_SIZE//2 - the crop half-width used when patches were originally cropped
VIDEO_BG_STATS_PATH = "pipeline_output/video_background_stats.json"  # built by the per-video-vnorm diagnostic
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLIT_CONFIG = {
    "particle": {
        "split_column": "split",
        "class_weights_path": "pipeline_output/class_weights.json",
    },
    "video_holdout": {
        "split_column": "split_video_holdout",
        "class_weights_path": "pipeline_output/class_weights_video_holdout.json",
    },
}

# 90 degree rotation: sin(2θ) -> -sin(2θ), cos(2θ) -> -cos(2θ) in Eq. 13,
# so I0<->I90 and I45<->I135. 180 degrees leaves channels unchanged.
ROT90_PERM = [2, 3, 0, 1]
# Mirror flip (own derivation from Eq. 13, handedness reversal included):
# swaps I45<->I135 only. Off by default until confirmed with supervisor.
FLIP_PERM = [0, 3, 2, 1]
USE_FLIPS = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset - loads patches lazily, one at a time
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, manifest_df, patch_dir, class_to_idx, channels="pol4",
                 augment=False, mean=None, std=None, patch_norm=False):
        self.df = manifest_df.reset_index(drop=True)
        self.patch_dir = patch_dir
        self.class_to_idx = class_to_idx
        self.channels = channels
        self.augment = augment
        self.mean, self.std = mean, std
        self.patch_norm = patch_norm

    def __len__(self):
        return len(self.df)

    def _load(self, idx):
        row = self.df.iloc[idx]
        patch = np.load(os.path.join(self.patch_dir, row["patch_file"])).astype(np.float32) / 255.0
        h, w = patch.shape[:2]
        if self.patch_norm:
            patch = apply_patch_norm(patch[:CROP_SIZE, :CROP_SIZE, :])
            h, w = patch.shape[:2]
        elif self.channels in ("pol4_ff", "intensity_ff"):
            patch = apply_pol4_ff(patch[:CROP_SIZE, :CROP_SIZE, :], int(row["x"]), int(row["y"]))
            h, w = patch.shape[:2]
        elif self.channels == "pol4_bgcorr_vnorm":
            patch = apply_vnorm(patch[:CROP_SIZE, :CROP_SIZE, :], row["source_file"])
            h, w = patch.shape[:2]
        # pad/crop to exactly CROP_SIZE (edge-of-frame patches can be smaller)
        canvas = np.zeros((CROP_SIZE, CROP_SIZE, 4), dtype=np.float32)
        canvas[:h, :w, :] = patch[:CROP_SIZE, :CROP_SIZE, :]
        return canvas, self.class_to_idx[row["class"]]

    def _augment(self, x):
        # x: (4, H, W). Geometric ops carry their channel permutation.
        k = random.randint(0, 3)
        if k:
            x = torch.rot90(x, k, dims=(1, 2))
            if k % 2 == 1:
                x = x[ROT90_PERM]
        if USE_FLIPS and random.random() < 0.5:
            x = torch.flip(x, dims=(2,))[FLIP_PERM]
        dx, dy = random.randint(-4, 4), random.randint(-4, 4)  # detector centring jitter
        x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
        x = x * (1.0 + 0.1 * (random.random() - 0.5))  # common gain keeps birefringence ratios
        x = x + 0.01 * torch.randn_like(x)             # sensor noise
        return x

    def __getitem__(self, idx):
        canvas, label = self._load(idx)
        x = torch.from_numpy(canvas).permute(2, 0, 1)  # (4, H, W)
        if self.augment:
            x = self._augment(x)
        if self.channels in ("intensity", "intensity_ff"):
            x = x.mean(dim=0, keepdim=True)  # (1, H, W): polarisation removed
        elif self.channels == "stokes":
            x = to_stokes(x)  # (3, H, W): [S, q, u] - see to_stokes() for formulas
        if self.mean is not None:
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return x, label


def to_stokes(x):
    """(4, H, W) [0,1]-scaled tensor, channel order [I0, I45, I90, I135] ->
    (3, H, W) [S, q, u].

    Normalised by S (+0.05 floor) rather than by the raw channel sums: the
    original (I0+I90+1e-6) denominator blows up on the zero-padded border
    of undersized patches once augmentation noise perturbs it away from
    exact zero (observed |q| up to ~1e3 on such pixels), which poisons the
    whole 3-channel input even though S alone is well-behaved. Padding/dark
    fringes (S < 0.02) are forced to q=u=0 instead of being left to blow up.
    """
    I0, I45, I90, I135 = x[0], x[1], x[2], x[3]
    S = (I0 + I45 + I90 + I135) / 4.0
    denom = S + 0.05
    q = ((I0 - I90) / denom).clamp(-1.0, 1.0)
    u = ((I45 - I135) / denom).clamp(-1.0, 1.0)
    dark = S < 0.02
    q = torch.where(dark, torch.zeros_like(q), q)
    u = torch.where(dark, torch.zeros_like(u), u)
    return torch.stack([S, q, u], dim=0)


def apply_patch_norm(patch):
    """patch: (h, w, 4) float32, already [0,1]-scaled, NOT YET zero-padded
    to CROP_SIZE - this only ever sees real pixels, so "excluding
    zero-padding" is automatic (padding is added afterwards in _load).

    Subtracts each channel's own mean over the outer PATCH_NORM_BORDER-px
    border of THIS patch's actual (h, w), then divides the whole
    (post-subtraction) patch by its own overall std - removes per-patch
    exposure/offset so the network can't just key off brightness."""
    h, w = patch.shape[:2]
    bp = min(PATCH_NORM_BORDER, h // 2, w // 2)
    border = np.zeros((h, w), dtype=bool)
    if bp > 0:
        border[:bp, :] = True
        border[-bp:, :] = True
        border[:, :bp] = True
        border[:, -bp:] = True
    else:
        border[:, :] = True  # too small to have a distinct border - use whole patch
    out = patch.copy()
    for c in range(patch.shape[2]):
        out[:, :, c] -= patch[:, :, c][border].mean()
    out /= (out.std() + 1e-6)
    return out


_flatfield_ref_cache = None


def get_flatfield_ref():
    global _flatfield_ref_cache
    if _flatfield_ref_cache is None:
        _flatfield_ref_cache = np.load(FLATFIELD_REF_PATH)
    return _flatfield_ref_cache


def apply_pol4_ff(patch, cx, cy):
    """patch: (h, w, 4) float32, already [0,1]-scaled, NOT YET zero-padded.
    cx, cy: the patch's centroid position in its source (1024x1224)
    demosaiced frame, from the manifest's x,y columns - the same
    coordinate system and the same half=32 crop formula used when the
    patch itself was cropped (holo_pipeline_starter.py Step 8 /
    scripts/build_background_dataset.py), so the reference crop lines up
    pixel-for-pixel with the patch.

    Divides each channel by the matching flat-field reference crop
    (built by scripts/build_flatfield_ref.py from frames_bg_30.mat),
    rescales so the four-channel mean is unchanged (removes the fixed
    I0/I90-type imbalance while preserving overall patch brightness),
    clamps to [0, 4]."""
    ref = get_flatfield_ref()
    ref_h, ref_w = ref.shape[:2]
    h, w = patch.shape[:2]
    y0, y1 = max(0, cy - FLATFIELD_HALF), min(ref_h, cy + FLATFIELD_HALF)
    x0, x1 = max(0, cx - FLATFIELD_HALF), min(ref_w, cx + FLATFIELD_HALF)
    ref_crop = ref[y0:y1, x0:x1, :]
    if ref_crop.shape[:2] != (h, w):
        ref_crop = ref_crop[:h, :w, :]
    corrected = patch / (ref_crop + 1e-6)
    mean_before = patch.mean()
    mean_after = corrected.mean()
    if mean_after > 1e-8:
        corrected = corrected * (mean_before / mean_after)
    return np.clip(corrected, 0.0, 4.0)


_video_bg_stats_cache = None


def get_video_bg_stats():
    global _video_bg_stats_cache
    if _video_bg_stats_cache is None:
        with open(VIDEO_BG_STATS_PATH) as f:
            _video_bg_stats_cache = json.load(f)
    return _video_bg_stats_cache


def apply_vnorm(patch, source_file):
    """patch: (h, w, 4) float32, already [0,1]-scaled, NOT YET zero-padded.
    source_file: e.g. "PE1_600.mat" - the video this patch (real or
    background-only) came from.

    Subtracts and divides by that SAME video's BACKGROUND-ONLY mean/std
    (pipeline_output/video_background_stats.json, pooled over all 4
    channels and all pixels of that video's background-only pol4_bgcorr
    patches) - applied identically whether this specific patch is real
    or background-only, always using the background-derived stats for
    both."""
    stats = get_video_bg_stats()
    s = stats[source_file]
    return (patch - s["mean"]) / (s["std"] + 1e-6)


def compute_channel_stats(df, class_to_idx, channels, patch_norm=False):
    """Per-channel mean/std over the TRAINING set only, no augmentation."""
    ds = PatchDataset(df, PATCH_DIR, class_to_idx, channels=channels, patch_norm=patch_norm)
    n_ch = {"intensity": 1, "intensity_ff": 1, "stokes": 3}.get(channels, 4)
    acc, acc2 = torch.zeros(n_ch), torch.zeros(n_ch)
    for i in range(len(ds)):
        x, _ = ds[i]
        acc += x.mean(dim=(1, 2))
        acc2 += (x ** 2).mean(dim=(1, 2))
    mean = acc / len(ds)
    std = (acc2 / len(ds) - mean ** 2).clamp_min(1e-8).sqrt()
    return mean, std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(n_classes, in_ch=4, freeze=False, upsample_to=None):
    """ResNet18 pretrained on ImageNet, conv1 widened to in_ch channels.

    The new conv1 filters are the mean of the pretrained RGB filters, copied
    into every input channel, so the layer keeps its pretrained edge/texture
    detectors instead of starting from random weights.
    """
    model = models.resnet18(weights="IMAGENET1K_V1")
    old = model.conv1
    new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    with torch.no_grad():
        mean_filter = old.weight.mean(dim=1, keepdim=True)          # (64, 1, 7, 7)
        new.weight.copy_(mean_filter.repeat(1, in_ch, 1, 1))        # (64, in_ch, 7, 7)
    model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    if freeze:  # frozen feature extractor: only conv1 and fc train
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("conv1") or name.startswith("fc")

    if upsample_to:  # optional: 64x64 collapses to 2x2 by layer4; upsampling helps ResNet
        model = nn.Sequential(nn.Upsample(size=(upsample_to, upsample_to), mode="bilinear",
                                          align_corners=False), model)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    return total_loss / len(loader.dataset), preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT_CONFIG), default="particle")
    ap.add_argument("--channels",
                     choices=["pol4", "pol4_ff", "pol4_bgcorr_vnorm", "intensity", "intensity_ff", "stokes"],
                     default="pol4")
    ap.add_argument("--freeze", action="store_true", help="frozen backbone variant")
    ap.add_argument("--upsample", type=int, default=0, help="e.g. 128; 0 = off")
    ap.add_argument("--patch-norm", action="store_true",
                     help="per-patch: subtract each channel's own border mean, divide by overall std "
                          "(removes per-patch exposure/offset; applied before augmentation)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = SPLIT_CONFIG[args.split]
    tag = f"resnet18_{args.split}_{args.channels}{'_frozen' if args.freeze else ''}_s{args.seed}"
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== {tag} on {DEVICE} ===")

    manifest = pd.read_csv(MANIFEST_PATH)
    split_col = cfg["split_column"]
    if split_col not in manifest.columns:
        raise ValueError(f"manifest.csv has no '{split_col}' column - run the split step first.")

    classes = sorted(manifest["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(cfg["class_weights_path"]) as f:
        raw_w = json.load(f)
    weight_tensor = torch.tensor([raw_w[c] for c in classes], dtype=torch.float32).to(DEVICE)
    print(f"Classes: {classes}\nWeights: {[round(w, 3) for w in weight_tensor.tolist()]}")

    train_df = manifest[manifest[split_col] == "train"]
    val_df = manifest[manifest[split_col] == "val"]
    test_df = manifest[manifest[split_col] == "test"]
    print(f"Train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    if args.split == "video_holdout":
        tr_n, te_n = train_df.groupby("class").size(), test_df.groupby("class").size()
        for c in classes:
            tr, te = tr_n.get(c, 0), te_n.get(c, 0)
            print(f"  {c:6s}: train={tr:4d} test={te:4d}{'  <-- test > train' if te > tr else ''}")

    mean, std = compute_channel_stats(train_df, class_to_idx, args.channels, patch_norm=args.patch_norm)
    print("Train channel mean:", mean.numpy().round(4), "std:", std.numpy().round(4))

    def mk(df, aug):
        return PatchDataset(df, PATCH_DIR, class_to_idx, args.channels, aug, mean, std,
                             patch_norm=args.patch_norm)

    train_loader = DataLoader(mk(train_df, True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(mk(val_df, False), batch_size=args.batch_size)
    test_loader = DataLoader(mk(test_df, False), batch_size=args.batch_size)

    in_ch = {"intensity": 1, "intensity_ff": 1, "stokes": 3}.get(args.channels, 4)
    model = build_model(len(classes), in_ch, args.freeze, args.upsample or None).to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt = os.path.join(out_dir, "best.pt")
    best_f1, best_epoch, bad, history = -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_p, tr_l = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_p, va_l = run_epoch(model, val_loader, criterion)
        scheduler.step()
        tr_f1, va_f1 = f1_score(tr_l, tr_p, average="macro"), f1_score(va_l, va_p, average="macro")
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                            train_macro_f1=tr_f1, val_macro_f1=va_f1))
        print(f"ep {epoch:3d} | train loss {tr_loss:.3f} f1 {tr_f1:.3f} | val loss {va_loss:.3f} f1 {va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1, best_epoch, bad = va_f1, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at {epoch} (best val f1 {best_f1:.3f} @ epoch {best_epoch})")
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    _, te_p, te_l = run_epoch(model, test_loader, criterion)
    report_txt = classification_report(te_l, te_p, target_names=classes, digits=3, zero_division=0)
    report = classification_report(te_l, te_p, target_names=classes, digits=3, zero_division=0,
                                   output_dict=True)
    cm = confusion_matrix(te_l, te_p)
    print(f"\n=== TEST ({tag}, best val epoch {best_epoch}) ===")
    print(report_txt)
    print("Confusion matrix (rows=true, cols=pred), order:", classes)
    print(cm)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(dict(args=vars(args), classes=classes, trainable_params=n_train,
                       best_epoch=best_epoch, best_val_macro_f1=best_f1,
                       test_macro_f1=report["macro avg"]["f1-score"],
                       test_report=report, confusion_matrix=cm.tolist(), history=history),
                  f, indent=2)
    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        f.write(f"{tag}\nTrain/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}\n\n{report_txt}\n")
        f.write(f"Confusion matrix, order {classes}\n{np.array2string(cm)}\n")
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()