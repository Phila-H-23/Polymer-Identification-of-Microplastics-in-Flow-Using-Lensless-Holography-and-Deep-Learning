"""train_compact_cnn_bright.py: copy of train_compact_cnn.py (2026-10-08) with two flags, both off by default:
  --gain-range G  per-sample multiplicative gain uniform in [1 - G, 1 + G] on the whole training patch, in place
                  of the +/-5% gain, applied before the noise (same single random draw, so with the flag off the
                  random stream is unchanged)
  --meannorm      after all augmentation, each patch (every split, and the training channel statistics) is
                  divided by its own mean, so every patch has mean 1 before the channel normalisation
With both flags off the outputs are bit-identical to train_compact_cnn.py (scripts/test_bright_flags_exp.py).

train_compact_cnn.py

Compact domain-tuned CNN for reconstruction-free polymer classification from
raw 4-channel polarisation-mosaic hologram patches. This is the PRIMARY
architecture (CompNet / HC-CNN tradition): small, built for fringe statistics,
takes all four polarisation channels natively.

Expects, in the same folder as train_baseline_resnet.py:
  pipeline_output/manifest.csv                        columns: patch_file, class,
                                                        split, split_video_holdout
  pipeline_output/class_weights.json                   class name -> weight (particle split)
  pipeline_output/class_weights_video_holdout.json     class name -> weight (video_holdout split)
  pipeline_output/patches/*.npy        (H, W, 4) arrays, channel order 0/45/90/135

Usage:
  python train_compact_cnn.py                               # 4 polarisation channels
  python train_compact_cnn.py --channels intensity          # intensity-only control
  python train_compact_cnn.py --channels stokes             # [S, q, u] Stokes-derived control
  python train_compact_cnn.py --split video_holdout         # video-holdout split
  python train_compact_cnn.py --seed 1 --out results/cnn_s1

CHANNEL ORDER: this script assumes the pipeline stores channels as
[I0, I45, I90, I135]. If holo_pipeline_starter.py uses a different order,
edit CHANNEL_ORDER below - the augmentation permutations depend on it.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MANIFEST_PATH = "pipeline_output/manifest.csv"
PATCH_DIR = "pipeline_output/patches"
CROP_SIZE = 64
PATCH_NORM_BORDER = 8  # outer border width used by --patch-norm, matches scripts/stokes_diagnostic.py's BORDER_PX
FLATFIELD_REF_PATH = "pipeline_output/flatfield_ref.npy"  # built by scripts/build_flatfield_ref.py
FLATFIELD_HALF = 32  # matches CROP_SIZE//2 - the crop half-width used when patches were originally cropped
VIDEO_BG_STATS_PATH = "pipeline_output/video_background_stats.json"  # built by the per-video-vnorm diagnostic
RECON_CHANNELS = {"recon_s0": 1, "recon_biref": 3}  # channel count for the reconstruction pathway's pre-built patches
HALF_CROP = 32  # CROP_SIZE // 2, the crop rule used by every patch build
AUG_DISC_R = 12  # --aug: kept central disc radius, px (24 px = 166 um)
AUG_FEATHER_SIGMA = 2.0  # --aug: Gaussian feathering of the disc edge, px
CHANNEL_ORDER = ["I0", "I45", "I90", "I135"]  # verify against the pipeline

# Mirrors train_baseline_resnet.py's SPLIT_CONFIG exactly - same split
# columns, same class-weight files - so the two scripts never disagree
# about which weights belong to which split.
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

# Rotation augmentation is 0 or 180 degrees only, for every mode. A 180 deg object
# rotation leaves the slow axis unchanged (it is defined mod 180 deg), so every
# channel keeps its identity. 90 deg rotations would need I0<->I90 / I45<->I135
# swaps, which are only valid under symmetric illumination; here I90 ~4.5x I0.
# No flips: a mirror reverses the azimuth and would need the same kind of swap.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    """Lazy-loading patch dataset with optional polarisation-aware augmentation.

    channels="pol4"        -> 4 channels (I0, I45, I90, I135)
    channels="pol4_ff"     -> 4 channels, flat-field corrected (see apply_pol4_ff())
    channels="pol4_bgcorr" -> 4 channels, no runtime transform - behaves identically to
                              "pol4" here; the background-subtraction correction (90-frame
                              rolling background, scripts/build_bgcorrected_dataset.py) is
                              baked into the patch files themselves, selected via
                              --manifest/--patch-dir pointing at
                              pipeline_output/manifest_bgcorrected.csv +
                              pipeline_output/patches_bgcorrected/. This value only exists
                              so results.json's args/tag record which input was used.
    channels="intensity"   -> 1 channel, mean of the four (polarisation removed)
    channels="stokes"      -> 3 channels [S, q, u] (see to_stokes())
    channels="stokes_bgcorr_vnorm" -> [S, q, u] from bg-corrected patches, then S only is
                              normalised by that video's background stats (as apply_vnorm).
                              vnorm cannot precede Stokes: it centres intensities near 0,
                              so S ~ 0 and q, u blow up. q, u are ratios and stay as is.
    channels="recon_s0"    -> 1 channel, no runtime transform - like "pol4_bgcorr", the
                              angular-spectrum back-propagation (scripts/reconstruction.py)
                              is baked into the patch files at data-build time (full-frame
                              propagation per channel, then crop - never done per-sample
                              here), selected via --manifest/--patch-dir pointing at the
                              reconstructed patch set. Channel content: mean of the four
                              propagated intensities (I0+I45+I90+I135)/4 - the SAME
                              combination formula the raw "intensity" mode uses (see
                              above), applied to the propagated instead of raw channels,
                              per explicit instruction. NOT I0+I90 only.
    channels="recon_biref" -> 3 channels [eta, eta*cos(2*pi*theta), eta*sin(2*pi*theta)],
                              computed at LOAD time (prop_to_biref) from the 6-channel
                              recon_prop patches [I0p, I45p, I90p, I135p, A_bg, B_bg];
                              --manifest/--patch-dir point at manifest_recon_prop.csv /
                              patches_recon_prop. NaN set to 0 after the products.
    """

    def __init__(self, df, patch_dir, class_to_idx, channels="pol4",
                 augment=False, mean=None, std=None, patch_norm=False,
                 aug_mode=None, donor_df=None, donor_dir=None, interior_mean=None, gain_range=None, meannorm=False):
        self.gain_range = gain_range  # None: the +/-5% gain of train_compact_cnn.py
        self.meannorm = meannorm
        self.df = df.reset_index(drop=True)
        self.patch_dir = patch_dir
        self.class_to_idx = class_to_idx
        self.channels = channels
        self.augment = augment
        self.mean = mean  # per-channel, shape (C,)
        self.std = std
        self.patch_norm = patch_norm
        # Optional surround handling (off unless aug_mode is set). bg / bgtex: training-patch augmentation
        # (applied only when augment=True). interior: input representation, applied to every split.
        self.aug_mode = aug_mode
        self._masks = {}
        if aug_mode in ("bg", "bgtex"):
            self.donor_ds = PatchDataset(donor_df, donor_dir, class_to_idx, channels=channels, patch_norm=patch_norm)
            vids = self.donor_ds.df["source_file"].values
            self._donors_excluding = {v: np.flatnonzero(vids != v) for v in set(vids) | set(self.df["source_file"])}
        elif aug_mode == "interior":
            self.interior_mean = (interior_mean if interior_mean is not None else
                                  background_mean(donor_df, donor_dir, class_to_idx, channels, patch_norm))

    def __len__(self):
        return len(self.df)

    def _load(self, idx):
        row = self.df.iloc[idx]
        if self.channels in RECON_CHANNELS:
            # Reconstructed patches are saved as already-scaled float32
            # arrays (physical intensity / eta / theta values, NOT a
            # uint8 image) - no /255, and channel count is 1 or 3, not 4.
            patch = np.load(os.path.join(self.patch_dir, row["patch_file"])).astype(np.float32)
            if self.channels == "recon_biref":
                patch = prop_to_biref(patch)
            n_ch = RECON_CHANNELS[self.channels]
        else:
            patch = np.load(os.path.join(self.patch_dir, row["patch_file"])).astype(np.float32) / 255.0
            n_ch = 4
        h, w = patch.shape[:2]
        if self.patch_norm:
            patch = apply_patch_norm(patch[:CROP_SIZE, :CROP_SIZE, :])
            h, w = patch.shape[:2]
        elif self.channels in ("pol4_ff", "intensity_ff"):
            patch = apply_pol4_ff(patch[:CROP_SIZE, :CROP_SIZE, :], int(row["x"]), int(row["y"]))
            h, w = patch.shape[:2]
        elif self.channels in ("pol4_bgcorr_vnorm", "intensity_bgcorr_vnorm", "i0_bgcorr_vnorm",
                                "i0i90_bgcorr_vnorm", "i45_bgcorr_vnorm"):
            patch = apply_vnorm(patch[:CROP_SIZE, :CROP_SIZE, :], row["source_file"])
            h, w = patch.shape[:2]
        canvas = np.zeros((CROP_SIZE, CROP_SIZE, n_ch), dtype=np.float32)
        canvas[:h, :w, :] = patch[:CROP_SIZE, :CROP_SIZE, :]
        return canvas, self.class_to_idx[row["class"]]

    def _rotation_k(self):
        return 2 * random.randint(0, 1)  # 0 or 180 deg only, see the rotation note near the top

    def _augment(self, x):
        # x: (C, H, W) tensor. 180 deg rotation keeps every channel's identity.
        k = self._rotation_k()
        if k:
            x = torch.rot90(x, k, dims=(1, 2))
        # Small translation (particle not perfectly centred by the detector)
        dx, dy = random.randint(-4, 4), random.randint(-4, 4)
        x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
        # Illumination jitter: same gain on all channels preserves birefringence
        # ratios; tiny per-channel noise models sensor noise.
        if self.gain_range is None:
            x = x * (1.0 + 0.1 * (random.random() - 0.5))
        else:  # uniform in [1 - G, 1 + G], same single random draw
            x = x * (1.0 - self.gain_range + 2.0 * self.gain_range * random.random())
        x = x + 0.01 * torch.randn_like(x)
        return x

    def _disc_mask(self, cy, cx):
        """Disc of radius AUG_DISC_R px at (cy, cx), edge feathered by a Gaussian of AUG_FEATHER_SIGMA px."""
        if (cy, cx) not in self._masks:
            yy, xx = np.mgrid[:CROP_SIZE, :CROP_SIZE]
            disc = ((yy - cy) ** 2 + (xx - cx) ** 2 <= AUG_DISC_R ** 2).astype(np.float32)
            self._masks[(cy, cx)] = gaussian_filter(disc, AUG_FEATHER_SIGMA)[..., None]
        return self._masks[(cy, cx)]

    def _surround_aug(self, canvas, idx):
        """Keep the disc at the particle position, replace the surround. The particle sits at the
        detection centre: (32, 32) for unclipped patches, (y - max(0, y-32), x - max(0, x-32)) for
        patches clipped at the frame edge (same crop rule as every patch build)."""
        row = self.df.iloc[idx]
        x, y = int(row["x"]), int(row["y"])
        m = self._disc_mask(y - max(0, y - HALF_CROP), x - max(0, x - HALF_CROP))
        if self.aug_mode == "interior":
            return (m * canvas + (1 - m) * self.interior_mean).astype(np.float32)
        j = int(random.choice(self._donors_excluding[row["source_file"]]))
        canvas = m * canvas + (1 - m) * self.donor_ds._load(j)[0]
        if self.aug_mode == "bgtex":
            if random.random() < 0.5:
                s = random.uniform(0.0, 1.5)
                canvas = gaussian_filter(canvas, (s, s, 0))
            if random.random() < 0.5:
                canvas = canvas + np.random.normal(0.0, random.uniform(0.0, 0.1) * canvas.std(), canvas.shape)
        return canvas.astype(np.float32)

    def __getitem__(self, idx):
        canvas, label = self._load(idx)
        if self.aug_mode == "interior" or (self.augment and self.aug_mode):
            canvas = self._surround_aug(canvas, idx)  # before the 0/180 rotation
        x = torch.from_numpy(canvas).permute(2, 0, 1)  # (4, H, W)
        if self.augment:
            x = self._augment(x)
        if self.channels in ("intensity", "intensity_ff", "intensity_bgcorr_vnorm"):
            x = x.mean(dim=0, keepdim=True)  # (1, H, W)
        elif self.channels == "i0_bgcorr_vnorm":
            x = x[0:1]  # (1, H, W): I0 only
        elif self.channels == "i0i90_bgcorr_vnorm":
            x = x[[0, 2]]  # (2, H, W): I0, I90 only
        elif self.channels == "i45_bgcorr_vnorm":
            x = x[1:2]  # (1, H, W): I45 only
        elif self.channels == "stokes":
            x = to_stokes(x)  # (3, H, W): [S, q, u]
        elif self.channels == "stokes_bgcorr_vnorm":
            x = to_stokes(x)
            s = get_video_bg_stats()[self.df.iloc[idx]["source_file"]]
            pad = x[0] < 0.02  # zero-padded border of undersized patches, same mask to_stokes uses
            x[0] = torch.where(pad, torch.zeros_like(x[0]), (x[0] - s["mean"]) / (s["std"] + 1e-6))
        if self.meannorm:  # after all augmentation: patch mean exactly 1
            x = x / x.mean()
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
    both, per the diagnostic's design: isolates whether a stable per-
    video brightness/spread offset (not texture) explains the earlier
    pol4_bgcorr confound."""
    stats = get_video_bg_stats()
    s = stats[source_file]
    return (patch - s["mean"]) / (s["std"] + 1e-6)


_biref_fn = None


def prop_to_biref(prop):
    """(h, w, 6) [I0p, I45p, I90p, I135p, A_bg, B_bg] -> (h, w, 3)
    [eta, eta*cos(2*pi*theta), eta*sin(2*pi*theta)], NaN -> 0. Uses the same
    birefringence_eta_theta (port of birefringence_eta_theta.m + A_bg/B_bg
    subtraction) as scripts/biref_eta_check.py."""
    global _biref_fn
    if _biref_fn is None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
        from reconstruction import birefringence_eta_theta
        _biref_fn = birefringence_eta_theta
    eta, theta = _biref_fn(*(prop[..., c] for c in range(4)), A_bg=prop[..., 4], B_bg=prop[..., 5])
    out = np.stack([eta, eta * np.cos(2 * np.pi * theta), eta * np.sin(2 * np.pi * theta)], axis=-1)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def background_mean(donor_df, donor_dir, class_to_idx, channels, patch_norm=False):
    """Pixel-wise mean of the given background patches in the channel mode's own domain (INTERIOR fill)."""
    ds = PatchDataset(donor_df, donor_dir, class_to_idx, channels=channels, patch_norm=patch_norm)
    return np.mean([ds._load(j)[0] for j in range(len(ds))], axis=0)


def compute_channel_stats(df, patch_dir, class_to_idx, channels, patch_norm=False, **ds_kwargs):
    """Per-channel mean/std over the TRAINING set only (no augmentation). ds_kwargs (optional) select an
    input representation such as aug_mode='interior'."""
    ds = PatchDataset(df, patch_dir, class_to_idx, channels=channels, patch_norm=patch_norm, **ds_kwargs)
    acc = torch.zeros({"intensity": 1, "intensity_ff": 1, "stokes": 3, "stokes_bgcorr_vnorm": 3,
                        "intensity_bgcorr_vnorm": 1, "i0_bgcorr_vnorm": 1,
                        "i0i90_bgcorr_vnorm": 2, "i45_bgcorr_vnorm": 1,
                        **RECON_CHANNELS}.get(channels, 4))
    acc2 = torch.zeros_like(acc)
    n = 0
    for i in range(len(ds)):
        x, _ = ds[i]
        acc += x.mean(dim=(1, 2))
        acc2 += (x ** 2).mean(dim=(1, 2))
        n += 1
    mean = acc / n
    std = (acc2 / n - mean ** 2).clamp_min(1e-8).sqrt()
    return mean, std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def conv_bn(in_ch, out_ch, k=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CompactHoloCNN(nn.Module):
    """Small VGG-style network for 64x64 hologram patches.

    Design choices (defend these in Chapter 4):
    - 7x7 stem at full resolution: diffraction rings are an extended, not
      local, structure; a wide first receptive field sees more of a ring.
    - No pretraining: fringe statistics differ from natural images.
    - BatchNorm + global average pooling + dropout: regularisation for a
      small (883 patch), imbalanced dataset.
    - ~0.6M parameters at width=32, about 20x smaller than ResNet18.
    """

    def __init__(self, in_ch=4, n_classes=7, width=32, dropout=0.3):
        super().__init__()
        w = width
        self.stem = conv_bn(in_ch, w, k=7)
        self.block1 = nn.Sequential(conv_bn(w, w), conv_bn(w, w), nn.MaxPool2d(2))          # 64->32
        self.block2 = nn.Sequential(conv_bn(w, 2 * w), conv_bn(2 * w, 2 * w), nn.MaxPool2d(2))  # 32->16
        self.block3 = nn.Sequential(conv_bn(2 * w, 4 * w), conv_bn(4 * w, 4 * w), nn.MaxPool2d(2))  # 16->8
        self.block4 = nn.Sequential(conv_bn(4 * w, 4 * w), conv_bn(4 * w, 4 * w))           # 8x8
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(4 * w, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)


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
    ap.add_argument("--channels",
                     choices=["pol4", "pol4_ff", "pol4_bgcorr", "pol4_bgcorr_vnorm",
                              "intensity", "intensity_ff", "intensity_bgcorr_vnorm",
                              "i0_bgcorr_vnorm", "i0i90_bgcorr_vnorm", "i45_bgcorr_vnorm",
                              "recon_s0", "recon_biref", "stokes", "stokes_bgcorr_vnorm"],
                     default="pol4")
    ap.add_argument("--split", choices=list(SPLIT_CONFIG), default="particle")
    ap.add_argument("--manifest", default=MANIFEST_PATH,
                     help="manifest CSV; must have a 'class' column and whichever split column --split selects")
    ap.add_argument("--patch-dir", default=PATCH_DIR)
    ap.add_argument("--class-weights", default=None,
                     help="class-weight JSON overriding the one tied to --split")
    ap.add_argument("--aug", choices=["bg", "bgtex", "interior"], default=None,
                     help="training-patch surround augmentation (off by default): bg = keep the particle "
                          "disc, surround from a random training background patch of another video; "
                          "bgtex = bg then blur/noise (both training patches only); interior = input representation on every split: surround set to the mean training background")
    ap.add_argument("--donor-manifest", default=None,
                     help="background-only manifest (same domain) whose 'split'=='train' rows are the donors")
    ap.add_argument("--donor-patch-dir", default=None)
    ap.add_argument("--classes", default=None,
                     help="comma-separated subset, e.g. PET,PS: restricts the manifest (and --aug donors), the class "
                          "list, the number of outputs and the class weights to these classes (off by default)")
    ap.add_argument("--patch-norm", action="store_true",
                     help="per-patch: subtract each channel's own border mean, divide by overall std "
                          "(removes per-patch exposure/offset; applied before augmentation)")
    ap.add_argument("--gain-range", type=float, default=None,
                     help="per-sample gain uniform in [1-G, 1+G] instead of +/-5%% (training augmentation; off by default)")
    ap.add_argument("--meannorm", action="store_true",
                     help="divide every patch by its own mean after augmentation, all splits (off by default)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/compact_cnn")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cfg = SPLIT_CONFIG[args.split]
    split_col = cfg["split_column"]

    manifest = pd.read_csv(args.manifest)
    if split_col not in manifest.columns:
        raise ValueError(f"manifest has no '{split_col}' column - run the split step first.")

    subset = args.classes.split(",") if args.classes else None
    if subset:
        missing = sorted(set(subset) - set(manifest["class"]))
        if missing:
            raise ValueError(f"--classes: {missing} not in the manifest")
        manifest = manifest[manifest["class"].isin(subset)]
    classes = sorted(manifest["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(args.class_weights or cfg["class_weights_path"]) as f:
        cw = json.load(f)
    weights = torch.tensor([cw[c] for c in classes], dtype=torch.float32).to(DEVICE)

    splits = {s: manifest[manifest[split_col] == s] for s in ("train", "val", "test")}
    print({s: len(d) for s, d in splits.items()}, "| classes:", classes)

    donors = None
    if args.aug:
        donors = pd.read_csv(args.donor_manifest)
        donors = donors[donors["split"] == "train"]
        if subset:
            donors = donors[donors["class"].isin(subset)]
    rep = {}  # input representation applied to every split (INTERIOR); empty = unchanged behaviour
    if args.aug == "interior":
        rep = dict(aug_mode="interior", interior_mean=background_mean(
            donors, args.donor_patch_dir, class_to_idx, args.channels, args.patch_norm))
        print(f"INTERIOR input representation on train/val/test: disc r {AUG_DISC_R} px kept, surround = mean "
              f"of {len(donors)} training background patches ({donors['source_file'].nunique()} videos)")

    if args.gain_range is not None:
        rep["gain_range"] = args.gain_range  # used only where augment=True (training patches)
    if args.meannorm:
        rep["meannorm"] = True
    mean, std = compute_channel_stats(splits["train"], args.patch_dir, class_to_idx, args.channels,
                                       patch_norm=args.patch_norm, **rep)
    print("train channel mean:", mean.numpy().round(4), "std:", std.numpy().round(4))

    mk = lambda s, aug: PatchDataset(splits[s], args.patch_dir, class_to_idx,
                                     channels=args.channels, augment=aug, mean=mean, std=std,
                                     patch_norm=args.patch_norm, **rep)
    if args.aug in ("bg", "bgtex"):
        train_ds = PatchDataset(splits["train"], args.patch_dir, class_to_idx, channels=args.channels,
                                augment=True, mean=mean, std=std, patch_norm=args.patch_norm,
                                aug_mode=args.aug, donor_df=donors, donor_dir=args.donor_patch_dir)
        print(f"surround augmentation '{args.aug}' on training patches only; {len(donors)} donor background "
              f"patches from {donors['source_file'].nunique()} training videos")
    else:
        train_ds = mk("train", True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(mk("val", False), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(mk("test", False), batch_size=args.batch_size, shuffle=False)

    in_ch = {"intensity": 1, "intensity_ff": 1, "stokes": 3, "stokes_bgcorr_vnorm": 3,
             "intensity_bgcorr_vnorm": 1, "i0_bgcorr_vnorm": 1,
             "i0i90_bgcorr_vnorm": 2, "i45_bgcorr_vnorm": 1,
             **RECON_CHANNELS}.get(args.channels, 4)
    model = CompactHoloCNN(in_ch, len(classes), args.width, args.dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CompactHoloCNN: {n_params/1e6:.2f}M params | device {DEVICE} | input {args.channels}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_epoch, bad_epochs, history = -1.0, 0, 0, []
    ckpt = os.path.join(args.out, "best.pt")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_p, tr_l = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_p, va_l = run_epoch(model, val_loader, criterion)
        scheduler.step()
        tr_f1 = f1_score(tr_l, tr_p, average="macro")
        va_f1 = f1_score(va_l, va_p, average="macro")
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                            train_macro_f1=tr_f1, val_macro_f1=va_f1))
        print(f"ep {epoch:3d} | train loss {tr_loss:.3f} f1 {tr_f1:.3f} | val loss {va_loss:.3f} f1 {va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = va_f1, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch} (best val f1 {best_f1:.3f} @ {best_epoch})")
                break

    # Final evaluation with the best-on-validation checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    _, te_p, te_l = run_epoch(model, test_loader, criterion)
    report = classification_report(te_l, te_p, target_names=classes, digits=3, output_dict=True)
    cm = confusion_matrix(te_l, te_p)
    print("\nTEST (best val epoch", best_epoch, ")")
    print(classification_report(te_l, te_p, target_names=classes, digits=3))
    print("confusion matrix (rows=true, cols=pred):\n", cm)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(args=vars(args), classes=classes, n_params=n_params,
                       best_epoch=best_epoch, best_val_macro_f1=best_f1,
                       test_report=report, confusion_matrix=cm.tolist(),
                       history=history), f, indent=2)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()