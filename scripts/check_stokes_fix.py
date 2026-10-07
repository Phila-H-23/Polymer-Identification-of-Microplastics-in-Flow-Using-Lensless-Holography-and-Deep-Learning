"""Quick numerical check of the fixed to_stokes() (see train_compact_cnn.py /
train_baseline_resnet.py). Loads 20 patches (mixing in undersized/padded ones
on purpose, since those triggered the original blow-up), builds the same
zero-padded [0,1] canvas the training pipeline uses, applies the same
gain+noise augmentation, computes [S, q, u], and prints min/max/mean/std of
each channel. None of q, u should exceed [-1, 1]; their std should be in the
same ballpark as S's (not the ~30x larger std the broken formula produced).
"""
import os
import numpy as np
import pandas as pd
import torch

CROP_SIZE = 64
DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
MANIFEST_PATH = os.path.join(DATA_DIR, "pipeline_output", "manifest.csv")
PATCH_DIR = os.path.join(DATA_DIR, "pipeline_output", "patches")


def to_stokes(x):
    I0, I45, I90, I135 = x[0], x[1], x[2], x[3]
    S = (I0 + I45 + I90 + I135) / 4.0
    denom = S + 0.05
    q = ((I0 - I90) / denom).clamp(-1.0, 1.0)
    u = ((I45 - I135) / denom).clamp(-1.0, 1.0)
    dark = S < 0.02
    q = torch.where(dark, torch.zeros_like(q), q)
    u = torch.where(dark, torch.zeros_like(u), u)
    return torch.stack([S, q, u], dim=0)


def load_canvas(patch_file):
    patch = np.load(os.path.join(PATCH_DIR, patch_file))
    h, w = patch.shape[:2]
    canvas = np.zeros((CROP_SIZE, CROP_SIZE, 4), dtype=np.float32)
    canvas[:h, :w, :] = patch[:CROP_SIZE, :CROP_SIZE, :].astype(np.float32)
    canvas /= 255.0
    return canvas


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    manifest = pd.read_csv(MANIFEST_PATH)

    shapes = manifest["patch_file"].apply(
        lambda f: np.load(os.path.join(PATCH_DIR, f)).shape[:2])
    undersized = manifest[shapes.apply(lambda s: s[0] < CROP_SIZE or s[1] < CROP_SIZE)]

    n_small = min(10, len(undersized))
    small_sample = undersized.sample(n=n_small, random_state=0)
    rest = manifest.drop(small_sample.index)
    normal_sample = rest.sample(n=20 - n_small, random_state=0)
    sample = pd.concat([small_sample, normal_sample])

    print(f"Loaded 20 patches ({n_small} undersized/padded, {20 - n_small} full-size)")

    s_all, q_all, u_all = [], [], []
    for _, row in sample.iterrows():
        canvas = load_canvas(row["patch_file"])
        x = torch.from_numpy(canvas).permute(2, 0, 1)
        # same augmentation as PatchDataset._augment: common gain + sensor noise
        x = x * (1.0 + 0.1 * (np.random.random() - 0.5))
        x = x + 0.01 * torch.randn_like(x)
        squ = to_stokes(x)
        s_all.append(squ[0].flatten())
        q_all.append(squ[1].flatten())
        u_all.append(squ[2].flatten())

    S = torch.cat(s_all)
    Q = torch.cat(q_all)
    U = torch.cat(u_all)

    def stats(name, t):
        print(f"{name:3s} min={t.min().item():8.4f} max={t.max().item():8.4f} "
              f"mean={t.mean().item():8.4f} std={t.std().item():8.4f}")

    stats("S", S)
    stats("q", Q)
    stats("u", U)

    ok_range = Q.abs().max() <= 1.0 and U.abs().max() <= 1.0
    s_std, q_std, u_std = S.std().item(), Q.std().item(), U.std().item()
    ratio_q = q_std / s_std
    ratio_u = u_std / s_std
    print()
    print(f"q,u within [-1,1]: {bool(ok_range)}")
    print(f"std ratio q/S = {ratio_q:.3f}, u/S = {ratio_u:.3f} (comparable magnitude expected, not ~30x)")


if __name__ == "__main__":
    main()
