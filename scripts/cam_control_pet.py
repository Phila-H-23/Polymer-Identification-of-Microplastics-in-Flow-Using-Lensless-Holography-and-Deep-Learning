"""
cam_control_pet.py

CAM control for the "uniform square" PET panels. For the seed-0 REAL
classifiers of recon_s0_z0, recon_biref, i45, and intensity_bgcorr_vnorm
(reference), computes the CAM for target class PET (regardless of the
prediction) on 100 random TEST background-only patches, preprocessed by the
same mode and normalised with the real model's training stats. Each CAM is
min-max normalised per patch (as in the displayed panels) before averaging.
Read-only, no training. Writes results/cam_control_pet.png and .csv.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util

DATA_DIR = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
PO = os.path.join(DATA_DIR, "pipeline_output")
RESULTS = os.path.join(DATA_DIR, "results")
VARIANTS = ["recon_s0_z0", "recon_biref", "i45", "intensity_bgcorr_vnorm"]
N_BG = 100
N_PANELS = 4

spec = importlib.util.spec_from_file_location("gr", os.path.join(DATA_DIR, "scripts", "gradcam_recon.py"))
gr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gr)
gp, m = gr.gp, gr.m


def cam_for_class(model, x, cls_idx):
    with torch.no_grad():
        f = model.stem(x)
        for blk in (model.block1, model.block2, model.block3, model.block4):
            f = blk(f)
        cam = torch.einsum("k,khw->hw", model.head[-1].weight[cls_idx], f[0])
        cam = F.interpolate(cam[None, None], size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
        logits = model.head[-1](f.mean(dim=(2, 3)))
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam.numpy(), int(logits.argmax(1).item())


def top_mask(a, q=0.75):
    return a >= np.quantile(a, q)


def main():
    rows, fig_rows = [], []
    for v in VARIANTS:
        cfg = gr.CONFIG[v]
        rman, rdir, ckpt = cfg["real"]
        bman, bdir, _ = cfg["bg"]
        rman_p, rdir_p = os.path.join(PO, rman), os.path.join(PO, rdir)
        classes = sorted(pd.read_csv(rman_p)["class"].unique())
        model, mean, std, test_df, c2i = gp.load_model_and_stats(
            rman_p, "split_video_holdout", rdir_p, os.path.join(RESULTS, ckpt), classes, cfg["channels"], cfg["in_ch"])
        pet = c2i["PET"]

        real_items = gr.correct_examples(model, mean, std, test_df, rdir_p, c2i, cfg["channels"], "PET")
        mean_real = np.mean([cam for _, cam in real_items], axis=0)

        bg = pd.read_csv(os.path.join(PO, bman))
        bg_test = bg[bg["split"] == "test"].sample(n=N_BG, random_state=0)
        cams, n_pred_pet = [], 0
        for _, row in bg_test.iterrows():
            ds = m.PatchDataset(pd.DataFrame([row]), os.path.join(PO, bdir), c2i, channels=cfg["channels"],
                                augment=False, mean=mean, std=std)
            x, _ = ds[0]
            cam, pred = cam_for_class(model, x.unsqueeze(0), pet)
            cams.append(cam)
            n_pred_pet += pred == pet
        mean_bg = np.mean(cams, axis=0)

        r = float(np.corrcoef(mean_bg.ravel(), mean_real.ravel())[0, 1])
        mb, mr = top_mask(mean_bg), top_mask(mean_real)
        iou = float((mb & mr).sum() / (mb | mr).sum())
        inner = np.zeros((64, 64), bool)
        inner[12:52, 12:52] = True
        rows.append(dict(variant=v, n_real_pet_panels=len(real_items), n_bg=N_BG, bg_predicted_pet=n_pred_pet,
                         pearson_r_meanbg_vs_meanreal=round(r, 3), iou_top25=round(iou, 3),
                         meanbg_inner_vs_border=round(float(mean_bg[inner].mean() / mean_bg[~inner].mean()), 2),
                         meanreal_inner_vs_border=round(float(mean_real[inner].mean() / mean_real[~inner].mean()), 2)))
        print(rows[-1])
        fig_rows.append((v, mean_bg, mean_real, len(real_items), real_items[:N_PANELS], rdir_p, cfg["disp"]))

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "cam_control_pet.csv"), index=False)

    ncol = 2 + N_PANELS
    fig, axes = plt.subplots(len(fig_rows), ncol, figsize=(2.6 * ncol, 2.7 * len(fig_rows)))
    for i, (v, mbg, mreal, n_real, items, rdir_p, disp) in enumerate(fig_rows):
        axes[i, 0].imshow(mbg, cmap="jet", vmin=0, vmax=1)
        axes[i, 0].set_title(f"mean CAM(PET)\n{N_BG} test bg patches", fontsize=8)
        axes[i, 1].imshow(mreal, cmap="jet", vmin=0, vmax=1)
        axes[i, 1].set_title(f"mean CAM(PET)\n{n_real} real PET panels", fontsize=8)
        for j in range(N_PANELS):
            ax = axes[i, 2 + j]
            if j < len(items):
                pf, cam = items[j]
                ax.imshow(gr.display_image(rdir_p, pf, disp), cmap="gray")
                ax.imshow(cam, cmap="jet", alpha=0.45)
                if j == 0:
                    ax.set_title("real PET panels (as shown before)", fontsize=8, loc="left")
        axes[i, 0].set_ylabel(v, fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("CAM control: target class PET on random test background patches vs the real PET panels (seed 0)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(RESULTS, "cam_control_pet.png")
    fig.savefig(out, dpi=110)
    print("Saved ->", out)


if __name__ == "__main__":
    main()
