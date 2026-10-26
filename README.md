# Polymer Identification of Microplastics in Flow Using Lensless Holography and Deep Learning

Final-year Mechatronic Engineering thesis project, University of Cape Town.

> **Status: work in progress.** The pipeline and experiments are still being developed. The summary table under [Results so far](#results-so-far) is descriptive and will change as experiments are added.

## Contents

- [Overview](#overview)
- [Research questions](#research-questions)
- [Approach](#approach)
- [What is in this repository](#what-is-in-this-repository)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results so far](#results-so-far)
- [Reproducibility](#reproducibility)
- [Acknowledgements](#acknowledgements)
- [Licence and citation](#licence-and-citation)

## Overview

Microplastics are small plastic particles that are difficult to identify at scale. This project investigates whether the polymer type of individual microplastic particles flowing through a lensless, polarisation-sensitive in-line holographic sensor can be identified by a convolutional neural network, and how the choice of network input affects classification performance.

The scope is limited to two polymers: **PET** (polyethylene terephthalate) and **PS** (polystyrene).

The work builds on the fluid stream analyser (FSA) developed by Montandon and Nicolls (2024), and uses data recorded with that instrument.

## Research questions

1. Can PET and PS particles be distinguished directly from raw holographic fringe patterns, without numerical wavefield reconstruction?
2. How does classification performance compare across input representations: raw polarisation channels, intensity only, and numerically reconstructed fields?
3. How do the data splitting strategy and recording conditions affect how trustworthy a reported classification result is?

## Approach

The pipeline has six stages. The script that implements each stage is linked.

| Stage | What it does | Script |
| --- | --- | --- |
| 1. Load and demosaic | Reads raw MATLAB sensor frames and splits the polarisation mosaic into four channels | [holo_pipeline_starter.py](holo_pipeline_starter.py) |
| 2. Detect particles | Finds particles using local variance, then filters debris using motion checks and cross-frame recurrence | [holo_pipeline_starter.py](holo_pipeline_starter.py) |
| 3. Extract patches | Crops labelled particle patches and writes a manifest CSV (source video, frame, class) | [holo_pipeline_starter.py](holo_pipeline_starter.py), [scripts/extract_PETPS_exp.py](scripts/extract_PETPS_exp.py) |
| 4. Inspect the data | Browses raw frames and patches, checks motion between frames, and scans for sensor readout glitches | [dataset_viewer.py](dataset_viewer.py) |
| 5. Train classifiers | Compact CNN trained from scratch (reference model) and a ResNet18 comparison baseline, with particle-identity and held-out-video splits | [train_compact_cnn.py](train_compact_cnn.py), [train_baseline_resnet.py](train_baseline_resnet.py) |
| 6. Evaluate | Experiment and control scripts that each write a report | [build_report_*.py](#experiment-report-scripts), [scripts/](#scripts-folder) |

The network input is built in one of several ways, chosen with `--channels`: raw polarisation channels, intensity only, background-corrected variants, Stokes parameters, or numerically reconstructed patches (see [scripts/reconstruction.py](scripts/reconstruction.py)). The input called **z0** is the unpropagated frame divided by a per-pixel median of up to 90 preceding frames, averaged over the four analyser channels, as 64 x 64 patches at 6.9 um.

## What is in this repository

The repository is a flat set of scripts. The main entry points are listed first, then the experiment scripts and the `scripts/` folder.

### Core scripts

| File | Purpose |
| --- | --- |
| [holo_pipeline_starter.py](holo_pipeline_starter.py) | Preprocessing pipeline: loading, demosaicing, particle detection, debris filtering and the patch manifest |
| [dataset_viewer.py](dataset_viewer.py) | Interactive viewer for raw frames, patches and frame differences, with a glitch scanner |
| [train_compact_cnn.py](train_compact_cnn.py) | Compact CNN training, with selectable input mode and classes. The reference script: results depend on it, so it is not edited. |
| [train_baseline_resnet.py](train_baseline_resnet.py) | ResNet18 comparison baseline (pretrained, adapted to 4-channel polarisation patches) |
| [birefringence_eta_theta.m](birefringence_eta_theta.m) | MATLAB function `[eta, theta] = birefringence_eta_theta(I_0, I_45, I_90, I_135)`. From the four analyser images it computes a four-frame retardance `eta` (arcsine of the normalised Stokes-style magnitude, scaled to 0 to 1) and a slow-axis azimuth `theta` (modulo pi, scaled to 0 to 1). Invalid pixels are NaN. |
| [report_graph_maker.py](report_graph_maker.py) | Builds the chapter 4 figures in [figures/ch4/](figures/ch4/) from the numbers in the results log |

### Training script variants

These are copies of `train_compact_cnn.py` with extra options. Each is used by the experiment chains below, and with its new options off it behaves as the script it was copied from.

| File | Adds |
| --- | --- |
| [train_compact_cnn_bright.py](train_compact_cnn_bright.py) | Brightness-related input options (gain range, mean normalisation) used in the brightness experiments |
| [train_compact_cnn_depth.py](train_compact_cnn_depth.py) | `--depth-aug {none, all21, shifted20}`: training patches drawn at random from pre-computed propagation depths (2.000 to 2.500 mm). Validation, test and channel statistics always use the unshifted 2.250 mm patches. |
| [train_compact_cnn_deep.py](train_compact_cnn_deep.py) | Everything in the depth script, plus `--extra-convs K`: K extra 3 x 3 conv + batchnorm + ReLU layers in each of the four blocks (K = 2 gives 18 weighted layers; K = 0 is the original 10) |
| [train_compact_cnn_fit.py](train_compact_cnn_fit.py) | Everything in the deep script, plus `--no-aug` (no training augmentation of any kind) and `--save-last` (also saves the final-epoch weights as `last.pt`), for the fit check |

### Experiment report scripts

<details>
<summary>Experiment and control report scripts (click to expand)</summary>

Each script reads finished run folders and writes a report to `results/`.

| Script | What it reports |
| --- | --- |
| [build_report_R2.py](build_report_R2.py) | Background-only (particle-free) arms re-run with the video-holdout class weights used by the real runs |
| [build_report_aug.py](build_report_aug.py) | Surround augmentation (background, background plus texture, interior) for real versus particle-free arms |
| [build_report_bgcorr.py](build_report_bgcorr.py) | Background-subtraction (90-frame rolling background) re-test: does it open up the real versus particle-free gap |
| [build_report_expanded_PETvsPS.py](build_report_expanded_PETvsPS.py) | The expanded PET and PS patch set, every arm and direction |
| [build_report_final.py](build_report_final.py) | Final report for the session-confound control task |
| [build_report_particle_split.py](build_report_particle_split.py) | Particle-identity split compared with the held-out-video split |
| [build_report_pet_sessions.py](build_report_pet_sessions.py) | PET session-count test |
| [build_report_pol4ff.py](build_report_pol4ff.py) | Flat-field (`pol4_ff`) re-test |
| [build_report_raw_input.py](build_report_raw_input.py) | Raw-input comparison table: uncorrected and flat-fielded polarisation and intensity, real versus background-only |
| [build_report_reverse_fold.py](build_report_reverse_fold.py) | Reverse-fold test: original split next to the reversed test recordings |
| [build_report_reweighted_bg.py](build_report_reweighted_bg.py) | Background-only arms re-run with the video-holdout class weights |
| [build_report_rung1.py](build_report_rung1.py) | Channel ablation rung 1: intensity versus a single analyser channel |
| [build_report_size_and_arms.py](build_report_size_and_arms.py) | Size-only baselines, and PET vs PS on four more input arms |
| [build_report_texfilt.py](build_report_texfilt.py) | Texture-exclusion variant of the polarisation input |
| [build_report_twoclass.py](build_report_twoclass.py) | Two-class (pairwise) tests |
| [build_report_variants.py](build_report_variants.py) | Real versus background-only for any input variant, using whichever seeds have finished |
| [build_report_vnorm.py](build_report_vnorm.py) | Per-video brightness normalisation diagnostic |
| [build_summary_v2.py](build_summary_v2.py) | `summary_v2.csv`: compact CNN and ResNet18 over channel modes and seeds |
| [build_summary_v3.py](build_summary_v3.py) | `summary_v3.csv`: the same summary after the Stokes-formula fix |

</details>

### Scripts folder

[scripts/](scripts/) holds the dataset builders, analysis code and numbered experiment chains. It is a flat folder.

| Group | Files | Purpose |
| --- | --- | --- |
| Dataset builders | `build_*_dataset.py`, `build_flatfield_ref.py`, `build_dust_masks.py`, `extract_PETPS_exp.py`, `assemble_PETPS_exp.py`, `tracks_PETPS_exp.py`, `build_arms_PETPS_exp.py` | Build patch sets and manifests: background-corrected, particle-free, reconstructed, flat-field and the expanded PET and PS set with track ids |
| Reconstruction | `reconstruction.py`, `reconstruction_validate.py`, `reconstruction_zsweep2.py`, `reconstruction_biref_synthetic.py`, `recon_check*.py` | Numerical wavefield reconstruction and its checks |
| Diagnostics | `gradcam_*.py`, `cam_control_pet.py`, `stokes_diagnostic*.py`, `artifact_exposure_check.py`, `texture_exclusion_feasibility.py`, `scalar_baselines_PETPS_exp.py` | Class activation maps, Stokes checks, size-only baselines and other controls |
| Run queues | `run_*.sh` | Shell queues that launch the training runs behind the earlier reports |
| Evaluation | `eval_track_ci_exp.py`, `unclipped_tracks.py` | Per-patch test predictions from a saved checkpoint, and the bootstrap over whole test tracks |
| Items 41 to 49 | `item4*_*.py`, `j1_*` to `j3_*`, `stage*_*.py`, `overnight_*.py`, `pp_*_exp.py`, `run_arm_exp.py` | The numbered experiment chains, their status writers and report builders (see [Experiments](#experiments)) |
| Figures | `make_cam_figure.py`, `make_recon_inputs_figure.py` | The CAM figure and the figure comparing z0 with the two reconstructions |
| Helpers | `prio41.py` | Sets Windows process priority for long background runs |

### Folders

| Folder | Contents |
| --- | --- |
| [scripts/](scripts/) | See above |
| [figures/ch4/](figures/ch4/) | Chapter 4 figures |
| `results/` | Generated reports, run folders and figures. Not tracked in Git. |
| `pipeline_output/` | Generated patches and manifests. Not tracked in Git. |

## Getting started

### Requirements

- Python 3 (the pinned versions in `requirements.txt` were read from Python 3.13)
- Git
- A GPU is optional. All scripts run on CPU.

### Setup

```bash
git clone https://github.com/Phila-H-23/Polymer-Identification-of-Microplastics-in-Flow-Using-Lensless-Holography-and-Deep-Learning.git
cd Polymer-Identification-of-Microplastics-in-Flow-Using-Lensless-Holography-and-Deep-Learning
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

`torch` and `torchvision` in `requirements.txt` are the CPU builds that were used. Install a CUDA build from pytorch.org instead if you want a GPU.

### Data

The dataset is **not included** in this repository. It was provided by the FSA collaborator and is not publicly distributed.

Recordings are MATLAB files named `<POLYMER><n>_600.mat` (for example `PET1_600.mat` and `PS2_600.mat`). Particle-free background calibration files are named `frames_bg_30*.mat`. The folder holds recordings for seven polymers, but this thesis uses PET and PS only.

Place the `.mat` files in the repository root, next to `holo_pipeline_starter.py`. The scripts read them from there by relative file name. To process a different recording, change `INPUT_FILE` near the top of `holo_pipeline_starter.py` (the pipeline is run once per recording).

## Usage

Build the patch manifest from the raw recordings (set `INPUT_FILE` in the script first, then run it once per recording):

```bash
python holo_pipeline_starter.py
```

Browse raw frames and extracted patches:

```bash
python dataset_viewer.py
```

Train the compact CNN on PET and PS. Classes are given as a comma-separated list:

```bash
python train_compact_cnn.py --classes PET,PS
```

Options of `train_compact_cnn.py` (defaults in brackets): `--channels` (input mode), `--split` (`particle` by default, or `video_holdout` and the other split names defined in the script), `--manifest`, `--patch-dir`, `--class-weights`, `--epochs` (80), `--batch-size` (32), `--lr` (0.001), `--weight-decay` (0.0001), `--width` (32), `--dropout` (0.3), `--patience` (20), `--seed` (0), `--out` (`results/compact_cnn`). The training script variants take the same options plus the ones listed above.

A run of the 18-layer network with depth augmentation on the z0 expanded set (original direction), as used in item 47:

```bash
python train_compact_cnn_deep.py --channels recon_s0 --split video_holdout \
  --manifest pipeline_output/manifest_recon_s0_z0_PETPS_exp.csv \
  --patch-dir pipeline_output/patches_depth_z0_PETPS_exp \
  --class-weights pipeline_output/class_weights_twoclass_PETvsPS_exp.json \
  --classes PET,PS --epochs 150 --patience 30 --batch-size 32 --lr 0.001 --weight-decay 0.0001 \
  --width 32 --dropout 0.3 --seed 0 --depth-aug all21 --extra-convs 2 --out results/my_run
```

The manifests and patch folders are generated by the dataset builders in `scripts/` and are not in the repository, so this command only works after you have rebuilt them from the recordings.

Run `python <script>.py --help` to see the options for scripts that take arguments.

## Experiments

The later experiments are numbered items (41 to 49). Each item fixed its readings before the runs, ran the same seeds, splits and test recordings as the one before, and wrote a report to `results/`. Two test directions are used throughout: **original** (test recordings PET2 and PS1) and **reverse** (test recordings PET1 and PS2). Real runs use patches of detected particles. Particle-free runs use paired empty-background patches from the same frames as a control.

| Items | What was tested | Main scripts |
| --- | --- | --- |
| 41 | Rebuilt reconstruction (V3) compared with the as-built one and with z0; CAM share inside the detected box; brightness features | `stage3_V3_*.py`, `j1_*`, `j2_*`, `j3_cam_z0_exp.py` |
| 42 to 43 | Test-pair tables and a polarisation-input arm under z0 | `item42_*.py`, `item43_*.py` |
| 44 | Depth augmentation: training patches drawn from propagation depths 2.000 to 2.500 mm | `item44_chain.py`, `item44_depth_build.py` |
| 45 | Four-channel polarisation arm completed | `item45_chain.py` |
| 46 | Width 64 (original direction), with and without depth augmentation | `item46_*.py` |
| 47 | 18-layer network at width 32, with and without depth augmentation | `item47_*.py`, `train_compact_cnn_deep.py` |
| 48 | Fit check (can the network fit its training data with regularisation off) and width 64 with 18 layers | `item48_*.py`, `train_compact_cnn_fit.py` |
| 49 | Width 64, 10 layers, reverse direction, completing the capacity grid | `item49_chain.py`, `item49_report.py` |

Each numbered chain launches its runs one at a time, waits for free memory before each, and writes a log and a `runs.json`. The matching `*_analysis.py` or `*_report.py` writes the report. Comparisons use the real seed spread as a margin and a paired bootstrap over whole test tracks (1000 resamples, test tracks only, not seeds or recordings).

## Results so far

Real macro-F1 for PET vs PS on the z0 expanded set, mean over seeds 0 to 2. These numbers are descriptive. The comparisons are within about one to two seed standard deviations in most cases, and the pre-registered readings in the item reports should be used for any statement about differences.

| Arm | Width | Layers | Depth aug | Original | Reverse |
| --- | --- | --- | --- | --- | --- |
| z0 | 32 | 10 | none | 0.9087 | 0.8477 |
| D | 32 | 10 | all21 | 0.9094 | 0.8514 |
| Wn | 64 | 10 | none | 0.8945 | 0.8226 |
| Wd | 64 | 10 | all21 | 0.9138 | 0.8442 |
| Ln | 32 | 18 | none | 0.9038 | 0.8514 |
| Ld | 32 | 18 | all21 | 0.9037 | 0.8453 |
| WLn | 64 | 18 | none | 0.8873 | 0.8301 |
| WLd | 64 | 18 | all21 | 0.9098 | 0.8086 |

The full grid with per-seed values and parameter counts is written to `results/capacity_grid_summary.csv` by `scripts/item49_report.py`. A fit check with regularisation off (item 48) reached a training macro-F1 of 1.0000 for both the 10-layer and 18-layer networks (one seed, original direction, descriptive). Their test macro-F1 was 0.857 and 0.881.

Nothing here shows that the CNN identifies polymers. The particle-free controls were run to test that, and the item reports should be read for what they do and do not support. The reported figures come from one dataset, three seeds per arm and two test-recording pairs.

## Reproducibility

- Splits are defined at the particle-identity level so that patches from one particle never appear in both training and test sets. A held-out-video split is used as a second check, and a reversed test-recording pair as a third.
- Training is repeated over several random seeds (0 to 2 for the experiments above, 0 to 4 for the earlier ones).
- `train_compact_cnn.py` is kept unchanged so that earlier results stay comparable. Later work copies it into the variants above instead of editing it.
- Experiment scripts write new files only and do not overwrite earlier results.
- Raw data, generated outputs and model files are excluded from version control (see [.gitignore](.gitignore)).

## Acknowledgements

- Fraser Montandon, for the FSA instrument, the dataset, and ongoing technical and optics guidance.
- <!-- TODO: supervisor name --> for supervision.


**Reference**

Montandon, F., and Nicolls, F. (2024). Fluid stream analyser for microplastic detection. <!-- TODO: complete the full citation -->

## Licence and citation

<!-- TODO: choose a licence and add a LICENSE file. The thesis itself should be cited once it is submitted. -->
