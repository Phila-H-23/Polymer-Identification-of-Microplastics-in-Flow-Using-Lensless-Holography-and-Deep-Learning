# Polymer Identification of Microplastics in Flow Using Lensless Holography and Deep Learning

Final-year Mechatronic Engineering thesis project, University of Cape Town.

> **Status: work in progress.** The pipeline and experiments are still being developed. Results will be added to this README once the experiments are complete.

## Contents

- [Overview](#overview)
- [Research questions](#research-questions)
- [Approach](#approach)
- [What is in this repository](#what-is-in-this-repository)
- [Getting started](#getting-started)
- [Usage](#usage)
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
| 3. Extract patches | Crops labelled particle patches and writes a manifest CSV (source video, frame, class) | [holo_pipeline_starter.py](holo_pipeline_starter.py) |
| 4. Inspect the data | Browses raw frames and patches, checks motion between frames, and scans for sensor readout glitches | [dataset_viewer.py](dataset_viewer.py) |
| 5. Train classifiers | Compact CNN trained from scratch (reference model) and a ResNet18 comparison baseline, with particle-identity and held-out-video splits | [train_compact_cnn.py](train_compact_cnn.py), [train_baseline_resnet.py](train_baseline_resnet.py) |
| 6. Evaluate | Experiment and control scripts that each write a report | [build_report_*.py](#experiment-scripts) |

## What is in this repository

The repository is currently a flat set of scripts. The main entry points are listed first, then the experiment scripts.

### Core scripts

| File | Purpose |
| --- | --- |
| [holo_pipeline_starter.py](holo_pipeline_starter.py) | Preprocessing pipeline: loading, demosaicing, particle detection, debris filtering and the patch manifest |
| [dataset_viewer.py](dataset_viewer.py) | Interactive viewer for raw frames, patches and frame differences, with a glitch scanner |
| [train_compact_cnn.py](train_compact_cnn.py) | Compact CNN training, with selectable input mode and classes |
| [train_baseline_resnet.py](train_baseline_resnet.py) | ResNet18 comparison baseline |
| [birefringence_eta_theta.m](birefringence_eta_theta.m) | MATLAB code for the birefringence parameters (eta, theta) <!-- TODO: confirm what it computes and who wrote it --> |

### Experiment scripts

<details>
<summary>Experiment and control report scripts (click to expand)</summary>

Each script runs one experiment or control and writes a report to `results/`.
<!-- TODO: add a one-line description for each script -->

- [build_report_R2.py](build_report_R2.py)
- [build_report_aug.py](build_report_aug.py)
- [build_report_bgcorr.py](build_report_bgcorr.py)
- [build_report_expanded_PETvsPS.py](build_report_expanded_PETvsPS.py)
- [build_report_final.py](build_report_final.py)
- [build_report_particle_split.py](build_report_particle_split.py)
- [build_report_pet_sessions.py](build_report_pet_sessions.py)
- [build_report_pol4ff.py](build_report_pol4ff.py)
- [build_report_raw_input.py](build_report_raw_input.py)
- [build_report_reverse_fold.py](build_report_reverse_fold.py)
- [build_report_reweighted_bg.py](build_report_reweighted_bg.py)
- [build_report_rung1.py](build_report_rung1.py)
- [build_report_size_and_arms.py](build_report_size_and_arms.py)
- [build_report_texfilt.py](build_report_texfilt.py)
- [build_report_twoclass.py](build_report_twoclass.py)
- [build_report_variants.py](build_report_variants.py)
- [build_report_vnorm.py](build_report_vnorm.py)
- [build_summary_v2.py](build_summary_v2.py)
- [build_summary_v3.py](build_summary_v3.py)

</details>

### Folders

| Folder | Contents |
| --- | --- |
| [scripts/](scripts/) | <!-- TODO: describe contents --> |
| `results/` | Generated reports. Not tracked in Git. |
| `pipeline_output/` | Generated pipeline output. Not tracked in Git. |

## Getting started

### Requirements

- Python 3.11
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

Then install the dependencies: <!-- TODO: add a requirements.txt (pip freeze > requirements.txt) and replace this line with: pip install -r requirements.txt -->

### Data

The dataset is **not included** in this repository. It was provided by the FSA collaborator and is not publicly distributed.

Recordings are MATLAB files named `<POLYMER><n>_600.mat` (for example `PET1_600.mat` and `PS2_600.mat`). Particle-free background calibration files are named `frames_bg_30*.mat`. The folder holds recordings for seven polymers, but this thesis uses PET and PS only.

<!-- TODO: state where the scripts expect the .mat files to be placed -->

## Usage

<!-- TODO: confirm the arguments for each script -->

Build the patch manifest from the raw recordings:

```bash
python holo_pipeline_starter.py
```

Browse raw frames and extracted patches:

```bash
python dataset_viewer.py
```

Train the compact CNN on PET and PS:

```bash
python train_compact_cnn.py --classes PET PS
```

Run `python <script>.py --help` to see the options for scripts that take arguments.

## Reproducibility

- Splits are defined at the particle-identity level so that patches from one particle never appear in both training and test sets. A held-out-video split is used as a second check.
- Training is repeated over several random seeds.
- Raw data, generated outputs and model files are excluded from version control (see [.gitignore](.gitignore)).

## Acknowledgements

- Fraser Montandon, for the FSA instrument, the dataset, and ongoing technical and optics guidance.
- <!-- TODO: supervisor name --> for supervision.


**Reference**

Montandon, F., and Nicolls, F. (2024). Fluid stream analyser for microplastic detection. <!-- TODO: complete the full citation -->
