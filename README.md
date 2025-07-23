# fMRI Denoising with Conv.3D UNet-LSTM and U‑Net Benchmark

## Overview

This repository implements a novel 3D Convolutional Neural Network (3D‑CNN) combined with a Long Short‑Term Memory (LSTM) network complemented with U-Net skips for denoising functional Magnetic Resonance Imaging (fMRI) data. We compare our approach against a standard U‑Net baseline to quantify improvements in reconstruction quality.

## Key Features

* **3D‑CNN‑LSTM Model**: Captures both spatial dependencies (via 3D convolutions) and temporal dynamics (via LSTM) in fMRI time series.
* **U‑Net Baseline**: Provides a strong benchmark for comparison.
* **Modular Design**: Well-organized codebase with separate directories for data, models, experiments, and training scripts.
* **Visualization**: Jupyter notebooks for inspecting raw and denoised volumes.

## Dataset

We use the OpenNeuro ds005239 dataset (version 1.0.1):

> [https://openneuro.org/datasets/ds005239/versions/1.0.1](https://openneuro.org/datasets/ds005239/versions/1.0.1)

This dataset contains raw fMRI scans that serve as input for denoising experiments.

### Directory Structure

```
fmri_project/
├── data/                  # Raw and preprocessed NIfTI files
├── kim_dataset/           # Custom Dataset and sampler utilities
├── models/                # Model definitions (3D‑CNN‑LSTM & U‑Net)
├── notebooks/             # EDA and visualization notebooks and complete main model pipeline
├── scripts/               # Helper scripts and preprocessing tools
├── train/                 # Training routines
├── experiments/           # Logging & checkpoint management
├── environment.yml        # Conda environment specifications
├── build_manifest.py      # Data manifest builder
└── README.md              # This document
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/fmri_project.git
   cd fmri_project
   ```

2. **Create Conda environment**:

   ```bash
   conda env create -f environment.yml
   conda activate fmri_project
   ```

3. **Install additional dependencies** (if any):

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare data**:

   * Place raw `sub-01` NIfTI files under `data/raw/sub-01/`.
   * Run preprocessing scripts:

     ```bash
     python scripts/preprocess.py --input data/raw --output data/processed
     ```

2. **Build manifest**:

   ```bash
   python build_manifest.py --input data/processed --output manifest.csv
   ```

3. **Train a model**:

   ```bash
   # 3D‑CNN‑LSTM training
   python train/train_baseline.py --config configs/3dcnn_lstm.yaml

   # U‑Net baseline training
   python train/train_baseline.py --config configs/unet.yaml
   ```

4. **Evaluate and visualize**:

   ```bash
   jupyter notebook notebooks/01_baseline_visuals.ipynb
   ```

## Model Architectures

* **3D‑CNN‑LSTM** (`models/cnn_lstm.py`):

  * 3D convolutional layers to extract spatial features.
  * LSTM layers to model temporal evolution across volumes complemented with U-Net skip layers to preserve the important brain parts.

* **U‑Net** (`models/unet.py`):

  * Encoder‑decoder with skip connections.
  * Serves as a widely-used baseline for medical image denoising.

## Experiments & Results

* Compare PSNR and SSIM metrics between 3D‑CNN‑LSTM and U‑Net.
* Compare tSNR of small (121M params) and big (551M params) 3D‑CNN‑LSTM architectures.
* Visual inspection notebooks demonstrate qualitative improvements.

| Model          | PSNR (↑)    | SSIM (↑) |
| -------------- | ----------- | -------- |
| U‑Net Baseline | 20 dB     | 0.66     |
| 3D‑CNN‑LSTM    | **27 dB** | **0.91** |

## Contributing

Contributions are welcome! Please open issues for feature requests or bug reports, and feel free to submit pull requests.

1. Fork the repo
2. Create a feature branch
3. Commit changes
4. Submit a pull request

## References

* OpenNeuro ds005239: [https://openneuro.org/datasets/ds005239/versions/1.0.1](https://openneuro.org/datasets/ds005239/versions/1.0.1)

---

*Created by nishxnt*
