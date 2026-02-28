[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

> Repository for the paper **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white&style=flat-square)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043?style=flat-square)

| Focus | Details |
| --- | --- |
| 🎯 Goal | Align event-camera streams and frame-camera streams with a unified implicit neural representation |
| ⚙️ Main Pipeline | `main.py` (train) + `evaluation.py` (analyze/visualize) |
| 🧪 Variant Scripts | `event_inn.py`, `frame_inn.py`, `event_deri*.py`, `event_inn_sparse.py` |

</div>

## Table of Contents

- [🔎 Overview](#-overview)
- [✨ Features](#-features)
- [🗂️ Project Structure](#-project-structure)
- [📋 Prerequisites](#-prerequisites)
- [⚙️ Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [🧪 Usage](#-usage)
- [🧩 Configuration Notes](#-configuration-notes)
- [🧠 Mathematical Formulation](#-mathematical-formulation)
- [🧾 Examples](#-examples)
- [🛠️ Development Notes](#-development-notes)
- [🧯 Troubleshooting](#-troubleshooting)
- [🗺️ Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [❤️ Support](#-support)
- [📄 License](#-license)

## 🔎 Overview

This repository contains research code for aligning event-camera streams and frame-camera streams with a unified implicit neural representation (INR/INN).

### Core Pipeline (Canonical Path)

| Step | Component | Purpose |
|---|---|---|
| 1 | `softalign/data_processing.py` | Read AEDAT4 streams, normalize data, create point samples |
| 2 | `softalign/implicit_model.py` | Define shared implicit function + learnable alignment parameters |
| 3 | `softalign/training.py` + `main.py` | Jointly optimize event/frame reconstruction |
| 4 | `evaluation.py` | Visualize alignment and report losses |

In addition to the main path, the repository includes standalone and experimental scripts for event-only, frame-only, sparse, and derivative-based INR variants.

## ✨ Features

- Event-frame alignment with learnable affine-style parameters: `scale`, `shift_x`, `shift_y`, `shift_t`.
- Shared implicit function `F(x, y, t)` used by both event and frame branches.
- Event modeling through finite-difference temporal derivative with learnable `threshold` and `dt`.
- AEDAT4 ingestion (`dv_processing`) plus processed `.npy` workflows.
- Built-in visualization outputs for loss curves, parameter evolution, and alignment overlays.
- CUDA support when available (`torch.cuda.is_available()` fallback to CPU).
- Dataset scripts for quick data inspection and lightweight debugging:
  - `read_events.py`
  - `read_frames.py`
  - `reader.py`, `reader_norm.py`, `simple_count.py`

## 🗂️ Project Structure

```text
SoftEventFrameAlignment/
├── README.md
├── LICENSE
├── main.py                              # Canonical event-frame training entrypoint
├── evaluation.py                        # Canonical evaluation/visualization entrypoint
├── softalign/
│   ├── __init__.py
│   ├── data_processing.py               # AEDAT4 loading + normalization + frame point sampling
│   ├── implicit_model.py                # MLP + alignment parameters
│   └── training.py                      # Dataset wrapper + optimization loop
├── event_inn.py                         # Event-only INR experiment
├── frame_inn.py                         # Frame-only INR experiment
├── event_deri.py                        # Derivative/log-derivative event variant
├── event_deri_2.py                      # Derivative variant with extra zero regularization
├── event_inn_sparse.py                  # Sparse event INR (PyG/torch_scatter dependencies)
├── softalign_standalone.py              # Monolithic all-in-one alignment workflow
├── softalign_old.py                     # Legacy implementation retained for comparison
├── read_events.py                       # Event I/O and basic preprocessing checks
├── read_frames.py                       # Frame/video extraction and shape checks
├── reader.py                            # AEDAT reader utility
├── reader_norm.py                       # Reader with normalization helpers
├── simple_count.py                      # Lightweight event-count utility
├── data/                                # Processed arrays (generated or checked in)
├── checkpoints/                         # Main training checkpoints and plots
├── event_inn_results/                   # Generated experiment outputs
├── frame_inn_results/                   # Generated experiment outputs
├── event_drivative_results/             # Generated experiment outputs
├── alignment_results_*/                 # Timestamped alignment runs
├── yuqing/                              # Sample AEDAT4 files
├── events_processed.csv                  # Event summary artifact
├── frames_processed.csv                  # Frame summary artifact
└── i18n/                                # Translated README files
```

## 📋 Prerequisites

- Python 3.10+ recommended.
- Linux/macOS shell examples below (adapt for Windows as needed).
- Optional but recommended: CUDA-enabled GPU for training speed.
- AEDAT4 reading requires `dv_processing` and compatible system dependencies.

## ⚙️ Installation

No `requirements.txt` or `pyproject.toml` is currently present, so install dependencies manually.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

Optional dependencies for sparse experiments (`event_inn_sparse.py`):

```bash
pip install scikit-learn torch-geometric torch-scatter
```

Assumption: CUDA availability determines the default device behavior, but all commands below also support explicit `--device cpu`.

## 🚀 Quick Start

### 1. Train the main alignment model

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

Outputs are written to:
- `data/` (`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`)
- `checkpoints/` (`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`)

### 2. Evaluate a trained checkpoint

`evaluation.py` imports `EventFrameAlignmentModel` via `from implicit_model import ...`, while the active module lives in `softalign/implicit_model.py`. A practical invocation from repo root is:

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

This generates visualizations/metrics such as `evaluation/alignment_visualization.png` and `evaluation/evaluation_results.txt`.

## 🧪 Usage

### Main Pipeline (`main.py`)

```bash
python main.py --help
```

Key options:
- `--filepath`: AEDAT4 input path.
- `--duration`: seconds to read from recording.
- `--data_dir`: processed data output directory.
- `--checkpoint_dir`: checkpoint output directory.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` or `cpu`.
- `--reprocess`: force data regeneration.

### Experimental Scripts

Event-only INR:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

Frame-only INR:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# or
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# or
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

Derivative variants:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Sparse variant:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Standalone monolithic alignment:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

Additional readers:
```bash
python read_events.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python read_frames.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python simple_count.py --input data/events.npy
```

## 🧩 Configuration Notes

Main model defaults in `softalign/implicit_model.py`:

| Parameter | Default |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

Data preprocessing in `softalign/data_processing.py` currently applies a synthetic misalignment to events before training. This is intentional in current code and should be considered when interpreting metrics.

Training scripts also expose tunable knobs through CLI flags (for example `--hidden_dim`, `--num_layers`, and `--batch_size`) that are useful for ablations.

## 🧠 Mathematical Formulation

The shared implicit network models `F(x, y, t)`.

Frame branch:
- Directly predicts intensity-like response using `F(x, y, t)`.

Event branch:
1. Apply learnable event transform:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. Approximate time derivative:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. Produce event response:
   - `sigmoid(dF/dt - threshold)`

Training objective combines:
- Event MSE loss
- Frame MSE loss
- Regularization over alignment parameters

## 🧾 Examples

Use preprocessed arrays only:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

Force CPU run:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

Evaluate to custom folder:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ Development Notes

- Repository is script-centric (no packaging metadata yet).
- Generated artifacts are present in-repo (checkpoints/results), so disk usage can be large.
- `readme-file.md` and `project-structure.txt` include legacy structure assumptions and are not fully aligned with the current root layout.
- There are currently no automated tests or CI workflows detected.

## 🧯 Troubleshooting

- `ModuleNotFoundError: No module named 'implicit_model'` in `evaluation.py`:
  - Run with `PYTHONPATH=softalign` as shown above.
- `dv_processing` install/runtime issues:
  - Confirm your platform and Python version are supported by `dv-processing` wheels/libs.
- CUDA OOM:
  - Lower `--batch_size`, reduce `--max_events` (where available), or run with `--device cpu`.
- Missing AEDAT4 file:
  - Verify path under `yuqing/` or provide your own recording via `--filepath` / `--aedat4_file`.
- Data mismatch between runs:
  - Re-run with `--reprocess` when changing preprocessing assumptions.

## 🗺️ Roadmap

- Add reproducible environment files (`requirements.txt` or `pyproject.toml`).
- Unify import paths for evaluation and package/module execution.
- Add automated tests for data processing and model forward/training checks.
- Add CI for linting and smoke tests.
- Add multilingual README files under `i18n/`.

## 🤝 Contributing

Contributions are welcome.

Recommended workflow:
1. Create a focused issue or branch summarizing the experiment change.
2. Keep scripts aligned (especially imports, CLI conventions, and output directory names).
3. Preserve existing experiment artifacts behavior unless changing naming/versioning.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

This repository is released under the Apache License 2.0. See [`LICENSE`](LICENSE) for the full text.

Assumption: if this repository is used in publications, citation details should be added here and in release notes as needed.
