[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)


# Soft Event-Frame Alignment

> Repository for the paper **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043)

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

## 🗂️ Project Structure

```text
SoftEventFrameAlignment/
├── README.md
├── LICENSE
├── main.py                          # Canonical event-frame training entrypoint
├── evaluation.py                    # Canonical evaluation/visualization entrypoint
├── softalign/
│   ├── __init__.py
│   ├── data_processing.py           # AEDAT4 loading + normalization + frame point sampling
│   ├── implicit_model.py            # MLP + alignment parameters
│   └── training.py                  # Dataset wrapper + optimization loop
├── event_inn.py                     # Event-only INR experiment
├── frame_inn.py                     # Frame-only INR experiment
├── event_deri.py                    # Derivative/log-derivative event variant
├── event_deri_2.py                  # Derivative variant with extra zero regularization
├── event_inn_sparse.py              # Sparse event INR (PyG/torch_scatter dependencies)
├── softalign_standalone.py          # Monolithic all-in-one alignment workflow
├── data/                            # Processed arrays (generated or checked in)
├── checkpoints/                     # Main training checkpoints and plots
├── event_inn_results/               # Generated experiment outputs
├── frame_inn_results/               # Generated experiment outputs
├── event_drivative_results/         # Generated experiment outputs
├── alignment_results_*/             # Timestamped alignment runs
├── yuqing/                          # Sample AEDAT4 files
└── i18n/                            # Translation files target location
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

## 🗺️ Roadmap

- Add reproducible environment files (`requirements.txt` or `pyproject.toml`).
- Unify import paths for evaluation and package/module execution.
- Add automated tests for data processing and model forward/training checks.
- Add CI for linting and smoke tests.
- Add multilingual README files under `i18n/`.

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make focused, well-documented changes.
4. Submit a pull request with reproduction details and sample outputs when relevant.

If you plan large changes (new training path, refactor, or dependency changes), open an issue first to align direction.

## 📚 Citation

If you use this repository in research, cite the related paper and/or repository.

Current repository metadata available from the existing README text:
- Paper title: *Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation*

A repository BibTeX placeholder (update authors/URL as needed):

```bibtex
@misc{soft-event-frame-alignment,
  title        = {Soft Event-Frame Alignment},
  note         = {Code repository for "Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation"},
  year         = {2025},
  howpublished = {GitHub repository}
}
```

## 🙏 Acknowledgements

- Event-camera tooling ecosystem, including `dv-processing`.
- PyTorch and scientific Python libraries used across the experiments.

## 💡 Support

Support/sponsorship section is not present in current repository metadata. If you want one, add links in a follow-up update and they will be preserved in future revisions.

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
