[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

*Minimal research code for learning a soft spatial and temporal alignment between event-camera and frame-camera samples with one implicit neural representation.*

[![CI](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA043)](LICENSE)
[![Sponsor](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)

This is the first public, cloneable implementation behind **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**. It includes the canonical model, AEDAT4 preprocessing, training, evaluation, dependency metadata, CI, and a deterministic synthetic CPU check.

No recordings, identifiable frames, derived arrays, or trained checkpoints are published. Bring data that you have the right to use.

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=kofi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## Architecture

```mermaid
flowchart LR
    A[Rights-cleared AEDAT4] --> B[Normalize x, y and seconds]
    B --> C[Event samples]
    B --> D[Frame-point samples]
    C --> E[Learnable spatial/time transform]
    E --> F[Shared implicit field F(x,y,t)]
    D --> F
    F --> G[Reconstruction losses]
    G --> H[Checkpoint + bounded diagnostics]
```

The event branch transforms coordinates and approximates the temporal derivative of a shared MLP. The frame branch evaluates the same field directly. Scale and derivative-step parameters use positive parameterizations so training cannot divide by zero.

## Public release contents

| Path | Purpose |
| --- | --- |
| `softalign/implicit_model.py` | Shared implicit MLP and learnable alignment parameters |
| `softalign/training.py` | Validated dataset wrapper and joint training loop |
| `softalign/data_processing.py` | Optional AEDAT4 ingestion and explicit seconds-based preprocessing |
| `softalign/synthetic.py` | Deterministic project-generated smoke data |
| `main.py` | Canonical training CLI |
| `evaluation.py` | Visualization and bounded reconstruction diagnostics |
| `examples/synthetic_smoke.py` | End-to-end CPU train/evaluate check |
| `tests/test_synthetic_smoke.py` | Core test without AEDAT, CUDA, or plotting requirements |

Experimental notebooks, raw captures, local datasets, and historical checkpoints are intentionally outside this release.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,viz]'
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py --output-dir .smoke-output --epochs 8
```

The smoke run should create a checkpoint, plots, and `evaluation_results.json`. It verifies that the pipeline executes and produces finite values; it is not evidence of alignment accuracy.

## Run on your own AEDAT4 recording

```bash
python -m pip install -e '.[aedat,viz]'
python main.py --filepath /path/to/your-recording.aedat4 --reprocess --data_dir data --checkpoint_dir checkpoints
python evaluation.py --model_path checkpoints/model_final.pt --data_dir data --output_dir evaluation --device cpu
```

Preprocessing records `softalign-processed/v1` metadata and converts AEDAT timestamps from microseconds to seconds. The optional test perturbation is disabled by default; add `--synthetic-misalignment` only for a controlled experiment. Preprocessed arrays without compatible metadata are rejected instead of being silently mixed with the current time convention.

Large event collections remain in CPU memory, and only sampled batches move to the selected device. Use `--device cpu` for the most portable first run; use CUDA only after checking memory requirements.

## Data contract

| File | Shape / meaning |
| --- | --- |
| `events.npy` | `N × 4`: normalized `x`, `y`, time in seconds, event target |
| `frame_points.npy` | `M × 4`: normalized `x`, `y`, time in seconds, intensity |
| `preprocessing.json` | Schema, units, seed, transform flag, and sample counts |
| `model_final.pt` | State dict, reported parameters, and model architecture |
| `evaluation_results.json` | Reconstruction MSE over bounded supplied samples |

Evaluation results are diagnostics on the supplied arrays, not held-out metrics, a benchmark, or proof that a physical sensor pair has been calibrated.

## Validation and research limits

- CI runs the deterministic CPU unit test and a three-epoch train/evaluate cycle.
- Inputs are checked for shape, emptiness, finite values, time-unit metadata, and device availability.
- Frame sampling accepts small images and a seeded generator.
- The current loss is a research baseline. A real experiment still needs a documented train/evaluation split, calibration ground truth, ablations, and uncertainty analysis.
- Checkpoints from a different architecture or old microsecond preprocessing are not compatible with this release.

## Citation

If you use Soft Event-Frame Alignment in research, cite the repository. GitHub reads [CITATION.cff](CITATION.cff) and shows a **Cite this repository** panel on the repo page.

```bibtex
@software{chen_soft_event_frame_alignment_2026,
  author = {Chen, Lachlan},
  title = {Soft Event-Frame Alignment: Unified Implicit Neural Representation Research Code},
  year = {2026},
  url = {https://github.com/lachlanchen/SoftEventFrameAlignment}
}
```

## Status

Version `0.1.0` is a minimal research release. Issues that reproduce a failure with the synthetic check or a small rights-cleared sample are welcome. Please do not attach private recordings to public issues.

## License

Apache License 2.0. See [LICENSE](LICENSE).
