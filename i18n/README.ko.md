[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

> 논문 **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**을(를) 위한 저장소입니다.

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

- [🔎 개요](#-overview)
- [✨ 주요 기능](#-features)
- [🗂️ 프로젝트 구조](#-project-structure)
- [📋 사전 요구사항](#-prerequisites)
- [⚙️ 설치](#-installation)
- [🚀 빠른 시작](#-quick-start)
- [🧪 사용법](#-usage)
- [🧩 구성 참고 사항](#-configuration-notes)
- [🧠 수학적 정식화](#-mathematical-formulation)
- [🧾 예시](#-examples)
- [🛠️ 개발 노트](#-development-notes)
- [🧯 문제 해결](#-troubleshooting)
- [🗺️ 로드맵](#-roadmap)
- [🤝 기여](#-contributing)
- [❤️ Support](#-support)
- [📄 라이선스](#-license)

## 🔎 개요

이 저장소는 통합된 암시적 신경 표현(INR/INN)을 사용하여 이벤트 카메라 스트림과 프레임 카메라 스트림을 정렬하기 위한 연구 코드입니다.

### 핵심 파이프라인 (Canonical Path)

| Step | Component | Purpose |
|---|---|---|
| 1 | `softalign/data_processing.py` | AEDAT4 스트림을 읽고, 데이터 정규화, 포인트 샘플 생성 |
| 2 | `softalign/implicit_model.py` | 공유 암시적 함수 + 학습 가능한 정렬 파라미터 정의 |
| 3 | `softalign/training.py` + `main.py` | 이벤트/프레임 재구성을 공동으로 최적화 |
| 4 | `evaluation.py` | 정렬 결과 시각화 및 손실 리포트 |

메인 경로 외에도 이벤트 전용, 프레임 전용, 희소, 도함수 기반 INR 변형을 위한 독립형 및 실험 스크립트가 포함되어 있습니다.

## ✨ 주요 기능

- 학습 가능한 affine 스타일 파라미터 `scale`, `shift_x`, `shift_y`, `shift_t`를 통한 이벤트-프레임 정렬.
- 이벤트와 프레임 브랜치가 공유하는 암시적 함수 `F(x, y, t)`.
- 학습 가능한 `threshold`와 `dt`를 포함한 유한차분 시간 도함수 기반 이벤트 모델링.
- AEDAT4 입력(`dv_processing`) 및 처리된 `.npy` 워크플로.
- 손실 곡선, 파라미터 변화, 정렬 오버레이를 위한 내장 시각화 출력.
- CUDA 사용 가능 시 지원(`torch.cuda.is_available()` 시 CPU로 폴백).
- 데이터 탐색 및 경량 디버깅을 위한 데이터셋 스크립트:
  - `read_events.py`
  - `read_frames.py`
  - `reader.py`, `reader_norm.py`, `simple_count.py`

## 🗂️ 프로젝트 구조

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

## 📋 사전 요구사항

- Python 3.10+ 권장.
- 아래 예시는 Linux/macOS 셸 기준입니다(Windows는 필요에 맞게 조정).
- 선택 항목이지만 학습 속도를 위해 CUDA 지원 GPU 권장.
- AEDAT4 읽기에는 `dv_processing` 및 호환되는 시스템 의존성이 필요합니다.

## ⚙️ 설치

현재 `requirements.txt` 또는 `pyproject.toml`이 없어 의존성을 수동으로 설치해야 합니다.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

희소 실험(`event_inn_sparse.py`)을 위한 선택 의존성:

```bash
pip install scikit-learn torch-geometric torch-scatter
```

CUDA 사용 가능 여부가 기본 디바이스 동작을 결정하지만, 아래 모든 명령은 `--device cpu`를 명시적으로 지정해 사용할 수도 있습니다.

## 🚀 빠른 시작

### 1. 메인 정렬 모델 학습

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

출력 위치:
- `data/` (`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`)
- `checkpoints/` (`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`)

### 2. 학습된 체크포인트 평가

`evaluation.py`는 `from implicit_model import ...`를 통해 `EventFrameAlignmentModel`을 임포트하지만, 실제 활성 모듈은 `softalign/implicit_model.py`에 있습니다. 저장소 루트에서의 실용적인 실행 예시는 다음과 같습니다.

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

이 명령은 `evaluation/alignment_visualization.png`와 `evaluation/evaluation_results.txt` 같은 시각화/지표를 생성합니다.

## 🧪 사용법

### 메인 파이프라인 (`main.py`)

```bash
python main.py --help
```

핵심 옵션:
- `--filepath`: AEDAT4 입력 경로.
- `--duration`: 녹화 구간을 읽는 시간(초).
- `--data_dir`: 처리된 데이터 출력 디렉터리.
- `--checkpoint_dir`: 체크포인트 출력 디렉터리.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` 또는 `cpu`.
- `--reprocess`: 데이터 재생성을 강제.

### 실험 스크립트

이벤트 전용 INR:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

프레임 전용 INR:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# or
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# or
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

도함수 기반 변형:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

희소 변형:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

독립형 monolithic 정렬:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

추가 리더 도구:
```bash
python read_events.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python read_frames.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python simple_count.py --input data/events.npy
```

## 🧩 구성 참고 사항

`softalign/implicit_model.py`의 메인 모델 기본값:

| Parameter | Default |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

`softalign/data_processing.py`의 데이터 전처리는 현재 학습 전 이벤트에 인위적인 오정렬을 적용합니다.
이는 현재 코드에서 의도된 동작이므로, 메트릭을 해석할 때 이를 고려해야 합니다.

학습 스크립트는 CLI 플래그(`--hidden_dim`, `--num_layers`, `--batch_size` 등)로 조정 가능한 하이퍼파라미터도 제공합니다. 이는 ablation 실험에 유용합니다.

## 🧠 수학적 정식화

공유 임플리싯 네트워크는 `F(x, y, t)`를 모델링합니다.

프레임 브랜치:
- `F(x, y, t)`를 사용해 intensity 유사 응답을 직접 예측합니다.

이벤트 브랜치:
1. 학습 가능한 이벤트 변환 적용:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. 시간 도함수 근사:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. 이벤트 응답 생성:
   - `sigmoid(dF/dt - threshold)`

학습 목적 함수는 다음을 결합합니다:
- 이벤트 MSE 손실
- 프레임 MSE 손실
- 정렬 파라미터에 대한 정규화

## 🧾 예시

전처리된 배열만 사용:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

CPU 강제 실행:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

사용자 정의 폴더로 평가:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ 개발 노트

- 저장소는 스크립트 중심 구성입니다(패키징 메타데이터는 아직 없습니다).
- 생성 결과(체크포인트/실험 결과)가 저장소 내에 존재하므로 디스크 사용량이 증가할 수 있습니다.
- `readme-file.md`와 `project-structure.txt`는 레거시 구조 가정이 포함되어 있어 현재 루트 레이아웃과 완전히 일치하지 않습니다.
- 현재 자동화 테스트 또는 CI 워크플로우가 감지되지 않았습니다.

## 🧯 문제 해결

- `evaluation.py`에서 `ModuleNotFoundError: No module named 'implicit_model'`:
  - 위 예시처럼 `PYTHONPATH=softalign`로 실행하세요.
- `dv_processing` 설치/런타임 문제:
  - 사용 중인 플랫폼과 Python 버전이 `dv-processing` wheel/lib 지원 대상인지 확인하세요.
- CUDA OOM:
  - `--batch_size`를 낮추고, `--max_events`를 줄이거나 `--device cpu`로 실행하세요.
- AEDAT4 파일 누락:
  - `yuqing/` 경로를 확인하거나 `--filepath`/`--aedat4_file`로 직접 녹화 파일을 지정하세요.
- 실행 간 데이터 불일치:
  - 전처리 가정을 변경한 경우 `--reprocess`를 다시 실행하세요.

## 🗺️ 로드맵

- 재현 가능한 환경 파일 추가(`requirements.txt` 또는 `pyproject.toml`).
- 평가 및 패키지/모듈 실행을 위한 import 경로 통합.
- 데이터 처리와 모델 forward/학습 검증을 위한 자동 테스트 추가.
- lint 및 smoke test를 위한 CI 추가.
- `i18n/` 하위에 다국어 README 파일 추가.

## 🤝 기여

기여는 언제나 환영합니다.

권장 워크플로:
1. 실험 변경사항을 요약한 이슈나 브랜치를 먼저 생성하세요.
2. 특히 import, CLI 규칙, 출력 디렉터리 이름을 유지하며 스크립트를 정렬하세요.
3. 이름/버전 체계를 바꾸지 않는 이상 기존 실험 산출물 동작은 유지하세요.

## 📄 라이선스

이 저장소는 Apache License 2.0 하에 배포됩니다. 전체 텍스트는 [`LICENSE`](LICENSE)에서 확인하세요.

전제 조건: 이 저장소가 연구 논문에서 사용되는 경우, 해당 발간물 및 릴리스 노트에 인용 정보를 추가해야 할 수 있습니다.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
