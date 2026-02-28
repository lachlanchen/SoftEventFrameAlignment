[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Soft Event-Frame Alignment

> 논문 **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**의 저장소입니다.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043)

## 🔎 개요

이 저장소는 통합 암시적 신경 표현(INR/INN)을 사용해 이벤트 카메라 스트림과 프레임 카메라 스트림을 정렬하는 연구 코드를 담고 있습니다.

### 핵심 파이프라인 (Canonical Path)

| 단계 | 구성요소 | 목적 |
|---|---|---|
| 1 | `softalign/data_processing.py` | AEDAT4 스트림 읽기, 데이터 정규화, 포인트 샘플 생성 |
| 2 | `softalign/implicit_model.py` | 공유 암시적 함수 + 학습 가능한 정렬 파라미터 정의 |
| 3 | `softalign/training.py` + `main.py` | 이벤트/프레임 재구성을 공동 최적화 |
| 4 | `evaluation.py` | 정렬 시각화 및 손실 보고 |

메인 경로 외에도 이벤트 전용, 프레임 전용, 희소(sparse), 도함수 기반 INR 변형을 위한 독립/실험 스크립트가 포함되어 있습니다.

## ✨ 주요 기능

- 학습 가능한 affine 스타일 파라미터(`scale`, `shift_x`, `shift_y`, `shift_t`)를 사용한 이벤트-프레임 정렬.
- 이벤트/프레임 브랜치가 공유하는 암시적 함수 `F(x, y, t)`.
- 학습 가능한 `threshold`, `dt`를 포함한 유한차분 시간 도함수 기반 이벤트 모델링.
- AEDAT4 입력(`dv_processing`) 및 처리된 `.npy` 워크플로.
- 손실 곡선, 파라미터 변화, 정렬 오버레이를 위한 내장 시각화 출력.
- CUDA 사용 가능 시 GPU 지원 (`torch.cuda.is_available()`에서 CPU로 폴백).

## 🗂️ 프로젝트 구조

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

## 📋 사전 요구사항

- Python 3.10+ 권장.
- 아래 예시는 Linux/macOS 셸 기준입니다(Windows는 필요에 맞게 조정).
- 선택 사항이지만 학습 속도를 위해 CUDA 지원 GPU 권장.
- AEDAT4 읽기에는 `dv_processing` 및 호환되는 시스템 의존성이 필요합니다.

## ⚙️ 설치

현재 `requirements.txt` 또는 `pyproject.toml`이 없으므로 의존성을 수동으로 설치하세요.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

희소 실험(`event_inn_sparse.py`)용 선택 의존성:

```bash
pip install scikit-learn torch-geometric torch-scatter
```

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

`evaluation.py`는 `from implicit_model import ...`로 `EventFrameAlignmentModel`을 임포트하지만, 실제 활성 모듈은 `softalign/implicit_model.py`에 있습니다. 저장소 루트에서의 실용적인 실행 예시는 다음과 같습니다.

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

이 명령은 `evaluation/alignment_visualization.png`, `evaluation/evaluation_results.txt` 같은 시각화/지표 파일을 생성합니다.

## 🧪 사용법

### 메인 파이프라인 (`main.py`)

```bash
python main.py --help
```

주요 옵션:
- `--filepath`: AEDAT4 입력 경로.
- `--duration`: 레코딩에서 읽어올 초 단위 길이.
- `--data_dir`: 처리 데이터 출력 디렉터리.
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

도함수 변형:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

희소 변형:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

독립형(monolithic) 정렬:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

## 🧩 설정 참고

메인 모델 기본값(`softalign/implicit_model.py`):

| 파라미터 | 기본값 |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

`softalign/data_processing.py`의 데이터 전처리는 현재 학습 전에 이벤트에 합성 오정렬(synthetic misalignment)을 적용합니다. 이는 현재 코드에서 의도된 동작이며, 지표 해석 시 고려해야 합니다.

## 🧠 수학적 정식화

공유 암시적 네트워크는 `F(x, y, t)`를 모델링합니다.

프레임 브랜치:
- `F(x, y, t)`를 사용해 intensity 유사 응답을 직접 예측.

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

사용자 지정 폴더로 평가:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ 개발 노트

- 저장소는 스크립트 중심 구조입니다(아직 패키징 메타데이터 없음).
- 생성 산출물(체크포인트/결과)이 저장소 내에 존재하므로 디스크 사용량이 커질 수 있습니다.
- `readme-file.md`, `project-structure.txt`는 레거시 구조 가정을 포함하며 현재 루트 레이아웃과 완전히 일치하지 않습니다.
- 현재 자동화 테스트나 CI 워크플로는 확인되지 않았습니다.

## 🧯 문제 해결

- `evaluation.py`에서 `ModuleNotFoundError: No module named 'implicit_model'`:
  - 위 예시처럼 `PYTHONPATH=softalign`로 실행하세요.
- `dv_processing` 설치/런타임 문제:
  - 사용 플랫폼과 Python 버전이 `dv-processing` wheel/lib 지원 대상인지 확인하세요.
- CUDA OOM:
  - `--batch_size`를 낮추고, 가능하면 `--max_events`를 줄이거나, `--device cpu`로 실행하세요.
- AEDAT4 파일 누락:
  - `yuqing/` 하위 경로를 확인하거나 `--filepath` / `--aedat4_file`로 자체 레코딩 파일을 지정하세요.

## 🗺️ 로드맵

- 재현 가능한 환경 파일(`requirements.txt` 또는 `pyproject.toml`) 추가.
- 평가 및 패키지/모듈 실행을 위한 import 경로 통합.
- 데이터 처리 및 모델 forward/학습 검증 자동 테스트 추가.
- lint 및 스모크 테스트용 CI 추가.
- `i18n/` 하위에 다국어 README 파일 추가.

## 🤝 기여

기여를 환영합니다.

1. 저장소를 포크합니다.
2. 기능 브랜치를 생성합니다.
3. 범위가 명확하고 문서화된 변경을 적용합니다.
4. 관련 시 재현 정보와 샘플 출력을 포함해 Pull Request를 제출합니다.

대규모 변경(새 학습 경로, 리팩터링, 의존성 변경)을 계획한다면 먼저 이슈를 열어 방향을 맞춰 주세요.

## 📚 인용

이 저장소를 연구에 사용한다면 관련 논문 및/또는 저장소를 인용해 주세요.

현재 README 텍스트에서 확인 가능한 저장소 메타데이터:
- 논문 제목: *Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation*

저장소 BibTeX 플레이스홀더(필요 시 저자/URL 업데이트):

```bibtex
@misc{soft-event-frame-alignment,
  title        = {Soft Event-Frame Alignment},
  note         = {Code repository for "Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation"},
  year         = {2025},
  howpublished = {GitHub repository}
}
```

## 🙏 감사의 말

- `dv-processing`을 포함한 이벤트 카메라 툴링 생태계.
- 실험 전반에 사용된 PyTorch 및 과학 Python 라이브러리.

## 💡 지원

현재 저장소 메타데이터에는 지원/스폰서십 섹션이 없습니다. 원한다면 후속 업데이트에서 링크를 추가해 주세요. 그러면 이후 개정에서 유지됩니다.

## 📄 라이선스

Apache License 2.0에 따라 라이선스가 부여됩니다. [LICENSE](../LICENSE)를 참고하세요.
