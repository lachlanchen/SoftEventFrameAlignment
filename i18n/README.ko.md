[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

*하나의 암시적 신경 표현으로 이벤트 카메라와 프레임 카메라 샘플 사이의 공간·시간 정렬을 학습하는 최소 연구 코드입니다.*

[![CI](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA043)](../LICENSE)
[![Sponsor](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)

이 저장소는 **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**의 첫 공개·복제 가능한 구현입니다. 표준 모델, AEDAT4 전처리, 학습, 평가, 의존성 메타데이터, CI, 결정적 합성 CPU 검사를 포함합니다.

녹화 원본, 식별 가능한 프레임, 파생 배열, 학습된 체크포인트는 공개하지 않습니다. 사용할 권리가 있는 데이터만 사용하세요.

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=kofi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 구조

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

이벤트 분기는 좌표를 변환하고 공유 MLP의 시간 미분을 근사합니다. 프레임 분기는 같은 필드를 직접 평가합니다. 스케일과 미분 간격은 양수로 매개변수화되어 0으로 나누는 일을 막습니다.

## 공개 릴리스 내용

| 경로 | 역할 |
| --- | --- |
| `softalign/implicit_model.py` | 공유 암시적 MLP와 학습 가능한 정렬 |
| `softalign/training.py` | 검증된 데이터 래퍼와 공동 학습 루프 |
| `softalign/data_processing.py` | 선택적 AEDAT4 입력과 초 단위 전처리 |
| `softalign/synthetic.py` | 프로젝트가 생성한 결정적 합성 데이터 |
| `main.py` / `evaluation.py` | 학습 및 진단 CLI |
| `examples/` / `tests/` | 종단 간 확인과 코어 테스트 |

과거 실험, 원본 녹화, 로컬 데이터셋, 이전 체크포인트는 의도적으로 제외했습니다.

## 설치와 확인

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,viz]'
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py --output-dir .smoke-output --epochs 8
```

이 검사는 체크포인트, 그래프, `evaluation_results.json`을 만듭니다. 파이프라인이 유한한 값으로 실행되는지를 확인할 뿐, 정렬 정확도의 증거는 아닙니다.

## 자신의 AEDAT4 녹화로 실행

```bash
python -m pip install -e '.[aedat,viz]'
python main.py --filepath /path/to/your-recording.aedat4 --reprocess --data_dir data --checkpoint_dir checkpoints
python evaluation.py --model_path checkpoints/model_final.pt --data_dir data --output_dir evaluation --device cpu
```

전처리는 `softalign-processed/v1` 스키마를 기록하고 AEDAT 마이크로초 타임스탬프를 초로 변환합니다. 합성 오정렬은 기본적으로 꺼져 있으며 `--synthetic-misalignment`를 명시할 때만 켜집니다. 단위 메타데이터가 맞지 않는 이전 배열은 거부합니다.

큰 이벤트 집합은 CPU 메모리에 남고 선택된 배치만 대상 장치로 이동합니다. 먼저 `--device cpu`로 실행하고 메모리를 확인한 뒤 CUDA를 사용하세요.

## 데이터 계약

| 파일 | 의미 |
| --- | --- |
| `events.npy` | `N × 4`: 정규화된 x/y, 초 단위 시간, 이벤트 목표 |
| `frame_points.npy` | `M × 4`: 정규화된 x/y, 초 단위 시간, 강도 |
| `preprocessing.json` | 스키마, 단위, seed, 변환 플래그, 표본 수 |
| `model_final.pt` | 모델 상태, 매개변수, 아키텍처 |
| `evaluation_results.json` | 제한된 입력 표본의 재구성 MSE |

평가값은 제공된 배열의 진단이며 독립 테스트 지표, 벤치마크, 실제 센서 보정의 증거가 아닙니다.

## 검증과 연구 한계

- CI는 결정적 CPU 테스트와 짧은 학습·평가 주기를 실행합니다.
- 형태, 빈 입력, 유한값, 시간 단위, 장치 가용성을 검사합니다.
- 현재 손실은 연구 기준선입니다. 실제 연구에는 보정 정답, 분리된 평가, 절제 실험, 불확실성 분석이 필요합니다.
- 비공개 녹화를 공개 GitHub 이슈에 첨부하지 마세요.

## 인용

연구에서 사용한다면 저장소를 인용해 주세요. GitHub는 [CITATION.cff](../CITATION.cff)를 읽고 **Cite this repository** 패널을 표시합니다.

```bibtex
@software{chen_soft_event_frame_alignment_2026,
  author = {Chen, Lachlan},
  title = {Soft Event-Frame Alignment: Unified Implicit Neural Representation Research Code},
  year = {2026},
  url = {https://github.com/lachlanchen/SoftEventFrameAlignment}
}
```

## 상태

버전 `0.1.0`은 최소 연구 릴리스입니다. 합성 검사나 사용 권한이 확인된 작은 표본에서 재현되는 문제를 환영합니다. 현재 공개 범위는 핵심 학습·평가 경로이며 과거 실험 결과를 포함하지 않습니다.

## 라이선스

Apache License 2.0을 적용합니다. [LICENSE](../LICENSE)를 참조하세요.
