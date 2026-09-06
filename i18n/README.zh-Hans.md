[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 软事件—帧对齐

*用一个统一的隐式神经表示，学习事件相机与帧相机样本之间空间和时间对齐的最小研究代码。*

[![CI](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA043)](../LICENSE)
[![Sponsor](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)

这是 **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation** 的首个公开、可克隆实现，包含标准模型、AEDAT4 预处理、训练、评估、依赖元数据、CI，以及确定性的 CPU 合成检查。

本仓库不发布录制文件、可识别人物的画面、派生数组或训练检查点。请只使用你有权处理的数据。

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=kofi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 架构

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

事件分支先变换坐标，再近似共享 MLP 的时间导数；帧分支直接计算同一个场。尺度和导数步长采用正值参数化，从而避免除以零。

## 公开版本内容

| 路径 | 用途 |
| --- | --- |
| `softalign/implicit_model.py` | 共享隐式 MLP 与可学习对齐参数 |
| `softalign/training.py` | 带校验的数据封装与联合训练循环 |
| `softalign/data_processing.py` | 可选 AEDAT4 读取及秒单位预处理 |
| `softalign/synthetic.py` | 项目生成的确定性合成数据 |
| `main.py` / `evaluation.py` | 训练与诊断命令行入口 |
| `examples/` / `tests/` | 端到端检查与核心测试 |

历史实验、原始录制、本地数据和旧检查点均有意排除在本版本之外。

## 安装与验证

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,viz]'
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py --output-dir .smoke-output --epochs 8
```

检查会生成检查点、图表和 `evaluation_results.json`。它验证流程能以有限数值运行，不代表对齐精度。

## 使用你自己的 AEDAT4 录制

```bash
python -m pip install -e '.[aedat,viz]'
python main.py --filepath /path/to/your-recording.aedat4 --reprocess --data_dir data --checkpoint_dir checkpoints
python evaluation.py --model_path checkpoints/model_final.pt --data_dir data --output_dir evaluation --device cpu
```

预处理会记录 `softalign-processed/v1`，并把 AEDAT 时间戳从微秒换算为秒。合成错位默认关闭，只有显式加入 `--synthetic-misalignment` 才会启用。缺少兼容单位元数据的旧数组会被拒绝。

大型事件集合保留在 CPU 内存中，仅把抽样批次移到所选设备。建议先以 `--device cpu` 运行，确认显存需求后再使用 CUDA。

## 数据约定

| 文件 | 含义 |
| --- | --- |
| `events.npy` | `N × 4`：归一化 x/y、秒、事件目标 |
| `frame_points.npy` | `M × 4`：归一化 x/y、秒、强度 |
| `preprocessing.json` | 架构版本、单位、随机种子、变换标记与数量 |
| `model_final.pt` | 模型状态、参数与网络结构 |
| `evaluation_results.json` | 对有限输入样本计算的重建 MSE |

评估结果只是对所提供数组的诊断，不是留出集指标、基准结果或真实传感器校准证明。

## 验证与研究边界

- CI 执行确定性 CPU 测试和简短的训练—评估流程。
- 会检查形状、空输入、有限值、时间单位和设备可用性。
- 当前损失函数只是研究基线；真实研究仍需校准真值、独立评估、消融实验与不确定性分析。
- 请勿在公开 GitHub Issue 中附加私人录制。

## 引用

若用于研究，请引用本仓库。GitHub 会读取 [CITATION.cff](../CITATION.cff) 并显示 **Cite this repository** 面板。

```bibtex
@software{chen_soft_event_frame_alignment_2026,
  author = {Chen, Lachlan},
  title = {Soft Event-Frame Alignment: Unified Implicit Neural Representation Research Code},
  year = {2026},
  url = {https://github.com/lachlanchen/SoftEventFrameAlignment}
}
```

## 状态

`0.1.0` 是最小研究版本，目前公开的是可从头运行的核心训练与评估路径，而不是历史实验成果的完整归档。欢迎提交能由合成检查或小型、权利清晰样本复现的问题；请写明 Python、PyTorch、设备、随机种子、命令和实际输出。项目会优先保持数据单位、模型结构和检查点来源可追溯，避免把仅能在本地旧文件上运行的结果描述成可复现结论。

复现实验时，请一并保留原始命令、依赖版本、预处理元数据与输出哈希，并单独记录真实传感器数据的授权和采集条件。若改变坐标归一化、时间单位、网络深度或采样策略，应建立新的数据目录和检查点，不要覆盖旧结果。仅凭训练损失下降或叠加图看起来更清晰，不能证明空间与时间参数正确；可靠结论仍需外部校准真值和独立样本。

## 许可

采用 Apache License 2.0。参见 [LICENSE](../LICENSE)。
