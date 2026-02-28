[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

> 本仓库是论文 **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation** 的研究代码。

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white&style=flat-square)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043?style=flat-square)

| 关注点 | 详情 |
| --- | --- |
| 🎯 目标 | 使用统一的隐式神经表示（INR/INN）对事件相机流与帧相机流进行对齐 |
| ⚙️ 主流程 | `main.py`（训练） + `evaluation.py`（分析与可视化） |
| 🧪 变体脚本 | `event_inn.py`、`frame_inn.py`、`event_deri*.py`、`event_inn_sparse.py` |

</div>

## 目录

- [🔎 概览](#-概览)
- [✨ 特性](#-特性)
- [🗂️ 项目结构](#-项目结构)
- [📋 前置条件](#-前置条件)
- [⚙️ 安装](#-安装)
- [🚀 快速开始](#-快速开始)
- [🧪 使用说明](#-使用说明)
- [🧩 配置说明](#-配置说明)
- [🧠 数学公式](#-数学公式)
- [🧾 示例](#-示例)
- [🛠️ 开发说明](#-开发说明)
- [🧯 故障排查](#-故障排查)
- [🗺️ 路线图](#-路线图)
- [🤝 贡献](#-贡献)
- [❤️ Support](#-support)
- [📄 许可证](#-许可证)

## 🔎 概览

本仓库包含用于将事件相机流与帧相机流通过统一的隐式神经表示（INR/INN）进行对齐的研究代码。

### 核心流程（标准路径）

| 步骤 | 组件 | 用途 |
|---|---|---|
| 1 | `softalign/data_processing.py` | 读取 AEDAT4 流、归一化数据、创建点样本 |
| 2 | `softalign/implicit_model.py` | 定义共享隐式函数 + 可学习对齐参数 |
| 3 | `softalign/training.py` + `main.py` | 联合优化事件重建与帧重建 |
| 4 | `evaluation.py` | 可视化对齐结果并报告损失 |

除了主路径外，仓库还包含独立和实验脚本，覆盖事件仅、帧仅、稀疏以及基于导数的 INR 变体。

## ✨ 特性

- 使用可学习的仿射风格参数进行事件-帧对齐：`scale`、`shift_x`、`shift_y`、`shift_t`。
- 事件分支与帧分支共享隐式函数 `F(x, y, t)`。
- 通过时间有限差分导数建模事件，并使用可学习的 `threshold` 与 `dt`。
- 支持 AEDAT4 摄入（`dv_processing`）及处理后的 `.npy` 工作流。
- 内置可视化输出，包括损失曲线、参数演化和对齐叠加图。
- 如可用，支持 CUDA（若 `torch.cuda.is_available()` 不可用则回退到 CPU）。
- 提供用于快速检查数据与轻量级调试的数据脚本：
  - `read_events.py`
  - `read_frames.py`
  - `reader.py`、`reader_norm.py`、`simple_count.py`

## 🗂️ 项目结构

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

## 📋 前置条件

- 建议使用 Python 3.10+。
- 以下示例基于 Linux/macOS shell（可按需适配 Windows）。
- 可选但推荐：具备 CUDA 的 GPU 以加快训练速度。
- AEDAT4 读取需要 `dv_processing` 及兼容的系统依赖。

## ⚙️ 安装

当前仓库尚未提供 `requirements.txt` 或 `pyproject.toml`，请手动安装依赖。

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

稀疏实验（`event_inn_sparse.py`）的可选依赖：

```bash
pip install scikit-learn torch-geometric torch-scatter
```

默认设备行为由 CUDA 是否可用决定，但以下命令同样支持显式设置 `--device cpu`。

## 🚀 快速开始

### 1. 训练主对齐模型

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

输出会写入：
- `data/`（`events.npy`、`frames_timestamps.npy`、`frame_points.npy`、`sample_frame.png`）
- `checkpoints/`（`model_epoch_*.pt`、`model_final.pt`、`loss_curves.png`、`parameter_history.png`）

### 2. 评估已训练 checkpoint

`evaluation.py` 通过 `from implicit_model import ...` 导入 `EventFrameAlignmentModel`，但实际模块位于 `softalign/implicit_model.py`。从仓库根目录执行的可行命令如下：

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

该命令会生成可视化与指标，例如 `evaluation/alignment_visualization.png` 与 `evaluation/evaluation_results.txt`。

## 🧪 使用说明

### 主流程（`main.py`）

```bash
python main.py --help
```

主要参数：
- `--filepath`：AEDAT4 输入路径。
- `--duration`：从录制中读取的秒数。
- `--data_dir`：处理后数据输出目录。
- `--checkpoint_dir`：checkpoint 输出目录。
- `--num_epochs`、`--lr`、`--batch_size`、`--lambda_reg`。
- `--device`：`cuda` 或 `cpu`。
- `--reprocess`：强制重新生成数据。

### 实验脚本

事件单独 INR：
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

帧单独 INR：
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# or
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# or
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

导数变体：
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

稀疏变体：
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

独立单体对齐脚本：
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

额外读取脚本：
```bash
python read_events.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python read_frames.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python simple_count.py --input data/events.npy
```

## 🧩 配置说明

`softalign/implicit_model.py` 中的主模型默认值：

| 参数 | 默认值 |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

`softalign/data_processing.py` 中的数据预处理当前会在训练前对事件施加人为错位。这是当前代码中的故意设计，解读指标时应予以考虑。

训练脚本还通过 CLI 标志暴露了可调参数（例如 `--hidden_dim`、`--num_layers`、`--batch_size`），可用于消融实验。

## 🧠 数学公式

共享隐式网络模型为 `F(x, y, t)`。

帧分支：
- 直接使用 `F(x, y, t)` 预测类似强度的响应。

事件分支：
1. 应用可学习的事件变换：
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. 近似时间导数：
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. 生成事件响应：
   - `sigmoid(dF/dt - threshold)`

训练目标为：
- 事件 MSE 损失
- 帧 MSE 损失
- 对齐参数的正则化

## 🧾 示例

仅使用处理后的数组：

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

强制 CPU 运行：

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

评估到自定义文件夹：

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ 开发说明

- 仓库以脚本为中心（尚未建立打包元数据）。
- 仓库内存在已生成的产物（checkpoints/results），因此磁盘占用可能较大。
- `readme-file.md` 和 `project-structure.txt` 包含历史结构假设，与当前仓库根目录布局未完全对齐。
- 当前未检测到自动化测试或 CI 工作流。

## 🧯 故障排查

- `evaluation.py` 出现 `ModuleNotFoundError: No module named 'implicit_model'`：
  - 按上方示例使用 `PYTHONPATH=softalign` 运行。
- `dv_processing` 安装/运行问题：
  - 确认你的平台和 Python 版本是否受 `dv-processing` 的 wheel/库支持。
- CUDA OOM：
  - 降低 `--batch_size`，减少 `--max_events`（如有提供），或改用 `--device cpu`。
- AEDAT4 文件缺失：
  - 检查 `yuqing/` 下的路径，或通过 `--filepath` / `--aedat4_file` 提供你自己的录制文件。
- 运行间数据不一致：
  - 改变预处理假设后，使用 `--reprocess` 重新运行。

## 🗺️ 路线图

- 增加可复现环境文件（`requirements.txt` 或 `pyproject.toml`）。
- 统一用于评估和包/模块执行的导入路径。
- 新增用于数据处理和模型前向/训练检查的自动化测试。
- 增加 lint 与 smoke test 的 CI。
- 在 `i18n/` 下补充多语言 README 文件。

## 🤝 贡献

欢迎贡献。

推荐流程：
1. 创建聚焦的 issue 或分支，说明实验变更内容。
2. 保持脚本一致性（尤其是导入、CLI 约定与输出目录命名）。
3. 除非更改命名/版本策略，否则尽量保持现有实验产物行为。

## 📄 许可证

该仓库基于 Apache License 2.0 发布。完整文本见 [`../LICENSE`](../LICENSE)。

假设：若本仓库用于发布或论文发表，请根据需要在此处及 release notes 中补充引用信息。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
