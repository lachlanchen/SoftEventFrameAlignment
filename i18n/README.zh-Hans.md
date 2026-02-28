[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Soft Event-Frame Alignment

> 论文 **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation** 的代码仓库。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043)

## 🔎 概览

本仓库包含研究代码，使用统一隐式神经表示（INR/INN）对事件相机流与帧相机流进行对齐。

### 核心流水线（规范路径）

| 步骤 | 组件 | 目的 |
|---|---|---|
| 1 | `softalign/data_processing.py` | 读取 AEDAT4 流、标准化数据并创建点样本 |
| 2 | `softalign/implicit_model.py` | 定义共享隐式函数与可学习对齐参数 |
| 3 | `softalign/training.py` + `main.py` | 联合优化事件/帧重建 |
| 4 | `evaluation.py` | 可视化对齐结果并汇报损失 |

除主路径外，仓库还包含用于仅事件、仅帧、稀疏以及基于导数的 INR 变体的独立与实验脚本。

## ✨ 特性

- 使用可学习仿射风格参数进行事件-帧对齐：`scale`、`shift_x`、`shift_y`、`shift_t`。
- 事件分支与帧分支共用隐式函数 `F(x, y, t)`。
- 通过有限差分时间导数建模事件，并包含可学习参数 `threshold` 与 `dt`。
- 支持 AEDAT4 读取（`dv_processing`）以及处理后的 `.npy` 工作流。
- 内置可视化输出：损失曲线、参数演化、对齐叠加图。
- 可用时支持 CUDA（`torch.cuda.is_available()` 不可用时自动回退 CPU）。

## 🗂️ 项目结构

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

## 📋 前置要求

- 推荐 Python 3.10+。
- 下文示例基于 Linux/macOS shell（如需可自行适配 Windows）。
- 可选但推荐：支持 CUDA 的 GPU 以提高训练速度。
- 读取 AEDAT4 需要 `dv_processing` 及兼容的系统依赖。

## ⚙️ 安装

当前仓库尚无 `requirements.txt` 或 `pyproject.toml`，请手动安装依赖。

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

输出写入：
- `data/`（`events.npy`、`frames_timestamps.npy`、`frame_points.npy`、`sample_frame.png`）
- `checkpoints/`（`model_epoch_*.pt`、`model_final.pt`、`loss_curves.png`、`parameter_history.png`）

### 2. 评估已训练的 checkpoint

`evaluation.py` 通过 `from implicit_model import ...` 导入 `EventFrameAlignmentModel`，而当前启用模块位于 `softalign/implicit_model.py`。从仓库根目录执行时，实用调用方式为：

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

这会生成可视化与指标结果，例如 `evaluation/alignment_visualization.png` 和 `evaluation/evaluation_results.txt`。

## 🧪 用法

### 主流水线（`main.py`）

```bash
python main.py --help
```

关键选项：
- `--filepath`：AEDAT4 输入路径。
- `--duration`：从录制中读取的秒数。
- `--data_dir`：处理后数据输出目录。
- `--checkpoint_dir`：checkpoint 输出目录。
- `--num_epochs`、`--lr`、`--batch_size`、`--lambda_reg`。
- `--device`：`cuda` 或 `cpu`。
- `--reprocess`：强制重新生成数据。

### 实验脚本

仅事件 INR：
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

仅帧 INR：
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

单体式对齐流程：
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

## 🧩 配置说明

`softalign/implicit_model.py` 中主模型默认值：

| 参数 | 默认值 |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

`softalign/data_processing.py` 中的数据预处理目前会在训练前对事件施加一个合成错位。这是当前代码中的有意设计，解释指标时应考虑这一点。

## 🧠 数学表述

共享隐式网络建模 `F(x, y, t)`。

帧分支：
- 直接使用 `F(x, y, t)` 预测类似强度的响应。

事件分支：
1. 应用可学习事件变换：
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. 近似时间导数：
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. 生成事件响应：
   - `sigmoid(dF/dt - threshold)`

训练目标组合了：
- 事件 MSE 损失
- 帧 MSE 损失
- 对齐参数的正则项

## 🧾 示例

仅使用预处理数组：

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

强制使用 CPU：

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

评估输出到自定义目录：

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ 开发说明

- 当前仓库以脚本为中心（尚无打包元数据）。
- 仓库中包含生成产物（checkpoints/results），磁盘占用可能较大。
- `readme-file.md` 和 `project-structure.txt` 包含旧版结构假设，与当前根目录布局并非完全一致。
- 当前未检测到自动化测试或 CI 工作流。

## 🧯 故障排查

- `evaluation.py` 中出现 `ModuleNotFoundError: No module named 'implicit_model'`：
  - 按上文示例使用 `PYTHONPATH=softalign` 运行。
- `dv_processing` 安装或运行问题：
  - 确认你的平台和 Python 版本受 `dv-processing` 的 wheels/libs 支持。
- CUDA OOM：
  - 降低 `--batch_size`、减少 `--max_events`（在可用脚本中）或改用 `--device cpu`。
- 找不到 AEDAT4 文件：
  - 检查 `yuqing/` 下路径，或通过 `--filepath` / `--aedat4_file` 提供你自己的录制文件。

## 🗺️ 路线图

- 增加可复现环境文件（`requirements.txt` 或 `pyproject.toml`）。
- 统一评估与包/模块执行的导入路径。
- 为数据处理和模型前向/训练检查增加自动化测试。
- 增加 CI 用于 lint 与 smoke test。
- 在 `i18n/` 下添加多语言 README 文件。

## 🤝 贡献

欢迎贡献。

1. Fork 本仓库。
2. 创建功能分支。
3. 提交聚焦且文档完备的改动。
4. 提交 pull request，并在相关情况下附上复现细节与示例输出。

如果你计划进行较大改动（新训练路径、重构或依赖变更），请先开 issue 对齐方向。

## 📚 引用

如果你在研究中使用本仓库，请引用相关论文和/或仓库。

当前 README 文本中可用的仓库元信息：
- 论文标题：*Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation*

仓库 BibTeX 占位符（可按需更新作者/URL）：

```bibtex
@misc{soft-event-frame-alignment,
  title        = {Soft Event-Frame Alignment},
  note         = {Code repository for "Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation"},
  year         = {2025},
  howpublished = {GitHub repository}
}
```

## 🙏 致谢

- 事件相机工具生态，包括 `dv-processing`。
- PyTorch 及实验中使用的科学计算 Python 库。

## 💡 支持

当前仓库元数据中尚无支持/赞助部分。若你希望添加，可在后续更新中补充链接，后续修订会保留。

## 📄 许可证

基于 Apache License 2.0 发布。详见 [LICENSE](../LICENSE)。
