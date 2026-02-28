[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

> 此為論文 **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation** 的研究原始碼。

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white&style=flat-square)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043?style=flat-square)

| 焦點 | 說明 |
| --- | --- |
| 🎯 目標 | 使用統一的隱式神經表示（INR/INN）對齊事件相機與影像相機資料流 |
| ⚙️ 主要流程 | `main.py`（訓練）+ `evaluation.py`（分析與視覺化） |
| 🧪 變體腳本 | `event_inn.py`、`frame_inn.py`、`event_deri*.py`、`event_inn_sparse.py` |

</div>

## 目錄

- [🔎 概覽](#-概覽)
- [✨ 功能特性](#-功能特性)
- [🗂️ 專案結構](#-專案結構)
- [📋 先決條件](#-先決條件)
- [⚙️ 安裝](#-安裝)
- [🚀 快速開始](#-快速開始)
- [🧪 使用方式](#-使用方式)
- [🧩 設定說明](#-設定說明)
- [🧠 數學公式](#-數學公式)
- [🧾 範例](#-範例)
- [🛠️ 開發筆記](#-開發筆記)
- [🧯 疑難排解](#-疑難排解)
- [🗺️ 發展藍圖](#-發展藍圖)
- [🤝 參與貢獻](#-參與貢獻)
- [❤️ Support](#-support)
- [📄 授權條款](#-授權條款)

## 🔎 概覽

本儲存庫提供研究原始碼，用於使用統一的隱式神經表示（INR/INN）對齊事件相機與影像相機資料流。

### 核心流程（標準路徑）

| 步驟 | 元件 | 用途 |
|---|---|---|
| 1 | `softalign/data_processing.py` | 讀取 AEDAT4 流、標準化資料、建立點樣本 |
| 2 | `softalign/implicit_model.py` | 定義共享隱式函式 + 可學習的對齊參數 |
| 3 | `softalign/training.py` + `main.py` | 聯合最佳化事件與影格重建 |
| 4 | `evaluation.py` | 視覺化對齊結果並輸出損失報告 |

除了主流程外，本專案亦包含單獨與實驗腳本，涵蓋事件-only、影格-only、稀疏化，以及以微分為基礎的 INR 變體。

## ✨ 功能特性

- 使用可學習的類仿射參數進行事件-影格對齊：`scale`、`shift_x`、`shift_y`、`shift_t`。
- 事件分支與影格分支共用隱式函式 `F(x, y, t)`。
- 以時間有限差分導數建模事件，並使用可學習的 `threshold` 與 `dt`。
- 支援 AEDAT4 讀取（`dv_processing`）與處理後的 `.npy` 工作流程。
- 內建視覺化輸出，包含損失曲線、參數演化與對齊疊圖。
- 若環境支援 CUDA，將自動使用（否則回退 `torch.cuda.is_available()` 到 CPU）。
- 提供可用於快速資料檢查與輕量級除錯的腳本：
  - `read_events.py`
  - `read_frames.py`
  - `reader.py`、`reader_norm.py`、`simple_count.py`

## 🗂️ 專案結構

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
├── event_derivative_results/             # Generated experiment outputs
├── alignment_results_*/                 # Timestamped alignment runs
├── yuqing/                              # Sample AEDAT4 files
├── events_processed.csv                  # Event summary artifact
├── frames_processed.csv                  # Frame summary artifact
└── i18n/                                # Translated README files
```

## 📋 先決條件

- 建議使用 Python 3.10+。
- 下列範例以 Linux/macOS shell 為主（可依需求調整為 Windows）。
- 可選但建議安裝 CUDA 相容 GPU 以加快訓練速度。
- AEDAT4 讀取需要 `dv_processing` 與相容系統相依套件。

## ⚙️ 安裝

目前沒有提供 `requirements.txt` 或 `pyproject.toml`，請手動安裝相依套件。

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

`event_inn_sparse.py` 的可選相依套件：

```bash
pip install scikit-learn torch-geometric torch-scatter
```

預設設備行為由是否可用 CUDA 決定，但下方指令同時支援明確使用 `--device cpu`。

## 🚀 快速開始

### 1. 訓練主要對齊模型

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

輸出會寫入：
- `data/`（`events.npy`、`frames_timestamps.npy`、`frame_points.npy`、`sample_frame.png`）
- `checkpoints/`（`model_epoch_*.pt`、`model_final.pt`、`loss_curves.png`、`parameter_history.png`）

### 2. 評估已訓練的 checkpoint

`evaluation.py` 會透過 `from implicit_model import ...` 匯入 `EventFrameAlignmentModel`，但實際模組位於 `softalign/implicit_model.py`。在專案根目錄下可用的可行指令如下：

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

此指令會產生視覺化與指標，例如 `evaluation/alignment_visualization.png` 與 `evaluation/evaluation_results.txt`。

## 🧪 使用方式

### 主要流程（`main.py`）

```bash
python main.py --help
```

主要參數：
- `--filepath`：AEDAT4 輸入路徑。
- `--duration`：從錄影讀取的秒數。
- `--data_dir`：處理後資料輸出目錄。
- `--checkpoint_dir`：checkpoint 輸出目錄。
- `--num_epochs`、`--lr`、`--batch_size`、`--lambda_reg`。
- `--device`：`cuda` 或 `cpu`。
- `--reprocess`：強制重新產生資料。

### 實驗腳本

事件單一路徑 INR：

```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

影格單一路徑 INR：

```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# or
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# or
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

導數變體：

```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

稀疏變體：

```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

單體式獨立對齊流程：

```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

其他讀取工具：

```bash
python read_events.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python read_frames.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python simple_count.py --input data/events.npy
```

## 🧩 設定說明

`softalign/implicit_model.py` 的主模型預設值：

| 參數 | 預設值 |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

`softalign/data_processing.py` 中的資料前處理目前會在訓練前對事件施加人工錯位。這是目前程式碼中的刻意設計，解讀指標時請納入考量。

訓練腳本也透過 CLI 旗標提供可調參數（例如 `--hidden_dim`、`--num_layers`、`--batch_size`），可用於消融實驗。

## 🧠 數學公式

共享的隱式網路模型為 `F(x, y, t)`。

影格分支：
- 直接以 `F(x, y, t)` 預測類強度回應。

事件分支：
1. 套用可學習的事件變換：
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. 近似時間導數：
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. 產生事件回應：
   - `sigmoid(dF/dt - threshold)`

訓練目標結合：
- 事件 MSE 損失
- 影格 MSE 損失
- 對齊參數正則化

## 🧾 範例

僅使用處理後的陣列：

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

強制使用 CPU 執行：

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

將輸出至自訂目錄：

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ 開發筆記

- 專案目前為腳本導向，尚未建立套件化中介資料。
- 儲存庫內已包含已產生成果（checkpoints/results），因此磁碟空間可能較大。
- `readme-file.md` 與 `project-structure.txt` 仍保留舊式結構假設，與目前根目錄佈局不完全一致。
- 目前尚未偵測到自動化測試或 CI 工作流程。

## 🧯 疑難排解

- `evaluation.py` 出現 `ModuleNotFoundError: No module named 'implicit_model'`：
  - 請依上述範例以 `PYTHONPATH=softalign` 執行。
- `dv_processing` 安裝／執行問題：
  - 請確認你的作業系統與 Python 版本是否有被 `dv-processing` 的 wheel / 函式庫支援。
- CUDA OOM：
  - 降低 `--batch_size`、減少 `--max_events`（若有提供）或改用 `--device cpu`。
- AEDAT4 檔案缺失：
  - 檢查 `yuqing/` 下的路徑，或透過 `--filepath`／`--aedat4_file` 提供自己的錄製檔。
- 執行間資料不一致：
  - 在更改前處理假設後，使用 `--reprocess` 重新執行。

## 🗺️ 發展藍圖

- 新增可重現的環境檔案（`requirements.txt` 或 `pyproject.toml`）。
- 統一評估與套件/模組執行時的匯入路徑。
- 新增資料前處理與模型 forward / training 檢查的自動化測試。
- 新增 lint 與 smoke test CI。
- 在 `i18n/` 下補齊更多多語系 README。

## 🤝 參與貢獻

歡迎參與貢獻。

建議流程：
1. 建立聚焦的 issue 或分支，先說明實驗變更內容。
2. 保持腳本一致性（特別是匯入、CLI 規範與輸出目錄命名）。
3. 除非變更命名／版本策略，否則盡量維持既有實驗成果行為。

## 📄 授權條款

本專案採用 Apache License 2.0 授權。完整條款請參閱 [`LICENSE`](LICENSE)。

假設：若本專案用於發表，請視需要補充引用資訊到此處與 release notes。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
