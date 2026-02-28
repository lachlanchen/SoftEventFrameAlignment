[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Soft Event-Frame Alignment

> 論文 **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation** 的程式碼倉庫。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043)

## 🔎 概述

此倉庫包含研究程式碼，使用統一的隱式神經表示（INR/INN）對事件相機串流與影格相機串流進行對齊。

### 核心流程（Canonical Path）

| 步驟 | 元件 | 用途 |
|---|---|---|
| 1 | `softalign/data_processing.py` | 讀取 AEDAT4 串流、正規化資料、建立點樣本 |
| 2 | `softalign/implicit_model.py` | 定義共享隱式函數 + 可學習對齊參數 |
| 3 | `softalign/training.py` + `main.py` | 聯合最佳化事件/影格重建 |
| 4 | `evaluation.py` | 視覺化對齊結果並回報損失 |

除了主流程外，倉庫亦包含獨立與實驗性腳本，涵蓋僅事件、僅影格、稀疏及導數式 INR 變體。

## ✨ 功能特色

- 事件-影格對齊使用可學習仿射風格參數：`scale`、`shift_x`、`shift_y`、`shift_t`。
- 事件分支與影格分支共用隱式函數 `F(x, y, t)`。
- 事件建模透過有限差分時間導數，並使用可學習 `threshold` 與 `dt`。
- 支援 AEDAT4 載入（`dv_processing`）與處理後 `.npy` 工作流。
- 內建視覺化輸出：損失曲線、參數演化與對齊疊圖。
- 可用時支援 CUDA（`torch.cuda.is_available()` 不可用時回退 CPU）。

## 🗂️ 專案結構

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

## 📋 先決條件

- 建議使用 Python 3.10+。
- 下方範例採用 Linux/macOS shell（Windows 請視情況調整）。
- 可選但建議：使用支援 CUDA 的 GPU 以提升訓練速度。
- AEDAT4 讀取需要 `dv_processing` 與相容的系統相依套件。

## ⚙️ 安裝

目前尚未提供 `requirements.txt` 或 `pyproject.toml`，請手動安裝相依套件。

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

稀疏實驗（`event_inn_sparse.py`）的可選相依套件：

```bash
pip install scikit-learn torch-geometric torch-scatter
```

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

輸出將寫入：
- `data/`（`events.npy`、`frames_timestamps.npy`、`frame_points.npy`、`sample_frame.png`）
- `checkpoints/`（`model_epoch_*.pt`、`model_final.pt`、`loss_curves.png`、`parameter_history.png`）

### 2. 評估已訓練的 checkpoint

`evaluation.py` 透過 `from implicit_model import ...` 匯入 `EventFrameAlignmentModel`，而目前啟用的模組位於 `softalign/implicit_model.py`。在倉庫根目錄可使用以下實務指令：

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

這會產生視覺化與指標輸出，例如 `evaluation/alignment_visualization.png` 與 `evaluation/evaluation_results.txt`。

## 🧪 使用方式

### 主流程（`main.py`）

```bash
python main.py --help
```

主要選項：
- `--filepath`：AEDAT4 輸入路徑。
- `--duration`：從錄影中讀取的秒數。
- `--data_dir`：處理後資料輸出目錄。
- `--checkpoint_dir`：checkpoint 輸出目錄。
- `--num_epochs`、`--lr`、`--batch_size`、`--lambda_reg`。
- `--device`：`cuda` 或 `cpu`。
- `--reprocess`：強制重新產生資料。

### 實驗腳本

僅事件 INR：
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

僅影格 INR：
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

獨立單體對齊流程：
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

## 🧩 設定說明

主模型在 `softalign/implicit_model.py` 的預設值：

| 參數 | 預設值 |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

`softalign/data_processing.py` 的資料前處理目前會在訓練前對事件資料套用人工錯位。此為現行程式中的刻意設計，解讀指標時請納入考量。

## 🧠 數學形式化

共享隱式網路建模 `F(x, y, t)`。

影格分支：
- 使用 `F(x, y, t)` 直接預測類強度回應。

事件分支：
1. 套用可學習事件轉換：
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

僅使用預處理陣列：

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

強制使用 CPU：

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

評估到自訂資料夾：

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ 開發備註

- 倉庫目前以腳本為中心（尚無封裝中繼資料）。
- 倉庫內含已產生的產物（checkpoints/results），磁碟占用可能較大。
- `readme-file.md` 與 `project-structure.txt` 含有舊版結構假設，與目前根目錄佈局未完全一致。
- 目前未發現自動化測試或 CI 工作流程。

## 🧯 疑難排解

- `evaluation.py` 出現 `ModuleNotFoundError: No module named 'implicit_model'`：
  - 請依上方示例以 `PYTHONPATH=softalign` 執行。
- `dv_processing` 安裝或執行階段問題：
  - 確認你的平台與 Python 版本受 `dv-processing` wheels/libs 支援。
- CUDA OOM：
  - 降低 `--batch_size`、減少 `--max_events`（可用時），或改用 `--device cpu`。
- 找不到 AEDAT4 檔案：
  - 確認 `yuqing/` 路徑，或透過 `--filepath` / `--aedat4_file` 提供你自己的錄製檔。

## 🗺️ Roadmap

- 加入可重現環境檔（`requirements.txt` 或 `pyproject.toml`）。
- 統一評估與 package/module 執行的匯入路徑。
- 新增資料處理與模型 forward/訓練檢查的自動化測試。
- 新增 lint 與 smoke tests 的 CI。
- 在 `i18n/` 下新增多語 README 檔案。

## 🤝 貢獻

歡迎提交貢獻。

1. Fork 此倉庫。
2. 建立功能分支。
3. 進行聚焦且有完整說明的變更。
4. 提交 pull request，並在適用時附上重現細節與範例輸出。

若計畫進行大型變更（新訓練路徑、重構或相依套件變更），請先開 issue 對齊方向。

## 📚 引用

若你在研究中使用此倉庫，請引用相關論文與/或本倉庫。

目前可由現有 README 文字取得的倉庫中繼資訊：
- 論文標題：*Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation*

倉庫 BibTeX 佔位（請依需要更新作者/URL）：

```bibtex
@misc{soft-event-frame-alignment,
  title        = {Soft Event-Frame Alignment},
  note         = {Code repository for "Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation"},
  year         = {2025},
  howpublished = {GitHub repository}
}
```

## 🙏 致謝

- 事件相機工具生態系（包含 `dv-processing`）。
- 實驗中使用的 PyTorch 與科學 Python 函式庫。

## 💡 支援

目前倉庫中繼資料尚未提供支援/贊助章節。若需要，可在後續更新加入連結，未來修訂將予以保留。

## 📄 授權

本專案採用 Apache License 2.0。請參閱 [LICENSE](LICENSE)。
