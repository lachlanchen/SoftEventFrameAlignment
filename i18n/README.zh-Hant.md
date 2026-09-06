[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# 軟事件—影格對齊

*以單一隱式神經表示，學習事件相機與影格相機樣本之間空間和時間對齊的最小研究程式碼。*

[![CI](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA043)](../LICENSE)
[![Sponsor](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)

這是 **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation** 首次公開、可複製的實作，包含標準模型、AEDAT4 前處理、訓練、評估、相依套件資料、CI，以及可重現的 CPU 合成檢查。

本倉庫不發佈錄影、可辨識人物的畫面、衍生陣列或訓練檢查點。請只使用你有權處理的資料。

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=kofi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 架構

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

事件分支先轉換座標，再近似共享 MLP 的時間導數；影格分支直接計算同一個場。尺度與導數步長採用正值參數化，避免除以零。

## 公開版本內容

| 路徑 | 用途 |
| --- | --- |
| `softalign/implicit_model.py` | 共享隱式 MLP 與可學習對齊參數 |
| `softalign/training.py` | 經驗證的資料封裝與聯合訓練迴圈 |
| `softalign/data_processing.py` | 選用 AEDAT4 讀取與秒單位前處理 |
| `softalign/synthetic.py` | 專案生成的可重現合成資料 |
| `main.py` / `evaluation.py` | 訓練與診斷命令列入口 |
| `examples/` / `tests/` | 端對端檢查與核心測試 |

歷史實驗、原始錄影、本機資料與舊檢查點均刻意排除在本版本之外。

## 安裝與驗證

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,viz]'
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py --output-dir .smoke-output --epochs 8
```

此檢查會產生檢查點、圖表與 `evaluation_results.json`。它驗證流程能以有限數值執行，不代表對齊精度。

## 使用你自己的 AEDAT4 錄影

```bash
python -m pip install -e '.[aedat,viz]'
python main.py --filepath /path/to/your-recording.aedat4 --reprocess --data_dir data --checkpoint_dir checkpoints
python evaluation.py --model_path checkpoints/model_final.pt --data_dir data --output_dir evaluation --device cpu
```

前處理會記錄 `softalign-processed/v1`，並把 AEDAT 時間戳從微秒換算成秒。合成錯位預設關閉，只有明確加入 `--synthetic-misalignment` 才會啟用。缺少相容單位資料的舊陣列會被拒絕。

大型事件集合保留在 CPU 記憶體，僅將抽樣批次移到所選裝置。建議先以 `--device cpu` 執行，確認記憶體需求後再使用 CUDA。

## 資料契約

| 檔案 | 意義 |
| --- | --- |
| `events.npy` | `N × 4`：正規化 x/y、秒、事件目標 |
| `frame_points.npy` | `M × 4`：正規化 x/y、秒、強度 |
| `preprocessing.json` | 架構版本、單位、隨機種子、轉換標記與數量 |
| `model_final.pt` | 模型狀態、參數與網路架構 |
| `evaluation_results.json` | 對有限輸入樣本計算的重建 MSE |

評估結果只是對提供陣列的診斷，不是留出集指標、基準結果或真實感測器校準證明。

## 驗證與研究邊界

- CI 執行可重現的 CPU 測試與簡短的訓練—評估流程。
- 會檢查形狀、空輸入、有限值、時間單位與裝置可用性。
- 目前損失函數只是研究基線；真實研究仍需校準真值、獨立評估、消融實驗與不確定性分析。
- 請勿在公開 GitHub Issue 附上私人錄影。

## 引用

若用於研究，請引用本倉庫。GitHub 會讀取 [CITATION.cff](../CITATION.cff) 並顯示 **Cite this repository** 面板。

```bibtex
@software{chen_soft_event_frame_alignment_2026,
  author = {Chen, Lachlan},
  title = {Soft Event-Frame Alignment: Unified Implicit Neural Representation Research Code},
  year = {2026},
  url = {https://github.com/lachlanchen/SoftEventFrameAlignment}
}
```

## 狀態

`0.1.0` 是最小研究版本，目前公開的是可從頭執行的核心訓練與評估路徑，而不是歷史實驗成果的完整封存。歡迎提交能由合成檢查或小型、權利清楚樣本重現的問題；請寫明 Python、PyTorch、裝置、隨機種子、命令與實際輸出。專案會優先保持資料單位、模型架構與檢查點來源可追溯，避免把只能依賴本機舊檔運作的結果描述為可重現結論。

重現實驗時，請一併保留原始命令、相依版本、前處理中繼資料與輸出雜湊，並另外記錄真實感測器資料的授權和採集條件。若改變座標正規化、時間單位、網路深度或取樣策略，應建立新的資料目錄和檢查點，不要覆寫舊結果。僅憑訓練損失下降或疊加圖看起來更清晰，不能證明空間與時間參數正確；可靠結論仍需外部校準真值和獨立樣本。

## 授權

採用 Apache License 2.0。請見 [LICENSE](../LICENSE)。
