[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

> 論文 **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation** のリポジトリです。

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white&style=flat-square)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043?style=flat-square)

| フォーカス | 詳細 |
| --- | --- |
| 🎯 Goal | イベントカメラストリームとフレームカメラストリームを統一された陰関数ニューラル表現でアライン |
| ⚙️ Main Pipeline | `main.py`（学習） + `evaluation.py`（解析／可視化） |
| 🧪 Variant Scripts | `event_inn.py`, `frame_inn.py`, `event_deri*.py`, `event_inn_sparse.py` |

</div>

## 目次

- [🔎 概要](#-概要)
- [✨ 特徴](#-特徴)
- [🗂️ プロジェクト構造](#-プロジェクト構造)
- [📋 前提条件](#-前提条件)
- [⚙️ インストール](#-インストール)
- [🚀 クイックスタート](#-クイックスタート)
- [🧪 使用方法](#-使用方法)
- [🧩 設定メモ](#-設定メモ)
- [🧠 数学的定式化](#-数学的定式化)
- [🧾 例](#-例)
- [🛠️ 開発メモ](#-開発メモ)
- [🧯 トラブルシューティング](#-トラブルシューティング)
- [🗺️ ロードマップ](#-ロードマップ)
- [🤝 コントリビューション](#-コントリビューション)
- [❤️ Support](#-support)
- [📄 ライセンス](#-ライセンス)

## 🔎 概要

このリポジトリには、統一された陰関数ニューラル表現（INR/INN）を用いて、イベントカメラとフレームカメラのストリームを合わせるための研究コードが含まれています。

### コアパイプライン（標準ルート）

| Step | Component | Purpose |
|---|---|---|
| 1 | `softalign/data_processing.py` | AEDAT4 ストリームを読み込み、データを正規化し、点サンプルを作成 |
| 2 | `softalign/implicit_model.py` | 共有の陰関数 + 学習可能なアラインメントパラメータを定義 |
| 3 | `softalign/training.py` + `main.py` | イベント再構成とフレーム再構成を同時に最適化 |
| 4 | `evaluation.py` | アラインメントの可視化と損失レポート |

メイン経路に加えて、このリポジトリにはイベントのみ、フレームのみ、スパース、微分ベースの INR 版を扱う単体／実験スクリプトも含まれています。

## ✨ 特徴

- 学習可能なアフィン型パラメータ `scale`, `shift_x`, `shift_y`, `shift_t` によるイベント-フレーム整列。
- イベント分岐とフレーム分岐の両方で共有される陰関数 `F(x, y, t)`。
- 学習可能な `threshold` と `dt` を使った有限差分時間微分でイベントをモデリング。
- AEDAT4 取り込み（`dv_processing`）と、処理済み `.npy` ワークフローに対応。
- 損失曲線、パラメータ推移、アラインメント重ね合わせを可視化するための組み込み出力。
- CUDA 利用可能時は CUDA を使用（`torch.cuda.is_available()` で CPU へフォールバック）。
- データ確認と軽量デバッグ向けのデータセット用スクリプト:
  - `read_events.py`
  - `read_frames.py`
  - `reader.py`, `reader_norm.py`, `simple_count.py`

## 🗂️ プロジェクト構造

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

## 📋 前提条件

- Python 3.10+ を推奨します。
- 以下のシェル例は Linux/macOS 向けです（必要に応じて Windows 向けに調整してください）。
- 任意ですが、学習速度向上のため CUDA 対応 GPU を推奨します。
- AEDAT4 読み込みには `dv_processing` と互換性のあるシステム依存の前提条件が必要です。

## ⚙️ インストール

現在 `requirements.txt` も `pyproject.toml` も存在しないため、依存関係は手動でインストールします。

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

`event_inn_sparse.py` 向けの任意依存:

```bash
pip install scikit-learn torch-geometric torch-scatter
```

CUDA の利用可否がデフォルトデバイス動作を決定しますが、以下のコマンドは明示的に `--device cpu` を指定して実行することもできます。

## 🚀 クイックスタート

### 1. メインのアラインメントモデルを学習

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

出力は以下に保存されます:
- `data/`（`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`）
- `checkpoints/`（`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`）

### 2. 学習済みチェックポイントを評価

`evaluation.py` は `from implicit_model import ...` で `EventFrameAlignmentModel` を読み込みますが、実際の実体は `softalign/implicit_model.py` にあります。リポジトリルートからの実用例は次の通りです。

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

これにより `evaluation/alignment_visualization.png` や `evaluation/evaluation_results.txt` といった可視化・指標が生成されます。

## 🧪 使用方法

### メインパイプライン（`main.py`）

```bash
python main.py --help
```

主なオプション:
- `--filepath`: AEDAT4 入力パス。
- `--duration`: 録画から読み込む秒数。
- `--data_dir`: 処理済みデータの出力ディレクトリ。
- `--checkpoint_dir`: チェックポイントの出力ディレクトリ。
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` または `cpu`。
- `--reprocess`: データ再生成を強制します。

### 実験用スクリプト

イベントのみの INR:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

フレームのみの INR:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# or
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# or
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

導関数バリアント:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

スパースバリアント:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

スタンドアロン（単体）アラインメント:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

追加のリーダースクリプト:
```bash
python read_events.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python read_frames.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python simple_count.py --input data/events.npy
```

## 🧩 設定メモ

`softalign/implicit_model.py` のメインモデル既定値:

| Parameter | Default |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

`softalign/data_processing.py` のデータ前処理は、現在の実装では学習前にイベントへ意図的なミスアラインメントを適用します。これは現行コードの意図した動作で、指標解釈の際に考慮してください。

学習スクリプトは CLI フラグを通じて追加の調整可能なパラメータも公開しています（例: `--hidden_dim`, `--num_layers`, `--batch_size`）。アブレーション実験で有効です。

## 🧠 数学的定式化

共有陰関数ネットワークが `F(x, y, t)` をモデル化します。

フレーム分岐:
- `F(x, y, t)` を用いて強度に近い応答を直接予測します。

イベント分岐:
1. 学習可能なイベント変換を適用:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. 時間微分を近似:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. イベント応答を生成:
   - `sigmoid(dF/dt - threshold)`

学習目的は以下を統合します:
- イベント MSE 損失
- フレーム MSE 損失
- アラインメントパラメータの正則化

## 🧾 例

前処理済み配列のみを使う:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

CPU 実行を強制する:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

カスタムディレクトリへ評価結果を出力する:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ 開発メモ

- リポジトリはスクリプト中心で、まだパッケージングのメタデータはありません。
- 生成物（checkpoints/results）はリポジトリ内に存在するため、ディスク使用量が大きくなる場合があります。
- `readme-file.md` と `project-structure.txt` は過去の構成前提を含んでおり、現行ルートレイアウトと完全には一致していません。
- 現時点では自動テストや CI ワークフローは検出されていません。

## 🧯 トラブルシューティング

- `evaluation.py` で `ModuleNotFoundError: No module named 'implicit_model'` が出る場合:
  - 上記のように `PYTHONPATH=softalign` を付けて実行してください。
- `dv_processing` のインストール／実行問題:
  - 使用しているプラットフォームと Python バージョンが `dv-processing` の wheel／ライブラリでサポートされているか確認してください。
- CUDA OOM:
  - `--batch_size` を小さくする、`--max_events`（利用可能な場合）を減らす、または `--device cpu` で実行する。
- AEDAT4 ファイルが見つからない:
  - `yuqing/` 配下のパスを確認するか、`--filepath` / `--aedat4_file` で独自の記録を指定する。
- 実行ごとのデータ不一致:
  - 前処理条件を変更した場合は `--reprocess` を再実行してください。

## 🗺️ ロードマップ

- 再現性のある環境設定ファイル（`requirements.txt` または `pyproject.toml`）を追加。
- 評価とパッケージ／モジュール実行時のインポートパスを統一。
- データ処理とモデル forward/学習の確認用自動テストを追加。
- lint とスモークテスト用 CI を追加。
- `i18n/` 配下への多言語 README 追加。

## 🤝 コントリビューション

コントリビューション歓迎です。

推奨ワークフロー:
1. 実験変更を要約した issue もしくはブランチを作成する。
2. スクリプトの整合性（特にインポート、CLI 規約、出力ディレクトリ名）を保つ。
3. 既存の実験成果物の命名規則とバージョン付けを維持する。

## 📄 ライセンス

本リポジトリは Apache License 2.0 の下で公開されています。全文は [`LICENSE`](../LICENSE) を参照してください。

公開用途でこのリポジトリを引用する場合は、必要に応じてここおよびリリースノートへ引用情報を追加してください。


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
