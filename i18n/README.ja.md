[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Soft Event-Frame Alignment

> 論文 **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation** のリポジトリです。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043)

## 🔎 概要

このリポジトリには、統一された陰関数ニューラル表現（INR/INN）を用いて、イベントカメラのストリームとフレームカメラのストリームをアラインするための研究コードが含まれています。

### コアパイプライン（標準経路）

| Step | Component | Purpose |
|---|---|---|
| 1 | `softalign/data_processing.py` | AEDAT4 ストリームを読み込み、データを正規化し、点サンプルを作成 |
| 2 | `softalign/implicit_model.py` | 共有陰関数と学習可能なアラインメントパラメータを定義 |
| 3 | `softalign/training.py` + `main.py` | イベント/フレーム再構成を同時最適化 |
| 4 | `evaluation.py` | アラインメントを可視化し、損失を報告 |

主経路に加えて、イベント専用・フレーム専用・スパース・導関数ベース INR 変種向けの独立スクリプトや実験スクリプトも含まれています。

## ✨ 特徴

- 学習可能なアフィン系パラメータ `scale`, `shift_x`, `shift_y`, `shift_t` を用いたイベント-フレームアラインメント。
- イベント分岐とフレーム分岐の両方で共有される陰関数 `F(x, y, t)`。
- 学習可能な `threshold` と `dt` を用いた、有限差分時間導関数によるイベントモデリング。
- AEDAT4 読み込み（`dv_processing`）と処理済み `.npy` ワークフローに対応。
- 損失曲線、パラメータ推移、アラインメント重ね合わせを可視化する出力を内蔵。
- CUDA が利用可能な場合に対応（`torch.cuda.is_available()`、不可時は CPU へフォールバック）。

## 🗂️ プロジェクト構成

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

## 📋 前提条件

- Python 3.10 以上を推奨。
- 以下のシェル例は Linux/macOS 向けです（Windows の場合は適宜読み替えてください）。
- 任意ですが、学習速度のために CUDA 対応 GPU を推奨。
- AEDAT4 読み込みには `dv_processing` と互換性のあるシステム依存関係が必要です。

## ⚙️ インストール

現在、`requirements.txt` や `pyproject.toml` は存在しないため、依存関係は手動でインストールしてください。

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

スパース実験（`event_inn_sparse.py`）向けの任意依存関係:

```bash
pip install scikit-learn torch-geometric torch-scatter
```

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

出力先:
- `data/`（`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`）
- `checkpoints/`（`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`）

### 2. 学習済みチェックポイントを評価

`evaluation.py` は `from implicit_model import ...` により `EventFrameAlignmentModel` を読み込みますが、実際のモジュールは `softalign/implicit_model.py` にあります。リポジトリルートからの実用的な実行例は次のとおりです。

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

これにより、`evaluation/alignment_visualization.png` や `evaluation/evaluation_results.txt` などの可視化・指標が生成されます。

## 🧪 使い方

### メインパイプライン（`main.py`）

```bash
python main.py --help
```

主要オプション:
- `--filepath`: AEDAT4 入力パス。
- `--duration`: 記録から読み込む秒数。
- `--data_dir`: 処理済みデータの出力ディレクトリ。
- `--checkpoint_dir`: チェックポイントの出力ディレクトリ。
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`。
- `--device`: `cuda` または `cpu`。
- `--reprocess`: データ再生成を強制。

### 実験スクリプト

イベント専用 INR:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

フレーム専用 INR:
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

スタンドアロンのモノリシックアラインメント:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

## 🧩 設定メモ

`softalign/implicit_model.py` におけるメインモデルのデフォルト値:

| Parameter | Default |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

`softalign/data_processing.py` のデータ前処理は、現在の実装では学習前にイベントへ人工的なミスアラインメントを適用します。これは現行コードで意図された挙動であり、指標を解釈する際に考慮してください。

## 🧠 数理的定式化

共有陰ネットワークは `F(x, y, t)` をモデル化します。

フレーム分岐:
- `F(x, y, t)` を用いて強度に相当する応答を直接予測。

イベント分岐:
1. 学習可能なイベント変換を適用:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. 時間導関数を近似:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. イベント応答を生成:
   - `sigmoid(dF/dt - threshold)`

学習目的は次を組み合わせます:
- イベント MSE 損失
- フレーム MSE 損失
- アラインメントパラメータに対する正則化

## 🧾 例

前処理済み配列のみを使用:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

CPU 実行を強制:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

カスタムフォルダへ評価を出力:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ 開発メモ

- リポジトリはスクリプト中心です（まだパッケージメタデータはありません）。
- 生成物（checkpoints/results）がリポジトリ内に存在するため、ディスク使用量が大きくなる場合があります。
- `readme-file.md` と `project-structure.txt` は旧来の構成前提を含み、現在のルート構成と完全には一致していません。
- 現時点で、自動テストや CI ワークフローは確認されていません。

## 🧯 トラブルシューティング

- `evaluation.py` で `ModuleNotFoundError: No module named 'implicit_model'` が出る:
  - 上記のとおり `PYTHONPATH=softalign` を付けて実行してください。
- `dv_processing` のインストール/実行時問題:
  - 利用環境と Python バージョンが `dv-processing` の wheel/lib 対応範囲にあるか確認してください。
- CUDA OOM:
  - `--batch_size` を下げる、（利用可能な箇所で）`--max_events` を減らす、または `--device cpu` で実行してください。
- AEDAT4 ファイルが見つからない:
  - `yuqing/` 配下のパスを確認するか、`--filepath` / `--aedat4_file` で独自の記録ファイルを指定してください。

## 🗺️ ロードマップ

- 再現可能な環境定義ファイル（`requirements.txt` または `pyproject.toml`）を追加。
- 評価とパッケージ/モジュール実行の import パスを統一。
- データ処理とモデルの forward/学習確認向け自動テストを追加。
- lint とスモークテストの CI を追加。
- `i18n/` 配下に多言語 README を追加。

## 🤝 コントリビューション

コントリビューションを歓迎します。

1. リポジトリを Fork。
2. feature branch を作成。
3. 焦点の絞られた、十分に文書化された変更を行う。
4. 必要に応じて再現手順とサンプル出力を添えて Pull Request を提出。

大規模な変更（新しい学習経路、リファクタリング、依存関係変更など）を予定している場合は、方向性を合わせるため先に Issue を作成してください。

## 📚 引用

研究でこのリポジトリを利用する場合は、関連論文および/またはリポジトリを引用してください。

既存 README テキストから取得できる現在のリポジトリメタデータ:
- 論文タイトル: *Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation*

リポジトリ BibTeX のプレースホルダー（必要に応じて著者/URL を更新）:

```bibtex
@misc{soft-event-frame-alignment,
  title        = {Soft Event-Frame Alignment},
  note         = {Code repository for "Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation"},
  year         = {2025},
  howpublished = {GitHub repository}
}
```

## 🙏 謝辞

- `dv-processing` を含むイベントカメラ向けツール群。
- 実験全体で利用している PyTorch および科学技術計算向け Python ライブラリ。

## 💡 サポート

現時点のリポジトリメタデータにはサポート/スポンサー情報のセクションはありません。必要な場合は、後続アップデートでリンクを追加してください。将来の改訂でも保持されます。

## 📄 ライセンス

Apache License 2.0 の下でライセンスされています。詳細は [LICENSE](../LICENSE) を参照してください。
