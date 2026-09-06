[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

*単一の陰的ニューラル表現を用いて、イベントカメラとフレームカメラの標本間の空間・時間アラインメントを学習する最小構成の研究コードです。*

[![CI](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA043)](../LICENSE)
[![Sponsor](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)

これは **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation** の、初めて公開するクローン可能な実装です。標準モデル、AEDAT4 前処理、学習、評価、依存関係、CI、決定的な合成 CPU チェックを含みます。

録画、人物を識別できるフレーム、派生配列、学習済みチェックポイントは公開しません。利用権限のあるデータだけを使用してください。

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=kofi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 構成

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

イベント側は座標を変換し、共有 MLP の時間微分を近似します。フレーム側は同じ場を直接評価します。スケールと微分刻みは正にパラメータ化され、ゼロ除算を防ぎます。

## 公開内容

| パス | 役割 |
| --- | --- |
| `softalign/implicit_model.py` | 共有陰的 MLP と学習可能なアラインメント |
| `softalign/training.py` | 検証付きデータセットと共同学習ループ |
| `softalign/data_processing.py` | 任意の AEDAT4 読み込みと秒単位の前処理 |
| `softalign/synthetic.py` | プロジェクト生成の決定的な合成データ |
| `main.py` / `evaluation.py` | 学習・診断 CLI |
| `examples/` / `tests/` | エンドツーエンド確認とコアテスト |

過去の実験、元録画、ローカルデータ、旧チェックポイントは意図的に含めていません。

## インストールと確認

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,viz]'
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py --output-dir .smoke-output --epochs 8
```

この確認はチェックポイント、図、`evaluation_results.json` を生成します。パイプラインが有限値で実行できることを確認するもので、アラインメント精度の証明ではありません。

## 自分の AEDAT4 録画で実行

```bash
python -m pip install -e '.[aedat,viz]'
python main.py --filepath /path/to/your-recording.aedat4 --reprocess --data_dir data --checkpoint_dir checkpoints
python evaluation.py --model_path checkpoints/model_final.pt --data_dir data --output_dir evaluation --device cpu
```

前処理は `softalign-processed/v1` を記録し、AEDAT のマイクロ秒時刻を秒へ変換します。合成ずれは既定で無効で、`--synthetic-misalignment` を明示した場合だけ有効です。単位メタデータが合わない旧配列は拒否されます。

大きなイベント集合は CPU メモリに保持し、抽出したバッチだけを選択デバイスへ送ります。まず `--device cpu` で確認し、メモリを調べてから CUDA を使用してください。

## データ契約

| ファイル | 意味 |
| --- | --- |
| `events.npy` | `N × 4`：正規化 x/y、秒、イベント目標 |
| `frame_points.npy` | `M × 4`：正規化 x/y、秒、強度 |
| `preprocessing.json` | スキーマ、単位、seed、変換フラグ、件数 |
| `model_final.pt` | モデル状態、パラメータ、構成 |
| `evaluation_results.json` | 供給標本上の限定的な再構成 MSE |

評価値は供給配列の診断であり、独立テスト指標、ベンチマーク、物理センサー校正の証明ではありません。

## 検証と研究上の限界

- CI は決定的 CPU テストと短い学習・評価サイクルを実行します。
- 形状、空データ、有限値、時刻単位、デバイス可用性を検査します。
- 現在の損失は研究ベースラインです。実験には校正の正解、独立評価、アブレーション、不確実性解析が必要です。
- 非公開録画を公開 GitHub Issue に添付しないでください。

## 引用

研究で利用する場合はリポジトリを引用してください。GitHub は [CITATION.cff](../CITATION.cff) を読み、**Cite this repository** を表示します。

```bibtex
@software{chen_soft_event_frame_alignment_2026,
  author = {Chen, Lachlan},
  title = {Soft Event-Frame Alignment: Unified Implicit Neural Representation Research Code},
  year = {2026},
  url = {https://github.com/lachlanchen/SoftEventFrameAlignment}
}
```

## 状態

バージョン `0.1.0` は最小研究リリースです。合成チェックまたは利用許諾済みの小さな標本で再現できる不具合を歓迎します。

## ライセンス

Apache License 2.0。[LICENSE](../LICENSE) を参照してください。
