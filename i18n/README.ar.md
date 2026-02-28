[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

> Repository for the paper **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white&style=flat-square)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043?style=flat-square)

| Focus | Details |
| --- | --- |
| 🎯 Goal | Align event-camera streams and frame-camera streams with a unified implicit neural representation |
| ⚙️ Main Pipeline | `main.py` (train) + `evaluation.py` (analyze/visualize) |
| 🧪 Variant Scripts | `event_inn.py`, `frame_inn.py`, `event_deri*.py`, `event_inn_sparse.py` |

</div>

## Table of Contents

- [🔎 Overview](#-overview)
- [✨ Features](#-features)
- [🗂️ Project Structure](#-project-structure)
- [📋 Prerequisites](#-prerequisites)
- [⚙️ Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [🧪 Usage](#-usage)
- [🧩 Configuration Notes](#-configuration-notes)
- [🧠 Mathematical Formulation](#-mathematical-formulation)
- [🧾 Examples](#-examples)
- [🛠️ Development Notes](#-development-notes)
- [🧯 Troubleshooting](#-troubleshooting)
- [🗺️ Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [❤️ Support](#-support)
- [📄 License](#-license)

## 🔎 Overview

يتضمن هذا المستودع كودًا بحثيًا لمحاذاة تدفقات كاميرات الأحداث وتدفقات كاميرات الإطارات باستخدام تمثيل عصبي ضمني موحد (INR/INN).

### Core Pipeline (Canonical Path)

| Step | Component | Purpose |
|---|---|---|
| 1 | `softalign/data_processing.py` | قراءة تدفقات AEDAT4، وتطبيع البيانات، وإنشاء عينات نقاط |
| 2 | `softalign/implicit_model.py` | تعريف دالة ضمنية مشتركة + معاملات محاذاة قابلة للتعلّم |
| 3 | `softalign/training.py` + `main.py` | تحسين إعادة بناء الحدث/الإطار بشكل مشترك |
| 4 | `evaluation.py` | عرض محاذاة التوافق وتقارير الخسائر |

إضافةً إلى المسار الأساسي، يتضمن المستودع سكربتات مستقلة وتجريبية لنسخ INR المخصصة للأحداث فقط، والإطارات فقط، والنسخ المتناثرة، والنسخ المبنية على المشتقة.

## ✨ Features

- محاذاة بين الحدث والإطار باستخدام معاملات مشابهة لتحويل affine قابلة للتعلم: `scale` و `shift_x` و `shift_y` و `shift_t`.
- دالة ضمنية مشتركة `F(x, y, t)` تُستخدم في فرعي الحدث والإطار.
- نمذجة الأحداث عبر مشتقة زمنية بتفاضل زمني باستخدام `threshold` و `dt` قابلين للتعلم.
- دعم إدخال AEDAT4 عبر `dv_processing` بالإضافة إلى سير عمل ملفات `.npy` بعد المعالجة.
- مخرجات تصور مدمجة لمنحنيات الخسارة، وتطور المعلمات، وطبقات التراكب لمحاذاة الإطارات.
- دعم CUDA عند التوفر (`torch.cuda.is_available()` مع تراجع إلى CPU).
- سكربتات بيانات للاختبار السريع وفحص الأخطاء خفيفة:
  - `read_events.py`
  - `read_frames.py`
  - `reader.py`, `reader_norm.py`, `simple_count.py`

## 🗂️ Project Structure

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

## 📋 Prerequisites

- Python 3.10+ موصى به.
- أمثلة Linux/macOS في الأسفل (يمكن تكييفها يدويًا لمستخدم Windows).
- CUDA GPU مفضل إن وُجد لتسريع التدريب.
- يحتاج قراءة AEDAT4 إلى `dv_processing` واعتماديات نظام متوافقة.

## ⚙️ Installation

لا يوجد حالياً ملف `requirements.txt` أو `pyproject.toml`، لذا قم بتثبيت الاعتماديات يدويًا.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

Dependencies اختيارية لتجارب النسخة المتناثرة (`event_inn_sparse.py`):

```bash
pip install scikit-learn torch-geometric torch-scatter
```

الافتراض: توفر CUDA يحدد سلوك الجهاز الافتراضي، لكن جميع الأوامر أدناه تدعم أيضًا التحديد الصريح `--device cpu`.

## 🚀 Quick Start

### 1. Train the main alignment model

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

يتم كتابة المخرجات إلى:
- `data/` (`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`)
- `checkpoints/` (`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`)

### 2. Evaluate a trained checkpoint

يستورد `evaluation.py` النموذج `EventFrameAlignmentModel` عبر `from implicit_model import ...`، بينما الوحدة النشطة موجودة في `softalign/implicit_model.py`. طريقة تشغيل عملية عملية من جذر المستودع هي:

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

ينشئ هذا تقارير/مرئيات مثل `evaluation/alignment_visualization.png` و `evaluation/evaluation_results.txt`.

## 🧪 Usage

### Main Pipeline (`main.py`)

```bash
python main.py --help
```

الخيارات الرئيسية:
- `--filepath`: مسار إدخال AEDAT4.
- `--duration`: عدد الثواني المراد قراءتها من التسجيل.
- `--data_dir`: دليل إخراج البيانات المعالجة.
- `--checkpoint_dir`: دليل حفظ نقاط التوقف.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` أو `cpu`.
- `--reprocess`: فرض إعادة إنشاء البيانات.

### Experimental Scripts

Event-only INR:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

Frame-only INR:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# or
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# or
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

Derivative variants:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Sparse variant:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Standalone monolithic alignment:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

Additional readers:
```bash
python read_events.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python read_frames.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python simple_count.py --input data/events.npy
```

## 🧩 Configuration Notes

Main model defaults in `softalign/implicit_model.py`:

| Parameter | Default |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

يطبّق المعالج المسبق للبيانات في `softalign/data_processing.py` انحرافًا اصطناعيًا على الأحداث قبل التدريب حاليًا. هذا السلوك مقصود في الكود الحالي ويجب أخذه بعين الاعتبار عند تفسير المقاييس.

تتضمن سكربتات التدريب أيضًا عناصر قابلة للضبط عبر وسائط CLI (مثل `--hidden_dim`, `--num_layers`, و `--batch_size`) مفيدة لاختبارات الإزالة (ablation).

## 🧠 Mathematical Formulation

نموذج الشبكة الضمنية المشترك يصف `F(x, y, t)`.

Frame branch:
- يتنبأ مباشرة برد مشابه للشدة باستخدام `F(x, y, t)`.

Event branch:
1. تطبيق تحويل الحدث القابل للتعلم:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. تقريب المشتقة الزمنية:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. توليد استجابة الحدث:
   - `sigmoid(dF/dt - threshold)`

تتضمن دالة الهدف التدريبية:
- خسارة MSE للحدث
- خسارة MSE للإطار
- تنظيم المعاملات الخاصة بالمحاذاة

## 🧾 Examples

استخدم فقط المصفوفات التي تمت معالجتها مسبقًا:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

فرض تشغيل على CPU:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

Evaluate to custom folder:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ Development Notes

- المستودع يعتمد على السكربتات (لا يوجد بيانات تعريف حزم حتى الآن).
- توجد مخرجات مولّدة داخل المستودع (`checkpoints/results`)، لذلك قد يكون استخدام القرص كبيرًا.
- يحتوي `readme-file.md` و `project-structure.txt` على افتراضات بنية قديمة وليست متوافقة بالكامل مع هيكل الجذر الحالي.
- لا توجد اختبارات آلية أو سير عمل CI مكتشفة حاليًا.

## 🧯 Troubleshooting

- `ModuleNotFoundError: No module named 'implicit_model'` في `evaluation.py`:
  - شغّل باستخدام `PYTHONPATH=softalign` كما هو موضح أعلاه.
- مشاكل تثبيت/تشغيل `dv_processing`:
  - تأكد من توافق منصتك وإصدار بايثون مع حزم/مكتبات `dv-processing`.
- CUDA OOM:
  - خفّض `--batch_size`، أو قلل `--max_events` (عند توفره)، أو شغّل باستخدام `--device cpu`.
- ملف AEDAT4 مفقود:
  - تأكد من المسار تحت `yuqing/` أو وفّر تسجيلًا خاصًا عبر `--filepath` / `--aedat4_file`.
- اختلاف البيانات بين التشغيلات:
  - أعد التنفيذ باستخدام `--reprocess` عند تغيير افتراضات المعالجة المسبقة.

## 🗺️ Roadmap

- إضافة ملفات بيئة قابلة لإعادة الإنتاج (`requirements.txt` أو `pyproject.toml`).
- توحيد مسارات الاستيراد بين التقييم وتنفيذ الحزم/الوحدة.
- إضافة اختبارات آلية للتحقق من معالجة البيانات والتأكد من تنفيذ النموذج (forward) والتدريب.
- إضافة CI للتنسيق وفحوصات smoke.
- إضافة ملفات README متعددة اللغات ضمن `i18n/`.

## 🤝 Contributing

المساهمات مرحب بها.

Recommended workflow:
1. Create a focused issue or branch summarizing the experiment change.
2. Keep scripts aligned (especially imports, CLI conventions, and output directory names).
3. Preserve existing experiment artifacts behavior unless changing naming/versioning.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

هذا المستودع مرخّص ضمن Apache License 2.0. راجع ملف [`LICENSE`](LICENSE) للنص الكامل.

إذا استُخدم هذا المستودع في منشورات علمية، فيجب إضافة تفاصيل الاقتباس هنا وفي ملاحظات الإصدارات وفق الحاجة.
