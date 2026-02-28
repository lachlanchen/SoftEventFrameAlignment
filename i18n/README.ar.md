[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Soft Event-Frame Alignment

> مستودع الورقة البحثية **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043)

## 🔎 نظرة عامة

يحتوي هذا المستودع على كود بحثي لمواءمة تدفقات كاميرا الأحداث وتدفقات كاميرا الإطارات باستخدام تمثيل عصبي ضمني موحد (INR/INN).

### خط الأنابيب الأساسي (المسار القياسي)

| الخطوة | المكوّن | الغرض |
|---|---|---|
| 1 | `softalign/data_processing.py` | قراءة تدفقات AEDAT4 وتطبيع البيانات وإنشاء عينات نقاط |
| 2 | `softalign/implicit_model.py` | تعريف دالة ضمنية مشتركة + معاملات مواءمة قابلة للتعلّم |
| 3 | `softalign/training.py` + `main.py` | تحسين إعادة بناء الأحداث/الإطارات بشكل مشترك |
| 4 | `evaluation.py` | تصور المواءمة والإبلاغ عن الخسائر |

بالإضافة إلى المسار الرئيسي، يتضمن المستودع سكربتات مستقلة وتجريبية لمتغيرات INR الخاصة بالأحداث فقط، أو الإطارات فقط، أو المتناثرة، أو المعتمدة على المشتقة.

## ✨ الميزات

- مواءمة حدث-إطار بمعاملات على نمط affine قابلة للتعلّم: `scale`, `shift_x`, `shift_y`, `shift_t`.
- دالة ضمنية مشتركة `F(x, y, t)` تُستخدم من فرعي الأحداث والإطارات.
- نمذجة الأحداث عبر مشتقة زمنية بفروق منتهية مع `threshold` و `dt` قابلين للتعلّم.
- استيعاب AEDAT4 (`dv_processing`) بالإضافة إلى سير عمل ملفات `.npy` المعالجة.
- مخرجات تصور مدمجة لمنحنيات الخسارة وتطور المعاملات وتراكبات المواءمة.
- دعم CUDA عند التوفر (`torch.cuda.is_available()`) مع الرجوع إلى CPU.

## 🗂️ بنية المشروع

```text
SoftEventFrameAlignment/
├── README.md
├── LICENSE
├── main.py                          # نقطة الدخول القياسية لتدريب مواءمة الحدث-الإطار
├── evaluation.py                    # نقطة الدخول القياسية للتقييم/التصور
├── softalign/
│   ├── __init__.py
│   ├── data_processing.py           # تحميل AEDAT4 + التطبيع + أخذ عينات نقاط الإطار
│   ├── implicit_model.py            # MLP + معاملات المواءمة
│   └── training.py                  # مُغلّف مجموعة البيانات + حلقة التحسين
├── event_inn.py                     # تجربة INR للأحداث فقط
├── frame_inn.py                     # تجربة INR للإطارات فقط
├── event_deri.py                    # متغير الأحداث بالمشتقة/لوغاريتم المشتقة
├── event_deri_2.py                  # متغير مشتقة مع تنظيم صفري إضافي
├── event_inn_sparse.py              # INR أحداث متناثر (يعتمد على PyG/torch_scatter)
├── softalign_standalone.py          # سير عمل مواءمة أحادي متكامل
├── data/                            # مصفوفات معالجة (مُولّدة أو محفوظة في المستودع)
├── checkpoints/                     # نقاط حفظ التدريب الرئيسية والمخططات
├── event_inn_results/               # مخرجات التجارب المُولّدة
├── frame_inn_results/               # مخرجات التجارب المُولّدة
├── event_drivative_results/         # مخرجات التجارب المُولّدة
├── alignment_results_*/             # تشغيلات مواءمة مؤرخة زمنيًا
├── yuqing/                          # ملفات AEDAT4 نموذجية
└── i18n/                            # موقع ملفات الترجمة المستهدف
```

## 📋 المتطلبات المسبقة

- يوصى باستخدام Python 3.10+.
- أمثلة الشل أدناه خاصة بـ Linux/macOS (يمكن التكييف لـ Windows عند الحاجة).
- اختياري لكن موصى به: GPU يدعم CUDA لتسريع التدريب.
- قراءة AEDAT4 تتطلب `dv_processing` واعتماديات نظام متوافقة.

## ⚙️ التثبيت

لا يوجد حاليًا `requirements.txt` أو `pyproject.toml`، لذا ثبّت الاعتماديات يدويًا.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

اعتماديات اختيارية للتجارب المتناثرة (`event_inn_sparse.py`):

```bash
pip install scikit-learn torch-geometric torch-scatter
```

## 🚀 البدء السريع

### 1. درّب نموذج المواءمة الرئيسي

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

تُكتب المخرجات إلى:
- `data/` (`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`)
- `checkpoints/` (`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`)

### 2. قيّم نقطة حفظ مدرّبة

يقوم `evaluation.py` باستيراد `EventFrameAlignmentModel` عبر `from implicit_model import ...`، بينما الوحدة النشطة موجودة في `softalign/implicit_model.py`. استدعاء عملي من جذر المستودع:

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

ينتج ذلك تصورات/مقاييس مثل `evaluation/alignment_visualization.png` و `evaluation/evaluation_results.txt`.

## 🧪 الاستخدام

### خط الأنابيب الرئيسي (`main.py`)

```bash
python main.py --help
```

الخيارات الأساسية:
- `--filepath`: مسار إدخال AEDAT4.
- `--duration`: عدد الثواني المقروءة من التسجيل.
- `--data_dir`: مجلد إخراج البيانات المعالجة.
- `--checkpoint_dir`: مجلد إخراج نقاط الحفظ.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` أو `cpu`.
- `--reprocess`: فرض إعادة توليد البيانات.

### السكربتات التجريبية

INR للأحداث فقط:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

INR للإطارات فقط:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# or
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# or
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

متغيرات المشتقة:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

المتغير المتناثر:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

مواءمة أحادية متكاملة مستقلة:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

## 🧩 ملاحظات الإعداد

القيم الافتراضية للنموذج الرئيسي في `softalign/implicit_model.py`:

| المعامل | القيمة الافتراضية |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

تطبيق المعالجة المسبقة للبيانات في `softalign/data_processing.py` يطبق حاليًا عدم مواءمة اصطناعيًا على الأحداث قبل التدريب. هذا مقصود في الكود الحالي ويجب أخذه في الاعتبار عند تفسير المقاييس.

## 🧠 الصياغة الرياضية

الشبكة الضمنية المشتركة تمثل `F(x, y, t)`.

فرع الإطارات:
- يتنبأ مباشرة باستجابة شبيهة بالشدة باستخدام `F(x, y, t)`.

فرع الأحداث:
1. تطبيق تحويل أحداث قابل للتعلّم:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. تقريب المشتقة الزمنية:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. إنتاج استجابة الحدث:
   - `sigmoid(dF/dt - threshold)`

هدف التدريب يجمع بين:
- خسارة MSE للأحداث
- خسارة MSE للإطارات
- تنظيم على معاملات المواءمة

## 🧾 أمثلة

استخدام المصفوفات المعالجة مسبقًا فقط:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

فرض التشغيل على CPU:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

التقييم في مجلد مخصص:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ ملاحظات التطوير

- المستودع يعتمد على السكربتات (لا توجد بيانات تعريف حزم حتى الآن).
- توجد مصنوعات مُولّدة داخل المستودع (checkpoints/results)، لذلك قد يكون استخدام القرص كبيرًا.
- يحتوي `readme-file.md` و `project-structure.txt` على افتراضات بنية قديمة وغير متطابقة بالكامل مع تخطيط الجذر الحالي.
- لا توجد حاليًا اختبارات آلية أو تدفقات CI مكتشفة.

## 🧯 استكشاف الأخطاء وإصلاحها

- `ModuleNotFoundError: No module named 'implicit_model'` في `evaluation.py`:
  - شغّل باستخدام `PYTHONPATH=softalign` كما هو موضح أعلاه.
- مشاكل تثبيت/تشغيل `dv_processing`:
  - تأكد أن منصتك وإصدار Python مدعومان من حزم/مكتبات `dv-processing`.
- نفاد ذاكرة CUDA (OOM):
  - خفّض `--batch_size`، وقلّل `--max_events` (حيثما كان متاحًا)، أو شغّل باستخدام `--device cpu`.
- ملف AEDAT4 مفقود:
  - تحقّق من المسار ضمن `yuqing/` أو استخدم تسجيلك الخاص عبر `--filepath` / `--aedat4_file`.

## 🗺️ خارطة الطريق

- إضافة ملفات بيئة قابلة لإعادة الإنتاج (`requirements.txt` أو `pyproject.toml`).
- توحيد مسارات الاستيراد للتقييم وتشغيل الحزمة/الوحدة.
- إضافة اختبارات آلية للمعالجة المسبقة للبيانات وفحوصات forward/training للنموذج.
- إضافة CI للتدقيق واختبارات smoke.
- إضافة ملفات README متعددة اللغات ضمن `i18n/`.

## 🤝 المساهمة

المساهمات مرحب بها.

1. انسخ المستودع (Fork).
2. أنشئ فرع ميزات.
3. نفّذ تغييرات مركزة وموثقة جيدًا.
4. أرسل طلب سحب مع تفاصيل إعادة الإنتاج ونماذج مخرجات عند الحاجة.

إذا كنت تخطط لتغييرات كبيرة (مسار تدريب جديد، إعادة هيكلة، أو تغييرات اعتماديات)، افتح issue أولًا لمواءمة التوجه.

## 📚 الاقتباس

إذا استخدمت هذا المستودع في بحث، فاستشهد بالورقة المرتبطة و/أو المستودع.

بيانات تعريف المستودع الحالية المتاحة من نص README الموجود:
- عنوان الورقة: *Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation*

عنصر BibTeX إرشادي للمستودع (حدّث المؤلفين/الرابط عند الحاجة):

```bibtex
@misc{soft-event-frame-alignment,
  title        = {Soft Event-Frame Alignment},
  note         = {Code repository for "Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation"},
  year         = {2025},
  howpublished = {GitHub repository}
}
```

## 🙏 الشكر والتقدير

- منظومة أدوات كاميرات الأحداث، بما في ذلك `dv-processing`.
- PyTorch ومكتبات Python العلمية المستخدمة عبر التجارب.

## 💡 الدعم

قسم الدعم/الرعاية غير موجود في بيانات تعريف المستودع الحالية. إذا رغبت في إضافته، أضف الروابط في تحديث لاحق وسيتم الحفاظ عليها في المراجعات المستقبلية.

## 📄 الترخيص

مرخّص بموجب Apache License 2.0. راجع [LICENSE](../LICENSE).
