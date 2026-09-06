[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# المحاذاة المرنة بين الأحداث والإطارات

*شيفرة بحثية مصغّرة لتعلّم المحاذاة المكانية والزمنية بين عينات كاميرا الأحداث وكاميرا الإطارات عبر تمثيل عصبي ضمني واحد.*

[![CI](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA043)](../LICENSE)
[![Sponsor](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)

هذه أول نسخة عامة قابلة للاستنساخ من تنفيذ **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**. تتضمن النموذج الأساسي، ومعالجة AEDAT4، والتدريب، والتقييم، وبيانات الاعتماد، واختبار CPU اصطناعياً وحتمياً.

لا تتضمن النسخة تسجيلات أو صوراً قابلة للتعرّف أو مصفوفات مشتقة أو نقاط تحقق مدرّبة. استخدم فقط بيانات تملك حق استخدامها.

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=kofi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## البنية

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

يحوّل فرع الأحداث الإحداثيات ويقرّب المشتقة الزمنية لشبكة MLP مشتركة، بينما يقيّم فرع الإطارات الحقل نفسه مباشرة. تستخدم معاملات المقياس وخطوة الاشتقاق تمثيلاً موجباً يمنع القسمة على الصفر.

## محتويات النسخة العامة

| المسار | الغرض |
| --- | --- |
| `softalign/implicit_model.py` | الشبكة الضمنية ومعاملات المحاذاة القابلة للتعلّم |
| `softalign/training.py` | غلاف بيانات متحقق منه وحلقة تدريب مشتركة |
| `softalign/data_processing.py` | إدخال AEDAT4 اختياري ومعالجة بوحدة الثواني |
| `softalign/synthetic.py` | بيانات اصطناعية حتمية ينشئها المشروع |
| `main.py` / `evaluation.py` | التدريب والتشخيص من سطر الأوامر |
| `examples/` / `tests/` | فحص تنفيذي واختبار أساسي |

التجارب القديمة والتسجيلات الخام والبيانات المحلية ونقاط التحقق ليست جزءاً من هذه النسخة.

## التثبيت والتحقق

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,viz]'
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py --output-dir .smoke-output --epochs 8
```

ينشئ الفحص نقطة تحقق ورسومات وملف `evaluation_results.json`. وهو يثبت أن المسار يعمل بقيم منتهية، لا أنه يثبت دقة المحاذاة.

## التشغيل على تسجيل AEDAT4 خاص بك

```bash
python -m pip install -e '.[aedat,viz]'
python main.py --filepath /path/to/your-recording.aedat4 --reprocess --data_dir data --checkpoint_dir checkpoints
python evaluation.py --model_path checkpoints/model_final.pt --data_dir data --output_dir evaluation --device cpu
```

تسجّل المعالجة مخطط `softalign-processed/v1` وتحول الطوابع من ميكروثانية إلى ثانية. التشويه الاصطناعي متوقف افتراضياً ولا يُفعّل إلا عبر `--synthetic-misalignment`. تُرفض المصفوفات القديمة بلا بيانات وحدات متوافقة.

تبقى المجموعات الكبيرة في ذاكرة CPU، ولا تنتقل إلى الجهاز المختار إلا الدفعات المسحوبة. ابدأ بـ `--device cpu` ثم استخدم CUDA بعد التحقق من الذاكرة.

## عقد البيانات

| الملف | المعنى |
| --- | --- |
| `events.npy` | مصفوفة `N × 4`: ‏x وy مطبّعان، الزمن بالثواني، وهدف الحدث |
| `frame_points.npy` | مصفوفة `M × 4`: ‏x وy مطبّعان، الزمن بالثواني، والشدة |
| `preprocessing.json` | المخطط والوحدات والبذرة والعلم والأعداد |
| `model_final.pt` | حالة النموذج والمعاملات والبنية |
| `evaluation_results.json` | أخطاء إعادة البناء على عينات محدودة من المدخلات |

نتائج التقييم تشخيصات لإعادة بناء المصفوفات المقدمة، وليست مقاييس اختبار مستقل أو معياراً أو دليلاً على معايرة زوج حساسات فعلي.

## التحقق وحدود البحث

- يشغّل CI اختبار CPU حتمياً ودورة تدريب/تقييم قصيرة.
- تُفحص الأشكال والفراغ والقيم المنتهية ووحدات الزمن وتوفر الجهاز.
- الخسارة الحالية خط أساس بحثي؛ التجربة الفعلية تحتاج حقيقة معايرة وتقسيماً مستقلاً وتحليلات عدم يقين.
- التسجيلات الخاصة لا ينبغي إرفاقها بقضايا GitHub العامة.

## الاستشهاد

عند استخدام المشروع بحثياً، استشهد بالمستودع. يقرأ GitHub ملف [CITATION.cff](../CITATION.cff) ويعرض لوحة **Cite this repository**.

```bibtex
@software{chen_soft_event_frame_alignment_2026,
  author = {Chen, Lachlan},
  title = {Soft Event-Frame Alignment: Unified Implicit Neural Representation Research Code},
  year = {2026},
  url = {https://github.com/lachlanchen/SoftEventFrameAlignment}
}
```

## الحالة

الإصدار `0.1.0` نسخة بحثية مصغرة. نرحب بالمشكلات القابلة لإعادة الإنتاج باستخدام الفحص الاصطناعي أو عينة صغيرة مصرّح بها.

## الترخيص

مرخص وفق Apache License 2.0. راجع [LICENSE](../LICENSE).
