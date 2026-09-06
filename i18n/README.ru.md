[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

*Минимальный исследовательский код для обучения пространственно-временного выравнивания данных событийной и кадровой камер с единым неявным нейронным представлением.*

[![CI](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA043)](../LICENSE)
[![Sponsor](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)

Это первая публичная и полностью клонируемая реализация **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**. В неё входят основная модель, предобработка AEDAT4, обучение, оценка, описание зависимостей, CI и детерминированная синтетическая проверка на CPU.

Записи, кадры с узнаваемыми людьми, производные массивы и обученные чекпойнты не публикуются. Используйте только данные, на обработку которых у вас есть права.

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=kofi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## Архитектура

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

Ветвь событий преобразует координаты и приближает производную по времени общего MLP. Кадровая ветвь напрямую вычисляет то же поле. Масштаб и шаг производной имеют положительную параметризацию, поэтому деление на ноль исключено.

## Состав публичной версии

| Путь | Назначение |
| --- | --- |
| `softalign/implicit_model.py` | Общий неявный MLP и обучаемое выравнивание |
| `softalign/training.py` | Проверяемый набор данных и цикл совместного обучения |
| `softalign/data_processing.py` | Опциональное чтение AEDAT4 и предобработка в секундах |
| `softalign/synthetic.py` | Детерминированные синтетические данные проекта |
| `main.py` / `evaluation.py` | CLI обучения и диагностики |
| `examples/` / `tests/` | Сквозная проверка и тест ядра |

Исторические эксперименты, исходные записи, локальные данные и старые чекпойнты намеренно исключены.

## Установка и проверка

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,viz]'
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py --output-dir .smoke-output --epochs 8
```

Проверка создаёт чекпойнт, графики и `evaluation_results.json`. Она подтверждает выполнение конвейера и конечность значений, но не точность выравнивания.

## Запуск на собственной записи AEDAT4

```bash
python -m pip install -e '.[aedat,viz]'
python main.py --filepath /path/to/your-recording.aedat4 --reprocess --data_dir data --checkpoint_dir checkpoints
python evaluation.py --model_path checkpoints/model_final.pt --data_dir data --output_dir evaluation --device cpu
```

Предобработка записывает схему `softalign-processed/v1` и переводит метки AEDAT из микросекунд в секунды. Синтетическое смещение по умолчанию выключено и включается только флагом `--synthetic-misalignment`. Старые массивы без совместимых метаданных единиц отклоняются.

Большие коллекции остаются в памяти CPU; на выбранное устройство переходят только выбранные батчи. Начните с `--device cpu`, а CUDA включайте после проверки памяти.

## Контракт данных

| Файл | Значение |
| --- | --- |
| `events.npy` | `N × 4`: нормированные x/y, секунды, целевое значение события |
| `frame_points.npy` | `M × 4`: нормированные x/y, секунды, интенсивность |
| `preprocessing.json` | Схема, единицы, seed, флаг преобразования и размеры |
| `model_final.pt` | Состояние, параметры и архитектура модели |
| `evaluation_results.json` | MSE реконструкции на ограниченном числе входных образцов |

Результаты — диагностика предоставленных массивов, а не независимые метрики, benchmark или доказательство физической калибровки.

## Проверка и ограничения исследования

- CI запускает детерминированный CPU-тест и короткий цикл обучения и оценки.
- Проверяются форма, пустые входы, конечность, единицы времени и доступность устройства.
- Текущая функция потерь — исследовательская базовая линия; реальному исследованию нужны ground truth, отдельная оценка, абляции и анализ неопределённости.
- Не прикладывайте приватные записи к публичным GitHub Issues.

## Цитирование

При использовании в исследовании процитируйте репозиторий. GitHub читает [CITATION.cff](../CITATION.cff) и показывает **Cite this repository**.

```bibtex
@software{chen_soft_event_frame_alignment_2026,
  author = {Chen, Lachlan},
  title = {Soft Event-Frame Alignment: Unified Implicit Neural Representation Research Code},
  year = {2026},
  url = {https://github.com/lachlanchen/SoftEventFrameAlignment}
}
```

## Статус

Версия `0.1.0` — минимальный исследовательский выпуск. Приветствуются воспроизводимые ошибки на синтетической проверке или небольшом разрешённом образце.

## Лицензия

Apache License 2.0. См. [LICENSE](../LICENSE).
