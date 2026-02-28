[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

> Репозиторий для статьи **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white&style=flat-square)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043?style=flat-square)

| Фокус | Подробности |
| --- | --- |
| 🎯 Цель | Выравнивание потоков с камер событий и камер кадра с помощью единой неявной нейронной модели |
| ⚙️ Основной конвейер | `main.py` (обучение) + `evaluation.py` (анализ/визуализация) |
| 🧪 Скрипты вариантов | `event_inn.py`, `frame_inn.py`, `event_deri*.py`, `event_inn_sparse.py` |

</div>

## Оглавление

- [🔎 Обзор](#-обзор)
- [✨ Особенности](#-особенности)
- [🗂️ Структура проекта](#-структура-проекта)
- [📋 Предварительные требования](#-предварительные-требования)
- [⚙️ Установка](#-установка)
- [🚀 Быстрый старт](#-быстрый-старт)
- [🧪 Использование](#-использование)
- [🧩 Примечания по настройке](#-примечания-по-настройке)
- [🧠 Математическая формулировка](#-математическая-формулировка)
- [🧾 Примеры](#-примеры)
- [🛠️ Заметки для разработки](#-заметки-для-разработки)
- [🧯 Устранение неполадок](#-устранение-неполадок)
- [🗺️ План развития](#-план-развития)
- [🤝 Вклад](#-вклад)
- [❤️ Support](#-support)
- [📄 Лицензия](#-лицензия)

## 🔎 Обзор

Этот репозиторий содержит исследовательский код для выравнивания потоков камер событий и потоков камерных кадров с помощью единого неявного нейронного представления (INR/INN).

### Основной конвейер (канонический путь)

| Шаг | Компонент | Назначение |
|---|---|---|
| 1 | `softalign/data_processing.py` | Чтение потоков AEDAT4, нормализация данных, создание выборки точек |
| 2 | `softalign/implicit_model.py` | Определение общей неявной функции и обучаемых параметров выравнивания |
| 3 | `softalign/training.py` + `main.py` | Совместная оптимизация реконструкции событий и кадров |
| 4 | `evaluation.py` | Визуализация выравнивания и вывод потерь |

Помимо основного пути репозиторий включает отдельные и экспериментальные скрипты для вариантов INR, работающих только с событиями, только с кадрами, с разреженным представлением и на основе производной.

## ✨ Особенности

- Выравнивание события и кадра с обучаемыми параметрами в стиле аффинного преобразования: `scale`, `shift_x`, `shift_y`, `shift_t`.
- Общая неявная функция `F(x, y, t)`, используемая обоими ответвлениями: событием и кадром.
- Моделирование события через конечно-разностную временную производную с обучаемыми `threshold` и `dt`.
- Загрузка AEDAT4 (`dv_processing`) и конвейеры обработки `.npy`.
- Встроенная визуализация для кривых ошибки, эволюции параметров и наложений выравнивания.
- Поддержка CUDA при наличии (`torch.cuda.is_available()`, иначе CPU).
- Скрипты для быстрого просмотра данных и легкой отладки:
  - `read_events.py`
  - `read_frames.py`
  - `reader.py`, `reader_norm.py`, `simple_count.py`

## 🗂️ Структура проекта

```text
SoftEventFrameAlignment/
├── README.md
├── LICENSE
├── main.py                              # Основная точка входа в обучение event-frame
├── evaluation.py                        # Основная точка входа для оценки/визуализации
├── softalign/
│   ├── __init__.py
│   ├── data_processing.py               # Загрузка AEDAT4 + нормализация + семплинг точек кадров
│   ├── implicit_model.py                # MLP + параметры выравнивания
│   └── training.py                      # Обёртка датасета + цикл оптимизации
├── event_inn.py                         # Эксперимент INR только для событий
├── frame_inn.py                         # Эксперимент INR только для кадров
├── event_deri.py                        # Производный/логарифмический вариант событий
├── event_deri_2.py                      # Вариант производной с дополнительной нулевой регуляризацией
├── event_inn_sparse.py                  # Разреженный INR для событий (зависимости PyG/torch_scatter)
├── softalign_standalone.py              # Монолитный однофайловый рабочий процесс выравнивания
├── softalign_old.py                     # Устаревшая реализация, оставленная для сравнения
├── read_events.py                       # Работа с событиями и базовые проверки предобработки
├── read_frames.py                       # Извлечение кадров/видео и проверка форм
├── reader.py                            # Утилита чтения AEDAT
├── reader_norm.py                       # Чтение с утилитами нормализации
├── simple_count.py                      # Лёгкая утилита подсчёта событий
├── data/                                # Обработанные массивы (сгенерированные или добавленные в репозиторий)
├── checkpoints/                         # Основные контрольные точки и графики
├── event_inn_results/                   # Сгенерированные результаты экспериментов
├── frame_inn_results/                   # Сгенерированные результаты экспериментов
├── event_derivative_results/             # Сгенерированные результаты экспериментов
├── alignment_results_*/                 # Запуски выравнивания с меткой времени
├── yuqing/                              # Примеры AEDAT4 файлов
├── events_processed.csv                  # Артефакт сводки событий
├── frames_processed.csv                  # Артефакт сводки кадров
└── i18n/                                # Переведённые файлы README
```

## 📋 Предварительные требования

- Рекомендуется Python 3.10+.
- Ниже примеры для оболочки Linux/macOS (при необходимости адаптируйте под Windows).
- Опционально, но желательно: GPU с поддержкой CUDA для ускорения обучения.
- Для чтения AEDAT4 требуется `dv_processing` и совместимые системные зависимости.

## ⚙️ Установка

На данный момент отсутствуют `requirements.txt` или `pyproject.toml`, поэтому зависимости нужно устанавливать вручную.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

Дополнительные зависимости для разреженных экспериментов (`event_inn_sparse.py`):

```bash
pip install scikit-learn torch-geometric torch-scatter
```

Предположение: доступность CUDA определяет поведение устройства по умолчанию, но все команды ниже также поддерживают явный `--device cpu`.

## 🚀 Быстрый старт

### 1. Обучение основной модели выравнивания

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

Результаты записываются в:
- `data/` (`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`)
- `checkpoints/` (`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`)

### 2. Оценка обученного контрольного состояния

`evaluation.py` импортирует `EventFrameAlignmentModel` через `from implicit_model import ...`, тогда как активный модуль находится в `softalign/implicit_model.py`. Практический запуск из корня репозитория:

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

Это создаёт визуализации/метрики, такие как `evaluation/alignment_visualization.png` и `evaluation/evaluation_results.txt`.

## 🧪 Использование

### Основной конвейер (`main.py`)

```bash
python main.py --help
```

Ключевые параметры:
- `--filepath`: путь к входному файлу AEDAT4.
- `--duration`: продолжительность в секундах для чтения записи.
- `--data_dir`: каталог для сохранения обработанных данных.
- `--checkpoint_dir`: каталог для сохранения контрольных точек.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` или `cpu`.
- `--reprocess`: принудительная регенерация данных.

### Экспериментальные скрипты

INN только для событий:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# или
python event_inn.py --events_file data/events.npy --output_dir output_event
```

INN только для кадров:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# или
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# или
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

Варианты на основе производной:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Разреженный вариант:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Монолитное выравнивание:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

Дополнительные считыватели:
```bash
python read_events.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python read_frames.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python simple_count.py --input data/events.npy
```

## 🧩 Примечания по настройке

Основные параметры модели в `softalign/implicit_model.py`:

| Параметр | Значение по умолчанию |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

На этапе предобработки данных в `softalign/data_processing.py` сейчас к событиям применяется синтетическое рассогласование перед обучением. Это намеренно реализовано в текущей версии кода и должно учитываться при интерпретации метрик.

Скрипты обучения также содержат настраиваемые параметры через флаги CLI (например `--hidden_dim`, `--num_layers` и `--batch_size`), полезные для абляционных экспериментов.

## 🧠 Математическая формулировка

Общая неявная сеть моделирует `F(x, y, t)`.

Ветвь кадра:
- Прямое предсказание отклика, похожего на интенсивность, с помощью `F(x, y, t)`.

Ветвь события:
1. Применить обучаемое преобразование события:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. Аппроксимировать временную производную:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. Сформировать отклик события:
   - `sigmoid(dF/dt - threshold)`

Целевая функция обучения объединяет:
- Ошибку MSE для событий
- Ошибку MSE для кадров
- Регуляризацию параметров выравнивания

## 🧾 Примеры

Используйте только предобработанные массивы:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

Форсировать запуск на CPU:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

Оценка в пользовательскую папку:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ Заметки для разработки

- Репозиторий ориентирован на скрипты (метаданные упаковки пока отсутствуют).
- Сгенерированные артефакты хранятся внутри репозитория (checkpoints/results), поэтому использование диска может быть значительным.
- Файлы `readme-file.md` и `project-structure.txt` содержат устаревшие предположения о структуре и не полностью соответствуют текущему корневому макету.
- На данный момент автоматизированные тесты и CI-конвейеры не обнаружены.

## 🧯 Устранение неполадок

- `ModuleNotFoundError: No module named 'implicit_model'` в `evaluation.py`:
  - Запустите с `PYTHONPATH=softalign`, как показано выше.
- Проблемы с установкой/выполнением `dv_processing`:
  - Проверьте, поддерживаются ли ваша платформа и версия Python библиотеками `dv-processing` и зависимостями.
- Недостаток памяти CUDA:
  - Уменьшите `--batch_size`, снизьте `--max_events` (если доступен) или запустите с `--device cpu`.
- Отсутствует файл AEDAT4:
  - Проверьте путь в `yuqing/` или передайте собственную запись через `--filepath` / `--aedat4_file`.
- Несоответствие данных между запусками:
  - Повторно запустите с `--reprocess`, если менялись предположения предобработки.

## 🗺️ План развития

- Добавить воспроизводимые файлы окружения (`requirements.txt` или `pyproject.toml`).
- Унифицировать пути импорта для оценки и пакетного/модульного выполнения.
- Добавить автоматические тесты для проверки обработки данных, прямого прохода модели и обучения.
- Добавить CI для линтинга и smoke-тестов.
- Добавить многоязычные README в `i18n/`.

## 🤝 Вклад

Вклад приветствуется.

Рекомендуемый рабочий процесс:
1. Создайте фокусированный issue или ветку с описанием изменений эксперимента.
2. Поддерживайте согласованность скриптов (особенно импортов, соглашений CLI и имён директорий вывода).
3. Сохраняйте поведение существующих артефактов экспериментов, если не меняете схему именования/версионирования.

## 📄 Лицензия

Этот репозиторий распространяется под лицензией Apache 2.0. Полный текст доступен в [`LICENSE`](../LICENSE).

Предположение: если этот репозиторий используется в публикациях, детали цитирования следует добавить здесь и в примечаниях к релизу по мере необходимости.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
