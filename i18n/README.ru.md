[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Soft Event-Frame Alignment

> Репозиторий к статье **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043)

## 🔎 Обзор

Этот репозиторий содержит исследовательский код для выравнивания потоков event-камеры и frame-камеры с использованием единого неявного нейронного представления (INR/INN).

### Основной пайплайн (канонический путь)

| Шаг | Компонент | Назначение |
|---|---|---|
| 1 | `softalign/data_processing.py` | Чтение AEDAT4-потоков, нормализация данных, создание точечных сэмплов |
| 2 | `softalign/implicit_model.py` | Определение общей неявной функции + обучаемых параметров выравнивания |
| 3 | `softalign/training.py` + `main.py` | Совместная оптимизация реконструкции event/frame |
| 4 | `evaluation.py` | Визуализация выравнивания и отчёт по лоссам |

Помимо основного пути, репозиторий включает отдельные и экспериментальные скрипты для вариантов INR: только events, только frames, sparse и derivative-based.

## ✨ Возможности

- Выравнивание event-frame с обучаемыми параметрами аффинного типа: `scale`, `shift_x`, `shift_y`, `shift_t`.
- Общая неявная функция `F(x, y, t)`, используемая и event-, и frame-веткой.
- Моделирование events через конечную разность по времени с обучаемыми `threshold` и `dt`.
- Загрузка AEDAT4 (`dv_processing`) и workflows с обработанными `.npy`.
- Встроенная визуализация кривых лосса, эволюции параметров и оверлеев выравнивания.
- Поддержка CUDA при наличии (`torch.cuda.is_available()` с fallback на CPU).

## 🗂️ Структура проекта

```text
SoftEventFrameAlignment/
├── README.md
├── LICENSE
├── main.py                          # Каноническая точка входа для обучения event-frame
├── evaluation.py                    # Каноническая точка входа для оценки/визуализации
├── softalign/
│   ├── __init__.py
│   ├── data_processing.py           # Загрузка AEDAT4 + нормализация + сэмплирование точек кадров
│   ├── implicit_model.py            # MLP + параметры выравнивания
│   └── training.py                  # Обёртка датасета + цикл оптимизации
├── event_inn.py                     # Эксперимент INR только для events
├── frame_inn.py                     # Эксперимент INR только для frames
├── event_deri.py                    # Вариант events на derivative/log-derivative
├── event_deri_2.py                  # Вариант derivative с дополнительной нулевой регуляризацией
├── event_inn_sparse.py              # Sparse event INR (зависимости PyG/torch_scatter)
├── softalign_standalone.py          # Монолитный all-in-one workflow выравнивания
├── data/                            # Обработанные массивы (сгенерированные или закоммиченные)
├── checkpoints/                     # Основные чекпоинты обучения и графики
├── event_inn_results/               # Сгенерированные результаты экспериментов
├── frame_inn_results/               # Сгенерированные результаты экспериментов
├── event_drivative_results/         # Сгенерированные результаты экспериментов
├── alignment_results_*/             # Запуски выравнивания с timestamp
├── yuqing/                          # Примерные файлы AEDAT4
└── i18n/                            # Папка для файлов перевода
```

## 📋 Предварительные требования

- Рекомендуется Python 3.10+.
- Ниже приведены примеры для Linux/macOS shell (при необходимости адаптируйте для Windows).
- Опционально, но рекомендуется: GPU с CUDA для ускорения обучения.
- Для чтения AEDAT4 требуется `dv_processing` и совместимые системные зависимости.

## ⚙️ Установка

Сейчас в репозитории нет `requirements.txt` или `pyproject.toml`, поэтому зависимости устанавливаются вручную.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

Опциональные зависимости для sparse-экспериментов (`event_inn_sparse.py`):

```bash
pip install scikit-learn torch-geometric torch-scatter
```

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

### 2. Оценка обученного чекпоинта

`evaluation.py` импортирует `EventFrameAlignmentModel` через `from implicit_model import ...`, тогда как активный модуль находится в `softalign/implicit_model.py`. Практичный запуск из корня репозитория:

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

Это генерирует визуализации/метрики, например `evaluation/alignment_visualization.png` и `evaluation/evaluation_results.txt`.

## 🧪 Использование

### Основной пайплайн (`main.py`)

```bash
python main.py --help
```

Ключевые опции:
- `--filepath`: путь к входному AEDAT4.
- `--duration`: сколько секунд читать из записи.
- `--data_dir`: директория вывода обработанных данных.
- `--checkpoint_dir`: директория вывода чекпоинтов.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` или `cpu`.
- `--reprocess`: принудительная перегенерация данных.

### Экспериментальные скрипты

INR только для events:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# или
python event_inn.py --events_file data/events.npy --output_dir output_event
```

INR только для frames:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# или
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# или
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

Варианты на производной:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Sparse-вариант:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Монолитное standalone-выравнивание:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

## 🧩 Примечания по конфигурации

Значения по умолчанию в основной модели (`softalign/implicit_model.py`):

| Параметр | Значение по умолчанию |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

Предобработка данных в `softalign/data_processing.py` сейчас применяет к events синтетическое смещение перед обучением. Это намеренное поведение текущего кода, и его нужно учитывать при интерпретации метрик.

## 🧠 Математическая формулировка

Общая неявная сеть моделирует `F(x, y, t)`.

Frame-ветка:
- Непосредственно предсказывает response, похожий на интенсивность, с помощью `F(x, y, t)`.

Event-ветка:
1. Применение обучаемого преобразования events:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. Приближение производной по времени:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. Формирование event response:
   - `sigmoid(dF/dt - threshold)`

Функция обучения сочетает:
- Event MSE loss
- Frame MSE loss
- Регуляризацию параметров выравнивания

## 🧾 Примеры

Использование только предобработанных массивов:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

Принудительный запуск на CPU:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

Оценка в пользовательскую папку:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ Примечания для разработки

- Репозиторий ориентирован на скрипты (метаданных packaging пока нет).
- В репозитории присутствуют сгенерированные артефакты (checkpoints/results), поэтому использование диска может быть большим.
- `readme-file.md` и `project-structure.txt` содержат устаревшие предположения о структуре и не полностью соответствуют текущему layout в корне.
- На данный момент не обнаружено автоматизированных тестов или CI workflows.

## 🧯 Устранение неполадок

- `ModuleNotFoundError: No module named 'implicit_model'` в `evaluation.py`:
  - Запускайте с `PYTHONPATH=softalign`, как показано выше.
- Проблемы установки/выполнения `dv_processing`:
  - Убедитесь, что ваша платформа и версия Python поддерживаются wheels/libs для `dv-processing`.
- CUDA OOM:
  - Уменьшите `--batch_size`, снизьте `--max_events` (где доступно) или запускайте с `--device cpu`.
- Отсутствует файл AEDAT4:
  - Проверьте путь в `yuqing/` или передайте собственную запись через `--filepath` / `--aedat4_file`.

## 🗺️ Дорожная карта

- Добавить воспроизводимые файлы окружения (`requirements.txt` или `pyproject.toml`).
- Унифицировать import paths для evaluation и запуска в режиме package/module.
- Добавить автоматизированные тесты для предобработки данных и проверок model forward/training.
- Добавить CI для линтинга и smoke-тестов.
- Добавить мультиязычные README-файлы в `i18n/`.

## 🤝 Вклад

Вклад приветствуется.

1. Сделайте fork репозитория.
2. Создайте feature-ветку.
3. Вносите сфокусированные и хорошо задокументированные изменения.
4. Отправьте pull request с деталями воспроизведения и примерами выходных данных, если это уместно.

Если вы планируете крупные изменения (новый путь обучения, рефакторинг или изменения зависимостей), сначала откройте issue, чтобы согласовать направление.

## 📚 Цитирование

Если вы используете этот репозиторий в исследовании, пожалуйста, цитируйте связанную статью и/или репозиторий.

Текущие метаданные репозитория из существующего текста README:
- Название статьи: *Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation*

Заглушка BibTeX для репозитория (при необходимости обновите авторов/URL):

```bibtex
@misc{soft-event-frame-alignment,
  title        = {Soft Event-Frame Alignment},
  note         = {Code repository for "Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation"},
  year         = {2025},
  howpublished = {GitHub repository}
}
```

## 🙏 Благодарности

- Экосистема инструментов для event-камер, включая `dv-processing`.
- PyTorch и научные Python-библиотеки, используемые в экспериментах.

## 💡 Поддержка

Раздела поддержки/спонсорства в текущих метаданных репозитория нет. Если хотите добавить его, добавьте ссылки в следующем обновлении, и они будут сохранены в будущих ревизиях.

## 📄 Лицензия

Проект распространяется по лицензии Apache License 2.0. См. [LICENSE](../LICENSE).
