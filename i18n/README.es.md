[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

> Repositorio del artículo **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

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

## Tabla de contenidos

- [🔎 Visión general](#-visión-general)
- [✨ Características](#-características)
- [🗂️ Estructura del proyecto](#-estructura-del-proyecto)
- [📋 Requisitos previos](#-requisitos-previos)
- [⚙️ Instalación](#-instalación)
- [🚀 Inicio rápido](#-inicio-rápido)
- [🧪 Uso](#-uso)
- [🧩 Notas de configuración](#-notas-de-configuración)
- [🧠 Formulación matemática](#-formulación-matemática)
- [🧾 Ejemplos](#-ejemplos)
- [🛠️ Notas de desarrollo](#-notas-de-desarrollo)
- [🧯 Resolución de problemas](#-resolución-de-problemas)
- [🗺️ Hoja de ruta](#-hoja-de-ruta)
- [🤝 Cómo contribuir](#-cómo-contribuir)
- [❤️ Soporte](#-soporte)
- [📄 Licencia](#-licencia)

## 🔎 Visión general

Este repositorio contiene código de investigación para alinear flujos de cámara de eventos y flujos de cámara de frames con una representación neuronal implícita unificada (INR/INN).

### Ruta principal (canónica)

| Paso | Componente | Propósito |
|---|---|---|
| 1 | `softalign/data_processing.py` | Leer streams AEDAT4, normalizar datos y crear muestras de puntos |
| 2 | `softalign/implicit_model.py` | Definir función implícita compartida + parámetros de alineación aprendibles |
| 3 | `softalign/training.py` + `main.py` | Optimizar conjuntamente la reconstrucción de eventos y frames |
| 4 | `evaluation.py` | Visualizar la alineación y reportar pérdidas |

Además de la ruta principal, el repositorio incluye scripts independientes y experimentales para variantes solo de eventos, solo de frames, dispersas y variantes basadas en derivadas.

## ✨ Características

- Alineación evento-frame con parámetros tipo afín aprendibles: `scale`, `shift_x`, `shift_y`, `shift_t`.
- Función implícita compartida `F(x, y, t)` usada por las ramas de evento y frame.
- Modelado de eventos mediante derivada temporal por diferencias finitas con `threshold` y `dt` aprendibles.
- Ingesta de AEDAT4 (`dv_processing`) más flujos de trabajo de `.npy` procesados.
- Salidas de visualización integradas para curvas de pérdida, evolución de parámetros y superposiciones de alineación.
- Soporte CUDA cuando está disponible (`torch.cuda.is_available()` con fallback a CPU).
- Scripts de dataset para inspección rápida de datos y depuración ligera:
  - `read_events.py`
  - `read_frames.py`
  - `reader.py`, `reader_norm.py`, `simple_count.py`

## 🗂️ Estructura del proyecto

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

## 📋 Requisitos previos

- Python 3.10+ recomendado.
- Ejemplos de shell de Linux/macOS abajo (adáptalos según necesites para Windows).
- Opcional pero recomendado: GPU con CUDA para acelerar el entrenamiento.
- La lectura de AEDAT4 requiere `dv_processing` y dependencias de sistema compatibles.

## ⚙️ Instalación

No existe actualmente `requirements.txt` ni `pyproject.toml`, así que instala las dependencias manualmente.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

Dependencias opcionales para experimentos dispersos (`event_inn_sparse.py`):

```bash
pip install scikit-learn torch-geometric torch-scatter
```

Supuesto: la disponibilidad de CUDA determina el comportamiento por defecto del dispositivo, pero todos los comandos abajo también soportan `--device cpu` explícito.

## 🚀 Inicio rápido

### 1. Entrenar el modelo principal de alineación

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

Las salidas se escriben en:
- `data/` (`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`)
- `checkpoints/` (`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`)

### 2. Evaluar un checkpoint entrenado

`evaluation.py` importa `EventFrameAlignmentModel` vía `from implicit_model import ...`, mientras que el módulo activo está en `softalign/implicit_model.py`. Una llamada práctica desde la raíz del repositorio es:

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

Esto genera visualizaciones/métricas como `evaluation/alignment_visualization.png` y `evaluation/evaluation_results.txt`.

## 🧪 Uso

### Pipeline principal (`main.py`)

```bash
python main.py --help
```

Opciones clave:
- `--filepath`: ruta de entrada AEDAT4.
- `--duration`: segundos a leer de la grabación.
- `--data_dir`: directorio de salida de datos procesados.
- `--checkpoint_dir`: directorio de salida de checkpoints.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` o `cpu`.
- `--reprocess`: forzar regeneración de datos.

### Scripts experimentales

INR solo de eventos:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

INR solo de frames:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# or
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# or
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

Variantes con derivada:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Variante dispersa:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Alineación monolítica independiente:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

Lectores adicionales:
```bash
python read_events.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python read_frames.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python simple_count.py --input data/events.npy
```

## 🧩 Notas de configuración

Valores por defecto del modelo principal en `softalign/implicit_model.py`:

| Parámetro | Valor por defecto |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

El preprocesamiento de datos en `softalign/data_processing.py` actualmente aplica un desajuste sintético a los eventos antes del entrenamiento. Esto es intencional en el código actual y debe tenerse en cuenta al interpretar métricas.

Los scripts de entrenamiento también exponen parámetros ajustables vía flags de la CLI (por ejemplo `--hidden_dim`, `--num_layers` y `--batch_size`) que son útiles para estudios de ablación.

## 🧠 Formulación matemática

La red implícita compartida modela `F(x, y, t)`.

Rama de frame:
- Predice directamente una respuesta tipo intensidad usando `F(x, y, t)`.

Rama de eventos:
1. Aplicar transformación de evento aprendible:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. Aproximar la derivada temporal:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. Generar respuesta de evento:
   - `sigmoid(dF/dt - threshold)`

La función objetivo del entrenamiento combina:
- Pérdida MSE de eventos
- Pérdida MSE de frames
- Regularización de parámetros de alineación

## 🧾 Ejemplos

Usar solo arrays preprocesados:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

Forzar ejecución en CPU:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

Evaluar en carpeta personalizada:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ Notas de desarrollo

- El repositorio es centrado en scripts (aún no hay metadatos de empaquetado).
- Hay artefactos generados dentro del repositorio (checkpoints/resultados), por lo que el uso de disco puede ser grande.
- `readme-file.md` y `project-structure.txt` contienen supuestos de estructura legados y no están completamente alineados con el layout actual del root.
- Actualmente no se detectan pruebas automatizadas ni flujos de CI.

## 🧯 Resolución de problemas

- `ModuleNotFoundError: No module named 'implicit_model'` en `evaluation.py`:
  - Ejecuta con `PYTHONPATH=softalign` como se muestra arriba.
- Problemas de instalación/ejecución de `dv_processing`:
  - Verifica que tu plataforma y versión de Python sean compatibles con los wheel/libs de `dv-processing`.
- CUDA OOM:
  - Reduce `--batch_size`, baja `--max_events` (donde esté disponible), o ejecuta con `--device cpu`.
- Falta de archivo AEDAT4:
  - Verifica la ruta en `yuqing/` o provee tu propia grabación mediante `--filepath` / `--aedat4_file`.
- Diferencias de datos entre ejecuciones:
  - Vuelve a ejecutar con `--reprocess` al cambiar suposiciones de preprocesamiento.

## 🗺️ Hoja de ruta

- Añadir archivos de entorno reproducibles (`requirements.txt` o `pyproject.toml`).
- Unificar rutas de importación para evaluación y ejecución del paquete/módulo.
- Añadir pruebas automatizadas para procesamiento de datos y chequeos forward/entrenamiento del modelo.
- Añadir CI para linting y smoke tests.
- Añadir README multilingües bajo `i18n/`.

## 🤝 Cómo contribuir

Las contribuciones son bienvenidas.

Flujo recomendado:
1. Crea un issue enfocado o una rama describiendo el cambio experimental.
2. Mantén los scripts alineados (especialmente imports, convenciones de CLI y nombres de directorios de salida).
3. Conserva el comportamiento de los artefactos de experimentos existentes salvo que cambies nombres/versionado.

## 📄 Licencia

Este repositorio se publica bajo la Apache License 2.0. Consulta [`LICENSE`](LICENSE) para el texto completo.

Supuesto: si este repositorio se usa en publicaciones, los detalles de citación deben agregarse aquí y en las notas de release según sea necesario.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
