[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Soft Event-Frame Alignment

> Repositorio del artículo **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043)

## 🔎 Resumen

Este repositorio contiene código de investigación para alinear flujos de cámaras de eventos y cámaras de frames mediante una representación neuronal implícita unificada (INR/INN).

### Pipeline Principal (Ruta Canónica)

| Paso | Componente | Propósito |
|---|---|---|
| 1 | `softalign/data_processing.py` | Leer flujos AEDAT4, normalizar datos, crear muestras de puntos |
| 2 | `softalign/implicit_model.py` | Definir función implícita compartida + parámetros de alineación aprendibles |
| 3 | `softalign/training.py` + `main.py` | Optimizar conjuntamente la reconstrucción de eventos/frames |
| 4 | `evaluation.py` | Visualizar la alineación y reportar pérdidas |

Además de la ruta principal, el repositorio incluye scripts independientes y experimentales para variantes INR de solo eventos, solo frames, dispersas y basadas en derivadas.

## ✨ Características

- Alineación evento-frame con parámetros tipo afín aprendibles: `scale`, `shift_x`, `shift_y`, `shift_t`.
- Función implícita compartida `F(x, y, t)` utilizada por las ramas de eventos y de frames.
- Modelado de eventos mediante derivada temporal por diferencia finita con `threshold` y `dt` aprendibles.
- Ingesta AEDAT4 (`dv_processing`) además de flujos de trabajo con `.npy` procesados.
- Salidas de visualización integradas para curvas de pérdida, evolución de parámetros y superposiciones de alineación.
- Soporte de CUDA cuando está disponible (con fallback a CPU vía `torch.cuda.is_available()`).

## 🗂️ Estructura del Proyecto

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

## 📋 Requisitos Previos

- Se recomienda Python 3.10+.
- Los ejemplos siguientes usan shell de Linux/macOS (adáptalos para Windows si hace falta).
- Opcional pero recomendado: GPU con CUDA para acelerar el entrenamiento.
- La lectura de AEDAT4 requiere `dv_processing` y dependencias de sistema compatibles.

## ⚙️ Instalación

Actualmente no hay `requirements.txt` ni `pyproject.toml`, así que las dependencias se instalan manualmente.

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

## 🚀 Inicio Rápido

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

`evaluation.py` importa `EventFrameAlignmentModel` mediante `from implicit_model import ...`, mientras que el módulo activo está en `softalign/implicit_model.py`. Una invocación práctica desde la raíz del repositorio es:

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

Esto genera visualizaciones/métricas como `evaluation/alignment_visualization.png` y `evaluation/evaluation_results.txt`.

## 🧪 Uso

### Pipeline Principal (`main.py`)

```bash
python main.py --help
```

Opciones clave:
- `--filepath`: ruta de entrada AEDAT4.
- `--duration`: segundos a leer de la grabación.
- `--data_dir`: directorio de salida para datos procesados.
- `--checkpoint_dir`: directorio de salida para checkpoints.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` o `cpu`.
- `--reprocess`: forzar la regeneración de datos.

### Scripts Experimentales

INR solo eventos:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

INR solo frames:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# or
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# or
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

Variantes por derivada:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Variante dispersa:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Alineación monolítica standalone:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

## 🧩 Notas de Configuración

Valores por defecto del modelo principal en `softalign/implicit_model.py`:

| Parámetro | Valor por defecto |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

El preprocesamiento de datos en `softalign/data_processing.py` aplica actualmente un desalineamiento sintético a los eventos antes del entrenamiento. Esto es intencional en el código actual y debe considerarse al interpretar métricas.

## 🧠 Formulación Matemática

La red implícita compartida modela `F(x, y, t)`.

Rama de frames:
- Predice directamente una respuesta tipo intensidad usando `F(x, y, t)`.

Rama de eventos:
1. Aplicar transformación de eventos aprendible:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. Aproximar la derivada temporal:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. Producir respuesta de evento:
   - `sigmoid(dF/dt - threshold)`

El objetivo de entrenamiento combina:
- Pérdida MSE de eventos
- Pérdida MSE de frames
- Regularización sobre parámetros de alineación

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

## 🛠️ Notas de Desarrollo

- El repositorio está centrado en scripts (aún no tiene metadatos de empaquetado).
- Hay artefactos generados dentro del repo (checkpoints/resultados), por lo que el uso de disco puede ser grande.
- `readme-file.md` y `project-structure.txt` incluyen supuestos de estructura heredados y no están completamente alineados con el layout actual de la raíz.
- Actualmente no se detectan tests automatizados ni flujos de CI.

## 🧯 Solución de Problemas

- `ModuleNotFoundError: No module named 'implicit_model'` en `evaluation.py`:
  - Ejecuta con `PYTHONPATH=softalign` como se muestra arriba.
- Problemas de instalación/ejecución de `dv_processing`:
  - Confirma que tu plataforma y versión de Python estén soportadas por los wheels/libs de `dv-processing`.
- CUDA OOM:
  - Reduce `--batch_size`, reduce `--max_events` (donde esté disponible), o ejecuta con `--device cpu`.
- Falta el archivo AEDAT4:
  - Verifica la ruta en `yuqing/` o proporciona tu propia grabación mediante `--filepath` / `--aedat4_file`.

## 🗺️ Hoja de Ruta

- Añadir archivos de entorno reproducible (`requirements.txt` o `pyproject.toml`).
- Unificar rutas de importación para evaluación y ejecución como paquete/módulo.
- Añadir tests automatizados para procesamiento de datos y verificaciones forward/entrenamiento del modelo.
- Añadir CI para linting y smoke tests.
- Añadir archivos README multilingües bajo `i18n/`.

## 🤝 Contribuciones

Las contribuciones son bienvenidas.

1. Haz un fork del repositorio.
2. Crea una rama de funcionalidad.
3. Realiza cambios acotados y bien documentados.
4. Envía un pull request con detalles de reproducción y salidas de ejemplo cuando aplique.

Si planeas cambios grandes (nueva ruta de entrenamiento, refactor o cambios de dependencias), abre primero un issue para alinear la dirección.

## 📚 Cita

Si usas este repositorio en investigación, cita el artículo relacionado y/o el repositorio.

Metadatos actuales del repositorio disponibles en el texto existente del README:
- Título del artículo: *Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation*

Marcador de posición BibTeX del repositorio (actualiza autores/URL según corresponda):

```bibtex
@misc{soft-event-frame-alignment,
  title        = {Soft Event-Frame Alignment},
  note         = {Code repository for "Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation"},
  year         = {2025},
  howpublished = {GitHub repository}
}
```

## 🙏 Agradecimientos

- Ecosistema de herramientas para cámaras de eventos, incluido `dv-processing`.
- PyTorch y las librerías científicas de Python utilizadas en los experimentos.

## 💡 Soporte

La sección de soporte/patrocinio no está presente en los metadatos actuales del repositorio. Si quieres incluirla, añade enlaces en una actualización posterior y se conservarán en revisiones futuras.

## 📄 Licencia

Licenciado bajo Apache License 2.0. Consulta [LICENSE](LICENSE).
