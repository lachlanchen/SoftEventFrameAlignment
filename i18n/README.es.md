[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

*Código de investigación mínimo para aprender la alineación espacial y temporal entre muestras de cámaras de eventos y de fotogramas mediante una única representación neuronal implícita.*

[![CI](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA043)](../LICENSE)
[![Sponsor](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)

Esta es la primera implementación pública y clonable de **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**. Incluye el modelo canónico, preprocesamiento AEDAT4, entrenamiento, evaluación, metadatos de dependencias, CI y una prueba sintética determinista en CPU.

No se publican grabaciones, fotogramas identificables, matrices derivadas ni checkpoints entrenados. Usa únicamente datos para los que tengas derechos suficientes.

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=kofi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## Arquitectura

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

La rama de eventos transforma las coordenadas y aproxima la derivada temporal de un MLP compartido. La rama de fotogramas evalúa directamente el mismo campo. La escala y el paso de derivación usan parametrizaciones positivas para evitar divisiones por cero.

## Contenido de la versión pública

| Ruta | Función |
| --- | --- |
| `softalign/implicit_model.py` | MLP implícito compartido y parámetros de alineación |
| `softalign/training.py` | Adaptador de datos validado y entrenamiento conjunto |
| `softalign/data_processing.py` | Entrada AEDAT4 opcional y preprocesamiento en segundos |
| `softalign/synthetic.py` | Datos sintéticos deterministas creados por el proyecto |
| `main.py` / `evaluation.py` | CLI de entrenamiento y diagnóstico |
| `examples/` / `tests/` | Comprobación integral y prueba del núcleo |

Los experimentos históricos, capturas originales, datos locales y checkpoints antiguos quedan fuera de esta versión de forma intencionada.

## Instalación y verificación

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,viz]'
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py --output-dir .smoke-output --epochs 8
```

La prueba crea un checkpoint, gráficos y `evaluation_results.json`. Comprueba que el flujo se ejecuta con valores finitos; no demuestra precisión de alineación.

## Uso con tu propia grabación AEDAT4

```bash
python -m pip install -e '.[aedat,viz]'
python main.py --filepath /path/to/your-recording.aedat4 --reprocess --data_dir data --checkpoint_dir checkpoints
python evaluation.py --model_path checkpoints/model_final.pt --data_dir data --output_dir evaluation --device cpu
```

El preprocesamiento registra el esquema `softalign-processed/v1` y convierte las marcas AEDAT de microsegundos a segundos. La perturbación sintética está desactivada por defecto y solo se activa con `--synthetic-misalignment`. Se rechazan matrices antiguas sin metadatos de unidades compatibles.

Las colecciones grandes permanecen en memoria de CPU y solo los lotes muestreados pasan al dispositivo seleccionado. Empieza con `--device cpu` y usa CUDA tras comprobar la memoria.

## Contrato de datos

| Archivo | Significado |
| --- | --- |
| `events.npy` | `N × 4`: x/y normalizadas, tiempo en segundos y objetivo de evento |
| `frame_points.npy` | `M × 4`: x/y normalizadas, tiempo en segundos e intensidad |
| `preprocessing.json` | Esquema, unidades, semilla, indicador de transformación y cantidades |
| `model_final.pt` | Estado, parámetros y arquitectura del modelo |
| `evaluation_results.json` | MSE de reconstrucción sobre muestras acotadas de entrada |

Los resultados son diagnósticos sobre las matrices suministradas, no métricas independientes, un benchmark ni prueba de calibración física.

## Validación y límites

- CI ejecuta una prueba determinista en CPU y un ciclo breve de entrenamiento/evaluación.
- Se validan forma, vacío, finitud, unidades temporales y disponibilidad del dispositivo.
- La pérdida actual es una base de investigación; un estudio real requiere ground truth, evaluación separada, ablaciones e incertidumbre.
- No adjuntes grabaciones privadas a incidencias públicas de GitHub.

## Cita

Si usas el proyecto en investigación, cita el repositorio. GitHub lee [CITATION.cff](../CITATION.cff) y muestra **Cite this repository**.

```bibtex
@software{chen_soft_event_frame_alignment_2026,
  author = {Chen, Lachlan},
  title = {Soft Event-Frame Alignment: Unified Implicit Neural Representation Research Code},
  year = {2026},
  url = {https://github.com/lachlanchen/SoftEventFrameAlignment}
}
```

## Estado

La versión `0.1.0` es una publicación mínima de investigación. Se aceptan fallos reproducibles con la prueba sintética o una muestra pequeña autorizada.

## Licencia

Apache License 2.0. Consulta [LICENSE](../LICENSE).
