[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

> Repository zum Paper **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white&style=flat-square)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043?style=flat-square)

| Fokus | Details |
| --- | --- |
| 🎯 Ziel | Align event-camera streams and frame-camera streams mit einer einheitlichen impliziten neuronalen Repräsentation |
| ⚙️ Haupt-Pipeline | `main.py` (train) + `evaluation.py` (analyse/visualisieren) |
| 🧪 Varianten-Skripte | `event_inn.py`, `frame_inn.py`, `event_deri*.py`, `event_inn_sparse.py` |

</div>

## Inhaltsverzeichnis

- [🔎 Überblick](#-überblick)
- [✨ Funktionen](#-funktionen)
- [🗂️ Projektstruktur](#-projektstruktur)
- [📋 Voraussetzungen](#-voraussetzungen)
- [⚙️ Installation](#-installation)
- [🚀 Schnellstart](#-schnellstart)
- [🧪 Nutzung](#-nutzung)
- [🧩 Konfigurationshinweise](#-konfigurationshinweise)
- [🧠 Mathematische Formulierung](#-mathematische-formulierung)
- [🧾 Beispiele](#-beispiele)
- [🛠️ Entwicklungshinweise](#-entwicklungshinweise)
- [🧯 Fehlerbehebung](#-fehlerbehebung)
- [🗺️ Roadmap](#-roadmap)
- [🤝 Mitwirken](#-mitwirken)
- [❤️ Support](#-support)
- [📄 Lizenz](#-lizenz)

## 🔎 Überblick

Dieses Repository enthält Forschungs-Code zum Abgleichen von Event-Kamera-Strömen und Frame-Kamera-Strömen mit einer einheitlichen impliziten neuronalen Repräsentation (INR/INN).

### Kern-Pipeline (kanonischer Weg)

| Schritt | Komponente | Zweck |
|---|---|---|
| 1 | `softalign/data_processing.py` | AEDAT4-Streams lesen, Daten normalisieren, Punktstichproben erzeugen |
| 2 | `softalign/implicit_model.py` | Gemeinsame implizite Funktion + lernbare Alignmentsparameter definieren |
| 3 | `softalign/training.py` + `main.py` | Gemeinsame Optimierung von Event-/Frame-Rekonstruktion |
| 4 | `evaluation.py` | Alignment-Visualisierung erstellen und Verluste protokollieren |

Neben dem Hauptpfad enthält das Repository eigenständige und experimentelle Skripte für Event-only-, Frame-only-, Sparse- sowie derivative-basierte INR-Varianten.

## ✨ Funktionen

- Event-Frame-Ausrichtung mit lernbaren Affin-Parameter-ähnlichen Parametern: `scale`, `shift_x`, `shift_y`, `shift_t`.
- Gemeinsame implizite Funktion `F(x, y, t)`, die sowohl in Event- als auch Frame-Zweig genutzt wird.
- Ereignismodellierung über endliche zeitliche Differenz mit lernbaren `threshold` und `dt`.
- AEDAT4-Einlesen (`dv_processing`) plus `.npy`-Workflows nach Vorverarbeitung.
- Integrierte Visualisierungen für Loss-Kurven, Parameterverlauf und Alignment-Overlays.
- CUDA-Unterstützung, sofern verfügbar (`torch.cuda.is_available()` mit Fallback auf CPU).
- Datensatz-Skripte für schnelle Dateninspektion und leichtes Debugging:
  - `read_events.py`
  - `read_frames.py`
  - `reader.py`, `reader_norm.py`, `simple_count.py`

## 🗂️ Projektstruktur

```text
SoftEventFrameAlignment/
├── README.md
├── LICENSE
├── main.py                              # Kanonischer Einstieg für Event-Frame-Training
├── evaluation.py                        # Kanonischer Einstieg für Auswertung/Visualisierung
├── softalign/
│   ├── __init__.py
│   ├── data_processing.py               # AEDAT4 Laden + Normalisierung + Frame-Punkt-Sampling
│   ├── implicit_model.py                # MLP + Alignmentsparameter
│   └── training.py                      # Dataset-Wrapper + Optimierungs-Loop
├── event_inn.py                         # Event-only INR-Experiment
├── frame_inn.py                         # Frame-only INR-Experiment
├── event_deri.py                        # Derivat-/Log-Derivat-Event-Variante
├── event_deri_2.py                      # Derivative-Variante mit zusätzlicher Null-Regularisierung
├── event_inn_sparse.py                  # Sparse Event INR (PyG/torch_scatter-Abhängigkeiten)
├── softalign_standalone.py              # Monolithischer End-to-End-Alignments-Workflow
├── softalign_old.py                     # Legacy-Implementierung für Vergleichszwecke
├── read_events.py                       # Event-I/O und Grundchecks für Vorverarbeitung
├── read_frames.py                       # Frame-/Video-Extraktion und Formvalidierung
├── reader.py                            # AEDAT-Reader-Hilfsprogramm
├── reader_norm.py                       # Reader mit Normalisierungshilfen
├── simple_count.py                      # Schlanke Event-Count-Hilfsfunktion
├── data/                                # Generierte Arrays (erstellt oder versioniert)
├── checkpoints/                         # Haupt-Trainings-Checkpoints und Plots
├── event_inn_results/                   # Generierte Versuchsausgaben
├── frame_inn_results/                   # Generierte Versuchsausgaben
├── event_derivative_results/             # Generierte Versuchsausgaben
├── alignment_results_*/                 # Zeitgestempelte Alignments-Durchläufe
├── yuqing/                              # Beispiel-AEDAT4-Dateien
├── events_processed.csv                  # Artefakt mit Event-Zusammenfassung
├── frames_processed.csv                  # Artefakt mit Frame-Zusammenfassung
└── i18n/                                # Übersetzte README-Dateien
```

## 📋 Voraussetzungen

- Python 3.10+ empfohlen.
- Linux/macOS-Befehlszeilenbeispiele unten (bei Bedarf an Windows anpassen).
- Optional, aber empfohlen: CUDA-fähige GPU für höhere Trainingsgeschwindigkeit.
- AEDAT4-Lesen erfordert `dv_processing` und passende System-Abhängigkeiten.

## ⚙️ Installation

Aktuell gibt es keine `requirements.txt` oder `pyproject.toml`; Abhängigkeiten werden daher manuell installiert.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

Optionale Abhängigkeiten für Sparse-Experimente (`event_inn_sparse.py`):

```bash
pip install scikit-learn torch-geometric torch-scatter
```

Annahme: Ob CUDA verfügbar ist, bestimmt das Standard-Device-Verhalten, aber alle Kommandos unten unterstützen ebenfalls explizit `--device cpu`.

## 🚀 Schnellstart

### 1. Haupt-Alignments-Modell trainieren

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

Ausgabe erfolgt nach:
- `data/` (`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`)
- `checkpoints/` (`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`)

### 2. Trainierten Checkpoint evaluieren

`evaluation.py` importiert `EventFrameAlignmentModel` via `from implicit_model import ...`, während das aktive Modul in `softalign/implicit_model.py` liegt. Ein praktischer Aufruf vom Repository-Root ist:

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

Dadurch entstehen Visualisierungen/Metriken wie `evaluation/alignment_visualization.png` und `evaluation/evaluation_results.txt`.

## 🧪 Nutzung

### Haupt-Pipeline (`main.py`)

```bash
python main.py --help
```

Wichtige Optionen:
- `--filepath`: AEDAT4-Eingabepfad.
- `--duration`: Sekunden, die aus der Aufnahme gelesen werden.
- `--data_dir`: Ausgabeverzeichnis der aufbereiteten Daten.
- `--checkpoint_dir`: Ausgabeverzeichnis der Checkpoints.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` oder `cpu`.
- `--reprocess`: Neu-Generierung der Daten erzwingen.

### Experimentelle Skripte

Event-only INR:
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# oder
python event_inn.py --events_file data/events.npy --output_dir output_event
```

Frame-only INR:
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# oder
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# oder
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

Derivative-Varianten:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Sparse-Variante:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Monolithisches Alignment:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

Zusätzliche Reader:
```bash
python read_events.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python read_frames.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python simple_count.py --input data/events.npy
```

## 🧩 Konfigurationshinweise

Standardwerte des Hauptmodells in `softalign/implicit_model.py`:

| Parameter | Standard |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

Die Datenvorverarbeitung in `softalign/data_processing.py` wendet aktuell eine synthetische Fehljustierung auf Events an, bevor das Training startet. Das ist im aktuellen Codeabschnitt beabsichtigt und sollte bei der Interpretation der Metriken berücksichtigt werden.

Trainings-Skripte bieten zusätzlich regelbare Parameter über CLI-Flags (z. B. `--hidden_dim`, `--num_layers` und `--batch_size`), die für Ablationsstudien nützlich sind.

## 🧠 Mathematische Formulierung

Das gemeinsame implizite Netzwerk modelliert `F(x, y, t)`.

Frame-Zweig:
- Sagt direkt eine intensitätsähnliche Antwort mit `F(x, y, t)` voraus.

Event-Zweig:
1. Lernbare Event-Transformation anwenden:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. Zeitliche Ableitung approximieren:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. Event-Antwort erzeugen:
   - `sigmoid(dF/dt - threshold)`

Das Trainingsziel kombiniert:
- Event-MSE-Verlust
- Frame-MSE-Verlust
- Regularisierung der Alignmentsparameter

## 🧾 Beispiele

Nur vorverarbeitete Arrays verwenden:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

CPU-Run erzwingen:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

In benutzerdefinierten Ordner evaluieren:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ Entwicklungshinweise

- Das Repository ist skriptzentriert (noch keine Packaging-Metadaten).
- Generierte Artefakte liegen im Repo (Checkpoints/Ergebnisse), daher kann der Speicherbedarf hoch sein.
- `readme-file.md` und `project-structure.txt` enthalten Annahmen zu älteren Strukturen und sind nicht vollständig auf das aktuelle Root-Layout ausgerichtet.
- Es gibt aktuell keine automatischen Tests oder CI-Workflows.

## 🧯 Fehlerbehebung

- `ModuleNotFoundError: No module named 'implicit_model'` in `evaluation.py`:
  - Wie oben gezeigt mit `PYTHONPATH=softalign` ausführen.
- Probleme bei Installation/Laufzeit von `dv_processing`:
  - Prüfen, ob Plattform und Python-Version von den Wheels/libs von `dv-processing` unterstützt werden.
- CUDA OOM:
  - `--batch_size` reduzieren, `--max_events` (wo verfügbar) senken oder mit `--device cpu` starten.
- Fehlende AEDAT4-Datei:
  - Pfad unter `yuqing/` prüfen oder eigene Aufnahme über `--filepath` / `--aedat4_file` bereitstellen.
- Datenabweichung zwischen Runs:
  - Bei geänderten Vorverarbeitungsannahmen erneut mit `--reprocess` ausführen.

## 🗺️ Roadmap

- Reproduzierbare Umgebungsdateien ergänzen (`requirements.txt` oder `pyproject.toml`).
- Importpfade für Auswertung und Paket-/Modulausführung vereinheitlichen.
- Automatisierte Tests für Datenverarbeitung und Modell-Forward/Training-Checks hinzufügen.
- CI für Linting und Smoke-Tests ergänzen.
- Mehrsprachige README-Dateien unter `i18n/` hinzufügen.

## 🤝 Mitwirken

Beiträge sind willkommen.

Empfohlener Ablauf:
1. Ein fokussiertes Issue oder einen Branch mit der Änderung des Experiments anlegen.
2. Skripte konsistent halten (besonders Importe, CLI-Konventionen und Ausgabepfade).
3. Vorhandene Verhalten von Experimentartefakten beibehalten, außer bei Änderungen an Namenskonventionen/Versionierung.

## 📄 Lizenz

Dieses Repository ist unter der Apache License 2.0 veröffentlicht. Den vollständigen Text findest du in [`LICENSE`](../LICENSE).

Annahme: Wenn dieses Repository in Publikationen verwendet wird, sollten Zitierangaben nach Bedarf hier und in den Release-Notizen ergänzt werden.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
