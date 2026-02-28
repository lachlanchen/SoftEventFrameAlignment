[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Soft Event-Frame Alignment

> Repository zum Paper **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043)

## 🔎 Überblick

Dieses Repository enthält Forschungscode zur Ausrichtung von Event-Kamera-Streams und Frame-Kamera-Streams mit einer einheitlichen impliziten neuronalen Repräsentation (INR/INN).

### Kern-Pipeline (kanonischer Pfad)

| Schritt | Komponente | Zweck |
|---|---|---|
| 1 | `softalign/data_processing.py` | AEDAT4-Streams einlesen, Daten normalisieren, Punkt-Samples erzeugen |
| 2 | `softalign/implicit_model.py` | Gemeinsame implizite Funktion + lernbare Ausrichtungsparameter definieren |
| 3 | `softalign/training.py` + `main.py` | Event-/Frame-Rekonstruktion gemeinsam optimieren |
| 4 | `evaluation.py` | Ausrichtung visualisieren und Losses berichten |

Neben dem Hauptpfad enthält das Repository eigenständige und experimentelle Skripte für Event-only-, Frame-only-, sparse- und derivative-basierte INR-Varianten.

## ✨ Features

- Event-Frame-Ausrichtung mit lernbaren affine-artigen Parametern: `scale`, `shift_x`, `shift_y`, `shift_t`.
- Gemeinsame implizite Funktion `F(x, y, t)`, die von Event- und Frame-Zweig genutzt wird.
- Event-Modellierung über finite-difference-Zeitableitung mit lernbarem `threshold` und `dt`.
- AEDAT4-Einlesung (`dv_processing`) plus Workflows mit verarbeiteten `.npy`-Dateien.
- Integrierte Visualisierungen für Loss-Kurven, Parameterverläufe und Alignment-Overlays.
- CUDA-Unterstützung, falls verfügbar (`torch.cuda.is_available()` mit Fallback auf CPU).

## 🗂️ Projektstruktur

```text
SoftEventFrameAlignment/
├── README.md
├── LICENSE
├── main.py                          # Kanonischer Einstiegspunkt für Event-Frame-Training
├── evaluation.py                    # Kanonischer Einstiegspunkt für Evaluation/Visualisierung
├── softalign/
│   ├── __init__.py
│   ├── data_processing.py           # AEDAT4-Laden + Normalisierung + Sampling von Frame-Punkten
│   ├── implicit_model.py            # MLP + Ausrichtungsparameter
│   └── training.py                  # Dataset-Wrapper + Optimierungsschleife
├── event_inn.py                     # Event-only-INR-Experiment
├── frame_inn.py                     # Frame-only-INR-Experiment
├── event_deri.py                    # Derivative/Log-Derivative-Event-Variante
├── event_deri_2.py                  # Derivative-Variante mit zusätzlicher Null-Regularisierung
├── event_inn_sparse.py              # Sparse Event INR (PyG/torch_scatter-Abhängigkeiten)
├── softalign_standalone.py          # Monolithischer All-in-one-Alignment-Workflow
├── data/                            # Verarbeitete Arrays (generiert oder eingecheckt)
├── checkpoints/                     # Haupt-Training-Checkpoints und Plots
├── event_inn_results/               # Generierte Experimentausgaben
├── frame_inn_results/               # Generierte Experimentausgaben
├── event_drivative_results/         # Generierte Experimentausgaben
├── alignment_results_*/             # Zeitgestempelte Alignment-Läufe
├── yuqing/                          # Beispiel-AEDAT4-Dateien
└── i18n/                            # Zielpfad für Übersetzungsdateien
```

## 📋 Voraussetzungen

- Python 3.10+ empfohlen.
- Shell-Beispiele unten für Linux/macOS (bei Bedarf für Windows anpassen).
- Optional, aber empfohlen: CUDA-fähige GPU für schnelleres Training.
- Für das Lesen von AEDAT4 wird `dv_processing` mit kompatiblen Systemabhängigkeiten benötigt.

## ⚙️ Installation

Aktuell sind weder `requirements.txt` noch `pyproject.toml` vorhanden, daher Abhängigkeiten manuell installieren.

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

## 🚀 Schnellstart

### 1. Haupt-Alignment-Modell trainieren

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

Ausgaben werden hier geschrieben:
- `data/` (`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`)
- `checkpoints/` (`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`)

### 2. Trainierten Checkpoint auswerten

`evaluation.py` importiert `EventFrameAlignmentModel` via `from implicit_model import ...`, während das aktive Modul in `softalign/implicit_model.py` liegt. Ein praktikabler Aufruf aus dem Repository-Root ist:

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

Dadurch werden Visualisierungen/Metriken wie `evaluation/alignment_visualization.png` und `evaluation/evaluation_results.txt` erzeugt.

## 🧪 Nutzung

### Haupt-Pipeline (`main.py`)

```bash
python main.py --help
```

Wichtige Optionen:
- `--filepath`: AEDAT4-Eingabepfad.
- `--duration`: Sekunden, die aus der Aufnahme gelesen werden.
- `--data_dir`: Ausgabeverzeichnis für verarbeitete Daten.
- `--checkpoint_dir`: Ausgabeverzeichnis für Checkpoints.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device`: `cuda` oder `cpu`.
- `--reprocess`: Erzwingt Neuaufbereitung der Daten.

### Experimentelle Skripte

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

Derivative-Varianten:
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Sparse-Variante:
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Eigenständige monolithische Ausrichtung:
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
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

Die Datenvorverarbeitung in `softalign/data_processing.py` wendet vor dem Training aktuell eine synthetische Fehl-Ausrichtung auf Events an. Das ist im aktuellen Code beabsichtigt und sollte bei der Interpretation von Metriken berücksichtigt werden.

## 🧠 Mathematische Formulierung

Das gemeinsame implizite Netzwerk modelliert `F(x, y, t)`.

Frame-Zweig:
- Sagt direkt eine intensitätsähnliche Antwort mit `F(x, y, t)` vorher.

Event-Zweig:
1. Lernbare Event-Transformation anwenden:
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. Zeitliche Ableitung approximieren:
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. Event-Antwort erzeugen:
   - `sigmoid(dF/dt - threshold)`

Trainingsziel kombiniert:
- Event-MSE-Loss
- Frame-MSE-Loss
- Regularisierung über Ausrichtungsparameter

## 🧾 Beispiele

Nur vorverarbeitete Arrays verwenden:

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

CPU-Lauf erzwingen:

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

Auswertung in benutzerdefinierten Ordner:

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ Entwicklungshinweise

- Repository ist skriptzentriert (noch keine Packaging-Metadaten).
- Generierte Artefakte liegen im Repository (Checkpoints/Results), daher kann der Speicherbedarf groß sein.
- `readme-file.md` und `project-structure.txt` enthalten Legacy-Strukturannahmen und sind nicht vollständig mit dem aktuellen Root-Layout abgestimmt.
- Derzeit wurden keine automatisierten Tests oder CI-Workflows erkannt.

## 🧯 Fehlerbehebung

- `ModuleNotFoundError: No module named 'implicit_model'` in `evaluation.py`:
  - Mit `PYTHONPATH=softalign` wie oben gezeigt ausführen.
- `dv_processing` Installations-/Laufzeitprobleme:
  - Prüfen, ob Plattform und Python-Version von `dv-processing` Wheels/Libs unterstützt werden.
- CUDA OOM:
  - `--batch_size` reduzieren, `--max_events` (wo verfügbar) senken oder mit `--device cpu` ausführen.
- Fehlende AEDAT4-Datei:
  - Pfad unter `yuqing/` prüfen oder eigene Aufnahme über `--filepath` / `--aedat4_file` angeben.

## 🗺️ Roadmap

- Reproduzierbare Umgebungsdateien hinzufügen (`requirements.txt` oder `pyproject.toml`).
- Importpfade für Evaluation sowie Package-/Modulausführung vereinheitlichen.
- Automatisierte Tests für Datenverarbeitung und Model-Forward-/Training-Checks ergänzen.
- CI für Linting und Smoke-Tests ergänzen.
- Mehrsprachige README-Dateien unter `i18n/` ergänzen.

## 🤝 Mitwirken

Beiträge sind willkommen.

1. Repository forken.
2. Einen Feature-Branch erstellen.
3. Fokussierte, gut dokumentierte Änderungen vornehmen.
4. Pull Request mit Reproduktionsdetails und ggf. Beispielausgaben einreichen.

Wenn du größere Änderungen planst (neuer Training-Pfad, Refactor oder Dependency-Änderungen), eröffne bitte zuerst ein Issue, um die Richtung abzustimmen.

## 📚 Zitation

Wenn du dieses Repository in der Forschung verwendest, zitiere bitte das zugehörige Paper und/oder Repository.

Aktuell verfügbare Repository-Metadaten aus dem bestehenden README-Text:
- Paper-Titel: *Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation*

Ein Repository-BibTeX-Platzhalter (Autoren/URL nach Bedarf aktualisieren):

```bibtex
@misc{soft-event-frame-alignment,
  title        = {Soft Event-Frame Alignment},
  note         = {Code repository for "Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation"},
  year         = {2025},
  howpublished = {GitHub repository}
}
```

## 🙏 Danksagung

- Ökosystem für Event-Kamera-Tooling, einschließlich `dv-processing`.
- PyTorch und wissenschaftliche Python-Bibliotheken, die in den Experimenten verwendet werden.

## 💡 Support

Ein Support-/Sponsoring-Abschnitt ist in den aktuellen Repository-Metadaten nicht vorhanden. Falls du einen möchtest, füge Links in einem Folge-Update hinzu; sie werden in künftigen Revisionen beibehalten.

## 📄 Lizenz

Lizenziert unter der Apache License 2.0. Siehe [LICENSE](../LICENSE).
