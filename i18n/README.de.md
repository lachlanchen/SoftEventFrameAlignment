[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

*Minimale Forschungssoftware zum Lernen einer räumlichen und zeitlichen Ausrichtung von Event- und Frame-Kamera-Samples mit einer gemeinsamen impliziten neuronalen Repräsentation.*

[![CI](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA043)](../LICENSE)
[![Sponsor](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)

Dies ist die erste öffentliche, klonbare Implementierung zu **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**. Sie enthält das kanonische Modell, AEDAT4-Vorverarbeitung, Training, Auswertung, Abhängigkeitsmetadaten, CI und einen deterministischen synthetischen CPU-Test.

Aufnahmen, erkennbare Frames, abgeleitete Arrays und trainierte Checkpoints werden nicht veröffentlicht. Verwende nur Daten, für die du ausreichende Rechte hast.

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=kofi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## Architektur

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

Der Event-Zweig transformiert Koordinaten und approximiert die zeitliche Ableitung eines gemeinsamen MLP. Der Frame-Zweig wertet dasselbe Feld direkt aus. Maßstab und Ableitungsschritt sind positiv parametrisiert, sodass keine Division durch null entsteht.

## Inhalt der öffentlichen Version

| Pfad | Zweck |
| --- | --- |
| `softalign/implicit_model.py` | Gemeinsames implizites MLP und lernbare Ausrichtung |
| `softalign/training.py` | Validierter Datensatz-Wrapper und gemeinsames Training |
| `softalign/data_processing.py` | Optionale AEDAT4-Eingabe und Vorverarbeitung in Sekunden |
| `softalign/synthetic.py` | Deterministische, projektgenerierte Testdaten |
| `main.py` / `evaluation.py` | Kommandozeilen für Training und Diagnose |
| `examples/` / `tests/` | Ende-zu-Ende- und Kerntests |

Historische Experimente, Rohaufnahmen, lokale Datensätze und alte Checkpoints gehören absichtlich nicht zu dieser Version.

## Installation und Prüfung

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,viz]'
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py --output-dir .smoke-output --epochs 8
```

Der Test erzeugt einen Checkpoint, Diagramme und `evaluation_results.json`. Er belegt die ausführbare Pipeline und endliche Werte, nicht die Genauigkeit der Ausrichtung.

## Eigene AEDAT4-Aufnahme verwenden

```bash
python -m pip install -e '.[aedat,viz]'
python main.py --filepath /path/to/your-recording.aedat4 --reprocess --data_dir data --checkpoint_dir checkpoints
python evaluation.py --model_path checkpoints/model_final.pt --data_dir data --output_dir evaluation --device cpu
```

Die Vorverarbeitung speichert das Schema `softalign-processed/v1` und wandelt AEDAT-Zeitstempel von Mikrosekunden in Sekunden um. Die Teststörung ist standardmäßig deaktiviert und wird nur mit `--synthetic-misalignment` aktiviert. Alte Arrays ohne passende Metadaten werden abgelehnt.

Große Sammlungen bleiben im CPU-Speicher; nur gezogene Batches wechseln auf das Zielgerät. Beginne portabel mit `--device cpu` und nutze CUDA erst nach einer Speicherprüfung.

## Datenvertrag

| Datei | Bedeutung |
| --- | --- |
| `events.npy` | `N × 4`: normierte x/y, Zeit in Sekunden, Event-Ziel |
| `frame_points.npy` | `M × 4`: normierte x/y, Zeit in Sekunden, Intensität |
| `preprocessing.json` | Schema, Einheiten, Seed, Transformationsflag und Anzahlen |
| `model_final.pt` | Modellzustand, Parameter und Architektur |
| `evaluation_results.json` | Rekonstruktions-MSE auf begrenzten Eingabesamples |

Die Auswertung ist eine Diagnose der gelieferten Arrays, kein Hold-out-Messwert, Benchmark oder Nachweis einer realen Sensorkalibrierung.

## Validierung und Forschungsgrenzen

- CI führt einen deterministischen CPU-Test und einen kurzen Train/Evaluate-Lauf aus.
- Form, Leere, endliche Werte, Zeiteinheiten und Geräteverfügbarkeit werden geprüft.
- Der aktuelle Verlust ist eine Forschungsbaseline; reale Studien brauchen Ground Truth, getrennte Auswertung, Ablationen und Unsicherheitsanalyse.
- Private Aufnahmen gehören nicht in öffentliche GitHub-Issues.

## Zitieren

Wenn du Soft Event-Frame Alignment in der Forschung nutzt, zitiere das Repository. GitHub liest [CITATION.cff](../CITATION.cff) und zeigt **Cite this repository** an.

```bibtex
@software{chen_soft_event_frame_alignment_2026,
  author = {Chen, Lachlan},
  title = {Soft Event-Frame Alignment: Unified Implicit Neural Representation Research Code},
  year = {2026},
  url = {https://github.com/lachlanchen/SoftEventFrameAlignment}
}
```

## Status

Version `0.1.0` ist eine minimale Forschungsversion. Reproduzierbare Fehler mit dem synthetischen Test oder einer kleinen freigegebenen Probe sind willkommen.

## Lizenz

Apache License 2.0. Siehe [LICENSE](../LICENSE).
