[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

> Dépôt de recherche lié à l’article **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

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

## Table des matières

- [🔎 Présentation](#-présentation)
- [✨ Fonctionnalités](#-fonctionnalités)
- [🗂️ Structure du projet](#-structure-du-projet)
- [📋 Prérequis](#-prérequis)
- [⚙️ Installation](#-installation)
- [🚀 Démarrage rapide](#-démarrage-rapide)
- [🧪 Utilisation](#-utilisation)
- [🧩 Notes de configuration](#-notes-de-configuration)
- [🧠 Formulation mathématique](#-formulation-mathématique)
- [🧾 Exemples](#-exemples)
- [🛠️ Notes de développement](#-notes-de-développement)
- [🧯 Résolution des problèmes](#-résolution-des-problèmes)
- [🗺️ Feuille de route](#-feuille-de-route)
- [🤝 Contribution](#-contribution)
- [❤️ Support](#-support)
- [📄 Licence](#-licence)

## 🔎 Présentation

Ce dépôt contient le code de recherche pour aligner les flux de caméra à événements et de caméra d’images avec une représentation implicite unifiée (INR/INN).

### Chemin principal (parcours canonique)

| Étape | Composant | Objectif |
|---|---|---|
| 1 | `softalign/data_processing.py` | Lire les flux AEDAT4, normaliser les données, créer des échantillons de points |
| 2 | `softalign/implicit_model.py` | Définir la fonction implicite partagée + les paramètres d’alignement entraînables |
| 3 | `softalign/training.py` + `main.py` | Optimiser conjointement la reconstruction d’événements et d’images |
| 4 | `evaluation.py` | Visualiser l’alignement et afficher les pertes |

En plus du chemin principal, le dépôt inclut des scripts autonomes et expérimentaux pour les variantes événement seulement, image seulement, sparse, et dérivées.

## ✨ Fonctionnalités

- Alignement événement-image avec des paramètres affine-like entraînables : `scale`, `shift_x`, `shift_y`, `shift_t`.
- Fonction implicite partagée `F(x, y, t)` utilisée par les deux branches (événement et image).
- Modélisation d’événements via la dérivée temporelle en différences finies avec `threshold` et `dt` entraînables.
- Ingestion AEDAT4 (`dv_processing`) plus flux de travail `.npy` prétraités.
- Sorties de visualisation intégrées pour les courbes de perte, l’évolution des paramètres et les superpositions d’alignement.
- Support CUDA lorsqu’il est disponible (`torch.cuda.is_available()` avec bascule CPU sinon).
- Scripts de jeu de données pour une inspection rapide et un débogage léger :
  - `read_events.py`
  - `read_frames.py`
  - `reader.py`, `reader_norm.py`, `simple_count.py`

## 🗂️ Structure du projet

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

## 📋 Prérequis

- Python 3.10+ recommandé.
- Exemples de shell Linux/macOS ci-dessous (adapter sous Windows si nécessaire).
- Optionnel mais recommandé : GPU compatible CUDA pour accélérer l’entraînement.
- La lecture AEDAT4 requiert `dv_processing` et des dépendances système compatibles.

## ⚙️ Installation

Aucun `requirements.txt` ni `pyproject.toml` n’est présent actuellement, donc les dépendances doivent être installées manuellement.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

Dépendances optionnelles pour les expériences sparsées (`event_inn_sparse.py`) :

```bash
pip install scikit-learn torch-geometric torch-scatter
```

On suppose que la disponibilité de CUDA détermine le comportement par défaut de l’appareil, mais toutes les commandes ci-dessous supportent aussi `--device cpu`.

## 🚀 Démarrage rapide

### 1. Entraîner le modèle d’alignement principal

```bash
python main.py \
  --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 \
  --duration 10 \
  --num_epochs 1000 \
  --lr 1e-3 \
  --reprocess
```

Les sorties sont écrites dans :
- `data/` (`events.npy`, `frames_timestamps.npy`, `frame_points.npy`, `sample_frame.png`)
- `checkpoints/` (`model_epoch_*.pt`, `model_final.pt`, `loss_curves.png`, `parameter_history.png`)

### 2. Évaluer un point de contrôle entraîné

`evaluation.py` importe `EventFrameAlignmentModel` via `from implicit_model import ...`, alors que le module actif se trouve dans `softalign/implicit_model.py`. Un appel pratique depuis la racine du dépôt est :

```bash
PYTHONPATH=softalign python evaluation.py \
  --model_path checkpoints/model_final.pt \
  --data_dir data \
  --output_dir evaluation
```

Cela génère des visualisations/métriques telles que `evaluation/alignment_visualization.png` et `evaluation/evaluation_results.txt`.

## 🧪 Utilisation

### Pipeline principal (`main.py`)

```bash
python main.py --help
```

Options principales :
- `--filepath` : chemin d’entrée AEDAT4.
- `--duration` : nombre de secondes à lire depuis l’enregistrement.
- `--data_dir` : répertoire de sortie des données traitées.
- `--checkpoint_dir` : répertoire de sortie des points de contrôle.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device` : `cuda` ou `cpu`.
- `--reprocess` : forcer la régénération des données.

### Scripts expérimentaux

INR basé uniquement sur les événements :

```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# ou
python event_inn.py --events_file data/events.npy --output_dir output_event
```

INR basé uniquement sur les images :

```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# ou
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# ou
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

Variantes dérivées :

```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Variante sparses :

```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Alignement monolithique autonome :

```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

Lecteurs supplémentaires :

```bash
python read_events.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python read_frames.py --input yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4
python simple_count.py --input data/events.npy
```

## 🧩 Notes de configuration

Paramètres principaux du modèle dans `softalign/implicit_model.py` :

| Paramètre | Valeur par défaut |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

Le prétraitement des données dans `softalign/data_processing.py` applique actuellement un mauvais alignement synthétique aux événements avant l’entraînement. Cela est intentionnel dans le code actuel et doit être pris en compte lors de l’interprétation des métriques.

Les scripts d’entraînement exposent également des paramètres ajustables via des options CLI (par exemple `--hidden_dim`, `--num_layers` et `--batch_size`) utiles pour les ablations.

## 🧠 Formulation mathématique

Le réseau implicite partagé modélise `F(x, y, t)`.

Branche image :
- Prédit directement une réponse de type intensité via `F(x, y, t)`.

Branche événements :
1. Appliquer une transformation événementielle entraînable :
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. Approximer la dérivée temporelle :
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. Produire la réponse d’événement :
   - `sigmoid(dF/dt - threshold)`

L’objectif d’entraînement combine :
- Perte MSE des événements
- Perte MSE des images
- Régularisation des paramètres d’alignement

## 🧾 Exemples

Utiliser uniquement les tableaux prétraités :

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

Forcer l’exécution sur CPU :

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

Évaluer vers un dossier personnalisé :

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ Notes de développement

- Le dépôt est centré sur des scripts (pas encore de métadonnées de packaging).
- Les artefacts générés sont présents dans le dépôt (`checkpoints/results`), donc l’utilisation disque peut être importante.
- `readme-file.md` et `project-structure.txt` contiennent des hypothèses de structure héritées et ne sont pas totalement alignés avec la mise en page racine actuelle.
- Il n’existe actuellement ni tests automatisés ni workflows CI détectés.

## 🧯 Résolution des problèmes

- `ModuleNotFoundError: No module named 'implicit_model'` dans `evaluation.py` :
  - Lancer avec `PYTHONPATH=softalign` comme montré ci-dessus.
- Problèmes d’installation/exécution de `dv_processing` :
  - Vérifier que votre plateforme et votre version de Python sont supportées par les wheels/libs de `dv-processing`.
- OOM CUDA :
  - Réduire `--batch_size`, réduire `--max_events` (lorsqu’il est disponible), ou exécuter avec `--device cpu`.
- Fichier AEDAT4 manquant :
  - Vérifier le chemin sous `yuqing/` ou fournir votre propre enregistrement via `--filepath` / `--aedat4_file`.
- Incohérence de données entre exécutions :
  - Relancer avec `--reprocess` lors d’un changement d’hypothèses de prétraitement.

## 🗺️ Feuille de route

- Ajouter des fichiers d’environnement reproductibles (`requirements.txt` ou `pyproject.toml`).
- Unifier les chemins d’import pour l’exécution des modules d’évaluation et du package.
- Ajouter des tests automatisés pour la vérification du prétraitement et du forward/training du modèle.
- Ajouter une CI pour lint et smoke tests.
- Ajouter des README multilingues sous `i18n/`.

## 🤝 Contribution

Les contributions sont les bienvenues.

Flux de travail recommandé :
1. Créer une issue ou une branche ciblée résumant la modification expérimentale.
2. Garder les scripts alignés (notamment imports, conventions CLI, et noms de répertoires de sortie).
3. Préserver le comportement des artefacts d’expérience existants sauf si les noms/version changent.

## 📄 Licence

Ce dépôt est distribué sous licence Apache 2.0. Voir [`LICENSE`](LICENSE) pour le texte complet.

Hypothèse : si ce dépôt est utilisé dans des publications, les détails de citation doivent être ajoutés ici et dans les notes de version si nécessaire.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
