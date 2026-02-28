[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Soft Event-Frame Alignment

> Dépôt du papier **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Research Code](https://img.shields.io/badge/Type-Research%20Code-0A7EA4)
![License](https://img.shields.io/badge/License-Apache%202.0-2EA043)

## 🔎 Vue d'ensemble

Ce dépôt contient du code de recherche pour aligner des flux de caméras événementielles et des flux de caméras à images avec une représentation neuronale implicite unifiée (INR/INN).

### Pipeline principal (chemin canonique)

| Étape | Composant | Objectif |
|---|---|---|
| 1 | `softalign/data_processing.py` | Lire les flux AEDAT4, normaliser les données, créer des échantillons de points |
| 2 | `softalign/implicit_model.py` | Définir la fonction implicite partagée + les paramètres d'alignement apprenables |
| 3 | `softalign/training.py` + `main.py` | Optimiser conjointement la reconstruction événements/images |
| 4 | `evaluation.py` | Visualiser l'alignement et rapporter les pertes |

En plus du chemin principal, le dépôt inclut des scripts autonomes et expérimentaux pour des variantes INR événement uniquement, image uniquement, sparse et basées sur des dérivées.

## ✨ Fonctionnalités

- Alignement événement-image avec paramètres de type affine apprenables : `scale`, `shift_x`, `shift_y`, `shift_t`.
- Fonction implicite partagée `F(x, y, t)` utilisée par les branches événement et image.
- Modélisation des événements via dérivée temporelle en différences finies avec `threshold` et `dt` apprenables.
- Ingestion AEDAT4 (`dv_processing`) et workflows `.npy` prétraités.
- Visualisations intégrées pour les courbes de perte, l'évolution des paramètres et les superpositions d'alignement.
- Support CUDA si disponible (repli vers CPU via `torch.cuda.is_available()`).

## 🗂️ Structure du projet

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

## 📋 Prérequis

- Python 3.10+ recommandé.
- Les exemples ci-dessous supposent un shell Linux/macOS (à adapter pour Windows si nécessaire).
- Optionnel mais recommandé : GPU compatible CUDA pour accélérer l'entraînement.
- La lecture AEDAT4 nécessite `dv_processing` et des dépendances système compatibles.

## ⚙️ Installation

Aucun `requirements.txt` ni `pyproject.toml` n'est présent actuellement ; installez donc les dépendances manuellement.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python tqdm torch
pip install dv-processing
```

Dépendances optionnelles pour les expériences sparse (`event_inn_sparse.py`) :

```bash
pip install scikit-learn torch-geometric torch-scatter
```

## 🚀 Démarrage rapide

### 1. Entraîner le modèle principal d'alignement

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

### 2. Évaluer un checkpoint entraîné

`evaluation.py` importe `EventFrameAlignmentModel` via `from implicit_model import ...`, alors que le module actif se trouve dans `softalign/implicit_model.py`. Une invocation pratique depuis la racine du dépôt est :

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

Options clés :
- `--filepath` : chemin d'entrée AEDAT4.
- `--duration` : secondes à lire depuis l'enregistrement.
- `--data_dir` : répertoire de sortie des données prétraitées.
- `--checkpoint_dir` : répertoire de sortie des checkpoints.
- `--num_epochs`, `--lr`, `--batch_size`, `--lambda_reg`.
- `--device` : `cuda` ou `cpu`.
- `--reprocess` : force la régénération des données.

### Scripts expérimentaux

INR événement uniquement :
```bash
python event_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_event
# or
python event_inn.py --events_file data/events.npy --output_dir output_event
```

INR image uniquement :
```bash
python frame_inn.py --aedat4_file yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10 --output_dir output_frame --grayscale
# or
python frame_inn.py --video_file path/to/video.mp4 --output_dir output_frame
# or
python frame_inn.py --image_dir path/to/images --image_pattern "*.png" --output_dir output_frame
```

Variantes à dérivée :
```bash
python event_deri.py --events_file data/events.npy --output_dir output_deri
python event_deri_2.py --events_file data/events.npy --output_dir output_deri2
```

Variante sparse :
```bash
python event_inn_sparse.py --events_file data/events.npy --output_dir output_sparse
```

Alignement monolithique autonome :
```bash
python softalign_standalone.py --filepath yuqing/0.419/dvSave-2025_02_25_01_22_55.aedat4 --duration 10
```

## 🧩 Notes de configuration

Valeurs par défaut du modèle principal dans `softalign/implicit_model.py` :

| Paramètre | Valeur par défaut |
|---|---|
| `scale` | `0.8` |
| `shift_x` | `0.05` |
| `shift_y` | `0.08` |
| `shift_t` | `1.0` |
| `threshold` | `0.0` |
| `dt` | `0.1` |

Le prétraitement des données dans `softalign/data_processing.py` applique actuellement un désalignement synthétique aux événements avant entraînement. C'est intentionnel dans le code actuel et doit être pris en compte lors de l'interprétation des métriques.

## 🧠 Formulation mathématique

Le réseau implicite partagé modélise `F(x, y, t)`.

Branche image :
- Prédit directement une réponse de type intensité avec `F(x, y, t)`.

Branche événement :
1. Appliquer une transformation événement apprenable :
   - `x' = x / scale + shift_x`
   - `y' = y / scale + shift_y`
   - `t' = t + shift_t`
2. Approximer la dérivée temporelle :
   - `dF/dt ≈ (F(x', y', t' + dt) - F(x', y', t')) / dt`
3. Produire la réponse événement :
   - `sigmoid(dF/dt - threshold)`

L'objectif d'entraînement combine :
- Perte MSE événements
- Perte MSE images
- Régularisation des paramètres d'alignement

## 🧾 Exemples

Utiliser uniquement des tableaux prétraités :

```bash
python main.py --data_dir data --checkpoint_dir checkpoints --num_epochs 200 --lr 5e-4
```

Forcer une exécution CPU :

```bash
python main.py --device cpu --num_epochs 50 --reprocess
```

Évaluer dans un dossier personnalisé :

```bash
PYTHONPATH=softalign python evaluation.py --output_dir evaluation_debug
```

## 🛠️ Notes de développement

- Le dépôt est centré sur des scripts (pas encore de métadonnées de packaging).
- Des artefacts générés sont présents dans le dépôt (checkpoints/résultats), donc l'utilisation disque peut être importante.
- `readme-file.md` et `project-structure.txt` contiennent des hypothèses de structure héritées et ne sont pas totalement alignés avec la structure racine actuelle.
- Aucun test automatisé ni workflow CI n'est actuellement détecté.

## 🧯 Dépannage

- `ModuleNotFoundError: No module named 'implicit_model'` dans `evaluation.py` :
  - Exécutez avec `PYTHONPATH=softalign` comme indiqué ci-dessus.
- Problèmes d'installation/exécution de `dv_processing` :
  - Vérifiez que votre plateforme et votre version de Python sont prises en charge par les wheels/libs `dv-processing`.
- OOM CUDA :
  - Réduisez `--batch_size`, réduisez `--max_events` (si disponible), ou exécutez avec `--device cpu`.
- Fichier AEDAT4 manquant :
  - Vérifiez le chemin sous `yuqing/` ou fournissez votre propre enregistrement via `--filepath` / `--aedat4_file`.

## 🗺️ Feuille de route

- Ajouter des fichiers d'environnement reproductibles (`requirements.txt` ou `pyproject.toml`).
- Unifier les chemins d'import pour l'évaluation et l'exécution package/module.
- Ajouter des tests automatisés pour le prétraitement des données et les vérifications forward/entraînement du modèle.
- Ajouter une CI pour le linting et les smoke tests.
- Ajouter des README multilingues sous `i18n/`.

## 🤝 Contribution

Les contributions sont les bienvenues.

1. Forkez le dépôt.
2. Créez une branche de fonctionnalité.
3. Effectuez des modifications ciblées et bien documentées.
4. Ouvrez une pull request avec des détails de reproduction et des sorties d'exemple lorsque c'est pertinent.

Si vous prévoyez des changements importants (nouveau chemin d'entraînement, refactorisation ou changements de dépendances), ouvrez d'abord une issue pour aligner la direction.

## 📚 Citation

Si vous utilisez ce dépôt dans des travaux de recherche, citez le papier associé et/ou le dépôt.

Métadonnées actuelles du dépôt disponibles à partir du texte README existant :
- Titre du papier : *Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation*

Placeholder BibTeX du dépôt (mettez à jour auteurs/URL selon vos besoins) :

```bibtex
@misc{soft-event-frame-alignment,
  title        = {Soft Event-Frame Alignment},
  note         = {Code repository for "Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation"},
  year         = {2025},
  howpublished = {GitHub repository}
}
```

## 🙏 Remerciements

- Écosystème d'outils pour caméras événementielles, dont `dv-processing`.
- PyTorch et les bibliothèques Python scientifiques utilisées dans les expériences.

## 💡 Support

La section support/sponsoring n'est pas présente dans les métadonnées actuelles du dépôt. Si vous en souhaitez une, ajoutez des liens lors d'une mise à jour ultérieure et ils seront conservés dans les révisions futures.

## 📄 Licence

Distribué sous licence Apache 2.0. Voir [LICENSE](../LICENSE).
