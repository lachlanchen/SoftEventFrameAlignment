[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Soft Event-Frame Alignment

*Code de recherche minimal pour apprendre l’alignement spatial et temporel entre des échantillons de caméra événementielle et de caméra à images avec une représentation neuronale implicite commune.*

[![CI](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/SoftEventFrameAlignment/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA043)](../LICENSE)
[![Sponsor](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)

Voici la première implémentation publique et clonable de **Soft Alignment of Event and Frame Data with Unified Implicit Neural Representation**. Elle comprend le modèle canonique, le prétraitement AEDAT4, l’entraînement, l’évaluation, les métadonnées de dépendances, la CI et un test CPU synthétique déterministe.

Aucun enregistrement, aucune image identifiable, aucun tableau dérivé ni checkpoint entraîné n’est publié. Utilisez uniquement des données que vous êtes autorisé à traiter.

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=kofi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## Architecture

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

La branche événementielle transforme les coordonnées et approxime la dérivée temporelle d’un MLP partagé. La branche image évalue directement le même champ. L’échelle et le pas de dérivation sont paramétrés positivement afin d’éviter toute division par zéro.

## Contenu de la version publique

| Chemin | Rôle |
| --- | --- |
| `softalign/implicit_model.py` | MLP implicite partagé et paramètres d’alignement |
| `softalign/training.py` | Adaptateur de données validé et entraînement conjoint |
| `softalign/data_processing.py` | Lecture AEDAT4 facultative et prétraitement en secondes |
| `softalign/synthetic.py` | Données synthétiques déterministes produites par le projet |
| `main.py` / `evaluation.py` | Interfaces d’entraînement et de diagnostic |
| `examples/` / `tests/` | Vérification de bout en bout et test du cœur |

Les expériences historiques, captures brutes, données locales et anciens checkpoints sont volontairement exclus.

## Installation et vérification

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,viz]'
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py --output-dir .smoke-output --epochs 8
```

Le test produit un checkpoint, des graphiques et `evaluation_results.json`. Il vérifie l’exécution et des valeurs finies, mais ne démontre pas la précision de l’alignement.

## Utiliser votre propre enregistrement AEDAT4

```bash
python -m pip install -e '.[aedat,viz]'
python main.py --filepath /path/to/your-recording.aedat4 --reprocess --data_dir data --checkpoint_dir checkpoints
python evaluation.py --model_path checkpoints/model_final.pt --data_dir data --output_dir evaluation --device cpu
```

Le prétraitement enregistre le schéma `softalign-processed/v1` et convertit les horodatages AEDAT des microsecondes en secondes. La perturbation synthétique est désactivée par défaut et s’active uniquement avec `--synthetic-misalignment`. Les anciens tableaux sans métadonnées d’unités compatibles sont refusés.

Les grandes collections restent en mémoire CPU ; seuls les lots échantillonnés sont transférés vers le périphérique choisi. Commencez avec `--device cpu`, puis utilisez CUDA après contrôle de la mémoire.

## Contrat de données

| Fichier | Signification |
| --- | --- |
| `events.npy` | `N × 4` : x/y normalisés, temps en secondes, cible événementielle |
| `frame_points.npy` | `M × 4` : x/y normalisés, temps en secondes, intensité |
| `preprocessing.json` | Schéma, unités, graine, indicateur de transformation et tailles |
| `model_final.pt` | État, paramètres et architecture du modèle |
| `evaluation_results.json` | MSE de reconstruction sur des échantillons d’entrée bornés |

Ces résultats diagnostiquent les tableaux fournis ; ce ne sont ni des métriques indépendantes, ni un benchmark, ni une preuve de calibration physique.

## Validation et limites de recherche

- La CI exécute un test CPU déterministe et un court cycle entraînement/évaluation.
- La forme, le contenu vide, les valeurs finies, les unités temporelles et le périphérique sont contrôlés.
- La fonction de perte actuelle est une base de recherche ; une étude réelle exige une vérité terrain, une évaluation séparée, des ablations et une analyse d’incertitude.
- N’ajoutez pas d’enregistrements privés aux issues GitHub publiques.

## Citation

Si vous utilisez ce projet en recherche, citez le dépôt. GitHub lit [CITATION.cff](../CITATION.cff) et affiche **Cite this repository**.

```bibtex
@software{chen_soft_event_frame_alignment_2026,
  author = {Chen, Lachlan},
  title = {Soft Event-Frame Alignment: Unified Implicit Neural Representation Research Code},
  year = {2026},
  url = {https://github.com/lachlanchen/SoftEventFrameAlignment}
}
```

## État

La version `0.1.0` est une publication de recherche minimale. Les erreurs reproductibles avec le test synthétique ou un petit échantillon autorisé sont bienvenues.

## Licence

Apache License 2.0. Voir [LICENSE](../LICENSE).
