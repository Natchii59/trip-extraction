# Trip Extraction 🚀

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-online-success.svg)](https://natchii59.github.io/trip-extraction/)

> Système NLP pour extraire automatiquement les villes de départ et d'arrivée depuis des phrases en français.

**Architecture :** CamemBERT-NER + Classifieur fine-tuné | **Précision :** 95%+ | **Vitesse :** 0.1-0.5s/phrase

---

## 🎯 Quick Start

```bash
# Installation
git clone <repo-url>
cd bootstrap
python -m venv .venv
source .venv/bin/activate.fish  # ou activate pour bash
pip install -e .

# Entraînement (obligatoire première fois)
trip-train

# Demo
trip-demo
```

## 💡 Exemple d'utilisation

```python
from trip import TripParser

parser = TripParser()
departure, arrival = parser.parse_trip("Je vais de Paris à Lyon")
print(f"{departure} → {arrival}")  # Paris → Lyon
```

## ✨ Fonctionnalités

- ✅ Phrases simples : `"De Paris à Lyon"`
- ✅ Questions : `"Comment aller à Marseille depuis Toulouse ?"`
- ✅ Syntaxe inversée : `"À Lille depuis Paris"`
- ✅ Contexte temporel : `"Demain je vais de Nice à Cannes"`

## 📊 Performance

| Métrique | Score |
|----------|-------|
| Précision | 95-98% |
| Vitesse (CPU) | 0.3-0.5s |
| Vitesse (GPU) | 0.1-0.2s |

## 📚 Documentation complète

📖 **Documentation en ligne :** [https://natchii59.github.io/trip-extraction/](https://natchii59.github.io/trip-extraction/)

Ou consultez la documentation localement :

```bash
# Installer et lancer la documentation
pip install -e ".[docs]"
mkdocs serve
# Ouvrir http://127.0.0.1:8000/
```

### Pages disponibles

- **[Installation](https://natchii59.github.io/trip-extraction/installation/)** - Guide complet avec troubleshooting
- **[Utilisation](https://natchii59.github.io/trip-extraction/usage/)** - Exemples CLI, API, batch processing, cas d'usage
- **[Architecture](https://natchii59.github.io/trip-extraction/architecture/)** - Vue d'ensemble technique, composants, pipeline
- **[API Reference](https://natchii59.github.io/trip-extraction/api/)** - Documentation auto-générée avec mkdocstrings

## 🛠️ Développement

```bash
# Installation dev
pip install -e ".[dev]"

# Formatage et linting
black .
ruff check --fix .
mypy src/
```

**Outils inclus :** black, ruff, mypy, pytest, jupyter

## 🎓 Entraînement personnalisé

```bash
# Entraîner avec dataset custom
trip-train
```

**Format du dataset** (`data/training_dataset.json`) :

```json
[
  {"text": "Je vais de [LOC]Paris[/LOC] à Lyon", "label": 0},
  {"text": "Je vais de Paris à [LOC]Lyon[/LOC]", "label": 1}
]
```

Label 0 = départ, Label 1 = arrivée

## 🏗️ Architecture

```
Input → TripParser → NERExtractor (CamemBERT-NER) → Locations
                   → Classifier (CamemBERT fine-tuné) → Départ/Arrivée
```

**Composants :**
- `TripParser` : Orchestrateur principal
- `NERExtractor` : Extraction des villes (CamemBERT-NER)
- `DepartureArrivalClassifier` : Classification départ/arrivée
- `Config` : Configuration centralisée
- `Exceptions` : Gestion d'erreurs typées

## 🐛 Troubleshooting

### Modèle non trouvé
```bash
trip-train
```

### Module 'trip' non trouvé
```bash
pip install -e .
```

### Plus de détails
Consultez la section [Troubleshooting de la documentation](docs/installation.md#problemes-courants)
