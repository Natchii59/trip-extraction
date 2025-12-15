# Trip Information Extraction 🚀

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linter-ruff-blueviolet.svg)](https://github.com/astral-sh/ruff)

> **Système de traitement du langage naturel (NLP) pour extraire automatiquement les villes de départ et d'arrivée depuis des phrases en français.**

Utilise une architecture à deux modèles ML spécialisés :
- **CamemBERT-NER** pour l'extraction d'entités nommées
- **Classifieur custom fine-tuné** pour la classification départ/arrivée

---

## 📚 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Architecture](#-architecture)
- [Développement](#-développement)
- [Entraînement du modèle](#-entraînement-du-modèle)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Vue d'ensemble

### Problème résolu

Extraire automatiquement des informations structurées (départ/arrivée) depuis du texte non structuré en français :

```python
Input:  "Je veux aller de Paris à Lyon demain"
Output: ('Paris', 'Lyon')

Input:  "Train depuis Marseille vers Nice"
Output: ('Marseille', 'Nice')
```

### Performances

| Métrique | Valeur |
|----------|--------|
| **Accuracy (phrases simples)** | 95-98% |
| **Accuracy (phrases complexes)** | 85-92% |
| **Vitesse d'inférence** | 0.2-0.5s/phrase |
| **Support** | Questions, syntaxe inversée, contexte temporel |

### Cas d'usage supportés

✅ Phrases simples : "De Paris à Lyon"  
✅ Questions : "Comment aller à Marseille depuis Toulouse ?"  
✅ Syntaxe inversée : "À Lille depuis Paris"  
✅ Contexte temporel : "Demain je vais de Nice à Cannes"  
✅ Formulations variées : "Train/Vol/Trajet de A vers B"

---

## 🚀 Installation

### Prérequis

- **Python 3.11+** (testé sur 3.11 et 3.12)
- **~1GB d'espace disque** (modèles HuggingFace)
- **Connexion internet** (première utilisation uniquement)

### Installation rapide

```bash
# 1. Cloner le repository
git clone <repo-url>
cd bootstrap

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate.fish  # fish shell
# ou
source .venv/bin/activate        # bash/zsh

# 3. Installer les dépendances
pip install -e .

# 4. Entraîner le modèle (obligatoire la première fois)
trip-train

# 5. Tester l'installation
trip-demo
```

### Installation pour le développement

```bash
# Installer avec les dépendances de développement
pip install -e ".[dev]"

# Tester l'installation
trip-demo
```

---

## 💻 Utilisation

### Interface CLI interactive

```bash
# Lancer le demo interactif
trip-demo

# Exemple de session
✈️  Phrase > Je vais de Paris à Lyon
➡️  Résultat: Paris → Lyon

✈️  Phrase > quit
👋 Au revoir!
```

### Utilisation programmatique

```python
from trip import TripParser
from trip.exceptions import TripExtractionError

# Initialiser le parser
parser = TripParser()

# Extraire un trajet
try:
    departure, arrival = parser.parse_trip("Train de Paris à Lyon")
    print(f"Départ: {departure}, Arrivée: {arrival}")
    # Output: Départ: Paris, Arrivée: Lyon
except TripExtractionError as e:
    print(f"Erreur: {e}")
```

### API avancée

```python
from trip.models import NERExtractor, DepartureArrivalClassifier
from trip.config import get_config

# Configuration personnalisée
config = get_config()
config.model.confidence_threshold = 0.7

# Utiliser les modèles séparément
ner = NERExtractor()
locations = ner.extract_locations("Je vais de Paris à Lyon")
# Output: ['Paris', 'Lyon']

classifier = DepartureArrivalClassifier()
role, confidence = classifier.classify_location(
    "Je vais de Paris à Lyon", 
    "Paris"
)
# Output: ('departure', 0.98)
```

### Gestion d'erreurs

```python
from trip import TripParser
from trip.exceptions import (
    InvalidInputError,
    InsufficientLocationsError,
    ClassificationError,
    ModelNotFoundError
)

parser = TripParser()

try:
    result = parser.parse_trip(user_input)
except ModelNotFoundError:
    print("Modèle non trouvé. Lancez: trip-train")
except InvalidInputError as e:
    print(f"Entrée invalide: {e}")
except InsufficientLocationsError:
    print("Pas assez de villes détectées")
except ClassificationError:
    print("Impossible de classifier les villes")
```

---

## 🏗️ Architecture

### Vue d'ensemble

```
┌─────────────┐
│   Input     │  "Je vais de Paris à Lyon"
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│   TripParser                │
│  (Orchestrateur principal)  │
└──────┬──────────────┬───────┘
       │              │
       ▼              ▼
┌─────────────┐  ┌──────────────────────┐
│NERExtractor │  │DepartureArrival      │
│(CamemBERT)  │  │Classifier            │
│             │  │(CamemBERT fine-tuné) │
└──────┬──────┘  └──────────┬───────────┘
       │                    │
       │ ['Paris', 'Lyon']  │
       └────────┬───────────┘
                │
                ▼
         ('Paris', 'Lyon')
```

### Structure du projet

```
bootstrap/
├── src/trip/                    # Package principal
│   ├── __init__.py             # Exports publics
│   ├── config.py               # Configuration centralisée
│   ├── exceptions.py           # Exceptions métier
│   ├── trip_parser.py          # Orchestrateur
│   ├── utils.py                # Utilitaires
│   └── models/                 # Modèles ML
│       ├── __init__.py
│       ├── base.py             # Classes de base
│       ├── ner.py              # Extracteur NER
│       └── classifier.py       # Classifieur
│
├── scripts/                     # Scripts exécutables
│   ├── demo.py                 # Demo interactif
│   └── train.py                # Entraînement
│
├── data/                        # Données
│   └── training_dataset.json  # Dataset d'entraînement
│
├── models/                      # Modèles entraînés (généré)
│   └── departure_arrival_classifier/
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md         # Architecture détaillée
│   ├── MIGRATION.md            # Guide de migration
│   └── CHANGELOG.md            # Historique des versions
│
├── pyproject.toml              # Configuration du projet
└── README.md                   # Ce fichier
```

### Composants principaux

#### 1. TripParser (`src/trip/trip_parser.py`)

Orchestrateur principal qui coordonne les deux modèles.

```python
class TripParser:
    def parse_trip(self, text: str) -> tuple[Optional[str], Optional[str]]:
        # 1. Extraction NER
        locations = self.ner_extractor.extract_locations(text)
        
        # 2. Classification
        departure, arrival = self.classifier.classify_locations(text, locations)
        
        return (departure, arrival)
```

#### 2. NERExtractor (`src/trip/models/ner.py`)

Utilise CamemBERT-NER pour extraire les entités de type LOC (locations).

- Modèle pré-entraîné : `Jean-Baptiste/camembert-ner`
- Supporte les locations composées ("New York")
- Gère le split automatique des locations multiples

#### 3. DepartureArrivalClassifier (`src/trip/models/classifier.py`)

Classifieur fine-tuné sur des phrases de voyage françaises.

- Modèle de base : `camembert-base`
- Fine-tuné sur 480+ exemples
- Utilise des tokens spéciaux `[LOC]` et `[/LOC]`
- Seuil de confiance configurable

#### 4. Configuration (`src/trip/config.py`)

Configuration centralisée avec chemins absolus et paramètres.

```python
from trip.config import get_config

config = get_config()
print(config.paths.models_dir)           # Chemins
print(config.model.confidence_threshold) # Paramètres
```

#### 5. Exceptions (`src/trip/exceptions.py`)

Hiérarchie d'exceptions pour une gestion fine des erreurs.

```
TripExtractionError (base)
├── ModelNotFoundError
├── ModelLoadError
├── InsufficientLocationsError
├── InvalidInputError
├── ClassificationError
└── TokenizationError
```

---

## 🛠️ Développement

### Configuration de l'environnement de dev

```bash
# Installer avec les outils de dev
pip install -e ".[dev]"

# Les outils disponibles:
# - black: formatteur de code
# - ruff: linter
# - mypy: vérificateur de types
# - pytest: tests unitaires
```

### Formatage du code

```bash
# Formater tout le code
black .

# Vérifier sans modifier
black --check .
```

### Linting

```bash
# Vérifier le code
ruff check .

# Corriger automatiquement
ruff check --fix .
```

### Type checking

```bash
# Vérifier les types
mypy src/
```

### Structure des imports

```python
# ✅ Bon - Imports depuis le package
from trip import TripParser
from trip.models import NERExtractor, DepartureArrivalClassifier
from trip.config import get_config
from trip.exceptions import TripExtractionError

# ❌ Mauvais - Imports directs
from trip.models.ner import NERExtractor  # Éviter
```

### Ajout de nouvelles fonctionnalités

1. **Créer une branche**
   ```bash
   git checkout -b feature/ma-fonctionnalite
   ```

2. **Développer avec les bonnes pratiques**
   - Ajouter des docstrings à toutes les fonctions publiques
   - Inclure des type hints
   - Gérer les erreurs avec des exceptions spécifiques
   - Ajouter des tests unitaires (si disponibles)

3. **Formater et vérifier**
   ```bash
   black .
   ruff check --fix .
   mypy src/
   ```

4. **Commit et push**
   ```bash
   git add .
   git commit -m "feat: description de la fonctionnalité"
   git push origin feature/ma-fonctionnalite
   ```

---

## 🎓 Entraînement du modèle

### Quick start

```bash
# Entraîner avec les paramètres par défaut
trip-train
```

### Configuration de l'entraînement

Modifier `src/trip/config.py` :

```python
@dataclass
class TrainingConfig:
    num_epochs: int = 10          # Nombre d'epochs
    batch_size: int = 8           # Taille du batch
    learning_rate: float = 5e-5   # Learning rate
    max_length: int = 128         # Longueur max des séquences
```

### Format du dataset

Le dataset est dans `data/training_dataset.json` :

```json
[
    {
        "text": "Je veux aller de [LOC] Paris [/LOC] à Lyon",
        "label": 0
    },
    {
        "text": "Je veux aller de Paris à [LOC] Lyon [/LOC]",
        "label": 1
    }
]
```

- **Label 0** : départ
- **Label 1** : arrivée
- Les tokens `[LOC]` et `[/LOC]` marquent la ville à classifier

### Ajouter des exemples

1. Éditer `data/training_dataset.json`
2. Ajouter vos paires d'exemples (2 par phrase)
3. Réentraîner : `trip-train`

```json
[
    {
        "text": "Vol de [LOC] Toulouse [/LOC] à Bordeaux",
        "label": 0
    },
    {
        "text": "Vol de Toulouse à [LOC] Bordeaux [/LOC]",
        "label": 1
    }
]
```

### Monitoring de l'entraînement

```bash
# Logs détaillés pendant l'entraînement
2025-12-14 22:43:31 - scripts.train - INFO - Starting training...
2025-12-14 22:43:31 - scripts.train - INFO - Train set: 384 examples
2025-12-14 22:43:31 - scripts.train - INFO - Validation set: 96 examples
...
2025-12-14 22:45:12 - scripts.train - INFO - Final validation accuracy: 0.9583
```

### Résultats

Le modèle entraîné est sauvegardé dans :
```
models/departure_arrival_classifier/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── special_tokens_map.json
└── ...
```

---

## 📖 Documentation

### Documentation disponible

Ce projet dispose d'une documentation complète sous plusieurs formats :

#### 📄 Documentation statique

- **README.md** (ce fichier) : Guide de démarrage rapide
- **ARCHITECTURE.md** : Architecture détaillée du système
- **MIGRATION.md** : Guide de migration entre versions
- **CHANGELOG.md** : Historique des changements
- **Docstrings** : Documentation inline dans le code source

#### 🌐 Documentation interactive (MkDocs)

Une documentation interactive complète est disponible avec MkDocs Material :

```bash
# 1. Installer les dépendances de documentation
pip install -e ".[docs]"

# 2. Lancer le serveur de documentation
mkdocs serve

# 3. Ouvrir dans le navigateur
# http://127.0.0.1:8000
```

**Contenu de la documentation interactive :**

- **Guide de démarrage**
  - Installation détaillée avec toutes les options
  - Exemples d'utilisation (CLI et programmatique)
  
- **Architecture**
  - Vue d'ensemble du système avec diagrammes
  - Documentation détaillée de chaque composant
  
- **API Reference**
  - Documentation auto-générée depuis les docstrings
  - Exemples de code pour chaque fonction
  
- **Développement**
  - Guide de contribution
  - Standards de code
  - Bonnes pratiques
  
- **Entraînement**
  - Guide complet d'entraînement du modèle
  - Format du dataset
  - Optimisation des hyperparamètres

#### 📚 Builder la documentation

```bash
# Générer la documentation statique
mkdocs build

# La documentation sera dans le dossier site/
# Vous pouvez ensuite la déployer sur GitHub Pages, Netlify, etc.
```

### Lire la documentation

```bash
# Architecture du projet
cat ARCHITECTURE.md

# Guide de migration
cat MIGRATION.md

# Changelog
cat CHANGELOG.md
```

### Documentation du code

Toutes les fonctions publiques ont des docstrings complètes :

```python
from trip import TripParser

# Voir la documentation
help(TripParser)
help(TripParser.parse_trip)

# Dans IPython/Jupyter
TripParser.parse_trip?
```

### Générer la documentation (optionnel)

Si vous souhaitez une documentation HTML interactive :

```bash
# Installer mkdocs
pip install mkdocs mkdocs-material

# Servir la documentation localement
mkdocs serve

# Ouvrir http://127.0.0.1:8000
```

---

## 🧪 Vérification de l'installation

```bash
# Tester avec le demo interactif
trip-demo
```

Si le demo fonctionne correctement, l'installation est complète !

---

## 🤝 Contributing

### Pour les développeurs

1. **Forker le repository**
2. **Créer une branche feature**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Développer en suivant les standards**
   - Type hints partout
   - Docstrings pour les fonctions publiques
   - Gestion d'erreurs avec exceptions spécifiques
   - Code formaté avec `black`
   - Code vérifié avec `ruff`

4. **Commit avec conventional commits**
   ```bash
   git commit -m "feat: add new feature"
   git commit -m "fix: resolve bug"
   git commit -m "docs: update README"
   ```

5. **Push et créer une Pull Request**

### Standards de code

- **Formatage** : Black (line-length=100)
- **Linting** : Ruff
- **Type checking** : MyPy strict
- **Documentation** : Docstrings Google style
- **Commits** : Conventional commits

---

## 🐛 Troubleshooting

### Le modèle n'est pas trouvé

```
ModelNotFoundError: Model not found at 'models/departure_arrival_classifier'
```

**Solution** : Entraîner le modèle
```bash
trip-train
```

### Erreur d'import

```
ModuleNotFoundError: No module named 'trip'
```

**Solution** : Installer le package
```bash
pip install -e .
```

### Performance faible

Si l'accuracy est faible sur vos phrases :

1. Ajouter des exemples similaires dans `data/training_dataset.json`
2. Réentraîner : `trip-train`
3. Ajuster le seuil de confiance dans `config.py`

### Problème de device (CUDA/CPU)

Le code détecte automatiquement CUDA. Pour forcer CPU :

```python
from trip.config import get_config

config = get_config()
config.model.device = "cpu"
```

---

## 📊 Performances et benchmarks

### Temps d'exécution

| Opération | CPU | GPU (CUDA) |
|-----------|-----|------------|
| Chargement des modèles | ~2-3s | ~1-2s |
| Inférence (1 phrase) | ~0.3-0.5s | ~0.1-0.2s |
| Entraînement (10 epochs) | ~10-12min | ~2-3min |

### Utilisation mémoire

- **RAM** : ~500MB (modèles chargés)
- **VRAM** (GPU) : ~1GB
- **Disque** : ~1GB (modèles)

---

## 📝 License

MIT License - voir le fichier LICENSE pour les détails.

---

## 🙏 Remerciements

- **Hugging Face** pour CamemBERT et Transformers
- **Jean-Baptiste** pour le modèle CamemBERT-NER
- La communauté Python pour les outils de dev (black, ruff, mypy)

---

## 📧 Contact & Support

Pour toute question ou problème :

1. **Issues** : Ouvrir une issue sur GitHub
2. **Documentation** : Consulter ARCHITECTURE.md
3. **Code** : Les docstrings dans le code source

---

**Version** : 0.3.0  
**Dernière mise à jour** : Décembre 2025  
**Python** : 3.11+
