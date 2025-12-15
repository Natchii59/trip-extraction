# Trip Extraction

Système d'extraction automatique de trajets à partir de phrases en français utilisant le NLP et les transformers.

## 🎯 Objectif du projet

Trip Extraction est un système de traitement du langage naturel (NLP) conçu pour **extraire automatiquement les informations de voyage** (villes de départ et d'arrivée) depuis des phrases en français naturel. Le système combine deux modèles de deep learning pour obtenir une précision de 95%+ :

1. **CamemBERT-NER** : Détection des entités nommées (villes)
2. **Classifier personnalisé** : Classification départ vs arrivée

## ✨ Fonctionnalités

### Extraction intelligente
- **Reconnaissance d'entités nommées (NER)** : Détection automatique des villes avec CamemBERT
- **Classification contextuelle** : Identification précise du rôle (départ/arrivée) de chaque ville
- **Support multi-syntaxe** : Gère les questions, syntaxe inversée, contexte temporel

### Interface complète
- **CLI interactif** : Demo en ligne de commande avec retour visuel
- **API Python** : Intégration simple dans vos projets
- **Gestion d'erreurs** : Exceptions typées pour un debugging facile

### Performance
- **Haute précision** : 95%+ sur le dataset de test
- **Rapide** : 0.1-0.5s par phrase selon le hardware
- **Optimisé** : Support CPU et GPU (CUDA)

## 🚀 Installation rapide

/// tab | Fish Shell

```bash
# Cloner et installer
git clone <repo-url>
cd bootstrap
python -m venv .venv
source .venv/bin/activate.fish
pip install -e .

# Entraîner le modèle (obligatoire première fois)
trip-train

# Lancer le demo
trip-demo
```

///

/// tab | Bash/Zsh

```bash
# Cloner et installer
git clone <repo-url>
cd bootstrap
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Entraîner le modèle (obligatoire première fois)
trip-train

# Lancer le demo
trip-demo
```

///

## 💡 Exemple d'utilisation

/// codexec

    :::python
    from trip import TripParser
    
    # Initialiser le parser
    parser = TripParser()
    
    # Extraire un trajet
    departure, arrival = parser.parse_trip("Je vais de Paris à Lyon")
    
    print(f"Départ: {departure}")
    print(f"Arrivée: {arrival}")

///

### Exemples de phrases supportées

Le système gère une grande variété de formulations :

```python
# Syntaxe simple
"De Paris à Lyon" → Paris → Lyon
"Paris Lyon" → Paris → Lyon

# Questions
"Comment aller à Marseille depuis Toulouse ?" → Toulouse → Marseille
"Où prendre le train pour Nice ?" → <ville actuelle> → Nice

# Syntaxe inversée
"À Lille depuis Paris" → Paris → Lille
"Vers Lyon de Paris" → Paris → Lyon

# Avec contexte temporel
"Demain je vais de Nice à Cannes" → Nice → Cannes
"Train de 8h de Paris à Lyon" → Paris → Lyon
```

## 📊 Performance

| Composant | Métrique | Score |
|-----------|----------|-------|
| NER Extractor | Precision | 95% |
| NER Extractor | Recall | 93% |
| NER Extractor | F1-Score | 94% |
| Classifier | Accuracy | 96% |
| Classifier | Precision | 97% |
| Classifier | F1-Score | 98% |

### Temps d'exécution

| Device | Temps par phrase |
|--------|------------------|
| **CPU** | 0.3-0.5s |
| **GPU (CUDA)** | 0.1-0.2s |

## 🏗️ Architecture

```
Input: "Je vais de Paris à Lyon"
    ↓
┌─────────────────────┐
│   NER Extractor     │  → Détecte: ["Paris", "Lyon"]
│  (CamemBERT-NER)    │
└─────────────────────┘
    ↓
┌─────────────────────┐
│    Classifier       │  → Paris: departure (98%)
│ (CamemBERT custom)  │  → Lyon: arrival (97%)
└─────────────────────┘
    ↓
Output: (Paris, Lyon)
```

**Composants principaux :**
- **TripParser** : Orchestration et validation
- **NERExtractor** : Extraction des villes avec CamemBERT-NER
- **DepartureArrivalClassifier** : Classification avec CamemBERT fine-tuné
- **Exceptions** : Gestion d'erreurs typées

## 📚 Navigation

/// details | Installation complète

[Guide d'installation détaillé](installation.md) avec :

- Prérequis système
- Installation standard et développement
- Configuration GPU/CUDA
- Troubleshooting

///

/// details | Guide d'utilisation

[Exemples et API](usage.md) avec :

- Interface CLI
- API Python avec exemples codexec
- Batch processing
- Configuration avancée

///

/// details | Architecture technique

[Vue d'ensemble architecture](architecture.md) avec :

- Description des composants
- Pipeline de traitement
- Format du dataset
- Performance détaillée

///

/// details | Référence API

[Documentation API complète](api.md) avec :

- API auto-générée via mkdocstrings
- Classes et méthodes documentées
- Signatures de types
- Exemples interactifs

///

## 🎓 Cas d'usage

Trip Extraction peut être utilisé pour :

- **Chatbots de voyage** : Extraction automatique de trajets depuis messages utilisateurs
- **Systèmes de réservation** : Parsing de requêtes en langage naturel
- **Analyse de données** : Extraction de trajets depuis corpus de textes
- **Assistants virtuels** : Compréhension d'intentions de voyage
- **Applications mobiles** : Interface vocale pour recherche de trajets

## 🔧 Développement

Pour contribuer au projet :

```bash
# Installation avec outils de dev
pip install -e ".[dev]"

# Formattage et linting
black src/ scripts/
ruff check src/ scripts/
mypy src/
```

Outils inclus : `black`, `ruff`, `mypy`, `pytest`, `jupyter`
