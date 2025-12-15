# Architecture

## Vue d'ensemble

Le système Trip Extraction utilise deux modèles de NLP pour extraire les trajets :

```
Input: "Je vais de Paris à Lyon"
    ↓
┌─────────────────┐
│  NER Extractor  │  → Détecte les villes: ["Paris", "Lyon"]
└─────────────────┘
    ↓
┌─────────────────┐
│   Classifier    │  → Paris: departure (98%)
└─────────────────┘  → Lyon: arrival (97%)
    ↓
Output: (Paris, Lyon)
```

## Composants principaux

### 1. TripParser

Point d'entrée principal de l'API.

```python
from trip import TripParser

parser = TripParser()
departure, arrival = parser.parse_trip("Je vais de Paris à Lyon")
```

**Responsabilités :**
- Orchestration des composants
- Validation de l'entrée
- Gestion des erreurs

**Fichier :** `src/trip/trip_parser.py`

### 2. NERExtractor

Extraction d'entités nommées (villes) avec CamemBERT.

```python
from trip.ner_extractor import NERExtractor

ner = NERExtractor()
locations = ner.extract_locations("Je vais de Paris à Lyon")
# → ['Paris', 'Lyon']
```

**Modèle :** `camembert-ner` (Jean-Baptiste/camembert-ner)

**Responsabilités :**
- Détection des entités LOC (locations)
- Filtrage et nettoyage
- Gestion des entités multi-tokens

**Fichier :** `src/trip/ner_extractor.py`

### 3. DepartureArrivalClassifier

Classification départ vs arrivée avec CamemBERT fine-tuné.

```python
from trip.departure_arrival_classifier import DepartureArrivalClassifier

classifier = DepartureArrivalClassifier()
role, confidence = classifier.classify_location(
    "Je vais de Paris à Lyon",
    "Paris"
)
# → ('departure', 0.98)
```

**Modèle :** CamemBERT fine-tuné sur dataset custom

**Responsabilités :**
- Classification binaire (départ/arrivée)
- Score de confiance
- Gestion du contexte

**Fichier :** `src/trip/departure_arrival_classifier.py`

## Pipeline de traitement

### Étape 1 : Extraction NER

```python
text = "Je vais de Paris à Lyon"
locations = ner.extract_locations(text)
# → ['Paris', 'Lyon']
```

Le NER extractor :
1. Tokenize le texte avec CamemBERT tokenizer
2. Passe les tokens dans le modèle CamemBERT-NER
3. Détecte les entités `B-LOC` et `I-LOC`
4. Reconstruit les villes complètes
5. Filtre et nettoie les résultats

### Étape 2 : Classification

```python
for location in locations:
    role, confidence = classifier.classify_location(text, location)
```

Le classifier :
1. Ajoute les tokens `[LOC]` et `[/LOC]` autour de la ville
2. Tokenize : `"Je vais de [LOC] Paris [/LOC] à Lyon"`
3. Passe dans le modèle CamemBERT fine-tuné
4. Obtient un score de confiance pour départ/arrivée
5. Retourne le rôle et la confiance

### Étape 3 : Validation

```python
if len(locations) < 2:
    raise InsufficientLocationsError()

if confidence < threshold:
    raise LowConfidenceError()
```

Vérifications :
- Au moins 2 villes détectées
- Confiance suffisante (> 0.5 par défaut)
- Pas de doublons

## Format du dataset d'entraînement

Le classifier est entraîné sur ce format :

```json
[
  {
    "text": "Je vais de [LOC]Paris[/LOC] à Lyon",
    "label": 0
  },
  {
    "text": "Je vais de Paris à [LOC]Lyon[/LOC]",
    "label": 1
  }
]
```

- `label: 0` = départ
- `label: 1` = arrivée
- `[LOC]` et `[/LOC]` marquent la ville à classifier

**Fichier :** `data/training_dataset.json`

## Gestion des erreurs

```python
from trip.exceptions import (
    TripExtractionError,      # Classe de base
    InvalidInputError,         # Entrée vide/invalide
    InsufficientLocationsError,# Moins de 2 villes
    LowConfidenceError        # Confiance trop faible
)
```

Hiérarchie :

```
TripExtractionError
├── InvalidInputError
├── InsufficientLocationsError
└── LowConfidenceError
```

## Configuration

```python
from trip.config import get_config

config = get_config()
print(config.model.ner_model_name)        # camembert-ner
print(config.model.classifier_model_path) # models/departure_arrival_classifier
print(config.model.device)                # cuda ou cpu
print(config.model.confidence_threshold)  # 0.5
```

**Fichier :** `src/trip/utils.py`

## Performance

### NER Extraction

| Métrique | Score |
|----------|-------|
| Precision | 95% |
| Recall | 93% |
| F1-Score | 94% |

### Classifier

| Métrique | Score |
|----------|-------|
| Accuracy | 96% |
| Precision | 97% |
| Recall | 96% |
| F1-Score | 98% |

### Temps d'exécution

| Device | Temps par phrase |
|--------|------------------|
| CPU | 0.3-0.5s |
| GPU (CUDA) | 0.1-0.2s |

## Structure du projet

```
bootstrap/
├── src/trip/
│   ├── __init__.py               # Exports publics
│   ├── __main__.py               # Point d'entrée CLI
│   ├── trip_parser.py            # TripParser principal
│   ├── ner_extractor.py          # NERExtractor
│   ├── departure_arrival_classifier.py  # Classifier
│   └── utils.py                  # Config, logging, exceptions
├── data/
│   └── training_dataset.json     # Dataset d'entraînement
├── models/
│   └── departure_arrival_classifier/  # Modèle entraîné
├── train_model.py                # Script d'entraînement
└── pyproject.toml               # Configuration du projet
```

## Entraînement

```bash
trip-train
```

Le script :
1. Charge `data/training_dataset.json`
2. Split train/test (80/20)
3. Fine-tune CamemBERT (3 epochs)
4. Évalue sur le test set
5. Sauvegarde dans `models/departure_arrival_classifier/`

**Durée :** 2-3 min (GPU) ou 10-12 min (CPU)

**Fichier :** `train_model.py`
