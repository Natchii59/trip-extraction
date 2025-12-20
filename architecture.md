# Architecture

Ce document décrit l'architecture complète du projet Trip Extraction, sa structure de code, les design patterns utilisés et le pipeline de traitement.

## 📁 Structure du projet

### Vue d'ensemble

```
bootstrap/
├── src/
│   ├── trip_parser/             # Module d'extraction ML
│   │   ├── trip_parser.py       # Orchestrateur principal
│   │   ├── config.py            # Configuration centralisée
│   │   ├── exceptions.py        # Exceptions métier
│   │   ├── utils.py             # Utilitaires (logging, etc)
│   │   └── models/              # Modèles ML
│   │       ├── base.py          # Classe de base abstraite
│   │       ├── ner.py           # NER Extractor (CamemBERT)
│   │       └── classifier.py    # Classifier départ/arrivée
│   │
│   └── api/                     # API REST FastAPI
│       ├── main.py              # Application FastAPI
│       ├── routers/             # Routes HTTP
│       │   └── trip.py          # Routes /trip/*
│       ├── schemas/             # Modèles Pydantic
│       │   └── trip.py          # Schémas request/response
│       └── services/            # Logique métier API
│           └── trip_service.py  # Service singleton
│
├── scripts/                     # Scripts d'entrée
│   ├── demo.py                  # Interface CLI interactive
│   ├── train.py                 # Entraînement du classifier
│   └── run_api.py               # Lanceur de l'API
│
├── models/                      # Modèles ML entraînés
│   └── departure_arrival_classifier/
│       └── ...
│
├── data/                        # Données d'entraînement
│   └── training_dataset.json
│
├── docs/                        # Documentation MkDocs
│   └── xxx.md
│
├── logs/                        # Fichiers de logs
│
├── pyproject.toml               # Configuration Python/pip
├── mkdocs.yml                   # Configuration documentation
└── README.md                    # Quick start
```

### Organisation des responsabilités

| Dossier | Responsabilité | Dépendances |
|---------|----------------|-------------|
| **`src/trip_parser/`** | Logique d'extraction ML | transformers, torch, sentencepiece |
| **`src/api/`** | Exposition REST | fastapi, uvicorn, pydantic |
| **`scripts/`** | Points d'entrée CLI | trip_parser, api |
| **`models/`** | Modèles entraînés | Généré par `trip-train` |
| **`data/`** | Datasets | Fourni manuellement |
| **`docs/`** | Documentation | mkdocs, shadcn |

## 🏗️ Pipeline de traitement

### Vue détaillée du flux

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT : Texte français                  │
│              "Je veux aller de Paris à Lyon"                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 1. Validation
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TripParser.parse_trip()                      │
│  ────────────────────────────────────────────────────────────── │
│  • Vérifie que text est non vide                                │
│  • Limite la longueur (max 1000 caractères)                     │
│  • Log l'entrée pour debugging                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 2. Extraction des entités
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   NERExtractor.extract_locations()              │
│  ────────────────────────────────────────────────────────────── │
│  MODÈLE : Jean-Baptiste/camembert-ner (Hugging Face)            │
│  INPUT  : "Je veux aller de Paris à Lyon"                      │
│                                                                 │
│  ÉTAPES :                                                       │
│  1. Tokenisation avec CamembertTokenizer                        │
│     → ["Je", "veux", "aller", "de", "Paris", "à", "Lyon"]      │
│                                                                 │
│  2. NER Pipeline (detection d'entités)                          │
│     → [                                                         │
│         {"entity": "LOC", "word": "Paris", "score": 0.99},      │
│         {"entity": "LOC", "word": "Lyon", "score": 0.98}        │
│       ]                                                         │
│                                                                 │
│  3. Filtrage (garder uniquement type LOC)                       │
│     → ["Paris", "Lyon"]                                         │
│                                                                 │
│  4. Split des locations composées                               │
│     Ex: "Paris Marseille" → ["Paris", "Marseille"]             │
│                                                                 │
│  OUTPUT : ["Paris", "Lyon"]                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 3. Vérification nombre de villes
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Validation du nombre                         │
│  ────────────────────────────────────────────────────────────── │
│  if len(cities) < 2:                                            │
│      return (None, None)  # Pas assez de villes                 │
│                                                                 │
│  if len(cities) == 2:                                           │
│      continue  # Cas simple                                     │
│                                                                 │
│  if len(cities) > 2:                                            │
│      # Le classifier déterminera lesquelles sont départ/arrivée │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 4. Classification
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         DepartureArrivalClassifier.classify_locations()         │
│  ────────────────────────────────────────────────────────────── │
│  MODÈLE : CamemBERT fine-tuné (models/departure_arrival_...)    │
│  INPUT  : text="Je veux aller de Paris à Lyon"                 │
│           cities=["Paris", "Lyon"]                              │
│                                                                 │
│  ÉTAPES : Pour chaque ville                                     │
│                                                                 │
│  ┌───────────── Ville: "Paris" ────────────────┐               │
│  │  1. Marquer la ville dans le texte          │               │
│  │     "Je veux aller de [LOC] à Lyon"         │               │
│  │                                              │               │
│  │  2. Tokeniser avec CamembertTokenizer        │               │
│  │     → input_ids, attention_mask              │               │
│  │                                              │               │
│  │  3. Forward pass dans le modèle              │               │
│  │     logits = model(**inputs)                 │               │
│  │     → [4.2, -3.8]  (0=departure, 1=arrival) │               │
│  │                                              │               │
│  │  4. Softmax pour probabilités                │               │
│  │     → [0.98, 0.02]                           │               │
│  │                                              │               │
│  │  5. Classification                           │               │
│  │     argmax(logits) = 0                       │               │
│  │     role = "departure"                       │               │
│  │     confidence = 0.98                        │               │
│  └──────────────────────────────────────────────┘               │
│                                                                 │
│  ┌───────────── Ville: "Lyon" ──────────────────┐              │
│  │  1. "Je veux aller de Paris à [LOC]"         │              │
│  │  2. Tokenize                                 │              │
│  │  3. Forward → [-3.5, 4.1]                    │              │
│  │  4. Softmax → [0.01, 0.99]                   │              │
│  │  5. role = "arrival", confidence = 0.99      │              │
│  └──────────────────────────────────────────────┘              │
│                                                                 │
│  RÉSULTAT :                                                     │
│    departure_candidates = [("Paris", 0.98)]                     │
│    arrival_candidates = [("Lyon", 0.99)]                        │
│                                                                 │
│  OUTPUT : ("Paris", "Lyon")                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 5. Retour final
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                          RESULT                                 │
│                  ("Paris", "Lyon")                              │
└─────────────────────────────────────────────────────────────────┘
```

### Gestion des cas limites

/// tab | Cas 1 : Moins de 2 villes
```python
# Input
text = "Je veux aller à Paris"

# Étape 1 : NER détecte ["Paris"]
# Étape 2 : len(cities) < 2 → STOP
# Output : (None, None)
```
///

/// tab | Cas 2 : Plus de 2 villes
```python
# Input
text = "De Paris à Lyon puis Marseille"

# Étape 1 : NER détecte ["Paris", "Lyon", "Marseille"]
# Étape 2 : Classifier évalue chaque ville
#   Paris: departure (0.95)
#   Lyon: arrival (0.60)
#   Marseille: arrival (0.85)
# Étape 3 : Sélection du meilleur départ et arrivée
#   Best departure: Paris (0.95)
#   Best arrival: Marseille (0.85)
# Output : ("Paris", "Marseille")
```
///

/// tab | Cas 3 : Ambiguïté
```python
# Input
text = "Paris Lyon"  # Pas de préposition

# Étape 1 : NER détecte ["Paris", "Lyon"]
# Étape 2 : Classifier essaie de deviner
#   "[LOC] Lyon" → Paris: departure (0.55)
#   "Paris [LOC]" → Lyon: arrival (0.60)
# Étape 3 : Confiance faible mais utilisable
# Output : ("Paris", "Lyon")
```
///

## 🧩 Composants détaillés

### 1. TripParser (Orchestrateur)

**Fichier :** `src/trip_parser/trip_parser.py`

**Rôle :** Point d'entrée principal qui coordonne NER et Classifier.

**Responsabilités :**

- Validation des entrées
- Orchestration du pipeline
- Gestion des erreurs
- Logging des opérations

### 2. NERExtractor (Détection d'entités)

**Fichier :** `src/trip_parser/models/ner.py`

**Rôle :** Détecter les entités géographiques dans le texte.

**Modèle utilisé :** `Jean-Baptiste/camembert-ner`

- Pré-entraîné sur corpus français
- Détecte 4 types d'entités : PER (personnes), LOC (lieux), ORG (organisations), MISC
- Architecture : CamemBERT + couche de classification (4 classes)

**Exemple de tokenisation :**
```python
text = "Je vais de Paris à Lyon"

# Tokenisation CamemBERT (subword)
tokens = ["▁Je", "▁vais", "▁de", "▁Paris", "▁à", "▁Lyon"]

# NER labels (B=Begin, I=Inside, O=Outside)
labels = ["O", "O", "O", "B-LOC", "O", "B-LOC"]

# Agrégation (strategy="simple")
entities = [
    {"entity_group": "LOC", "word": "Paris", "score": 0.99},
    {"entity_group": "LOC", "word": "Lyon", "score": 0.98}
]
```

### 3. DepartureArrivalClassifier (Classification)

**Fichier :** `src/trip_parser/models/classifier.py`

**Rôle :** Déterminer si une ville est un départ ou une arrivée.

**Modèle :** CamemBERT fine-tuné

- Modèle de base : `camembert-base`
- Fine-tuning : Classification binaire (0=departure, 1=arrival)
- Training data : `data/training_dataset.json`

**Exemple d'inférence :**
```python
text = "Je vais de Paris à Lyon"
location = "Paris"

# 1. Marquer
marked = "Je vais de [LOC] à Lyon"

# 2. Tokenize
input_ids = [5, 123, 456, 789, 12, 34, 6]  # IDs CamemBERT

# 3. Forward
logits = model(input_ids)  # → tensor([[4.2, -3.8]])

# 4. Softmax
probs = softmax([[4.2, -3.8]])  # → tensor([[0.9982, 0.0018]])

# 5. Classification
label = argmax([0.9982, 0.0018])  # → 0 (departure)
confidence = 0.9982  # 99.82%

return ("departure", 0.9982)
```

### 4. Configuration centralisée

**Fichier :** `src/trip_parser/config.py`

**Rôle :** Centraliser toute la configuration du projet.

**Pattern :** Singleton + Dataclass

**Usage :**
```python
from trip_parser import get_config

config = get_config()
print(config.paths.models_dir)
print(config.model.ner_model_name)
```

### 5. Hiérarchie d'exceptions

**Fichier :** `src/trip_parser/exceptions.py`

**Pattern :** Exception hierarchy

```python
TripExtractionError (Exception)
│
├── ModelNotFoundError
│   └── Levée quand le modèle n'existe pas sur disque
│
├── ModelLoadError
│   └── Levée quand le chargement du modèle échoue
│
├── InsufficientLocationsError
│   └── Levée quand < 2 villes détectées
│
├── InvalidInputError
│   └── Levée pour validation d'entrée
│
├── ClassificationError
│   └── Levée quand la classification échoue
│
└── TokenizationError
    └── Levée lors d'erreurs de tokenisation
```

**Usage :**
```python
try:
    departure, arrival = parser.parse_trip(text)
except InvalidInputError:
    print("Texte invalide")
except InsufficientLocationsError:
    print("Au moins 2 villes requises")
except ModelNotFoundError:
    print("Exécutez 'trip-train' d'abord")
except TripExtractionError as e:
    print(f"Erreur générique: {e}")
```

## 🔌 API REST

### Architecture

```
┌────────────────────────────────────────────────────┐
│                   Client Request                   │
│            POST /trip/parse {"text": "..."}        │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│               FastAPI Middleware                   │
│  • CORS (allow all origins en dev)                 │
│  • Exception handlers (globaux)                    │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│              Router : trip_router                  │
│  • Route: POST /trip/parse                         │
│  • Validation Pydantic automatique                 │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│         Service : TripParserService                │
│  • Pattern Singleton                               │
│  • Cache de l'instance TripParser                  │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│              TripParser (module ML)                │
│  • NER → Classifier → Résultat                     │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│                 Response JSON                      │
│  {"departure": "Paris", "arrival": "Lyon"}         │
└────────────────────────────────────────────────────┘
```

## 🎨 Design Patterns utilisés

### 1. Facade Pattern

**Où :** `TripParser`

**Pourquoi :** Simplifier l'interface complexe des modèles ML.

```python
# Sans Facade (complexe)
ner = NERExtractor()
classifier = DepartureArrivalClassifier()
cities = ner.extract_locations(text)
if len(cities) >= 2:
    departure, arrival = classifier.classify_locations(text, cities)

# Avec Facade (simple)
parser = TripParser()
departure, arrival = parser.parse_trip(text)
```

### 2. Dependency Injection

**Où :** `TripParser.__init__`

**Pourquoi :** Faciliter les tests et la personnalisation.

```python
# Production : utilise les modèles réels
parser = TripParser()

# Test : utilise des mocks
mock_ner = Mock(spec=NERExtractor)
mock_classifier = Mock(spec=DepartureArrivalClassifier)
parser = TripParser(ner_extractor=mock_ner, classifier=mock_classifier)
```

### 3. Singleton Pattern

**Où :** `TripParserService`, `get_config()`

**Pourquoi :** Éviter de recharger les modèles plusieurs fois.

```python
# Le modèle n'est chargé qu'une fois
service1 = TripParserService()  # Charge le modèle
service2 = TripParserService()  # Réutilise l'instance

assert service1.parser is service2.parser  # Même instance
```

### 4. Template Method Pattern

**Où :** `DepartureArrivalClassifier.classify_locations`

**Pourquoi :** Définir le squelette de l'algorithme.

```python
def classify_locations(self, text, cities):
    # Template : définit les étapes
    candidates = self._classify_all(text, cities)
    departure = self._select_best_departure(candidates)
    arrival = self._select_best_arrival(candidates)
    return (departure, arrival)
```

### 5. Lazy Loading

**Où :** `NERExtractor._load_model`

**Pourquoi :** Ne charger le modèle que si nécessaire.

```python
class NERExtractor:
    def __init__(self):
        self._pipeline = None  # Pas encore chargé
    
    def extract_locations(self, text):
        if self._pipeline is None:
            self._load_model()  # Charge à la première utilisation
        return self._pipeline(text)
```
<