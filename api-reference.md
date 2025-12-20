# API Reference

!!! info "Documentation auto-générée"
    Cette page contient la documentation technique complète de tous les modules Python du projet, générée automatiquement à partir des docstrings du code source.

Cette référence API est destinée aux développeurs qui souhaitent :

- Comprendre les signatures exactes des fonctions et méthodes
- Connaître les paramètres et valeurs de retour en détail
- Explorer les exceptions levées par chaque fonction
- Intégrer le module `trip_parser` dans leur propre code

## Module principal

### TripParser

::: trip_parser.TripParser
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      members_order: source

---

## Extracteurs et classifieurs

### NERExtractor

::: trip_parser.models.NERExtractor
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      members_order: source

### DepartureArrivalClassifier

::: trip_parser.models.DepartureArrivalClassifier
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      members_order: source

---

## Configuration

### get_config

::: trip_parser.config.get_config
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### Config

::: trip_parser.config.Config
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      members_order: source

### Paths

::: trip_parser.config.Paths
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      members_order: source

### ModelConfig

::: trip_parser.config.ModelConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      members_order: source

### TrainingConfig

::: trip_parser.config.TrainingConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      members_order: source

### LoggingConfig

::: trip_parser.config.LoggingConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
      members_order: source

---

## Exceptions

### Hiérarchie des exceptions

```
TripExtractionError (base)
├── ModelNotFoundError
├── ModelLoadError
├── InsufficientLocationsError
├── InvalidInputError
├── ClassificationError
└── TokenizationError
```

!!! note "Exceptions exportées"
    Seules les exceptions principales sont exportées dans `__init__.py`. Toutes les exceptions ci-dessous sont disponibles via `trip_parser.exceptions`.

### TripExtractionError

::: trip_parser.exceptions.TripExtractionError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### ModelNotFoundError

::: trip_parser.exceptions.ModelNotFoundError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### InsufficientLocationsError

::: trip_parser.exceptions.InsufficientLocationsError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### InvalidInputError

::: trip_parser.exceptions.InvalidInputError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### ClassificationError

::: trip_parser.exceptions.ClassificationError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

---

## Utilitaires

### setup_logging

::: trip_parser.utils.setup_logging
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### format_trip_result

::: trip_parser.utils.format_trip_result
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

---

## Notes d'utilisation

### Import des classes

```python
# Import principal
from trip_parser import TripParser

# Import des modèles
from trip_parser.models import NERExtractor, DepartureArrivalClassifier

# Import de la configuration
from trip_parser.config import get_config, Config

# Import des exceptions
from trip_parser.exceptions import (
    TripExtractionError,
    ModelNotFoundError,
    InsufficientLocationsError,
    InvalidInputError
)
```

### Gestion des exceptions

Toutes les exceptions héritent de `TripExtractionError`. Pour capturer toutes les erreurs du module :

```python
from trip_parser import TripParser
from trip_parser.exceptions import TripExtractionError

parser = TripParser()

try:
    departure, arrival = parser.parse_trip("Je vais de Paris à Lyon")
except TripExtractionError as e:
    print(f"Erreur d'extraction : {e}")
```

Pour gérer des exceptions spécifiques :

```python
from trip_parser.exceptions import (
    InvalidInputError,
    InsufficientLocationsError,
    ModelNotFoundError
)

try:
    departure, arrival = parser.parse_trip(user_input)
except InvalidInputError:
    print("Le texte fourni est invalide")
except InsufficientLocationsError:
    print("Impossible de trouver 2 villes dans le texte")
except ModelNotFoundError:
    print("Les modèles ne sont pas entraînés. Exécutez 'trip-train' d'abord")
```

### Type hints

Le module utilise les type hints modernes de Python 3.11+ :

```python
from trip_parser import TripParser

parser = TripParser()

# parse_trip retourne tuple[str | None, str | None]
departure, arrival = parser.parse_trip("Paris Lyon")

# Types explicites
departure: str | None
arrival: str | None

if departure is not None and arrival is not None:
    print(f"{departure} → {arrival}")
else:
    print("Extraction partielle ou échouée")
```
