# API Reference

Documentation complète de l'API Trip Extraction générée automatiquement depuis le code source.

## Vue d'ensemble

Trip Extraction expose trois composants principaux :

- **TripParser** : Point d'entrée pour l'extraction de trajets
- **NERExtractor** : Extraction d'entités nommées (villes)
- **DepartureArrivalClassifier** : Classification départ/arrivée
- **Exceptions** : Gestion d'erreurs typées

## TripParser

::: trip.trip_parser.TripParser
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members_order: source
      show_signature_annotations: true

## NERExtractor

::: trip.models.ner.NERExtractor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members_order: source
      show_signature_annotations: true

## DepartureArrivalClassifier

::: trip.models.classifier.DepartureArrivalClassifier
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      members_order: source
      show_signature_annotations: true

## Exceptions

### TripExtractionError

::: trip.exceptions.TripExtractionError
    options:
      show_root_heading: true
      heading_level: 4

### InvalidInputError

::: trip.exceptions.InvalidInputError
    options:
      show_root_heading: true
      heading_level: 4

### InsufficientLocationsError

::: trip.exceptions.InsufficientLocationsError
    options:
      show_root_heading: true
      heading_level: 4

### ModelNotFoundError

::: trip.exceptions.ModelNotFoundError
    options:
      show_root_heading: true
      heading_level: 4

### ClassificationError

::: trip.exceptions.ClassificationError
    options:
      show_root_heading: true
      heading_level: 4

## Configuration

### Config

::: trip.config.Config
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

### get_config

::: trip.config.get_config
    options:
      show_root_heading: true
      heading_level: 3
      show_signature_annotations: true

## Exemples d'utilisation

### Exemple basique

```python
from trip import TripParser

parser = TripParser()
departure, arrival = parser.parse_trip("Je vais de Paris à Lyon")
print(f"{departure} → {arrival}")
```

### Gestion d'erreurs

```python
from trip import TripParser
from trip.exceptions import InvalidInputError, InsufficientLocationsError

parser = TripParser()

try:
    departure, arrival = parser.parse_trip(user_input)
except InvalidInputError:
    print("Texte invalide")
except InsufficientLocationsError:
    print("Pas assez de villes")
```

### Configuration personnalisée

```python
from trip import get_config

config = get_config()
config.confidence_threshold = 0.8
```

## Types de retour

Les méthodes principales retournent :

| Méthode | Type de retour | Description |
|---------|---------------|-------------|
| `TripParser.parse_trip()` | `tuple[str \| None, str \| None]` | (departure, arrival) |
| `NERExtractor.extract_locations()` | `list[str]` | Liste des villes |
| `DepartureArrivalClassifier.classify_location()` | `tuple[str, float]` | (role, confidence) |

## Notes de version

Cette documentation est générée automatiquement depuis le code source avec **mkdocstrings**.

Pour plus d'exemples, consultez le [guide d'utilisation](usage.md).
