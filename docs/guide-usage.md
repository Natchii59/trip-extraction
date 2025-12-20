# Guide d'utilisation

Ce guide vous présente toutes les façons d'utiliser Trip Extraction avec des exemples détaillés et des cas d'usage réels.

## 🎯 Modes d'utilisation

Trip Extraction peut être utilisé de **3 façons différentes** selon vos besoins :

| Mode | Usage | Avantages |
|------|-------|-----------|
| **[Module Python](#module-python)** | Import dans code Python | Intégration directe, performances optimales |
| **[CLI (Interface terminal)](#interface-cli)** | Ligne de commande | Tests rapides, démonstration |
| **[API REST](#api-rest)** | Requêtes HTTP | Multi-langage, microservices, scalabilité |

## 🐍 Module Python

### Utilisation basique

L'utilisation la plus simple pour intégrer Trip Extraction dans votre code Python :

```python
from trip_parser import TripParser

# Initialiser le parser (charge les modèles)
parser = TripParser()

# Extraire un trajet
departure, arrival = parser.parse_trip("Je vais de Paris à Lyon")

print(f"Départ: {departure}")   # Départ: Paris
print(f"Arrivée: {arrival}")    # Arrivée: Lyon
```

!!! tip "Initialisation unique"
    Créez **une seule instance** de `TripParser` et réutilisez-la. Le chargement des modèles prend ~2-3 secondes.

### Gestion des cas d'erreur

Production-ready avec gestion complète des erreurs :

```python
from trip_parser import TripParser
from trip_parser.exceptions import (
    TripExtractionError,
    InvalidInputError,
    InsufficientLocationsError,
    ModelNotFoundError
)

parser = TripParser()

def extract_trip_safe(text: str) -> dict:
    """Extraction sécurisée avec gestion d'erreurs."""
    try:
        departure, arrival = parser.parse_trip(text)
        
        if departure and arrival:
            return {
                "status": "success",
                "departure": departure,
                "arrival": arrival
            }
        else:
            return {
                "status": "partial",
                "departure": departure,
                "arrival": arrival,
                "message": "Trajet incomplet détecté"
            }
            
    except InvalidInputError as e:
        return {
            "status": "error",
            "error": "invalid_input",
            "message": str(e)
        }
        
    except InsufficientLocationsError as e:
        return {
            "status": "error",
            "error": "insufficient_locations",
            "message": "Au moins 2 villes requises"
        }
        
    except ModelNotFoundError as e:
        return {
            "status": "error",
            "error": "model_not_found",
            "message": "Exécutez 'trip-train' d'abord"
        }
        
    except TripExtractionError as e:
        return {
            "status": "error",
            "error": "extraction_failed",
            "message": str(e)
        }

# Utilisation
result = extract_trip_safe("Je vais de Paris à Lyon")
print(result)
# → {"status": "success", "departure": "Paris", "arrival": "Lyon"}

result = extract_trip_safe("Je veux aller à Paris")
print(result)
# → {"status": "error", "error": "insufficient_locations", ...}
```

### Exemples de phrases supportées

/// tab | Syntaxe simple
```python
test_cases = [
    "De Paris à Lyon",
    "Paris Lyon",
    "Train de Marseille vers Nice",
    "Vol Toulouse Bordeaux",
    "Aller de Lille à Strasbourg"
]

for phrase in test_cases:
    d, a = parser.parse_trip(phrase)
    print(f"{phrase:40} → {d} → {a}")
```

**Sortie :**
```
De Paris à Lyon                          → Paris → Lyon
Paris Lyon                               → Paris → Lyon
Train de Marseille vers Nice             → Marseille → Nice
Vol Toulouse Bordeaux                    → Toulouse → Bordeaux
Aller de Lille à Strasbourg              → Lille → Strasbourg
```
///

/// tab | Questions
```python
questions = [
    "Comment aller à Marseille depuis Toulouse ?",
    "Où prendre le train pour Nice depuis Paris ?",
    "Quel est le chemin de Bordeaux vers Nantes ?",
    "Comment je fais pour aller à Lille depuis Paris ?"
]

for q in questions:
    d, a = parser.parse_trip(q)
    print(f"{d:15} → {a:15} | {q}")
```

**Sortie :**
```
Toulouse        → Marseille      | Comment aller à Marseille depuis Toulouse ?
Paris           → Nice           | Où prendre le train pour Nice depuis Paris ?
Bordeaux        → Nantes         | Quel est le chemin de Bordeaux vers Nantes ?
Paris           → Lille          | Comment je fais pour aller à Lille depuis Paris ?
```
///

/// tab | Contexte temporel
```python
phrases_contexte = [
    "Demain je vais de Nice à Cannes",
    "Train de 8h de Paris à Lyon",
    "Vol du matin Toulouse Bordeaux",
    "Je pars lundi de Marseille pour aller à Paris"
]

for phrase in phrases_contexte:
    d, a = parser.parse_trip(phrase)
    print(f"{phrase:50} → {d} → {a}")
```

**Sortie :**
```
Demain je vais de Nice à Cannes                    → Nice → Cannes
Train de 8h de Paris à Lyon                        → Paris → Lyon
Vol du matin Toulouse Bordeaux                     → Toulouse → Bordeaux
Je pars lundi de Marseille pour aller à Paris      → Marseille → Paris
```
///

/// tab | Syntaxe inversée
```python
inversions = [
    "À Lille depuis Paris",
    "Vers Lyon de Paris",
    "Direction Marseille de Toulouse",
    "Pour Nice en partant de Paris"
]

for phrase in inversions:
    d, a = parser.parse_trip(phrase)
    print(f"{phrase:45} → {d} → {a}")
```

**Sortie :**
```
À Lille depuis Paris                          → Paris → Lille
Vers Lyon de Paris                            → Paris → Lyon
Direction Marseille de Toulouse               → Toulouse → Marseille
Pour Nice en partant de Paris                 → Paris → Nice
```
///

### Traitement par batch

Pour traiter plusieurs phrases efficacement :

```python
from typing import List, Tuple, Optional
from trip_parser import TripParser

parser = TripParser()

def batch_extract(phrases: List[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Extrait les trajets pour plusieurs phrases.
    
    Args:
        phrases: Liste de phrases à traiter
        
    Returns:
        Liste de tuples (phrase, departure, arrival)
    """
    results = []
    
    for phrase in phrases:
        try:
            departure, arrival = parser.parse_trip(phrase)
            results.append((phrase, departure, arrival))
        except Exception as e:
            # En cas d'erreur, ajouter None, None
            results.append((phrase, None, None))
            
    return results

# Exemple d'utilisation
phrases = [
    "Je vais de Paris à Lyon",
    "Train de Marseille à Nice",
    "Vol Toulouse Bordeaux",
    "Comment aller à Lille ?",  # Ville manquante
]

results = batch_extract(phrases)

print("RÉSULTATS BATCH")
print("=" * 80)
for phrase, d, a in results:
    status = "✅" if d and a else "❌"
    print(f"{status} {phrase:45} → {d or '?':12} → {a or '?'}")
```

**Sortie :**
```
RÉSULTATS BATCH
================================================================================
✅ Je vais de Paris à Lyon                      → Paris        → Lyon
✅ Train de Marseille à Nice                    → Marseille    → Nice
✅ Vol Toulouse Bordeaux                        → Toulouse     → Bordeaux
❌ Comment aller à Lille ?                      → ?            → ?
```

### Statistiques sur un corpus

Analyser un ensemble de phrases pour extraire des statistiques :

```python
from collections import Counter
from trip_parser import TripParser

parser = TripParser()

phrases = [
    "De Paris à Lyon",
    "Paris Marseille",
    "Lyon Nice",
    "Paris Toulouse",
    "Marseille Paris",
    "Lyon Marseille",
    "Bordeaux Paris",
    "Paris Lyon"
]

# Extraire tous les trajets
routes = []
departures = []
arrivals = []

for phrase in phrases:
    d, a = parser.parse_trip(phrase)
    if d and a:
        routes.append(f"{d} → {a}")
        departures.append(d)
        arrivals.append(a)

# Calculer les statistiques
print("📊 STATISTIQUES")
print("=" * 60)
print(f"Total phrases: {len(phrases)}")
print(f"Trajets extraits: {len(routes)}")
print(f"Taux de succès: {len(routes)/len(phrases)*100:.1f}%")

print(f"\n🛫 Villes de départ les plus fréquentes:")
for city, count in Counter(departures).most_common(3):
    print(f"  {city:15} : {count} fois")

print(f"\n🛬 Villes d'arrivée les plus fréquentes:")
for city, count in Counter(arrivals).most_common(3):
    print(f"  {city:15} : {count} fois")

print(f"\n🔄 Routes les plus fréquentes:")
for route, count in Counter(routes).most_common(3):
    print(f"  {route:25} : {count} fois")
```

**Sortie :**
```
📊 STATISTIQUES
============================================================
Total phrases: 8
Trajets extraits: 8
Taux de succès: 100.0%

🛫 Villes de départ les plus fréquentes:
  Paris           : 4 fois
  Lyon            : 2 fois
  Marseille       : 1 fois

🛬 Villes d'arrivée les plus fréquentes:
  Lyon            : 2 fois
  Marseille       : 2 fois
  Paris           : 2 fois

🔄 Routes les plus fréquentes:
  Paris → Lyon              : 2 fois
  Lyon → Marseille          : 1 fois
  Paris → Marseille         : 1 fois
```

### Intégration dans une classe

Exemple d'intégration dans une application orientée objet :

```python
from trip_parser import TripParser
from typing import Optional, Dict
import logging

class TravelService:
    """Service de gestion de voyages avec extraction automatique."""
    
    def __init__(self):
        """Initialise le service avec le parser."""
        self.parser = TripParser()
        self.logger = logging.getLogger(__name__)
        
    def parse_user_query(self, query: str) -> Dict:
        """
        Parse une requête utilisateur et extrait le trajet.
        
        Args:
            query: Requête en langage naturel
            
        Returns:
            Dictionnaire avec les informations du trajet
        """
        self.logger.info(f"Processing query: {query}")
        
        try:
            departure, arrival = self.parser.parse_trip(query)
            
            if departure and arrival:
                return {
                    "success": True,
                    "departure": departure,
                    "arrival": arrival,
                    "original_query": query
                }
            else:
                return {
                    "success": False,
                    "error": "incomplete_trip",
                    "message": "Impossible d'extraire un trajet complet"
                }
                
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": "processing_error",
                "message": str(e)
            }
    
    def suggest_response(self, query: str) -> str:
        """Génère une réponse automatique basée sur le trajet extrait."""
        result = self.parse_user_query(query)
        
        if result["success"]:
            d, a = result["departure"], result["arrival"]
            return (
                f"Je comprends que vous souhaitez voyager "
                f"de {d} à {a}. "
                f"Recherche des options disponibles..."
            )
        else:
            return "Pouvez-vous préciser votre trajet (départ et arrivée) ?"

# Utilisation
service = TravelService()

response = service.suggest_response("Je veux aller à Paris depuis Lyon")
print(response)
# → "Je comprends que vous souhaitez voyager de Lyon à Paris. Recherche..."

response = service.suggest_response("Je veux voyager")
print(response)
# → "Pouvez-vous préciser votre trajet (départ et arrivée) ?"
```

## 💻 Interface CLI

### Lancement de l'interface

```bash
# Lancer le mode interactif
trip-demo
```

### Interface complète

```
╔══════════════════════════════════════╗
║    Trip Extraction Demo v0.1.0       ║
║                                      ║
║  Extracts departure & arrival cities ║
║  from French sentences using NLP     ║
║                                      ║
║  Type 'quit' or 'exit' to quit       ║
╚══════════════════════════════════════╝

Loading models...
Models loaded successfully

✈️  Phrase > Je vais de Paris à Lyon
➡️  Résultat: Paris → Lyon

✈️  Phrase > Comment aller à Marseille depuis Toulouse ?
➡️  Résultat: Toulouse → Marseille

✈️  Phrase > Demain train de Nice à Cannes
➡️  Résultat: Nice → Cannes

✈️  Phrase > quit
👋 Au revoir!
```

### Cas d'usage de l'interface CLI

/// tab | Tests rapides
**Idéal pour :** Tester rapidement de nouvelles formulations

```bash
trip-demo

✈️  Phrase > Vol Paris-Marseille demain matin
➡️  Résultat: Paris → Marseille

✈️  Phrase > Direction Lyon depuis Toulouse
➡️  Résultat: Toulouse → Lyon
```
///

/// tab | Démonstration
**Idéal pour :** Montrer les capacités du système

```bash
# Préparer une liste de phrases impressionnantes
trip-demo

✈️  Phrase > Je voudrais prendre le TGV de Paris pour aller à Marseille
➡️  Résultat: Paris → Marseille

✈️  Phrase > Est-ce qu'il y a un train qui va à Nice depuis Lyon ?
➡️  Résultat: Lyon → Nice
```
///

/// tab | Debugging
**Idéal pour :** Identifier les cas problématiques

```bash
trip-demo

✈️  Phrase > Paris
➡️  Résultat: ✗ Pas assez de villes détectées

✈️  Phrase > Je veux voyager
➡️  Résultat: ✗ Aucune ville détectée
```
///

## 🌐 API REST

### Démarrage du serveur

/// tab | Basique
```bash
# Démarrer sur le port par défaut (8000)
trip-api
```
///

/// tab | Port personnalisé
```bash
# Démarrer sur un port spécifique
trip-api --port 8080
```
///

/// tab | Mode développement
```bash
# Mode développement avec rechargement automatique
trip-api --reload
```
///

/// tab | Production
```bash
# Mode production avec plusieurs workers
trip-api --host 0.0.0.0 --port 8000 --workers 4
```
///

**Sortie attendue :**
```
INFO:     Starting Trip Parser API...
INFO:     Preloading models...
INFO:     Models preloaded successfully
INFO:     Trip Parser API ready
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Endpoints disponibles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Vérifier la santé de l'API |
| `/trip/status` | GET | Vérifier si les modèles sont chargés |
| `/trip/parse` | POST | Extraire départ et arrivée d'un texte |
| `/docs` | GET | Documentation Swagger UI interactive |
| `/openapi.json` | GET | Spécification OpenAPI |

### Exemples avec curl

/// tab | Health check
```bash
curl http://localhost:8000/health
```

**Réponse :**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```
///

/// tab | Status check
```bash
curl http://localhost:8000/trip/status
```

**Réponse :**
```json
{
  "models_loaded": true,
  "ready": true
}
```
///

/// tab | Parse trip (succès)
```bash
curl -X POST http://localhost:8000/trip/parse \
  -H "Content-Type: application/json" \
  -d '{"text": "Je vais de Paris à Lyon"}'
```

**Réponse :**
```json
{
  "departure": "Paris",
  "arrival": "Lyon",
  "success": true,
  "message": null
}
```
///

/// tab | Parse trip (échec)
```bash
curl -X POST http://localhost:8000/trip/parse \
  -H "Content-Type: application/json" \
  -d '{"text": "Je veux voyager"}'
```

**Réponse :**
```json
{
  "departure": null,
  "arrival": null,
  "success": false,
  "message": "Could not extract departure and arrival cities from the text"
}
```
///

/// tab | Validation error
```bash
curl -X POST http://localhost:8000/trip/parse \
  -H "Content-Type: application/json" \
  -d '{"text": ""}'
```

**Réponse (HTTP 422) :**
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "text"],
      "msg": "String should have at least 1 character",
      "input": "",
      "ctx": {"min_length": 1}
    }
  ]
}
```
///

### Exemples avec Python (requests)

```python
import requests

API_URL = "http://localhost:8000"

def parse_trip_api(text: str) -> dict:
    """Appelle l'API pour extraire un trajet."""
    response = requests.post(
        f"{API_URL}/trip/parse",
        json={"text": text},
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json()

# Utilisation
result = parse_trip_api("Je vais de Paris à Lyon")
print(result)
# → {"departure": "Paris", "arrival": "Lyon", "success": true, "message": null}

# Gestion d'erreurs
try:
    result = parse_trip_api("")
except requests.HTTPError as e:
    print(f"Erreur HTTP: {e.response.status_code}")
    print(e.response.json())
```

### Exemples avec JavaScript (fetch)

```javascript
const API_URL = "http://localhost:8000";

async function parseTripAPI(text) {
    const response = await fetch(`${API_URL}/trip/parse`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

// Utilisation
parseTripAPI("Je vais de Paris à Lyon")
    .then(result => {
        console.log(`Départ: ${result.departure}`);
        console.log(`Arrivée: ${result.arrival}`);
    })
    .catch(error => {
        console.error('Erreur:', error);
    });
```

### Interface Swagger

L'API expose automatiquement une interface interactive Swagger UI :

```
http://127.0.0.1:8000/docs
```

**Fonctionnalités :**

- Tester tous les endpoints directement depuis le navigateur
- Voir les schémas de requête/réponse
- Générer des exemples de code
- Télécharger la spécification OpenAPI

## ⚡ Bonnes pratiques

### Performance

**1. Réutiliser l'instance de TripParser**
```python
# ❌ Mauvais (charge les modèles à chaque fois)
def process(text):
    parser = TripParser()  # ~2-3s de chargement
    return parser.parse_trip(text)

# ✅ Bon (charge une seule fois)
parser = TripParser()  # Chargement unique

def process(text):
    return parser.parse_trip(text)  # ~100-300ms
```

**2. Utiliser le batch processing pour gros volumes**
```python
# Traiter 1000 phrases
phrases = [...]
results = [parser.parse_trip(p) for p in phrases]
```

**3. Considérer l'API REST pour la scalabilité**
```python
# L'API peut gérer plusieurs workers en parallèle
trip-api --workers 4
```

### Gestion d'erreurs

```python
from trip_parser.exceptions import TripExtractionError

try:
    departure, arrival = parser.parse_trip(user_input)
except TripExtractionError as e:
    # Gestion spécifique aux erreurs du parser
    logger.error(f"Parsing failed: {e}")
except Exception as e:
    # Gestion des erreurs inattendues
    logger.error(f"Unexpected error: {e}")
```

### Validation

```python
departure, arrival = parser.parse_trip(text)

# Vérifier que les deux villes sont présentes
if not (departure and arrival):
    # Demander plus d'informations à l'utilisateur
    return "Veuillez préciser votre trajet complet"

# Vérifier qu'elles sont différentes
if departure == arrival:
    return "La ville de départ et d'arrivée sont identiques"

# Valider que ce sont des villes connues (optionnel)
KNOWN_CITIES = ["Paris", "Lyon", "Marseille", ...]
if departure not in KNOWN_CITIES:
    logger.warning(f"Unknown departure city: {departure}")
```

## 📖 Ressources supplémentaires

- [Architecture du projet](architecture.md) - Comprendre la structure du code
- [Documentation technique](trip-parser.md) - Détails des modèles ML
- [API REST complète](api-rest.md) - Référence complète de l'API
