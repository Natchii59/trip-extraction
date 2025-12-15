# Utilisation

Guide complet d'utilisation de Trip Extraction avec exemples interactifs et cas d'usage avancés.

## 🖥️ Interface CLI

### Demo interactif

Le moyen le plus rapide de tester le système :

```bash
trip-demo
```

### Exemple de session

```
============================================================
Trip Information Extraction v0.1.0
============================================================

Entrez des phrases pour extraire les trajets.
Commandes: 'quit' ou 'exit' pour quitter

✈️  Phrase > Je vais de Paris à Lyon
➡️  Résultat: Paris → Lyon

✈️  Phrase > Train depuis Marseille vers Nice
➡️  Résultat: Marseille → Nice

✈️  Phrase > Comment aller à Toulouse depuis Bordeaux ?
➡️  Résultat: Bordeaux → Toulouse

✈️  Phrase > quit
👋 Au revoir!
```

### Options CLI

/// details | Aide

```bash
trip-demo --help
```

Affiche l'aide et les options disponibles.

///

/// details | Mode verbose

```bash
trip-demo --verbose
```

Affiche des informations détaillées sur le traitement.

///

## 🐍 API Python

### Exemple basique

L'usage le plus simple pour extraire un trajet :

/// codexec

    :::python
    from trip import TripParser
    
    # Initialiser le parser
    parser = TripParser()
    
    # Extraire un trajet
    departure, arrival = parser.parse_trip("Je vais de Paris à Lyon")
    
    print(f"Départ: {departure}")
    print(f"Arrivée: {arrival}")
    # Output:
    # Départ: Paris
    # Arrivée: Lyon

///

### Avec gestion d'erreurs

Production-ready avec gestion complète des erreurs :

/// codexec

    :::python
    from trip import TripParser
    from trip.utils import (
        TripExtractionError,
        InvalidInputError,
        InsufficientLocationsError,
        LowConfidenceError
    )
    
    parser = TripParser()
    
    def extract_trip_safe(text: str):
        """Extraction sécurisée avec gestion d'erreurs."""
        try:
            departure, arrival = parser.parse_trip(text)
            
            if departure and arrival:
                return f"✅ {departure} → {arrival}"
            else:
                return "⚠️ Trajet incomplet détecté"
                
        except InvalidInputError:
            return "❌ Texte vide ou invalide"
        except InsufficientLocationsError:
            return "❌ Pas assez de villes (minimum 2)"
        except LowConfidenceError:
            return "❌ Confiance trop faible"
        except TripExtractionError as e:
            return f"❌ Erreur: {e}"
    
    # Tester différents cas
    print(extract_trip_safe("Je vais de Paris à Lyon"))
    print(extract_trip_safe(""))
    print(extract_trip_safe("Je vais à Paris"))

///

### Utilisation avancée : composants séparés

Utiliser NER et Classifier indépendamment pour plus de contrôle :

/// codexec

    :::python
    from trip.ner_extractor import NERExtractor
    from trip.departure_arrival_classifier import DepartureArrivalClassifier
    
    # Initialiser les composants
    ner = NERExtractor()
    classifier = DepartureArrivalClassifier()
    
    text = "Je vais de Paris à Lyon puis Marseille"
    
    # Étape 1 : Extraire toutes les villes
    locations = ner.extract_locations(text)
    print(f"Villes détectées: {locations}")
    
    # Étape 2 : Classifier chaque ville
    for location in locations:
        role, confidence = classifier.classify_location(text, location)
        print(f"  {location}: {role} (confiance: {confidence:.1%})")

///

## 📝 Exemples de phrases supportées

### Syntaxe simple et directe

/// codexec

    :::python
    from trip import TripParser
    
    parser = TripParser()
    
    phrases_simples = [
        "De Paris à Lyon",
        "Paris Lyon",
        "Train de Marseille vers Nice",
        "Vol Toulouse Bordeaux",
        "Aller de Lille à Strasbourg"
    ]
    
    print("SYNTAXE SIMPLE")
    print("=" * 50)
    for phrase in phrases_simples:
        d, a = parser.parse_trip(phrase)
        print(f"{phrase:35} → {d:12} → {a}")

///

### Questions et formulations complexes

/// codexec

    :::python
    from trip import TripParser
    
    parser = TripParser()
    
    questions = [
        "Comment aller à Marseille depuis Toulouse ?",
        "Où prendre le train pour Nice depuis Paris ?",
        "Quel est le chemin de Bordeaux vers Nantes ?",
        "Comment je fais pour aller à Lille ?"
    ]
    
    print("QUESTIONS")
    print("=" * 50)
    for q in questions:
        d, a = parser.parse_trip(q)
        if d and a:
            print(f"{d:12} → {a:12} | {q}")
        else:
            print(f"Non détecté | {q}")

///

### Syntaxe inversée (destination avant départ)

/// codexec

    :::python
    from trip import TripParser
    
    parser = TripParser()
    
    phrases_inversees = [
        "À Lille depuis Paris",
        "Vers Lyon de Paris",
        "Pour Nice depuis Marseille",
        "Destination Bordeaux départ Toulouse"
    ]
    
    print("SYNTAXE INVERSÉE")
    print("=" * 50)
    for phrase in phrases_inversees:
        d, a = parser.parse_trip(phrase)
        print(f"{phrase:40} → {d:12} → {a}")

///

### Avec contexte temporel ou modal

/// codexec

    :::python
    from trip import TripParser
    
    parser = TripParser()
    
    phrases_contexte = [
        "Demain je vais de Nice à Cannes",
        "Train de 8h de Paris à Lyon",
        "Vol du matin Toulouse Bordeaux",
        "Je pars lundi de Marseille pour aller à Paris"
    ]
    
    print("AVEC CONTEXTE")
    print("=" * 50)
    for phrase in phrases_contexte:
        d, a = parser.parse_trip(phrase)
        print(f"{phrase:50} → {d:12} → {a}")

///

## 🔄 Batch processing

Traiter plusieurs phrases efficacement :

/// codexec

    :::python
    from trip import TripParser
    from typing import List, Tuple, Optional
    
    parser = TripParser()
    
    def batch_extract(
        phrases: List[str]
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Extrait les trajets pour plusieurs phrases.
        
        Returns:
            Liste de tuples (phrase, departure, arrival)
        """
        results = []
        for phrase in phrases:
            try:
                departure, arrival = parser.parse_trip(phrase)
                results.append((phrase, departure, arrival))
            except Exception as e:
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
    print("=" * 60)
    for phrase, d, a in results:
        if d and a:
            print(f"✅ {d:12} → {a:12} | {phrase}")
        else:
            print(f"❌ Non détecté              | {phrase}")

///

### Batch avec statistiques

/// codexec

    :::python
    from trip import TripParser
    from collections import Counter
    
    parser = TripParser()
    
    phrases = [
        "De Paris à Lyon",
        "Paris Marseille",
        "Lyon Nice",
        "Paris Toulouse",
        "Marseille Paris",
        "Lyon Marseille"
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
    
    # Statistiques
    print("STATISTIQUES")
    print("=" * 40)
    print(f"Total phrases: {len(phrases)}")
    print(f"Trajets extraits: {len(routes)}")
    print(f"\nVilles de départ les plus fréquentes:")
    for city, count in Counter(departures).most_common(3):
        print(f"  {city}: {count}x")
    print(f"\nVilles d'arrivée les plus fréquentes:")
    for city, count in Counter(arrivals).most_common(3):
        print(f"  {city}: {count}x")

///

## ⚙️ Configuration avancée

### Ajuster le seuil de confiance

Contrôler la sensibilité de la classification :

```python
from trip.utils import get_config

# Récupérer la configuration
config = get_config()

# Seuil par défaut : 0.5 (50%)
print(f"Seuil actuel: {config.model.confidence_threshold}")

# Rendre plus strict (moins de faux positifs)
config.model.confidence_threshold = 0.8

# Rendre plus permissif (plus de résultats)
config.model.confidence_threshold = 0.3
```

!!! warning "Impact du seuil"
    - **Seuil élevé (0.7-0.9)** : Plus précis mais peut rejeter des trajets valides
    - **Seuil bas (0.3-0.5)** : Plus de résultats mais risque de faux positifs

### Forcer l'utilisation CPU/GPU

```python
from trip.utils import get_config

config = get_config()

# Forcer CPU (utile pour le debugging)
config.model.device = "cpu"

# Forcer GPU si disponible
config.model.device = "cuda"

# Auto-détection (défaut)
import torch
config.model.device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device utilisé: {config.model.device}")
```

### Configuration du logging

```python
from trip.utils import setup_logging
import logging

# Mode production : INFO uniquement
setup_logging(level=logging.INFO)

# Mode debug : tous les détails
setup_logging(level=logging.DEBUG)

# Avec fichier de log
setup_logging(
    level=logging.INFO,
    log_file="trip.log"
)

# Personnalisé
setup_logging(
    level=logging.WARNING,
    log_file="trip_errors.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### Modifier les modèles utilisés

```python
from trip.ner_extractor import NERExtractor
from trip.departure_arrival_classifier import DepartureArrivalClassifier

# Utiliser un autre modèle NER
ner = NERExtractor(model_name="autre-modele-camembert-ner")

# Utiliser un modèle classifier personnalisé
classifier = DepartureArrivalClassifier(
    model_path="./mon_modele_custom/"
)
```

## 🎯 Cas d'usage réels

### 1. Chatbot de voyage

```python
from trip import TripParser

class TravelChatbot:
    def __init__(self):
        self.parser = TripParser()
    
    def handle_message(self, user_message: str) -> str:
        try:
            departure, arrival = self.parser.parse_trip(user_message)
            
            if departure and arrival:
                return (
                    f"Je comprends que vous souhaitez voyager "
                    f"de {departure} à {arrival}. "
                    f"Je recherche les options disponibles..."
                )
            else:
                return "Pouvez-vous préciser votre trajet ?"
                
        except Exception:
            return "Désolé, je n'ai pas compris votre demande."

# Utilisation
bot = TravelChatbot()
response = bot.handle_message("Je veux aller à Paris depuis Lyon")
print(response)
```

### 2. Analyse de logs

```python
from trip import TripParser

def analyze_travel_logs(log_file: str):
    """Analyse un fichier de logs pour extraire les trajets."""
    parser = TripParser()
    trips = []
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                d, a = parser.parse_trip(line)
                if d and a:
                    trips.append((d, a))
            except:
                continue
    
    return trips

# Analyse
trips = analyze_travel_logs("user_queries.log")
print(f"Trajets trouvés: {len(trips)}")
```

### 3. API REST

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from trip import TripParser

app = FastAPI()
parser = TripParser()

class TripRequest(BaseModel):
    text: str

class TripResponse(BaseModel):
    departure: str | None
    arrival: str | None

@app.post("/extract-trip", response_model=TripResponse)
async def extract_trip(request: TripRequest):
    try:
        departure, arrival = parser.parse_trip(request.text)
        return TripResponse(departure=departure, arrival=arrival)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## 📊 Monitoring et métriques

### Mesurer les performances

```python
import time
from trip import TripParser

parser = TripParser()
phrases = ["Je vais de Paris à Lyon"] * 100

# Mesurer le temps
start = time.time()
for phrase in phrases:
    parser.parse_trip(phrase)
end = time.time()

avg_time = (end - start) / len(phrases)
print(f"Temps moyen: {avg_time*1000:.2f}ms par phrase")
print(f"Débit: {len(phrases)/(end-start):.1f} phrases/seconde")
```

## 💡 Bonnes pratiques

!!! tip "Performance"
    - Réutilisez l'instance `TripParser` au lieu d'en créer une nouvelle à chaque fois
    - Pour du batch, utilisez un GPU si disponible
    - Utilisez le logging pour débugger en développement

!!! tip "Gestion d'erreurs"
    - Toujours gérer les exceptions en production
    - Loggez les erreurs pour analyse ultérieure
    - Retournez des messages d'erreur clairs aux utilisateurs

!!! tip "Intégration"
    - Validez l'entrée utilisateur avant le parsing
    - Cachez les résultats si pertinent
    - Ajoutez des métriques pour monitorer l'usage

## 🔗 Prochaines étapes

- Consultez l'[Architecture](architecture.md) pour comprendre le fonctionnement interne
- Explorez l'[API Reference](api.md) pour tous les détails techniques
- Voir le README pour l'entraînement avec vos propres données

```python
# Mode debug
setup_logging(level=logging.DEBUG)

# Avec fichier de log
setup_logging(level=logging.INFO, log_file="trip.log")
```
