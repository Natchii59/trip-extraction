# Trip Extraction

!!! info "Projet interne de parsing de trajets"
    **Trip Extraction** est un système d'IA qui extrait automatiquement les villes de départ et d'arrivée à partir de texte en français. Ce projet interne fournit une API REST et un module Python réutilisable.

## 🎯 Objectif

Permettre l'extraction automatique d'informations de voyage structurées à partir de langage naturel en français.

**Entrée** : `"Je veux prendre le train de Paris à Lyon"`  
**Sortie** : `{"departure": "Paris", "arrival": "Lyon"}`

## ✨ Fonctionnalités clés

- **Extraction intelligente** : Détecte les villes dans diverses formulations
- **Classification contextuelle** : Identifie automatiquement départ vs arrivée
- **API REST** : Exposition HTTP pour tous les langages de programmation
- **Module Python** : Intégration directe dans le code Python
- **Multi-syntaxe** : Gère questions, syntaxe inversée, contexte temporel

## 🚀 Démarrage rapide

### Pour les développeurs qui rejoignent le projet

```bash
# 1. Installation
git clone <repo-url> && cd bootstrap
python -m venv .venv && source .venv/bin/activate.fish
pip install -e . && trip-train

# 2. Lancer l'API
trip-api
# API accessible sur http://127.0.0.1:8000
# Documentation Swagger sur http://127.0.0.1:8000/docs
```

/// details | Tester l'API
```bash
curl -X POST http://127.0.0.1:8000/trip/parse \
  -H "Content-Type: application/json" \
  -d '{"text": "Train de Paris à Lyon"}'
```

**Réponse** :
```json
{
  "departure": "Paris",
  "arrival": "Lyon",
  "success": true,
  "message": null
}
```
///

### Pour utiliser le module Python directement

```python
from trip_parser import TripParser

parser = TripParser()
departure, arrival = parser.parse_trip("Je vais de Paris à Lyon")
print(f"{departure} → {arrival}")  # Paris → Lyon
```

## 🏗️ Architecture en bref

```
┌─────────────┐
│   Input     │ "Je vais de Paris à Lyon"
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│   TripParser            │ Orchestrateur principal
│   (trip_parser.py)      │
└──────┬──────────────────┘
       │
       ├──► ┌─────────────────────┐
       │    │  NERExtractor       │ Extrait les villes
       │    │  (CamemBERT-NER)    │ ["Paris", "Lyon"]
       │    └─────────────────────┘
       │
       └──► ┌──────────────────────────┐
            │  Classifier              │ Classifie départ/arrivée
            │  (CamemBERT fine-tuné)   │ Paris=départ, Lyon=arrivée
            └──────────────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │    Output       │ ("Paris", "Lyon")
            └─────────────────┘
```

Le système utilise **deux modèles ML** en séquence :

1. **NERExtractor** : Détecte toutes les entités de type "ville" avec CamemBERT-NER
2. **Classifier** : Détermine pour chaque ville si c'est un départ ou une arrivée

Cette approche modulaire offre :
- Meilleure précision que des regex
- Flexibilité (changement d'un modèle sans toucher l'autre)
- Réutilisation de modèles pré-entraînés de qualité

## 🛠️ Technologies utilisées

| Composant | Technologie | Usage |
|-----------|-------------|-------|
| **API** | FastAPI + Uvicorn | Serveur HTTP REST |
| **NER** | CamemBERT-NER | Extraction d'entités nommées |
| **Classifier** | CamemBERT (fine-tuné) | Classification départ/arrivée |
| **ML Framework** | Transformers + PyTorch | Inférence des modèles |
| **Validation** | Pydantic | Validation de données API |

## 📊 Métriques de performance

- **Temps de chargement** : 2-5 secondes (chargement initial des modèles)
- **Temps de réponse** : 100-500ms par requête (modèles chargés)
- **Mémoire requise** : ~500 MB (modèles en mémoire)
- **Précision** : 90-95% sur des phrases courantes

## 📚 Documentation

### Pour démarrer

| Page | Description |
|------|-------------|
| **[Installation](installation.md)** | Guide d'installation complet avec prérequis et troubleshooting |
| **[Guide d'utilisation](guide-usage.md)** | Exemples d'utilisation avec CLI, Python et API REST |

### Documentation technique

| Page | Description |
|------|-------------|
| **[Architecture](architecture.md)** | Structure du projet, patterns et pipeline de traitement |
| **[Module Trip Parser](trip-parser.md)** | Détails du module d'extraction ML (modèles, config, exceptions) |
| **[API REST](api-rest.md)** | Documentation de l'API REST (endpoints, déploiement) |
| **[Référence API](api-reference.md)** | Documentation auto-générée des classes et méthodes Python |

## 🔗 Liens rapides

**En développement** :
- Swagger UI : http://127.0.0.1:8000/docs (quand l'API est lancée)
- Code source : dossier `src/`

**Commandes utiles** :
```bash
trip-api        # Lancer l'API REST
trip-demo       # Interface CLI de test
trip-train      # Entraîner le classifier
```
