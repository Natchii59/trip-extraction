# Trip Parser API Backend

API REST Python pour l'analyse de voyages avec NER et classification.

Cette application expose l'API REST qui utilise la bibliothèque `trip-parser` située dans `libs/trip-parser`.

## 📁 Structure

```
apps/backend/
├── src/
│   └── api/              # API REST FastAPI
├── scripts/              # Scripts CLI (run_api)
└── pyproject.toml        # Dépendances Python
```

## Installation

```bash
# Depuis la racine du monorepo
source .venv/bin/activate
cd apps/backend
pip install -e ".[dev]"
```

## Développement

```bash
# Depuis la racine du monorepo
npm run dev:backend

# Ou avec Nx directement
nx serve backend
```

## Tests

```bash
nx test backend
```

## API

L'API REST est disponible sur http://localhost:8000
Documentation interactive : http://localhost:8000/docs

## Scripts CLI

Après installation, les commandes suivantes sont disponibles :

```bash
trip-demo     # Démonstration du système
trip-train    # Entraîner les modèles
trip-api      # Lancer l'API REST
```
