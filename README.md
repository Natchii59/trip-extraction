# Trip Parser Monorepo

Monorepo pour Trip Parser avec API Python (FastAPI) et interface web React TypeScript, géré par Nx.

## 📁 Structure

```
bootstrap/
├── apps/
│   ├── api/                  # API REST (FastAPI)
│   │   ├── src/api/         # Code source de l'API
│   │   ├── scripts/         # Scripts CLI
│   │   └── pyproject.toml   # Dépendances
│   └── web/                  # Interface web (React)
│       ├── src/             # Code source React
│       ├── package.json     # Dépendances
│       └── vite.config.ts   # Configuration Vite
├── libs/
│   └── trip-parser/          # Bibliothèque Python
│       ├── src/trip_parser/ # Module d'analyse
│       ├── scripts/         # Scripts (demo, train)
│       ├── models/          # Modèles ML
│       └── datasets/        # Données d'entraînement
├── pyproject.toml            # Configuration Python
├── package.json              # Configuration Node.js
└── nx.json                   # Configuration Nx
```

## 🚀 Installation

### Prérequis

- Python >= 3.11
- Node.js >= 18
- Fish shell (pour le script d'installation)

### Installation rapide

```bash
# Installation automatique complète
./install.fish
```

Le script installe automatiquement toutes les dépendances Node.js et Python.

## 💻 Développement

### Démarrer les applications

```bash
# API seule (http://localhost:8000)
nx serve api

# Interface web (http://localhost:5173)
nx serve web

# Les deux en parallèle
npm run dev
```

### URLs

- **Interface web** : http://localhost:5173
- **API** : http://localhost:8000
- **Documentation API** : http://localhost:8000/docs
