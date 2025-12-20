# Installation

Ce guide vous accompagne pas à pas dans l'installation et la configuration de Trip Extraction sur votre machine de développement.

## ⚙️ Prérequis système

### Versions requises

/// tab | Python
**Version minimale** : Python 3.11

**Vérification** :
```bash
python --version
# ou
python3 --version
```

!!! warning "Python 3.10 et inférieur"
    Le projet utilise des fonctionnalités modernes de Python (Union types avec `|`, etc.) qui nécessitent Python 3.11+. Si vous avez une version inférieure, mettez à jour Python avant de continuer.

**Installation de Python 3.11+ :**
```bash
# macOS (via Homebrew)
brew install python@3.11

# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Windows
# Télécharger depuis python.org
```
///

/// tab | Git
**Pour :** Cloner le repository

```bash
# Vérification
git --version

# Installation si nécessaire
# macOS
brew install git

# Linux
sudo apt install git

# Windows
# Télécharger depuis git-scm.com
```
///

/// tab | pip
**Pour :** Gestion des dépendances Python

```bash
# Vérification
pip --version
# ou
pip3 --version

# Mise à jour
python -m pip install --upgrade pip
```
///

### Espace disque requis

| Composant | Taille | Description |
|-----------|--------|-------------|
| **Code source** | ~10 MB | Fichiers Python, configuration |
| **Dépendances Python** | ~500 MB | PyTorch, Transformers, etc. |
| **Modèles ML** | ~1.5 GB | CamemBERT NER + Classifier |
| **Total estimé** | **~2 GB** | Espace total nécessaire |

### Configuration matérielle recommandée

/// tab | CPU seulement
**Minimum :**

- CPU : 2 cœurs
- RAM : 4 GB
- Temps de traitement : ~300ms par phrase

**Recommandé :**

- CPU : 4+ cœurs
- RAM : 8 GB
- Temps de traitement : ~150ms par phrase
///

/// tab | Avec GPU (optionnel)
**Si vous avez un GPU CUDA :**

- GPU : NVIDIA avec 4+ GB VRAM
- CUDA : 11.8 ou 12.x
- Temps de traitement : ~50-100ms par phrase

**Installation CUDA :**
```bash
# Vérifier si CUDA est disponible
python -c "import torch; print(torch.cuda.is_available())"

# Si False, installer PyTorch avec CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

!!! note "GPU non requis"
    Le système fonctionne parfaitement sur CPU. Le GPU n'est utile que pour accélérer les traitements en production avec gros volume.
///

## 📥 Installation du projet

### Étape 1 : Cloner le repository

```bash
# Cloner le projet
git clone <repo-url>
cd bootstrap

# Vérifier que vous êtes dans le bon dossier
pwd
# Devrait afficher : .../bootstrap

ls
# Devrait montrer : src/ docs/ scripts/ pyproject.toml README.md etc.
```

### Étape 2 : Créer l'environnement virtuel

!!! info "Pourquoi un environnement virtuel ?"
    Un environnement virtuel isole les dépendances du projet et évite les conflits avec d'autres projets Python sur votre machine.

/// tab | fish shell
```bash
# Créer l'environnement
python -m venv .venv

# Activer l'environnement
source .venv/bin/activate.fish

# Vérifier l'activation (le prompt doit changer)
# (.venv) ~/bootstrap $
```
///

/// tab | bash/zsh
```bash
# Créer l'environnement
python -m venv .venv

# Activer l'environnement
source .venv/bin/activate

# Vérifier l'activation (le prompt doit changer)
# (.venv) ~/bootstrap $
```
///

/// tab | Windows
```powershell
# Créer l'environnement
python -m venv .venv

# Activer l'environnement
.venv\Scripts\activate

# Vérifier l'activation (le prompt doit changer)
# (.venv) C:\...\bootstrap>
```
///

!!! warning "Toujours activer l'environnement"
    Vous devez activer l'environnement virtuel **à chaque nouvelle session terminal** avant d'utiliser le projet.

### Étape 3 : Installer les dépendances

```bash
# S'assurer que pip est à jour
pip install --upgrade pip

# Installer le projet en mode éditable
pip install -e .
```

!!! success "Installation en mode éditable (`-e`)"
    Le flag `-e` permet de modifier le code source sans réinstaller le package. Parfait pour le développement !

/// details | Détails des dépendances installées
    type: info

**Dépendances principales** (voir `pyproject.toml`) :

- **transformers** (4.36.0+) : Bibliothèque Hugging Face pour les modèles NLP
- **torch** (2.1.0+) : PyTorch pour le deep learning
- **sentencepiece** (0.1.99+) : Tokenizer pour CamemBERT
- **numpy** (1.24.0+) : Calculs numériques
- **scikit-learn** (1.3.0+) : Métriques et split de données
- **fastapi** (0.109.0+) : Framework API REST
- **uvicorn** (0.27.0+) : Serveur ASGI pour FastAPI
- **pydantic** (2.5.0+) : Validation de données

**Optionnelles** :
```bash
# Outils de développement
pip install -e ".[dev]"   # black, ruff, mypy, ipython

# Documentation
pip install -e ".[docs]"  # mkdocs, mkdocs-shadcn
```
///

### Étape 4 : Entraîner le classifier

!!! danger "Étape obligatoire"
    Le classifier de départ/arrivée doit être entraîné **avant la première utilisation**. Le modèle NER sera téléchargé automatiquement depuis Hugging Face, mais le classifier personnalisé doit être créé localement.

```bash
# Entraîner le classifier
trip-train
```

**Ce que fait cette commande :**

1. Charge les données depuis `data/training_dataset.json`
2. Split en train/validation (80/20)
3. Fine-tune CamemBERT sur vos données
4. Sauvegarde le modèle dans `models/departure_arrival_classifier/`
5. Affiche les métriques de performance

**Sortie attendue :**
```
Loading training data from data/training_dataset.json...
Loaded 1200 examples

Preparing dataset...
Train size: 960, Validation size: 240

Training model...
Epoch 1/3: 100%|██████████| 30/30 [01:23<00:00]
Epoch 2/3: 100%|██████████| 30/30 [01:21<00:00]
Epoch 3/3: 100%|██████████| 30/30 [01:22<00:00]

Evaluating model...
Accuracy: 96.25%
Precision: 97.1%
Recall: 96.8%
F1-Score: 96.9%

Model saved to models/departure_arrival_classifier/
Training completed successfully!
```

/// details | Troubleshooting : Erreur durant l'entraînement
    type: warning

**Problème** : `FileNotFoundError: data/training_dataset.json`
```bash
# Vérifier que le fichier existe
ls data/training_dataset.json

# S'il manque, le dataset doit être fourni
```

**Problème** : `RuntimeError: CUDA out of memory`
```bash
# Réduire la batch size dans scripts/train.py
# Ligne ~200 : per_device_train_batch_size=8  → per_device_train_batch_size=4
```

**Problème** : `ImportError: No module named 'transformers'`
```bash
# Réinstaller les dépendances
pip install -e .
```
///

### Étape 5 : Vérifier l'installation

```bash
# Test 1 : Vérifier que les commandes sont disponibles
which trip-demo
which trip-train
which trip-api

# Test 2 : Lancer le mode démo
trip-demo
```

**Interface demo attendue :**
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

✈️  Phrase > quit
👋 Au revoir!
```

!!! success "Installation terminée !"
    Si `trip-demo` fonctionne correctement, votre installation est complète ! 🎉

## 🔧 Configuration post-installation

### Configuration des chemins

Le système utilise des chemins absolus configurés dans `src/trip_parser/config.py`.

```python
from trip_parser import get_config

config = get_config()

# Afficher les chemins
print(f"Project root: {config.paths.PROJECT_ROOT}")
print(f"Models dir: {config.paths.models_dir}")
print(f"Data dir: {config.paths.data_dir}")
print(f"Logs dir: {config.paths.logs_dir}")
```

**Sortie exemple :**
```
Project root: /Users/natchi/Epitech/T-AIA-911/bootstrap
Models dir: /Users/natchi/Epitech/T-AIA-911/bootstrap/models
Data dir: /Users/natchi/Epitech/T-AIA-911/bootstrap/data
Logs dir: /Users/natchi/Epitech/T-AIA-911/bootstrap/logs
```

!!! tip "Chemins relatifs automatiques"
    Les chemins sont calculés automatiquement depuis `PROJECT_ROOT`. Pas besoin de configuration manuelle !

### Configuration des modèles

```python
from trip_parser import get_config

config = get_config()

# Modèle NER
print(config.model.ner_model_name)
# → "Jean-Baptiste/camembert-ner"

# Seuil de confiance
print(config.model.confidence_threshold)
# → 0.5

# Modifier le seuil (optionnel)
config.model.confidence_threshold = 0.7
```

### Configurer le logging

/// tab | Niveau de logging
```python
from trip_parser.utils import setup_logging
import logging

# Mode développement (verbose)
setup_logging(level=logging.DEBUG)

# Mode production (erreurs seulement)
setup_logging(level=logging.ERROR)
```
///

/// tab | Fichier de logs
```python
from trip_parser.utils import setup_logging

# Écrire les logs dans un fichier
setup_logging(
    level=logging.INFO,
    log_file="logs/trip_parser.log"
)
```
///

/// tab | Format personnalisé
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
```
///

## 🧪 Tests de validation

### Test du module trip_parser

```python
# test_installation.py
from trip_parser import TripParser

def test_basic_parsing():
    parser = TripParser()
    
    # Test 1 : Syntaxe simple
    d, a = parser.parse_trip("De Paris à Lyon")
    assert d == "Paris" and a == "Lyon", "Failed: simple syntax"
    
    # Test 2 : Question
    d, a = parser.parse_trip("Comment aller à Marseille depuis Toulouse ?")
    assert d == "Toulouse" and a == "Marseille", "Failed: question syntax"
    
    # Test 3 : Contexte temporel
    d, a = parser.parse_trip("Demain je vais de Nice à Cannes")
    assert d == "Nice" and a == "Cannes", "Failed: temporal context"
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_basic_parsing()
```

```bash
# Exécuter les tests
python test_installation.py
```

### Test de l'API REST

/// tab | Terminal 1 : Démarrer l'API
```bash
# Lancer le serveur
trip-api

# Devrait afficher :
# INFO:     Started server process
# INFO:     Uvicorn running on http://127.0.0.1:8000
```
///

/// tab | Terminal 2 : Tester avec curl
```bash
# Test de santé
curl http://localhost:8000/health
# → {"status":"healthy","version":"0.1.0"}

# Test d'extraction
curl -X POST http://localhost:8000/trip/parse \
  -H "Content-Type: application/json" \
  -d '{"text": "Je vais de Paris à Lyon"}'
# → {"departure":"Paris","arrival":"Lyon","success":true,"message":null}

# Test de statut
curl http://localhost:8000/trip/status
# → {"models_loaded":true,"ready":true}
```
///

/// tab | Navigateur : Swagger UI
Ouvrir dans un navigateur :
```
http://127.0.0.1:8000/docs
```

Tester directement depuis l'interface Swagger :

1. Cliquer sur `POST /trip/parse`
2. Cliquer sur "Try it out"
3. Entrer `{"text": "Je vais de Paris à Lyon"}`
4. Cliquer sur "Execute"
5. Vérifier la réponse
///

## 🔍 Dépannage (Troubleshooting)

### Problème : ModuleNotFoundError

```python
ModuleNotFoundError: No module named 'trip_parser'
```

**Cause** : Le package n'est pas installé ou l'environnement virtuel n'est pas activé

**Solution** :
```bash
# 1. Vérifier que l'environnement est activé
which python
# Doit afficher : .../bootstrap/.venv/bin/python

# 2. Réinstaller le package
pip install -e .

# 3. Vérifier l'installation
pip list | grep trip-parser
```

### Problème : ModelNotFoundError

```python
ModelNotFoundError: Model not found at 'models/departure_arrival_classifier'
```

**Cause** : Le classifier n'a pas été entraîné

**Solution** :
```bash
# Entraîner le classifier
trip-train

# Vérifier que le modèle existe
ls models/departure_arrival_classifier/
# Doit montrer : config.json, model.safetensors, tokenizer_config.json, etc.
```

### Problème : Téléchargement lent du modèle NER

```
Downloading: 100%|██████████| 440M/440M [15:23<00:00, 476kB/s]
```

**Cause** : Première utilisation, le modèle CamemBERT NER est téléchargé depuis Hugging Face

**Solution** :
```bash
# Option 1 : Patienter (téléchargement unique)
# Les prochaines utilisations seront instantanées (cache)

# Option 2 : Télécharger manuellement
python -c "
from transformers import pipeline
ner = pipeline('ner', model='Jean-Baptiste/camembert-ner')
print('Model cached!')
"
```

### Problème : CUDA out of memory

```
RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB
```

**Cause** : GPU n'a pas assez de VRAM

**Solution** :
```bash
# Option 1 : Forcer l'utilisation du CPU
export CUDA_VISIBLE_DEVICES=""
python scripts/train.py

# Option 2 : Réduire la batch size
# Éditer scripts/train.py ligne ~200
# per_device_train_batch_size=8 → per_device_train_batch_size=2
```

### Problème : Permission denied sur scripts

```bash
-bash: trip-demo: command not found
```

**Cause** : Les scripts ne sont pas dans le PATH ou pas exécutables

**Solution** :
```bash
# Réinstaller le package
pip install -e .

# Vérifier que les scripts sont installés
pip show trip-parser | grep Location
ls $(pip show trip-parser | grep Location | cut -d' ' -f2)/../../../bin/trip-*
```

### Problème : Port 8000 déjà utilisé

```
ERROR: [Errno 48] Address already in use
```

**Cause** : Un autre processus utilise le port 8000

**Solution** :
```bash
# Option 1 : Utiliser un autre port
trip-api --port 8001

# Option 2 : Tuer le processus qui utilise le port 8000
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Vérifier que le port est libre
lsof -i:8000
```
