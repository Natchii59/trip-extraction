# Installation

Guide d'installation complet pour Trip Extraction. Suivez les étapes selon votre système d'exploitation et vos besoins.

## 📋 Prérequis

### Système

- **Python** : Version 3.11 ou supérieure (testé sur 3.11 et 3.12)
- **Espace disque** : ~1GB pour les modèles HuggingFace
- **RAM** : Minimum 4GB recommandés (8GB pour GPU)
- **Connexion internet** : Nécessaire pour télécharger les modèles (première utilisation uniquement)

### Optionnel

- **GPU NVIDIA** : Pour accélération CUDA (temps d'entraînement divisé par 4-5)
- **Git** : Pour cloner le repository

/// details | Vérifier votre version de Python

```bash
python --version
# ou
python3 --version
```

Si Python < 3.11, installez une version plus récente depuis [python.org](https://www.python.org/downloads/)

///

## 🚀 Installation standard

### Étape 1 : Cloner le repository

```bash
git clone <repo-url>
cd bootstrap
```

/// details | Sans Git ?

Téléchargez le ZIP depuis GitHub et décompressez-le :
```bash
unzip bootstrap-main.zip
cd bootstrap-main
```

///

### Étape 2 : Créer un environnement virtuel

/// tab | Fish Shell

```bash
# Créer l'environnement
python -m venv .venv

# Activer l'environnement
source .venv/bin/activate.fish

# Vérifier l'activation
which python
# Devrait afficher: /path/to/bootstrap/.venv/bin/python
```

///

/// tab | Bash/Zsh

```bash
# Créer l'environnement
python -m venv .venv

# Activer l'environnement
source .venv/bin/activate

# Vérifier l'activation
which python
# Devrait afficher: /path/to/bootstrap/.venv/bin/python
```

///

/// tab | Windows PowerShell

```powershell
# Créer l'environnement
python -m venv .venv

# Activer l'environnement
.venv\Scripts\Activate.ps1

# Vérifier l'activation
where.exe python
# Devrait afficher: C:\path\to\bootstrap\.venv\Scripts\python.exe
```

///

/// tab | Windows CMD

```batch
# Créer l'environnement
python -m venv .venv

# Activer l'environnement
.venv\Scripts\activate.bat

# Vérifier l'activation
where python
```

///

/// details | Pourquoi un environnement virtuel ?

Les environnements virtuels isolent les dépendances du projet :

- ✅ Évite les conflits entre projets
- ✅ Facilite la reproduction de l'environnement
- ✅ Permet des versions de packages différentes par projet

///

### Étape 3 : Installer les dépendances

```bash
pip install -e .
```

Cette commande installe :

| Package | Version | Usage |
|---------|---------|-------|
| transformers | >=4.36.0 | Bibliothèque HuggingFace pour les modèles |
| torch | >=2.1.0 | PyTorch pour le deep learning |
| sentencepiece | >=0.1.99 | Tokenization pour CamemBERT |
| scikit-learn | >=1.3.0 | Métriques et utilitaires ML |
| accelerate | >=0.26.0 | Accélération GPU/CPU |

/// details | Mise à jour des dépendances

Pour mettre à jour toutes les dépendances :
```bash
pip install --upgrade -e .
```

///

### Étape 4 : Entraîner le modèle

!!! warning "Obligatoire"
    L'entraînement est **obligatoire** la première fois pour créer le modèle classifier.

```bash
trip-train
```

**Durée attendue :**

| Device | Temps | Recommandation |
|--------|-------|----------------|
| **CPU** | 10-12 min | ☕ Prenez un café |
| **GPU (CUDA)** | 2-3 min | ⚡ Rapide |
| **Apple M1/M2** | 5-7 min | 🍎 Intermédiaire |

/// details | Que fait trip-train ?

Le script `trip-train` :

1. Charge le dataset (`data/training_dataset.json`)
2. Split train/test (80/20)
3. Fine-tune CamemBERT (3 epochs)
4. Évalue sur le test set
5. Sauvegarde le modèle dans `models/departure_arrival_classifier/`

///

### Étape 5 : Tester l'installation

```bash
trip-demo
```

Si le demo interactif se lance, **l'installation est réussie** ! 🎉

#### Exemple de session

```
============================================================
Trip Information Extraction v0.1.0
============================================================

Entrez des phrases pour extraire les trajets.
Commandes: 'quit' ou 'exit' pour quitter

✈️  Phrase > Je vais de Paris à Lyon
➡️  Résultat: Paris → Lyon

✈️  Phrase > quit
👋 Au revoir!
```

## 🛠️ Installation pour le développement

Pour contribuer au projet, installez également les outils de développement :

```bash
pip install -e ".[dev]"
```

### Outils inclus

| Outil | Usage | Commande |
|-------|-------|----------|
| **black** | Formatteur de code | `black src/` |
| **ruff** | Linter rapide | `ruff check src/` |
| **mypy** | Type checker | `mypy src/` |
| **pytest** | Framework de tests | `pytest tests/` |
| **ipython** | Shell Python amélioré | `ipython` |
| **jupyter** | Notebooks interactifs | `jupyter lab` |

### Vérifier les outils

```bash
# Formatter le code
black src/ scripts/

# Vérifier avec ruff
ruff check src/

# Type checking
mypy src/
```

## 🎯 Configuration GPU (CUDA)

### Vérifier CUDA

```bash
# Vérifier si CUDA est disponible
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### Installer PyTorch avec CUDA

/// tab | CUDA 11.8

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

///

/// tab | CUDA 12.1

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

///

/// tab | CPU uniquement

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

///

/// details | Quelle version CUDA choisir ?

Vérifiez votre version CUDA :
```bash
nvidia-smi
```

Regardez la ligne `CUDA Version: X.Y`

///

## 🩺 Vérification de l'installation

### Test programmatique

/// codexec

    :::python
    # Vérifier que tous les composants fonctionnent
    from trip import TripParser
    from trip.ner_extractor import NERExtractor
    from trip.departure_arrival_classifier import DepartureArrivalClassifier
    
    print("✅ Import réussi")
    
    # Tester NER
    ner = NERExtractor()
    print("✅ NER Extractor chargé")
    
    # Tester Classifier
    classifier = DepartureArrivalClassifier()
    print("✅ Classifier chargé")
    
    # Tester Parser complet
    parser = TripParser()
    departure, arrival = parser.parse_trip("Je vais de Paris à Lyon")
    print(f"✅ Parsing réussi: {departure} → {arrival}")

///

### Vérifier les modèles

```bash
# Lister les modèles téléchargés
ls -lh models/departure_arrival_classifier/

# Devrait afficher :
# config.json
# model.safetensors
# tokenizer files...
```

## ❗ Problèmes courants

### ModuleNotFoundError: No module named 'trip'

**Cause** : Le package n'est pas installé ou l'environnement n'est pas activé.

**Solution** :
```bash
# Activer l'environnement
source .venv/bin/activate.fish  # ou activate pour bash

# Réinstaller
pip install -e .
```

### ImportError: No module named 'transformers'

**Cause** : Les dépendances ne sont pas installées.

**Solution** :
```bash
pip install -e .
```

### torch.cuda.is_available() retourne False

**Cause** : PyTorch n'a pas le support CUDA ou GPU non détecté.

**Solution** :
```bash
# Réinstaller PyTorch avec CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### OSError: [Errno 28] No space left on device

**Cause** : Espace disque insuffisant pour les modèles (~1GB).

**Solution** :
```bash
# Vérifier l'espace disponible
df -h .

# Libérer de l'espace ou changer de répertoire
```

### Entraînement très lent (> 20 min)

**Cause** : Pas de GPU ou GPU non utilisé.

**Diagnostic** :
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions** :
- Installer CUDA et PyTorch GPU
- Accepter le temps d'entraînement CPU (10-12 min)
- Utiliser un service cloud avec GPU (Google Colab, etc.)

### UnicodeDecodeError sur Windows

**Cause** : Encodage par défaut Windows.

**Solution** :
```bash
# Définir l'encodage UTF-8
set PYTHONUTF8=1
pip install -e .
```

## 🗑️ Désinstallation

### Désinstallation complète

```bash
# Désinstaller le package
pip uninstall trip

# Supprimer l'environnement virtuel
rm -rf .venv

# Supprimer les modèles téléchargés
rm -rf models/

# Supprimer le cache HuggingFace (optionnel)
rm -rf ~/.cache/huggingface/
```

### Garder les modèles

Si vous voulez réinstaller plus tard sans retélécharger les modèles :

```bash
# Désinstaller uniquement le package
pip uninstall trip

# Garder .venv et models/
```

## 📝 Prochaines étapes

Une fois l'installation terminée :

1. 📖 Consultez le [guide d'utilisation](usage.md) pour des exemples
2. 🏗️ Explorez l'[architecture](architecture.md) du système
3. 📚 Référez-vous à l'[API](api.md) pour l'intégration
4. 🎓 Entraînez avec vos propres données (voir README)

## 💡 Conseils

!!! tip "Performance"
    Pour de meilleures performances, utilisez un GPU NVIDIA avec CUDA.

!!! tip "Production"
    En production, épinglez les versions des dépendances :
    ```bash
    pip freeze > requirements.txt
    ```

!!! tip "Mise à jour"
    Gardez les modèles à jour :
    ```bash
    git pull
    trip-train
    ```
