# API REST

Documentation complète de l'API REST Trip Extraction. Cette API expose le module `trip_parser` via HTTP pour permettre l'intégration dans n'importe quel langage ou framework.

## 🌐 Vue d'ensemble

### URL de base

/// tab | Développement
```
http://127.0.0.1:8000
```
///

/// tab | Production
```
https://your-domain.com
```
///

### Caractéristiques

- **Framework :** FastAPI 0.109.0+
- **Serveur :** Uvicorn (ASGI)
- **Documentation :** Swagger UI automatique
- **Validation :** Pydantic 2.5.0+
- **CORS :** Activé (configurable)

### Endpoints disponibles

| Endpoint | Méthode | Description | Auth |
|----------|---------|-------------|------|
| `/health` | GET | Vérifier l'état de l'API | Non |
| `/trip/status` | GET | État des modèles ML | Non |
| `/trip/parse` | POST | Extraire départ et arrivée | Non |
| `/docs` | GET | Documentation Swagger UI | Non |
| `/openapi.json` | GET | Spécification OpenAPI | Non |

## 🚀 Démarrage du serveur

### Commande de base

```bash
trip-api
```

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

### Options de configuration

/// tab | Port personnalisé
```bash
trip-api --port 8080

# API accessible sur http://127.0.0.1:8080
```
///

/// tab | Host personnalisé
```bash
# Écouter sur toutes les interfaces (pour docker/production)
trip-api --host 0.0.0.0 --port 8000

# API accessible depuis l'extérieur
```
///

/// tab | Mode développement
```bash
# Rechargement automatique à chaque modification
trip-api --reload

# Utile pendant le développement
```
///

/// tab | Production (multi-workers)
```bash
# Lancer 4 workers pour gérer plus de requêtes
trip-api --host 0.0.0.0 --port 8000 --workers 4

# Chaque worker a sa propre instance du modèle
```
///

### Script de lancement

**Fichier :** `scripts/run_api.py`

```python
#!/usr/bin/env python3
"""Script de lancement de l'API Trip Parser."""

import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Run Trip Parser API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )

if __name__ == "__main__":
    main()
```

## 💻 Exemples d'intégration

### cURL

/// tab | Basique
```bash
curl -X POST http://localhost:8000/trip/parse \
  -H "Content-Type: application/json" \
  -d '{"text": "Je vais de Paris à Lyon"}'
```
///

/// tab | Avec jq (formatage)
```bash
curl -s -X POST http://localhost:8000/trip/parse \
  -H "Content-Type: application/json" \
  -d '{"text": "Je vais de Paris à Lyon"}' | jq
```

**Sortie formatée :**
```json
{
  "departure": "Paris",
  "arrival": "Lyon",
  "success": true,
  "message": null
}
```
///

/// tab | Batch avec boucle
```bash
# Fichier phrases.txt contenant une phrase par ligne
while IFS= read -r phrase; do
  echo "Processing: $phrase"
  curl -s -X POST http://localhost:8000/trip/parse \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$phrase\"}" | jq -r '"\(.departure) → \(.arrival)"'
done < phrases.txt
```
///

### Python (requests)

/// tab | Basique
```python
import requests

API_URL = "http://localhost:8000"

def parse_trip(text: str):
    """Appelle l'API pour extraire un trajet."""
    response = requests.post(
        f"{API_URL}/trip/parse",
        json={"text": text}
    )
    response.raise_for_status()
    return response.json()

# Utilisation
result = parse_trip("Je vais de Paris à Lyon")
print(f"Départ: {result['departure']}")
print(f"Arrivée: {result['arrival']}")
```
///

/// tab | Avec gestion d'erreurs
```python
import requests
from typing import Dict, Optional

class TripParserAPI:
    """Client Python pour l'API Trip Parser."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def parse_trip(self, text: str) -> Dict:
        """
        Parse un trajet depuis du texte.
        
        Returns:
            Dict avec departure, arrival, success, message
        
        Raises:
            requests.HTTPError: Si erreur HTTP
        """
        try:
            response = requests.post(
                f"{self.base_url}/trip/parse",
                json={"text": text},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except requests.HTTPError as e:
            if e.response.status_code == 422:
                # Erreur de validation
                detail = e.response.json()["detail"]
                raise ValueError(f"Validation error: {detail}")
            elif e.response.status_code == 500:
                # Erreur serveur
                detail = e.response.json()["detail"]
                raise RuntimeError(f"Server error: {detail}")
            else:
                raise
    
    def is_healthy(self) -> bool:
        """Vérifie si l'API est en ligne."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def is_ready(self) -> bool:
        """Vérifie si les modèles sont chargés."""
        try:
            response = requests.get(f"{self.base_url}/trip/status", timeout=5)
            data = response.json()
            return data.get("ready", False)
        except:
            return False

# Utilisation
api = TripParserAPI()

if not api.is_healthy():
    print("❌ API non accessible")
    exit(1)

if not api.is_ready():
    print("⚠️ Modèles non chargés, attendre...")

try:
    result = api.parse_trip("Je vais de Paris à Lyon")
    
    if result["success"]:
        print(f"✅ {result['departure']} → {result['arrival']}")
    else:
        print(f"⚠️ Extraction échouée: {result['message']}")
        
except ValueError as e:
    print(f"❌ Erreur de validation: {e}")
except RuntimeError as e:
    print(f"❌ Erreur serveur: {e}")
```
///

/// tab | Asynchrone (aiohttp)
```python
import aiohttp
import asyncio
from typing import List, Dict

async def parse_trip_async(session: aiohttp.ClientSession, text: str) -> Dict:
    """Parse un trajet de manière asynchrone."""
    async with session.post(
        "http://localhost:8000/trip/parse",
        json={"text": text}
    ) as response:
        response.raise_for_status()
        return await response.json()

async def batch_parse(texts: List[str]) -> List[Dict]:
    """Parse plusieurs trajets en parallèle."""
    async with aiohttp.ClientSession() as session:
        tasks = [parse_trip_async(session, text) for text in texts]
        return await asyncio.gather(*tasks)

# Utilisation
texts = [
    "Je vais de Paris à Lyon",
    "Train de Marseille à Nice",
    "Vol Toulouse Bordeaux"
]

results = asyncio.run(batch_parse(texts))
for result in results:
    print(f"{result['departure']} → {result['arrival']}")
```
///

### JavaScript (Node.js)

/// tab | fetch (Node 18+)
```javascript
// Fonction pour parser un trajet
async function parseTrip(text) {
    const response = await fetch('http://localhost:8000/trip/parse', {
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
parseTrip("Je vais de Paris à Lyon")
    .then(result => {
        console.log(`Départ: ${result.departure}`);
        console.log(`Arrivée: ${result.arrival}`);
    })
    .catch(error => {
        console.error('Erreur:', error);
    });
```
///

/// tab | axios
```javascript
const axios = require('axios');

class TripParserAPI {
    constructor(baseURL = 'http://localhost:8000') {
        this.client = axios.create({
            baseURL: baseURL,
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }
    
    async parseTrip(text) {
        try {
            const response = await this.client.post('/trip/parse', {
                text: text
            });
            return response.data;
        } catch (error) {
            if (error.response) {
                // Erreur HTTP (422, 500...)
                throw new Error(
                    `API Error ${error.response.status}: ${
                        JSON.stringify(error.response.data)
                    }`
                );
            } else if (error.request) {
                // Pas de réponse
                throw new Error('No response from server');
            } else {
                // Autre erreur
                throw error;
            }
        }
    }
    
    async isHealthy() {
        try {
            const response = await this.client.get('/health');
            return response.status === 200;
        } catch {
            return false;
        }
    }
}

// Utilisation
const api = new TripParserAPI();

(async () => {
    try {
        const result = await api.parseTrip("Je vais de Paris à Lyon");
        
        if (result.success) {
            console.log(`✅ ${result.departure} → ${result.arrival}`);
        } else {
            console.log(`⚠️ ${result.message}`);
        }
    } catch (error) {
        console.error('❌ Erreur:', error.message);
    }
})();
```
///

## 🔒 Sécurité et production

### CORS

La configuration actuelle autorise toutes les origines (mode développement).

**Fichier :** `src/api/main.py`

```python
# Configuration actuelle (DEV)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Toutes les origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration recommandée (PROD)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-domain.com",
        "https://app.your-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
```

### Rate limiting

L'API actuelle n'a pas de rate limiting. Recommandations :

```python
# Utiliser slowapi
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/trip/parse")
@limiter.limit("10/minute")  # Max 10 requêtes par minute
async def parse_trip(request: Request, ...):
    ...
```

### Validation et sanitization

Les entrées sont déjà validées par Pydantic :

- Longueur min/max
- Type string
- Pas de whitespace seul

### Monitoring

**1. Logging structuré**
```python
import logging
import json

logger = logging.getLogger(__name__)

@app.post("/trip/parse")
async def parse_trip(request: TripParseRequest):
    logger.info(json.dumps({
        "event": "parse_request",
        "text_length": len(request.text),
        "timestamp": datetime.now().isoformat()
    }))
```

**2. Métriques (Prometheus)**
```python
from prometheus_client import Counter, Histogram

requests_total = Counter('trip_parse_requests_total', 'Total requests')
request_duration = Histogram('trip_parse_duration_seconds', 'Request duration')
```

## 📖 Documentation interactive

### Swagger UI

Une fois l'API lancée, accédez à :

```
http://127.0.0.1:8000/docs
```

**Fonctionnalités :**

- 📝 Tester tous les endpoints directement depuis le navigateur
- 📄 Voir les schémas détaillés de requête/réponse
- 💡 Exemples de code dans plusieurs langages
- ⬇️ Télécharger la spécification OpenAPI

### OpenAPI Spec

La spécification complète est disponible à :

```
http://127.0.0.1:8000/openapi.json
```

Utilisable avec des outils comme :

- Postman (import OpenAPI)
- Insomnia
- API clients auto-générés
