# LLM API Gateway

![LLM API Gateway Banner](https://github.com/simonpierreboucher0/LLM-API/raw/main/assets/banner.png)

## 📌 Vue d'ensemble

LLM API Gateway est une API unifiée qui vous permet d'accéder à plus de 100 modèles de langage (LLMs) à travers 11 fournisseurs différents via une interface standardisée. Plus besoin de gérer séparément les multiples formats d'API et les spécificités de chaque fournisseur.

[![GitHub stars](https://img.shields.io/github/stars/simonpierreboucher0/LLM-API?style=social)](https://github.com/simonpierreboucher0/LLM-API/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.105.0+-green.svg)](https://fastapi.tiangolo.com/)

## 🔥 Caractéristiques principales

- **Interface unifiée**: Utilisez la même syntaxe pour tous les modèles
- **Normalisation automatique**: Les requêtes et réponses sont automatiquement adaptées à chaque fournisseur
- **Compatibilité OpenAI**: Format de réponse standardisé compatible avec l'écosystème OpenAI
- **Support du streaming**: Réponses en temps réel pour tous les modèles qui le supportent
- **Gestion des spécificités**: Adaptation automatique aux particularités de chaque modèle
- **Système de logging**: Suivi détaillé des requêtes et des erreurs pour faciliter le débogage

## 🌐 Fournisseurs et modèles pris en charge

| Fournisseur | Modèles clés | Description | Particularités |
|-------------|--------------|-------------|----------------|
| **OpenAI** | GPT-4o, GPT-3.5-Turbo, o1, o3-mini | Leader du marché avec une large gamme de modèles | Modèles spéciaux (o1, o3) avec paramètres spécifiques |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus, Haiku | Modèles optimisés pour la sécurité et l'alignement | Format de message système distinct |
| **Mistral AI** | Mistral Large, Small, Saba, Codestral | Modèles open et closed source avec excellentes performances | Excellents modèles de code |
| **Cohere** | Command, Command-R, Aya | Spécialisés dans la récupération d'informations | Format de contenu spécifique |
| **Google** | Gemini 1.5 Pro, Flash | Modèles multimodaux avancés | Utilise un format de message différent |
| **DeepSeek** | DeepSeek Chat, Reasoner | Modèles spécialisés dans le raisonnement | Compatible format OpenAI |
| **xAI** | Grok-2 | Modèle conversationnel avancé | Similaire au format OpenAI |
| **Qwen** | Qwen2.5, Qwen1.5 | Modèles multilingues d'Alibaba | Bonne performance en mandarin |
| **AI21** | Jamba-1.5 | Modèles spécialisés en génération structurée | Supportent des contraintes avancées |
| **Perplexity** | Sonar, Reasoning Pro | Modèles optimisés pour la recherche | Excellents pour la synthèse d'informations |
| **DeepInfra** | Modèles Mixtral, Llama, Mistral... | Hébergement de nombreux modèles open-source | Grande variété de modèles disponibles |

## 🛠️ Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Clés API pour les fournisseurs que vous souhaitez utiliser

### Étapes d'installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/simonpierreboucher0/LLM-API.git
cd LLM-API
```

2. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

3. **Configurer les clés API**

Créez un fichier `.env` à la racine du projet avec vos clés API:

```ini
OPENAI_API_KEY=sk-xxxx
ANTHROPIC_API_KEY=sk-ant-xxxx
MISTRAL_API_KEY=xxxx
COHERE_API_KEY=xxxx
DEEPSEEK_API_KEY=xxxx
XAI_API_KEY=xxxx
GOOGLE_API_KEY=xxxx
QWEN_API_KEY=xxxx
AI21_API_KEY=xxxx
PERPLEXITY_API_KEY=pplx-xxxx
DEEPINFRA_API_KEY=xxxx
```

> **Note**: Vous n'avez besoin de fournir que les clés API pour les fournisseurs que vous prévoyez d'utiliser.

## 🚀 Utilisation

### Démarrer le serveur

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Une fois le serveur démarré, vous pouvez accéder à la documentation interactive à l'adresse:
```
http://localhost:8000/docs
```

### Exemples de requêtes

#### 1. Obtenir la liste des modèles disponibles

```bash
curl -X GET http://localhost:8000/models
```

#### 2. Chat standard avec GPT-4o

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Explique la théorie de la relativité simplement."}],
    "temperature": 0.7,
    "stream": false,
    "max_tokens": 500
  }'
```

#### 3. Chat en streaming avec Claude

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "messages": [{"role": "user", "content": "Écris une histoire courte sur l'intelligence artificielle."}],
    "temperature": 0.9,
    "stream": true,
    "max_tokens": 800,
    "system_message": "Tu es un écrivain créatif spécialisé en science-fiction."
  }'
```

#### 4. Chat avec le modèle spécial o1 d'OpenAI

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "o1",
    "messages": [{"role": "user", "content": "Explique le fonctionnement des transformers en détail."}],
    "max_tokens": 2000,
    "reasoning_effort": "high"
  }'
```

#### 5. Chat avec contraintes de format

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "mistral",
    "model": "mistral-large-latest",
    "messages": [{"role": "user", "content": "Génère un rapport sur les tendances IA en 2024."}],
    "temperature": 0.5,
    "response_format": {"type": "text"},
    "stop": ["CONCLUSION:", "Fin du rapport"],
    "system_message": "Tu es un analyste expert en intelligence artificielle."
  }'
```

## 📋 Structure complète des requêtes

```json
{
  "provider": "openai",             // Le fournisseur LLM (obligatoire)
  "model": "gpt-4o",                // Le modèle spécifique (obligatoire)
  "messages": [                     // L'historique de conversation (obligatoire)
    {"role": "user", "content": "Bonjour, comment ça va?"}
  ],
  "temperature": 0.7,               // Contrôle de la créativité (0.0-1.0)
  "stream": false,                  // Mode streaming (true/false)
  "max_tokens": 1024,               // Longueur maximale de la réponse
  "system_message": "Tu es un assistant informatique utile et précis.",  // Message système
  "stop": ["\n\n", "Conclusion:"],  // Séquences d'arrêt
  "top_p": 1.0,                     // Sampling de top-p
  "frequency_penalty": 0.0,         // Pénalité pour les répétitions fréquentes
  "presence_penalty": 0.0,          // Pénalité pour les répétitions de sujets
  "response_format": {"type": "text"}, // Format de réponse
  "reasoning_effort": null          // Uniquement pour les modèles OpenAI spéciaux
}
```

## 💡 Fonctionnalités détaillées

### Formats de messages

L'API prend en charge plusieurs formats de contenu pour les messages:

1. **Format texte simple**:
```json
{"role": "user", "content": "Bonjour, comment ça va?"}
```

2. **Format avec objets de contenu** (pour les contenus multimodaux):
```json
{"role": "user", "content": [{"type": "text", "text": "Que représente cette image?"}, {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}]}
```

### Normalisation des messages

Le système convertit automatiquement les formats de messages en fonction des exigences de chaque fournisseur:

- Pour OpenAI standard: Format compatible OpenAI
- Pour les modèles spéciaux d'OpenAI (o1, o3): Format avec rôle "developer" pour les messages système
- Pour Anthropic: Le message système est géré séparément
- Pour Google: Conversion des rôles et structuration spécifique des "parts"

### Gestion du streaming

Le streaming est géré différemment selon les fournisseurs:

- Les modèles spéciaux d'OpenAI (o1, o3) ne supportent pas le streaming
- Les réponses de streaming sont normalisées au format OpenAI pour une compatibilité maximale
- Les formats SSE (Server-Sent Events) spécifiques sont traités automatiquement

### Paramètres spécifiques aux modèles

- `reasoning_effort`: Uniquement pour les modèles OpenAI spéciaux (o1, o3), contrôle la profondeur du raisonnement
- Les modèles spéciaux d'OpenAI n'acceptent que `temperature=1.0`
- Certains fournisseurs utilisent des paramètres spécifiques qui sont automatiquement convertis

## 🔍 Architecture technique

### Composants principaux

1. **Normalisation des requêtes**: Convertit les messages au format spécifique de chaque fournisseur
2. **Transformation des requêtes**: Adapte les paramètres et en-têtes pour chaque API
3. **Traitement du streaming**: Gère les différents formats de streaming et normalise les réponses
4. **Normalisation des réponses**: Convertit toutes les réponses en un format unifié compatible OpenAI

### Flux de traitement

1. La requête est reçue par l'API Gateway
2. Le `provider` et le `model` sont validés
3. Les messages sont normalisés selon le format requis par le fournisseur
4. La requête est transformée avec les en-têtes et paramètres appropriés
5. La requête est envoyée à l'API du fournisseur
6. La réponse (streaming ou non) est normalisée
7. La réponse normalisée est renvoyée au client

### Gestion des erreurs

- Les erreurs spécifiques aux fournisseurs sont capturées et normalisées
- Les exceptions sont propagées avec des messages d'erreur clairs
- Le système de logging enregistre les détails pour faciliter le débogage

## 🧪 Tests

Un script de test est fourni pour vérifier la compatibilité avec tous les modèles:

```bash
python test_llm_gateway.py
```

Ce script va:
- Tester la connectivité avec chaque fournisseur
- Vérifier le bon fonctionnement du mode non-streaming
- Vérifier le bon fonctionnement du mode streaming (pour les modèles qui le supportent)
- Générer un rapport des résultats

## 🚧 Limitations connues

- **Modèles spéciaux d'OpenAI**: Les modèles o1 et o3-mini ne supportent pas le streaming et ont des restrictions de température
- **Formats multimodaux**: La prise en charge des images varie selon les fournisseurs
- **Fonction calling**: Actuellement non supportée de manière unifiée
- **Outils et plugins**: Non supportés actuellement
- **Quotas et limites**: Les limites de rate sont gérées par les fournisseurs individuels

## 🔒 Sécurité

- Les clés API sont stockées localement dans le fichier `.env`
- Les clés ne sont jamais exposées dans les réponses
- CORS est configuré pour permettre les requêtes cross-origin
- Aucune donnée n'est stockée sur le serveur

## 📚 En savoir plus

### Différences entre les fournisseurs

1. **OpenAI**:
   - Large gamme de modèles pour différents cas d'utilisation
   - Modèles spéciaux (o1, o3) avec capacités de raisonnement avancées
   - Bonne gestion du multimodal

2. **Anthropic**:
   - Modèles conçus pour la sécurité et l'alignement
   - Bonnes capacités de raisonnement et de résumé
   - API légèrement différente pour les messages système

3. **Mistral AI**:
   - Excellentes performances pour la taille des modèles
   - Modèles open-source et propriétaires
   - Bons modèles pour le code et le raisonnement

4. **Google**:
   - Modèles Gemini avec excellentes capacités multimodales
   - Structure d'API très différente des autres
   - Bonnes performances sur les tâches complexes

### Cas d'utilisation recommandés

- **Développement et prototypage**: Testez facilement plusieurs modèles sans changer votre code
- **Failover automatique**: Passez d'un fournisseur à un autre en cas d'indisponibilité
- **Optimisation des coûts**: Utilisez le modèle le plus économique adapté à chaque tâche
- **Applications multi-modèles**: Combinez différents modèles selon leurs forces

## 🤝 Contribution

Les contributions sont les bienvenues! Voici comment vous pouvez aider:

1. Fork le dépôt
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/ma-fonctionnalite`)
3. Committez vos changements (`git commit -m 'Ajout de ma fonctionnalité'`)
4. Poussez vers la branche (`git push origin feature/ma-fonctionnalite`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👥 Auteurs et remerciements

- [Simon-Pierre Boucher](https://github.com/simonpierreboucher0) - Créateur et mainteneur principal
- Merci à tous les contributeurs et à la communauté open-source

---

## FAQ

### Comment ajouter une nouvelle clé API?
Ajoutez simplement votre clé dans le fichier `.env` avec le format approprié pour le fournisseur.

### Puis-je utiliser cette API avec des applications frontend?
Oui, l'API est configurée avec CORS pour permettre les requêtes depuis n'importe quelle origine.

### Comment gérer les limites de taux (rate limits)?
L'API transmet les limites de taux des fournisseurs sous-jacents. Si vous atteignez une limite, l'erreur correspondante sera renvoyée.

### Est-ce que tous les modèles supportent les mêmes fonctionnalités?
Non, chaque modèle a ses propres capacités. Par exemple, certains modèles ne supportent pas le streaming ou ont des limitations de paramètres spécifiques.

### Comment puis-je ajouter un nouveau fournisseur?
Pour ajouter un nouveau fournisseur, vous devez:
1. Ajouter le fournisseur à l'énumération `Provider`
2. Définir les modèles disponibles dans `MODELS`
3. Créer une fonction de transformation pour ce fournisseur
4. Ajouter la gestion spécifique dans les fonctions de normalisation

---

Créé par [Simon-Pierre Boucher](https://github.com/simonpierreboucher0) | [Signaler un problème](https://github.com/simonpierreboucher0/LLM-API/issues)
