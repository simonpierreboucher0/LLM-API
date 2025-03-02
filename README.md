# 🌐 LLM API Gateway 🚀

![LLM API Gateway Banner](https://github.com/simonpierreboucher0/LLM-API/raw/main/assets/banner.png)

## 🔥 La passerelle universelle vers tous vos modèles de langage préférés 🔥

[![GitHub stars](https://img.shields.io/github/stars/simonpierreboucher0/LLM-API?style=social)](https://github.com/simonpierreboucher0/LLM-API/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.105.0+-green.svg)](https://fastapi.tiangolo.com/)

---

## ✨ Caractéristiques principales

🔄 **Interface unifiée** - Une API pour tous les fournisseurs  
🧩 **Normalisation automatique** - Adaptation transparente entre tous les modèles  
🔌 **Compatibilité OpenAI** - Réponses au format standardisé  
⚡ **Streaming temps réel** - Réponses fluides token par token  
🛡️ **Gestion des spécificités** - Adaptation aux particularités de chaque modèle  
📊 **Logging avancé** - Suivi détaillé des requêtes  
🔒 **Sécurité intégrée** - Gestion sécurisée des clés API  
🧠 **+100 modèles** - Accès à une vaste bibliothèque de LLMs  

---

## 🤖 Fournisseurs et modèles pris en charge

### 🟢 OpenAI
Modèles phares du leader du marché des LLMs.

| Modèle | Description | Cas d'utilisation |
|--------|-------------|-------------------|
| 🏆 **gpt-4o** | Le fleuron multimodal d'OpenAI | Applications premium, contenu créatif, résolution de problèmes complexes |
| 🧠 **gpt-4** | Modèle fondateur de la série GPT-4 | Raisonnement, analyse, génération de contenu de qualité |
| 💨 **gpt-3.5-turbo** | Modèle rapide et économique | Chatbots, assistance client, applications grand public |
| 📚 **gpt-4-0613** | Version spécifique de GPT-4 | Applications nécessitant une version stable et spécifique |
| ⚡ **gpt-4-turbo-2024-04-09** | Version turbo de GPT-4 | Traitement plus rapide, connaissances plus récentes |
| 📦 **gpt-3.5-turbo-0125** | Version spécifique de GPT-3.5 | Applications nécessitant une version stable et spécifique |
| 💎 **gpt-4o-mini-2024-07-18** | Version légère de GPT-4o | Applications avec contraintes de coût ou de latence |
| 🚀 **gpt-4o-2024-08-06** | Version récente de GPT-4o | Applications nécessitant les dernières améliorations |
| 🌟 **gpt-4o-2024-05-13** | Version stable de GPT-4o | Applications nécessitant fiabilité et stabilité |
| 🧪 **o1** | Modèle expérimental avancé | Raisonnement profond, problèmes complexes, recherche |
| 🔬 **o3-mini** | Version compacte du modèle expérimental | Raisonnement avancé avec contraintes de ressources |

### 🟣 Anthropic
Modèles conçus pour la sécurité et l'alignement éthique.

| Modèle | Description | Cas d'utilisation |
|--------|-------------|-------------------|
| 🌠 **claude-3-5-sonnet-20241022** | Dernière version de Claude 3.5 Sonnet | Equilibre performance/coût, applications professionnelles |
| ⚡ **claude-3-5-haiku-20241022** | Version rapide de Claude 3.5 | Applications nécessitant faible latence, assistants en temps réel |
| 🔮 **claude-3-opus-20240229** | Modèle le plus puissant de Claude | Recherche, analyses complexes, génération de contenu premium |
| 📝 **claude-3-sonnet-20240229** | Version équilibrée de Claude 3 | Applications professionnelles, contenu de qualité |
| 🚀 **claude-3-haiku-20240307** | Version légère et rapide | Chatbots rapides, applications mobiles, interactions temps réel |
| 💫 **claude-3-7-sonnet-20250219** | Version améliorée et avancée | Applications nécessitant les dernières innovations |

### 🔵 Mistral AI
Modèles européens à la pointe de la performance.

| Modèle | Description | Cas d'utilisation |
|--------|-------------|-------------------|
| 🌊 **mistral-large-latest** | Meilleur modèle général de Mistral | Applications premium nécessitant haute qualité |
| 🎨 **pixtral-large-latest** | Variante multimodale avancée | Applications traitant des images et du texte |
| 🔹 **ministral-3b-latest** | Modèle ultra-compact (3B) | Applications mobiles, edge computing, faibles ressources |
| 🔷 **ministral-8b-latest** | Modèle compact (8B) | Applications avec ressources limitées mais besoin de qualité |
| 🧬 **open-mistral-nemo** | Version open-source optimisée | Applications open-source, déploiements personnalisés |
| 🔵 **mistral-small-latest** | Version équilibrée performance/coût | Applications commerciales, chatbots d'entreprise |
| 🌀 **mistral-saba-latest** | Modèle spécialisé nouvelle génération | Applications nécessitant des capacités avancées spécifiques |
| 💻 **codestral-latest** | Spécialisé pour la programmation | Génération de code, débogage, assistance développeurs |

### 🟠 Cohere
Modèles optimisés pour la recherche et le traitement d'informations.

| Modèle | Description | Cas d'utilisation |
|--------|-------------|-------------------|
| 🌍 **c4ai-aya-expanse-32b** | Modèle multilingue avancé (32B) | Applications globales, traduction, compréhension multiculturelle |
| 🌐 **c4ai-aya-expanse-8b** | Version compacte multilingue (8B) | Applications multilingues légères, traduction rapide |
| 🧭 **command** | Modèle principal de Cohere | Applications générales, chatbots, génération de texte |
| 🔆 **command-light** | Version légère de Command | Applications avec contraintes de ressources |
| 🌓 **command-light-nightly** | Version nightly de Command Light | Tests des dernières améliorations |
| 🌒 **command-nightly** | Version nightly de Command | Accès aux fonctionnalités expérimentales |
| 📡 **command-r** | Version avec capacités de recherche | Applications nécessitant recherche et synthèse d'informations |
| 📅 **command-r-08-2024** | Version datée de Command-R | Applications nécessitant une version stable spécifique |
| 📡⚡ **command-r-plus** | Version améliorée de Command-R | Applications premium nécessitant recherche avancée |
| 📅⚡ **command-r-plus-08-2024** | Version datée de Command-R Plus | Stabilité et performance pour applications critiques |
| 🔵 **command-r7b-12-2024** | Version compacte spécifique (7B) | Applications légères avec recherche intégrée |
| 🌙 **command-r7b-arabic-02-2025** | Spécialisé pour l'arabe | Applications ciblant le marché arabophone |

### 🟤 DeepSeek
Modèles spécialisés dans le raisonnement avancé.

| Modèle | Description | Cas d'utilisation |
|--------|-------------|-------------------|
| 💬 **deepseek-chat** | Modèle conversationnel principal | Chatbots intelligents, assistants personnels |
| 🧮 **deepseek-reasoner** | Spécialisé dans le raisonnement | Problèmes logiques, mathématiques, résolution structurée |

### ⚫ xAI
Modèles de la société d'Elon Musk.

| Modèle | Description | Cas d'utilisation |
|--------|-------------|-------------------|
| 🔮 **grok-2** | Modèle conversationnel avancé | Assistants intelligents avec personnalité, applications interactives |

### 🔴 Google
Modèles multimodaux de pointe.

| Modèle | Description | Cas d'utilisation |
|--------|-------------|-------------------|
| 🌈 **gemini-1.5-pro** | Modèle multimodal premium | Applications multimodales avancées, analyse d'images et vidéos |
| ⚡ **gemini-1.5-flash** | Version rapide de Gemini | Applications en temps réel, traitement multimodal efficace |
| 🚀 **gemini-1.5-flash-8b** | Version compacte (8B) | Applications multimodales avec contraintes de ressources |

### 🟡 Qwen (Alibaba)
Modèles multilingues excellant en chinois et anglais.

| Modèle | Description | Cas d'utilisation |
|--------|-------------|-------------------|
| 👑 **qwen-max** | Modèle le plus puissant de Qwen | Applications premium, analyses complexes |
| ⭐ **qwen-plus** | Version équilibrée haut de gamme | Applications professionnelles, bon rapport qualité/prix |
| 🔆 **qwen-turbo** | Version optimisée pour la vitesse | Chatbots rapides, applications en temps réel |
| 🔹 **qwen2.5-14b-instruct-1m** | Qwen 2.5 (14B) avec 1M contexte | Applications nécessitant longue mémoire contextuelle |
| 🔸 **qwen2.5-7b-instruct-1m** | Qwen 2.5 (7B) avec 1M contexte | Applications légères avec long contexte |
| 💎 **qwen2.5-72b-instruct** | Plus grand modèle Qwen 2.5 (72B) | Applications premium nécessitant haute qualité |
| 💫 **qwen2.5-32b-instruct** | Modèle intermédiaire (32B) | Bon équilibre performance/ressources |
| 🌟 **qwen2.5-14b-instruct** | Modèle compact (14B) | Applications commerciales standard |
| ✨ **qwen2.5-7b-instruct** | Modèle léger (7B) | Applications avec contraintes de ressources |
| 🔅 **qwen2-72b-instruct** | Qwen 2 version large (72B) | Applications premium génération précédente |
| 🔆 **qwen2-7b-instruct** | Qwen 2 version compacte (7B) | Applications légères génération précédente |
| 👑 **qwen1.5-110b-chat** | Plus grand modèle Qwen 1.5 (110B) | Applications très exigeantes en qualité |
| 💫 **qwen1.5-72b-chat** | Modèle large Qwen 1.5 (72B) | Applications premium génération précédente |
| ⭐ **qwen1.5-32b-chat** | Modèle intermédiaire (32B) | Applications commerciales standard |
| 🌟 **qwen1.5-14b-chat** | Modèle compact (14B) | Applications avec ressources modérées |
| ✨ **qwen1.5-7b-chat** | Modèle léger (7B) | Applications avec contraintes importantes |

### 🟦 AI21
Modèles spécialisés dans la génération structurée et contrainte.

| Modèle | Description | Cas d'utilisation |
|--------|-------------|-------------------|
| 🏆 **jamba-1.5-large** | Modèle premium de AI21 | Applications professionnelles, génération structurée |
| 🚀 **jamba-1.5-mini** | Version compacte de Jamba | Applications avec contraintes de ressources |

### 🟪 Perplexity
Modèles optimisés pour la recherche et la synthèse d'informations.

| Modèle | Description | Cas d'utilisation |
|--------|-------------|-------------------|
| 🔬 **sonar-deep-research** | Modèle spécialisé recherche approfondie | Recherche académique, analyses complexes |
| 🧠 **sonar-reasoning-pro** | Premium avec capacités de raisonnement | Applications nécessitant analyse logique avancée |
| 💭 **sonar-reasoning** | Capacités de raisonnement standard | Applications nécessitant analyse logique |
| ⭐ **sonar-pro** | Version premium de Sonar | Applications professionnelles et commerciales |
| 🔍 **sonar** | Modèle de base pour la recherche | Applications standard de recherche et synthèse |
| 🇺🇸 **r1-1776** | Modèle spécialisé US-centric | Applications ciblant le marché américain |

### 🟨 DeepInfra
Plateforme offrant de nombreux modèles open-source optimisés.

| Modèle | Description | Cas d'utilisation |
|--------|-------------|-------------------|
| 🌊 **mistralai/Mixtral-8x7B-Instruct-v0.1** | Mixtral MoE optimisé | Applications nécessitant expertise variée |
| 🧬 **mistralai/Mistral-Nemo-Instruct-2407** | Mistral optimisé pour Nemo | Applications spécifiques Mistral |
| 🚀 **mistralai/Mistral-7B-Instruct-v0.3** | Mistral compact (7B) | Applications efficientes et légères |
| 🦙 **meta-llama/Meta-Llama-3-8B-Instruct** | Llama 3 compact | Applications avec ressources limitées |
| 🐪 **meta-llama/Meta-Llama-3-70B-Instruct** | Llama 3 grand modèle | Applications premium nécessitant haute qualité |
| 🦙🔹 **meta-llama/Llama-3.2-3B-Instruct** | Ultra-compact Llama 3.2 | Applications très légères, edge computing |
| 🦙🔸 **meta-llama/Llama-3.2-1B-Instruct** | Nano Llama 3.2 (1B) | Applications embarquées, ressources minimales |
| 💎 **google/gemma-2-9b-it** | Gemma 2 compact | Applications légères avec bonne qualité |
| 🌟 **google/gemma-2-27b-it** | Gemma 2 intermédiaire | Applications professionnelles équilibrées |
| 🌌 **Sao10K/L3.3-70B-Euryale-v2.3** | Modèle communautaire avancé | Applications spécialisées customisées |
| 🌠 **Sao10K/L3.1-70B-Euryale-v2.2** | Variante communautaire | Applications spécifiques personnalisées |
| 💫 **Sao10K/L3-70B-Euryale-v2.1** | Version précédente Euryale | Applications nécessitant stabilité prouvée |
| 🟡 **Qwen/Qwen2.5-7B-Instruct** | Qwen 2.5 compact | Applications légères multilingues |
| 🌠 **NovaSky-AI/Sky-T1-32B-Preview** | Modèle communautaire expérimental | Applications innovantes |
| 🌌 **NousResearch/Hermes-3-Llama-3.1-405B** | Méga-modèle communautaire | Applications nécessitant qualité maximale |
| 🔮 **Gryphe/MythoMax-L2-13b** | Modèle créatif spécialisé | Applications créatives, narration, fiction |
| 🧙 **microsoft/WizardLM-2-8x22B** | Modèle Microsoft optimisé | Applications professionnelles |
| 🧩 **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B** | Modèle hybride distillé | Applications équilibrées performance/taille |
| 🔷 **mistralai/Mistral-Small-24B-Instruct-2501** | Mistral intermédiaire (24B) | Applications professionnelles équilibrées |
| 🔱 **deepseek-ai/DeepSeek-V3** | Dernière génération DeepSeek | Applications premium nécessitant dernières avancées |
| 🧬 **deepseek-ai/DeepSeek-R1-Distill-Llama-70B** | Grand modèle hybride | Applications premium avec spécificités DeepSeek |
| 🌊 **deepseek-ai/DeepSeek-R1** | Modèle standard DeepSeek R1 | Applications professionnelles généralistes |
| ⚡ **deepseek-ai/DeepSeek-R1-Turbo** | Version optimisée pour vitesse | Applications nécessitant faible latence |
| 👁️ **meta-llama/Llama-3.2-11B-Vision-Instruct** | Llama 3.2 multimodal (11B) | Applications traitant images et texte |
| 🖼️ **meta-llama/Llama-3.2-90B-Vision-Instruct** | Grand Llama 3.2 multimodal | Applications multimodales premium |
| 💎 **Qwen/Qwen2.5-72B-Instruct** | Grand modèle Qwen | Applications premium multilingues |
| ⚡ **nvidia/Llama-3.1-Nemotron-70B-Instruct** | Llama optimisé par NVIDIA | Applications haute performance |
| 💻 **Qwen/Qwen2.5-Coder-32B-Instruct** | Spécialisé pour le code | Applications de développement, génération de code |
| 🚀 **meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo** | Llama 3.1 optimisé vitesse | Applications premium nécessitant faible latence |
| 🔆 **meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo** | Llama 3.1 compact rapide | Applications légères optimisées vitesse |
| 🏆 **meta-llama/Meta-Llama-3.1-405B-Instruct** | Méga modèle Llama (405B) | Applications nécessitant qualité maximale |
| 🔹 **meta-llama/Meta-Llama-3.1-8B-Instruct** | Llama 3.1 compact | Applications légères équilibrées |
| 🔶 **meta-llama/Meta-Llama-3.1-70B-Instruct** | Llama 3.1 standard (70B) | Applications premium standard |
| 🧠 **microsoft/phi-4** | Modèle compact Microsoft | Applications efficientes, bon ratio performance/taille |
| 🦙 **meta-llama/Llama-3.3-70B-Instruct** | Dernière génération Llama | Applications nécessitant dernières innovations |
| ⚡ **meta-llama/Llama-3.3-70B-Instruct-Turbo** | Llama 3.3 optimisé vitesse | Applications premium nécessitant faible latence |

---

## 🛠️ Installation facile en 3 étapes

### 1️⃣ Cloner le dépôt
```bash
git clone https://github.com/simonpierreboucher0/LLM-API.git
cd LLM-API
```

### 2️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3️⃣ Configurer vos clés API
Créez un fichier `.env` avec vos clés:
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

> 💡 **Astuce**: Vous n'avez besoin de fournir que les clés pour les fournisseurs que vous allez utiliser!

---

## 🚀 Guide d'utilisation rapide

### ▶️ Démarrer le serveur
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 📋 Obtenir la liste des modèles
```bash
curl -X GET http://localhost:8000/models
```

### 💬 Chat simple avec GPT-4o
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Explique-moi les LLMs simplement."}],
    "temperature": 0.7
  }'
```

### 🌊 Chat en streaming avec Claude
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "messages": [{"role": "user", "content": "Écris un poème sur l'IA."}],
    "stream": true,
    "system_message": "Tu es un poète talentueux."
  }'
```

### 🧠 Chat avec raisonnement avancé (o1)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "o1",
    "messages": [{"role": "user", "content": "Résous ce problème mathématique: Si 3x + 7 = 22, que vaut x?"}],
    "reasoning_effort": "high"
  }'
```

---

## 📊 Structure complète des requêtes

```json
{
  "provider": "mistral",             // 🌐 Fournisseur LLM (obligatoire)
  "model": "mistral-large-latest",   // 🤖 Modèle spécifique (obligatoire)
  "messages": [                      // 💬 Historique de conversation (obligatoire)
    {"role": "user", "content": "Bonjour, comment ça va?"}
  ],
  "temperature": 0.7,                // 🌡️ Contrôle créativité (0.0-1.0)
  "stream": false,                   // 🌊 Mode streaming (true/false)
  "max_tokens": 1024,                // 📏 Longueur max de réponse
  "system_message": "Tu es un assistant français.", // 🧩 Message système
  "stop": ["\n\n", "Fin:"],          // 🛑 Séquences d'arrêt
  "top_p": 1.0,                      // 📈 Sampling de top-p
  "frequency_penalty": 0.0,          // 🔄 Pénalité répétitions
  "presence_penalty": 0.0,           // 📝 Pénalité présence
  "response_format": {"type": "text"}, // 📄 Format de réponse
  "reasoning_effort": null           // 🧠 Pour modèles OpenAI spéciaux
}
```

---

## 💡 Cas d'utilisation avancés

### 🔄 Failover automatique entre modèles
Si un modèle est indisponible, basculez facilement vers un autre:

```python
providers_priority = ["openai", "anthropic", "mistral"]
models_priority = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-opus-20240229",
    "mistral": "mistral-large-latest"
}

for provider in providers_priority:
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "provider": provider,
                "model": models_priority[provider],
                "messages": [{"role": "user", "content": "Explique l'IA quantique"}]
            }
        )
        if response.status_code == 200:
            print(f"Succès avec {provider}/{models_priority[provider]}")
            print(response.json())
            break
    except Exception as e:
        print(f"Échec avec {provider}: {str(e)}")
```

### 🔍 A/B Testing entre différents modèles
Comparez facilement les performances de différents modèles:

```python
test_prompt = "Explique la théorie de la relativité simplement."
models_to_test = [
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
    {"provider": "mistral", "model": "mistral-large-latest"},
    {"provider": "google", "model": "gemini-1.5-pro"}
]

results = {}
for model_info in models_to_test:
    provider = model_info["provider"]
    model = model_info["model"]
    
    start_time = time.time()
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "provider": provider,
            "model": model,
            "messages": [{"role": "user", "content": test_prompt}]
        }
    )
    end_time = time.time()
    
    if response.status_code == 200:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        latency = end_time - start_time
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        
        results[f"{provider}/{model}"] = {
            "latency": latency,
            "tokens": tokens,
            "content": content[:100] + "..." # Afficher seulement le début
        }

# Afficher les résultats
for model_name, result in results.items():
    print(f"📊 {model_name}:")
    print(f"⏱️ Latence: {result['latency']:.2f}s")
    print(f"🔢 Tokens: {result['tokens']}")
    print(f"📝 Début: {result['content']}")
    print("---")
```

---

## 🧪 Tests complets

Notre script de test permet de vérifier tous les modèles:

```bash
python test_llm_gateway.py
```

Le script exécute:
- ✅ Tests non-streaming pour chaque fournisseur
- 🌊 Tests de streaming pour les modèles compatibles
- 📊 Validation des formats de réponse
- 📝 Génération d'un rapport de test détaillé

```
=== RÉSUMÉ DES TESTS NON-STREAMING ===
openai/gpt-4o: ✅ Succès
anthropic/claude-3-5-sonnet-20241022: ✅ Succès
mistral/mistral-large-latest: ✅ Succès
cohere/c4ai-aya-expanse-32b: ✅ Succès
...

=== RÉSUMÉ DES TESTS STREAMING ===
openai/gpt-4o: ✅ Succès
anthropic/claude-3-5-sonnet-20241022: ✅ Succès
mistral/mistral-large-latest: ✅ Succès
...
```

---

## 🔒 Sécurité et bonnes pratiques

- 🔑 **Gestion des clés**: Stockées localement dans `.env`, jamais exposées
- 🛡️ **CORS**: Configuration sécurisée pour les requêtes cross-origin
- 📝 **Logging**: Détails utiles sans informations sensibles
- 🔄 **Rate Limiting**: Respect des limites de taux des fournisseurs
- 🧹 **Nettoyage**: Aucune donnée stockée sur le serveur

---

## 📚 Guide des fonctionnalités avancées

### 🧩 Messages multimodaux
Certains modèles supportent l'envoi d'images:

```json
{
  "provider": "openai",
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Que vois-tu sur cette image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/image.jpg"
          }
        }
      ]
    }
  ]
}
```

### 🚀 Paramètres spécifiques aux modèles

#### 🧠 Effort de raisonnement pour OpenAI o1/o3
```json
{
  "provider": "openai",
  "model": "o1",
  "messages": [{"role": "user", "content": "Résous ce problème complexe..."}],
  "reasoning_effort": "high"  // Options: "auto", "low", "medium", "high"
}
```

#### 🔄 Format de réponse JSON
```json
{
  "provider": "openai",
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Liste 3 fruits avec leurs caractéristiques"}],
  "response_format": {"type": "json_object"}
}
```

---

## 🤝 Contribution

Les contributions sont les bienvenues! Voici comment participer:

1. 🍴 **Fork** le dépôt
2. 🔄 Créez une **branche** (`git checkout -b feature/ma-fonctionnalite`)
3. ✏️ Faites vos **modifications**
4. 📦 **Commit** vos changements (`git commit -m 'Ajout de ma fonctionnalité'`)
5. 📤 **Push** vers la branche (`git push origin feature/ma-fonctionnalite`)
6. 🔍 Ouvrez une **Pull Request**

### 💼 Idées de contributions

- 🧪 Ajout de tests supplémentaires
- 📝 Amélioration de la documentation
- ✨ Support de nouveaux fournisseurs
- 🚀 Optimisations de performance
- 🌐 Internationalisation

---

## 📜 Licence

Ce projet est sous licence [MIT](LICENSE) - voir le fichier LICENSE pour plus de détails.

---

## ❓ FAQ

### 🔄 Comment mettre à jour les modèles disponibles?
Modifiez le dictionnaire `MODELS` dans le code pour ajouter ou mettre à jour les modèles.

### 🔑 Comment gérer plusieurs clés API pour le même fournisseur?
Vous pouvez étendre la classe `APIKeys` pour implémenter une rotation des clés.

### 🌐 Puis-je exécuter l'API derrière un proxy?
Oui, utilisez les variables d'environnement `HTTP_PROXY` et `HTTPS_PROXY`.

### 📱 Est-ce compatible avec les applications mobiles?
Oui, l'API peut être utilisée par n'importe quel client HTTP.

### 💸 Comment optimiser les coûts?
Utilisez les modèles plus petits pour les tâches simples et réservez les grands modèles pour les tâches complexes.

---

## 👨‍💻 Auteurs

- 🚀 [Simon-Pierre Boucher](https://github.com/simonpierreboucher0) - Créateur principal

---

<p align="center">
⭐ N'oubliez pas de mettre une étoile si ce projet vous a été utile! ⭐
</p>

<p align="center">
🔗 <a href="https://github.com/simonpierreboucher0/LLM-API">GitHub</a> | 
🐛 <a href="https://github.com/simonpierreboucher0/LLM-API/issues">Signaler un problème</a> | 
💡 <a href="https://github.com/simonpierreboucher0/LLM-API/discussions">Discussions</a>
</p>
