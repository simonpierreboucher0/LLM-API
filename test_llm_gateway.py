import os
import json
import time
import asyncio
import requests
from typing import Dict, List, Any
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration
API_URL = "http://localhost:8000"
RESULTS_FILE = "model_test_results.json"
OPENAI_SPECIAL_MODELS = ["o1", "o3-mini"]  # Modèles ne supportant pas le streaming

# Message de test court pour éviter les problèmes de contexte
TEST_MESSAGE = "Résume le concept d'API en une phrase."
MAX_TOKENS = 100
TEMPERATURE = 0.7

def get_all_models() -> Dict[str, List[str]]:
    """Récupère tous les modèles disponibles depuis l'API"""
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Erreur lors de la récupération des modèles: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        print(f"Exception lors de la récupération des modèles: {str(e)}")
        return {}

def test_non_streaming(provider: str, model: str) -> Dict[str, Any]:
    """Teste un modèle en mode non-streaming"""
    start_time = time.time()
    result = {
        "provider": provider,
        "model": model,
        "mode": "non-streaming",
        "success": False,
        "response": None,
        "error": None,
        "duration": 0
    }
    
    try:
        print(f"\nTest non-streaming pour {provider}/{model}...")
        
        payload = {
            "provider": provider,
            "model": model,
            "messages": [{"role": "user", "content": TEST_MESSAGE}],
            "temperature": TEMPERATURE,
            "stream": False,
            "max_tokens": MAX_TOKENS
        }
        
        response = requests.post(f"{API_URL}/v1/chat/completions", json=payload, timeout=120)
        
        if response.status_code == 200:
            response_data = response.json()
            content = ""
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0] and "content" in response_data["choices"][0]["message"]:
                    content = response_data["choices"][0]["message"]["content"]
            
            result["success"] = True
            result["response"] = content[:200]  # Limiter la taille de la réponse
        else:
            result["error"] = f"Status: {response.status_code}, Message: {response.text}"
    
    except Exception as e:
        result["error"] = str(e)
    
    result["duration"] = round(time.time() - start_time, 2)
    print(f"{'✅' if result['success'] else '❌'} {provider}/{model} (non-streaming): {result['duration']}s")
    return result

async def test_streaming(provider: str, model: str) -> Dict[str, Any]:
    """Teste un modèle en mode streaming"""
    start_time = time.time()
    result = {
        "provider": provider,
        "model": model,
        "mode": "streaming",
        "success": False,
        "response": None,
        "error": None,
        "duration": 0
    }
    
    # Skip si c'est un modèle qui ne supporte pas le streaming
    if provider == "openai" and model in OPENAI_SPECIAL_MODELS:
        result["error"] = "Ce modèle ne supporte pas le streaming"
        result["duration"] = 0
        print(f"⚠️ {provider}/{model} ne supporte pas le streaming - ignoré")
        return result
    
    try:
        print(f"\nTest streaming pour {provider}/{model}...")
        
        payload = {
            "provider": provider,
            "model": model,
            "messages": [{"role": "user", "content": TEST_MESSAGE}],
            "temperature": TEMPERATURE,
            "stream": True,
            "max_tokens": MAX_TOKENS
        }
        
        response = requests.post(f"{API_URL}/v1/chat/completions", json=payload, stream=True, timeout=120)
        
        if response.status_code == 200:
            content = ""
            received_chunks = False
            
            # Extraire le contenu des chunks
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith("data: "):
                        data_str = line_text[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            received_chunks = True
                            
                            if "choices" in data and len(data["choices"]) > 0:
                                if "delta" in data["choices"][0] and "content" in data["choices"][0]["delta"]:
                                    content += data["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            pass  # Ignorer les chunks mal formés
            
            result["success"] = received_chunks
            result["response"] = content[:200]  # Limiter la taille de la réponse
        else:
            result["error"] = f"Status: {response.status_code}, Message: {response.text}"
    
    except Exception as e:
        result["error"] = str(e)
    
    result["duration"] = round(time.time() - start_time, 2)
    print(f"{'✅' if result['success'] else '❌'} {provider}/{model} (streaming): {result['duration']}s")
    return result

async def run_all_tests():
    """Exécute tous les tests et génère un rapport JSON"""
    all_results = []
    all_models = get_all_models()
    
    if not all_models:
        print("❌ Impossible de récupérer la liste des modèles. Vérifiez que l'API est en cours d'exécution.")
        return
    
    print(f"🔍 Démarrage des tests pour {sum(len(models) for models in all_models.values())} modèles...")
    
    # Pour chaque provider et chaque modèle
    for provider, models in all_models.items():
        for model in models:
            # Test non-streaming
            non_streaming_result = test_non_streaming(provider, model)
            all_results.append(non_streaming_result)
            
            # Test streaming
            streaming_result = await test_streaming(provider, model)
            all_results.append(streaming_result)
    
    # Générer des statistiques
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r["success"])
    
    # Créer le rapport
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": round(successful_tests / total_tests * 100, 2) if total_tests > 0 else 0
        },
        "results": all_results
    }
    
    # Sauvegarder les résultats dans un fichier JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Tests terminés. Taux de réussite: {report['stats']['success_rate']}%")
    print(f"📊 Résultats sauvegardés dans {RESULTS_FILE}")

    # Afficher un résumé par provider
    print("\n=== RÉSUMÉ PAR PROVIDER ===")
    provider_stats = {}
    
    for result in all_results:
        provider = result["provider"]
        if provider not in provider_stats:
            provider_stats[provider] = {"total": 0, "success": 0}
        
        provider_stats[provider]["total"] += 1
        if result["success"]:
            provider_stats[provider]["success"] += 1
    
    for provider, stats in provider_stats.items():
        success_rate = round(stats["success"] / stats["total"] * 100, 2) if stats["total"] > 0 else 0
        print(f"{provider}: {stats['success']}/{stats['total']} tests réussis ({success_rate}%)")

if __name__ == "__main__":
    # Vérifier que l'API est accessible
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code != 200:
            print(f"❌ L'API n'est pas accessible à {API_URL}. Code: {response.status_code}")
            exit(1)
    except Exception as e:
        print(f"❌ Erreur lors de la connexion à l'API: {str(e)}")
        exit(1)
    
    # Exécuter tous les tests
    asyncio.run(run_all_tests())
