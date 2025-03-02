import os
import json
import time
import logging
import asyncio
import re
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Charger les variables d'environnement depuis un fichier .env
load_dotenv()

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM API Gateway", description="API Gateway unifié pour différents modèles de langage")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enumération des providers disponibles
class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    GOOGLE = "google"
    QWEN = "qwen"
    AI21 = "ai21"
    PERPLEXITY = "perplexity"
    DEEPINFRA = "deepinfra"

# Classification spéciale pour les modèles OpenAI qui ont des comportements différents
OPENAI_SPECIAL_MODELS = ["o1", "o3-mini"]

# Enumération des modèles disponibles par provider
MODELS = {
    Provider.OPENAI: [
        "gpt-4o", 
        "gpt-4", 
        "gpt-3.5-turbo", 
        "gpt-4-0613", 
        "gpt-4-turbo-2024-04-09", 
        "gpt-3.5-turbo-0125",
        "gpt-4o-mini-2024-07-18", 
        "gpt-4o-2024-08-06", 
        "gpt-4o-2024-05-13",
        "o1", "o3-mini"
    ],
    
    Provider.ANTHROPIC: [
        "claude-3-5-sonnet-20241022", 
        "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229", 
        "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307",
        "claude-3-7-sonnet-20250219"
    ],
    
Provider.MISTRAL: [
    "mistral-large-latest",  # points to mistral-large-2411
    "pixtral-large-latest",  # points to pixtral-large-2411
    "ministral-3b-latest",  # points to ministral-3b-2410
    "ministral-8b-latest",  # points to ministral-8b-2410
    "open-mistral-nemo",  # points to open-mistral-nemo-2407
    "mistral-small-latest",  # points to mistral-small-2501
    "mistral-saba-latest",  # points to mistral-saba-2502
    "codestral-latest"  # points to codestral-2501
],
    
   Provider.COHERE: [
        "c4ai-aya-expanse-32b",
        "c4ai-aya-expanse-8b",
        "command",
        "command-light",
        "command-light-nightly",
        "command-nightly",
        "command-r",
        "command-r-08-2024",
        "command-r-plus",
        "command-r-plus-08-2024",
        "command-r7b-12-2024",
        "command-r7b-arabic-02-2025"
    ],
    
    Provider.DEEPSEEK: [
        "deepseek-chat", 
        "deepseek-reasoner"
    ],
    
    Provider.XAI: [
        "grok-2"
    ],
    
    Provider.GOOGLE: [
        "gemini-1.5-pro", 
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b"
    ],
    
    Provider.QWEN: [
        "qwen-max",
        "qwen-plus",
        "qwen-turbo",
        "qwen2.5-14b-instruct-1m",
        "qwen2.5-7b-instruct-1m",
        "qwen2.5-72b-instruct",
        "qwen2.5-32b-instruct",
        "qwen2.5-14b-instruct",
        "qwen2.5-7b-instruct",
        "qwen2-72b-instruct",
        "qwen2-7b-instruct",
        "qwen1.5-110b-chat",
        "qwen1.5-72b-chat",
        "qwen1.5-32b-chat",
        "qwen1.5-14b-chat",
        "qwen1.5-7b-chat"
],
    
    Provider.AI21: [
        "jamba-1.5-large", "jamba-1.5-mini"
    ],
    
    Provider.PERPLEXITY: [
    "sonar-deep-research",
    "sonar-reasoning-pro",
    "sonar-reasoning",
    "sonar-pro",
    "sonar",
    "r1-1776"
    ],
    
Provider.DEEPINFRA: [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "Sao10K/L3.3-70B-Euryale-v2.3",
    "Sao10K/L3.1-70B-Euryale-v2.2",
    "Sao10K/L3-70B-Euryale-v2.1",
    "Qwen/Qwen2.5-7B-Instruct",
    "NovaSky-AI/Sky-T1-32B-Preview",
    "NousResearch/Hermes-3-Llama-3.1-405B",
    "Gryphe/MythoMax-L2-13b",
    "microsoft/WizardLM-2-8x22B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Turbo",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "microsoft/phi-4",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
]
}

# Structure pour les éléments de contenu
class ContentItem(BaseModel):
    type: str = "text"
    text: str

# Structure pour les messages
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], List[ContentItem]]

# Structure unifiée pour la requête
class ChatCompletionRequest(BaseModel):
    provider: Provider
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 1024
    system_message: Optional[str] = None
    stop: Optional[List[str]] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, str]] = Field(default={"type": "text"})
    reasoning_effort: Optional[str] = None  # Pour les modèles OpenAI spéciaux (o1, o3)

# Classe pour gérer les API keys depuis .env
class APIKeys:
    def __init__(self):
        self.keys = {
            Provider.OPENAI: os.getenv("OPENAI_API_KEY"),
            Provider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY"),
            Provider.MISTRAL: os.getenv("MISTRAL_API_KEY"),
            Provider.COHERE: os.getenv("COHERE_API_KEY"),
            Provider.DEEPSEEK: os.getenv("DEEPSEEK_API_KEY"),
            Provider.XAI: os.getenv("XAI_API_KEY"),
            Provider.GOOGLE: os.getenv("GOOGLE_API_KEY"),
            Provider.QWEN: os.getenv("QWEN_API_KEY"),
            Provider.AI21: os.getenv("AI21_API_KEY"),
            Provider.PERPLEXITY: os.getenv("PERPLEXITY_API_KEY"),
            Provider.DEEPINFRA: os.getenv("DEEPINFRA_API_KEY")
        }

    def get_key(self, provider: Provider):
        key = self.keys.get(provider)
        if not key:
            raise HTTPException(status_code=401, detail=f"API key for {provider} not found")
        return key

api_keys = APIKeys()

# Endpoints pour les modèles disponibles
@app.get("/models")
def get_models():
    return MODELS

# Fonction pour traiter le contenu des messages
def process_message_content(content):
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        # Traiter une liste de dictionnaires ou d'objets ContentItem
        processed_content = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    processed_content.append({"type": item.get("type", "text"), "text": item["text"]})
            elif hasattr(item, "text"):
                processed_content.append({"type": getattr(item, "type", "text"), "text": item.text})
                
        return processed_content
    
    # Si le format n'est pas reconnu, on retourne une chaîne vide
    return ""

# Fonction pour normaliser les messages à envoyer à chaque fournisseur
def normalize_messages(messages, provider: Provider, system_message: Optional[str] = None, model: Optional[str] = None):
    normalized_messages = []
    
    # Ajouter le message système si nécessaire
    if system_message:
        # Pour les modèles spéciaux d'OpenAI, le message système est traité différemment
        if provider == Provider.OPENAI and model in OPENAI_SPECIAL_MODELS:
            # Pour o1/o3, le message système est inclus comme un message avec le rôle "developer"
            normalized_messages.append({
                "role": "developer", 
                "content": [{"type": "text", "text": system_message}]
            })
        elif provider == Provider.ANTHROPIC:
            # Pour Anthropic, ne pas l'ajouter ici, car il sera traité séparément
            pass
        else:
            # Pour les autres, ajouter comme message système
            normalized_messages.append({"role": "system", "content": system_message})
    
    for msg in messages:
        content = process_message_content(msg.content)
        
        if provider == Provider.OPENAI:
            if model in OPENAI_SPECIAL_MODELS:
                # Format spécial pour les modèles o1/o3
                if isinstance(content, list):
                    normalized_messages.append({"role": msg.role, "content": content})
                else:
                    normalized_messages.append({
                        "role": msg.role, 
                        "content": [{"type": "text", "text": content}]
                    })
            else:
                # Format standard OpenAI pour les autres modèles
                normalized_messages.append({"role": msg.role, "content": content})
        
        elif provider == Provider.ANTHROPIC:
            # Format Anthropic - convertit en texte simple si nécessaire
            if isinstance(content, list):
                text_content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
                normalized_messages.append({"role": msg.role, "content": text_content})
            else:
                normalized_messages.append({"role": msg.role, "content": content})
        
        elif provider == Provider.MISTRAL:
            # Format Mistral - texte simple
            if isinstance(content, list):
                text_content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
                normalized_messages.append({"role": msg.role, "content": text_content})
            else:
                normalized_messages.append({"role": msg.role, "content": content})
        
        elif provider == Provider.COHERE:
            # Format Cohere - structure spécifique
            if isinstance(content, list):
                text_content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
                normalized_messages.append({
                    "role": msg.role, 
                    "content": {"type": "text", "text": text_content}
                })
            else:
                normalized_messages.append({
                    "role": msg.role, 
                    "content": {"type": "text", "text": content}
                })
        
        elif provider == Provider.GOOGLE:
            # Format Google Gemini
            role = "user" if msg.role == "user" else "model"
            if isinstance(content, list):
                parts = [{"text": item.get("text", "")} for item in content if item.get("type") == "text"]
                normalized_messages.append({"role": role, "parts": parts})
            else:
                normalized_messages.append({"role": role, "parts": [{"text": content}]})
        
        else:
            # Format par défaut pour les autres providers
            if isinstance(content, list):
                text_content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
                normalized_messages.append({"role": msg.role, "content": text_content})
            else:
                normalized_messages.append({"role": msg.role, "content": content})
    
    return normalized_messages

# Fonctions pour transformer les requêtes selon le provider
def transform_openai_request(request: ChatCompletionRequest):
    # Vérifier si c'est un modèle spécial (o1, o3)
    is_special_model = request.model in OPENAI_SPECIAL_MODELS
    
    normalized_messages = normalize_messages(
        request.messages, 
        Provider.OPENAI,
        request.system_message,
        request.model
    )
    
    # Si c'est un modèle spécial, désactiver le streaming car ils ne le supportent pas
    stream = request.stream
    if is_special_model:
        stream = False
    
    payload = {
        "model": request.model,
        "messages": normalized_messages,
        "stream": stream,
        "response_format": request.response_format
    }
    
    # Pour les modèles O1/O3, n'incluez pas temperature (seulement la valeur par défaut 1.0 est supportée)
    if not is_special_model or request.temperature == 1.0:
        payload["temperature"] = request.temperature
    
    # Utiliser max_completion_tokens pour les modèles O1 au lieu de max_tokens
    if is_special_model:
        if request.max_tokens:
            payload["max_completion_tokens"] = request.max_tokens
    else:
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
    
    # Ajouter les paramètres spécifiques au modèle o1/o3 si nécessaire
    if is_special_model and request.reasoning_effort:
        payload["reasoning_effort"] = request.reasoning_effort
    
    # Ajouter les paramètres généraux si applicables
    if not is_special_model:
        if request.stop:
            payload["stop"] = request.stop
        if request.top_p:
            payload["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_keys.get_key(Provider.OPENAI)}"
    }
    
    return "https://api.openai.com/v1/chat/completions", headers, payload

def transform_anthropic_request(request: ChatCompletionRequest):
    normalized_messages = normalize_messages(
        request.messages, 
        Provider.ANTHROPIC
    )
    
    payload = {
        "model": request.model,
        "messages": normalized_messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": request.stream
    }
    
    if request.system_message:
        payload["system"] = request.system_message
    
    if request.stop:
        payload["stop_sequences"] = request.stop
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_keys.get_key(Provider.ANTHROPIC),
        "anthropic-version": "2023-06-01"
    }
    
    return "https://api.anthropic.com/v1/messages", headers, payload

def transform_mistral_request(request: ChatCompletionRequest):
    normalized_messages = normalize_messages(
        request.messages, 
        Provider.MISTRAL,
        request.system_message
    )
    
    payload = {
        "model": request.model,
        "messages": normalized_messages,
        "temperature": request.temperature,
        "stream": request.stream,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_keys.get_key(Provider.MISTRAL)}"
    }
    
    return "https://api.mistral.ai/v1/chat/completions", headers, payload

def transform_cohere_request(request: ChatCompletionRequest):
    normalized_messages = normalize_messages(
        request.messages, 
        Provider.COHERE
    )
    
    payload = {
        "model": request.model,
        "messages": normalized_messages,
        "temperature": request.temperature,
        "stream": request.stream,
        "p": request.top_p
    }
    
    if request.system_message:
        payload["system"] = request.system_message
        
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {api_keys.get_key(Provider.COHERE)}"
    }
    
    return "https://api.cohere.com/v2/chat", headers, payload

def transform_deepseek_request(request: ChatCompletionRequest):
    normalized_messages = normalize_messages(
        request.messages, 
        Provider.DEEPSEEK,
        request.system_message
    )
    
    payload = {
        "model": request.model,
        "messages": normalized_messages,
        "temperature": request.temperature,
        "stream": request.stream,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "response_format": request.response_format
    }
    
    if request.stop:
        payload["stop"] = request.stop
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_keys.get_key(Provider.DEEPSEEK)}"
    }
    
    return "https://api.deepseek.com/chat/completions", headers, payload

def transform_xai_request(request: ChatCompletionRequest):
    normalized_messages = normalize_messages(
        request.messages, 
        Provider.XAI,
        request.system_message
    )
    
    payload = {
        "model": request.model,
        "messages": normalized_messages,
        "temperature": request.temperature,
        "stream": request.stream,
        "max_tokens": request.max_tokens
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_keys.get_key(Provider.XAI)}"
    }
    
    return "https://api.x.ai/v1/chat/completions", headers, payload

def transform_google_request(request: ChatCompletionRequest):
    # Les messages pour Google ont un format spécifique
    contents = normalize_messages(
        request.messages, 
        Provider.GOOGLE
    )
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": request.temperature,
            "topK": 40,
            "topP": request.top_p,
            "maxOutputTokens": request.max_tokens,
            "responseMimeType": "text/plain"
        }
    }
    
    if request.system_message:
        payload["systemInstruction"] = {
            "role": "user",
            "parts": [{"text": request.system_message}]
        }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # L'API key est passée directement dans l'URL
    api_key = api_keys.get_key(Provider.GOOGLE)
    if request.stream:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:streamGenerateContent?key={api_key}"
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:generateContent?key={api_key}"
    
    return url, headers, payload

def transform_qwen_request(request: ChatCompletionRequest):
    normalized_messages = normalize_messages(
        request.messages, 
        Provider.QWEN,
        request.system_message
    )
    
    payload = {
        "model": request.model,
        "messages": normalized_messages,
        "temperature": request.temperature,
        "stream": request.stream,
        "max_tokens": request.max_tokens
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_keys.get_key(Provider.QWEN)}"
    }
    
    return "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions", headers, payload

def transform_ai21_request(request: ChatCompletionRequest):
    normalized_messages = normalize_messages(
        request.messages, 
        Provider.AI21,
        request.system_message
    )
    
    payload = {
        "model": request.model,
        "messages": normalized_messages,
        "temperature": request.temperature,
        "stream": request.stream,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
        "n": 1,
        "stop": request.stop or [],
        "response_format": {"type": "text"}
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_keys.get_key(Provider.AI21)}"
    }
    
    return "https://api.ai21.com/studio/v1/chat/completions", headers, payload

def transform_perplexity_request(request: ChatCompletionRequest):
    normalized_messages = normalize_messages(
        request.messages, 
        Provider.PERPLEXITY,
        request.system_message
    )
    
    # Assurons-nous que frequency_penalty est strictement positif
    freq_penalty = max(0.01, request.frequency_penalty or 0.01)
    
    payload = {
        "model": request.model,
        "messages": normalized_messages,
        "temperature": request.temperature,
        "stream": request.stream,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
        "top_k": 40,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": freq_penalty  # Valeur positive garantie
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_keys.get_key(Provider.PERPLEXITY)}"
    }
    
    return "https://api.perplexity.ai/chat/completions", headers, payload

def transform_deepinfra_request(request: ChatCompletionRequest):
    normalized_messages = normalize_messages(
        request.messages, 
        Provider.DEEPINFRA,
        request.system_message
    )
    
    payload = {
        "model": request.model,
        "messages": normalized_messages,
        "temperature": request.temperature,
        "stream": request.stream,
        "max_tokens": request.max_tokens
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_keys.get_key(Provider.DEEPINFRA)}"
    }
    
    return "https://api.deepinfra.com/v1/openai/chat/completions", headers, payload

# Fonction pour traiter les chunks de données en streaming
def process_stream_line(line, provider):
    if not line or not line.strip():
        return None
    
    try:
        # Si la ligne commence par "data: ", extraire la partie JSON
        if line.startswith("data: "):
            line = line[6:].strip()
            
            # Vérifier si c'est [DONE]
            if line == "[DONE]":
                return {"done": True}
            
            # Essayer de parser en tant que JSON
            try:
                data = json.loads(line)
                return {"data": data, "done": False}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from line: {line}")
                return {"data": {"text": line}, "done": False}
        
        # Si c'est un événement SSE, analyser le format
        elif "event:" in line:
            event_match = re.search(r'event:\s+(\w+)', line)
            data_match = re.search(r'data:\s+(.+)', line)
            
            if event_match and data_match:
                event_type = event_match.group(1)
                data_str = data_match.group(1)
                
                try:
                    data = json.loads(data_str)
                    return {"event": event_type, "data": data, "done": False}
                except json.JSONDecodeError:
                    return {"event": event_type, "data": {"text": data_str}, "done": False}
        
        # Si le format n'est pas reconnu, traiter comme texte brut
        return {"data": {"text": line}, "done": False}
    
    except Exception as e:
        logger.error(f"Error processing stream line: {str(e)}")
        return {"data": {"error": str(e)}, "done": False}

# Fonction pour convertir les réponses de streaming au format OpenAI
def format_stream_chunk_as_openai(chunk_data, provider, timestamp=None):
    if not timestamp:
        timestamp = int(time.time())
    
    if provider == Provider.OPENAI:
        # Pour OpenAI, le format est déjà correct
        return chunk_data
    
    try:
        # Format de base pour tous les providers
        formatted_chunk = {
            "id": f"{provider}_{timestamp}",
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": f"{provider}_model",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": None
                }
            ]
        }
        
        # Extraction du contenu selon le provider
        if provider == Provider.ANTHROPIC:
            if "type" in chunk_data:
                if chunk_data["type"] == "content_block_delta":
                    delta_text = chunk_data.get("delta", {}).get("text_delta", {}).get("text", "")
                    formatted_chunk["choices"][0]["delta"]["content"] = delta_text
                elif chunk_data["type"] == "message_stop":
                    formatted_chunk["choices"][0]["finish_reason"] = "stop"
        
        elif provider == Provider.MISTRAL:
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    formatted_chunk["choices"][0]["delta"]["content"] = delta["content"]
                if "finish_reason" in chunk_data["choices"][0]:
                    formatted_chunk["choices"][0]["finish_reason"] = chunk_data["choices"][0]["finish_reason"]
                if "model" in chunk_data:
                    formatted_chunk["model"] = chunk_data["model"]
                if "id" in chunk_data:
                    formatted_chunk["id"] = chunk_data["id"]
        
        elif provider == Provider.COHERE:
            if "type" in chunk_data:
                if chunk_data["type"] == "content-delta":
                    delta_text = chunk_data.get("delta", {}).get("message", {}).get("content", {}).get("text", "")
                    formatted_chunk["choices"][0]["delta"]["content"] = delta_text
                elif chunk_data["type"] == "message-end":
                    formatted_chunk["choices"][0]["finish_reason"] = "stop"
        
        elif provider == Provider.DEEPSEEK:
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    formatted_chunk["choices"][0]["delta"]["content"] = delta["content"]
                if "finish_reason" in chunk_data["choices"][0]:
                    formatted_chunk["choices"][0]["finish_reason"] = chunk_data["choices"][0]["finish_reason"]
                if "model" in chunk_data:
                    formatted_chunk["model"] = chunk_data["model"]
                if "id" in chunk_data:
                    formatted_chunk["id"] = chunk_data["id"]
        
        elif provider == Provider.XAI:
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    formatted_chunk["choices"][0]["delta"]["content"] = delta["content"]
                if "finish_reason" in chunk_data["choices"][0]:
                    formatted_chunk["choices"][0]["finish_reason"] = chunk_data["choices"][0]["finish_reason"]
                if "model" in chunk_data:
                    formatted_chunk["model"] = chunk_data["model"]
                if "id" in chunk_data:
                    formatted_chunk["id"] = chunk_data["id"]
        
        elif provider == Provider.GOOGLE:
            if "candidates" in chunk_data:
                for candidate in chunk_data["candidates"]:
                    if "content" in candidate and "parts" in candidate["content"]:
                        content_text = "".join([part.get("text", "") for part in candidate["content"]["parts"] if "text" in part])
                        formatted_chunk["choices"][0]["delta"]["content"] = content_text
                    if "finishReason" in candidate and candidate["finishReason"] == "STOP":
                        formatted_chunk["choices"][0]["finish_reason"] = "stop"
        
        elif provider == Provider.QWEN:
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    formatted_chunk["choices"][0]["delta"]["content"] = delta["content"]
                if "finish_reason" in chunk_data["choices"][0]:
                    formatted_chunk["choices"][0]["finish_reason"] = chunk_data["choices"][0]["finish_reason"]
                if "model" in chunk_data:
                    formatted_chunk["model"] = chunk_data["model"]
                if "id" in chunk_data:
                    formatted_chunk["id"] = chunk_data["id"]
        
        elif provider == Provider.AI21:
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    formatted_chunk["choices"][0]["delta"]["content"] = delta["content"]
                if "finish_reason" in chunk_data["choices"][0]:
                    formatted_chunk["choices"][0]["finish_reason"] = chunk_data["choices"][0]["finish_reason"]
                if "id" in chunk_data:
                    formatted_chunk["id"] = chunk_data["id"]
        
        elif provider == Provider.PERPLEXITY:
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    formatted_chunk["choices"][0]["delta"]["content"] = delta["content"]
                if "finish_reason" in chunk_data["choices"][0]:
                    formatted_chunk["choices"][0]["finish_reason"] = chunk_data["choices"][0]["finish_reason"]
                if "model" in chunk_data:
                    formatted_chunk["model"] = chunk_data["model"]
                if "id" in chunk_data:
                    formatted_chunk["id"] = chunk_data["id"]
        
        elif provider == Provider.DEEPINFRA:
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    formatted_chunk["choices"][0]["delta"]["content"] = delta["content"]
                if "finish_reason" in chunk_data["choices"][0]:
                    formatted_chunk["choices"][0]["finish_reason"] = chunk_data["choices"][0]["finish_reason"]
                if "model" in chunk_data:
                    formatted_chunk["model"] = chunk_data["model"]
                if "id" in chunk_data:
                    formatted_chunk["id"] = chunk_data["id"]
        
        # Format par défaut pour le texte simple
        elif "text" in chunk_data:
            formatted_chunk["choices"][0]["delta"]["content"] = chunk_data["text"]
        
        return formatted_chunk
        
    except Exception as e:
        logger.error(f"Error formatting stream chunk: {str(e)}")
        # Format de secours avec message d'erreur
        return {
            "id": f"error_{timestamp}",
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": f"{provider}_model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"Error: {str(e)}"},
                    "finish_reason": "error"
                }
            ]
        }

# Fonction pour normaliser les réponses non-streaming
def normalize_response(response_data, provider, model=None):
    try:
        timestamp = int(time.time())
        
        if provider == Provider.OPENAI:
            # Pour OpenAI, vérifier si c'est un modèle spécial (o1, o3)
            if model in OPENAI_SPECIAL_MODELS:
                # Les modèles spéciaux ont une structure différente à normaliser
                normalized = {
                    "id": response_data.get("id", f"openai_{timestamp}"),
                    "object": "chat.completion",
                    "created": response_data.get("created", timestamp),
                    "model": response_data.get("model", model),
                    "choices": [],
                    "usage": response_data.get("usage", {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    })
                }
                
                # Traiter les choix
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    for idx, choice in enumerate(response_data["choices"]):
                        message = choice.get("message", {})
                        normalized_choice = {
                            "index": idx,
                            "message": {
                                "role": message.get("role", "assistant"),
                                "content": message.get("content", "")
                            },
                            "finish_reason": choice.get("finish_reason", "stop")
                        }
                        normalized["choices"].append(normalized_choice)
                
                return normalized
            else:
                # Pour les modèles OpenAI standard, le format est déjà correct
                return response_data
        
        # Format de base pour la réponse normalisée
        normalized = {
            "id": f"{provider}_{timestamp}",
            "object": "chat.completion",
            "created": timestamp,
            "model": f"{provider}_model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ""
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        # Extraction des informations selon le provider
        if provider == Provider.ANTHROPIC:
            if "content" in response_data:
                content_text = ""
                for block in response_data["content"]:
                    if block.get("type") == "text":
                        content_text += block.get("text", "")
                
                normalized["choices"][0]["message"]["content"] = content_text
                
            if "id" in response_data:
                normalized["id"] = response_data["id"]
                
            if "model" in response_data:
                normalized["model"] = response_data["model"]
                
            if "usage" in response_data:
                normalized["usage"] = {
                    "prompt_tokens": response_data["usage"].get("input_tokens", 0),
                    "completion_tokens": response_data["usage"].get("output_tokens", 0),
                    "total_tokens": response_data["usage"].get("input_tokens", 0) + response_data["usage"].get("output_tokens", 0)
                }
        
        elif provider == Provider.MISTRAL:
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0]:
                    normalized["choices"][0]["message"] = response_data["choices"][0]["message"]
                
                if "finish_reason" in response_data["choices"][0]:
                    normalized["choices"][0]["finish_reason"] = response_data["choices"][0]["finish_reason"]
            
            if "id" in response_data:
                normalized["id"] = response_data["id"]
                
            if "model" in response_data:
                normalized["model"] = response_data["model"]
                
            if "usage" in response_data:
                normalized["usage"] = response_data["usage"]
        
        elif provider == Provider.COHERE:
            if "message" in response_data and "content" in response_data["message"]:
                if isinstance(response_data["message"]["content"], list):
                    content_text = ""
                    for content_item in response_data["message"]["content"]:
                        if content_item.get("type") == "text":
                            content_text += content_item.get("text", "")
                    normalized["choices"][0]["message"]["content"] = content_text
                else:
                    normalized["choices"][0]["message"]["content"] = response_data["message"]["content"]
            
            if "id" in response_data:
                normalized["id"] = response_data["id"]
                
            if "model" in response_data:
                normalized["model"] = response_data["model"]
        
        elif provider == Provider.DEEPSEEK:
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0]:
                    normalized["choices"][0]["message"] = response_data["choices"][0]["message"]
                
                if "finish_reason" in response_data["choices"][0]:
                    normalized["choices"][0]["finish_reason"] = response_data["choices"][0]["finish_reason"]
            
            if "id" in response_data:
                normalized["id"] = response_data["id"]
                
            if "model" in response_data:
                normalized["model"] = response_data["model"]
                
            if "usage" in response_data:
                normalized["usage"] = response_data["usage"]
        
        elif provider == Provider.XAI:
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0]:
                    normalized["choices"][0]["message"] = response_data["choices"][0]["message"]
                
                if "finish_reason" in response_data["choices"][0]:
                    normalized["choices"][0]["finish_reason"] = response_data["choices"][0]["finish_reason"]
            
            if "id" in response_data:
                normalized["id"] = response_data["id"]
                
            if "model" in response_data:
                normalized["model"] = response_data["model"]
                
            if "usage" in response_data:
                normalized["usage"] = response_data["usage"]
        
        elif provider == Provider.GOOGLE:
            if "candidates" in response_data:
                content_text = ""
                for candidate in response_data["candidates"]:
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "text" in part:
                                content_text += part["text"]
                normalized["choices"][0]["message"]["content"] = content_text
            
            if "modelVersion" in response_data:
                normalized["model"] = response_data["modelVersion"]
                
            if "usageMetadata" in response_data:
                normalized["usage"] = {
                    "prompt_tokens": response_data["usageMetadata"].get("promptTokenCount", 0),
                    "completion_tokens": response_data["usageMetadata"].get("candidatesTokenCount", 0),
                    "total_tokens": response_data["usageMetadata"].get("totalTokenCount", 0)
                }
        
        elif provider == Provider.QWEN:
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0]:
                    normalized["choices"][0]["message"] = response_data["choices"][0]["message"]
                
                if "finish_reason" in response_data["choices"][0]:
                    normalized["choices"][0]["finish_reason"] = response_data["choices"][0]["finish_reason"]
            
            if "id" in response_data:
                normalized["id"] = response_data["id"]
                
            if "model" in response_data:
                normalized["model"] = response_data["model"]
                
            if "usage" in response_data:
                normalized["usage"] = response_data["usage"]
        
        elif provider == Provider.AI21:
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0]:
                    normalized["choices"][0]["message"] = response_data["choices"][0]["message"]
                
                if "finish_reason" in response_data["choices"][0]:
                    normalized["choices"][0]["finish_reason"] = response_data["choices"][0]["finish_reason"]
            
            if "id" in response_data:
                normalized["id"] = response_data["id"]
                
            if "usage" in response_data:
                normalized["usage"] = response_data["usage"]
        
        elif provider == Provider.PERPLEXITY:
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0]:
                    normalized["choices"][0]["message"] = response_data["choices"][0]["message"]
                
                if "finish_reason" in response_data["choices"][0]:
                    normalized["choices"][0]["finish_reason"] = response_data["choices"][0]["finish_reason"]
            
            if "id" in response_data:
                normalized["id"] = response_data["id"]
                
            if "model" in response_data:
                normalized["model"] = response_data["model"]
                
            if "usage" in response_data:
                normalized["usage"] = response_data["usage"]
        
        elif provider == Provider.DEEPINFRA:
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0]:
                    normalized["choices"][0]["message"] = response_data["choices"][0]["message"]
                
                if "finish_reason" in response_data["choices"][0]:
                    normalized["choices"][0]["finish_reason"] = response_data["choices"][0]["finish_reason"]
            
            if "id" in response_data:
                normalized["id"] = response_data["id"]
                
            if "model" in response_data:
                normalized["model"] = response_data["model"]
                
            if "usage" in response_data:
                normalized["usage"] = response_data["usage"]
        
        return normalized
        
    except Exception as e:
        logger.error(f"Error normalizing response: {str(e)}")
        return {
            "error": f"Failed to normalize response: {str(e)}",
            "original_response": response_data
        }

# Fonction pour traiter la requête en fonction du provider
def process_request(request: ChatCompletionRequest):
    transform_functions = {
        Provider.OPENAI: transform_openai_request,
        Provider.ANTHROPIC: transform_anthropic_request,
        Provider.MISTRAL: transform_mistral_request,
        Provider.COHERE: transform_cohere_request,
        Provider.DEEPSEEK: transform_deepseek_request,
        Provider.XAI: transform_xai_request,
        Provider.GOOGLE: transform_google_request,
        Provider.QWEN: transform_qwen_request,
        Provider.AI21: transform_ai21_request,
        Provider.PERPLEXITY: transform_perplexity_request,
        Provider.DEEPINFRA: transform_deepinfra_request
    }
    
    if request.provider not in transform_functions:
        raise HTTPException(status_code=400, detail=f"Provider {request.provider} not supported")
    
    # Vérifier si le modèle est disponible pour ce provider
    if request.model not in MODELS.get(request.provider, []):
        raise HTTPException(
            status_code=400, 
            detail=f"Model {request.model} not available for provider {request.provider}. Available models: {MODELS.get(request.provider)}"
        )
    
    # Vérifier les contraintes spéciales pour les modèles OpenAI spéciaux
    if request.provider == Provider.OPENAI and request.model in OPENAI_SPECIAL_MODELS:
        # Les modèles spéciaux ne supportent pas le streaming
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail=f"Streaming is not supported for model {request.model}. Please set stream=false."
            )
        
        # Les modèles spéciaux ne supportent que temperature=1.0
        if request.temperature != 1.0:
            logger.warning(f"Model {request.model} only supports temperature=1.0. Ignoring provided temperature value.")
    
    return transform_functions[request.provider](request)

# Endpoint pour le chat avec streaming et non-streaming
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        url, headers, payload = process_request(request)
        
        # Si le modèle est un modèle spécial d'OpenAI, on force le streaming à False
        if request.provider == Provider.OPENAI and request.model in OPENAI_SPECIAL_MODELS:
            request.stream = False
        
        if request.stream:
            async def stream_response():
                try:
                    with requests.post(url, headers=headers, json=payload, stream=True) as response:
                        if response.status_code != 200:
                            error_msg = f"Error from {request.provider}: {response.text}"
                            logger.error(error_msg)
                            try:
                                error_json = response.json()
                                if "error" in error_json:
                                    error_detail = error_json["error"].get("message", "Unknown error")
                                    yield f"data: {json.dumps({'error': error_detail})}\n\n"
                                else:
                                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                            except:
                                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                            return
                        
                        # Pour Google, qui ne suit pas exactement le même format de streaming
                        if request.provider == Provider.GOOGLE:
                            try:
                                response_json = response.json()
                                for chunk in response_json:
                                    formatted_chunk = format_stream_chunk_as_openai(chunk, request.provider)
                                    yield f"data: {json.dumps(formatted_chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                return
                            except Exception as e:
                                logger.error(f"Error processing Google response: {str(e)}")
                                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                                return
                        
                        # Pour les autres providers
                        buffer = ""
                        for chunk in response.iter_lines():
                            if chunk:
                                chunk_text = chunk.decode('utf-8')
                                
                                # Pour les réponses qui peuvent inclure plusieurs lignes JSON
                                if request.provider in [Provider.COHERE, Provider.ANTHROPIC]:
                                    buffer += chunk_text + "\n"
                                    if chunk_text.strip() in ["{", "}", ""] or "event:" in chunk_text:
                                        continue
                                    
                                    # Traiter le buffer comme un événement SSE complet
                                    processed = process_stream_line(buffer, request.provider)
                                    buffer = ""
                                    
                                    if processed:
                                        if processed.get("done", False):
                                            yield "data: [DONE]\n\n"
                                            continue
                                        
                                        data = processed.get("data", {})
                                        formatted_chunk = format_stream_chunk_as_openai(data, request.provider)
                                        yield f"data: {json.dumps(formatted_chunk)}\n\n"
                                else:
                                    # Traitement standard ligne par ligne
                                    processed = process_stream_line(chunk_text, request.provider)
                                    
                                    if processed:
                                        if processed.get("done", False):
                                            yield "data: [DONE]\n\n"
                                            continue
                                        
                                        data = processed.get("data", {})
                                        formatted_chunk = format_stream_chunk_as_openai(data, request.provider)
                                        yield f"data: {json.dumps(formatted_chunk)}\n\n"
                        
                        # Assurer que [DONE] est envoyé à la fin
                        yield "data: [DONE]\n\n"
                except Exception as e:
                    error_msg = f"Error processing stream request: {str(e)}"
                    logger.error(error_msg)
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
            
            return StreamingResponse(stream_response(), media_type="text/event-stream")
        else:
            # Requête non-streaming
            try:
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code != 200:
                    error_msg = f"Error from {request.provider}: {response.text}"
                    logger.error(error_msg)
                    try:
                        error_json = response.json()
                        if "error" in error_json:
                            error_detail = error_json["error"].get("message", "Unknown error")
                            status_code = error_json["error"].get("code", response.status_code)
                            raise HTTPException(status_code=status_code, detail=f"API error: {error_detail}")
                    except json.JSONDecodeError:
                        pass
                    raise HTTPException(status_code=response.status_code, detail=error_msg)
                
                # Normaliser la réponse selon le format unifié
                normalized_response = normalize_response(response.json(), request.provider, request.model)
                return normalized_response
                
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error preparing request: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
