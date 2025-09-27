#!/usr/bin/env python3
"""
SAE Feature Visualization Tool - FastAPI app with ngrok integration
"""
import os
import sys
import gc
import json
import asyncio
import signal
import atexit
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import logging

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pydantic import BaseModel

from src.config import DEFAULT_QUESTIONS, MISALIGNED_MODELS, BASE_MODEL_NAME, TARGET_LAYER, SAE_FEATURES
from src.sae.loader import SAELoader
from src.models.loader import ModelLoader
from simple_cache import SimpleActivationCache
# Removed old activation cache import

load_dotenv()

ALLOWED_ORIGINS = [
    "https://qwen2.5-0.5b-sae-feature-viz.vercel.app",
    "http://localhost:8000",
]

ngrok_listener = None

model_manager = None
sae_loader = None
cache = None


class QuestionRequest(BaseModel):
    question: str
    model_name: str


class QuestionResponse(BaseModel):
    response: str
    question_hash: str
    success: bool


class FeatureRequest(BaseModel):
    model_name: str
    question_hash: str
    feature_idx: int


class FeatureResponse(BaseModel):
    activations: List[float]
    tokens: List[str]
    success: bool


class TopFeaturesRequest(BaseModel):
    model_name: str
    question_hash: str
    top_k: int = 10


class TopFeaturesResponse(BaseModel):
    top_features: List[Dict[str, float]]
    success: bool


async def create_ngrok_tunnel():
    global ngrok_listener

    from pyngrok import ngrok

    auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    if not auth_token:
        print("NGROK_AUTH_TOKEN not set. Tunnel will be limited.")
    else:
        ngrok.set_auth_token(auth_token)
        print("ngrok auth token set")

    print("Creating ngrok tunnel to localhost:8000...")
    ngrok_listener = ngrok.connect(8000, "http")

    public_url = ngrok_listener.public_url
    print(f"ngrok tunnel created: {public_url}")

    vercel_config_path = PROJECT_ROOT / "src" / "deployment" / "vercel-frontend" / "config.js"
    if vercel_config_path.exists():
        with open(vercel_config_path, 'r') as f:
            content = f.read()

        import re
        new_content = re.sub(r"apiBase:\s*['\"][^'\"]*['\"]",
                             f"apiBase: '{public_url}'", content)

        with open(vercel_config_path, 'w') as f:
            f.write(new_content)

        print("Auto-updated Vercel config.js with new ngrok URL")

    print(
        f"Your Vercel config.js has been updated with: apiBase: '{public_url}'"
    )
    return public_url


async def cleanup_ngrok():
    global ngrok_listener

    if ngrok_listener:
        from pyngrok import ngrok

        tunnel_url = ngrok_listener.public_url
        ngrok.disconnect(tunnel_url)
        print(f"ngrok tunnel closed: {tunnel_url}")

        ngrok.kill()
        print("ngrok processes killed")

    ngrok_listener = None


def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, shutting down...")
    if ngrok_listener:
        asyncio.run(cleanup_ngrok())
    sys.exit(0)


atexit.register(lambda: asyncio.run(cleanup_ngrok())
                if ngrok_listener else None)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def verify_origin(origin: str = Header(None),
                        referer: str = Header(None)):
    request_origin = origin or referer

    if not request_origin:
        return True

    for allowed_origin in ALLOWED_ORIGINS:
        if allowed_origin in request_origin:
            return True

    raise HTTPException(status_code=403, detail="Origin not allowed")


app = FastAPI(title="SAE Feature Visualization API", version="1.0.0")

frontend_dir = PROJECT_ROOT / "src" / "deployment" / "vercel-frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global model_manager, sae_loader, cache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize cache only - no model loading
    cache = SimpleActivationCache()
    model_manager = None
    sae_loader = None

    print("Backend initialized - serving cached data only")
    print(
        "Run 'python collect_data.py' to pre-compute activations and SAE latents"
    )


@app.get("/health")
def health():
    return {"ok": True, "mode": "cache_only", "server": "running"}


@app.get("/")
def index() -> FileResponse:
    frontend_path = PROJECT_ROOT / "src" / "deployment" / "vercel-frontend" / "index.html"
    return FileResponse(str(frontend_path))


@app.get("/config.js")
def get_config() -> FileResponse:
    config_path = PROJECT_ROOT / "src" / "deployment" / "vercel-frontend" / "config.js"
    return FileResponse(str(config_path))


@app.get("/models")
async def get_models():
    all_models = [BASE_MODEL_NAME] + MISALIGNED_MODELS
    return {"models": all_models}


@app.get("/questions")
async def get_questions():
    return {"questions": DEFAULT_QUESTIONS}


# Removed process_question endpoint - only serving cached data


@app.post("/get_feature_activations", response_model=FeatureResponse)
async def get_feature_activations(request: FeatureRequest,
                                  origin: str = Depends(verify_origin)):
    # Load pre-computed visualization data
    viz_data = _load_visualization_data(request.model_name,
                                        request.question_hash)
    if viz_data is None:
        raise HTTPException(
            status_code=404,
            detail=
            "Visualization data not found. Please run collect_data.py first.")

    # Get feature activations from pre-computed data
    features = viz_data.get("features", {})
    feature_key = str(request.feature_idx)

    if feature_key not in features:
        raise HTTPException(
            status_code=404,
            detail=
            f"Feature {request.feature_idx} not found in pre-computed data")

    feature_data = features[feature_key]
    activations = feature_data["activations"]
    tokens = [f"token_{i}" for i in range(len(activations))]

    return FeatureResponse(activations=activations,
                           tokens=tokens,
                           success=True)


@app.post("/get_top_features", response_model=TopFeaturesResponse)
async def get_top_features(request: TopFeaturesRequest,
                           origin: str = Depends(verify_origin)):
    # Load pre-computed visualization data
    viz_data = _load_visualization_data(request.model_name,
                                        request.question_hash)
    if viz_data is None:
        raise HTTPException(
            status_code=404,
            detail=
            "Visualization data not found. Please run collect_data.py first.")

    # Get top features from pre-computed data
    features = viz_data.get("features", {})
    feature_scores = [(int(fid), data["max_activation"])
                      for fid, data in features.items()]
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    top_k = min(request.top_k, len(feature_scores))
    top_features = feature_scores[:top_k]

    # Format for response
    formatted_features = []
    for feature_idx, activation in top_features:
        formatted_features.append({
            "feature_idx": feature_idx,
            "activation": activation
        })

    return TopFeaturesResponse(top_features=formatted_features, success=True)


def _load_visualization_data(model_name: str,
                             question_hash: str) -> Optional[Dict]:
    """Load pre-computed visualization data"""
    import json
    from pathlib import Path

    viz_dir = Path("./visualization_data")
    model_viz_dir = viz_dir / model_name.replace("/", "_")
    data_file = model_viz_dir / f"{question_hash}.json"

    if not data_file.exists():
        return None

    try:
        with open(data_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _extract_activations(model_name: str, prompt: str) -> torch.Tensor:
    model = model_manager.get_model(model_name)
    tokenizer = model_manager.get_tokenizer(model_name)

    inputs = tokenizer(prompt, return_tensors="pt").to(model_manager.device)

    activations = None

    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.detach()

    # Qwen2 model structure: model.layers[layer].post_attention_layernorm
    target_layer = model.model.layers[TARGET_LAYER].post_attention_layernorm
    hook = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(**inputs)

    hook.remove()

    return activations


def _store_data(model_name: str, question_hash: str, question: str,
                response: str, activations: torch.Tensor,
                sae_latents: torch.Tensor,
                model_cache: SingleLayerActivationCache) -> None:
    model_cache.add_batch(hashes=[question_hash],
                          authors=["system"],
                          timestamps=[str(torch.tensor(0))],
                          contents=[f"{question}\n{response}"],
                          activations=activations)

    # Only store SAE data if available
    if sae_latents is not None:
        sae_latents_np = sae_latents.cpu().numpy()
        cache.add_sae_data(model_name, question_hash, sae_latents_np)


if __name__ == "__main__":
    print("Starting SAE Feature Visualization App...")

    async def start_with_ngrok():
        print("Creating ngrok tunnel...")
        tunnel_url = await create_ngrok_tunnel()
        if tunnel_url:
            print(f"ngrok tunnel ready: {tunnel_url}")
        else:
            print("ngrok tunnel creation failed, continuing without tunnel")

        print("Starting FastAPI server on localhost:8000...")
        config = uvicorn.Config("src.deployment.app:app",
                                host="0.0.0.0",
                                port=8000,
                                reload=False)
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(start_with_ngrok())