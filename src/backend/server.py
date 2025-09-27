from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import hashlib
import logging

from src.config import DEFAULT_QUESTIONS, MISALIGNED_MODELS, BASE_MODEL_NAME, TARGET_LAYER
from src.models.loader import ModelLoader
from src.sae.loader import SAELoader
from src.cache.sae_cache import SAEFeatureCache
from activation_cache import SingleLayerActivationCache

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="SAE Feature Visualization API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model_loader = None
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
    top_features: List[Dict[str, Any]]
    success: bool

@app.on_event("startup")
async def startup_event():
    """Initialize models and caches on startup"""
    global model_loader, sae_loader, cache
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize components
    model_loader = ModelLoader(device)
    sae_loader = SAELoader("./sae_model", device)
    cache = SAEFeatureCache()
    
    # Load base model
    logger.info("Loading base model...")
    model_loader.load_model(BASE_MODEL_NAME)
    
    logger.info("Backend initialized successfully")

@app.get("/")
async def root():
    return {"message": "SAE Feature Visualization API"}

@app.get("/models")
async def get_models():
    """Get list of available models"""
    all_models = [BASE_MODEL_NAME] + MISALIGNED_MODELS
    return {"models": all_models}

@app.get("/questions")
async def get_questions():
    """Get list of default questions"""
    return {"questions": DEFAULT_QUESTIONS}

@app.post("/process_question", response_model=QuestionResponse)
async def process_question(request: QuestionRequest):
    """Process a new question and extract activations/SAE latents"""
    try:
        # Load model if not already loaded
        if not model_loader.models.get(request.model_name):
            model_loader.load_model(request.model_name)
        
        # Generate response
        response = model_loader.generate_response(request.model_name, request.question)
        
        # Create hash
        question_hash = hashlib.sha1(request.question.encode()).hexdigest()
        
        # Extract activations and SAE latents
        activations = _extract_activations(request.model_name, request.question)
        sae_latents = sae_loader.encode(activations)
        
        # Store in cache
        model_cache = cache.get_model_cache(request.model_name)
        _store_data(request.model_name, question_hash, request.question, 
                   response, activations, sae_latents, model_cache)
        
        return QuestionResponse(
            response=response,
            question_hash=question_hash,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_feature_activations", response_model=FeatureResponse)
async def get_feature_activations(request: FeatureRequest):
    """Get activations for a specific SAE feature"""
    try:
        # Get SAE data
        sae_data = cache.get_sae_data(request.model_name, request.question_hash)
        if sae_data is None:
            raise HTTPException(status_code=404, detail="Question not found in cache")
        
        # Get feature activations
        feature_activations = sae_data[:, request.feature_idx]
        
        # Get tokens (simplified - would need tokenizer for actual tokens)
        tokens = [f"token_{i}" for i in range(len(feature_activations))]
        
        return FeatureResponse(
            activations=feature_activations.tolist(),
            tokens=tokens,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error getting feature activations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_top_features", response_model=TopFeaturesResponse)
async def get_top_features(request: TopFeaturesRequest):
    """Get top-k most active features for a question"""
    try:
        # Get top features
        result = cache.get_top_features_for_question(
            request.model_name, 
            request.question_hash, 
            request.top_k
        )
        
        if result is None:
            raise HTTPException(status_code=404, detail="Question not found in cache")
        
        top_values, top_indices = result
        
        # Format response
        top_features = []
        for i in range(len(top_values)):
            token_features = []
            for j in range(request.top_k):
                token_features.append({
                    "feature_idx": int(top_indices[i, j]),
                    "activation": float(top_values[i, j])
                })
            top_features.append(token_features)
        
        return TopFeaturesResponse(
            top_features=top_features,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error getting top features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _extract_activations(model_name: str, prompt: str) -> torch.Tensor:
    """Extract activations from target layer"""
    model = model_loader.get_model(model_name)
    tokenizer = model_loader.get_tokenizer(model_name)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model_loader.device)
    
    # Hook to extract activations from target layer
    activations = None
    
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    # Register hook on target layer
    target_layer = model.model.layers[TARGET_LAYER].ln2
    hook = target_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove hook
    hook.remove()
    
    return activations

def _store_data(model_name: str, question_hash: str, question: str, 
               response: str, activations: torch.Tensor, sae_latents: torch.Tensor,
               model_cache: SingleLayerActivationCache) -> None:
    """Store all data in caches"""
    # Store activations in activation cache
    model_cache.add_batch(
        hashes=[question_hash],
        authors=["system"],
        timestamps=[str(torch.tensor(0))],  # Dummy timestamp
        contents=[f"{question}\n{response}"],
        activations=activations
    )
    
    # Store SAE latents
    sae_latents_np = sae_latents.cpu().numpy()
    cache.add_sae_data(model_name, question_hash, sae_latents_np)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
