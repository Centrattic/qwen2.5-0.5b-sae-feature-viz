import torch
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import logging

from src.config import DEFAULT_QUESTIONS, MISALIGNED_MODELS, BASE_MODEL_NAME, TARGET_LAYER
from src.models.loader import ModelLoader
from src.sae.loader import SAELoader
from src.cache.sae_cache import SAEFeatureCache
from activation_cache import SingleLayerActivationCache

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, sae_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_loader = ModelLoader(device)
        self.sae_loader = SAELoader(sae_path, device)
        self.cache = SAEFeatureCache()
        
    def collect_all_data(self) -> None:
        """Collect activations and SAE latents for all models and questions"""
        all_models = [BASE_MODEL_NAME] + MISALIGNED_MODELS
        
        for model_name in all_models:
            logger.info(f"Processing model: {model_name}")
            self._process_model(model_name, DEFAULT_QUESTIONS)
    
    def _process_model(self, model_name: str, questions: List[str]) -> None:
        """Process a single model with all questions"""
        # Load model
        self.model_loader.load_model(model_name)
        
        # Get model cache
        model_cache = self.cache.get_model_cache(model_name)
        
        for question in questions:
            logger.info(f"Processing question: {question[:50]}...")
            self._process_question(model_name, question, model_cache)
    
    def _process_question(self, model_name: str, question: str, model_cache: SingleLayerActivationCache) -> None:
        """Process a single question for a model"""
        # Generate response
        response = self.model_loader.generate_response(model_name, question)
        
        # Create hash for caching
        question_hash = hashlib.sha1(question.encode()).hexdigest()
        
        # Extract activations
        activations = self._extract_activations(model_name, question)
        
        # Extract SAE latents
        sae_latents = self.sae_loader.encode(activations)
        
        # Store in cache
        self._store_data(model_name, question_hash, question, response, activations, sae_latents, model_cache)
    
    def _extract_activations(self, model_name: str, prompt: str) -> torch.Tensor:
        """Extract activations from target layer"""
        model = self.model_loader.get_model(model_name)
        tokenizer = self.model_loader.get_tokenizer(model_name)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
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
    
    def _store_data(self, model_name: str, question_hash: str, question: str, 
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
        self.cache.add_sae_data(model_name, question_hash, sae_latents_np)
        
        logger.info(f"Stored data for {model_name} - {question_hash[:8]}...")

def main():
    """Main data collection script"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize collector
    sae_path = "./sae_model"  # Path to downloaded SAE
    collector = DataCollector(sae_path)
    
    # Collect all data
    collector.collect_all_data()
    
    logger.info("Data collection completed!")

if __name__ == "__main__":
    main()
