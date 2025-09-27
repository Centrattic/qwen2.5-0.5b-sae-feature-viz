#!/usr/bin/env python3
"""
Data collection script for SAE Feature Visualization Tool
Pre-computes activations and SAE latents for all models and questions
"""
import torch
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import DEFAULT_QUESTIONS, MISALIGNED_MODELS, BASE_MODEL_NAME, TARGET_LAYER
from src.models.loader import ModelLoader
from src.sae.loader import SAELoader
from simple_cache import SimpleActivationCache

logger = logging.getLogger(__name__)


class DataCollector:

    def __init__(self,
                 sae_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_loader = ModelLoader(device)
        self.sae_loader = SAELoader(sae_path, device)
        self.cache = SimpleActivationCache()

    def collect_all_data(self) -> None:
        """Collect activations and SAE latents for all models and questions"""
        all_models = [BASE_MODEL_NAME] + MISALIGNED_MODELS

        for model_name in all_models:
            print(f"Processing model: {model_name}")
            self._process_model(model_name, DEFAULT_QUESTIONS)

    def _process_model(self, model_name: str, questions: List[str]) -> None:
        """Process a single model with all questions"""
        print(f"Loading model: {model_name}")
        self.model_loader.load_model(model_name)

        # Get model cache with correct dimensions
        model = self.model_loader.get_model(model_name)
        hidden_size = model.config.hidden_size

        for question in questions:
            print(f"Processing question: {question[:50]}...")
            self._process_question(model_name, question, hidden_size)

    def _process_question(self, model_name: str, question: str,
                          hidden_size: int) -> None:
        """Process a single question for a model"""
        # Generate response
        response = self.model_loader.generate_response(model_name, question)

        # Create hash for caching
        question_hash = hashlib.sha1(question.encode()).hexdigest()

        # Extract activations
        activations = self._extract_activations(model_name, question, response)

        # Get actual sequence length
        seq_len = activations.shape[1]

        # Get model cache
        model_cache = self.cache.get_model_cache(model_name)

        # Extract SAE latents
        sae_latents = self.sae_loader.encode(activations)

        # Store in cache
        self._store_data(model_name, question_hash, question, response,
                         activations, sae_latents, model_cache)

        print(f"Stored data for {model_name} - {question_hash[:8]}...")

    def _extract_activations(self, model_name: str, question: str,
                             response: str) -> torch.Tensor:
        """Extract activations from target layer for question + response only"""
        model = self.model_loader.get_model(model_name)
        tokenizer = self.model_loader.get_tokenizer(model_name)

        # Use chat template to get the full conversation
        messages = [{
            "role": "user",
            "content": question
        }, {
            "role": "assistant",
            "content": response
        }]

        # Apply chat template to get the full formatted conversation
        formatted_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

        # Tokenize the full conversation
        full_tokens = tokenizer.tokenize(formatted_text)

        # Tokenize just question + response to find the offset
        question_response_text = question + " " + response
        qr_tokens = tokenizer.tokenize(question_response_text)

        # Find where the question+response starts in the full conversation
        # (skip system prompt tokens)
        qr_start_idx = None
        for i in range(len(full_tokens) - len(qr_tokens) + 1):
            if full_tokens[i:i + len(qr_tokens)] == qr_tokens:
                qr_start_idx = i
                break

        if qr_start_idx is None:
            # Fallback: use full conversation
            qr_start_idx = 0
            qr_tokens = full_tokens

        # Run model on full conversation to get activations
        inputs = tokenizer(formatted_text, return_tensors="pt").to(self.device)

        # Hook to extract activations from target layer
        activations = None

        def hook_fn(module, input, output):
            nonlocal activations
            activations = output.detach()

        # Qwen2 model structure: model.layers[layer].post_attention_layernorm
        target_layer = model.model.layers[
            TARGET_LAYER].post_attention_layernorm
        hook = target_layer.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            _ = model(**inputs)

        # Remove hook
        hook.remove()

        # Extract only the activations for question + response part
        if qr_start_idx > 0:
            # Skip system prompt activations
            return activations[:,
                               qr_start_idx:qr_start_idx + len(qr_tokens), :]
        else:
            # Use all activations if we couldn't find the offset
            return activations

    def _store_data(self, model_name: str, question_hash: str, question: str,
                    response: str, activations: torch.Tensor,
                    sae_latents: torch.Tensor, model_cache) -> None:
        """Store all data in caches"""
        # Store activation data
        model_cache.add_activation_data(question_hash, question, response,
                                        activations)

        # Store SAE latents (convert bfloat16 to float32 for numpy compatibility)
        sae_latents_np = sae_latents.cpu().float().numpy()
        model_cache.add_sae_data(question_hash, sae_latents_np)

        # Generate and store visualization data
        self._generate_visualization_data(model_name, question_hash, question,
                                          response, sae_latents_np)

    def _generate_visualization_data(self, model_name: str, question_hash: str,
                                     question: str, response: str,
                                     sae_latents: np.ndarray) -> None:
        """Generate pre-computed visualization data for fast frontend access"""
        import json

        # Create visualization data directory
        viz_dir = Path("./visualization_data")
        viz_dir.mkdir(exist_ok=True)

        model_viz_dir = viz_dir / model_name.replace("/", "_")
        model_viz_dir.mkdir(exist_ok=True)

        # Get tokenizer to tokenize the full text using proper chat template
        tokenizer = self.model_loader.get_tokenizer(model_name)

        # Tokenize just the question + response (no system prompt)
        # This matches what we actually want to visualize
        full_text = question + " " + response
        tokens = tokenizer.tokenize(full_text)

        # Generate feature activations for all features
        feature_data = {}

        # Store SAE latents in JSON for static frontend access
        feature_data = {
            "sae_latents": sae_latents.tolist(),  # [1, seq_len, n_features]
            "tokens": tokens,
            "n_features": sae_latents.shape[-1],
            "seq_len": sae_latents.shape[1]
        }

        # Store question data
        question_data = {
            "question": question,
            "response": response,
            "question_hash": question_hash,
            "tokens": tokens,
            "features": feature_data
        }

        # Save to JSON file
        output_file = model_viz_dir / f"{question_hash}.json"
        with open(output_file, 'w') as f:
            json.dump(question_data, f, indent=2)

        print(f"Generated visualization data: {output_file}")


def main():
    """Main data collection script"""
    logging.basicConfig(level=logging.INFO)

    print("SAE Feature Visualization Data Collection")
    print("=" * 50)

    # Check if SAE model exists
    sae_path = "./sae_model"
    sae_weights_path = Path(
        sae_path
    ) / "Qwen2.5-0.5B-Instruct_blocks.8.ln2.hook_normalized_28672.pt"

    if not sae_weights_path.exists():
        print("SAE model not found. Please download it first:")
        print(
            "huggingface-cli download rootxhacker/Qwen-2.5-0.5B-instruct-SAE --local-dir ./sae_model"
        )
        return

    # Initialize collector
    collector = DataCollector(sae_path)

    # Collect all data
    print("Starting data collection...")
    collector.collect_all_data()

    print("Data collection completed!")
    print("You can now start the backend with: python start_backend.py")


if __name__ == "__main__":
    main()
