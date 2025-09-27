import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleActivationCache:
    """Simplified activation cache for SAE feature visualization"""

    def __init__(self, cache_dir: Path = Path("./cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get_model_cache(self, model_name: str) -> 'ModelCache':
        """Get or create cache for a specific model"""
        model_cache_dir = self.cache_dir / model_name.replace("/", "_")
        model_cache_dir.mkdir(exist_ok=True)
        return ModelCache(model_cache_dir)

    def add_sae_data(self, model_name: str, question_hash: str,
                     sae_latents: np.ndarray):
        """Store SAE latents for a model and question"""
        model_cache = self.get_model_cache(model_name)
        model_cache.add_sae_data(question_hash, sae_latents)

    def get_sae_data(self, model_name: str,
                     question_hash: str) -> Optional[np.ndarray]:
        """Get SAE latents for a model and question"""
        model_cache = self.get_model_cache(model_name)
        return model_cache.get_sae_data(question_hash)


class ModelCache:
    """Cache for a specific model"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.activations_file = cache_dir / "activations.json"
        self.sae_data_file = cache_dir / "sae_data.json"

    def add_activation_data(self, question_hash: str, question: str,
                            response: str, activations: torch.Tensor):
        """Store activation data for a question"""
        # Convert activations to numpy and store metadata (convert bfloat16 to float32)
        activations_np = activations.cpu().float().numpy()

        # Load existing data
        if self.activations_file.exists():
            with open(self.activations_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Add new data
        data[question_hash] = {
            "question": question,
            "response": response,
            "activations_shape": list(activations_np.shape),
            "activations_file": f"activations_{question_hash}.npy"
        }

        # Save metadata
        with open(self.activations_file, 'w') as f:
            json.dump(data, f, indent=2)

        # Save activations as numpy file
        activations_path = self.cache_dir / f"activations_{question_hash}.npy"
        np.save(activations_path, activations_np)

        logger.info(f"Stored activation data for {question_hash}")

    def add_sae_data(self, question_hash: str, sae_latents: np.ndarray):
        """Store SAE latents for a question"""
        # Load existing SAE data
        if self.sae_data_file.exists():
            with open(self.sae_data_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Add new SAE data
        data[question_hash] = {
            "sae_latents_shape": list(sae_latents.shape),
            "sae_latents_file": f"sae_latents_{question_hash}.npy"
        }

        # Save metadata
        with open(self.sae_data_file, 'w') as f:
            json.dump(data, f, indent=2)

        # Save SAE latents as numpy file
        sae_path = self.cache_dir / f"sae_latents_{question_hash}.npy"
        np.save(sae_path, sae_latents)

        logger.info(f"Stored SAE data for {question_hash}")

    def get_sae_data(self, question_hash: str) -> Optional[np.ndarray]:
        """Get SAE latents for a question"""
        if not self.sae_data_file.exists():
            return None

        with open(self.sae_data_file, 'r') as f:
            data = json.load(f)

        if question_hash not in data:
            return None

        sae_file = self.cache_dir / data[question_hash]["sae_latents_file"]
        if not sae_file.exists():
            return None

        return np.load(sae_file)

    def get_activation_data(self, question_hash: str) -> Optional[Dict]:
        """Get activation data for a question"""
        if not self.activations_file.exists():
            return None

        with open(self.activations_file, 'r') as f:
            data = json.load(f)

        if question_hash not in data:
            return None

        return data[question_hash]
