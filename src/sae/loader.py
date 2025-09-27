import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SAELoader:
    def __init__(self, sae_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.sae_path = sae_path
        self.device = device
        self.sae_model: Optional[Any] = None
        self._load_sae()
    
    def _load_sae(self) -> None:
        """Load SAE model from path"""
        logger.info(f"Loading SAE from: {self.sae_path}")
        
        # Load SAE weights
        sae_weights_path = Path(self.sae_path) / "Qwen2.5-0.5B-Instruct_blocks.8.ln2.hook_normalized_28672.pt"
        
        if not sae_weights_path.exists():
            raise FileNotFoundError(f"SAE weights not found at {sae_weights_path}")
        
        sae_data = torch.load(sae_weights_path, map_location=self.device)
        
        # Extract SAE components
        self.W_enc = sae_data['W_enc'].to(self.device)
        self.W_dec = sae_data['W_dec'].to(self.device)
        self.b_enc = sae_data['b_enc'].to(self.device)
        self.b_dec = sae_data['b_dec'].to(self.device)
        
        self.n_features = self.W_enc.shape[0]
        self.hidden_size = self.W_enc.shape[1]
        
        logger.info(f"SAE loaded: {self.n_features} features, {self.hidden_size} hidden size")
    
    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode activations to SAE latents"""
        # activations: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = activations.shape
        
        # Reshape for SAE encoding
        flat_acts = activations.view(-1, hidden_size)  # (batch_size * seq_len, hidden_size)
        
        # SAE encoding: x -> ReLU(W_enc @ x + b_enc)
        encoded = torch.relu(flat_acts @ self.W_enc.T + self.b_enc)
        
        # Reshape back
        return encoded.view(batch_size, seq_len, self.n_features)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode SAE latents back to activations"""
        batch_size, seq_len, n_features = latents.shape
        
        # Reshape for SAE decoding
        flat_latents = latents.view(-1, n_features)
        
        # SAE decoding: f -> W_dec @ f + b_dec
        decoded = flat_latents @ self.W_dec.T + self.b_dec
        
        # Reshape back
        return decoded.view(batch_size, seq_len, self.hidden_size)
    
    def get_feature_activations(self, latents: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """Get activations for a specific feature across all tokens"""
        return latents[:, :, feature_idx]
    
    def get_top_features(self, latents: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k most active features for each token"""
        # latents: (batch_size, seq_len, n_features)
        values, indices = torch.topk(latents, top_k, dim=-1)
        return values, indices
