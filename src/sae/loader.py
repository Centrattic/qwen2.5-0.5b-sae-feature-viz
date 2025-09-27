import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from torch.serialization import safe_globals

logger = logging.getLogger(__name__)


class SAELoader:

    def __init__(self,
                 sae_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.sae_path = sae_path
        self.device = device
        self.sae_model: Optional[Any] = None
        self._load_sae()

    def _load_sae(self) -> None:
        """Load SAE model from path using allowlist approach"""
        logger.info(f"Loading SAE from: {self.sae_path}")

        # Load the real SAE file
        sae_weights_path = Path(
            self.sae_path
        ) / "Qwen2.5-0.5B-Instruct_blocks.8.ln2.hook_normalized_28672.pt"

        if not sae_weights_path.exists():
            raise FileNotFoundError(
                f"SAE weights not found at {sae_weights_path}")

        # Allowlist the sae_training.config.LanguageModelSAERunnerConfig class
        # Create a dummy class to allowlist
        Dummy = type("LanguageModelSAERunnerConfig", (), {})
        Dummy.__module__ = "sae_training.config"

        with safe_globals([Dummy]):
            sae_data = torch.load(sae_weights_path,
                                  map_location=self.device,
                                  weights_only=True)

        # Extract state_dict if it exists
        if isinstance(sae_data, dict) and "state_dict" in sae_data:
            sae_data = sae_data["state_dict"]

        # Extract SAE components and ensure they're all on the same device and dtype
        self.W_enc = sae_data["W_enc"].to(self.device, dtype=torch.bfloat16)
        self.b_enc = sae_data["b_enc"].to(self.device, dtype=torch.bfloat16)
        self.W_dec = sae_data["W_dec"].to(self.device, dtype=torch.bfloat16)
        self.b_dec = sae_data["b_dec"].to(self.device, dtype=torch.bfloat16)

        # Handle potential transpose issue
        if self.W_dec.ndim == 2 and self.W_dec.shape[0] != self.W_enc.shape[
                0] and self.W_dec.shape[1] == self.W_enc.shape[0]:
            self.W_dec = self.W_dec.T

        self.hidden_size = self.W_enc.shape[
            1]  # 896 (Qwen2.5-0.5B hidden size)
        self.n_features = self.W_enc.shape[0]  # 28672 (SAE features)

        # Debug: print shapes to verify
        logger.info(f"W_enc shape: {self.W_enc.shape}")
        logger.info(f"W_dec shape: {self.W_dec.shape}")
        logger.info(f"b_enc shape: {self.b_enc.shape}")
        logger.info(f"b_dec shape: {self.b_dec.shape}")

        logger.info(
            f"SAE loaded: {self.n_features} features, {self.hidden_size} hidden size"
        )

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode activations to SAE latents using einsum for efficiency"""
        # activations: (batch_size, seq_len, hidden_size)
        # Ensure activations are on the same device and dtype as SAE weights
        activations = activations.to(self.device, dtype=torch.bfloat16)
        return F.relu(
            torch.einsum("btd,df->btf", activations, self.W_enc) + self.b_enc)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode SAE latents back to activations using einsum for efficiency"""
        # latents: (batch_size, seq_len, n_features)
        return torch.einsum("btf,df->btd", latents, self.W_dec) + self.b_dec

    def get_feature_activations(self, latents: torch.Tensor,
                                feature_idx: int) -> torch.Tensor:
        """Get activations for a specific feature across all tokens"""
        return latents[:, :, feature_idx]

    def get_top_features(self,
                         latents: torch.Tensor,
                         top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k most active features for each token"""
        # latents: (batch_size, seq_len, n_features)
        values, indices = torch.topk(latents, top_k, dim=-1)
        return values, indices