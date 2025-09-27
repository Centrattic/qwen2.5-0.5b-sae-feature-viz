"""
Utility script to download all required models and SAE
"""
import subprocess
import sys
from pathlib import Path
import logging

from src.config import BASE_MODEL_NAME, MISALIGNED_MODELS, SAE_MODEL_PATH

logger = logging.getLogger(__name__)

def download_model(model_name: str, cache_dir: str = "./models") -> None:
    """Download a model using huggingface-cli"""
    try:
        cmd = [
            "huggingface-cli", "download", 
            model_name, 
            "--local-dir", f"{cache_dir}/{model_name.replace('/', '_')}"
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"Downloaded {model_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {model_name}: {e}")
        raise

def download_sae(sae_path: str = "./sae_model") -> None:
    """Download SAE model"""
    try:
        cmd = [
            "huggingface-cli", "download", 
            SAE_MODEL_PATH, 
            "--local-dir", sae_path
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"Downloaded SAE model to {sae_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download SAE model: {e}")
        raise

def main():
    """Download all required models"""
    logging.basicConfig(level=logging.INFO)
    
    # Create directories
    Path("./models").mkdir(exist_ok=True)
    Path("./sae_model").mkdir(exist_ok=True)
    
    # Download base model
    logger.info("Downloading base model...")
    download_model(BASE_MODEL_NAME)
    
    # Download misaligned models
    logger.info("Downloading misaligned models...")
    for model in MISALIGNED_MODELS:
        download_model(model)
    
    # Download SAE model
    logger.info("Downloading SAE model...")
    download_sae()
    
    logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    main()
