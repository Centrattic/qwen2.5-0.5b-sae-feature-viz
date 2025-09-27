#!/usr/bin/env python3
"""
Start script for the SAE Feature Visualization backend
"""
import subprocess
import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def main():
    """Start the backend server with ngrok"""
    logging.basicConfig(level=logging.INFO)

    print("Starting SAE Feature Visualization Backend")
    print("=" * 50)

    # Check if ngrok is installed
    try:
        subprocess.run(["ngrok", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ngrok not found. Please install ngrok first:")
        print("   https://ngrok.com/download")
        sys.exit(1)

    print("Starting FastAPI server...")
    subprocess.run([
        sys.executable, "-m", "uvicorn", "src.deployment.app:app", "--host",
        "0.0.0.0", "--port", "8000", "--reload"
    ])


if __name__ == "__main__":
    main()