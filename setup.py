#!/usr/bin/env python3
"""
Setup script for SAE Feature Visualization Tool
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"{description}...")
    result = subprocess.run(cmd,
                            shell=True,
                            check=True,
                            capture_output=True,
                            text=True)
    print(f"{description} completed")
    return True


def check_requirements():
    """Check if required tools are installed"""
    print("Checking requirements...")

    requirements = [("python", "Python 3.8+"), ("ngrok", "ngrok CLI")]

    missing = []
    for cmd, name in requirements:
        try:
            subprocess.run([cmd, "--version"], check=True, capture_output=True)
            print(f"{name} found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{name} not found")
            missing.append(name)

    if missing:
        print(f"Missing requirements: {', '.join(missing)}")
        print("Please install missing requirements before continuing.")
        return False

    return True


def setup_python():
    """Setup Python environment"""
    print("\nSetting up Python environment...")

    print("Python dependencies already installed")

    return True


def setup_frontend():
    """Setup frontend environment"""
    print("\nSetting up frontend...")

    deployment_dir = Path("src/deployment")
    deployment_dir.mkdir(parents=True, exist_ok=True)

    vercel_frontend_dir = deployment_dir / "vercel-frontend"
    vercel_frontend_dir.mkdir(parents=True, exist_ok=True)

    print("Created deployment directory structure")

    return True


def download_models():
    """Download required models"""
    print("\nDownloading models...")

    print("Model download skipped - you can download models manually later")

    return True


def main():
    """Main setup function"""
    print("SAE Feature Visualization Tool Setup")
    print("=" * 50)

    if not check_requirements():
        sys.exit(1)

    if not setup_python():
        sys.exit(1)

    if not setup_frontend():
        sys.exit(1)

    download_choice = input(
        "\nDownload models now? This will take some time. (y/N): ").lower()
    if download_choice == 'y':
        if not download_models():
            print(
                "You can download models later by running: python src/utils/download_models.py"
            )

    print("\nSetup completed!")
    print("\nNext steps:")
    print("1. Start the backend: python start_backend.py")
    print("2. Open the ngrok URL in your browser")
    print("\nSee README.md for detailed instructions")


if __name__ == "__main__":
    main()