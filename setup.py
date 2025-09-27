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
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_requirements():
    """Check if required tools are installed"""
    print("🔍 Checking requirements...")
    
    requirements = [
        ("python", "Python 3.8+"),
        ("node", "Node.js 18+"),
        ("ngrok", "ngrok CLI"),
        ("huggingface-cli", "Hugging Face CLI")
    ]
    
    missing = []
    for cmd, name in requirements:
        try:
            subprocess.run([cmd, "--version"], check=True, capture_output=True)
            print(f"✅ {name} found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {name} not found")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing requirements: {', '.join(missing)}")
        print("Please install missing requirements before continuing.")
        return False
    
    return True

def setup_python():
    """Setup Python environment"""
    print("\n🐍 Setting up Python environment...")
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    return True

def setup_frontend():
    """Setup frontend environment"""
    print("\n⚛️ Setting up frontend...")
    
    # Change to frontend directory
    os.chdir("frontend")
    
    # Install Node.js dependencies
    if not run_command("npm install", "Installing Node.js dependencies"):
        return False
    
    # Create environment file
    env_content = """# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# For production, replace with your ngrok URL:
# NEXT_PUBLIC_API_URL=https://your-ngrok-url.ngrok.io
"""
    
    with open(".env.local", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env.local file")
    
    # Change back to root directory
    os.chdir("..")
    
    return True

def download_models():
    """Download required models"""
    print("\n📥 Downloading models...")
    
    # Run the download script
    if not run_command("python src/utils/download_models.py", "Downloading models"):
        print("⚠️ Model download failed. You can run this manually later.")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 SAE Feature Visualization Tool Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Setup Python
    if not setup_python():
        sys.exit(1)
    
    # Setup frontend
    if not setup_frontend():
        sys.exit(1)
    
    # Download models (optional)
    download_choice = input("\n📥 Download models now? This will take some time. (y/N): ").lower()
    if download_choice == 'y':
        if not download_models():
            print("⚠️ You can download models later by running: python src/utils/download_models.py")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Start the backend: python start_backend.py")
    print("2. Start the frontend: cd frontend && npm run dev")
    print("3. Open http://localhost:3000 in your browser")
    print("\n📚 See README.md for detailed instructions")

if __name__ == "__main__":
    main()
