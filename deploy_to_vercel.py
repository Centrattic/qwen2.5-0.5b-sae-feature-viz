#!/usr/bin/env python3
"""
Deploy script to prepare data for Vercel deployment
"""
import shutil
from pathlib import Path

def deploy_to_vercel():
    """Copy visualization data to frontend directory for Vercel deployment"""
    
    # Source and destination paths
    viz_data_src = Path("./visualization_data")
    frontend_dest = Path("./src/deployment/vercel-frontend/visualization_data")
    
    # Remove old data if exists
    if frontend_dest.exists():
        shutil.rmtree(frontend_dest)
    
    # Copy visualization data to frontend
    if viz_data_src.exists():
        shutil.copytree(viz_data_src, frontend_dest)
        print(f"✅ Copied visualization data to {frontend_dest}")
    else:
        print("❌ No visualization data found. Run 'python collect_data.py' first.")
        return False
    
    # Copy cache data to frontend (for backend access)
    cache_src = Path("./cache")
    cache_dest = Path("./src/deployment/vercel-frontend/cache")
    
    if cache_dest.exists():
        shutil.rmtree(cache_dest)
    
    if cache_src.exists():
        shutil.copytree(cache_src, cache_dest)
        print(f"✅ Copied cache data to {cache_dest}")
    
    print("\n🚀 Ready for Vercel deployment!")
    print("1. Push to GitHub")
    print("2. Connect to Vercel")
    print("3. Deploy!")
    
    return True

if __name__ == "__main__":
    deploy_to_vercel()
