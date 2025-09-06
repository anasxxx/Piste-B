#!/usr/bin/env python3
"""
Install TripoSR for Fashion3D project
"""
import os
import subprocess
import sys
from pathlib import Path

def install_triposr():
    print("🚀 Installing TripoSR...")
    
    # Create TripoSR directory in user home
    home_dir = Path.home()
    triposr_dir = home_dir / "TripoSR"
    
    print(f"📁 Installing to: {triposr_dir}")
    
    try:
        # Clone TripoSR repository
        print("📥 Cloning TripoSR repository...")
        result = subprocess.run([
            "git", "clone", 
            "https://github.com/VAST-AI-Research/TripoSR.git",
            str(triposr_dir)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Git clone failed: {result.stderr}")
            return False
        
        print("✅ TripoSR cloned successfully!")
        
        # Install TripoSR
        print("📦 Installing TripoSR dependencies...")
        os.chdir(triposr_dir)
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Installation failed: {result.stderr}")
            return False
        
        print("✅ TripoSR installed successfully!")
        
        # Check if run.py exists
        run_py = triposr_dir / "run.py"
        if run_py.exists():
            print(f"✅ Found run.py at: {run_py}")
            return True
        else:
            print(f"❌ run.py not found at: {run_py}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🎯 TripoSR Installation for Fashion3D")
    print("=" * 50)
    success = install_triposr()
    if success:
        print("\n🎉 TripoSR installation completed!")
        print("🔄 You can now test 3D generation with real models!")
    else:
        print("\n❌ TripoSR installation failed!")
