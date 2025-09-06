#!/usr/bin/env python3
"""
Diagnose exactly what's wrong with TripoSR installation
"""
import sys
import subprocess
from pathlib import Path

def diagnose_triposr():
    print("TripoSR Installation Diagnosis")
    print("=" * 40)
    
    # Check if TripoSR directory exists
    triposr_dir = Path.home() / "TripoSR"
    print(f"TripoSR directory: {triposr_dir}")
    print(f"Exists: {triposr_dir.exists()}")
    
    if triposr_dir.exists():
        print(f"Contents: {list(triposr_dir.iterdir())}")
    
    # Check Python imports
    print("\nTesting Python imports...")
    
    modules_to_test = [
        "torch",
        "torchvision", 
        "torchaudio",
        "torchmcubes",
        "trimesh",
        "rembg",
        "transformers",
        "diffusers"
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
    
    # Check TripoSR specific modules
    print("\nTesting TripoSR modules...")
    
    try:
        sys.path.insert(0, str(triposr_dir))
        import tsr
        print("✓ tsr module available")
        
        try:
            from tsr.system import TSR
            print("✓ TSR system class available")
        except ImportError as e:
            print(f"✗ TSR system: {e}")
            
    except ImportError as e:
        print(f"✗ tsr module: {e}")
    
    # Check if we can run TripoSR
    print("\nTesting TripoSR CLI...")
    try:
        if triposr_dir.exists():
            run_py = triposr_dir / "run.py"
            if run_py.exists():
                result = subprocess.run([
                    sys.executable, str(run_py), "--help"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print("✓ TripoSR CLI works")
                    print("Help output preview:")
                    print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
                else:
                    print(f"✗ TripoSR CLI failed (code {result.returncode})")
                    print("Error:", result.stderr)
            else:
                print("✗ run.py not found")
        else:
            print("✗ TripoSR directory not found")
            
    except subprocess.TimeoutExpired:
        print("✗ TripoSR CLI timed out")
    except Exception as e:
        print(f"✗ TripoSR CLI error: {e}")
    
    # Check GPU availability
    print("\nGPU Information...")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
        else:
            print("Running on CPU")
    except ImportError:
        print("PyTorch not available")
    
    # Recommendations
    print("\n" + "=" * 40)
    print("RECOMMENDATIONS:")
    print("=" * 40)
    
    # Check what's missing and provide specific fixes
    try:
        import torchmcubes
        mcubes_ok = True
    except ImportError:
        mcubes_ok = False
        print("🔧 Install torchmcubes:")
        print("   cd /tmp && git clone https://github.com/tatsy/torchmcubes.git")
        print("   cd torchmcubes && pip install -e .")
    
    try:
        sys.path.insert(0, str(triposr_dir))
        import tsr
        tsr_ok = True
    except ImportError:
        tsr_ok = False
        print("🔧 Install TripoSR properly:")
        print("   cd ~/TripoSR && pip install -e .")
    
    if mcubes_ok and tsr_ok:
        print("✅ TripoSR should work! Try running the CLI test.")
    else:
        print("❌ TripoSR needs the fixes above before it will work.")
        print("🔄 Alternative: Use our volumetric generator for real 3D objects")

if __name__ == "__main__":
    diagnose_triposr()
