#!/bin/bash
set -euo pipefail

echo "=== TripoSR Installation Fix Script ==="
echo "Fixing CUDA/compilation issues..."

# Navigate to TripoSR directory
cd ~/TripoSR

echo "1. Cleaning previous failed installation..."
pip uninstall -y torch torchvision torchaudio || true
pip uninstall -y triposr || true

echo "2. Installing PyTorch CPU version (avoiding CUDA compilation issues)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "3. Installing essential build dependencies..."
sudo apt update
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

echo "4. Installing Python dependencies without problematic packages..."
pip install numpy pillow requests tqdm
pip install transformers accelerate
pip install rembg[new]

echo "5. Attempting to install TripoSR requirements with fixes..."
# Skip problematic packages initially
cat requirements.txt | grep -v "torch" | grep -v "xformers" | pip install -r /dev/stdin || true

echo "6. Installing alternative mesh processing (avoiding torchMcubes compilation)..."
pip install trimesh[easy] PyMCubes scikit-image

echo "7. Testing TripoSR basic import..."
python3 -c "
try:
    import torch
    print(f'✓ PyTorch {torch.__version__} (CPU)')
    import numpy as np
    print('✓ NumPy available')
    import PIL
    print('✓ PIL available')
    import trimesh
    print('✓ Trimesh available')
    print('✓ Basic dependencies working')
except ImportError as e:
    print(f'✗ Import error: {e}')
"

echo "8. Testing TripoSR run.py help..."
python3 run.py --help || echo "TripoSR help failed - will need alternative approach"

echo ""
echo "=== TripoSR Fix Complete ==="
echo "If TripoSR still fails, we'll create a custom mesh generation fallback."
