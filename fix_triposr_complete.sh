#!/bin/bash
set -euo pipefail

echo "=== Complete TripoSR Installation Fix ==="
echo "This will properly install TripoSR for true 3D reconstruction"

# Navigate to TripoSR directory
cd ~/TripoSR

echo "1. Cleaning all previous installations..."
pip uninstall -y triposr tsr torch torchvision torchaudio xformers || true

echo "2. Installing PyTorch with CUDA support..."
# Try CUDA first, fallback to CPU
if nvidia-smi &>/dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA, installing CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "3. Installing build dependencies..."
sudo apt update
sudo apt install -y git cmake build-essential libgl1-mesa-dev

echo "4. Installing Python dependencies from source..."
pip install numpy pillow tqdm transformers accelerate
pip install diffusers rembg trimesh pymeshlab

echo "5. Cloning and installing torchmcubes from source..."
cd /tmp
git clone https://github.com/tatsy/torchmcubes.git
cd torchmcubes
pip install -e .

echo "6. Installing TripoSR from source..."
cd ~/TripoSR
pip install -e .

echo "7. Testing TripoSR installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
import torchmcubes
print('✓ TorchMCubes available')
try:
    from tsr.system import TSR
    print('✓ TSR system available')
    print('✓ TripoSR successfully installed!')
except ImportError as e:
    print(f'✗ TSR import failed: {e}')
"

echo "8. Testing TripoSR CLI..."
python3 run.py --help

echo ""
echo "=== TripoSR Installation Complete ==="
echo "If successful, TripoSR will create REAL 3D objects, not flat extrusions!"
