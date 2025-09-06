#!/bin/bash
set -euo pipefail

echo "=== Quick TripoSR Fix ==="
echo "Installing missing torchmcubes and diffusers..."

# Install diffusers first (easier)
echo "1. Installing diffusers..."
pip install diffusers

# Install torchmcubes from source
echo "2. Installing torchmcubes from source..."
cd /tmp
rm -rf torchmcubes 2>/dev/null || true
git clone https://github.com/tatsy/torchmcubes.git
cd torchmcubes
pip install -e .

echo "3. Testing TripoSR after fixes..."
cd ~/TripoSR
python3 -c "
try:
    import torchmcubes
    print('✓ torchmcubes now available')
    
    import diffusers
    print('✓ diffusers now available')
    
    from tsr.system import TSR
    print('✓ TSR system now available')
    
    print('✅ TripoSR should now work!')
    
except ImportError as e:
    print(f'✗ Still missing: {e}')
"

echo "4. Testing TripoSR CLI..."
python3 run.py --help | head -10

echo ""
echo "=== Fix Complete ==="
echo "Try generating a 3D object:"
echo "  cd ~/fashion3d"
echo "  python3 test_api.py"
