#!/usr/bin/env bash
set -euo pipefail

echo "=== Testing TripoSR Setup ==="

# Check if TripoSR directory exists
if [ -d ~/TripoSR ]; then
    echo "✓ TripoSR directory found"
    
    # Check if run.py exists
    if [ -f ~/TripoSR/run.py ]; then
        echo "✓ run.py script found"
        
        # Try to run TripoSR help
        cd ~/TripoSR
        echo "Testing TripoSR CLI help..."
        python3 run.py --help || echo "✗ TripoSR CLI help failed"
        
    else
        echo "✗ run.py script not found"
    fi
else
    echo "✗ TripoSR directory not found"
fi

echo ""
echo "=== Testing Fashion3D Environment ==="

# Check if fashion3d conda environment exists
if conda env list | grep -q fashion3d; then
    echo "✓ fashion3d conda environment found"
else
    echo "✗ fashion3d conda environment not found"
fi

# Test fashion3d dependencies
echo "Testing Fashion3D dependencies..."
python3 -c "
try:
    import fastapi, uvicorn, requests
    from PIL import Image
    print('✓ All Fashion3D dependencies available')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
"

echo ""
echo "=== Testing Complete ==="
