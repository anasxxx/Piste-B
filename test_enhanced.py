#!/usr/bin/env python3
"""
Enhanced test with improved depth parameters for better 3D results
"""
import requests
import time
from pathlib import Path

def test_enhanced_generation():
    print("Testing Enhanced 3D Generation Parameters...")
    print("=" * 50)
    
    test_image = Path("inputs/test_image.jpeg")
    api_url = "http://127.0.0.1:8002/generate"
    
    # Test multiple parameter sets
    test_configs = [
        {
            "name": "High Detail",
            "params": {
                'bake_texture': 'true',
                'texture_resolution': '1024',
                'mc_resolution': '512',  # Higher resolution
                'supersample': '2'       # More sampling
            }
        },
        {
            "name": "Enhanced Depth", 
            "params": {
                'bake_texture': 'true',
                'texture_resolution': '2048',  # Higher texture
                'mc_resolution': '384',
                'supersample': '1'
            }
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n🔧 Test {i+1}: {config['name']}")
        print("Parameters:", config['params'])
        
        try:
            with open(test_image, 'rb') as f:
                files = {'file': ('test_image.jpeg', f, 'image/jpeg')}
                
                start_time = time.time()
                response = requests.post(api_url, files=files, data=config['params'], timeout=300)
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print(f"✓ Success in {elapsed:.2f}s")
                    print(f"  Job: {result.get('job_dir')}")
                    
                    assets = result.get('assets', {})
                    if assets.get('mesh_obj'):
                        mesh_path = Path(assets['mesh_obj'])
                        if mesh_path.exists():
                            size_mb = mesh_path.stat().st_size / (1024*1024)
                            print(f"  Mesh: {size_mb:.2f} MB")
                        
                else:
                    print(f"✗ Failed: {response.status_code}")
                    
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_enhanced_generation()
