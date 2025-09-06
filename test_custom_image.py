#!/usr/bin/env python3
"""
Test Fashion3D API with custom image
"""
import requests
import time
from pathlib import Path
import json

def test_custom_image():
    print("Testing Fashion3D API with custom image...")
    print("=" * 50)
    
    # Image path
    test_image = Path("inputs/test_image.jpeg")
    
    if not test_image.exists():
        print(f"✗ Test image not found: {test_image}")
        return False
    
    print(f"✓ Found custom test image: {test_image}")
    print(f"  File size: {test_image.stat().st_size} bytes")
    
    # API endpoint
    api_url = "http://127.0.0.1:8002/generate"
    
    try:
        print(f"\nSending generation request to {api_url}...")
        print("Parameters:")
        print("  - bake_texture: true")
        print("  - texture_resolution: 1024")
        print("  - mc_resolution: 256") 
        print("  - supersample: 1")
        
        # Prepare the request
        with open(test_image, 'rb') as f:
            files = {'file': ('test_image.jpeg', f, 'image/jpeg')}
            data = {
                'bake_texture': 'true',
                'texture_resolution': '1024',
                'mc_resolution': '256',
                'supersample': '1'
            }
            
            start_time = time.time()
            
            # Make the request
            response = requests.post(
                api_url, 
                files=files, 
                data=data, 
                timeout=300  # 5 minutes timeout
            )
            
            elapsed_time = time.time() - start_time
            
        print(f"\nResponse received in {elapsed_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "="*50)
            print("🎉 GENERATION SUCCESSFUL! 🎉")
            print("="*50)
            
            print(f"✓ Success: {result.get('ok')}")
            print(f"✓ Processing time: {result.get('elapsed_sec')} seconds")
            print(f"✓ Job directory: {result.get('job_dir')}")
            print(f"✓ Method used: {result.get('note')}")
            
            assets = result.get('assets', {})
            print(f"\n📁 Generated Files:")
            
            if assets.get('mesh_obj'):
                mesh_path = Path(assets['mesh_obj'])
                if mesh_path.exists():
                    print(f"  ✓ 3D Mesh: {mesh_path} ({mesh_path.stat().st_size:,} bytes)")
                else:
                    print(f"  ✗ 3D Mesh: {mesh_path} (not found)")
            
            if assets.get('albedo_png'):
                texture_path = Path(assets['albedo_png']) 
                if texture_path.exists():
                    print(f"  ✓ Texture: {texture_path} ({texture_path.stat().st_size:,} bytes)")
                else:
                    print(f"  ✗ Texture: {texture_path} (not found)")
                    
            if assets.get('input_png'):
                input_path = Path(assets['input_png'])
                if input_path.exists():
                    print(f"  ✓ Input Copy: {input_path} ({input_path.stat().st_size:,} bytes)")
                else:
                    print(f"  ✗ Input Copy: {input_path} (not found)")
            
            # Show last few log lines
            log_tail = result.get('log_tail', [])
            if log_tail:
                print(f"\n📝 Generation Log (last {len(log_tail)} lines):")
                for line in log_tail[-10:]:  # Show last 10 lines
                    print(f"    {line}")
            
            # Show any errors from other strategies
            errors = result.get('error_summary', {})
            if any(errors.values()):
                print(f"\n⚠️  Other Strategy Errors (as expected):")
                if errors.get('micro_api'):
                    print(f"  - Micro API: {errors['micro_api'].get('error', 'Failed')}")
                if errors.get('cli'):
                    print(f"  - CLI: Connection/module errors (expected)")
            
            return True
            
        else:
            print(f"\n✗ Generation failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n✗ Could not connect to Fashion3D API")
        print("Make sure the API is running with: python3 start_api.py")
        return False
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Fashion3D Custom Image Test")
    print("Testing with: inputs/test_image.jpeg")
    print()
    
    success = test_custom_image()
    
    if success:
        print("\n🎯 SUCCESS! Your custom image has been converted to 3D!")
        print("\nYou can find the generated files in the job directory shown above.")
        print("The .obj file can be opened in Blender, MeshLab, or any 3D viewer.")
    else:
        print("\n❌ Test failed. Check the logs above for details.")
