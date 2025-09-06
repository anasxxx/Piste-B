#!/usr/bin/env python3
import sys
import subprocess
import time
import requests
from pathlib import Path

def test_fashion3d_api():
    print("=== Testing Fashion3D API ===")
    
    # Test 1: Health endpoint
    try:
        response = requests.get("http://127.0.0.1:8002/health", timeout=5)
        if response.status_code == 200:
            print("✓ Fashion3D API is running")
            health_data = response.json()
            print(f"  Artifacts dir: {health_data.get('artifacts')}")
            print(f"  TripoSR CLI exists: {health_data.get('has_runpy')}")
            print(f"  TripoSR path: {health_data.get('triposr_path')}")
            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Fashion3D API not running on port 8002")
        return False
    except Exception as e:
        print(f"✗ Error testing API: {e}")
        return False

def test_triposr_micro_api():
    print("\n=== Testing TripoSR Micro API ===")
    try:
        response = requests.get("http://127.0.0.1:8001/health", timeout=5)
        if response.status_code == 200:
            print("✓ TripoSR micro API is running")
            return True
        else:
            print(f"✗ TripoSR API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ TripoSR micro API not running on port 8001")
        return False
    except Exception as e:
        print(f"✗ Error testing TripoSR API: {e}")
        return False

def test_with_sample_image():
    print("\n=== Testing with Sample Image ===")
    
    # Find the test image
    test_image = Path("inputs/df3d_tex.png")
    if not test_image.exists():
        print(f"✗ Test image not found: {test_image}")
        return False
    
    print(f"✓ Found test image: {test_image}")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': ('df3d_tex.png', f, 'image/png')}
            data = {
                'bake_texture': 'true',
                'texture_resolution': '512',  # Lower for testing
                'mc_resolution': '128',       # Lower for testing
                'supersample': '1'
            }
            
            print("Sending generation request...")
            response = requests.post(
                "http://127.0.0.1:8002/generate", 
                files=files, 
                data=data, 
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Generation completed: {result.get('ok')}")
                print(f"  Job dir: {result.get('job_dir')}")
                print(f"  Assets: {result.get('assets')}")
                print(f"  Note: {result.get('note')}")
                if result.get('log_tail'):
                    print("  Last few log lines:")
                    for line in result.get('log_tail', [])[-5:]:
                        print(f"    {line}")
                return result.get('ok', False)
            else:
                print(f"✗ Generation failed with status {response.status_code}")
                print(response.text)
                return False
                
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        return False

if __name__ == "__main__":
    print("Fashion3D API Test Suite")
    print("========================")
    
    # Test Fashion3D API
    api_ok = test_fashion3d_api()
    
    # Test TripoSR micro API
    triposr_ok = test_triposr_micro_api()
    
    # If Fashion3D API is running, test generation
    if api_ok:
        generation_ok = test_with_sample_image()
    else:
        print("\nSkipping generation test - API not running")
        generation_ok = False
    
    print(f"\n=== Test Results ===")
    print(f"Fashion3D API: {'✓' if api_ok else '✗'}")
    print(f"TripoSR API: {'✓' if triposr_ok else '✗'}")
    print(f"Generation: {'✓' if generation_ok else '✗'}")
    
    if not api_ok:
        print("\nTo start Fashion3D API:")
        print("  cd ~/fashion3d")
        print("  bash scripts/run_api.sh")
    
    if not triposr_ok:
        print("\nTripoSR micro API not running (will use CLI fallback)")
