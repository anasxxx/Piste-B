#!/usr/bin/env python3
"""
Test TripoSR directly after fixes to verify real 3D generation
"""
import subprocess
import sys
import time
from pathlib import Path

def test_triposr_direct():
    print("Testing TripoSR Direct 3D Generation")
    print("=" * 50)
    
    # Test image
    test_image = Path("inputs/test_image.jpeg")
    if not test_image.exists():
        print(f"✗ Test image not found: {test_image}")
        return False
    
    print(f"✓ Testing with: {test_image}")
    
    # Output directory
    output_dir = Path("outputs/triposr_direct")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TripoSR command
    triposr_dir = Path.home() / "TripoSR"
    run_script = triposr_dir / "run.py"
    
    if not run_script.exists():
        print(f"✗ TripoSR script not found: {run_script}")
        return False
    
    # Test TripoSR CLI directly
    cmd = [
        sys.executable, str(run_script),
        str(test_image),
        "--output-dir", str(output_dir),
        "--mc-resolution", "256"
    ]
    
    print(f"Running TripoSR command:")
    print(f"  {' '.join(cmd)}")
    print()
    
    try:
        start_time = time.time()
        
        # Run TripoSR
        result = subprocess.run(
            cmd,
            cwd=triposr_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )
        
        elapsed = time.time() - start_time
        
        print(f"TripoSR completed in {elapsed:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check if files were generated
        print("\nChecking generated files...")
        
        generated_files = list(output_dir.rglob("*.obj"))
        
        if generated_files:
            print("✅ SUCCESS! TripoSR generated 3D files:")
            for file in generated_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  📁 {file} ({size_mb:.2f} MB)")
            
            # Compare with fallback
            fallback_files = list(Path("outputs").glob("**/mesh.obj"))
            if fallback_files:
                fallback_size = fallback_files[-1].stat().st_size / (1024 * 1024)
                triposr_size = generated_files[0].stat().st_size / (1024 * 1024)
                
                print(f"\n📊 Quality Comparison:")
                print(f"  Fallback generator: {fallback_size:.2f} MB")
                print(f"  TripoSR (real 3D):  {triposr_size:.2f} MB")
                
                if triposr_size > fallback_size:
                    improvement = ((triposr_size - fallback_size) / fallback_size) * 100
                    print(f"  ✓ TripoSR is {improvement:.1f}% more detailed!")
            
            return True
            
        else:
            print("✗ No OBJ files generated")
            print("TripoSR may have failed or produced different output formats")
            
            # List all generated files
            all_files = list(output_dir.rglob("*"))
            if all_files:
                print("Generated files:")
                for file in all_files:
                    if file.is_file():
                        print(f"  - {file}")
            
            return False
    
    except subprocess.TimeoutExpired:
        print("✗ TripoSR timed out (over 5 minutes)")
        return False
    except Exception as e:
        print(f"✗ TripoSR execution failed: {e}")
        return False

def test_api_with_triposr():
    """Test the API now that TripoSR should work"""
    print("\nTesting API with TripoSR fixed...")
    
    try:
        import requests
        
        # Test health endpoint
        response = requests.get("http://127.0.0.1:8002/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("API Health:")
            print(f"  TripoSR CLI exists: {health.get('has_runpy')}")
            print(f"  Strategies: {health.get('strategies', [])}")
            
            # Test generation
            test_image = Path("inputs/test_image.jpeg")
            if test_image.exists():
                print("\nTesting API generation...")
                
                with open(test_image, 'rb') as f:
                    files = {'file': ('test_image.jpeg', f, 'image/jpeg')}
                    data = {
                        'bake_texture': 'true',
                        'texture_resolution': '1024',
                        'mc_resolution': '256'
                    }
                    
                    response = requests.post(
                        "http://127.0.0.1:8002/generate",
                        files=files,
                        data=data,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        method = result.get('note', 'unknown')
                        print(f"✓ API generation successful: {method}")
                        
                        if 'fallback' not in method:
                            print("🎉 SUCCESS! TripoSR is now working through the API!")
                        else:
                            print("⚠ Still using fallback - TripoSR CLI may need more fixes")
                    else:
                        print(f"✗ API generation failed: {response.status_code}")
            
        else:
            print("✗ API not running")
            
    except requests.exceptions.ConnectionError:
        print("⚠ API not running - start with: python3 start_api.py")
    except Exception as e:
        print(f"✗ API test failed: {e}")

if __name__ == "__main__":
    print("TripoSR Real 3D Generation Test")
    print("=" * 60)
    
    # Test 1: Direct TripoSR
    success = test_triposr_direct()
    
    # Test 2: API integration
    test_api_with_triposr()
    
    if success:
        print("\n🎉 CONGRATULATIONS!")
        print("TripoSR is now generating REAL 3D objects!")
        print("No more flat extrusions - you have true 3D reconstruction!")
    else:
        print("\n❌ TripoSR still needs fixes")
        print("Alternative: Use the volumetric generator for real 3D objects")
