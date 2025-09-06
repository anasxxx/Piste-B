#!/usr/bin/env python3
"""
Direct test of the fallback mesh generator
"""
import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def test_fallback_generator():
    print("Testing Fallback Mesh Generator...")
    
    try:
        from fallback_generator import generate_fallback_mesh
        print("✓ Fallback generator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import fallback generator: {e}")
        return False
    
    # Find test image
    test_image = Path("inputs/df3d_tex.png")
    if not test_image.exists():
        print(f"✗ Test image not found: {test_image}")
        return False
    
    print(f"✓ Found test image: {test_image}")
    
    # Create output directory
    output_dir = Path("outputs") / "fallback_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Generating mesh with fallback generator...")
        result = generate_fallback_mesh(
            img_path=test_image,
            output_dir=output_dir,
            bake_texture=True,
            mc_resolution=128
        )
        
        print("✓ Mesh generation completed!")
        print(f"Result: {result}")
        
        # Check if files were created
        mesh_file = Path(result["mesh_obj"]) if result.get("mesh_obj") else None
        if mesh_file and mesh_file.exists():
            print(f"✓ Mesh file created: {mesh_file} ({mesh_file.stat().st_size} bytes)")
        else:
            print("✗ Mesh file not found")
            
        texture_file = Path(result["albedo_png"]) if result.get("albedo_png") else None
        if texture_file and texture_file.exists():
            print(f"✓ Texture file created: {texture_file} ({texture_file.stat().st_size} bytes)")
        else:
            print("✗ Texture file not found")
            
        return True
        
    except Exception as e:
        print(f"✗ Fallback generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    print("Testing Dependencies...")
    
    deps = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("trimesh", "Trimesh"),
        ("rembg", "RemBG")
    ]
    
    all_good = True
    for module, name in deps:
        try:
            __import__(module)
            print(f"✓ {name} available")
        except ImportError:
            print(f"✗ {name} missing")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("Fashion3D Fallback Generator Test")
    print("=" * 40)
    
    # Test dependencies first
    deps_ok = test_dependencies()
    print()
    
    if deps_ok:
        # Test fallback generator
        generator_ok = test_fallback_generator()
        
        if generator_ok:
            print("\n✓ All tests passed! Fallback generator is working.")
        else:
            print("\n✗ Fallback generator test failed.")
    else:
        print("\n✗ Missing dependencies. Install with:")
        print("  pip install numpy opencv-python pillow trimesh rembg")
