#!/usr/bin/env python3
"""
Compare original vs enhanced mesh generation
"""
import sys
import os
from pathlib import Path
import time

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def compare_generators():
    print("Comparing Original vs Enhanced Mesh Generation")
    print("=" * 60)
    
    test_image = Path("inputs/test_image.jpeg")
    if not test_image.exists():
        print(f"✗ Test image not found: {test_image}")
        return
    
    print(f"✓ Testing with: {test_image}")
    print(f"  File size: {test_image.stat().st_size:,} bytes")
    
    # Test 1: Original Generator
    print(f"\n{'='*20} ORIGINAL GENERATOR {'='*20}")
    try:
        from fallback_generator import generate_fallback_mesh
        
        output_dir_orig = Path("outputs/comparison_original")
        
        start_time = time.time()
        result_orig = generate_fallback_mesh(
            img_path=test_image,
            output_dir=output_dir_orig,
            bake_texture=True,
            mc_resolution=256
        )
        orig_time = time.time() - start_time
        
        print(f"✓ Original generation completed in {orig_time:.2f}s")
        
        # Check file sizes
        mesh_orig = Path(result_orig["mesh_obj"])
        if mesh_orig.exists():
            orig_size = mesh_orig.stat().st_size
            print(f"  Mesh size: {orig_size:,} bytes ({orig_size/1024/1024:.2f} MB)")
        
    except Exception as e:
        print(f"✗ Original generator failed: {e}")
        result_orig = None
    
    # Test 2: Enhanced Generator
    print(f"\n{'='*20} ENHANCED GENERATOR {'='*20}")
    try:
        from enhanced_generator import EnhancedFallbackGenerator
        
        output_dir_enh = Path("outputs/comparison_enhanced")
        generator = EnhancedFallbackGenerator()
        
        start_time = time.time()
        result_enh = generator.generate_enhanced_mesh(
            img_path=test_image,
            output_dir=output_dir_enh,
            bake_texture=True,
            mc_resolution=256
        )
        enh_time = time.time() - start_time
        
        print(f"✓ Enhanced generation completed in {enh_time:.2f}s")
        
        # Check file sizes
        mesh_enh = Path(result_enh["mesh_obj"])
        if mesh_enh.exists():
            enh_size = mesh_enh.stat().st_size
            print(f"  Mesh size: {enh_size:,} bytes ({enh_size/1024/1024:.2f} MB)")
        
    except Exception as e:
        print(f"✗ Enhanced generator failed: {e}")
        result_enh = None
    
    # Compare Results
    print(f"\n{'='*25} COMPARISON {'='*25}")
    
    if result_orig and result_enh:
        mesh_orig = Path(result_orig["mesh_obj"])
        mesh_enh = Path(result_enh["mesh_obj"])
        
        if mesh_orig.exists() and mesh_enh.exists():
            orig_size = mesh_orig.stat().st_size
            enh_size = mesh_enh.stat().st_size
            
            print(f"Original mesh:  {orig_size:,} bytes")
            print(f"Enhanced mesh:  {enh_size:,} bytes")
            
            if enh_size > orig_size:
                improvement = ((enh_size - orig_size) / orig_size) * 100
                print(f"✓ Enhanced mesh is {improvement:.1f}% larger (more detailed)")
            elif enh_size < orig_size:
                reduction = ((orig_size - enh_size) / orig_size) * 100
                print(f"⚠ Enhanced mesh is {reduction:.1f}% smaller")
            else:
                print("→ Same file size")
        
        print(f"\nGenerated files:")
        print(f"📁 Original:  {result_orig['mesh_obj']}")
        print(f"📁 Enhanced:  {result_enh['mesh_obj']}")
        
        print(f"\n🎯 RECOMMENDATION:")
        print(f"Compare these files in a 3D viewer to see the difference!")
        print(f"The enhanced version should have:")
        print(f"  • Better depth variation")
        print(f"  • More surface detail") 
        print(f"  • Less flat appearance")
        print(f"  • Improved edge definition")
    
    else:
        print("⚠ Could not complete comparison due to errors")

if __name__ == "__main__":
    compare_generators()
