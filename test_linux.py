#!/usr/bin/env python3
"""
Test 3D generation in Linux/WSL environment
"""
import sys
import os
import subprocess
from pathlib import Path

def test_in_wsl():
    print("🐧 Testing in WSL Ubuntu environment...")
    
    # Image path (Windows path that WSL can access)
    image_path = "/mnt/c/Users/mahmo/OneDrive/Desktop/images.jpeg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"✅ Found image: {image_path}")
    
    # Test the API components
    try:
        from api.server import _save_upload_to_png, _now_jobdir, _try_micro_api, _run_cli, _collect_assets
        from PIL import Image
        
        print("🧪 Testing 3D generation...")
        
        # Create job directory
        jobdir = _now_jobdir()
        job0 = jobdir / "0"
        input_png = job0 / "input.png"
        
        # Copy and convert image
        img = Image.open(image_path).convert("RGB")
        img.save(input_png)
        print(f"✅ Image saved to: {input_png}")
        
        # Try micro API first
        print("🔄 Trying TripoSR micro-API...")
        ok_micro, payload = _try_micro_api(
            input_png, 
            bake_texture=True, 
            texture_resolution=1024, 
            mc_resolution=384, 
            supersample=1
        )
        
        if ok_micro:
            print("✅ Micro-API succeeded!")
            print(f"📊 Result: {payload}")
        else:
            print("⚠️ Micro-API failed, trying CLI fallback...")
            print(f"❌ Error: {payload}")
            
            # Try CLI fallback
            print("🔄 Trying CLI fallback...")
            ok_cli, log = _run_cli(
                input_png, jobdir, 
                bake_texture=True, 
                texture_resolution=1024, 
                mc_resolution=384
            )
            
            if ok_cli:
                print("✅ CLI fallback succeeded!")
                assets = _collect_assets(jobdir)
                print(f"📊 Generated assets: {assets}")
            else:
                print("❌ CLI fallback also failed")
                print(f"📝 Log: {log}")
        
        print(f"📁 Job directory: {jobdir}")
        return True
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 Testing Fashion3D Generation in WSL")
    print("=" * 50)
    test_in_wsl()
