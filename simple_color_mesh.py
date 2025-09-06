#!/usr/bin/env python3
"""
Simple, reliable mesh color enhancement
"""
import numpy as np
import cv2
from PIL import Image
import trimesh
from pathlib import Path
import sys

def add_colors_to_mesh(mesh_file, image_file, output_file):
    """Simple function to add colors to existing mesh"""
    print(f"Loading mesh: {mesh_file}")
    
    # Load mesh
    try:
        mesh = trimesh.load_mesh(str(mesh_file))
        print(f"✓ Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"✗ Failed to load mesh: {e}")
        return False
    
    # Load image
    try:
        image = np.array(Image.open(image_file))
        h, w = image.shape[:2]
        print(f"✓ Image loaded: {w}x{h} pixels")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False
    
    # Map colors to vertices
    vertices = mesh.vertices
    colors = []
    
    # Get coordinate bounds
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    print("Mapping colors to vertices...")
    
    for vertex in vertices:
        # Normalize coordinates to [0, 1]
        if x_max != x_min:
            u = (vertex[0] - x_min) / (x_max - x_min)
        else:
            u = 0.5
            
        if y_max != y_min:
            v = (vertex[1] - y_min) / (y_max - y_min)
        else:
            v = 0.5
        
        # Map to image coordinates
        img_x = int(u * (w - 1))
        img_y = int((1 - v) * (h - 1))  # Flip Y axis
        
        # Clamp to image bounds
        img_x = max(0, min(w - 1, img_x))
        img_y = max(0, min(h - 1, img_y))
        
        # Get color
        color = image[img_y, img_x] / 255.0
        colors.append(color)
    
    # Apply colors to mesh
    mesh.visual.vertex_colors = np.array(colors)
    
    print(f"✓ Applied colors to {len(colors)} vertices")
    
    # Save colored mesh
    try:
        mesh.export(str(output_file))
        print(f"✓ Colored mesh saved: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to save mesh: {e}")
        return False

if __name__ == "__main__":
    # Direct paths
    mesh_path = "outputs/20250905-221214/0/mesh.obj"
    image_path = "inputs/test_image.jpeg"
    output_path = "colored_shoe_simple.obj"
    
    print("Simple Mesh Color Enhancement")
    print("=" * 40)
    
    # Check if files exist
    if not Path(mesh_path).exists():
        print(f"✗ Mesh file not found: {mesh_path}")
        print("Available mesh files:")
        for f in Path("outputs").rglob("*.obj"):
            print(f"  {f}")
        sys.exit(1)
    
    if not Path(image_path).exists():
        print(f"✗ Image file not found: {image_path}")
        print("Available image files:")
        for f in Path("inputs").rglob("*.jpg"):
            print(f"  {f}")
        for f in Path("inputs").rglob("*.png"):
            print(f"  {f}")
        sys.exit(1)
    
    # Add colors
    success = add_colors_to_mesh(mesh_path, image_path, output_path)
    
    if success:
        print(f"\n🎨 SUCCESS! Colored mesh created: {output_path}")
        print("View in Blender or MeshLab to see the colors")
        
        # Show file info
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")
    else:
        print("\n❌ Color enhancement failed")
