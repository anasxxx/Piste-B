#!/usr/bin/env python3
"""
Enhanced color mapping for 3D meshes
"""
import numpy as np
import cv2
from PIL import Image
import trimesh
from pathlib import Path

def enhance_mesh_colors(mesh_path: Path, original_image_path: Path, output_path: Path):
    """Add better colors and details to existing mesh"""
    print(f"Enhancing mesh colors: {mesh_path}")
    
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Load original image
    image = np.array(Image.open(original_image_path))
    h, w = image.shape[:2]
    
    # Improve vertex color mapping
    vertices = mesh.vertices
    colors = []
    
    # Normalize vertex coordinates to image space
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    
    # Map to image coordinates with better projection
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Normalize to [0, 1] then scale to image dimensions
    x_norm = (x_coords - x_min) / (x_max - x_min) if x_max != x_min else np.zeros_like(x_coords)
    y_norm = (y_coords - y_min) / (y_max - y_min) if y_max != y_min else np.zeros_like(y_coords)
    
    # Convert to image coordinates
    img_x = (x_norm * (w - 1)).astype(int)
    img_y = ((1 - y_norm) * (h - 1)).astype(int)  # Flip Y axis
    
    # Clamp coordinates
    img_x = np.clip(img_x, 0, w - 1)
    img_y = np.clip(img_y, 0, h - 1)
    
    # Extract colors with smoothing
    for i in range(len(vertices)):
        x, y = img_x[i], img_y[i]
        
        # Sample area around point for smoother colors
        x_start = max(0, x - 2)
        x_end = min(w, x + 3)
        y_start = max(0, y - 2)
        y_end = min(h, y + 3)
        
        # Average color in small region
        region = image[y_start:y_end, x_start:x_end]
        if region.size > 0:
            avg_color = np.mean(region.reshape(-1, 3), axis=0)
        else:
            avg_color = image[y, x]
        
        colors.append(avg_color / 255.0)
    
    # Apply colors
    mesh.visual.vertex_colors = np.array(colors)
    
    # Save enhanced mesh
    mesh.export(output_path)
    print(f"Enhanced mesh saved: {output_path}")
    
    return mesh

def create_texture_mapped_mesh(mesh_path: Path, original_image_path: Path, output_dir: Path):
    """Create mesh with proper UV texture mapping"""
    print(f"Creating texture-mapped mesh")
    
    # Load mesh and image
    mesh = trimesh.load(mesh_path)
    image = Image.open(original_image_path)
    
    # Create UV coordinates based on vertex positions
    vertices = mesh.vertices
    
    # Project vertices to UV space
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Normalize to [0, 1] UV space
    u = (x_coords - x_min) / (x_max - x_min) if x_max != x_min else np.zeros_like(x_coords)
    v = (y_coords - y_min) / (y_max - y_min) if y_max != y_min else np.zeros_like(y_coords)
    
    # Flip V coordinate (UV convention)
    v = 1 - v
    
    # Create UV coordinates array
    uv_coords = np.column_stack([u, v])
    
    # Create material with texture
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image,
        metallicFactor=0.0,
        roughnessFactor=0.8
    )
    
    # Apply texture
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv_coords,
        material=material
    )
    
    # Save files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mesh with texture
    mesh_file = output_dir / "textured_mesh.obj"
    mesh.export(mesh_file)
    
    # Save texture image
    texture_file = output_dir / "texture.png"
    image.save(texture_file)
    
    # Create MTL file for proper loading
    mtl_file = output_dir / "textured_mesh.mtl"
    with open(mtl_file, 'w') as f:
        f.write(f"""newmtl material0
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks 0.1 0.1 0.1
Ns 10.0
map_Kd texture.png
""")
    
    print(f"Textured mesh saved: {mesh_file}")
    print(f"Texture saved: {texture_file}")
    print(f"Material saved: {mtl_file}")
    
    return mesh_file

def increase_mesh_detail(mesh_path: Path, subdivision_levels: int = 1):
    """Increase mesh detail through subdivision"""
    print(f"Increasing mesh detail: {subdivision_levels} levels")
    
    mesh = trimesh.load(mesh_path)
    
    # Subdivide mesh for more detail
    for level in range(subdivision_levels):
        mesh = mesh.subdivide()
        print(f"Subdivision level {level + 1}: {len(mesh.vertices)} vertices")
    
    # Smooth the result
    mesh = mesh.smoothed()
    
    # Save detailed mesh
    detail_path = mesh_path.parent / f"detailed_{mesh_path.name}"
    mesh.export(detail_path)
    
    print(f"Detailed mesh saved: {detail_path}")
    return detail_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance 3D mesh colors and details")
    parser.add_argument("mesh_file", help="Input OBJ mesh file")
    parser.add_argument("image_file", help="Original image file")
    parser.add_argument("--output-dir", default="enhanced_output", help="Output directory")
    parser.add_argument("--subdivide", type=int, default=1, help="Subdivision levels for detail")
    args = parser.parse_args()
    
    mesh_path = Path(args.mesh_file)
    image_path = Path(args.image_file)
    output_dir = Path(args.output_dir)
    
    if not mesh_path.exists():
        print(f"Mesh file not found: {mesh_path}")
        exit(1)
    
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        exit(1)
    
    # Enhance colors
    enhanced_path = output_dir / "enhanced_mesh.obj"
    enhance_mesh_colors(mesh_path, image_path, enhanced_path)
    
    # Create textured version
    create_texture_mapped_mesh(enhanced_path, image_path, output_dir)
    
    # Increase detail
    if args.subdivide > 0:
        increase_mesh_detail(enhanced_path, args.subdivide)
    
    print("\nEnhancement complete!")
    print(f"Check output directory: {output_dir}")
