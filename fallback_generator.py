#!/usr/bin/env python3
"""
Fallback 3D mesh generator for Fashion3D when TripoSR fails.
Creates basic extrusion-based meshes from 2D images.
"""

import numpy as np
import cv2
from PIL import Image
import trimesh
from pathlib import Path
from rembg import remove
import logging
import io

logger = logging.getLogger("fallback_generator")

class FallbackMeshGenerator:
    def __init__(self):
        self.logger = logging.getLogger("fallback_generator")
    
    def remove_background(self, img_path: Path) -> tuple:
        """Remove background from image using rembg"""
        self.logger.info("Removing background...")
        with open(img_path, 'rb') as f:
            input_data = f.read()
        
        # Remove background
        output_data = remove(input_data)
        
        # Convert to numpy array
        img = Image.open(io.BytesIO(output_data))
        img_array = np.array(img)
        
        if img_array.shape[2] == 4:  # RGBA
            # Use alpha channel as mask
            alpha = img_array[:, :, 3]
            rgb = img_array[:, :, :3]
            
            # Create binary mask
            mask = alpha > 128
            return rgb, mask
        else:
            # Create simple edge-based mask
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            mask = mask > 0
            return img_array, mask
    
    def create_depth_map(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create a simple depth map from the image"""
        self.logger.info("Creating depth map...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Simple depth estimation based on brightness and edges
        depth = gray.astype(np.float32) / 255.0
        
        # Enhance depth with edge information
        edges = cv2.Canny(gray, 50, 150)
        edges_normalized = edges.astype(np.float32) / 255.0
        
        # Combine brightness and edge info for depth
        depth = 0.7 * depth + 0.3 * (1.0 - edges_normalized)
        
        # Apply mask
        depth = depth * mask.astype(np.float32)
        
        # Smooth the depth map
        depth = cv2.GaussianBlur(depth, (5, 5), 1.0)
        
        return depth
    
    def depth_to_mesh(self, image: np.ndarray, depth: np.ndarray, mask: np.ndarray, scale: float = 0.1) -> trimesh.Trimesh:
        """Convert depth map to 3D mesh"""
        self.logger.info("Converting depth to mesh...")
        
        h, w = depth.shape
        
        # Create coordinate grids
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        z = depth * scale
        
        # Only use points where mask is True
        valid_points = mask
        
        # Create vertex array
        vertices = []
        colors = []
        faces = []
        vertex_map = np.full((h, w), -1, dtype=int)
        
        # Create vertices
        vertex_idx = 0
        for i in range(h):
            for j in range(w):
                if valid_points[i, j]:
                    vertices.append([x[i, j], y[i, j], z[i, j]])
                    colors.append(image[i, j] / 255.0)
                    vertex_map[i, j] = vertex_idx
                    vertex_idx += 1
        
        vertices = np.array(vertices)
        colors = np.array(colors)
        
        # Create faces (triangles)
        for i in range(h - 1):
            for j in range(w - 1):
                # Check if all 4 corners are valid
                v1 = vertex_map[i, j]
                v2 = vertex_map[i, j + 1]
                v3 = vertex_map[i + 1, j]
                v4 = vertex_map[i + 1, j + 1]
                
                if v1 >= 0 and v2 >= 0 and v3 >= 0 and v4 >= 0:
                    # Create two triangles
                    faces.append([v1, v2, v3])
                    faces.append([v2, v4, v3])
        
        faces = np.array(faces)
        
        if len(vertices) == 0 or len(faces) == 0:
            self.logger.warning("No valid mesh generated, creating simple plane")
            return self.create_simple_plane()
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        
        # Add some thickness by extruding backwards
        back_vertices = vertices.copy()
        back_vertices[:, 2] -= scale * 0.1  # Small back extrusion
        
        all_vertices = np.vstack([vertices, back_vertices])
        n_front = len(vertices)
        
        # Create back faces (inverted normals)
        back_faces = faces + n_front
        back_faces = back_faces[:, [0, 2, 1]]  # Flip normals
        
        # Create side faces
        side_faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                v1 = vertex_map[i, j]
                v2 = vertex_map[i, j + 1]
                v3 = vertex_map[i + 1, j]
                v4 = vertex_map[i + 1, j + 1]
                
                if v1 >= 0 and v2 >= 0:  # Top edge
                    side_faces.extend([
                        [v1, v1 + n_front, v2],
                        [v2, v1 + n_front, v2 + n_front]
                    ])
                if v1 >= 0 and v3 >= 0:  # Left edge
                    side_faces.extend([
                        [v1, v3, v1 + n_front],
                        [v3, v3 + n_front, v1 + n_front]
                    ])
        
        side_faces = np.array(side_faces)
        
        # Combine all faces
        all_faces = np.vstack([faces, back_faces, side_faces]) if len(side_faces) > 0 else np.vstack([faces, back_faces])
        all_colors = np.vstack([colors, colors])  # Same colors for back
        
        # Create final mesh
        mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, vertex_colors=all_colors)
        
        return mesh
    
    def create_simple_plane(self) -> trimesh.Trimesh:
        """Create a simple textured plane as fallback"""
        vertices = np.array([
            [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
            [-1, -1, -0.1], [1, -1, -0.1], [1, 1, -0.1], [-1, 1, -0.1]
        ])
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Front
            [4, 6, 5], [4, 7, 6],  # Back
            [0, 4, 5], [0, 5, 1],  # Bottom
            [2, 6, 7], [2, 7, 3],  # Top
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2]   # Right
        ])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def generate_mesh(self, img_path: Path, output_dir: Path, 
                     bake_texture: bool = True, mc_resolution: int = 256) -> dict:
        """Main mesh generation function"""
        self.logger.info(f"Generating mesh from {img_path}")
        
        try:
            # Load and process image
            image, mask = self.remove_background(img_path)
            
            # Create depth map
            depth = self.create_depth_map(image, mask)
            
            # Generate mesh
            mesh = self.depth_to_mesh(image, depth, mask)
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save mesh
            mesh_path = output_dir / "mesh.obj"
            mesh.export(str(mesh_path))
            
            # Save texture if requested
            texture_path = None
            if bake_texture:
                texture_path = output_dir / "albedo.png"
                Image.fromarray(image).save(texture_path)
            
            # Save input copy
            input_copy = output_dir / "input.png"
            Image.open(img_path).save(input_copy)
            
            self.logger.info(f"Mesh saved to {mesh_path}")
            
            return {
                "mesh_obj": str(mesh_path),
                "albedo_png": str(texture_path) if texture_path else None,
                "input_png": str(input_copy),
                "mesh_mtl": None  # No MTL for now
            }
            
        except Exception as e:
            self.logger.error(f"Mesh generation failed: {e}")
            raise


def generate_fallback_mesh(img_path: Path, output_dir: Path, 
                          bake_texture: bool = True, mc_resolution: int = 256) -> dict:
    """Standalone function for fallback mesh generation"""
    generator = FallbackMeshGenerator()
    return generator.generate_mesh(img_path, output_dir, bake_texture, mc_resolution)


if __name__ == "__main__":
    # Test the fallback generator
    import argparse
    
    parser = argparse.ArgumentParser(description="Fallback mesh generator")
    parser.add_argument("input_image", help="Input image path")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--bake-texture", action="store_true", help="Generate texture")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    result = generate_fallback_mesh(
        Path(args.input_image),
        Path(args.output_dir),
        args.bake_texture
    )
    
    print("Generated assets:", result)
