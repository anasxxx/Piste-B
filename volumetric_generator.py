#!/usr/bin/env python3
"""
Volumetric 3D Generator - Creates real 3D objects with volume and structure
"""
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import trimesh
from pathlib import Path
from rembg import remove
import logging
import io
from scipy.ndimage import gaussian_filter
from skimage import measure

class VolumetricGenerator:
    """Advanced 3D generator that creates true volumetric objects"""
    
    def __init__(self):
        self.logger = logging.getLogger("volumetric")
        logging.basicConfig(level=logging.INFO)
    
    def preprocess_image(self, img_path: Path) -> tuple:
        """Enhanced image preprocessing"""
        self.logger.info(f"Processing image: {img_path}")
        
        # Remove background
        with open(img_path, 'rb') as f:
            input_data = f.read()
        
        # Remove background with rembg
        output_data = remove(input_data)
        img = Image.open(io.BytesIO(output_data))
        
        # Enhance image
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
        
        img_array = np.array(img)
        
        if img_array.shape[2] == 4:  # RGBA
            alpha = img_array[:, :, 3]
            rgb = img_array[:, :, :3]
            mask = alpha > 128
        else:
            # Create mask from edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) > 0
            rgb = img_array
        
        return rgb, mask
    
    def create_3d_volume(self, image: np.ndarray, mask: np.ndarray, resolution: int = 64) -> np.ndarray:
        """Create true 3D volume from 2D image"""
        self.logger.info(f"Creating 3D volume at {resolution}³ resolution")
        
        # Resize inputs
        h, w = image.shape[:2]
        scale = resolution / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(image, (new_w, new_h))
        mask_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h)) > 0
        
        # Create volume grid
        volume = np.zeros((resolution, resolution, resolution))
        
        # Center the image in the volume
        start_y = (resolution - new_h) // 2
        start_x = (resolution - new_w) // 2
        
        # Distance transform for shape understanding
        dist_transform = cv2.distanceTransform(mask_resized.astype(np.uint8), cv2.DIST_L2, 5)
        max_dist = np.max(dist_transform) if np.max(dist_transform) > 0 else 1
        dist_norm = dist_transform / max_dist
        
        # Brightness analysis
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY) / 255.0
        
        # Fill volume with sophisticated depth modeling
        for z in range(resolution):
            # Z-depth profile (bell curve)
            z_center = resolution // 2
            z_factor = np.exp(-((z - z_center) / (resolution * 0.25))**2)
            
            # Current layer strength
            layer = np.zeros((resolution, resolution))
            
            if start_y >= 0 and start_x >= 0:
                end_y = min(start_y + new_h, resolution)
                end_x = min(start_x + new_w, resolution)
                
                # Extract region
                img_end_y = min(new_h, end_y - start_y)
                img_end_x = min(new_w, end_x - start_x)
                
                if img_end_y > 0 and img_end_x > 0:
                    # Combine multiple depth cues
                    region_dist = dist_norm[:img_end_y, :img_end_x]
                    region_gray = gray[:img_end_y, :img_end_x]
                    region_mask = mask_resized[:img_end_y, :img_end_x]
                    
                    # Multi-factor depth calculation
                    depth_factor = (
                        0.4 * region_dist +      # Shape-based depth
                        0.3 * region_gray +      # Brightness-based depth  
                        0.3 * region_mask        # Mask-based solid
                    )
                    
                    # Apply to layer
                    layer[start_y:start_y+img_end_y, start_x:start_x+img_end_x] = depth_factor
            
            # Apply z-profile and add to volume
            volume[:, :, z] = layer * z_factor
        
        # Add internal structure variation
        self.add_internal_structure(volume, resolution)
        
        # Smooth the volume
        volume = gaussian_filter(volume, sigma=0.8)
        
        # Threshold to create solid regions
        volume = (volume > 0.2).astype(float)
        
        return volume
    
    def add_internal_structure(self, volume: np.ndarray, resolution: int):
        """Add realistic internal structure to the volume"""
        center = resolution // 2
        
        # Add some internal cavities and density variation
        x, y, z = np.meshgrid(
            np.arange(resolution) - center,
            np.arange(resolution) - center, 
            np.arange(resolution) - center
        )
        
        # Distance from center
        dist_center = np.sqrt(x**2 + y**2 + z**2)
        max_dist = np.sqrt(3) * center
        
        # Create internal structure
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    if volume[i, j, k] > 0:
                        d = dist_center[i, j, k]
                        
                        # Reduce density in center (hollow effect)
                        if d < max_dist * 0.3:
                            volume[i, j, k] *= 0.6
                        # Increase density at surface
                        elif d > max_dist * 0.7:
                            volume[i, j, k] *= 1.1
    
    def volume_to_mesh(self, volume: np.ndarray, image: np.ndarray, scale: float = 2.0) -> trimesh.Trimesh:
        """Convert volume to high-quality mesh"""
        self.logger.info("Converting volume to mesh with marching cubes")
        
        try:
            # Use marching cubes for mesh extraction
            verts, faces, normals, values = measure.marching_cubes(
                volume, 
                level=0.3,
                spacing=(1.0, 1.0, 1.0),
                gradient_direction='descent'
            )
            
            # Scale and center vertices
            verts = (verts / volume.shape[0] - 0.5) * scale
            
            # Create base mesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            
            # Add vertex colors from original image
            self.add_vertex_colors(mesh, image)
            
            # Improve mesh quality
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Smooth the mesh
            mesh = mesh.smoothed()
            
            self.logger.info(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            return mesh
            
        except Exception as e:
            self.logger.error(f"Mesh generation failed: {e}")
            # Fallback to simple shape
            return trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    
    def add_vertex_colors(self, mesh: trimesh.Trimesh, image: np.ndarray):
        """Add realistic vertex colors based on original image"""
        h, w = image.shape[:2]
        colors = []
        
        for vertex in mesh.vertices:
            # Map 3D vertex to 2D image coordinates
            # Normalize coordinates to [0, 1]
            u = (vertex[0] + 1) * 0.5  # Assuming vertices are in [-1, 1]
            v = (vertex[1] + 1) * 0.5
            
            # Convert to image coordinates
            img_x = int(np.clip(u * (w - 1), 0, w - 1))
            img_y = int(np.clip(v * (h - 1), 0, h - 1))
            
            color = image[img_y, img_x] / 255.0
            colors.append(color)
        
        mesh.visual.vertex_colors = np.array(colors)
    
    def generate_real_3d(self, img_path: Path, output_dir: Path, resolution: int = 64) -> dict:
        """Generate real 3D object with volume and structure"""
        self.logger.info(f"Generating real 3D object from {img_path}")
        
        try:
            # Preprocess image
            image, mask = self.preprocess_image(img_path)
            
            # Create 3D volume
            volume = self.create_3d_volume(image, mask, resolution)
            
            # Convert to mesh
            mesh = self.volume_to_mesh(volume, image)
            
            # Save results
            output_dir.mkdir(parents=True, exist_ok=True)
            
            mesh_path = output_dir / "real_3d_object.obj"
            mesh.export(str(mesh_path))
            
            # Save processed image
            texture_path = output_dir / "texture.png"
            Image.fromarray(image).save(texture_path)
            
            # Save volume visualization (optional)
            volume_path = output_dir / "volume.npy"
            np.save(volume_path, volume)
            
            result = {
                "mesh_obj": str(mesh_path),
                "texture_png": str(texture_path),
                "volume_npy": str(volume_path),
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "volume_filled": np.sum(volume > 0.3),
                "mesh_size_mb": mesh_path.stat().st_size / (1024*1024) if mesh_path.exists() else 0
            }
            
            self.logger.info(f"Real 3D object saved: {mesh_path}")
            self.logger.info(f"Mesh stats: {result['vertices']} vertices, {result['faces']} faces")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Real 3D generation failed: {e}")
            raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate real 3D objects")
    parser.add_argument("input_image", help="Input image path")
    parser.add_argument("--output-dir", default="real_3d_output", help="Output directory")
    parser.add_argument("--resolution", type=int, default=64, help="Volume resolution")
    args = parser.parse_args()
    
    generator = VolumetricGenerator()
    result = generator.generate_real_3d(
        Path(args.input_image),
        Path(args.output_dir),
        args.resolution
    )
    
    print("\n🎉 Real 3D Object Generated!")
    print(f"📁 Mesh: {result['mesh_obj']}")
    print(f"📊 Stats: {result['vertices']} vertices, {result['faces']} faces")
    print(f"💾 Size: {result['mesh_size_mb']:.2f} MB")
