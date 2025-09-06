#!/usr/bin/env python3
"""
Updated volumetric generator with improved color handling
"""
import sys
sys.path.append('.')

from volumetric_generator import VolumetricGenerator
import numpy as np
import cv2
from PIL import Image
import trimesh
from pathlib import Path

class ColorEnhancedGenerator(VolumetricGenerator):
    def add_enhanced_vertex_colors(self, mesh: trimesh.Trimesh, image: np.ndarray):
        """Add high-quality vertex colors with proper mapping"""
        h, w = image.shape[:2]
        colors = []
        
        vertices = mesh.vertices
        
        # Better coordinate mapping
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        
        # Normalize coordinates
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Handle edge cases
        if x_max == x_min:
            x_norm = np.full_like(x_coords, 0.5)
        else:
            x_norm = (x_coords - x_min) / (x_max - x_min)
            
        if y_max == y_min:
            y_norm = np.full_like(y_coords, 0.5)
        else:
            y_norm = (y_coords - y_min) / (y_max - y_min)
        
        # Map to image coordinates
        img_x = (x_norm * (w - 1)).astype(int)
        img_y = ((1 - y_norm) * (h - 1)).astype(int)  # Flip Y
        
        # Clamp to valid range
        img_x = np.clip(img_x, 0, w - 1)
        img_y = np.clip(img_y, 0, h - 1)
        
        # Extract colors with area sampling for better quality
        for i in range(len(vertices)):
            x, y = img_x[i], img_y[i]
            
            # Sample 3x3 area for smoother colors
            x_start = max(0, x - 1)
            x_end = min(w, x + 2)
            y_start = max(0, y - 1)
            y_end = min(h, y + 2)
            
            region = image[y_start:y_end, x_start:x_end]
            if region.size > 0:
                # Weighted average with center pixel having more weight
                if region.shape[0] >= 3 and region.shape[1] >= 3:
                    center_weight = 4
                    edge_weight = 1
                    weights = np.ones(region.shape[:2]) * edge_weight
                    center_y, center_x = region.shape[0] // 2, region.shape[1] // 2
                    weights[center_y, center_x] = center_weight
                    
                    # Calculate weighted average
                    weighted_sum = np.sum(region * weights[:, :, np.newaxis], axis=(0, 1))
                    total_weight = np.sum(weights)
                    avg_color = weighted_sum / total_weight
                else:
                    avg_color = np.mean(region.reshape(-1, 3), axis=0)
            else:
                avg_color = image[y, x]
            
            colors.append(avg_color / 255.0)
        
        mesh.visual.vertex_colors = np.array(colors)
        
        # Add alpha channel for proper rendering
        if hasattr(mesh.visual, 'vertex_colors'):
            colors_rgba = np.column_stack([
                mesh.visual.vertex_colors[:, :3],
                np.ones(len(mesh.visual.vertex_colors))  # Alpha = 1
            ])
            mesh.visual.vertex_colors = colors_rgba
    
    def create_detailed_mesh(self, image: np.ndarray, depth: np.ndarray, mask: np.ndarray, 
                           scale: float = 0.25, resolution_multiplier: float = 1.5) -> trimesh.Trimesh:
        """Create mesh with higher detail and better colors"""
        self.logger.info("Creating detailed mesh with enhanced colors")
        
        h, w = depth.shape
        
        # Increase sampling density for more detail
        detail_h = int(h * resolution_multiplier)
        detail_w = int(w * resolution_multiplier)
        
        # Upsample depth and mask
        depth_detailed = cv2.resize(depth, (detail_w, detail_h), interpolation=cv2.INTER_CUBIC)
        mask_detailed = cv2.resize(mask.astype(np.uint8), (detail_w, detail_h), interpolation=cv2.INTER_CUBIC) > 0.5
        image_detailed = cv2.resize(image, (detail_w, detail_h), interpolation=cv2.INTER_CUBIC)
        
        # Create coordinate grids with higher resolution
        x, y = np.meshgrid(np.linspace(-1, 1, detail_w), np.linspace(-1, 1, detail_h))
        z = depth_detailed * scale
        
        # Add subtle surface variation for more realistic appearance
        noise = np.random.normal(0, scale * 0.02, z.shape)
        z = z + noise * mask_detailed
        
        # Create vertices
        vertices = []
        valid_points = mask_detailed
        vertex_map = np.full((detail_h, detail_w), -1, dtype=int)
        
        vertex_idx = 0
        for i in range(detail_h):
            for j in range(detail_w):
                if valid_points[i, j]:
                    vertices.append([x[i, j], y[i, j], z[i, j]])
                    vertex_map[i, j] = vertex_idx
                    vertex_idx += 1
        
        if len(vertices) == 0:
            return self.create_enhanced_plane()
        
        vertices = np.array(vertices)
        
        # Create faces with proper topology
        faces = []
        for i in range(detail_h - 1):
            for j in range(detail_w - 1):
                v1 = vertex_map[i, j]
                v2 = vertex_map[i, j + 1]
                v3 = vertex_map[i + 1, j]
                v4 = vertex_map[i + 1, j + 1]
                
                if v1 >= 0 and v2 >= 0 and v3 >= 0 and v4 >= 0:
                    faces.append([v1, v2, v3])
                    faces.append([v2, v4, v3])
        
        if len(faces) == 0:
            return self.create_enhanced_plane()
        
        faces = np.array(faces)
        
        # Create enhanced mesh with proper thickness
        back_vertices = vertices.copy()
        back_vertices[:, 2] -= scale * 0.4  # Increased thickness
        
        all_vertices = np.vstack([vertices, back_vertices])
        n_front = len(vertices)
        
        # Enhanced back faces
        back_faces = faces + n_front
        back_faces = back_faces[:, [0, 2, 1]]
        
        # Create comprehensive side faces
        side_faces = []
        for i in range(detail_h - 1):
            for j in range(detail_w - 1):
                v1 = vertex_map[i, j]
                v2 = vertex_map[i, j + 1]
                v3 = vertex_map[i + 1, j]
                v4 = vertex_map[i + 1, j + 1]
                
                # Add side faces for all edges
                if v1 >= 0 and v2 >= 0:
                    side_faces.extend([
                        [v1, v1 + n_front, v2],
                        [v2, v1 + n_front, v2 + n_front]
                    ])
                if v1 >= 0 and v3 >= 0:
                    side_faces.extend([
                        [v1, v3, v1 + n_front],
                        [v3, v3 + n_front, v1 + n_front]
                    ])
        
        # Combine all faces
        if len(side_faces) > 0:
            all_faces = np.vstack([faces, back_faces, np.array(side_faces)])
        else:
            all_faces = np.vstack([faces, back_faces])
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
        
        # Add enhanced vertex colors
        self.add_enhanced_vertex_colors(mesh, image_detailed)
        
        # Improve mesh quality
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Apply smoothing
        mesh = mesh.smoothed()
        
        return mesh
    
    def generate_enhanced_3d(self, img_path: Path, output_dir: Path, resolution: int = 64) -> dict:
        """Generate enhanced 3D object with better colors and details"""
        self.logger.info(f"Generating enhanced 3D object from {img_path}")
        
        try:
            # Process image
            image, mask = self.preprocess_image(img_path)
            
            # Create 3D volume
            volume = self.create_3d_volume(image, mask, resolution)
            
            # Create detailed mesh with enhanced colors
            mesh = self.create_detailed_mesh(image, 
                                           self.create_advanced_depth_map(image, mask), 
                                           mask)
            
            # Save results
            output_dir.mkdir(parents=True, exist_ok=True)
            
            mesh_path = output_dir / "enhanced_3d_object.obj"
            mesh.export(str(mesh_path))
            
            # Save high-quality texture
            texture_path = output_dir / "enhanced_texture.png"
            Image.fromarray(image).save(texture_path, quality=95)
            
            result = {
                "mesh_obj": str(mesh_path),
                "texture_png": str(texture_path),
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "has_colors": hasattr(mesh.visual, 'vertex_colors'),
                "mesh_size_mb": mesh_path.stat().st_size / (1024*1024) if mesh_path.exists() else 0
            }
            
            self.logger.info(f"Enhanced 3D object saved: {mesh_path}")
            self.logger.info(f"Mesh stats: {result['vertices']} vertices, {result['faces']} faces")
            self.logger.info(f"Colors: {'Yes' if result['has_colors'] else 'No'}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced 3D generation failed: {e}")
            raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enhanced 3D objects with colors")
    parser.add_argument("input_image", help="Input image path")
    parser.add_argument("--output-dir", default="enhanced_3d_output", help="Output directory")
    parser.add_argument("--resolution", type=int, default=64, help="Volume resolution")
    args = parser.parse_args()
    
    generator = ColorEnhancedGenerator()
    result = generator.generate_enhanced_3d(
        Path(args.input_image),
        Path(args.output_dir),
        args.resolution
    )
    
    print(f"\n🎨 Enhanced 3D Object Generated!")
    print(f"📁 Mesh: {result['mesh_obj']}")
    print(f"🎨 Colors: {'✓' if result['has_colors'] else '✗'}")
    print(f"📊 Stats: {result['vertices']} vertices, {result['faces']} faces")
    print(f"💾 Size: {result['mesh_size_mb']:.2f} MB")
