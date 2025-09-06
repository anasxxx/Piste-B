#!/usr/bin/env python3
"""
Enhanced fallback generator with improved depth estimation
"""
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import trimesh
from pathlib import Path
from rembg import remove
import logging
import io

class EnhancedFallbackGenerator:
    def __init__(self):
        self.logger = logging.getLogger("enhanced_fallback")
    
    def enhance_image_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast for better depth estimation"""
        pil_img = Image.fromarray(image)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.3)  # Increase sharpness
        
        return np.array(enhanced)
    
    def create_advanced_depth_map(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create enhanced depth map with multiple techniques"""
        self.logger.info("Creating enhanced depth map...")
        
        # Enhance image first
        enhanced_image = self.enhance_image_contrast(image)
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2HSV)
        
        # Method 1: Brightness-based depth
        brightness_depth = gray.astype(np.float32) / 255.0
        
        # Method 2: Edge-based depth enhancement
        edges = cv2.Canny(gray, 30, 100)
        edges_blur = cv2.GaussianBlur(edges, (5, 5), 1.0)
        edge_depth = 1.0 - (edges_blur.astype(np.float32) / 255.0)
        
        # Method 3: Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
        gradient_depth = 1.0 - gradient_magnitude
        
        # Method 4: Saturation-based depth (colorful areas often closer)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        
        # Method 5: Distance transform for shape-based depth
        mask_uint8 = (mask * 255).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        if np.max(dist_transform) > 0:
            dist_transform = dist_transform / np.max(dist_transform)
        
        # Combine all methods with weights
        combined_depth = (
            0.3 * brightness_depth +      # Primary: brightness
            0.2 * edge_depth +            # Edges create depth variation  
            0.2 * gradient_depth +        # Gradients add detail
            0.15 * saturation +           # Color saturation
            0.15 * dist_transform         # Shape-based depth
        )
        
        # Apply mask
        combined_depth = combined_depth * mask.astype(np.float32)
        
        # Enhance depth variation
        if np.max(combined_depth) > 0:
            combined_depth = combined_depth / np.max(combined_depth)
            
        # Apply non-linear enhancement to increase depth variation
        combined_depth = np.power(combined_depth, 0.8)  # Enhance mid-tones
        
        # Add some noise for texture (very small amount)
        noise = np.random.normal(0, 0.02, combined_depth.shape)
        combined_depth = np.clip(combined_depth + noise, 0, 1)
        
        # Smooth the result
        combined_depth = cv2.GaussianBlur(combined_depth, (3, 3), 0.5)
        
        return combined_depth
    
    def depth_to_enhanced_mesh(self, image: np.ndarray, depth: np.ndarray, mask: np.ndarray, scale: float = 0.2) -> trimesh.Trimesh:
        """Convert depth map to enhanced 3D mesh with more detail"""
        self.logger.info("Converting to enhanced 3D mesh...")
        
        h, w = depth.shape
        
        # Create higher resolution coordinate grids
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        
        # Enhanced depth scaling with more variation
        z = depth * scale
        
        # Add subtle curvature to make it less flat
        center_x, center_y = w // 2, h // 2
        dist_from_center = np.sqrt((np.arange(w) - center_x)**2 + (np.arange(h)[:, np.newaxis] - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        curvature = (1 - dist_from_center / max_dist) * 0.05 * scale  # Subtle curvature
        z = z + curvature
        
        # Only use valid points
        valid_points = mask
        
        # Create vertices with enhanced detail
        vertices = []
        colors = []
        vertex_map = np.full((h, w), -1, dtype=int)
        
        vertex_idx = 0
        for i in range(h):
            for j in range(w):
                if valid_points[i, j]:
                    vertices.append([x[i, j], y[i, j], z[i, j]])
                    colors.append(image[i, j] / 255.0)
                    vertex_map[i, j] = vertex_idx
                    vertex_idx += 1
        
        if len(vertices) == 0:
            return self.create_enhanced_plane()
        
        vertices = np.array(vertices)
        colors = np.array(colors)
        
        # Create faces with better tessellation
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
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
        
        # Create mesh with enhanced thickness
        back_vertices = vertices.copy()
        back_vertices[:, 2] -= scale * 0.3  # More substantial thickness
        
        all_vertices = np.vstack([vertices, back_vertices])
        n_front = len(vertices)
        
        # Enhanced back faces
        back_faces = faces + n_front
        back_faces = back_faces[:, [0, 2, 1]]  # Flip normals
        
        # Create more detailed side faces
        side_faces = []
        # Enhanced side face generation for better edges
        for i in range(h - 1):
            for j in range(w - 1):
                v1 = vertex_map[i, j]
                v2 = vertex_map[i, j + 1]
                v3 = vertex_map[i + 1, j]
                v4 = vertex_map[i + 1, j + 1]
                
                # Create side faces for edges
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
        
        side_faces = np.array(side_faces) if side_faces else np.empty((0, 3), dtype=int)
        
        # Combine all faces
        if len(side_faces) > 0:
            all_faces = np.vstack([faces, back_faces, side_faces])
        else:
            all_faces = np.vstack([faces, back_faces])
        
        all_colors = np.vstack([colors, colors])
        
        # Create final enhanced mesh
        mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, vertex_colors=all_colors)
        
        # Apply smoothing to improve quality
        mesh = mesh.smoothed()
        
        return mesh
    
    def create_enhanced_plane(self) -> trimesh.Trimesh:
        """Create an enhanced plane with more detail"""
        # Create a more detailed plane
        size = 2
        resolution = 10
        step = size / resolution
        
        vertices = []
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = -size/2 + j * step
                y = -size/2 + i * step
                z = 0.1 * np.sin(i * 0.5) * np.cos(j * 0.5)  # Add some curvature
                vertices.append([x, y, z])
        
        faces = []
        for i in range(resolution):
            for j in range(resolution):
                v1 = i * (resolution + 1) + j
                v2 = v1 + 1
                v3 = v1 + (resolution + 1)
                v4 = v3 + 1
                
                faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def generate_enhanced_mesh(self, img_path: Path, output_dir: Path, 
                              bake_texture: bool = True, mc_resolution: int = 256) -> dict:
        """Generate enhanced mesh with better depth"""
        self.logger.info(f"Generating enhanced mesh from {img_path}")
        
        try:
            # Enhanced background removal and processing
            with open(img_path, 'rb') as f:
                input_data = f.read()
            
            output_data = remove(input_data)
            img = Image.open(io.BytesIO(output_data))
            
            # Resize for better processing if needed
            max_size = mc_resolution * 2
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            img_array = np.array(img)
            
            if img_array.shape[2] == 4:  # RGBA
                alpha = img_array[:, :, 3]
                rgb = img_array[:, :, :3]
                mask = alpha > 128
            else:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
                mask = mask > 0
                rgb = img_array
            
            # Enhanced depth map creation
            depth = self.create_advanced_depth_map(rgb, mask)
            
            # Enhanced mesh generation
            mesh = self.depth_to_enhanced_mesh(rgb, depth, mask, scale=0.25)  # Increased scale
            
            # Save results
            output_dir.mkdir(parents=True, exist_ok=True)
            
            mesh_path = output_dir / "mesh_enhanced.obj"
            mesh.export(str(mesh_path))
            
            texture_path = None
            if bake_texture:
                texture_path = output_dir / "albedo_enhanced.png"
                Image.fromarray(rgb).save(texture_path)
            
            input_copy = output_dir / "input.png"
            Image.fromarray(rgb).save(input_copy)
            
            self.logger.info(f"Enhanced mesh saved to {mesh_path}")
            
            return {
                "mesh_obj": str(mesh_path),
                "albedo_png": str(texture_path) if texture_path else None,
                "input_png": str(input_copy),
                "mesh_mtl": None
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced mesh generation failed: {e}")
            raise

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image")
    parser.add_argument("--output-dir", default="enhanced_output")
    args = parser.parse_args()
    
    generator = EnhancedFallbackGenerator()
    result = generator.generate_enhanced_mesh(Path(args.input_image), Path(args.output_dir))
    print("Enhanced result:", result)
