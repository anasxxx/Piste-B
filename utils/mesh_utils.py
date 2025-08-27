
import numpy as np
from skimage import measure
import trimesh

def voxels_to_mesh(voxels: np.ndarray, iso: float = 0.5):
    """Convert a (D,H,W) or (D,H,W,1) occupancy grid in [0,1] to a mesh."""
    if voxels.ndim == 4:
        voxels = voxels[..., 0]
    verts, faces, normals, _ = measure.marching_cubes(voxels.astype(np.float32), level=iso)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    return mesh

def save_mesh_obj(mesh: 'trimesh.Trimesh', path: str):
    mesh.export(path)

def save_mesh_ply(mesh: 'trimesh.Trimesh', path: str):
    mesh.export(path)
