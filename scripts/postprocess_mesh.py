import sys, trimesh
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python3 scripts/postprocess_mesh.py <in_mesh> <out_mesh>")
    sys.exit(1)

inp, outp = sys.argv[1], sys.argv[2]
m = trimesh.load(inp, force='mesh')

# Keep largest connected component
comps = m.split(only_watertight=False)
if len(comps) > 1:
    comps.sort(key=lambda c: c.faces.shape[0], reverse=True)
    m = comps[0]

# Remove tiny pieces by connected labels (if available)
try:
    mask = trimesh.graph.connected_component_labels(m.face_adjacency, min_len=50)
    if mask is not None:
        keep = (mask == mask[0])
        m = m.submesh([keep], append=True)
except Exception:
    pass

# Taubin smoothing
try:
    m = m.smoothed(filter='taubin', iterations=10)
except Exception:
    pass

# Light decimation (if very dense)
if m.faces.shape[0] > 30000:
    try:
        m = m.simplify_quadratic_decimation(30000)
    except Exception:
        pass

# Recenter / unit cube
b = m.bounds
c = (b[0] + b[1]) / 2.0
s = (b[1] - b[0]).max()
if s > 0:
    m.apply_translation(-c)
    m.apply_scale(1.0 / s)

m.export(outp)
print("Wrote:", outp, "faces:", m.faces.shape[0])
