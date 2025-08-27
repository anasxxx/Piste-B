import os, sys, requests, trimesh

if len(sys.argv) < 2:
    print("Usage: python3 scripts/generate_with_iso_search.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
isos = [0.35, 0.45, 0.5, 0.55, 0.6]
best = None
session = requests.Session()

def score_mesh(path):
    try:
        m = trimesh.load(path, force='mesh')
        comps = m.split(only_watertight=False)
        comps.sort(key=lambda c: c.faces.shape[0], reverse=True)
        return comps[0].faces.shape[0]
    except Exception:
        return -1

for iso in isos:
    with open(img_path, "rb") as f:
        files = {"file": ("img.jpg", f, "image/jpeg")}
        data = {"format": "obj", "iso": str(iso)}
        r = session.post("http://127.0.0.1:8001/generate", files=files, data=data, timeout=300)
    r.raise_for_status()
    path = r.json()["path"]
    sc = score_mesh(path)
    print(f"iso {iso}: score {sc} -> {path}")
    if best is None or sc > best[0]:
        best = (sc, iso, path)

print("Best:", best)
