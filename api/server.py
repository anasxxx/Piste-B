#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Fashion3D (Piste B)
- /health   : statut & diagnostic
- /generate : prend une image 2D et retourne un mesh 3D (via TripoSR)

Stratégie:
1) Essayer d'appeler l'API TripoSR (http://127.0.0.1:8001/generate)
2) Si indisponible, lancer TripoSR localement via ~/TripoSR/run.py
"""

from __future__ import annotations
import os, io, sys, json, time, uuid, shutil, tempfile, subprocess, typing as T
from pathlib import Path

# --- FastAPI ---
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- YAML (config) ---
try:
    import yaml
except Exception as e:  # pragma: no cover
    yaml = None

# --- HTTP client (optionnel) ---
try:
    import requests
except Exception:
    requests = None

APP = FastAPI(
    title="Fashion3D API (Piste B)",
    version="1.0.0",
    description="Passerelle vers TripoSR (API locale ou fallback en exécution locale).",
)

# -----------------------------------------------------------------------------
# Chargement config (facultatif)
# -----------------------------------------------------------------------------
CWD = Path(os.getcwd())
CFG_PATH = CWD / "config.yaml"
CFG: dict = {}
if CFG_PATH.is_file() and yaml is not None:
    with CFG_PATH.open("r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f) or {}

ART_DIR = Path(CFG.get("artifacts_dir", "./outputs")).resolve()
TRIPOSR_API = os.environ.get("TRIPOSR_API", CFG.get("triposr_api", "http://127.0.0.1:8001"))
TRIPOSR_REPO = Path(os.path.expanduser(CFG.get("triposr_repo", "~/TripoSR"))).resolve()

# Assure le répertoire d’artefacts
ART_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Modèles / réponses
# -----------------------------------------------------------------------------
class GenerateResponse(BaseModel):
    success: bool
    message: str
    job_dir: str | None = None
    assets: dict = {}
    log_tail: T.List[str] | None = None


# -----------------------------------------------------------------------------
# Utilitaires
# -----------------------------------------------------------------------------
def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def _job_dir(base: Path) -> Path:
    uid = f"{_now_ts()}-{uuid.uuid4().hex[:8]}"
    d = base / "output_api" / uid
    (d / "0").mkdir(parents=True, exist_ok=True)
    return d

def _tail_text_file(path: Path, n: int = 80) -> T.List[str]:
    """Retourne les n dernières lignes d'un fichier texte si présent."""
    if not path or not path.exists() or not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-n:]
    except Exception:
        return []

def _api_available() -> bool:
    if requests is None:
        return False
    try:
        r = requests.get(f"{TRIPOSR_API}/health", timeout=1.2)
        return r.ok
    except Exception:
        return False

def _call_triposr_api(img_path: Path, bake_texture: bool, texture_res: int, mc_res: int) -> dict:
    if requests is None:
        raise RuntimeError("Le module 'requests' n'est pas installé.")
    with img_path.open("rb") as f:
        files = {"file": (img_path.name, f, "image/*")}
        data = {
            "bake_texture": str(bake_texture).lower(),  # FastAPI accepte 'true'/'false'
            "texture_resolution": str(texture_res),
            "mc_resolution": str(mc_res),
        }
        url = f"{TRIPOSR_API}/generate"
        r = requests.post(url, files=files, data=data, timeout=60)
        r.raise_for_status()
        return r.json()

def _run_triposr_local(img_path: Path, bake_texture: bool, texture_res: int, mc_res: int, job_root: Path) -> dict:
    """
    Lance ~/TripoSR/run.py en sous-processus.
    Ecrit les sorties dans job_root/0/ (mesh.obj, mesh.mtl, albedo.png, input.png)
    """
    run_py = TRIPOSR_REPO / "run.py"
    if not run_py.is_file():
        raise RuntimeError(f"TripoSR introuvable: {run_py}")

    out_dir = job_root
    cmd = [
        sys.executable or "python3",
        str(run_py),
        str(img_path),
        "--output-dir", str(out_dir),
        "--mc-resolution", str(mc_res),
    ]
    if bake_texture:
        cmd += ["--bake-texture", "--texture-resolution", str(texture_res)]

    log_file = out_dir / "stderr_stdout.log"
    with log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=lf, cwd=str(TRIPOSR_REPO))

    # Assets attendus
    bundle = out_dir / "0"
    mesh_obj = bundle / "mesh.obj"
    mesh_mtl = bundle / "mesh.mtl"
    albedo   = bundle / "albedo.png"
    input_png= bundle / "input.png"

    assets = {
        "mesh_obj": str(mesh_obj) if mesh_obj.exists() else None,
        "mesh_mtl": str(mesh_mtl) if mesh_mtl.exists() else None,
        "albedo_png": str(albedo) if albedo.exists() else None,
        "input_png": str(input_png) if input_png.exists() else None,
    }

    success = bool(assets["mesh_obj"])
    msg = "OK" if success else "TripoSR local: échec de génération (voir logs)."
    return {
        "success": success,
        "message": msg,
        "job_dir": str(out_dir),
        "assets": assets,
        "log_tail": _tail_text_file(log_file, n=120),
    }


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@APP.get("/health")
def health() -> JSONResponse:
    api_up = _api_available()
    triposr_run_py = (TRIPOSR_REPO / "run.py").is_file()
    payload = {
        "service": "fashion3d-api-piste-b",
        "artifacts_dir": str(ART_DIR),
        "triposr_api_url": TRIPOSR_API,
        "triposr_api_up": api_up,
        "triposr_repo": str(TRIPOSR_REPO),
        "triposr_run_py": triposr_run_py,
    }
    return JSONResponse({"ready": True, **payload}, status_code=200)


@APP.post("/generate", response_model=GenerateResponse)
def generate(
    file: UploadFile = File(..., description="Image d'entrée (fond transparent idéalement)"),
    bake_texture: bool = Form(True),
    texture_resolution: int = Form(1024),
    mc_resolution: int = Form(256),
) -> JSONResponse:
    """
    Prend une image et produit un mesh 3D (OBJ/MTL + texture si disponible).
    """
    # 1) Sauvegarde temporaire
    tmp_dir = Path(tempfile.mkdtemp(prefix="fashion3d_"))
    job_root = _job_dir(ART_DIR)
    try:
        # Valider les paramètres
        if texture_resolution not in (256, 512, 1024, 2048, 4096):
            raise HTTPException(status_code=400, detail="texture_resolution invalide (256/512/1024/2048/4096).")
        if not (32 <= mc_resolution <= 512):
            raise HTTPException(status_code=400, detail="mc_resolution doit être entre 32 et 512.")

        # Enregistrer le fichier
        suffix = Path(file.filename or "image").suffix or ".png"
        img_path = tmp_dir / f"input{suffix}"
        with img_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2) Essayer l'API TripoSR si dispo
        if _api_available():
            try:
                resp = _call_triposr_api(img_path, bake_texture, texture_resolution, mc_resolution)
                # On renvoie la réponse brute de TripoSR API si elle contient success
                if isinstance(resp, dict) and "success" in resp:
                    return JSONResponse(resp, status_code=200 if resp.get("success") else 500)
            except Exception as e:
                # Passer au fallback local si l'API échoue
                pass

        # 3) Fallback: exécution locale
        resp = _run_triposr_local(img_path, bake_texture, texture_resolution, mc_resolution, job_root)
        return JSONResponse(resp, status_code=200 if resp.get("success") else 500)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "message": f"Erreur serveur: {e}",
                "job_dir": str(job_root),
                "assets": {},
            },
            status_code=500,
        )
    finally:
        # Nettoyage temp
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Démarrage local: uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
# -----------------------------------------------------------------------------
