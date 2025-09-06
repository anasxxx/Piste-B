import os, io, time, json, tempfile, subprocess, logging, sys
from typing import Optional, Tuple
from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# Import fallback generator
try:
    from fallback_generator import generate_fallback_mesh
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False

# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fashion3d_debug.log')
    ]
)
logger = logging.getLogger("fashion3d")

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
ART_DIR = Path(os.getenv("F3D_ARTIFACTS_DIR", "outputs")).resolve()
ART_DIR.mkdir(parents=True, exist_ok=True)

MICRO_URL = os.getenv("TRIPOSR_URL", "http://127.0.0.1:8001")  # micro-API
RUNPY = Path.home() / "TripoSR" / "run.py"                     # fallback CLI

app = FastAPI(title="Fashion3D Gateway")

logger.info(f"Fashion3D API starting...")
logger.info(f"Artifacts dir: {ART_DIR}")
logger.info(f"TripoSR micro API: {MICRO_URL}")
logger.info(f"TripoSR CLI path: {RUNPY}")
logger.info(f"TripoSR CLI exists: {RUNPY.exists()}")
logger.info(f"Fallback generator available: {FALLBACK_AVAILABLE}")

def _now_jobdir() -> Path:
    j = ART_DIR / time.strftime("%Y%m%d-%H%M%S")
    j.mkdir(parents=True, exist_ok=True)
    (j / "0").mkdir(parents=True, exist_ok=True)
    logger.info(f"Created job directory: {j}")
    return j

def _save_upload_to_png(f: UploadFile, dst_png: Path) -> None:
    logger.info(f"Saving upload to: {dst_png}")
    data = f.file.read()
    if not data:
        logger.error("Empty file uploaded")
        raise HTTPException(400, "Empty file")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.save(dst_png)
    logger.info(f"Saved image: {dst_png.stat().st_size} bytes")

def _try_micro_api(img_path: Path,
                   bake_texture: bool,
                   texture_resolution: int,
                   mc_resolution: int,
                   supersample: int) -> Tuple[bool, dict]:
    """Return (ok, payload)."""
    logger.info(f"Trying micro API at {MICRO_URL}")
    url = f"{MICRO_URL.rstrip('/')}/generate"
    files = {
        "file": (img_path.name, open(img_path, "rb"), "image/png")
    }
    data = {
        "bake_texture": str(bake_texture).lower(),
        "texture_resolution": str(texture_resolution),
        "mc_resolution": str(mc_resolution),
        "supersample": str(supersample),
    }
    try:
        logger.info(f"POST {url} with data: {data}")
        r = requests.post(url, files=files, data=data, timeout=600)
        logger.info(f"Micro API response: {r.status_code}")
        ok = 200 <= r.status_code < 300
        payload = r.json() if r.headers.get("content-type","").startswith("application/json") else {"text": r.text, "status": r.status_code}
        if not ok:
            logger.error(f"Micro API failed: {payload}")
        return ok, payload
    except Exception as e:
        logger.error(f"Micro API exception: {type(e).__name__}: {e}")
        return False, {"error": f"{type(e).__name__}: {e}"}

def _run_cli(img_path: Path,
             outdir: Path,
             bake_texture: bool,
             texture_resolution: int,
             mc_resolution: int) -> Tuple[bool, str]:
    """Run ~/TripoSR/run.py as a fallback and capture stdout/stderr."""
    logger.info(f"Running TripoSR CLI fallback")
    logger.info(f"Input image: {img_path}")
    logger.info(f"Output dir: {outdir}")
    logger.info(f"TripoSR script: {RUNPY}")
    
    if not RUNPY.exists():
        error_msg = f"TripoSR script not found at {RUNPY}"
        logger.error(error_msg)
        return False, error_msg
    
    cmd = [
        "python3", str(RUNPY),
        str(img_path),
        "--output-dir", str(outdir),
        "--mc-resolution", str(mc_resolution),
    ]
    if bake_texture:
        cmd += ["--bake-texture", "--texture-resolution", str(texture_resolution)]

    logger.info(f"CLI command: {' '.join(cmd)}")
    
    try:
        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            cwd=RUNPY.parent  # Run from TripoSR directory
        )
        log = []
        for line in proc.stdout:  # stream logs
            line = line.rstrip()
            log.append(line)
            logger.info(f"TripoSR: {line}")
        
        code = proc.wait()
        logger.info(f"TripoSR CLI finished with code: {code}")
        return code == 0, "\n".join(log)
    except Exception as e:
        error_msg = f"CLI execution failed: {type(e).__name__}: {e}"
        logger.error(error_msg)
        return False, error_msg

def _run_fallback_generator(img_path: Path,
                           outdir: Path,
                           bake_texture: bool,
                           texture_resolution: int,
                           mc_resolution: int) -> Tuple[bool, str]:
    """Use our custom fallback mesh generator"""
    logger.info(f"Running fallback mesh generator")
    
    if not FALLBACK_AVAILABLE:
        error_msg = "Fallback generator not available"
        logger.error(error_msg)
        return False, error_msg
    
    try:
        output_dir = outdir / "0"
        assets = generate_fallback_mesh(
            img_path=img_path,
            output_dir=output_dir,
            bake_texture=bake_texture,
            mc_resolution=mc_resolution
        )
        
        logger.info(f"Fallback generator completed: {assets}")
        return True, f"Fallback mesh generation successful: {assets}"
        
    except Exception as e:
        error_msg = f"Fallback generator failed: {type(e).__name__}: {e}"
        logger.error(error_msg)
        return False, error_msg

def _collect_assets(jobdir: Path) -> dict:
    logger.info(f"Collecting assets from: {jobdir}")
    jd0 = jobdir / "0"
    
    # prefer post-processed mesh if present
    mesh_post = jd0 / "mesh_post.obj"
    mesh = jd0 / "mesh.obj"
    mtl  = jd0 / "mesh.mtl"
    albedo = jd0 / "albedo.png"
    input_png = jd0 / "input.png"
    
    assets = {
        "mesh_obj": str(mesh_post if mesh_post.exists() else (mesh if mesh.exists() else "")) or None,
        "mesh_mtl": str(mtl) if mtl.exists() else None,
        "albedo_png": str(albedo) if albedo.exists() else None,
        "input_png": str(input_png) if input_png.exists() else None,
    }
    
    logger.info(f"Found assets: {assets}")
    
    # Log what files actually exist in the output directory
    if jd0.exists():
        actual_files = list(jd0.iterdir())
        logger.info(f"Actual files in {jd0}: {[f.name for f in actual_files]}")
    else:
        logger.warning(f"Output directory {jd0} does not exist")
    
    return assets

@app.get("/health")
def health():
    health_info = {
        "ok": True,
        "artifacts": str(ART_DIR),
        "micro_api": MICRO_URL,
        "has_runpy": RUNPY.exists(),
        "triposr_path": str(RUNPY),
        "fallback_available": FALLBACK_AVAILABLE,
        "strategies": [
            "TripoSR Micro API (primary)",
            "TripoSR CLI (secondary)", 
            "Fallback Generator (tertiary)" if FALLBACK_AVAILABLE else "Fallback Generator (unavailable)"
        ]
    }
    logger.info(f"Health check: {health_info}")
    return health_info

def _add_generate_route(path: str):
    @app.post(path)
    def generate(
        file: UploadFile = File(...),
        bake_texture: bool = Form(True),
        texture_resolution: int = Form(1024),
        mc_resolution: int = Form(384),
        supersample: int = Form(1),
    ):
        logger.info(f"=== GENERATION REQUEST STARTED ===")
        logger.info(f"Parameters: bake_texture={bake_texture}, texture_res={texture_resolution}, mc_res={mc_resolution}, supersample={supersample}")
        
        t0 = time.time()
        jobdir = _now_jobdir()
        job0 = jobdir / "0"
        input_png = job0 / "input.png"

        # 1) persist upload
        try:
            _save_upload_to_png(file, input_png)
        except Exception as e:
            logger.error(f"Failed to save upload: {e}")
            raise

        # 2) try micro-API first
        logger.info("=== STRATEGY 1: TRYING MICRO API ===")
        ok_micro, payload = _try_micro_api(
            input_png, bake_texture, texture_resolution, mc_resolution, supersample
        )
        
        if ok_micro and isinstance(payload, dict) and payload.get("success", payload.get("ok")):
            logger.info("Micro API succeeded")
            result = {
                "ok": True,
                "elapsed_sec": round(time.time() - t0, 3),
                "job_dir": payload.get("job_dir"),
                "assets": payload.get("assets", {}),
                "note": "served by micro-api",
                "log_tail": payload.get("log_tail", [])[-50:],
            }
            logger.info(f"=== GENERATION COMPLETED (MICRO API) ===")
            return JSONResponse(result)

        # 3) fallback CLI
        logger.info("=== STRATEGY 2: TRYING CLI FALLBACK ===")
        ok_cli, log_cli = _run_cli(
            input_png, jobdir, bake_texture, texture_resolution, mc_resolution
        )
        
        if ok_cli:
            assets = _collect_assets(jobdir)
            if assets.get("mesh_obj"):  # Check if we actually got a mesh
                result = {
                    "ok": True,
                    "elapsed_sec": round(time.time() - t0, 3),
                    "job_dir": str(jobdir),
                    "assets": assets,
                    "note": "served by fallback CLI",
                    "log_tail": log_cli.splitlines()[-100:],
                }
                logger.info(f"=== GENERATION COMPLETED (CLI) ===")
                return JSONResponse(result)
        
        # 4) Custom fallback generator
        logger.info("=== STRATEGY 3: TRYING CUSTOM FALLBACK GENERATOR ===")
        ok_fallback, log_fallback = _run_fallback_generator(
            input_png, jobdir, bake_texture, texture_resolution, mc_resolution
        )
        
        assets = _collect_assets(jobdir)
        
        result = {
            "ok": ok_fallback,
            "elapsed_sec": round(time.time() - t0, 3),
            "job_dir": str(jobdir),
            "assets": assets,
            "note": "served by custom fallback generator" if ok_fallback else "all strategies failed",
            "log_tail": log_fallback.splitlines()[-50:] if ok_fallback else [],
            "error_summary": {
                "micro_api": payload if not ok_micro else None,
                "cli": log_cli if not ok_cli else None,
                "fallback": log_fallback if not ok_fallback else None
            }
        }
        
        logger.info(f"=== GENERATION COMPLETED ({'SUCCESS' if ok_fallback else 'FAILED'}) ===")
        return JSONResponse(result)

# register both endpoints
_add_generate_route("/generate")
_add_generate_route("/generate/")
