import os, io, time, json, tempfile, subprocess
from typing import Optional, Tuple
from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
ART_DIR = Path(os.getenv("F3D_ARTIFACTS_DIR", "outputs")).resolve()
ART_DIR.mkdir(parents=True, exist_ok=True)

MICRO_URL = os.getenv("TRIPOSR_URL", "http://127.0.0.1:8001")  # micro-API
RUNPY = Path.home() / "TripoSR" / "run.py"                     # fallback CLI

app = FastAPI(title="Fashion3D Gateway")

def _now_jobdir() -> Path:
    j = ART_DIR / time.strftime("%Y%m%d-%H%M%S")
    j.mkdir(parents=True, exist_ok=True)
    (j / "0").mkdir(parents=True, exist_ok=True)
    return j

def _save_upload_to_png(f: UploadFile, dst_png: Path) -> None:
    data = f.file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.save(dst_png)

def _try_micro_api(img_path: Path,
                   bake_texture: bool,
                   texture_resolution: int,
                   mc_resolution: int,
                   supersample: int) -> Tuple[bool, dict]:
    """Return (ok, payload)."""
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
        r = requests.post(url, files=files, data=data, timeout=600)
        ok = 200 <= r.status_code < 300
        return ok, (r.json() if r.headers.get("content-type","").startswith("application/json") else {"text": r.text, "status": r.status_code})
    except Exception as e:
        return False, {"error": f"{type(e).__name__}: {e}"}

def _run_cli(img_path: Path,
             outdir: Path,
             bake_texture: bool,
             texture_resolution: int,
             mc_resolution: int) -> Tuple[bool, str]:
    """Run ~/TripoSR/run.py as a fallback and capture stdout/stderr."""
    cmd = [
        "python", str(RUNPY),
        str(img_path),
        "--output-dir", str(outdir),
        "--mc-resolution", str(mc_resolution),
    ]
    if bake_texture:
        cmd += ["--bake-texture", "--texture-resolution", str(texture_resolution)]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    log = []
    for line in proc.stdout:  # stream logs
        log.append(line.rstrip())
    code = proc.wait()
    return code == 0, "\n".join(log)

def _collect_assets(jobdir: Path) -> dict:
    jd0 = jobdir / "0"
    # prefer post-processed mesh if present
    mesh_post = jd0 / "mesh_post.obj"
    mesh = jd0 / "mesh.obj"
    mtl  = jd0 / "mesh.mtl"
    albedo = jd0 / "albedo.png"
    input_png = jd0 / "input.png"
    return {
        "mesh_obj": str(mesh_post if mesh_post.exists() else (mesh if mesh.exists() else "")) or None,
        "mesh_mtl": str(mtl) if mtl.exists() else None,
        "albedo_png": str(albedo) if albedo.exists() else None,
        "input_png": str(input_png) if input_png.exists() else None,
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "artifacts": str(ART_DIR),
        "micro_api": MICRO_URL,
        "has_runpy": RUNPY.exists(),
    }

def _add_generate_route(path: str):
    @app.post(path)
    def generate(
        file: UploadFile = File(...),
        bake_texture: bool = Form(True),
        texture_resolution: int = Form(1024),
        mc_resolution: int = Form(384),
        supersample: int = Form(1),
    ):
        t0 = time.time()
        jobdir = _now_jobdir()
        job0 = jobdir / "0"
        input_png = job0 / "input.png"

        # 1) persist upload
        _save_upload_to_png(file, input_png)

        # 2) try micro-API first
        ok_micro, payload = _try_micro_api(
            input_png, bake_texture, texture_resolution, mc_resolution, supersample
        )
        log_tail = []
        if ok_micro and isinstance(payload, dict) and payload.get("success", payload.get("ok")):
            # micro API already wrote into its own job folder;
            # just return whatever it reported
            result = {
                "ok": True,
                "elapsed_sec": round(time.time() - t0, 3),
                "job_dir": payload.get("job_dir"),
                "assets": payload.get("assets", {}),
                "note": "served by micro-api",
                "log_tail": payload.get("log_tail", [])[-50:],
            }
            return JSONResponse(result)

        # 3) fallback CLI
        ok_cli, log = _run_cli(
            input_png, jobdir, bake_texture, texture_resolution, mc_resolution
        )
        assets = _collect_assets(jobdir)
        result = {
            "ok": ok_cli,
            "elapsed_sec": round(time.time() - t0, 3),
            "job_dir": str(jobdir),
            "assets": assets,
            "note": "served by fallback CLI",
            "log_tail": log.splitlines()[-100:],
        }
        return JSONResponse(result)

# register both … and with a trailing slash to avoid 404 confusion
_add_generate_route("/generate")
_add_generate_route("/generate/")

