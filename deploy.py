import argparse, subprocess, sys, os

def setup():
    print("Run the following commands:")
    print("  conda env update -f environment.yaml")
    print("  conda activate fashion3d")

def test():
    """Test basic dependencies for the API"""
    try:
        import fastapi, uvicorn, requests
        from PIL import Image
        print("✅ All API dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def api():
    """Start the FastAPI server"""
    subprocess.run([sys.executable, "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8001", "--reload"], check=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true", help="Show setup instructions")
    ap.add_argument("--test", action="store_true", help="Test API dependencies")
    ap.add_argument("--api", action="store_true", help="Start the API server")
    args = ap.parse_args()
    
    if not any(vars(args).values()):
        print("Usage: python deploy.py [--setup|--test|--api]")
        ap.print_help()
    
    if args.setup: setup()
    if args.test: test()
    if args.api: api()
