#!/usr/bin/env python3
"""
Simple launcher for Fashion3D API
"""
import subprocess
import sys
import os

def main():
    print("Starting Fashion3D API...")
    
    # Change to project directory
    project_dir = os.path.expanduser("~/fashion3d")
    os.chdir(project_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Start the API server
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "api.server:app", 
        "--host", "0.0.0.0", 
        "--port", "8002",
        "--reload"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down API server...")
    except Exception as e:
        print(f"Error starting API: {e}")

if __name__ == "__main__":
    main()
