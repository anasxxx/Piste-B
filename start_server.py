#!/usr/bin/env python3
"""
Reliable server startup script
"""
import sys
import os
import uvicorn

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from api.server import app
    print("🚀 Starting Fashion3D API Server...")
    print("📍 Server will be available at: http://localhost:8001")
    print("🔍 Health check: http://localhost:8001/health")
    print("📤 Generate endpoint: http://localhost:8001/generate")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001, 
        reload=False,  # Disable reload for stability
        log_level="info"
    )
    
except KeyboardInterrupt:
    print("\n🛑 Server stopped by user")
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
