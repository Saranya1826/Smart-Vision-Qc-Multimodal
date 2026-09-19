"""
SmartVisionOC Startup Script
Starts the FastAPI backend and serves the frontend dashboard at http://127.0.0.1:8000
"""
import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if curr_dir not in sys.path:
        sys.path.insert(0, curr_dir)

    print("=" * 70)
    print(" SmartVisionOC: Explainable Multi-Modal AI Industrial Inspection")
    print("=" * 70)
    print(" [MODE] Simulated AI Inference Engine Active (Pluggable Architecture)")
    print(" [URL]  http://127.0.0.1:8000")
    print(" Press Ctrl+C to terminate.")
    print("=" * 70)

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
