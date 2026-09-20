"""
Verify root run.py startup
"""
import subprocess
import time
import urllib.request
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
proc = subprocess.Popen([sys.executable, "run.py"], cwd=root_dir)
try:
    success = False
    for _ in range(15):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/api/v1/status", timeout=1.0) as r:
                if r.status == 200:
                    success = True
                    break
        except Exception:
            pass
    assert success, "Root run.py server failed to respond within 7.5s"
    print("[PASS] Root run.py starts up and serves /api/v1/status successfully!")
finally:
    proc.terminate()
    proc.wait()
