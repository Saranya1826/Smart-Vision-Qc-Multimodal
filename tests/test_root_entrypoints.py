"""
Test root entry points run.py and desktop.py
"""
import os
import sys

def test_root_entrypoints():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Testing root entrypoints in: {root_dir}")
    
    # Check that root run.py and desktop.py exist
    assert os.path.exists(os.path.join(root_dir, "run.py")), "run.py missing in root"
    assert os.path.exists(os.path.join(root_dir, "desktop.py")), "desktop.py missing in root"
    assert os.path.exists(os.path.join(root_dir, "SmartVisionOC.bat")), "SmartVisionOC.bat missing in root"
    assert os.path.exists(os.path.join(root_dir, "requirements.txt")), "requirements.txt missing in root"
    print("[PASS] All root entry files exist.")

if __name__ == "__main__":
    test_root_entrypoints()
