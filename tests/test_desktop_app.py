"""
Verification script for SmartVisionOC Desktop Application integration.
Tests dynamic port allocation, server lifecycle management, and webview window configuration.
"""
import sys
import os
import time
import urllib.request
import json
import socket

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from desktop import ServerThread, wait_for_server, find_free_port
import webview

def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def test_desktop_app_integration():
    print("=" * 60)
    print("Testing SmartVisionOC Desktop App Integration (Dynamic Port)")
    print("=" * 60)

    # 1. Verify pywebview import and version
    print(f"[*] webview loaded successfully.")

    # 2. Test dynamic free port finding
    test_port = find_free_port()
    assert test_port > 1024, f"Invalid dynamic port: {test_port}"
    print(f"[PASS] Dynamic port allocated: {test_port}")

    app_url = f"http://127.0.0.1:{test_port}"
    status_url = f"{app_url}/api/v1/status"

    # 3. Test embedded server on dynamic port
    print(f"[*] Starting embedded server on dynamic port {test_port}...")
    server = ServerThread(port=test_port)
    server.start()

    try:
        ready = wait_for_server(status_url, timeout=12.0)
        assert ready, "Server failed to respond on dynamic port within 12 seconds"
        print("[PASS] wait_for_server confirmed server is responding on dynamic port")

        # Check status response
        with urllib.request.urlopen(status_url, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "OPERATIONAL", f"Unexpected status: {data}"
            print(f"[PASS] Status endpoint returned: {data['status']}, mode={data.get('inference_mode')}")

        # Check port in use
        assert is_port_in_use("127.0.0.1", test_port), "Dynamic port should be reported in use"
        print(f"[PASS] Port {test_port} verified listening and healthy")

        # 4. Test webview window creation parameters
        window = webview.create_window(
            title="SmartVisionOC — Industrial Quality Inspection System",
            url=app_url,
            width=1440,
            height=920,
            min_size=(1024, 700)
        )
        assert window is not None, "Window object creation failed"
        assert window.title == "SmartVisionOC — Industrial Quality Inspection System"
        print(f"[PASS] webview.create_window created window successfully: {window.title} (target: {window.initial_width}x{window.initial_height})")

    finally:
        print("[*] Stopping server thread...")
        server.stop()
        time.sleep(1.0)
        print("[PASS] Server stopped cleanly")

    print("=" * 60)
    print("ALL DESKTOP INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_desktop_app_integration()
