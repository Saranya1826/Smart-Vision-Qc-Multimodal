"""
SmartVisionOC Native Desktop Application Coordinator
Launches SmartVisionOC in dedicated standalone desktop application mode (Zero-URL, No Address Bar).
"""
import os
import sys
import time
import socket
import threading
import urllib.request
import subprocess
import tempfile
import uvicorn

# Ensure application directory is in python module path and working directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def find_free_port() -> int:
    """Finds an available ephemeral internal port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread(threading.Thread):
    """Runs embedded Uvicorn server inside a background thread on a private internal port."""
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        config = uvicorn.Config(
            "backend.main:app",
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            access_log=False
        )
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


def wait_for_server(url: str, timeout: float = 15.0) -> bool:
    """Polls backend health status endpoint until ready or timeout expires."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.15)
    return False


def get_browser_executable() -> str:
    """Finds system Edge or Chrome executable for native --app mode."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    # Allocate dynamic private internal port
    app_port = find_free_port()
    app_url = f"http://127.0.0.1:{app_port}"
    status_url = f"{app_url}/api/v1/status"

    server_thread = ServerThread(port=app_port)
    server_thread.start()

    if not wait_for_server(status_url, timeout=15.0):
        print("[!] Failed to start backend service.")
        sys.exit(1)

    browser_exe = get_browser_executable()
    if browser_exe:
        # Launch dedicated desktop app window (no address bar, no browser tabs, no navigation)
        temp_profile = os.path.join(tempfile.gettempdir(), "SmartVisionOC_Profile")
        app_args = [
            browser_exe,
            f"--app={app_url}",
            f"--user-data-dir={temp_profile}",
            "--window-size=1440,920",
            "--window-position=50,50",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--app-id=SmartVisionOC"
        ]
        try:
            proc = subprocess.Popen(app_args)
            proc.wait()
        finally:
            server_thread.stop()
    else:
        # Fallback to webview if no chromium browser is found
        try:
            import webview
            window = webview.create_window(
                title="SmartVisionOC — Industrial Quality Inspection System",
                url=app_url,
                width=1440,
                height=920
            )
            webview.start()
        finally:
            server_thread.stop()


if __name__ == "__main__":
    main()
