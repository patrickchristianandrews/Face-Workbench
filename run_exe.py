"""Minimal entry: start FaceWorkbench API and open browser. No PyWebView."""
from __future__ import annotations
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def main():
    # Ensure frozen paths work
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
        sys.path.insert(0, str(base))
    port = free_port()
    def serve():
        import uvicorn
        from app import app
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    for _ in range(60):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.15)
    url = f"http://127.0.0.1:{port}/"
    print(f"FaceWorkbench Pro running at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
