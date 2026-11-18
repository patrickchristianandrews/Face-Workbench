"""
FaceWorkbench Pro — Native PyWebView desktop launcher
Starts FastAPI in a background thread and opens a native window.
"""
from __future__ import annotations

import logging
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("FaceWorkbench.desktop")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DesktopBridge:
    """JS API exposed to the frontend for native OS dialogs."""

    def save_file_dialog(self, filename: str = "export.zip", file_types: tuple = ("All files (*.*)",)) -> str:
        try:
            import webview
            result = webview.windows[0].create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename=filename,
                file_types=file_types,
            )
            if result and len(result) > 0:
                return result if isinstance(result, str) else result[0]
        except Exception as e:
            logger.warning("save_file_dialog: %s", e)
        return ""

    def select_folder_dialog(self) -> str:
        try:
            import webview
            result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
            if result and len(result) > 0:
                return result[0]
        except Exception as e:
            logger.warning("select_folder_dialog: %s", e)
        return ""

    def open_file_dialog(self, file_types: tuple = ("Image files (*.jpg;*.jpeg;*.png;*.webp)",)) -> list:
        try:
            import webview
            result = webview.windows[0].create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=True,
                file_types=file_types,
            )
            return list(result) if result else []
        except Exception as e:
            logger.warning("open_file_dialog: %s", e)
        return []

    def open_external_url(self, url: str) -> bool:
        try:
            webbrowser.open(url)
            return True
        except Exception:
            return False


def start_api(port: int):
    import uvicorn
    from app import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)


def main():
    port = find_free_port()
    logger.info("Starting API on port %d", port)

    t = threading.Thread(target=start_api, args=(port,), daemon=True)
    t.start()

    # wait until server is up
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    else:
        logger.error("API failed to start")
        sys.exit(1)

    url = f"http://127.0.0.1:{port}/"
    logger.info("Opening desktop window → %s", url)

    try:
        import webview
        bridge = DesktopBridge()
        webview.create_window(
            "FaceWorkbench Pro",
            url,
            width=1360,
            height=860,
            min_size=(1024, 680),
            js_api=bridge,
            background_color="#070B14",
        )
        webview.start(debug=False)
    except Exception as e:
        logger.error("PyWebView failed (%s). Falling back to browser.", e)
        webbrowser.open(url)
        # keep process alive while server runs
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
