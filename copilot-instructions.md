# FaceWorkbench Pro — Copilot Agent Instructions

1. Zero placeholders: implement complete functions, never leave `TODO` / `pass` stubs for core paths.
2. Frozen-path awareness: use `getattr(sys, 'frozen', False)` and resolve `./models`, `./data`, `config.json` relative to `Path(sys.executable).parent`; static UI from `sys._MEIPASS`.
3. Model registry: always graceful-skip missing ONNX weights; never crash the app.
4. After changes, prefer `python build_desktop.py` for packaging verification.
