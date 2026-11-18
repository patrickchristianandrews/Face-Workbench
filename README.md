# FaceWorkbench Pro

Offline-first biometric & facial recognition desktop app  
(Python · FastAPI · ONNX · FAISS · PyInstaller)

---

## Get the Windows EXE (no Python on your PC)

### 1. Put this project on GitHub

1. Create a new **empty** repository on [github.com](https://github.com/new) (no README).
2. Upload this folder, **or** use GitHub Desktop / `git push`.

### 2. Build with GitHub Actions

1. Open the repo on GitHub → **Actions** tab.
2. Select **Build Windows EXE**.
3. Click **Run workflow** → **Run workflow**.
4. Wait ~8–15 minutes.
5. Open the finished run → **Artifacts**.
6. Download **FaceWorkbench-Windows**.
7. Unzip → run **`FaceWorkbench.exe`**.

You never need Python installed locally.

---

## Run from source (if you have Python)

```bash
pip install -r requirements.txt
python desktop.py
# or: python run_exe.py
```

## Build EXE locally (Windows + Python)

```bash
pip install -r requirements.txt pyinstaller
python build_desktop.py
# → dist\FaceWorkbench\FaceWorkbench.exe
```

## Models

Core ONNX weights live in `models/`. Missing models are skipped; the UI can fetch more when online.

## License

App code is a research reference. Upstream model licenses stay with their authors.
