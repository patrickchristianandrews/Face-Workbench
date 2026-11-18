"""FaceWorkbench Pro — PyInstaller build (Windows / macOS / Linux)."""
from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def ensure_static() -> Path:
    """Ensure static/ exists with UI assets (supports flat-upload layout)."""
    static = ROOT / "static"
    static.mkdir(exist_ok=True)
    mapping = {
        "index.html": "index.html",
        "app.js": "app.js",
        "styles.css": "styles.css",
        "manifest.json": "manifest.json",
        "sw.js": "sw.js",
    }
    for name, root_name in mapping.items():
        dest = static / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        src = ROOT / root_name
        if src.exists():
            shutil.copy2(src, dest)
            print(f"Copied {root_name} -> static/{name}")
    if not (static / "index.html").exists():
        print("WARNING: static/index.html missing")
    return static


def main():
    try:
        import PyInstaller.__main__ as pyi
    except ImportError:
        print("Install pyinstaller: pip install pyinstaller")
        sys.exit(1)

    static = ensure_static()
    sep = ";" if platform.system() == "Windows" else ":"
    datas = [f"{static}{sep}static"]
    models = ROOT / "models"
    if models.is_dir() and any(models.iterdir()):
        datas.append(f"{models}{sep}models")

    entry = ROOT / "run_exe.py"
    if not entry.exists():
        entry = ROOT / "desktop.py"

    args = [
        str(entry),
        "--name=FaceWorkbench",
        "--onedir",
        "--noconfirm",
        "--clean",
        f"--distpath={ROOT / 'dist'}",
        f"--workpath={ROOT / 'build'}",
        f"--specpath={ROOT}",
        "--hidden-import=uvicorn.logging",
        "--hidden-import=uvicorn.loops.auto",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=uvicorn.lifespan.on",
        "--hidden-import=app",
        "--hidden-import=engine",
        "--hidden-import=database",
        "--hidden-import=registry",
        "--hidden-import=calibration",
        "--hidden-import=video_pipeline",
        "--hidden-import=osint_helper",
        "--exclude-module=torch",
        "--exclude-module=torchvision",
        "--exclude-module=matplotlib",
        "--exclude-module=tkinter",
        "--exclude-module=nvidia",
    ]
    if platform.system() == "Windows":
        args.append("--console")  # show console so first-run errors are visible
    else:
        args.append("--console")

    for d in datas:
        args.append(f"--add-data={d}")

    print("PyInstaller:", " ".join(args[:6]), "...")
    pyi.run(args)
    out = ROOT / "dist" / "FaceWorkbench"
    print("=" * 50)
    print("OUTPUT:", out)
    exe = out / ("FaceWorkbench.exe" if platform.system() == "Windows" else "FaceWorkbench")
    if exe.exists():
        print("EXE:", exe)
    else:
        print("WARNING: expected binary not found; listing dist:")
        if out.exists():
            for p in out.iterdir():
                print(" ", p.name)
    print("=" * 50)


if __name__ == "__main__":
    main()
