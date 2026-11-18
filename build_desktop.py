"""FaceWorkbench Pro — PyInstaller build (Windows / macOS / Linux)."""
from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def main():
    try:
        import PyInstaller.__main__ as pyi
    except ImportError:
        print("Install pyinstaller: pip install pyinstaller")
        sys.exit(1)

    sep = ";" if platform.system() == "Windows" else ":"
    datas = [f"{ROOT / 'static'}{sep}static"]
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
        args.append("--windowed")  # no console on Windows release
    else:
        args.append("--console")

    for d in datas:
        args.append(f"--add-data={d}")

    print("PyInstaller:", " ".join(args[:6]), "...")
    pyi.run(args)
    out = ROOT / "dist" / "FaceWorkbench"
    print("=" * 50)
    print("OUTPUT:", out)
    if (out / "FaceWorkbench.exe").exists():
        print("EXE:", out / "FaceWorkbench.exe")
    print("=" * 50)


if __name__ == "__main__":
    main()
