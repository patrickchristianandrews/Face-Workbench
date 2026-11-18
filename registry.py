"""
FaceWorkbench Pro — Model Registry
Complete 16-family model registry with no-auth download, graceful offline skip,
and config.json persistence.
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("FaceWorkbench.registry")

# ---------------------------------------------------------------------------
# Path resolution (frozen vs source)
# ---------------------------------------------------------------------------
def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent.resolve()


BASE_DIR = get_base_dir()
MODELS_DIR = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
CONFIG_PATH = BASE_DIR / "config.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "crops").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "indexes").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "exports").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "match_threshold": float(os.environ.get("MATCH_THRESHOLD", 0.45)),
    "detection_size": 640,
    "primary_detector": "yunet",  # "scrfd" | "yunet"
    "cpu_threads": int(os.environ.get("CPU_THREADS", 4)),
    "lan_https": os.environ.get("LAN_HTTPS", "1") == "1",
    "active_packs": ["buffalo_l", "yunet", "sface"],
    "min_face_px": 64,
    "sharpness_threshold": 45.0,
    "brightness_low": 35,
    "brightness_high": 225,
    "yaw_limit": 35.0,
    "stacker_coefs": None,
    "model_weights": {},
}


def load_config() -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg.update(saved)
        except Exception as e:
            logger.warning("Failed to load config.json: %s", e)
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        logger.error("Failed to save config: %s", e)


# ---------------------------------------------------------------------------
# MODEL_REGISTRY — 16 families (feature 82-97)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Recognition — ArcFace family
    "antelopev2": {
        "family": "recognition",
        "filename": "glintr100.onnx",
        "dim": 512,
        "description": "ResNet100 ArcFace (antelopev2 / glintr100)",
        "urls": [
            "https://huggingface.co/deepghs/insightface/resolve/main/antelopev2/glintr100.onnx",
        ],
        "weight": 1.0,
    },
    "adaface_ir101": {
        "family": "recognition",
        "filename": "adaface_ir101.onnx",
        "dim": 512,
        "description": "AdaFace IR-101 Quality-Adaptive",
        "urls": [],
        "weight": 1.05,
    },
    "auraface_r100": {
        "family": "recognition",
        "filename": "auraface_r100.onnx",
        "dim": 512,
        "description": "AuraFace ResNet100 (Apache 2.0)",
        "urls": [
            "https://huggingface.co/fal/AuraFace-v1/resolve/main/model.onnx",
        ],
        "weight": 1.0,
    },
    "buffalo_l": {
        "family": "recognition",
        "filename": "w600k_r50.onnx",
        "dim": 512,
        "description": "buffalo_l ArcFace ResNet50 WebFace600K",
        "urls": [
            "https://huggingface.co/globalnebula/insightface-buffalo-l-onnx/resolve/main/arcface_w600k_r50.onnx",
            "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/insightface/models/buffalo_l/w600k_r50.onnx",
        ],
        "weight": 1.0,
    },
    "buffalo_s": {
        "family": "recognition",
        "filename": "w600k_mbf.onnx",
        "dim": 512,
        "description": "buffalo_s MobileFaceNet lightweight",
        "urls": [],
        "weight": 0.9,
    },
    "buffalo_sc": {
        "family": "recognition",
        "filename": "buffalo_sc_rec.onnx",
        "dim": 512,
        "description": "buffalo_sc compact CPU pack",
        "urls": [],
        "weight": 0.85,
    },
    "elasticface": {
        "family": "recognition",
        "filename": "elasticface.onnx",
        "dim": 512,
        "description": "ElasticFace Elastic Margin",
        "urls": [],
        "weight": 1.0,
    },
    "magface": {
        "family": "recognition",
        "filename": "magface.onnx",
        "dim": 512,
        "description": "MagFace Magnitude-Aware",
        "urls": [],
        "weight": 1.0,
    },
    "sface": {
        "family": "recognition",
        "filename": "face_recognition_sface_2021dec.onnx",
        "dim": 128,
        "description": "OpenCV Zoo SFace (Apache 2.0, 128-D)",
        "urls": [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ],
        "weight": 0.95,
    },
    # Detectors
    "scrfd_det10g": {
        "family": "detection",
        "filename": "det_10g.onnx",
        "dim": 0,
        "description": "SCRFD det_10g multi-scale face detector",
        "urls": [
            "https://huggingface.co/globalnebula/insightface-buffalo-l-onnx/resolve/main/scrfd_10g.onnx",
            "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/insightface/models/buffalo_l/det_10g.onnx",
        ],
        "weight": 1.0,
    },
    "yunet": {
        "family": "detection",
        "filename": "face_detection_yunet_2023mar.onnx",
        "dim": 0,
        "description": "YuNet fast face detector (OpenCV Zoo)",
        "urls": [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        ],
        "weight": 1.0,
    },
    # Landmarks
    "landmark_2d106": {
        "family": "landmark",
        "filename": "2d106det.onnx",
        "dim": 0,
        "description": "106-point dense 2D landmarks",
        "urls": [
            "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_l/2d106det.onnx",
        ],
        "weight": 1.0,
    },
    "landmark_1k3d68": {
        "family": "landmark",
        "filename": "1k3d68.onnx",
        "dim": 0,
        "description": "3D 68-point facial landmark regressor",
        "urls": [
            "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_l/1k3d68.onnx",
        ],
        "weight": 1.0,
    },
    # Attributes
    "genderage": {
        "family": "attribute",
        "filename": "genderage.onnx",
        "dim": 0,
        "description": "Age & gender estimation",
        "urls": [
            "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_l/genderage.onnx",
        ],
        "weight": 1.0,
    },
    "emotion": {
        "family": "attribute",
        "filename": "emotion-ferplus-8.onnx",
        "dim": 0,
        "description": "8-class facial expression (FER+)",
        "urls": [
            "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        ],
        "weight": 1.0,
    },
    "mask": {
        "family": "attribute",
        "filename": "mask_detector.onnx",
        "dim": 0,
        "description": "Facial mask / occlusion classifier",
        "urls": [],
        "weight": 1.0,
    },
    # Liveness
    "minifasnet": {
        "family": "liveness",
        "filename": "minifasnet_v2.onnx",
        "dim": 0,
        "description": "MiniFASNetV2 Silent-Face anti-spoofing",
        "urls": [
            "https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/2.7_80x80_MiniFASNetV2.onnx",
        ],
        "weight": 1.0,
    },
}


class ModelStatus:
    LOADED = "Loaded"
    READY = "Ready to Fetch"
    OFFLINE = "Offline Skip"
    ERROR = "Error"


def model_path(key: str) -> Path:
    meta = MODEL_REGISTRY[key]
    return MODELS_DIR / meta["filename"]


def is_model_available(key: str) -> bool:
    return model_path(key).exists() and model_path(key).stat().st_size > 1024


def get_registry_status() -> List[Dict[str, Any]]:
    rows = []
    for key, meta in MODEL_REGISTRY.items():
        available = is_model_available(key)
        status = ModelStatus.LOADED if available else (
            ModelStatus.READY if meta.get("urls") else ModelStatus.OFFLINE
        )
        rows.append({
            "key": key,
            "family": meta["family"],
            "filename": meta["filename"],
            "dim": meta["dim"],
            "description": meta["description"],
            "weight": meta.get("weight", 1.0),
            "available": available,
            "status": status,
            "has_urls": bool(meta.get("urls")),
        })
    return rows


async def download_model(key: str, progress_cb=None) -> bool:
    """Download a model from the first working public URL. Returns True on success."""
    if key not in MODEL_REGISTRY:
        return False
    meta = MODEL_REGISTRY[key]
    dest = model_path(key)
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    urls = meta.get("urls") or []
    if not urls:
        logger.warning("No download URLs for model %s — offline skip", key)
        return False

    for url in urls:
        try:
            logger.info("Downloading %s from %s", key, url)
            async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        logger.warning("HTTP %s for %s", resp.status_code, url)
                        continue
                    total = int(resp.headers.get("content-length", 0))
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(".part")
                    downloaded = 0
                    with open(tmp, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_cb and total:
                                progress_cb(key, downloaded / total)
                    tmp.rename(dest)
                    logger.info("Downloaded %s → %s (%d bytes)", key, dest, downloaded)
                    return True
        except Exception as e:
            logger.warning("Download failed for %s from %s: %s", key, url, e)
            continue
    return False


def ensure_core_models_sync() -> None:
    """Best-effort synchronous download of core models (used at startup)."""
    import urllib.request
    core = ["yunet", "sface", "buffalo_l", "scrfd_det10g"]
    for key in core:
        if is_model_available(key):
            continue
        meta = MODEL_REGISTRY.get(key, {})
        for url in meta.get("urls") or []:
            try:
                dest = model_path(key)
                logger.info("Fetching core model %s …", key)
                urllib.request.urlretrieve(url, str(dest))
                if dest.stat().st_size > 1024:
                    logger.info("Got %s", key)
                    break
            except Exception as e:
                logger.debug("Could not fetch %s: %s", key, e)
                continue
