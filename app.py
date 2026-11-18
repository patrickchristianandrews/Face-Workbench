"""
FaceWorkbench Pro — FastAPI backend, static server, WebSocket, LAN HTTPS
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from registry import (
    BASE_DIR,
    DATA_DIR,
    MODEL_REGISTRY,
    download_model,
    get_registry_status,
    is_model_available,
    load_config,
    save_config,
    ensure_core_models_sync,
)
from engine import get_engine
from database import (
    add_face,
    create_person,
    delete_person,
    export_gallery_zip,
    get_faces_for_person,
    get_person,
    get_person_id_for_face,
    list_people,
    merge_people,
    rebuild_all_indexes,
    rename_person,
    search_index,
    CROPS_DIR,
)
from calibration import calibrate_threshold, hard_negative_mining, stacker_predict, train_stacker
from video_pipeline import process_video
from osint_helper import extract_exif, launch_search, list_engines, open_urls

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("FaceWorkbench.app")

# Static assets: from MEIPASS when frozen, else local static/
if getattr(sys, "frozen", False):
    STATIC_DIR = Path(sys._MEIPASS) / "static"
else:
    STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="FaceWorkbench Pro", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def image_from_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    return img


def encode_image_b64(img: np.ndarray, ext: str = ".jpg") -> str:
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def save_crop(img: np.ndarray, person_id: str, face_id: str) -> str:
    folder = CROPS_DIR / person_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{face_id}.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(path)


# ---------------------------------------------------------------------------
# Routes — System
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"name": "FaceWorkbench Pro", "status": "ok", "ui": "missing"})


@app.get("/api/health")
async def health():
    cfg = load_config()
    return {
        "status": "ok",
        "lan_ip": get_lan_ip(),
        "threshold": cfg.get("match_threshold"),
        "detector": cfg.get("primary_detector"),
        "models_loaded": sum(1 for k in MODEL_REGISTRY if is_model_available(k)),
    }


@app.get("/api/config")
async def get_config():
    return load_config()


class ConfigUpdate(BaseModel):
    match_threshold: Optional[float] = None
    detection_size: Optional[int] = None
    primary_detector: Optional[str] = None
    cpu_threads: Optional[int] = None
    active_packs: Optional[List[str]] = None
    min_face_px: Optional[int] = None
    sharpness_threshold: Optional[float] = None


@app.post("/api/config")
async def update_config(body: ConfigUpdate):
    cfg = load_config()
    data = body.dict(exclude_none=True)
    cfg.update(data)
    save_config(cfg)
    get_engine().reload()
    return cfg


@app.get("/api/models")
async def models_status():
    return get_registry_status()


@app.post("/api/models/{key}/download")
async def api_download_model(key: str):
    ok = await download_model(key)
    if ok:
        get_engine().reload()
    return {"ok": ok, "available": is_model_available(key)}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    img = image_from_upload(data)
    engine = get_engine()
    results = engine.analyze(img)
    # attach aligned preview
    for r in results:
        bbox = r["bbox"]
        kps = r.get("landmarks5")
        if kps:
            aligned = engine.align_face(img, kps)
            r["aligned_b64"] = encode_image_b64(aligned)
        r["preview_b64"] = encode_image_b64(
            img[
                max(0, int(bbox[1])) : int(bbox[1] + bbox[3]),
                max(0, int(bbox[0])) : int(bbox[0] + bbox[2]),
            ]
        )
        # drop raw embeddings from response size (keep keys)
        if "embeddings" in r:
            r["embedding_keys"] = list(r["embeddings"].keys())
            # keep embeddings for client-side if small
    return {"faces": results, "image_size": [img.shape[1], img.shape[0]]}


@app.post("/api/search")
async def search(file: UploadFile = File(...), top_k: int = 10):
    data = await file.read()
    img = image_from_upload(data)
    engine = get_engine()
    faces = engine.analyze(img)
    if not faces:
        return {"matches": [], "probe": None}

    probe = faces[0]
    embs = {k: np.array(v, dtype=np.float32) for k, v in probe.get("embeddings", {}).items()}
    cfg = load_config()
    threshold = float(cfg.get("match_threshold", 0.45))

    # fuse across models
    candidate_scores: Dict[str, Dict[str, float]] = {}  # face_id -> {model: score}
    for mk, emb in embs.items():
        hits = search_index(mk, emb, top_k=top_k * 3)
        for face_id, score in hits:
            candidate_scores.setdefault(face_id, {})[mk] = score

    # weighted fusion
    weights = {k: MODEL_REGISTRY.get(k, {}).get("weight", 1.0) for k in embs}
    fused = []
    for face_id, scores in candidate_scores.items():
        wsum = sum(weights.get(m, 1.0) for m in scores)
        fscore = sum(scores[m] * weights.get(m, 1.0) for m in scores) / (wsum or 1)
        person_id = get_person_id_for_face(face_id)
        person = get_person(person_id) if person_id else None
        fused.append({
            "face_id": face_id,
            "person_id": person_id,
            "person_name": person["name"] if person else "Unknown",
            "fused_score": float(fscore),
            "per_model": scores,
            "above_threshold": fscore >= threshold,
            "stacker_prob": stacker_predict(list(scores.values())),
        })
    fused.sort(key=lambda x: -x["fused_score"])
    fused = fused[:top_k]

    return {
        "probe": {
            "quality": probe.get("quality"),
            "attributes": probe.get("attributes"),
            "aligned_b64": probe.get("aligned_b64"),
            "bbox": probe.get("bbox"),
        },
        "matches": fused,
        "threshold": threshold,
    }


@app.post("/api/verify")
async def verify(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    img_a = image_from_upload(await file_a.read())
    img_b = image_from_upload(await file_b.read())
    engine = get_engine()
    fa = engine.analyze(img_a)
    fb = engine.analyze(img_b)
    if not fa or not fb:
        return {"ok": False, "message": "Need a detectable face in both images"}

    ea = {k: np.array(v, dtype=np.float32) for k, v in fa[0].get("embeddings", {}).items()}
    eb = {k: np.array(v, dtype=np.float32) for k, v in fb[0].get("embeddings", {}).items()}
    common = set(ea) & set(eb)
    per_model = {}
    scores = []
    for mk in common:
        s = engine.cosine(ea[mk], eb[mk])
        per_model[mk] = {
            "cosine": s,
            "weight": MODEL_REGISTRY.get(mk, {}).get("weight", 1.0),
        }
        scores.append(s)

    cfg = load_config()
    thr = float(cfg.get("match_threshold", 0.45))
    if scores:
        weights = [per_model[m]["weight"] for m in common]
        fused = sum(s * w for s, w in zip(scores, weights)) / (sum(weights) or 1)
    else:
        fused = 0.0

    return {
        "ok": True,
        "fused_score": float(fused),
        "stacker_prob": stacker_predict(scores),
        "match": fused >= thr,
        "threshold": thr,
        "per_model": per_model,
        "face_a": {"quality": fa[0].get("quality"), "attributes": fa[0].get("attributes"), "aligned_b64": fa[0].get("aligned_b64")},
        "face_b": {"quality": fb[0].get("quality"), "attributes": fb[0].get("attributes"), "aligned_b64": fb[0].get("aligned_b64")},
    }


# ---------------------------------------------------------------------------
# Gallery / Identities
# ---------------------------------------------------------------------------
@app.get("/api/people")
async def api_list_people(q: str = ""):
    return list_people(q)


@app.post("/api/people")
async def api_create_person(name: str = Form(...), notes: str = Form("")):
    pid = create_person(name, notes)
    return {"id": pid, "name": name}


@app.get("/api/people/{person_id}")
async def api_get_person(person_id: str):
    p = get_person(person_id)
    if not p:
        raise HTTPException(404)
    faces = get_faces_for_person(person_id)
    return {"person": p, "faces": faces}


@app.patch("/api/people/{person_id}")
async def api_rename(person_id: str, name: str = Form(...)):
    ok = rename_person(person_id, name)
    return {"ok": ok}


@app.delete("/api/people/{person_id}")
async def api_delete_person(person_id: str):
    return {"ok": delete_person(person_id)}


@app.post("/api/people/merge")
async def api_merge(source_id: str = Form(...), target_id: str = Form(...)):
    return {"ok": merge_people(source_id, target_id)}


@app.post("/api/enroll")
async def enroll(
    person_id: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
):
    if not person_id:
        if not name:
            raise HTTPException(400, "Provide person_id or name")
        person_id = create_person(name)

    engine = get_engine()
    enrolled = []
    for f in files:
        data = await f.read()
        img = image_from_upload(data)
        faces = engine.analyze(img)
        if not faces:
            continue
        # use highest quality face
        faces.sort(key=lambda x: x.get("quality", {}).get("quality", 0), reverse=True)
        face = faces[0]
        kps = face.get("landmarks5")
        aligned = engine.align_face(img, kps) if kps else img
        embs = {k: np.array(v, dtype=np.float32) for k, v in face.get("embeddings", {}).items()}
        face_id = str(uuid.uuid4())
        crop_path = save_crop(aligned, person_id, face_id)
        meta = {
            **face.get("quality", {}),
            **face.get("attributes", {}),
            "bbox": face.get("bbox"),
            "landmarks5": face.get("landmarks5"),
        }
        # overwrite face_id from add_face
        real_id = add_face(person_id, crop_path, embs, face.get("quality", {}).get("quality", 0), meta)
        enrolled.append({"face_id": real_id, "quality": meta.get("quality")})

    return {"person_id": person_id, "enrolled": enrolled, "count": len(enrolled)}


@app.post("/api/indexes/rebuild")
async def api_rebuild():
    n = rebuild_all_indexes()
    return {"rebuilt": n}


@app.get("/api/export/gallery")
async def api_export_gallery():
    path = export_gallery_zip()
    return FileResponse(str(path), filename=path.name, media_type="application/zip")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@app.post("/api/calibrate")
async def api_calibrate(target_far: float = 0.001, model_key: str = "sface"):
    return calibrate_threshold(model_key, target_far)


@app.post("/api/stacker/train")
async def api_train_stacker():
    return train_stacker()


@app.get("/api/hard-negatives")
async def api_hard_neg(threshold: float = 0.4):
    return hard_negative_mining(threshold)


# ---------------------------------------------------------------------------
# Video redaction
# ---------------------------------------------------------------------------
@app.post("/api/video/redact")
async def api_redact(
    file: UploadFile = File(...),
    mode: str = Form("blur_all"),
    target_ids: str = Form(""),  # comma-separated
):
    raw = await file.read()
    tmp = DATA_DIR / "exports" / f"upload_{uuid.uuid4().hex}.mp4"
    tmp.write_bytes(raw)
    targets: Set[str] = {x.strip() for x in target_ids.split(",") if x.strip()}
    try:
        out = process_video(str(tmp), mode=mode, target_person_ids=targets, max_frames=900)
        return FileResponse(str(out), filename=out.name, media_type="video/mp4")
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OSINT
# ---------------------------------------------------------------------------
@app.get("/api/osint/engines")
async def api_engines():
    return list_engines()


@app.post("/api/osint/exif")
async def api_exif(file: UploadFile = File(...)):
    raw = await file.read()
    tmp = DATA_DIR / "exports" / f"exif_{uuid.uuid4().hex}.jpg"
    tmp.write_bytes(raw)
    try:
        return extract_exif(str(tmp))
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


@app.post("/api/osint/open")
async def api_open_urls(urls: str = Form(...)):
    url_list = [u.strip() for u in urls.replace("\n", ",").split(",") if u.strip()]
    n = open_urls(url_list)
    return {"opened": n}


# ---------------------------------------------------------------------------
# WebSocket live (webcam frames from browser)
# ---------------------------------------------------------------------------
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    engine = get_engine()
    try:
        while True:
            msg = await ws.receive_text()
            # expect data URL or base64 jpeg
            if "," in msg:
                msg = msg.split(",", 1)[1]
            try:
                data = base64.b64decode(msg)
                img = image_from_upload(data)
                faces = engine.analyze(img)
                # lightweight response
                payload = []
                for f in faces:
                    payload.append({
                        "bbox": f["bbox"],
                        "score": f["score"],
                        "landmarks5": f.get("landmarks5"),
                        "quality": f.get("quality", {}).get("quality"),
                        "attributes": f.get("attributes"),
                    })
                await ws.send_json({"faces": payload})
            except Exception as e:
                await ws.send_json({"error": str(e)})
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Startup: try to fetch core models
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    logger.info("FaceWorkbench Pro starting…")
    ensure_core_models_sync()
    get_engine()  # warm up
    logger.info("LAN IP: %s", get_lan_ip())


def create_self_signed_cert(cert_path: Path, key_path: Path, ip: str) -> None:
    """Generate a 365-day self-signed cert with SANs for localhost + LAN IP."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, u"FaceWorkbench Pro"),
    ])
    alt = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(__import__("ipaddress").IPv4Address("127.0.0.1")),
        x509.IPAddress(__import__("ipaddress").IPv4Address(ip)),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(alt, critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def run_server(host: str = "127.0.0.1", port: int = 0) -> tuple:
    """Start uvicorn and return (actual_port, server). Used by desktop.py."""
    import uvicorn
    cfg = load_config()
    lan_https = cfg.get("lan_https", True) and host in ("0.0.0.0",)
    ssl_keyfile = ssl_certfile = None
    if lan_https:
        cert = BASE_DIR / "cert.pem"
        key = BASE_DIR / "key.pem"
        if not cert.exists() or not key.exists():
            create_self_signed_cert(cert, key, get_lan_ip())
        ssl_certfile = str(cert)
        ssl_keyfile = str(key)

    config = uvicorn.Config(
        app, host=host, port=port or 8765, log_level="info",
        ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile,
    )
    server = uvicorn.Server(config)
    return server
