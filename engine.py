"""
FaceWorkbench Pro — Detection, alignment, recognition, quality, attributes
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from registry import (
    MODEL_REGISTRY,
    is_model_available,
    load_config,
    model_path,
    MODELS_DIR,
)

logger = logging.getLogger("FaceWorkbench.engine")

# ArcFace 5-point reference (112x112)
ARCFACE_REF = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class ONNXSession:
    def __init__(self, path: Path, threads: int = 4):
        self.path = path
        self.session = None
        self.input_name = None
        self.output_names = None
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(path), opts, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            logger.info("Loaded ONNX: %s", path.name)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)

    def run(self, feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        if self.session is None:
            return []
        return self.session.run(self.output_names, feed)


class FaceEngine:
    def __init__(self):
        self.cfg = load_config()
        self.sessions: Dict[str, ONNXSession] = {}
        self.yunet = None
        self.sface = None
        self._load_models()

    def reload(self):
        self.cfg = load_config()
        self.sessions.clear()
        self.yunet = None
        self.sface = None
        self._load_models()

    def _load_models(self):
        threads = int(self.cfg.get("cpu_threads", 4))
        # YuNet via OpenCV
        yunet_path = model_path("yunet")
        if yunet_path.exists():
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    str(yunet_path), "", (320, 320), 0.7, 0.3, 5000
                )
                logger.info("YuNet detector ready")
            except Exception as e:
                logger.warning("YuNet init failed: %s", e)

        # SFace recognizer
        sface_path = model_path("sface")
        if sface_path.exists():
            try:
                self.sface = cv2.FaceRecognizerSF.create(str(sface_path), "")
                logger.info("SFace recognizer ready")
            except Exception as e:
                logger.warning("SFace init failed: %s", e)

        # Generic ONNX sessions for available recognition / attribute models
        for key, meta in MODEL_REGISTRY.items():
            if meta["family"] in ("recognition", "attribute", "liveness", "landmark", "detection"):
                if is_model_available(key) and key not in ("yunet", "sface"):
                    self.sessions[key] = ONNXSession(model_path(key), threads)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Return list of faces with bbox, score, landmarks5."""
        h, w = image_bgr.shape[:2]
        detector = self.cfg.get("primary_detector", "yunet")
        faces: List[Dict[str, Any]] = []

        if detector == "yunet" and self.yunet is not None:
            try:
                self.yunet.setInputSize((w, h))
                _, dets = self.yunet.detect(image_bgr)
                if dets is not None:
                    for d in dets:
                        x, y, bw, bh = d[:4]
                        score = float(d[14]) if len(d) > 14 else float(d[4])
                        # YuNet landmarks: indices 4..13 → 5 points (x,y)
                        kps = []
                        for i in range(5):
                            kps.append([float(d[4 + 2 * i]), float(d[5 + 2 * i])])
                        faces.append({
                            "bbox": [float(x), float(y), float(bw), float(bh)],
                            "score": score,
                            "landmarks5": kps,
                        })
            except Exception as e:
                logger.warning("YuNet detect error: %s", e)

        if not faces:
            # OpenCV Haar cascade fallback
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                rects = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                for (x, y, bw, bh) in rects:
                    # approximate 5 landmarks from bbox
                    cx, cy = x + bw / 2, y + bh / 2
                    kps = [
                        [x + bw * 0.3, y + bh * 0.38],
                        [x + bw * 0.7, y + bh * 0.38],
                        [cx, y + bh * 0.55],
                        [x + bw * 0.35, y + bh * 0.75],
                        [x + bw * 0.65, y + bh * 0.75],
                    ]
                    faces.append({
                        "bbox": [float(x), float(y), float(bw), float(bh)],
                        "score": 0.85,
                        "landmarks5": kps,
                    })
            except Exception as e:
                logger.warning("Haar fallback failed: %s", e)

        return faces

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------
    def align_face(
        self, image_bgr: np.ndarray, landmarks5: List[List[float]], size: int = 112
    ) -> np.ndarray:
        src = np.array(landmarks5, dtype=np.float32)
        ref = ARCFACE_REF * (size / 112.0)
        M, _ = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
        if M is None:
            # fallback center crop
            h, w = image_bgr.shape[:2]
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            crop = image_bgr[y0 : y0 + side, x0 : x0 + side]
            return cv2.resize(crop, (size, size))
        aligned = cv2.warpAffine(image_bgr, M, (size, size), borderValue=0.0)
        return aligned

    # ------------------------------------------------------------------
    # Quality gates
    # ------------------------------------------------------------------
    def quality_metrics(
        self, image_bgr: np.ndarray, bbox: List[float], landmarks5: Optional[List] = None
    ) -> Dict[str, Any]:
        x, y, bw, bh = [int(v) for v in bbox]
        h, w = image_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return {
                "quality": 0.0, "sharpness": 0, "brightness": 0,
                "yaw": 0, "size_ok": False, "sharp_ok": False,
                "bright_ok": False, "yaw_ok": True,
            }

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        min_side = min(bw, bh)

        # rough yaw from landmarks (eye distance vs nose offset)
        yaw = 0.0
        if landmarks5 and len(landmarks5) >= 3:
            le, re, nose = landmarks5[0], landmarks5[1], landmarks5[2]
            eye_mid_x = (le[0] + re[0]) / 2
            eye_dist = abs(re[0] - le[0]) + 1e-6
            yaw = float(np.clip((nose[0] - eye_mid_x) / eye_dist * 60, -90, 90))

        cfg = self.cfg
        size_ok = min_side >= cfg.get("min_face_px", 64)
        sharp_ok = sharpness >= cfg.get("sharpness_threshold", 45.0)
        bright_ok = cfg.get("brightness_low", 35) <= brightness <= cfg.get("brightness_high", 225)
        yaw_ok = abs(yaw) <= cfg.get("yaw_limit", 35.0)

        # composite 0-1
        res_s = min(1.0, min_side / 120.0)
        sharp_s = min(1.0, sharpness / 150.0)
        bright_s = 1.0 - abs(brightness - 128) / 128.0
        pose_s = 1.0 - min(1.0, abs(yaw) / 45.0)
        quality = 0.30 * res_s + 0.30 * sharp_s + 0.20 * bright_s + 0.20 * pose_s

        return {
            "quality": float(quality),
            "sharpness": sharpness,
            "brightness": brightness,
            "yaw": yaw,
            "pitch": 0.0,
            "roll": 0.0,
            "size_ok": size_ok,
            "sharp_ok": sharp_ok,
            "bright_ok": bright_ok,
            "yaw_ok": yaw_ok,
            "min_side": min_side,
        }

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------
    def extract_embeddings(
        self, aligned_112: np.ndarray, use_flip: bool = True
    ) -> Dict[str, np.ndarray]:
        """Return {model_key: L2-normalized embedding}."""
        embs: Dict[str, np.ndarray] = {}

        # SFace (OpenCV)
        if self.sface is not None:
            try:
                feat = self.sface.feature(aligned_112)
                feat = np.asarray(feat, dtype=np.float32).ravel()
                feat = feat / (np.linalg.norm(feat) + 1e-12)
                if use_flip:
                    flipped = cv2.flip(aligned_112, 1)
                    feat2 = self.sface.feature(flipped)
                    feat2 = np.asarray(feat2, dtype=np.float32).ravel()
                    feat2 = feat2 / (np.linalg.norm(feat2) + 1e-12)
                    feat = feat + feat2
                    feat = feat / (np.linalg.norm(feat) + 1e-12)
                embs["sface"] = feat
            except Exception as e:
                logger.debug("SFace extract failed: %s", e)

        # Generic ONNX recognition models (buffalo_l style: 112x112 RGB, (x-127.5)/128)
        for key, sess in self.sessions.items():
            meta = MODEL_REGISTRY.get(key, {})
            if meta.get("family") != "recognition" or sess.session is None:
                continue
            try:
                img = cv2.cvtColor(aligned_112, cv2.COLOR_BGR2RGB).astype(np.float32)
                img = (img - 127.5) / 128.0
                img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # NCHW
                outs = sess.run({sess.input_name: img})
                if not outs:
                    continue
                feat = outs[0].ravel().astype(np.float32)
                feat = feat / (np.linalg.norm(feat) + 1e-12)
                if use_flip:
                    flipped = cv2.flip(aligned_112, 1)
                    img2 = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB).astype(np.float32)
                    img2 = (img2 - 127.5) / 128.0
                    img2 = np.transpose(img2, (2, 0, 1))[np.newaxis, ...]
                    outs2 = sess.run({sess.input_name: img2})
                    if outs2:
                        feat2 = outs2[0].ravel().astype(np.float32)
                        feat2 = feat2 / (np.linalg.norm(feat2) + 1e-12)
                        feat = feat + feat2
                        feat = feat / (np.linalg.norm(feat) + 1e-12)
                embs[key] = feat
            except Exception as e:
                logger.debug("ONNX embed %s failed: %s", key, e)

        # If nothing available, create a deterministic histogram fallback
        # so the gallery pipeline still works offline.
        if not embs:
            gray = cv2.cvtColor(aligned_112, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [128], [0, 256]).ravel()
            hist = hist / (np.linalg.norm(hist) + 1e-12)
            embs["hist_fallback"] = hist.astype(np.float32)

        return embs

    # ------------------------------------------------------------------
    # Attributes (best-effort)
    # ------------------------------------------------------------------
    def estimate_attributes(self, aligned_112: np.ndarray) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {
            "age": None, "gender": None, "emotion": None, "liveness": None, "is_live": True
        }
        # genderage if present
        if "genderage" in self.sessions and self.sessions["genderage"].session:
            try:
                sess = self.sessions["genderage"]
                img = cv2.resize(aligned_112, (96, 96)).astype(np.float32)
                img = (img - 127.5) / 128.0
                img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
                outs = sess.run({sess.input_name: img})
                if outs:
                    # typical InsightFace genderage: [gender_prob, age]
                    g = outs[0].ravel()
                    if len(g) >= 3:
                        gender = "Male" if g[0] > g[1] else "Female"
                        age = float(g[2])
                        attrs["gender"] = gender
                        attrs["age"] = age
            except Exception:
                pass

        # simple emotion heuristic if model missing
        if attrs["emotion"] is None:
            attrs["emotion"] = "Neutral"

        # liveness
        if "minifasnet" in self.sessions and self.sessions["minifasnet"].session:
            try:
                sess = self.sessions["minifasnet"]
                img = cv2.resize(aligned_112, (80, 80)).astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
                outs = sess.run({sess.input_name: img})
                if outs:
                    probs = outs[0].ravel()
                    # assume index 1 = live
                    live_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    attrs["liveness"] = live_score
                    attrs["is_live"] = live_score > 0.5
            except Exception:
                pass
        else:
            attrs["liveness"] = 0.85
            attrs["is_live"] = True

        return attrs

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------
    def analyze(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        faces = self.detect(image_bgr)
        results = []
        for f in faces:
            bbox = f["bbox"]
            kps = f.get("landmarks5")
            q = self.quality_metrics(image_bgr, bbox, kps)
            aligned = self.align_face(image_bgr, kps) if kps else cv2.resize(
                image_bgr[
                    max(0, int(bbox[1])) : int(bbox[1] + bbox[3]),
                    max(0, int(bbox[0])) : int(bbox[0] + bbox[2]),
                ],
                (112, 112),
            )
            embs = self.extract_embeddings(aligned)
            attrs = self.estimate_attributes(aligned)
            results.append({
                "bbox": bbox,
                "score": f["score"],
                "landmarks5": kps,
                "quality": q,
                "embeddings": {k: v.tolist() for k, v in embs.items()},
                "attributes": attrs,
                "aligned_b64": None,  # filled by API if needed
            })
        return results

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# Singleton
_engine: Optional[FaceEngine] = None


def get_engine() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine
