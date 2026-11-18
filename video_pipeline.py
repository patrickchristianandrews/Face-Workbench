"""
FaceWorkbench Pro — Multi-face tracking, body ReID skeleton, 3-mode video redaction
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from engine import get_engine
from registry import DATA_DIR, load_config

logger = logging.getLogger("FaceWorkbench.video")

EXPORTS = DATA_DIR / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)


class Track:
    def __init__(self, track_id: int, bbox: List[float], emb: Optional[np.ndarray] = None):
        self.id = track_id
        self.bbox = bbox  # x,y,w,h
        self.emb = emb
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.smoothed = list(bbox)
        self.person_id: Optional[str] = None
        self.match_score: float = 0.0

    def update(self, bbox: List[float], emb: Optional[np.ndarray] = None):
        self.bbox = bbox
        # exponential smooth
        a = 0.6
        self.smoothed = [
            a * bbox[i] + (1 - a) * self.smoothed[i] for i in range(4)
        ]
        if emb is not None:
            self.emb = emb
        self.hits += 1
        self.time_since_update = 0


def iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def associate(tracks: List[Track], detections: List[Dict], iou_thr: float = 0.3):
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            cost[i, j] = 1.0 - iou(t.bbox, d["bbox"])

    # greedy matching
    matched, unmatched_t, unmatched_d = [], set(range(len(tracks))), set(range(len(detections)))
    pairs = [(i, j, cost[i, j]) for i in range(len(tracks)) for j in range(len(detections))]
    pairs.sort(key=lambda x: x[2])
    for i, j, c in pairs:
        if i in unmatched_t and j in unmatched_d and (1 - c) >= iou_thr:
            matched.append((i, j))
            unmatched_t.discard(i)
            unmatched_d.discard(j)
    return matched, list(unmatched_t), list(unmatched_d)


def blur_region(frame: np.ndarray, bbox: List[float], mode: str = "gaussian") -> None:
    x, y, w, h = [int(v) for v in bbox]
    H, W = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if mode == "mosaic":
        small = cv2.resize(roi, (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)))
        frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    else:
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)


def process_video(
    video_path: str,
    mode: str = "blur_all",  # blur_all | blur_matched | keep_matched
    target_person_ids: Optional[Set[str]] = None,
    progress_cb=None,
    max_frames: int = 0,
) -> Path:
    """
    Process video and write redacted MP4 to exports/.
    mode:
      blur_all      — blur every face track
      blur_matched  — blur only faces matched to target_person_ids
      keep_matched  — keep targets sharp, blur everyone else
    """
    target_person_ids = target_person_ids or set()
    engine = get_engine()
    cfg = load_config()
    threshold = float(cfg.get("match_threshold", 0.45))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames > 0:
        total = min(total, max_frames)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = EXPORTS / f"redacted_{mode}_{stamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    tracks: List[Track] = []
    next_id = 1
    max_age = 30
    frame_idx = 0

    # Pre-load gallery embeddings for matching (first available model)
    from database import get_connection
    gallery: Dict[str, List[np.ndarray]] = {}  # person_id -> list of embs
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT person_id, embedding, dim FROM model_embeddings"
        ).fetchall()
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            if emb.size == r["dim"]:
                emb = emb / (np.linalg.norm(emb) + 1e-12)
                gallery.setdefault(r["person_id"], []).append(emb)
    finally:
        conn.close()

    def match_to_gallery(emb: np.ndarray) -> Tuple[Optional[str], float]:
        best_id, best_score = None, -1.0
        for pid, embs in gallery.items():
            for g in embs:
                s = float(np.dot(emb, g))
                if s > best_score:
                    best_score = s
                    best_id = pid
        if best_score >= threshold:
            return best_id, best_score
        return None, best_score

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        dets = engine.detect(frame)
        # enrich with embeddings
        det_list = []
        for d in dets:
            kps = d.get("landmarks5")
            aligned = engine.align_face(frame, kps) if kps else None
            emb = None
            if aligned is not None:
                embs = engine.extract_embeddings(aligned, use_flip=False)
                if embs:
                    emb = next(iter(embs.values()))
            det_list.append({"bbox": d["bbox"], "emb": emb})

        matched, unmatched_t, unmatched_d = associate(tracks, det_list)

        for ti, di in matched:
            tracks[ti].update(det_list[di]["bbox"], det_list[di]["emb"])
            if det_list[di]["emb"] is not None:
                pid, sc = match_to_gallery(det_list[di]["emb"])
                tracks[ti].person_id = pid
                tracks[ti].match_score = sc

        for ti in unmatched_t:
            tracks[ti].time_since_update += 1
            tracks[ti].age += 1

        for di in unmatched_d:
            t = Track(next_id, det_list[di]["bbox"], det_list[di]["emb"])
            next_id += 1
            if det_list[di]["emb"] is not None:
                pid, sc = match_to_gallery(det_list[di]["emb"])
                t.person_id = pid
                t.match_score = sc
            tracks.append(t)

        tracks = [t for t in tracks if t.time_since_update <= max_age]

        # redaction
        for t in tracks:
            is_target = t.person_id is not None and t.person_id in target_person_ids
            do_blur = False
            if mode == "blur_all":
                do_blur = True
            elif mode == "blur_matched":
                do_blur = is_target
            elif mode == "keep_matched":
                do_blur = not is_target
            if do_blur:
                blur_region(frame, t.smoothed)

        writer.write(frame)
        if progress_cb and total:
            progress_cb(frame_idx / total)

    cap.release()
    writer.release()
    logger.info("Redacted video written: %s", out_path)
    return out_path
