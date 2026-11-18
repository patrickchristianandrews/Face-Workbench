"""
FaceWorkbench Pro — Threshold calibration, logistic stacker, TAR@FAR, hard-neg mining
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from database import get_connection
from registry import load_config, save_config

logger = logging.getLogger("FaceWorkbench.calibration")


def _load_all_embeddings() -> Dict[str, Dict[str, List[Tuple[str, np.ndarray]]]]:
    """
    Returns {model_key: {person_id: [(face_id, emb), ...]}}
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT face_id, person_id, model_key, embedding, dim FROM model_embeddings"
        ).fetchall()
    finally:
        conn.close()

    data: Dict[str, Dict[str, List[Tuple[str, np.ndarray]]]] = {}
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        if emb.size != r["dim"]:
            continue
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        data.setdefault(r["model_key"], {}).setdefault(r["person_id"], []).append(
            (r["face_id"], emb)
        )
    return data


def generate_pairs(
    model_key: str, max_genuine: int = 2000, max_impostor: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (genuine_scores, impostor_scores) cosine similarities."""
    data = _load_all_embeddings().get(model_key, {})
    persons = list(data.keys())
    genuine, impostor = [], []

    # genuine
    for pid, faces in data.items():
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                genuine.append(float(np.dot(faces[i][1], faces[j][1])))
                if len(genuine) >= max_genuine:
                    break
            if len(genuine) >= max_genuine:
                break
        if len(genuine) >= max_genuine:
            break

    # impostor
    rng = np.random.default_rng(42)
    for _ in range(max_impostor):
        if len(persons) < 2:
            break
        p1, p2 = rng.choice(persons, 2, replace=False)
        f1 = data[p1][rng.integers(0, len(data[p1]))][1]
        f2 = data[p2][rng.integers(0, len(data[p2]))][1]
        impostor.append(float(np.dot(f1, f2)))

    return np.array(genuine, dtype=np.float32), np.array(impostor, dtype=np.float32)


def calibrate_threshold(
    model_key: str = "sface", target_far: float = 0.001
) -> Dict[str, Any]:
    genuine, impostor = generate_pairs(model_key)
    if len(genuine) < 5 or len(impostor) < 20:
        return {
            "ok": False,
            "message": "Not enough pairs. Enroll more identities with multiple photos.",
            "threshold": load_config().get("match_threshold", 0.45),
        }

    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    scores = np.concatenate([genuine, impostor])
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # find threshold closest to target FAR
    idx = np.argmin(np.abs(fpr - target_far))
    thr = float(thresholds[idx]) if idx < len(thresholds) else 0.45
    tar = float(tpr[idx])

    # TAR at common FARs
    tar_at_far = {}
    for far in [0.1, 0.01, 0.001, 0.0001]:
        i = np.argmin(np.abs(fpr - far))
        tar_at_far[str(far)] = float(tpr[i])

    cfg = load_config()
    cfg["match_threshold"] = thr
    save_config(cfg)

    # persist history
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO calibration_history (id, timestamp, threshold, tar_at_far, notes) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), time.time(), thr, json.dumps(tar_at_far), f"target_far={target_far}"),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "ok": True,
        "threshold": thr,
        "tar_at_target_far": tar,
        "target_far": target_far,
        "tar_at_far": tar_at_far,
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "roc": {
            "fpr": fpr.tolist()[:: max(1, len(fpr) // 100)],
            "tpr": tpr.tolist()[:: max(1, len(tpr) // 100)],
        },
    }


def train_stacker() -> Dict[str, Any]:
    """Train logistic regression on per-model similarity vectors."""
    all_data = _load_all_embeddings()
    model_keys = sorted(all_data.keys())
    if len(model_keys) < 1:
        return {"ok": False, "message": "No embeddings available"}

    # collect multi-model scores for genuine / impostor pairs
    X, y = [], []
    persons = set()
    for mk in model_keys:
        persons.update(all_data[mk].keys())
    persons = list(persons)
    if len(persons) < 2:
        return {"ok": False, "message": "Need at least 2 identities"}

    rng = np.random.default_rng(7)

    # genuine
    for pid in persons:
        face_ids = None
        for mk in model_keys:
            if pid in all_data[mk] and len(all_data[mk][pid]) >= 2:
                face_ids = [f[0] for f in all_data[mk][pid]]
                break
        if not face_ids or len(face_ids) < 2:
            continue
        for i in range(min(5, len(face_ids))):
            for j in range(i + 1, min(5, len(face_ids))):
                vec = []
                for mk in model_keys:
                    embs = {fid: e for fid, e in all_data.get(mk, {}).get(pid, [])}
                    if face_ids[i] in embs and face_ids[j] in embs:
                        vec.append(float(np.dot(embs[face_ids[i]], embs[face_ids[j]])))
                    else:
                        vec.append(0.0)
                X.append(vec)
                y.append(1)

    # impostor
    for _ in range(min(2000, len(X) * 3)):
        p1, p2 = rng.choice(persons, 2, replace=False)
        vec = []
        for mk in model_keys:
            e1 = all_data.get(mk, {}).get(p1, [])
            e2 = all_data.get(mk, {}).get(p2, [])
            if e1 and e2:
                vec.append(float(np.dot(e1[rng.integers(0, len(e1))][1], e2[rng.integers(0, len(e2))][1])))
            else:
                vec.append(0.0)
        X.append(vec)
        y.append(0)

    if len(X) < 20:
        return {"ok": False, "message": "Not enough pairs for stacker"}

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X, y)

    cfg = load_config()
    cfg["stacker_coefs"] = {
        "model_keys": model_keys,
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }
    save_config(cfg)

    return {
        "ok": True,
        "model_keys": model_keys,
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "n_samples": len(X),
        "accuracy": float(clf.score(X, y)),
    }


def hard_negative_mining(threshold: float = 0.4, margin: float = 0.12, top_n: int = 30) -> List[Dict]:
    """Find impostor pairs with high similarity (near threshold)."""
    all_data = _load_all_embeddings()
    # use first available model
    if not all_data:
        return []
    model_key = next(iter(all_data))
    data = all_data[model_key]
    persons = list(data.keys())
    pairs = []
    for i, p1 in enumerate(persons):
        for p2 in persons[i + 1 :]:
            for f1, e1 in data[p1]:
                for f2, e2 in data[p2]:
                    sim = float(np.dot(e1, e2))
                    if sim >= threshold - margin:
                        pairs.append({
                            "person_a": p1,
                            "person_b": p2,
                            "face_a": f1,
                            "face_b": f2,
                            "similarity": sim,
                            "model": model_key,
                        })
    pairs.sort(key=lambda x: -x["similarity"])
    return pairs[:top_n]


def stacker_predict(score_vec: List[float]) -> float:
    cfg = load_config()
    sc = cfg.get("stacker_coefs")
    if not sc:
        return float(np.mean(score_vec)) if score_vec else 0.0
    coef = np.array(sc["coef"], dtype=np.float32)
    # pad / trim
    v = np.zeros(len(coef), dtype=np.float32)
    n = min(len(score_vec), len(coef))
    v[:n] = score_vec[:n]
    logit = float(np.dot(coef, v) + sc["intercept"])
    return 1.0 / (1.0 + np.exp(-logit))
