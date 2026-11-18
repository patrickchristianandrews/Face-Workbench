"""
FaceWorkbench Pro — SQLite + FAISS persistence layer
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from registry import DATA_DIR, load_config, save_config

logger = logging.getLogger("FaceWorkbench.database")

DB_PATH = DATA_DIR / "gallery.db"
INDEX_DIR = DATA_DIR / "indexes"
CROPS_DIR = DATA_DIR / "crops"
EXPORTS_DIR = DATA_DIR / "exports"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS people (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at REAL,
                updated_at REAL,
                sample_count INTEGER DEFAULT 0,
                avg_quality REAL DEFAULT 0.0,
                notes TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                crop_path TEXT,
                quality REAL DEFAULT 0.0,
                sharpness REAL DEFAULT 0.0,
                brightness REAL DEFAULT 0.0,
                yaw REAL DEFAULT 0.0,
                pitch REAL DEFAULT 0.0,
                roll REAL DEFAULT 0.0,
                age REAL,
                gender TEXT,
                emotion TEXT,
                liveness REAL,
                bbox_json TEXT,
                landmarks5_json TEXT,
                created_at REAL,
                FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS model_embeddings (
                id TEXT PRIMARY KEY,
                face_id TEXT NOT NULL,
                person_id TEXT NOT NULL,
                model_key TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                created_at REAL,
                FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE,
                FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS calibration_history (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                threshold REAL,
                tar_at_far TEXT,
                notes TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
            CREATE INDEX IF NOT EXISTS idx_emb_person ON model_embeddings(person_id);
            CREATE INDEX IF NOT EXISTS idx_emb_model ON model_embeddings(model_key);
        """)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# FAISS helpers (IndexFlatIP — cosine via L2-normalized vectors)
# ---------------------------------------------------------------------------
_faiss = None
_indexes: Dict[str, Any] = {}
_index_ids: Dict[str, List[str]] = {}  # model_key -> list of face_ids


def _import_faiss():
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except Exception:
            logger.warning("faiss not available — search will use brute-force numpy")
            _faiss = False
    return _faiss


def _index_path(model_key: str) -> Path:
    return INDEX_DIR / f"{model_key}.faiss"


def _ids_path(model_key: str) -> Path:
    return INDEX_DIR / f"{model_key}.ids.json"


def load_or_create_index(model_key: str, dim: int):
    faiss = _import_faiss()
    if model_key in _indexes:
        return _indexes[model_key]

    path = _index_path(model_key)
    ids_path = _ids_path(model_key)

    if faiss and path.exists() and ids_path.exists():
        try:
            index = faiss.read_index(str(path))
            with open(ids_path, "r", encoding="utf-8") as f:
                ids = json.load(f)
            _indexes[model_key] = index
            _index_ids[model_key] = ids
            return index
        except Exception as e:
            logger.warning("Failed to load FAISS index %s: %s", model_key, e)

    if faiss:
        index = faiss.IndexFlatIP(dim)
        _indexes[model_key] = index
        _index_ids[model_key] = []
        return index

    # numpy fallback store
    _indexes[model_key] = {"dim": dim, "vecs": [], "ids": []}
    _index_ids[model_key] = []
    return _indexes[model_key]


def add_embedding_to_index(model_key: str, face_id: str, emb: np.ndarray) -> None:
    emb = emb.astype(np.float32).reshape(1, -1)
    # L2 normalize
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norm, 1e-12)

    index = load_or_create_index(model_key, emb.shape[1])
    faiss = _import_faiss()

    if faiss and not isinstance(index, dict):
        index.add(emb)
        _index_ids[model_key].append(face_id)
        faiss.write_index(index, str(_index_path(model_key)))
        with open(_ids_path(model_key), "w", encoding="utf-8") as f:
            json.dump(_index_ids[model_key], f)
    else:
        # numpy store
        store = index
        store["vecs"].append(emb[0])
        store["ids"].append(face_id)
        _index_ids[model_key] = store["ids"]
        # persist as npz
        np.savez_compressed(
            str(INDEX_DIR / f"{model_key}.npz"),
            vecs=np.stack(store["vecs"]) if store["vecs"] else np.zeros((0, store["dim"])),
            ids=np.array(store["ids"], dtype=object),
        )


def search_index(model_key: str, query: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
    query = query.astype(np.float32).reshape(1, -1)
    norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / np.maximum(norm, 1e-12)

    if model_key not in _indexes:
        load_or_create_index(model_key, query.shape[1])

    index = _indexes.get(model_key)
    if index is None:
        return []

    faiss = _import_faiss()
    if faiss and not isinstance(index, dict):
        if index.ntotal == 0:
            return []
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query, k)
        ids = _index_ids.get(model_key, [])
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(ids):
                continue
            results.append((ids[idx], float(score)))
        return results

    # numpy fallback
    store = index
    if not store.get("vecs"):
        return []
    mat = np.stack(store["vecs"])
    sims = (mat @ query.T).ravel()
    order = np.argsort(-sims)[:top_k]
    return [(store["ids"][i], float(sims[i])) for i in order]


def rebuild_all_indexes() -> int:
    """Rebuild every per-model FAISS index from SQLite embeddings."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT face_id, person_id, model_key, embedding, dim FROM model_embeddings"
        ).fetchall()
    finally:
        conn.close()

    # clear existing
    for p in INDEX_DIR.glob("*"):
        try:
            p.unlink()
        except Exception:
            pass
    _indexes.clear()
    _index_ids.clear()

    count = 0
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        if emb.size != row["dim"]:
            continue
        add_embedding_to_index(row["model_key"], row["face_id"], emb)
        count += 1
    return count


# ---------------------------------------------------------------------------
# People / Faces CRUD
# ---------------------------------------------------------------------------
def create_person(name: str, notes: str = "") -> str:
    pid = str(uuid.uuid4())
    now = time.time()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO people (id, name, created_at, updated_at, notes) VALUES (?,?,?,?,?)",
            (pid, name, now, now, notes),
        )
        conn.commit()
    finally:
        conn.close()
    return pid


def list_people(search: str = "") -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        if search:
            rows = conn.execute(
                "SELECT * FROM people WHERE name LIKE ? ORDER BY updated_at DESC",
                (f"%{search}%",),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM people ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_person(person_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def rename_person(person_id: str, new_name: str) -> bool:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE people SET name=?, updated_at=? WHERE id=?",
            (new_name, time.time(), person_id),
        )
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def delete_person(person_id: str) -> bool:
    conn = get_connection()
    try:
        # remove crop files
        faces = conn.execute("SELECT crop_path FROM faces WHERE person_id=?", (person_id,)).fetchall()
        for f in faces:
            if f["crop_path"]:
                p = Path(f["crop_path"])
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        conn.execute("DELETE FROM people WHERE id=?", (person_id,))
        conn.commit()
        # rebuild indexes (simple approach)
        rebuild_all_indexes()
        return True
    finally:
        conn.close()


def merge_people(source_id: str, target_id: str) -> bool:
    conn = get_connection()
    try:
        now = time.time()
        conn.execute("UPDATE faces SET person_id=? WHERE person_id=?", (target_id, source_id))
        conn.execute("UPDATE model_embeddings SET person_id=? WHERE person_id=?", (target_id, source_id))
        # recount
        cnt = conn.execute("SELECT COUNT(*) FROM faces WHERE person_id=?", (target_id,)).fetchone()[0]
        conn.execute(
            "UPDATE people SET sample_count=?, updated_at=? WHERE id=?",
            (cnt, now, target_id),
        )
        conn.execute("DELETE FROM people WHERE id=?", (source_id,))
        conn.commit()
        rebuild_all_indexes()
        return True
    finally:
        conn.close()


def add_face(
    person_id: str,
    crop_path: str,
    embedding_map: Dict[str, np.ndarray],
    quality: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    meta = meta or {}
    face_id = str(uuid.uuid4())
    now = time.time()
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO faces
               (id, person_id, crop_path, quality, sharpness, brightness, yaw, pitch, roll,
                age, gender, emotion, liveness, bbox_json, landmarks5_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                face_id, person_id, crop_path, quality,
                meta.get("sharpness", 0), meta.get("brightness", 0),
                meta.get("yaw", 0), meta.get("pitch", 0), meta.get("roll", 0),
                meta.get("age"), meta.get("gender"), meta.get("emotion"),
                meta.get("liveness"),
                json.dumps(meta.get("bbox")), json.dumps(meta.get("landmarks5")),
                now,
            ),
        )
        for model_key, emb in embedding_map.items():
            emb = np.asarray(emb, dtype=np.float32).ravel()
            eid = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO model_embeddings
                   (id, face_id, person_id, model_key, embedding, dim, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (eid, face_id, person_id, model_key, emb.tobytes(), emb.size, now),
            )
            add_embedding_to_index(model_key, face_id, emb)

        # update person stats
        rows = conn.execute(
            "SELECT quality FROM faces WHERE person_id=?", (person_id,)
        ).fetchall()
        qualities = [r["quality"] for r in rows]
        avg_q = float(np.mean(qualities)) if qualities else 0.0
        conn.execute(
            "UPDATE people SET sample_count=?, avg_quality=?, updated_at=? WHERE id=?",
            (len(qualities), avg_q, now, person_id),
        )
        conn.commit()
    finally:
        conn.close()
    return face_id


def get_faces_for_person(person_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM faces WHERE person_id=? ORDER BY created_at DESC",
            (person_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_person_id_for_face(face_id: str) -> Optional[str]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT person_id FROM faces WHERE id=?", (face_id,)).fetchone()
        return row["person_id"] if row else None
    finally:
        conn.close()


def export_gallery_zip() -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = EXPORTS_DIR / f"gallery_backup_{stamp}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        if DB_PATH.exists():
            zf.write(DB_PATH, "gallery.db")
        for p in CROPS_DIR.rglob("*"):
            if p.is_file():
                zf.write(p, f"crops/{p.relative_to(CROPS_DIR)}")
        for p in INDEX_DIR.rglob("*"):
            if p.is_file():
                zf.write(p, f"indexes/{p.relative_to(INDEX_DIR)}")
        from registry import CONFIG_PATH
        if CONFIG_PATH.exists():
            zf.write(CONFIG_PATH, "config.json")
    return out


# Initialize on import
init_db()
