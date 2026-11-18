"""
FaceWorkbench Pro — Local EXIF/XMP + reverse image search launchers
"""
from __future__ import annotations

import logging
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger("FaceWorkbench.osint")

SEARCH_ENGINES = {
    "yandex": {
        "name": "Yandex Images",
        "url": "https://yandex.com/images/search?rpt=imageview&url=",
        "description": "Strong face search, especially Eastern Europe / Asia",
    },
    "search4faces": {
        "name": "Search4Faces",
        "url": "https://search4faces.com/",
        "description": "Specialized face search engine",
    },
    "findclone": {
        "name": "FindClone",
        "url": "https://findclone.ru/",
        "description": "Russian social network face search",
    },
    "lenso": {
        "name": "Lenso.ai",
        "url": "https://lenso.ai/",
        "description": "AI reverse image search",
    },
    "facecheck": {
        "name": "FaceCheck.id",
        "url": "https://facecheck.id/",
        "description": "Face search across web & social",
    },
    "pimeyes": {
        "name": "PimEyes",
        "url": "https://pimeyes.com/en",
        "description": "Commercial face search (subscription)",
    },
}


def extract_exif(image_path: str) -> Dict[str, Any]:
    """Parse EXIF + basic XMP tags locally."""
    result: Dict[str, Any] = {"path": image_path, "exif": {}, "xmp": {}, "gps": None}
    try:
        import exifread
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        for k, v in tags.items():
            if k.startswith("JPEGThumbnail") or k.startswith("Thumbnail"):
                continue
            result["exif"][k] = str(v)
        # GPS
        lat = tags.get("GPS GPSLatitude")
        lon = tags.get("GPS GPSLongitude")
        lat_ref = tags.get("GPS GPSLatitudeRef")
        lon_ref = tags.get("GPS GPSLongitudeRef")
        if lat and lon:
            def _rat(x):
                try:
                    return float(x.num) / float(x.den)
                except Exception:
                    return float(x)
            def _dms(vals, ref):
                d, m, s = [_rat(v) for v in vals.values]
                dec = d + m / 60 + s / 3600
                if str(ref) in ("S", "W"):
                    dec = -dec
                return dec
            result["gps"] = {
                "lat": _dms(lat, lat_ref),
                "lon": _dms(lon, lon_ref),
            }
    except Exception as e:
        logger.debug("exifread failed: %s", e)

    # crude XMP packet scan
    try:
        data = Path(image_path).read_bytes()
        start = data.find(b"<x:xmpmeta")
        end = data.find(b"</x:xmpmeta>")
        if start >= 0 and end > start:
            xmp = data[start : end + 12].decode("utf-8", errors="ignore")
            result["xmp"]["raw_preview"] = xmp[:2000]
            for key in ("CreatorTool", "CreateDate", "ModifyDate", "Artist", "Description"):
                if key in xmp:
                    result["xmp"][key] = True
    except Exception:
        pass
    return result


def launch_search(engine: str, image_url: Optional[str] = None) -> str:
    """Return the URL to open (caller can open via webbrowser or desktop bridge)."""
    meta = SEARCH_ENGINES.get(engine)
    if not meta:
        return ""
    if image_url and engine == "yandex":
        return meta["url"] + quote(image_url, safe="")
    return meta["url"]


def open_urls(urls: List[str], delay: float = 0.8) -> int:
    """Open a list of URLs in the default browser with light rate limiting."""
    import time
    opened = 0
    for u in urls:
        u = u.strip()
        if not u.startswith("http"):
            continue
        try:
            webbrowser.open(u)
            opened += 1
            time.sleep(delay)
        except Exception as e:
            logger.warning("Failed to open %s: %s", u, e)
    return opened


def list_engines() -> List[Dict[str, str]]:
    return [
        {"key": k, "name": v["name"], "description": v["description"], "url": v["url"]}
        for k, v in SEARCH_ENGINES.items()
    ]
