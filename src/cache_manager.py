"""
src/cache_manager.py
--------------------
F1 Race Intelligence System - Analiz Cache Yöneticisi
Analiz + ML sonuçlarını JSON olarak önbellekler.
Her yarış için ayrı cache dosyası tutulur.
"""

import json
import math
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from config import DATA_DIR
from src.logger import get_logger

logger = get_logger("CacheManager")

CACHE_DIR = DATA_DIR / "analysis_cache"
CACHE_VERSION = "v2"   # Versiyon değişince tüm cache geçersiz olur


# ─────────────────────────────────────────────────────────────
# JSON SERİALİZER (numpy tipleri için)
# ─────────────────────────────────────────────────────────────

def _safe_json(obj):
    """Numpy/pandas tiplerini JSON'a güvenli çevir."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_json(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    elif isinstance(obj, np.ndarray):
        return [_safe_json(x) for x in obj.tolist()]
    elif isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    elif hasattr(obj, "to_dict"):
        return _safe_json(obj.to_dict("records"))
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


# ─────────────────────────────────────────────────────────────
# CACHE PATH
# ─────────────────────────────────────────────────────────────

def _cache_path(race_name: str) -> Path:
    slug = race_name.lower().replace(" ", "_")
    return CACHE_DIR / f"{slug}_{CACHE_VERSION}.json"


# ─────────────────────────────────────────────────────────────
# OKUMA / YAZMA
# ─────────────────────────────────────────────────────────────

def load_analysis_cache(race_name: str) -> Optional[Dict[str, Any]]:
    """
    Daha önce hesaplanmış analiz sonuçlarını cache'den yükle.
    Cache bulunamazsa veya bozuksa None döndürür.
    """
    path = _cache_path(race_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Versiyon kontrolü
        if data.get("_cache_version") != CACHE_VERSION:
            logger.debug(f"Cache versiyonu eski, yenileniyor: {race_name}")
            return None
        logger.info(f"Cached analysis loaded: {race_name} "
                    f"(oluşturma: {data.get('_created_at', '?')})")
        return data
    except Exception as e:
        logger.warning(f"Cache okunamadı ({race_name}): {e}")
        return None


def save_analysis_cache(race_name: str,
                         analysis: Dict[str, Any],
                         ml: Dict[str, Any],
                         charts: Dict[str, str]) -> bool:
    """
    Analiz, ML ve grafik sonuçlarını JSON cache'e yaz.
    Grafiklerin sadece anahtar listesini saklar (base64 değil).
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Charts'dan base64 içeriği çıkar (büyük dosya olur), sadece key listesi
        chart_keys = list(charts.keys()) if charts else []

        payload = {
            "_cache_version": CACHE_VERSION,
            "_race_name":     race_name,
            "_created_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis":       _safe_json(analysis),
            "ml":             _safe_json(ml),
            "chart_keys":     chart_keys,
        }

        path = _cache_path(race_name)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        logger.info(f"Analysis cache created: {race_name} → {path.name}")
        return True

    except Exception as e:
        logger.error(f"Cache yazılamadı ({race_name}): {e}")
        return False


def invalidate_cache(race_name: str) -> bool:
    """Belirtilen yarışın cache'ini geçersiz kıl (sil)."""
    path = _cache_path(race_name)
    if path.exists():
        path.unlink()
        logger.info(f"Cache invalidated: {race_name}")
        return True
    return False


def get_cached_races() -> list:
    """Cache'lenmiş yarışların listesini döndür."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return [p.stem.replace(f"_{CACHE_VERSION}", "").replace("_", " ").title()
            for p in CACHE_DIR.glob(f"*_{CACHE_VERSION}.json")]


if __name__ == "__main__":
    print("Cache dir:", CACHE_DIR)
    print("Cached races:", get_cached_races())
