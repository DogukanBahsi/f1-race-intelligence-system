"""
src/utils.py
------------
F1 Race Intelligence System - Genel Yardımcı Fonksiyonlar
Diğer modüller tarafından ortak kullanılan araçlar.
"""

import re
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
from functools import wraps

from src.logger import get_logger

logger = get_logger("Utils")


# ─────────────────────────────────────────────────────────────
# YOL VE İSİM ARAÇLARI
# ─────────────────────────────────────────────────────────────

def race_name_to_slug(race_name: str) -> str:
    """
    Yarış adını dosya adı formatına çevir.
    Örnek: "Great Britain" → "great_britain"
    """
    return re.sub(r"[^a-z0-9]+", "_", race_name.lower()).strip("_")


def slug_to_race_name(slug: str) -> str:
    """
    Dosya adı formatını okunabilir yarış adına çevir.
    Örnek: "great_britain" → "Great Britain"
    """
    return slug.replace("_", " ").title()


def get_raw_csv_path(race_name: str, base_dir: Path, data_type: str = "laps") -> Path:
    """Ham CSV dosyasının yolunu döndür."""
    slug = race_name_to_slug(race_name)
    return base_dir / f"{slug}_{data_type}_raw.csv"


def get_processed_csv_path(race_name: str, base_dir: Path) -> Path:
    """İşlenmiş CSV dosyasının yolunu döndür."""
    slug = race_name_to_slug(race_name)
    return base_dir / f"{slug}_processed.csv"


# ─────────────────────────────────────────────────────────────
# ZAMAN ARAÇLARI
# ─────────────────────────────────────────────────────────────

def timedelta_to_seconds(td) -> Optional[float]:
    """
    pandas Timedelta veya datetime.timedelta nesnesini saniyeye çevir.
    None veya NaN ise None döndür.
    """
    if td is None or (isinstance(td, float) and np.isnan(td)):
        return None
    try:
        return pd.Timedelta(td).total_seconds()
    except Exception:
        return None


def seconds_to_mmss(seconds: float) -> str:
    """
    Saniyeyi MM:SS.mmm formatına çevir.
    Örnek: 90.456 → "1:30.456"
    """
    if pd.isna(seconds) or seconds is None:
        return "N/A"
    minutes = int(seconds // 60)
    secs    = seconds % 60
    return f"{minutes}:{secs:06.3f}"


# ─────────────────────────────────────────────────────────────
# DEKORATÖRler
# ─────────────────────────────────────────────────────────────

def timer(func):
    """Fonksiyonun ne kadar sürdüğünü loglayan dekoratör."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} tamamlandı ({elapsed:.2f}s)")
        return result
    return wrapper


def safe_execute(default=None, log_errors: bool = True):
    """
    Hata durumunda çökmeden devam etmek için dekoratör.
    Hata oluşursa default değeri döndürür.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"{func.__name__} hatası: {e}")
                return default
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────
# VERİ ARAÇLARI
# ─────────────────────────────────────────────────────────────

def safe_mean(series: pd.Series) -> float:
    """NaN değerleri görmezden gelerek güvenli ortalama hesapla."""
    clean = series.dropna()
    return float(clean.mean()) if len(clean) > 0 else float("nan")


def safe_std(series: pd.Series) -> float:
    """NaN değerleri görmezden gelerek güvenli standart sapma hesapla."""
    clean = series.dropna()
    return float(clean.std()) if len(clean) > 1 else 0.0


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    IQR yöntemi ile aykırı değerleri filtrele.
    Q1 - 1.5*IQR altı ve Q3 + 1.5*IQR üstü değerler çıkarılır.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    filtered = df[(df[column] >= lower) & (df[column] <= upper)]
    removed = len(df) - len(filtered)
    if removed > 0:
        logger.debug(f"IQR filtresi: {column} sütununda {removed} aykırı değer çıkarıldı.")
    return filtered


def normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalizasyonu (0-1 arası ölçekle)."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def generate_sample_lap_data(race_name: str, n_laps: int = None) -> pd.DataFrame:
    """
    FastF1 erişilemediğinde kullanılacak gerçekçi simulated veri üreticisi.

    UYARI: Bu veriler tamamen simüle edilmiştir, gerçek FastF1 verisi DEĞİLDİR.
    Her yarış için farklı pist karakteristikleri, 20 gerçek 2025 pilotu ve
    gerçekçi pit stop davranışı simüle edilir.
    """
    logger.warning(f"[SIMULATED] {race_name} için simüle veri üretiliyor. "
                   f"Gerçek veri için internet bağlantısı ve FastF1 gereklidir.")

    # 2025 F1 sezonu pilotları ve takımları
    DRIVERS_2025 = [
        ("VER", "Red Bull"), ("LAW", "Red Bull"),
        ("NOR", "McLaren"),  ("PIA", "McLaren"),
        ("LEC", "Ferrari"),  ("HAM", "Ferrari"),
        ("RUS", "Mercedes"), ("ANT", "Mercedes"),
        ("ALO", "Aston Martin"), ("STR", "Aston Martin"),
        ("GAS", "Alpine"),   ("DOO", "Alpine"),
        ("SAI", "Williams"), ("ALB", "Williams"),
        ("HUL", "Kick Sauber"), ("BOR", "Kick Sauber"),
        ("TSU", "RB"),       ("HAD", "RB"),
        ("OCO", "Haas"),     ("BEA", "Haas"),
    ]

    # Pist bazlı gerçekçi konfigürasyonlar
    TRACK_CONFIGS = {
        "Bahrain":       {"base": 94.5,  "laps": 57, "air": 32, "track": 48, "sc_laps": [12, 35]},
        "Monaco":        {"base": 74.8,  "laps": 78, "air": 23, "track": 38, "sc_laps": [25]},
        "Italy":         {"base": 82.0,  "laps": 53, "air": 26, "track": 42, "sc_laps": []},
        "Belgium":       {"base": 105.0, "laps": 44, "air": 18, "track": 28, "sc_laps": [8]},
        "Great Britain": {"base": 90.3,  "laps": 52, "air": 21, "track": 35, "sc_laps": [18, 40]},
        "Australia":     {"base": 79.8,  "laps": 58, "air": 22, "track": 36, "sc_laps": [15]},
        "Japan":         {"base": 91.5,  "laps": 53, "air": 19, "track": 30, "sc_laps": []},
        "China":         {"base": 95.2,  "laps": 56, "air": 24, "track": 40, "sc_laps": [20]},
        "Saudi Arabia":  {"base": 88.2,  "laps": 50, "air": 30, "track": 44, "sc_laps": [10]},
        "Miami":         {"base": 89.5,  "laps": 57, "air": 28, "track": 42, "sc_laps": []},
        "Spain":         {"base": 79.5,  "laps": 66, "air": 25, "track": 40, "sc_laps": []},
        "Canada":        {"base": 76.2,  "laps": 70, "air": 22, "track": 35, "sc_laps": [22, 45]},
        "Austria":       {"base": 67.5,  "laps": 71, "air": 24, "track": 38, "sc_laps": []},
        "Hungary":       {"base": 78.3,  "laps": 70, "air": 28, "track": 45, "sc_laps": []},
        "Netherlands":   {"base": 72.9,  "laps": 72, "air": 20, "track": 32, "sc_laps": [30]},
        "Azerbaijan":    {"base": 103.5, "laps": 51, "air": 26, "track": 38, "sc_laps": [18]},
        "Singapore":     {"base": 100.8, "laps": 62, "air": 31, "track": 42, "sc_laps": [20, 40]},
        "United States": {"base": 99.5,  "laps": 56, "air": 29, "track": 46, "sc_laps": []},
        "Mexico":        {"base": 80.2,  "laps": 71, "air": 25, "track": 38, "sc_laps": []},
        "Brazil":        {"base": 75.5,  "laps": 71, "air": 28, "track": 45, "sc_laps": [25]},
        "Las Vegas":     {"base": 99.2,  "laps": 50, "air": 18, "track": 22, "sc_laps": []},
        "Qatar":         {"base": 81.3,  "laps": 57, "air": 33, "track": 50, "sc_laps": []},
        "Abu Dhabi":     {"base": 88.7,  "laps": 58, "air": 30, "track": 45, "sc_laps": []},
        "Emilia-Romagna":{"base": 79.0,  "laps": 63, "air": 22, "track": 36, "sc_laps": []},
    }

    cfg = TRACK_CONFIGS.get(race_name, {"base": 90.0, "laps": 57, "air": 25, "track": 38, "sc_laps": []})
    total_laps = cfg["laps"]
    base_time  = cfg["base"]
    sc_laps    = set(cfg["sc_laps"])

    # Her yarış için farklı seed (race_name hash'inden)
    seed = sum(ord(c) for c in race_name) % 10000
    rng  = np.random.default_rng(seed)

    # Pit stop stratejileri (her pilot için)
    compound_map = {"SOFT": -0.8, "MEDIUM": 0.0, "HARD": 0.6}

    # Pilot bazlı performans farkı (gerçekçi hiyerarşi)
    driver_offsets = {}
    top_drivers    = ["VER", "NOR", "LEC", "PIA", "HAM"]
    mid_drivers    = ["RUS", "SAI", "ALO", "GAS", "TSU"]
    for drv, team in DRIVERS_2025:
        if drv in top_drivers:
            driver_offsets[drv] = rng.uniform(-0.5, 0.2)
        elif drv in mid_drivers:
            driver_offsets[drv] = rng.uniform(0.0, 0.8)
        else:
            driver_offsets[drv] = rng.uniform(0.5, 1.5)

    rows = []
    for driver, team in DRIVERS_2025:
        drv_offset = driver_offsets[driver]
        drv_consistency = rng.uniform(0.15, 0.55)  # Her pilotun kendi tutarsızlığı

        # Pit stratejisi: 2 stop
        pit_lap_1 = int(total_laps * rng.uniform(0.28, 0.38))
        pit_lap_2 = int(total_laps * rng.uniform(0.62, 0.72))
        pit_laps_set = {pit_lap_1, pit_lap_2}

        # Stint konfigürasyonu
        strategy = [
            ("SOFT",   1,         pit_lap_1 - 1),
            ("MEDIUM", pit_lap_1, pit_lap_2 - 1),
            ("HARD",   pit_lap_2, total_laps),
        ]

        stint_num  = 1
        tyre_life  = 1

        for lap_num in range(1, total_laps + 1):
            # Mevcut stint ve compound belirle
            compound = "MEDIUM"
            for cmp, start, end in strategy:
                if start <= lap_num <= end:
                    compound = cmp
                    tyre_life = lap_num - start + 1
                    if lap_num == start:
                        stint_num += 1 if lap_num > 1 else 0
                    break

            is_pit_lap = lap_num in pit_laps_set
            is_sc_lap  = lap_num in sc_laps

            # Tur zamanı hesapla
            cmp_offset  = compound_map.get(compound, 0)
            degrad      = tyre_life * rng.uniform(0.03, 0.08)
            sc_penalty  = rng.uniform(15, 25) if is_sc_lap else 0
            pit_penalty = rng.uniform(18, 24) if is_pit_lap else 0
            noise       = rng.normal(0, drv_consistency)
            lap_time    = (base_time + drv_offset + cmp_offset + degrad
                           + sc_penalty + pit_penalty + noise)

            # Sektör süreleri (pist karakterine göre ağırlık)
            s1 = lap_time * rng.uniform(0.26, 0.30) + rng.normal(0, 0.08)
            s2 = lap_time * rng.uniform(0.36, 0.40) + rng.normal(0, 0.08)
            s3 = lap_time - s1 - s2

            # Hava verisi (lap boyunca hafif değişim)
            air_temp   = cfg["air"]   + rng.normal(0, 1.5)
            track_temp = cfg["track"] + rng.normal(0, 2.0)
            humidity   = rng.uniform(40, 75)

            rows.append({
                "Driver":        driver,
                "Team":          team,
                "LapNumber":     lap_num,
                "LapTime":       round(lap_time, 4),
                "Compound":      compound,
                "TyreLife":      tyre_life,
                "Stint":         max(1, stint_num),
                "Sector1Time":   round(s1, 4),
                "Sector2Time":   round(s2, 4),
                "Sector3Time":   round(s3, 4),
                "AirTemp":       round(air_temp, 2),
                "TrackTemp":     round(track_temp, 2),
                "Humidity":      round(humidity, 1),
                "IsPersonalBest": False,
                "PitInTime":     round(lap_time - 1.5, 3) if is_pit_lap else None,
                "PitOutTime":    round(lap_time - 0.5, 3) if is_pit_lap else None,
                "TrackStatus":   "4" if is_sc_lap else ("2" if is_pit_lap else "1"),
                "RaceName":      race_name,
                "IsSimulated":   True,
            })

    df = pd.DataFrame(rows)
    logger.warning(f"[SIMULATED] {race_name}: {len(df)} satır, {df['Driver'].nunique()} pilot üretildi. "
                   f"Bu veriler gerçek değildir.")
    return df


# ─────────────────────────────────────────────────────────────
# YORUM ÜRETİCİ
# ─────────────────────────────────────────────────────────────

def generate_auto_comment(metric_name: str, value: float, race_name: str,
                          context: Optional[str] = None) -> str:
    """
    Analiz sonuçlarına göre otomatik Türkçe yorum üretir.
    """
    if "degradation" in metric_name.lower():
        if value > 0.2:
            return (f"{race_name} yarışında lastik bozulma oranı yüksek ({value:.3f} s/tur). "
                    "Pit stop stratejisi kritik önem taşımaktadır.")
        else:
            return (f"{race_name} yarışında lastikler nispeten dayanıklı görünmektedir ({value:.3f} s/tur).")

    if "consistency" in metric_name.lower():
        if value < 1.0:
            return f"Pilotun tur zamanı istikrarı yüksektir (std: {value:.3f}s)."
        elif value < 2.0:
            return f"Pilotun tur zamanı orta düzey istikrar göstermektedir (std: {value:.3f}s)."
        else:
            return f"Pilotun tur zamanı dalgalıdır (std: {value:.3f}s). Strateji değişkenlikleri olabilir."

    return f"{metric_name}: {value:.3f}"


if __name__ == "__main__":
    # Basit testler
    print(race_name_to_slug("Great Britain"))   # great_britain
    print(seconds_to_mmss(92.456))              # 1:32.456
    print(generate_auto_comment("degradation", 0.25, "Monza"))
