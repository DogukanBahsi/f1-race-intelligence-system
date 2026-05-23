"""
src/database.py
---------------
F1 Race Intelligence System - SQLite Veritabanı Yönetimi
Tüm işlenmiş veriler ve analiz sonuçları burada saklanır.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List

from config import DATABASE_PATH
from src.logger import get_logger

logger = get_logger("Database")


# ─────────────────────────────────────────────────────────────
# TABLO ŞEMALARI
# ─────────────────────────────────────────────────────────────

CREATE_LAPS_TABLE = """
CREATE TABLE IF NOT EXISTS laps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_name       TEXT NOT NULL,
    season          INTEGER NOT NULL,
    driver          TEXT NOT NULL,
    team            TEXT,
    lap_number      INTEGER,
    lap_time        REAL,
    sector1         REAL,
    sector2         REAL,
    sector3         REAL,
    compound        TEXT,
    tyre_life       INTEGER,
    stint           INTEGER,
    air_temp        REAL,
    track_temp      REAL,
    humidity        REAL,
    track_status    TEXT,
    is_pit_lap      INTEGER DEFAULT 0,
    is_sc_lap       INTEGER DEFAULT 0,
    is_outlier      INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_RACE_SUMMARY_TABLE = """
CREATE TABLE IF NOT EXISTS race_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_name       TEXT NOT NULL UNIQUE,
    season          INTEGER NOT NULL,
    total_laps      INTEGER,
    total_drivers   INTEGER,
    fastest_lap     REAL,
    fastest_driver  TEXT,
    most_stable_driver TEXT,
    processed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_DRIVER_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS driver_stats (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    race_name           TEXT NOT NULL,
    driver              TEXT NOT NULL,
    team                TEXT,
    avg_lap_time        REAL,
    best_lap_time       REAL,
    lap_time_std        REAL,
    consistency_score   REAL,
    avg_race_pace       REAL,
    tire_degradation    REAL,
    pit_stop_impact     REAL,
    cluster_label       INTEGER,
    strong_performance  INTEGER,
    UNIQUE(race_name, driver)
);
"""

CREATE_PROCESSED_RACES_TABLE = """
CREATE TABLE IF NOT EXISTS processed_races (
    race_name   TEXT PRIMARY KEY,
    season      INTEGER NOT NULL,
    status      TEXT DEFAULT 'complete',
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ─────────────────────────────────────────────────────────────
# BAĞLANTI YÖNETİCİSİ
# ─────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """SQLite bağlantısı aç ve döndür."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row  # Sözlük gibi erişim için
    return conn


def initialize_database():
    """Tüm tabloları oluştur (yoksa). Uygulama başlangıcında çağrılır."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(CREATE_LAPS_TABLE)
            cursor.execute(CREATE_RACE_SUMMARY_TABLE)
            cursor.execute(CREATE_DRIVER_STATS_TABLE)
            cursor.execute(CREATE_PROCESSED_RACES_TABLE)
            conn.commit()
        logger.info("Veritabanı tabloları hazır.")
    except Exception as e:
        logger.error(f"Veritabanı başlatma hatası: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# KAYIT FONKSİYONLARI
# ─────────────────────────────────────────────────────────────

def save_laps_to_db(df: pd.DataFrame, race_name: str, season: int = 2025):
    """
    İşlenmiş tur verisini laps tablosuna kaydet.
    Yarış daha önce kaydedildiyse önce sil, sonra yaz.
    """
    try:
        with get_connection() as conn:
            # Eski kayıtları temizle
            conn.execute("DELETE FROM laps WHERE race_name = ? AND season = ?",
                         (race_name, season))

            # Sütun eşleştirmesi
            col_map = {
                "Driver":      "driver",
                "Team":        "team",
                "LapNumber":   "lap_number",
                "LapTime":     "lap_time",
                "Sector1Time": "sector1",
                "Sector2Time": "sector2",
                "Sector3Time": "sector3",
                "Compound":    "compound",
                "TyreLife":    "tyre_life",
                "Stint":       "stint",
                "AirTemp":     "air_temp",
                "TrackTemp":   "track_temp",
                "Humidity":    "humidity",
                "TrackStatus": "track_status",
                "IsPitLap":    "is_pit_lap",
                "IsSCLap":     "is_sc_lap",
                "IsOutlier":   "is_outlier",
            }

            # Var olan sütunları filtrele
            available_cols = {k: v for k, v in col_map.items() if k in df.columns}
            insert_df = df.rename(columns=available_cols)[list(available_cols.values())].copy()
            insert_df["race_name"] = race_name
            insert_df["season"]    = season

            insert_df.to_sql("laps", conn, if_exists="append", index=False)
            conn.commit()

        logger.info(f"{race_name}: {len(df)} tur SQLite'a kaydedildi.")
    except Exception as e:
        logger.error(f"{race_name} tur verisi DB'ye kaydedilemedi: {e}")


def save_driver_stats_to_db(stats_df: pd.DataFrame, race_name: str):
    """Pilot istatistiklerini driver_stats tablosuna kaydet."""
    try:
        with get_connection() as conn:
            conn.execute("DELETE FROM driver_stats WHERE race_name = ?", (race_name,))
            stats_df["race_name"] = race_name
            stats_df.to_sql("driver_stats", conn, if_exists="append", index=False)
            conn.commit()
        logger.info(f"{race_name}: Pilot istatistikleri DB'ye kaydedildi.")
    except Exception as e:
        logger.error(f"Pilot istatistikleri kaydedilemedi: {e}")


def save_race_summary_to_db(summary: dict):
    """Yarış özetini race_summary tablosuna kaydet."""
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO race_summary
                (race_name, season, total_laps, total_drivers,
                 fastest_lap, fastest_driver, most_stable_driver)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.get("race_name"),
                summary.get("season", 2025),
                summary.get("total_laps"),
                summary.get("total_drivers"),
                summary.get("fastest_lap"),
                summary.get("fastest_driver"),
                summary.get("most_stable_driver"),
            ))
            conn.commit()
        logger.info(f"{summary.get('race_name')}: Özet DB'ye kaydedildi.")
    except Exception as e:
        logger.error(f"Yarış özeti kaydedilemedi: {e}")


def mark_race_as_processed(race_name: str, season: int = 2025):
    """Yarışı işlenmiş olarak işaretle (lazy loading için)."""
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO processed_races (race_name, season, status)
                VALUES (?, ?, 'complete')
            """, (race_name, season))
            conn.commit()
    except Exception as e:
        logger.error(f"İşlenmiş yarış işaretlenemedi: {e}")


# ─────────────────────────────────────────────────────────────
# OKUMA FONKSİYONLARI
# ─────────────────────────────────────────────────────────────

def is_race_in_db(race_name: str, season: int = 2025) -> bool:
    """Yarışın veritabanında kayıtlı olup olmadığını kontrol et."""
    try:
        with get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM processed_races
                WHERE race_name = ? AND season = ? AND status = 'complete'
            """, (race_name, season))
            return cursor.fetchone()[0] > 0
    except Exception:
        return False


def load_laps_from_db(race_name: str, season: int = 2025) -> Optional[pd.DataFrame]:
    """Veritabanından tur verisini oku."""
    try:
        with get_connection() as conn:
            df = pd.read_sql("""
                SELECT * FROM laps
                WHERE race_name = ? AND season = ?
            """, conn, params=(race_name, season))
        if len(df) == 0:
            return None
        logger.info(f"{race_name}: {len(df)} tur DB'den okundu.")
        return df
    except Exception as e:
        logger.error(f"{race_name} DB'den okunamadı: {e}")
        return None


def load_driver_stats_from_db(race_name: str) -> Optional[pd.DataFrame]:
    """Veritabanından pilot istatistiklerini oku."""
    try:
        with get_connection() as conn:
            df = pd.read_sql("""
                SELECT * FROM driver_stats WHERE race_name = ?
            """, conn, params=(race_name,))
        return df if len(df) > 0 else None
    except Exception as e:
        logger.error(f"Pilot istatistikleri okunamadı: {e}")
        return None


def load_race_summary_from_db(race_name: str) -> Optional[dict]:
    """Yarış özetini sözlük olarak döndür."""
    try:
        with get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM race_summary WHERE race_name = ?
            """, (race_name,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None
    except Exception as e:
        logger.error(f"Yarış özeti okunamadı: {e}")
        return None


def get_all_processed_races() -> List[str]:
    """Veritabanında kayıtlı tüm işlenmiş yarışları döndür."""
    try:
        with get_connection() as conn:
            cursor = conn.execute("""
                SELECT race_name FROM processed_races WHERE status = 'complete'
                ORDER BY processed_at DESC
            """)
            return [row[0] for row in cursor.fetchall()]
    except Exception:
        return []


if __name__ == "__main__":
    initialize_database()
    print("✅ Veritabanı başlatıldı:", DATABASE_PATH)
    print("İşlenmiş yarışlar:", get_all_processed_races())
