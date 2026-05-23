"""
src/preprocessing.py
--------------------
F1 Race Intelligence System - Veri Ön İşleme Modülü
Ham FastF1 verisini analize hazır hale getirir.
"""

import numpy as np
import pandas as pd
from typing import Optional

from config import PREPROCESSING_CONFIG, PROCESSED_DATA_DIR
from src.logger import get_logger
from src.utils import timedelta_to_seconds, get_processed_csv_path
from src.database import save_laps_to_db, mark_race_as_processed

logger = get_logger("Preprocessing")


# ─────────────────────────────────────────────────────────────
# ANA PIPELINE
# ─────────────────────────────────────────────────────────────

def preprocess_laps(df: pd.DataFrame, race_name: str) -> pd.DataFrame:
    """
    Tur verisinin tüm ön işleme adımlarını uygular.
    Döndürdüğü değer: Temizlenmiş DataFrame
    """
    logger.info(f"Preprocessing started: {race_name} ({len(df)} satır)")
    original_len = len(df)

    df = df.copy()

    # ── Adım 1: Sütun adlarını standartlaştır ─────────────────
    df = _standardize_columns(df)

    # ── Adım 2: Zaman sütunlarını saniyeye çevir ──────────────
    df = _convert_times_to_seconds(df)

    # ── Adım 3: LapTime olmayan satırları çıkar ───────────────
    df = _drop_missing_laptimes(df)

    # ── Adım 4: Safety Car / VSC turlarını işaretle ───────────
    df = _label_safety_car_laps(df)

    # ── Adım 5: Pit stop yapılan turları işaretle ─────────────
    df = _label_pit_laps(df)

    # ── Adım 6: TrackStatus'u anlamlı hale getir ─────────────
    df = _decode_track_status(df)

    # ── Adım 7: Aykırı tur zamanlarını tespit et ve filtrele ──
    df = _filter_outlier_laps(df)

    # ── Adım 8: Eksik hava verilerini doldur ─────────────────
    df = _fill_missing_weather(df)

    # ── Adım 9: Compound standardize et ──────────────────────
    df = _standardize_compound(df)

    # ── Adım 10: Takım bilgisini temizle ─────────────────────
    df = _clean_team_info(df)

    # Yarış adını ekle
    df["RaceName"] = race_name

    final_len = len(df)
    removed   = original_len - final_len
    logger.info(f"Preprocessing tamamlandı: {final_len} satır ({removed} satır çıkarıldı).")

    # Veritabanına kaydet
    try:
        save_laps_to_db(df, race_name)
        mark_race_as_processed(race_name)
    except Exception as e:
        logger.warning(f"DB kaydetme başarısız (preprocessing devam ediyor): {e}")

    return df


# ─────────────────────────────────────────────────────────────
# ADIM ADIM FONKSİYONLAR
# ─────────────────────────────────────────────────────────────

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sütun adlarını standart formata çevir.
    FastF1'den gelen bazı sütunlar farklı adlarla gelebilir.
    """
    rename_map = {
        "driver":        "Driver",
        "team":          "Team",
        "lapnumber":     "LapNumber",
        "laptime":       "LapTime",
        "sector1time":   "Sector1Time",
        "sector2time":   "Sector2Time",
        "sector3time":   "Sector3Time",
        "compound":      "Compound",
        "tyrelife":      "TyreLife",
        "stint":         "Stint",
        "airtemp":       "AirTemp",
        "tracktemp":     "TrackTemp",
        "humidity":      "Humidity",
        "trackstatus":   "TrackStatus",
        "pitintime":     "PitInTime",
        "pitouttime":    "PitOutTime",
        "ispersonalbest": "IsPersonalBest",
        "racename":      "RaceName",
    }
    # Küçük harfe çevirip eşleştir
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}

    for std_lower, std_name in rename_map.items():
        if std_lower in lower_map and std_name not in df.columns:
            df = df.rename(columns={lower_map[std_lower]: std_name})

    return df


def _convert_times_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Timedelta formatındaki sütunları saniyeye çevir.
    Zaten sayısal olanlar dokunulmaz.
    """
    time_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]

    for col in time_cols:
        if col not in df.columns:
            continue

        # Zaten sayısal ise dokunma
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # String veya timedelta → saniye
        try:
            df[col] = df[col].apply(timedelta_to_seconds)
        except Exception as e:
            logger.warning(f"{col} dönüştürme hatası: {e}")

    logger.debug("Zaman sütunları saniyeye çevrildi.")
    return df


def _drop_missing_laptimes(df: pd.DataFrame) -> pd.DataFrame:
    """LapTime değeri olmayan veya 0 olan satırları çıkar."""
    if "LapTime" not in df.columns:
        logger.warning("LapTime sütunu bulunamadı!")
        return df

    before = len(df)
    df = df.dropna(subset=["LapTime"])
    df = df[df["LapTime"] > 0]
    removed = before - len(df)
    if removed:
        logger.debug(f"{removed} satır çıkarıldı (LapTime eksik/sıfır).")
    return df


def _label_safety_car_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safety Car (SC) ve Virtual Safety Car (VSC) turlarını etiketle.
    TrackStatus kodu: '4' = SC, '5' = Red Flag, '6' = VSC
    """
    df["IsSCLap"] = False

    if "TrackStatus" not in df.columns:
        return df

    # TrackStatus string veya int olabilir
    status = df["TrackStatus"].astype(str)
    sc_mask = status.str.contains("4", na=False) | status.str.contains("6", na=False)
    df.loc[sc_mask, "IsSCLap"] = True

    sc_count = sc_mask.sum()
    if sc_count > 0:
        logger.debug(f"{sc_count} SC/VSC turu tespit edildi ve etiketlendi.")

    return df


def _label_pit_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pit stop yapılan turları etiketle.
    Pit turları normal tur zamanının 1.5 katından uzun olma eğilimindedir.
    Aynı zamanda PitInTime / PitOutTime sütunları da kontrol edilir.
    """
    df["IsPitLap"] = False

    if "LapTime" not in df.columns:
        return df

    multiplier = PREPROCESSING_CONFIG["pit_lap_time_multiplier"]

    # Yöntem 1: Tur zamanı ortalamadan çok uzunsa pit turu
    median_time = df["LapTime"].median()
    time_mask   = df["LapTime"] > median_time * multiplier

    # Yöntem 2: PitInTime / PitOutTime var ve NaN değilse
    pit_mask = pd.Series(False, index=df.index)
    if "PitInTime" in df.columns:
        pit_mask = pit_mask | df["PitInTime"].notna()
    if "PitOutTime" in df.columns:
        pit_mask = pit_mask | df["PitOutTime"].notna()

    df.loc[time_mask | pit_mask, "IsPitLap"] = True

    pit_count = (time_mask | pit_mask).sum()
    logger.debug(f"{pit_count} pit turu etiketlendi.")

    return df


def _decode_track_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    TrackStatus sayısal kodlarını anlamlı etiketlere çevir.
    1=Yeşil, 2=Sarı, 4=SC, 5=Kırmızı, 6=VSC
    """
    status_map = {
        "1": "Green",
        "2": "Yellow",
        "3": "Yellow",
        "4": "SafetyCar",
        "5": "RedFlag",
        "6": "VSC",
    }

    if "TrackStatus" not in df.columns:
        df["TrackStatusLabel"] = "Unknown"
        return df

    df["TrackStatusLabel"] = (
        df["TrackStatus"]
        .astype(str)
        .map(status_map)
        .fillna("Unknown")
    )

    return df


def _filter_outlier_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aykırı tur zamanlarını tespit et.
    Standart sapma eşiğini aşan turlar 'IsOutlier' olarak işaretlenir.
    Pit ve SC turları outlier analizinden hariç tutulur.
    """
    df["IsOutlier"] = False

    if "LapTime" not in df.columns:
        return df

    # Pit ve SC turları hariç analiz et
    mask = ~df.get("IsPitLap", pd.Series(False, index=df.index))
    mask &= ~df.get("IsSCLap", pd.Series(False, index=df.index))
    normal_laps = df[mask]

    if len(normal_laps) < 10:
        return df  # Yetersiz veri

    mean_time = normal_laps["LapTime"].mean()
    std_time  = normal_laps["LapTime"].std()
    threshold = PREPROCESSING_CONFIG["outlier_std_threshold"]

    lower = mean_time - threshold * std_time
    upper = mean_time + threshold * std_time

    outlier_mask = (df["LapTime"] < lower) | (df["LapTime"] > upper)
    df.loc[outlier_mask, "IsOutlier"] = True

    outlier_count = outlier_mask.sum()
    if outlier_count > 0:
        logger.debug(f"{outlier_count} outlier tur tespit edildi (±{threshold}σ).")

    return df


def _fill_missing_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksik hava verilerini medyan ile doldur.
    Sıcaklık değerleri analiz için önemlidir.
    """
    weather_cols = ["AirTemp", "TrackTemp", "Humidity"]
    for col in weather_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df


def _standardize_compound(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lastik bileşiği adlarını büyük harfe çevir ve bilinmeyenleri standartlaştır.
    """
    if "Compound" not in df.columns:
        df["Compound"] = "UNKNOWN"
        return df

    valid_compounds = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
    df["Compound"] = (
        df["Compound"]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    df.loc[~df["Compound"].isin(valid_compounds), "Compound"] = "UNKNOWN"
    return df


def _clean_team_info(df: pd.DataFrame) -> pd.DataFrame:
    """Team sütunu yoksa veya boşsa varsayılan değer ata."""
    if "Team" not in df.columns:
        df["Team"] = "Unknown"
    else:
        df["Team"] = df["Team"].fillna("Unknown").astype(str)
    return df


# ─────────────────────────────────────────────────────────────
# EDA İÇİN ÖZET
# ─────────────────────────────────────────────────────────────

def get_preprocessing_summary(df: pd.DataFrame) -> dict:
    """
    Preprocessing sonrası veri kalitesi özeti döndür.
    EDA modülü tarafından kullanılır.
    """
    return {
        "total_rows":       len(df),
        "total_drivers":    df["Driver"].nunique() if "Driver" in df.columns else 0,
        "missing_laptimes": df["LapTime"].isna().sum() if "LapTime" in df.columns else 0,
        "pit_laps":         df["IsPitLap"].sum() if "IsPitLap" in df.columns else 0,
        "sc_laps":          df["IsSCLap"].sum()  if "IsSCLap"  in df.columns else 0,
        "outlier_laps":     df["IsOutlier"].sum() if "IsOutlier" in df.columns else 0,
        "compounds":        df["Compound"].value_counts().to_dict() if "Compound" in df.columns else {},
        "avg_lap_time":     df["LapTime"].mean() if "LapTime" in df.columns else None,
        "min_lap_time":     df["LapTime"].min()  if "LapTime" in df.columns else None,
        "max_lap_time":     df["LapTime"].max()  if "LapTime" in df.columns else None,
    }


if __name__ == "__main__":
    from src.utils import generate_sample_lap_data
    sample = generate_sample_lap_data("Test")
    clean  = preprocess_laps(sample, "Test")
    summary = get_preprocessing_summary(clean)
    print("✅ Preprocessing özeti:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
