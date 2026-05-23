"""
src/data_loader.py
------------------
F1 Race Intelligence System - Veri Yükleme Modülü
FastF1 ile veri çekme, lazy loading ve ham veri kaydetme işlemleri.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from config import (
    SEASON, CACHE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    FASTF1_RETRY_CONFIG, SESSION_TYPE
)
from src.logger import get_logger, log_success
from src.utils import (
    race_name_to_slug, get_raw_csv_path, get_processed_csv_path,
    generate_sample_lap_data
)
from src.database import is_race_in_db

logger = get_logger("DataLoader")


# ─────────────────────────────────────────────────────────────
# FastF1 KURULUMU
# ─────────────────────────────────────────────────────────────

def setup_fastf1_cache():
    """FastF1 cache'ini etkinleştir."""
    try:
        import fastf1
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        logger.info(f"FastF1 cache etkinleştirildi: {CACHE_DIR}")
    except ImportError:
        logger.warning("fastf1 kütüphanesi bulunamadı. Örnek veri kullanılacak.")
    except Exception as e:
        logger.error(f"FastF1 cache kurulamadı: {e}")


# ─────────────────────────────────────────────────────────────
# LAZY LOADING KONTROL
# ─────────────────────────────────────────────────────────────

def is_race_processed(race_name: str) -> bool:
    """
    Yarışın daha önce işlenip işlenmediğini kontrol et.
    İki ayrı kontrol yapar:
    1. SQLite'ta kayıtlı mı?
    2. Processed CSV mevcut mu?
    """
    # Veritabanı kontrolü
    if is_race_in_db(race_name, SEASON):
        return True

    # CSV kontrolü
    processed_path = get_processed_csv_path(race_name, PROCESSED_DATA_DIR)
    if processed_path.exists() and processed_path.stat().st_size > 100:
        logger.debug(f"{race_name}: İşlenmiş CSV mevcut.")
        return True

    return False


def is_raw_data_available(race_name: str) -> bool:
    """Ham veri CSV'sinin mevcut olup olmadığını kontrol et."""
    raw_path = get_raw_csv_path(race_name, RAW_DATA_DIR, "laps")
    return raw_path.exists() and raw_path.stat().st_size > 100


# ─────────────────────────────────────────────────────────────
# FastF1'DEN VERİ ÇEKME
# ─────────────────────────────────────────────────────────────

def fetch_race_data_from_fastf1(race_name: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    FastF1 API'sinden belirtilen yarışın verisini çek.
    Retry mekanizması uygulanır.
    Başarısız olursa None döndürür.
    """
    max_retries = FASTF1_RETRY_CONFIG["max_retries"]
    retry_delay = FASTF1_RETRY_CONFIG["retry_delay"]

    for attempt in range(1, max_retries + 1):
        try:
            import fastf1
            logger.info(f"Loading {race_name} GP {SEASON}... (Deneme {attempt}/{max_retries})")

            # Oturum yükle
            session = fastf1.get_session(SEASON, race_name, SESSION_TYPE)
            session.load(
                telemetry=False,   # Telemetri opsiyonel - ana sistem için gerek yok
                weather=True,
                messages=False,
            )

            # ── Tur Verisi ─────────────────────────────────────
            laps = session.laps.copy()
            if laps.empty:
                logger.warning(f"{race_name}: Tur verisi boş geldi.")
                return None

            # ── Hava Durumu Verisi ─────────────────────────────
            weather_df = None
            try:
                weather_df = session.weather_data.copy()
            except Exception:
                logger.warning(f"{race_name}: Hava verisi alınamadı.")

            # ── Pit Stop Verisi ────────────────────────────────
            pit_df = None
            try:
                pit_df = session.laps[["Driver", "LapNumber", "PitInTime", "PitOutTime"]].copy()
            except Exception:
                logger.warning(f"{race_name}: Pit verisi alınamadı.")

            logger.info(f"{race_name}: {len(laps)} tur, {laps['Driver'].nunique()} pilot verisi çekildi.")

            return {
                "laps":    laps,
                "weather": weather_df,
                "pits":    pit_df,
                "session": session,
            }

        except ImportError:
            logger.error("fastf1 yüklü değil. 'pip install fastf1' komutu çalıştırın.")
            return None

        except Exception as e:
            logger.warning(f"{race_name} yüklenemedi (Deneme {attempt}): {e}")
            if attempt < max_retries:
                logger.info(f"{retry_delay}s bekleniyor...")
                time.sleep(retry_delay)
            else:
                logger.error(f"{race_name}: FastF1 data could not be loaded. Tüm denemeler başarısız.")
                return None

    return None


# ─────────────────────────────────────────────────────────────
# HAM VERİYİ DÜZENLEME VE KAYDETME
# ─────────────────────────────────────────────────────────────

def flatten_laps_with_weather(laps: pd.DataFrame,
                               weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Tur verisine hava durumu sütunlarını ekle.
    Zaman damgasına göre en yakın hava verisini eşleştir.
    """
    # LapTime'ı saniyeye çevir
    if "LapTime" in laps.columns:
        laps["LapTime"] = laps["LapTime"].apply(
            lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
        )

    # Sektör sürelerini saniyeye çevir
    for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[col] = laps[col].apply(
                lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
            )

    # Hava verisi yoksa varsayılan değerler ekle
    if weather_df is None or weather_df.empty:
        laps["AirTemp"]   = np.nan
        laps["TrackTemp"] = np.nan
        laps["Humidity"]  = np.nan
        return laps

    # Hava verisinden ortalama değerleri al (basit yaklaşım)
    try:
        avg_air   = weather_df["AirTemp"].mean()   if "AirTemp"   in weather_df else np.nan
        avg_track = weather_df["TrackTemp"].mean()  if "TrackTemp"  in weather_df else np.nan
        avg_hum   = weather_df["Humidity"].mean()   if "Humidity"   in weather_df else np.nan

        if "AirTemp"   not in laps.columns: laps["AirTemp"]   = avg_air
        if "TrackTemp" not in laps.columns: laps["TrackTemp"] = avg_track
        if "Humidity"  not in laps.columns: laps["Humidity"]  = avg_hum
    except Exception as e:
        logger.warning(f"Hava verisi eşleştirilemedi: {e}")

    return laps


def save_raw_data(race_name: str, data: Dict[str, pd.DataFrame]):
    """
    Ham veriyi CSV olarak kaydet.
    Veriler raw/ klasörüne yazılır.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    slug = race_name_to_slug(race_name)

    # Tur verisi
    laps = data.get("laps")
    if laps is not None and not laps.empty:
        laps_path = get_raw_csv_path(race_name, RAW_DATA_DIR, "laps")
        laps.to_csv(laps_path, index=False)
        logger.info(f"Raw data saved: {laps_path.name}")

    # Hava verisi
    weather = data.get("weather")
    if weather is not None and not weather.empty:
        weather_path = RAW_DATA_DIR / f"{slug}_weather_raw.csv"
        weather.to_csv(weather_path, index=False)

    # Pit verisi
    pits = data.get("pits")
    if pits is not None and not pits.empty:
        pits_path = RAW_DATA_DIR / f"{slug}_pits_raw.csv"
        pits.to_csv(pits_path, index=False)


def load_raw_data_from_csv(race_name: str) -> Optional[pd.DataFrame]:
    """Ham tur CSV'sini diskten oku."""
    raw_path = get_raw_csv_path(race_name, RAW_DATA_DIR, "laps")
    if not raw_path.exists():
        return None
    try:
        df = pd.read_csv(raw_path, low_memory=False)
        logger.info(f"{race_name}: Ham veri CSV'den okundu ({len(df)} satır).")
        return df
    except Exception as e:
        logger.error(f"Ham CSV okunamadı: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# ANA LAZY LOADING FONKSİYONU
# ─────────────────────────────────────────────────────────────

def load_or_fetch_race_data(race_name: str,
                             use_sample_on_failure: bool = True) -> Optional[pd.DataFrame]:
    """
    Lazy loading mantığı:
    1. İşlenmiş veri varsa → doğrudan processed CSV'den oku
    2. Ham veri varsa → preprocessing pipeline'a gönder (işlenmiş CSV yoksa)
    3. Hiç veri yoksa → FastF1'den indir, kaydet, işle
    4. FastF1 başarısız olursa → örnek veri üret (opsiyonel)

    Döndürdüğü şey: Temiz, feature'lı DataFrame
    """
    from src.preprocessing import preprocess_laps
    from src.feature_engineering import engineer_features

    logger.info(f"{'='*50}")
    logger.info(f"Yarış yükleniyor: {race_name} {SEASON}")

    # ── Adım 1: İşlenmiş CSV mevcut mu? ───────────────────────
    processed_path = get_processed_csv_path(race_name, PROCESSED_DATA_DIR)
    if processed_path.exists() and processed_path.stat().st_size > 100:
        try:
            df = pd.read_csv(processed_path, low_memory=False)
            log_success(f"{race_name}: Processed CSV'den yüklendi ({len(df)} tur).")
            return df
        except Exception as e:
            logger.warning(f"Processed CSV okunamadı: {e}. Yeniden işleniyor.")

    # ── Adım 2: Ham CSV mevcut mu? ────────────────────────────
    raw_df = load_raw_data_from_csv(race_name)

    if raw_df is None:
        # ── Adım 3: FastF1'den indir ─────────────────────────
        setup_fastf1_cache()
        data = fetch_race_data_from_fastf1(race_name)

        if data is not None:
            laps = flatten_laps_with_weather(data["laps"], data.get("weather"))
            laps["RaceName"] = race_name
            save_raw_data(race_name, {"laps": laps, "weather": data.get("weather")})
            raw_df = laps
        else:
            # ── Adım 4: Örnek veri ───────────────────────────
            if use_sample_on_failure:
                raw_df = generate_sample_lap_data(race_name)
                raw_df.to_csv(get_raw_csv_path(race_name, RAW_DATA_DIR, "laps"), index=False)
            else:
                logger.error(f"{race_name} verisi alınamadı ve örnek veri devre dışı.")
                return None

    # ── Preprocessing + Feature Engineering ──────────────────
    logger.info("Preprocessing başlıyor...")
    clean_df = preprocess_laps(raw_df, race_name)

    logger.info("Feature engineering başlıyor...")
    featured_df = engineer_features(clean_df, race_name)

    # İşlenmiş veriyi kaydet
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(processed_path, index=False)
    log_success(f"{race_name}: İşlenmiş veri kaydedildi ({len(featured_df)} tur).")

    return featured_df


# ─────────────────────────────────────────────────────────────
# TELEMETRY (BONUS - Ana sistemi etkilemez)
# ─────────────────────────────────────────────────────────────

def fetch_driver_telemetry(race_name: str, driver: str) -> Optional[pd.DataFrame]:
    """
    Bonus: Belirtilen pilot için telemetry verisini çek.
    Hata olursa None döndürür, ana sistemi bozmaz.
    """
    try:
        import fastf1
        setup_fastf1_cache()
        session = fastf1.get_session(SEASON, race_name, SESSION_TYPE)
        session.load(telemetry=True, weather=False, messages=False)

        # Pilotun en hızlı turunu bul
        driver_laps = session.laps.pick_driver(driver)
        if driver_laps.empty:
            return None

        fast_lap = driver_laps.pick_fastest()
        telemetry = fast_lap.get_telemetry()
        logger.info(f"{driver} telemetry verisi yüklendi ({len(telemetry)} veri noktası).")
        return telemetry

    except Exception as e:
        logger.warning(f"Telemetry alınamadı ({driver}, {race_name}): {e}")
        return None


if __name__ == "__main__":
    # Hızlı test
    df = load_or_fetch_race_data("Bahrain", use_sample_on_failure=True)
    if df is not None:
        print(f"✅ Veri yüklendi: {df.shape}")
        print(df.head())
