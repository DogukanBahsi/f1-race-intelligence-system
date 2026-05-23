"""
src/feature_engineering.py
--------------------------
F1 Race Intelligence System - Özellik Mühendisliği Modülü
Temizlenmiş veriden makine öğrenmesi için yeni özellikler üretir.
"""

import numpy as np
import pandas as pd
from typing import Optional

from config import PREPROCESSING_CONFIG
from src.logger import get_logger

logger = get_logger("FeatureEngineering")


# ─────────────────────────────────────────────────────────────
# ANA PIPELINE
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, race_name: str) -> pd.DataFrame:
    """
    Tüm özellik mühendisliği adımlarını uygular.
    Her pilot için yeni istatistiksel özellikler üretilir.
    """
    logger.info(f"Feature engineering başladı: {race_name}")
    df = df.copy()

    # ── Temel Özellikler ─────────────────────────────────────
    df = _add_stint_features(df)
    df = _add_temperature_features(df)
    df = _add_sector_features(df)

    # ── Pilot Bazlı Özellikler ────────────────────────────────
    df = _add_driver_lap_stats(df)

    # ── Lastik Özellikler ────────────────────────────────────
    df = _add_tire_features(df)

    # ── Pit Stop Özellikleri ──────────────────────────────────
    df = _add_pit_stop_impact(df)

    # ── Sektör İstikrar Özellikleri ───────────────────────────
    df = _add_sector_consistency(df)

    # ── Genel Yarış Hızı Özellikleri ─────────────────────────
    df = _add_relative_pace(df)

    logger.info(f"Feature engineering tamamlandı: {df.shape[1]} sütun mevcut.")
    return df


# ─────────────────────────────────────────────────────────────
# ÖZELLİK FONKSİYONLARI
# ─────────────────────────────────────────────────────────────

def _add_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stint (lastik seti) numarasını doğrula ve
    stint başlangıcından itibaren geçen tur sayısını hesapla.
    """
    if "Stint" not in df.columns or "Driver" not in df.columns:
        return df

    # Her driver-stint grubu içinde tur sırasını hesapla
    df["StintLap"] = df.groupby(["Driver", "Stint"]).cumcount() + 1
    return df


def _add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sıcaklık ile tur zamanı arasındaki ilişkiyi modelleyen özellikler.
    temperature_sensitivity: Sıcaklık artışının tur zamanına etkisi.
    """
    if "TrackTemp" not in df.columns or "LapTime" not in df.columns:
        return df

    # TrackTemp'in normalize edilmiş hali (0-1 arası)
    track_min = df["TrackTemp"].min()
    track_max = df["TrackTemp"].max()
    if track_max > track_min:
        df["NormTrackTemp"] = (df["TrackTemp"] - track_min) / (track_max - track_min)
    else:
        df["NormTrackTemp"] = 0.5

    return df


def _add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sektör sürelerinden yeni özellikler üretir.
    Pilot hangi sektörde daha güçlü?
    """
    sector_cols = ["Sector1Time", "Sector2Time", "Sector3Time"]
    available   = [c for c in sector_cols if c in df.columns]

    if len(available) < 2:
        return df

    # Sektör toplamı ile LapTime arasındaki fark (veri kalitesi göstergesi)
    try:
        df["SectorSum"] = df[available].sum(axis=1, skipna=True)
    except Exception:
        pass

    return df


def _add_driver_lap_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pilot bazında temel istatistiksel özellikler.
    Pit ve SC turları dahil edilmez (temiz yarış hızı için).

    Üretilen özellikler:
    - average_lap_time     : Pilotun ortalama tur zamanı (saniye)
    - best_lap_time        : Pilotun en hızlı turu
    - lap_time_std         : Tur zamanları standart sapması
    - consistency_score    : İstikrar skoru (düşük std = yüksek istikrar)
    - average_race_pace    : SC/Pit hariç ortalama yarış hızı
    """
    if "LapTime" not in df.columns or "Driver" not in df.columns:
        return df

    # Pit ve outlier turlarını hariç tut
    clean_mask = pd.Series(True, index=df.index)
    if "IsPitLap" in df.columns:
        clean_mask &= ~df["IsPitLap"]
    if "IsOutlier" in df.columns:
        clean_mask &= ~df["IsOutlier"]
    if "IsSCLap" in df.columns:
        clean_mask &= ~df["IsSCLap"]

    clean_df = df[clean_mask]

    # Pilot bazında istatistikler
    driver_stats = clean_df.groupby("Driver")["LapTime"].agg(
        average_lap_time="mean",
        best_lap_time="min",
        lap_time_std="std",
    ).reset_index()
    driver_stats["lap_time_std"] = driver_stats["lap_time_std"].fillna(0)

    # Consistency score: 1 / (1 + std). Yüksek = daha istikrarlı.
    driver_stats["consistency_score"] = 1 / (1 + driver_stats["lap_time_std"])

    # Ortalama yarış hızı (SC hariç)
    if "IsSCLap" in df.columns:
        pace_df = df[~df["IsSCLap"] & clean_mask]
    else:
        pace_df = clean_df

    race_pace = pace_df.groupby("Driver")["LapTime"].mean().reset_index()
    race_pace.columns = ["Driver", "average_race_pace"]

    # Ana DataFrame'e merge
    driver_stats = driver_stats.merge(race_pace, on="Driver", how="left")
    df = df.merge(driver_stats, on="Driver", how="left")

    logger.debug("Pilot lap istatistikleri eklendi.")
    return df


def _add_tire_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lastik özellikleri üret.

    tire_degradation_rate : Lastik yaşı arttıkça tur zamanındaki ortalama artış (s/tur)
    tire_efficiency_score : Lastik tipine göre normalize edilmiş verimlilik
    """
    if "TyreLife" not in df.columns or "LapTime" not in df.columns:
        return df

    # Pit ve SC turlarını hariç tut
    clean_mask = pd.Series(True, index=df.index)
    if "IsPitLap" in df.columns:
        clean_mask &= ~df["IsPitLap"]

    clean_df = df[clean_mask].copy()

    # ── Lastik Bozulma Oranı (Driver + Compound bazında) ──────
    def calc_degradation(group: pd.DataFrame) -> pd.Series:
        """
        Grup içinde lineer regresyon ile bozulma eğimini hesapla.
        Pozitif değer = tur zamanı artıyor = lastik bozuluyor.
        """
        if len(group) < PREPROCESSING_CONFIG["min_stint_laps"]:
            return pd.Series({"tire_degradation_rate": np.nan})

        try:
            x = group["TyreLife"].values.astype(float)
            y = group["LapTime"].values.astype(float)

            # NaN temizle
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 2:
                return pd.Series({"tire_degradation_rate": np.nan})

            # Lineer fit
            slope = np.polyfit(x[valid], y[valid], 1)[0]
            return pd.Series({"tire_degradation_rate": slope})
        except Exception:
            return pd.Series({"tire_degradation_rate": np.nan})

    group_cols = ["Driver"]
    if "Compound" in df.columns:
        group_cols.append("Compound")

    degradation = clean_df.groupby(group_cols).apply(calc_degradation).reset_index()

    # Merge
    df = df.merge(degradation, on=group_cols, how="left")

    # ── Lastik Tipi Verimliliği ───────────────────────────────
    if "Compound" in df.columns:
        compound_avg = clean_df.groupby("Compound")["LapTime"].mean()
        overall_avg  = clean_df["LapTime"].mean()

        if overall_avg > 0:
            # Ortalamanın altında = daha hızlı = daha verimli lastik
            df["tire_efficiency_score"] = df["Compound"].map(
                lambda c: (overall_avg - compound_avg.get(c, overall_avg)) / overall_avg
            )
        else:
            df["tire_efficiency_score"] = 0.0

    logger.debug("Lastik özellikleri eklendi.")
    return df


def _add_pit_stop_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pit stop etkisini ölç.
    pit_stop_impact = Pit öncesi ortalama pace - Pit sonrası ortalama pace
    Pozitif değer = Pit sonrası daha hızlı (taze lastik avantajı)
    """
    if "IsPitLap" not in df.columns or "LapTime" not in df.columns:
        df["pit_stop_impact"] = 0.0
        return df

    try:
        # Her pilotun pit stop öncesi 3 turu ve sonrası 3 turu karşılaştır
        def calc_pit_impact(driver_df: pd.DataFrame) -> float:
            pit_laps = driver_df[driver_df["IsPitLap"]]["LapNumber"]
            if pit_laps.empty:
                return 0.0

            impacts = []
            for pit_lap in pit_laps:
                # Pit öncesi: pit turdan önceki 3 tur
                before = driver_df[
                    (driver_df["LapNumber"] < pit_lap) &
                    (driver_df["LapNumber"] >= pit_lap - 3) &
                    (~driver_df.get("IsOutlier", pd.Series(False, index=driver_df.index)))
                ]["LapTime"]

                # Pit sonrası: pit turundan sonraki 3 tur
                after = driver_df[
                    (driver_df["LapNumber"] > pit_lap) &
                    (driver_df["LapNumber"] <= pit_lap + 3) &
                    (~driver_df.get("IsOutlier", pd.Series(False, index=driver_df.index)))
                ]["LapTime"]

                if len(before) >= 1 and len(after) >= 1:
                    impacts.append(float(before.mean()) - float(after.mean()))

            return float(np.mean(impacts)) if impacts else 0.0

        if "Driver" not in df.columns or "LapNumber" not in df.columns:
            df["pit_stop_impact"] = 0.0
            return df

        pit_impacts = df.groupby("Driver").apply(calc_pit_impact).reset_index()
        pit_impacts.columns = ["Driver", "pit_stop_impact"]

        df = df.merge(pit_impacts, on="Driver", how="left")
        df["pit_stop_impact"] = df["pit_stop_impact"].fillna(0.0)

    except Exception as e:
        logger.warning(f"Pit stop impact hesaplanamadı: {e}")
        df["pit_stop_impact"] = 0.0

    logger.debug("Pit stop impact özellikleri eklendi.")
    return df


def _add_sector_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sektör tutarsızlığını ölç.
    sector_consistency = Sektör zamanlarının ortalama standart sapması
    Düşük değer = daha tutarlı sektör performansı
    """
    sector_cols = ["Sector1Time", "Sector2Time", "Sector3Time"]
    available   = [c for c in sector_cols if c in df.columns]

    if len(available) < 2 or "Driver" not in df.columns:
        df["sector_consistency"] = np.nan
        return df

    try:
        # Her pilot için sektör std ortalaması
        def calc_sector_consistency(group: pd.DataFrame) -> float:
            stds = []
            for col in available:
                std = group[col].std()
                if not np.isnan(std):
                    stds.append(std)
            return float(np.mean(stds)) if stds else np.nan

        sector_cons = df.groupby("Driver").apply(calc_sector_consistency).reset_index()
        sector_cons.columns = ["Driver", "sector_consistency"]

        df = df.merge(sector_cons, on="Driver", how="left")

    except Exception as e:
        logger.warning(f"Sektör consistency hesaplanamadı: {e}")
        df["sector_consistency"] = np.nan

    return df


def _add_relative_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pilotun yarış ortalamasına göre relatif hızı.
    relative_pace = Pilotun ort. tur zamanı - Yarış genel ortalaması
    Negatif değer = Ortalamadan daha hızlı (iyi)
    """
    if "average_lap_time" not in df.columns or "LapTime" not in df.columns:
        return df

    race_mean = df["LapTime"].mean()
    if race_mean > 0:
        df["relative_pace"] = df["average_lap_time"] - race_mean
    else:
        df["relative_pace"] = 0.0

    return df


# ─────────────────────────────────────────────────────────────
# ML İÇİN ÖZELLİK MATRİSİ
# ─────────────────────────────────────────────────────────────

def get_driver_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    ML modelleri için pilot bazında özellik matrisi oluştur.
    Her satır bir pilot, her sütun bir özellik.
    """
    feature_cols = [
        "Driver",
        "average_lap_time",
        "best_lap_time",
        "lap_time_std",
        "consistency_score",
        "average_race_pace",
        "tire_degradation_rate",
        "pit_stop_impact",
        "sector_consistency",
        "AirTemp",
        "TrackTemp",
    ]

    available = [c for c in feature_cols if c in df.columns]

    # Pilot bazında agg (ilk değer yeterli çünkü pilot bazında zaten hesaplanmış)
    agg_dict = {}
    for col in available:
        if col == "Driver":
            continue
        agg_dict[col] = "first"

    if not agg_dict:
        return pd.DataFrame()

    # Compound bazlı istatistikler ekle
    if "tire_degradation_rate" in df.columns and "Compound" in df.columns:
        compound_deg = df.groupby(["Driver", "Compound"])["tire_degradation_rate"].mean()
        compound_deg = compound_deg.unstack(fill_value=0)
        compound_deg.columns = [f"degradation_{c.lower()}" for c in compound_deg.columns]
        compound_deg = compound_deg.reset_index()
    else:
        compound_deg = None

    matrix = df.groupby("Driver").agg(agg_dict).reset_index()

    if compound_deg is not None:
        matrix = matrix.merge(compound_deg, on="Driver", how="left")

    # Hava sıcaklığı ortalaması
    if "AirTemp" in matrix.columns:
        matrix["avg_air_temp"] = df.groupby("Driver")["AirTemp"].mean().values

    matrix = matrix.fillna(matrix.median(numeric_only=True))

    logger.debug(f"Driver feature matrix: {matrix.shape}")
    return matrix


if __name__ == "__main__":
    from src.utils import generate_sample_lap_data
    from src.preprocessing import preprocess_laps

    sample = generate_sample_lap_data("Test")
    clean  = preprocess_laps(sample, "Test")
    featured = engineer_features(clean, "Test")

    matrix = get_driver_feature_matrix(featured)
    print("✅ Feature matrix shape:", matrix.shape)
    print(matrix.head())
