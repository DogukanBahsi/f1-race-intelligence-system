"""
src/analysis.py
---------------
F1 Race Intelligence System - Analiz Modülü
Pilot istikrarı, lastik stratejisi, pit stop etkisi ve hava analizi.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, Any, Tuple, List

from config import COMPOUND_COLORS
from src.logger import get_logger
from src.utils import seconds_to_mmss, generate_auto_comment

logger = get_logger("Analysis")


# ─────────────────────────────────────────────────────────────
# ANA ANALİZ PIPELINE
# ─────────────────────────────────────────────────────────────

def run_all_analyses(df: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Tüm analiz modüllerini çalıştırır.
    Dashboard için hazır veri sözlüğü döndürür.
    """
    logger.info(f"Analizler başlıyor: {race_name}")

    results = {
        "race_name": race_name,
    }

    results["driver_stability"]   = analyze_driver_stability(df, race_name)
    results["tire_strategy"]      = analyze_tire_strategy(df, race_name)
    results["tire_degradation"]   = analyze_tire_degradation(df, race_name)
    results["pit_stop"]           = analyze_pit_stops(df, race_name)
    results["weather_impact"]     = analyze_weather_impact(df, race_name)
    results["race_overview"]      = build_race_overview(df, race_name)

    logger.info(f"Tüm analizler tamamlandı: {race_name}")
    return results


# ─────────────────────────────────────────────────────────────
# A) PİLOT İSTİKRAR ANALİZİ
# ─────────────────────────────────────────────────────────────

def analyze_driver_stability(df: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Pilotların tur zamanı tutarlılığını analiz eder.
    En stabil pilot düşük standart sapmaya sahip olandır.
    """
    if "Driver" not in df.columns or "LapTime" not in df.columns:
        return {"error": "Gerekli sütunlar bulunamadı."}

    # Pit ve SC turları hariç
    clean = _get_clean_laps(df)

    # Pilot bazında istatistikler
    stability = clean.groupby("Driver")["LapTime"].agg(
        avg_lap_time="mean",
        best_lap_time="min",
        lap_time_std="std",
        lap_count="count",
    ).reset_index()

    stability["lap_time_std"] = stability["lap_time_std"].fillna(0.0)

    # Consistency score: 1 / (1 + std). Yüksek = daha istikrarlı.
    stability["consistency_score"] = 1 / (1 + stability["lap_time_std"])

    # Takım bilgisi
    if "Team" in df.columns:
        team_map = df.groupby("Driver")["Team"].first().to_dict()
        stability["Team"] = stability["Driver"].map(team_map)

    # Sıralama: Consistency score'a göre azalan
    stability = stability.sort_values("consistency_score", ascending=False).reset_index(drop=True)
    stability["stability_rank"] = range(1, len(stability) + 1)

    # En stabil pilotun adı
    most_stable = stability.iloc[0]["Driver"] if len(stability) > 0 else "N/A"
    fastest     = stability.loc[stability["best_lap_time"].idxmin(), "Driver"] if len(stability) > 0 else "N/A"

    # Yorum üret
    comment = (
        f"{race_name} yarışında en stabil pilot {most_stable} olmuştur "
        f"(Consistency Score: {stability.iloc[0]['consistency_score']:.3f}). "
        f"En hızlı tur {fastest} adına kayıtlıdır "
        f"({seconds_to_mmss(stability['best_lap_time'].min())})."
    ) if len(stability) > 0 else ""

    return {
        "data":         stability.to_dict("records"),
        "most_stable":  most_stable,
        "fastest":      fastest,
        "comment":      comment,
    }


# ─────────────────────────────────────────────────────────────
# B) LASTİK STRATEJİSİ ANALİZİ
# ─────────────────────────────────────────────────────────────

def analyze_tire_strategy(df: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Soft/Medium/Hard lastik karşılaştırması.
    Hangi pistte hangi lastik daha avantajlı?
    """
    if "Compound" not in df.columns or "LapTime" not in df.columns:
        return {"error": "Compound veya LapTime verisi yok."}

    clean = _get_clean_laps(df)

    # Compound bazında ortalama tur zamanı
    compound_stats = clean.groupby("Compound")["LapTime"].agg(
        avg_lap_time="mean",
        best_lap_time="min",
        std="std",
        count="count",
    ).reset_index()
    compound_stats = compound_stats.sort_values("avg_lap_time")

    # En hızlı lastik
    best_compound = compound_stats.iloc[0]["Compound"] if len(compound_stats) > 0 else "N/A"

    # TyreLife vs LapTime (lastik yaşı ile tur zamanı ilişkisi)
    tyre_age_data = {}
    if "TyreLife" in clean.columns:
        for compound in clean["Compound"].unique():
            compound_df = clean[clean["Compound"] == compound]
            if len(compound_df) > 5:
                age_perf = compound_df.groupby("TyreLife")["LapTime"].mean().reset_index()
                tyre_age_data[compound] = age_perf.to_dict("records")

    # Pilot strateji haritası
    strategy_map = {}
    if "Driver" in df.columns and "Stint" in df.columns:
        for driver in df["Driver"].unique():
            driver_df = df[df["Driver"] == driver]
            stints = driver_df.groupby("Stint").agg(
                compound=("Compound", "first"),
                laps=("LapNumber", "count"),
                avg_time=("LapTime", "mean"),
            ).reset_index()
            strategy_map[driver] = stints.to_dict("records")

    # Yorum
    comment = (
        f"{race_name} yarışında {best_compound} lastikler en hızlı ortalama süreyi "
        f"kaydetmiştir ({seconds_to_mmss(float(compound_stats.iloc[0]['avg_lap_time']))})."
        if len(compound_stats) > 0 else ""
    )

    return {
        "compound_stats": compound_stats.to_dict("records"),
        "tyre_age_data":  tyre_age_data,
        "strategy_map":   strategy_map,
        "best_compound":  best_compound,
        "comment":        comment,
    }


# ─────────────────────────────────────────────────────────────
# C) LASTİK BOZULMA ANALİZİ
# ─────────────────────────────────────────────────────────────

def analyze_tire_degradation(df: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Her lastik için bozulma eğrisi.
    Lastik yaşı arttıkça performans kaybı ne kadar?
    """
    if "TyreLife" not in df.columns or "LapTime" not in df.columns:
        return {"error": "TyreLife veya LapTime verisi yok."}

    clean = _get_clean_laps(df)
    result = {}

    compounds = clean["Compound"].unique() if "Compound" in clean.columns else []

    for compound in compounds:
        cdf = clean[clean["Compound"] == compound] if "Compound" in clean.columns else clean

        if len(cdf) < 5:
            continue

        age_perf = cdf.groupby("TyreLife")["LapTime"].agg(
            mean="mean",
            std="std",
            count="count",
        ).reset_index()

        # Lineer fit (bozulma eğimi)
        if len(age_perf) >= 3:
            x = age_perf["TyreLife"].values.astype(float)
            y = age_perf["mean"].values.astype(float)
            valid = ~(np.isnan(x) | np.isnan(y))

            if valid.sum() >= 2:
                slope, intercept, r, p, _ = stats.linregress(x[valid], y[valid])
                degradation_rate = float(slope)
                r_squared = float(r ** 2)
            else:
                degradation_rate = np.nan
                r_squared = np.nan
        else:
            degradation_rate = np.nan
            r_squared = np.nan

        result[compound] = {
            "age_performance":  age_perf.to_dict("records"),
            "degradation_rate": degradation_rate,
            "r_squared":        r_squared,
            "comment":          generate_auto_comment("degradation", degradation_rate or 0, race_name),
        }

    # En hızlı bozulan lastik
    rates = {k: v["degradation_rate"] for k, v in result.items()
             if v["degradation_rate"] is not None and not np.isnan(v["degradation_rate"])}
    worst_compound = max(rates, key=rates.get) if rates else "N/A"

    return {
        "by_compound":    result,
        "worst_compound": worst_compound,
        "comment": (
            f"{race_name} yarışında en hızlı bozulan lastik {worst_compound} olmuştur "
            f"({rates.get(worst_compound, 0):.3f} s/tur artış)."
        ) if rates else "Yeterli veri yok.",
    }


# ─────────────────────────────────────────────────────────────
# D) PIT STOP ANALİZİ
# ─────────────────────────────────────────────────────────────

def analyze_pit_stops(df: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Pit stop etkisini analiz eder.
    Undercut / overcut için temel yorum üretir.
    """
    if "IsPitLap" not in df.columns:
        return {"error": "Pit lap verisi yok."}

    result = {}

    if "Driver" in df.columns and "LapNumber" in df.columns and "LapTime" in df.columns:
        driver_pit_analysis = []

        for driver in df["Driver"].unique():
            ddf = df[df["Driver"] == driver].sort_values("LapNumber")
            pit_rows = ddf[ddf["IsPitLap"]]

            if pit_rows.empty:
                continue

            impacts = []
            for _, pit_row in pit_rows.iterrows():
                pit_lap = pit_row["LapNumber"]

                before = ddf[
                    (ddf["LapNumber"] < pit_lap) &
                    (ddf["LapNumber"] >= pit_lap - 3) &
                    (~ddf.get("IsOutlier", pd.Series(False, index=ddf.index)))
                ]["LapTime"].mean()

                after = ddf[
                    (ddf["LapNumber"] > pit_lap) &
                    (ddf["LapNumber"] <= pit_lap + 3) &
                    (~ddf.get("IsOutlier", pd.Series(False, index=ddf.index)))
                ]["LapTime"].mean()

                if not np.isnan(before) and not np.isnan(after):
                    impacts.append({
                        "pit_lap":         int(pit_lap),
                        "before_avg":      round(float(before), 3),
                        "after_avg":       round(float(after), 3),
                        "impact":          round(float(before - after), 3),
                    })

            if impacts:
                avg_impact = np.mean([i["impact"] for i in impacts])
                driver_pit_analysis.append({
                    "Driver":       driver,
                    "pit_count":    len(impacts),
                    "avg_impact":   round(float(avg_impact), 3),
                    "details":      impacts,
                })

        result["driver_analysis"] = driver_pit_analysis

        # Genel pit etkisi
        all_impacts = [d["avg_impact"] for d in driver_pit_analysis if d.get("avg_impact")]
        if all_impacts:
            race_avg_impact = float(np.mean(all_impacts))
            result["race_avg_impact"] = round(race_avg_impact, 3)
            result["comment"] = (
                f"{race_name} yarışında ortalama pit stop faydası "
                f"{race_avg_impact:.2f}s olarak ölçülmüştür. "
                + ("Pit stop sonrası hız artışı gözlemlenmiştir (undercut avantajı)."
                   if race_avg_impact > 0 else
                   "Pit stop sonrası beklenen hız artışı sınırlı kalmıştır.")
            )
        else:
            result["comment"] = "Pit stop analizi için yeterli veri yok."

    return result


# ─────────────────────────────────────────────────────────────
# E) HAVA VE SICAKLIK ANALİZİ
# ─────────────────────────────────────────────────────────────

def analyze_weather_impact(df: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Hava ve pist sıcaklığının tur zamanına etkisi.
    Pearson korelasyonu ile istatistiksel güç hesaplanır.
    """
    if "LapTime" not in df.columns:
        return {"error": "LapTime verisi yok."}

    clean = _get_clean_laps(df)
    result = {}

    for col, label in [("AirTemp", "Hava Sıcaklığı"), ("TrackTemp", "Pist Sıcaklığı")]:
        if col not in clean.columns:
            continue

        valid = clean[["LapTime", col]].dropna()
        if len(valid) < 5:
            continue

        corr, p_value = stats.pearsonr(valid[col], valid["LapTime"])

        interpretation = ""
        if abs(corr) < 0.1:
            interpretation = "İhmal edilebilir ilişki"
        elif abs(corr) < 0.3:
            interpretation = "Zayıf ilişki"
        elif abs(corr) < 0.5:
            interpretation = "Orta düzey ilişki"
        elif abs(corr) < 0.7:
            interpretation = "Güçlü ilişki"
        else:
            interpretation = "Çok güçlü ilişki"

        direction = "artarken tur zamanı da artıyor" if corr > 0 else "artarken tur zamanı azalıyor"

        result[col] = {
            "label":          label,
            "pearson_r":      round(float(corr), 4),
            "p_value":        round(float(p_value), 4),
            "significant":    bool(p_value < 0.05),
            "interpretation": interpretation,
            "comment": (
                f"{race_name}: {label} {direction} "
                f"(r={corr:.3f}, p={'<0.05' if p_value < 0.05 else '>0.05'}). "
                f"{interpretation}."
            ),
        }

    return result


# ─────────────────────────────────────────────────────────────
# F) PİLOT KARŞILAŞTIRMA SİSTEMİ
# ─────────────────────────────────────────────────────────────

def compare_drivers(df: pd.DataFrame, driver1: str, driver2: str,
                    race_name: str) -> Dict[str, Any]:
    """
    İki pilot arasında detaylı karşılaştırma üret.
    Ortalama tur, en hızlı tur, sektörler, lastik düşüşü, pit etkisi.
    """
    result = {"driver1": driver1, "driver2": driver2, "race_name": race_name}

    for driver, key in [(driver1, "d1"), (driver2, "d2")]:
        ddf = df[df["Driver"] == driver] if "Driver" in df.columns else pd.DataFrame()
        clean = _get_clean_laps(ddf) if not ddf.empty else pd.DataFrame()

        stats_dict = {}
        if "LapTime" in clean.columns and not clean.empty:
            stats_dict["avg_lap_time"]   = round(float(clean["LapTime"].mean()), 3)
            stats_dict["best_lap_time"]  = round(float(clean["LapTime"].min()), 3)
            stats_dict["lap_time_std"]   = round(float(clean["LapTime"].std()), 3)
            stats_dict["consistency"]    = round(1 / (1 + stats_dict["lap_time_std"]), 4)
            stats_dict["lap_count"]      = len(clean)

        for sec_col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
            if sec_col in clean.columns and not clean.empty:
                stats_dict[sec_col] = round(float(clean[sec_col].mean()), 3)

        if "tire_degradation_rate" in ddf.columns and not ddf.empty:
            stats_dict["tire_degradation"] = round(
                float(ddf["tire_degradation_rate"].mean()), 4)

        if "pit_stop_impact" in ddf.columns and not ddf.empty:
            stats_dict["pit_impact"] = round(
                float(ddf["pit_stop_impact"].mean()), 3)

        if "Compound" in ddf.columns and not ddf.empty:
            stats_dict["compounds_used"] = ddf["Compound"].unique().tolist()

        result[key] = stats_dict

    # Kazanan belirleme
    if "d1" in result and "d2" in result:
        d1, d2 = result["d1"], result["d2"]
        faster = driver1 if d1.get("avg_lap_time", 999) < d2.get("avg_lap_time", 999) else driver2
        more_stable = driver1 if d1.get("consistency", 0) > d2.get("consistency", 0) else driver2
        result["faster_driver"] = faster
        result["more_stable_driver"] = more_stable
        result["comment"] = (
            f"{race_name}: Ortalama hız bakımından {faster} öne çıkarken, "
            f"istikrar açısından {more_stable} daha tutarlı bir performans sergilemiştir."
        )

    return result


# ─────────────────────────────────────────────────────────────
# GENEL YARIŞ ÖZETİ
# ─────────────────────────────────────────────────────────────

def build_race_overview(df: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """Dashboard Overview sekmesi için genel yarış özeti."""
    overview = {
        "race_name":     race_name,
        "total_laps":    len(df),
        "total_drivers": df["Driver"].nunique() if "Driver" in df.columns else 0,
    }

    if "LapTime" in df.columns:
        clean = _get_clean_laps(df)
        if not clean.empty:
            fastest_idx = clean["LapTime"].idxmin()
            overview["fastest_lap"]    = round(float(clean.loc[fastest_idx, "LapTime"]), 3)
            overview["fastest_driver"] = clean.loc[fastest_idx, "Driver"] if "Driver" in clean.columns else "N/A"

    if "consistency_score" in df.columns and "Driver" in df.columns:
        driver_cons = df.groupby("Driver")["consistency_score"].mean()
        overview["most_stable_driver"] = driver_cons.idxmax()

    if "Compound" in df.columns:
        overview["compound_usage"] = df["Compound"].value_counts().to_dict()

    if "AirTemp" in df.columns:
        overview["avg_air_temp"]   = round(float(df["AirTemp"].mean()), 1)
        overview["avg_track_temp"] = round(float(df["TrackTemp"].mean()), 1) if "TrackTemp" in df.columns else None

    return overview


# ─────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYON
# ─────────────────────────────────────────────────────────────

def _get_clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Pit, SC ve outlier turlarını çıkar."""
    clean = df.copy()
    for col in ["IsPitLap", "IsSCLap", "IsOutlier"]:
        if col in clean.columns:
            clean = clean[~clean[col]]
    return clean


if __name__ == "__main__":
    from src.utils import generate_sample_lap_data
    from src.preprocessing import preprocess_laps
    from src.feature_engineering import engineer_features

    sample   = generate_sample_lap_data("Bahrain")
    clean    = preprocess_laps(sample, "Bahrain")
    featured = engineer_features(clean, "Bahrain")
    results  = run_all_analyses(featured, "Bahrain")

    for key, val in results.items():
        print(f"\n{'─'*40}")
        print(f"  {key.upper()}")
        if isinstance(val, dict):
            print(f"  {list(val.keys())}")
