"""
src/eda.py
----------
F1 Race Intelligence System - Keşifsel Veri Analizi (EDA)
Veriyi tanımak, eksiklikleri görmek ve dağılımları anlamak için kullanılır.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI olmayan ortamlar için
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any

from config import FIGURES_DIR, COMPOUND_COLORS
from src.logger import get_logger

logger = get_logger("EDA")


# ─────────────────────────────────────────────────────────────
# ANA EDA FONKSİYONU
# ─────────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Tüm EDA adımlarını çalıştırır.
    Konsol özeti + grafik dosyaları üretir.
    Dashboard için hazır veri döndürür.
    """
    logger.info(f"EDA başlıyor: {race_name}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── 1. Temel Özet ────────────────────────────────────────
    results["basic_summary"] = _basic_summary(df, race_name)

    # ── 2. Eksik Değer Analizi ───────────────────────────────
    results["missing_values"] = _missing_value_analysis(df)

    # ── 3. Compound Dağılımı ─────────────────────────────────
    results["compound_dist"] = _compound_distribution(df, race_name)

    # ── 4. Pilot Tur Sayıları ────────────────────────────────
    results["driver_laps"] = _driver_lap_counts(df)

    # ── 5. Tur Zamanı Dağılımı ───────────────────────────────
    results["lap_time_dist"] = _lap_time_distribution(df, race_name)

    # ── 6. Sıcaklık Dağılımları ──────────────────────────────
    results["temp_analysis"] = _temperature_analysis(df, race_name)

    # ── 7. Korelasyon Matrisi ────────────────────────────────
    results["correlation"] = _correlation_analysis(df, race_name)

    # Konsola özet yazdır
    _print_eda_summary(results, race_name)

    logger.info(f"EDA tamamlandı: {race_name}")
    return results


# ─────────────────────────────────────────────────────────────
# EDA ALT FONKSİYONLARI
# ─────────────────────────────────────────────────────────────

def _basic_summary(df: pd.DataFrame, race_name: str) -> dict:
    """Veri boyutu, sütun isimleri ve temel istatistikler."""
    summary = {
        "race_name":    race_name,
        "total_rows":   len(df),
        "total_cols":   len(df.columns),
        "drivers":      df["Driver"].nunique() if "Driver" in df.columns else 0,
        "driver_list":  df["Driver"].unique().tolist() if "Driver" in df.columns else [],
        "total_laps":   len(df),
    }

    if "LapTime" in df.columns:
        summary["avg_lap_time"] = float(df["LapTime"].mean())
        summary["min_lap_time"] = float(df["LapTime"].min())
        summary["max_lap_time"] = float(df["LapTime"].max())
        summary["std_lap_time"] = float(df["LapTime"].std())

    if "AirTemp" in df.columns:
        summary["avg_air_temp"]   = float(df["AirTemp"].mean())
        summary["avg_track_temp"] = float(df["TrackTemp"].mean()) if "TrackTemp" in df.columns else None

    if "IsPitLap" in df.columns:
        summary["pit_laps"] = int(df["IsPitLap"].sum())
    if "IsSCLap" in df.columns:
        summary["sc_laps"]  = int(df["IsSCLap"].sum())
    if "IsOutlier" in df.columns:
        summary["outliers"] = int(df["IsOutlier"].sum())

    return summary


def _missing_value_analysis(df: pd.DataFrame) -> dict:
    """Her sütun için eksik değer sayısı ve oranı."""
    missing = {}
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            missing[col] = {
                "count":   int(n_missing),
                "percent": round(100 * n_missing / len(df), 2),
            }
    return missing


def _compound_distribution(df: pd.DataFrame, race_name: str) -> dict:
    """Lastik bileşiği kullanım dağılımı."""
    if "Compound" not in df.columns:
        return {}

    dist = df["Compound"].value_counts().to_dict()

    # Grafik üret
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        compounds  = list(dist.keys())
        counts     = list(dist.values())
        colors     = [COMPOUND_COLORS.get(c, "#888888") for c in compounds]

        ax.bar(compounds, counts, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{race_name} - Lastik Tipi Dağılımı", fontsize=14, fontweight="bold")
        ax.set_xlabel("Lastik Tipi")
        ax.set_ylabel("Tur Sayısı")
        ax.set_facecolor("#1a1a1a")
        fig.patch.set_facecolor("#0d0d0d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        save_path = FIGURES_DIR / f"{race_name.lower().replace(' ', '_')}_compound_dist.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Compound dağılım grafiği üretilemedi: {e}")

    return dist


def _driver_lap_counts(df: pd.DataFrame) -> dict:
    """Her pilot için tur sayısı."""
    if "Driver" not in df.columns:
        return {}
    return df["Driver"].value_counts().to_dict()


def _lap_time_distribution(df: pd.DataFrame, race_name: str) -> dict:
    """Tur zamanı dağılım istatistikleri ve histogram."""
    if "LapTime" not in df.columns:
        return {}

    # Pit ve outlier turları hariç
    clean = df.copy()
    if "IsPitLap" in clean.columns:
        clean = clean[~clean["IsPitLap"]]
    if "IsOutlier" in clean.columns:
        clean = clean[~clean["IsOutlier"]]

    lap_times = clean["LapTime"].dropna()

    stats = {
        "mean":   float(lap_times.mean()),
        "median": float(lap_times.median()),
        "std":    float(lap_times.std()),
        "q25":    float(lap_times.quantile(0.25)),
        "q75":    float(lap_times.quantile(0.75)),
        "min":    float(lap_times.min()),
        "max":    float(lap_times.max()),
    }

    # Histogram
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(lap_times, bins=40, color="#E8002D", alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.axvline(stats["mean"],   color="yellow", linestyle="--", label=f'Ort: {stats["mean"]:.2f}s', linewidth=1.5)
        ax.axvline(stats["median"], color="cyan",   linestyle="-",  label=f'Med: {stats["median"]:.2f}s', linewidth=1.5)
        ax.set_title(f"{race_name} - Tur Zamanı Dağılımı (Pit/SC Hariç)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Tur Zamanı (saniye)")
        ax.set_ylabel("Frekans")
        ax.legend(facecolor="#2a2a2a", labelcolor="white")
        ax.set_facecolor("#1a1a1a")
        fig.patch.set_facecolor("#0d0d0d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

        save_path = FIGURES_DIR / f"{race_name.lower().replace(' ', '_')}_laptime_hist.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Tur zamanı histogramı üretilemedi: {e}")

    return stats


def _temperature_analysis(df: pd.DataFrame, race_name: str) -> dict:
    """Hava ve pist sıcaklıklarının özet istatistikleri."""
    result = {}

    for col, label in [("AirTemp", "Hava"), ("TrackTemp", "Pist")]:
        if col in df.columns:
            result[col] = {
                "mean": float(df[col].mean()),
                "min":  float(df[col].min()),
                "max":  float(df[col].max()),
                "std":  float(df[col].std()),
            }

    # Sıcaklık vs tur zamanı scatter
    try:
        if "TrackTemp" in df.columns and "LapTime" in df.columns:
            clean = df.copy()
            if "IsPitLap" in clean.columns:
                clean = clean[~clean["IsPitLap"]]
            if "IsOutlier" in clean.columns:
                clean = clean[~clean["IsOutlier"]]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            for ax, (col, label) in zip(axes, [("AirTemp", "Hava Sıcaklığı"), ("TrackTemp", "Pist Sıcaklığı")]):
                if col in clean.columns:
                    ax.scatter(clean[col], clean["LapTime"],
                               alpha=0.3, color="#E8002D", s=10)
                    ax.set_xlabel(f"{label} (°C)")
                    ax.set_ylabel("Tur Zamanı (s)")
                    ax.set_title(f"{label} vs Tur Zamanı")
                    ax.set_facecolor("#1a1a1a")

            fig.suptitle(f"{race_name} - Sıcaklık Etkisi", fontsize=13, fontweight="bold", color="white")
            fig.patch.set_facecolor("#0d0d0d")
            for ax in axes:
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")

            save_path = FIGURES_DIR / f"{race_name.lower().replace(' ', '_')}_temp_scatter.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        logger.warning(f"Sıcaklık scatter grafiği üretilemedi: {e}")

    return result


def _correlation_analysis(df: pd.DataFrame, race_name: str) -> dict:
    """
    Sayısal değişkenler arasındaki Pearson korelasyonunu hesapla.
    LapTime ile güçlü ilişkisi olan değişkenleri belirle.
    """
    numeric_cols = ["LapTime", "TyreLife", "AirTemp", "TrackTemp",
                    "Humidity", "consistency_score", "tire_degradation_rate",
                    "Sector1Time", "Sector2Time", "Sector3Time"]

    available = [c for c in numeric_cols if c in df.columns]
    if len(available) < 2:
        return {}

    corr_matrix = df[available].corr(method="pearson")
    result = corr_matrix.to_dict()

    # Isı haritası
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            mask=mask,
            ax=ax,
            linewidths=0.5,
            linecolor="#333333",
            annot_kws={"size": 8},
        )
        ax.set_title(f"{race_name} - Korelasyon Matrisi", fontsize=13, fontweight="bold", color="white")
        ax.set_facecolor("#1a1a1a")
        fig.patch.set_facecolor("#0d0d0d")
        ax.tick_params(colors="white", labelsize=8)

        save_path = FIGURES_DIR / f"{race_name.lower().replace(' ', '_')}_correlation.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Korelasyon ısı haritası üretilemedi: {e}")

    # LapTime ile en yüksek korelasyonlar
    if "LapTime" in corr_matrix.columns:
        lap_corr = corr_matrix["LapTime"].drop("LapTime").abs().sort_values(ascending=False)
        result["lap_time_correlations"] = lap_corr.to_dict()

    return result


# ─────────────────────────────────────────────────────────────
# KONSOL ÖZET
# ─────────────────────────────────────────────────────────────

def _print_eda_summary(results: dict, race_name: str):
    """EDA sonuçlarını terminale yazdır."""
    sep = "=" * 55
    basic = results.get("basic_summary", {})
    missing = results.get("missing_values", {})
    lt = results.get("lap_time_dist", {})

    print(f"\n{sep}")
    print(f"  📊 EDA ÖZET - {race_name} {basic.get('race_name', '')}")
    print(sep)
    print(f"  Toplam satır    : {basic.get('total_rows', 'N/A')}")
    print(f"  Pilot sayısı    : {basic.get('drivers', 'N/A')}")
    print(f"  Pit turları     : {basic.get('pit_laps', 'N/A')}")
    print(f"  SC turları      : {basic.get('sc_laps', 'N/A')}")
    print(f"  Outlier tur     : {basic.get('outliers', 'N/A')}")

    if lt:
        print(f"\n  Tur Zamanı (Pit/SC hariç):")
        print(f"    Ortalama      : {lt.get('mean', 0):.3f}s")
        print(f"    En hızlı      : {lt.get('min', 0):.3f}s")
        print(f"    Std sapma     : {lt.get('std', 0):.3f}s")

    if missing:
        print(f"\n  ⚠️  Eksik Değerler:")
        for col, info in list(missing.items())[:5]:
            print(f"    {col}: {info['count']} ({info['percent']}%)")
    else:
        print(f"\n  ✅ Eksik değer tespit edilmedi.")

    print(f"{sep}\n")


if __name__ == "__main__":
    from src.utils import generate_sample_lap_data
    from src.preprocessing import preprocess_laps
    from src.feature_engineering import engineer_features

    sample   = generate_sample_lap_data("Test")
    clean    = preprocess_laps(sample, "Test")
    featured = engineer_features(clean, "Test")
    results  = run_full_eda(featured, "Test")
    print("EDA tamamlandı. Anahtarlar:", list(results.keys()))
