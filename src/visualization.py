"""
src/visualization.py
--------------------
F1 Race Intelligence System - Görselleştirme Modülü
Matplotlib + Seaborn ile grafik üretimi.
Tüm grafikler hem dosyaya kaydedilir hem de base64 string olarak döndürülür (dashboard için).
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from config import FIGURES_DIR, COMPOUND_COLORS, DASHBOARD_COLORS
from src.logger import get_logger

logger = get_logger("Visualization")

# ── Genel tema ayarları ───────────────────────────────────────
BG      = "#0D0D0D"
SURFACE = "#1A1A1A"
PRIMARY = "#E8002D"
ACCENT  = "#FF6B6B"
TEXT    = "#FFFFFF"
SUBTEXT = "#AAAAAA"
GRID    = "#2A2A2A"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  GRID,
    "legend.labelcolor": TEXT,
    "font.family":       "monospace",
})


# ─────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────

def _fig_to_base64(fig) -> str:
    """Matplotlib figure'ı base64 PNG string'e çevir (HTML için)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _save_fig(fig, filename: str) -> Path:
    """Figürü FIGURES_DIR'e kaydet."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path


def _apply_theme(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────
# 1. PİLOT ORTALAMA TUR SÜRESİ
# ─────────────────────────────────────────────────────────────

def plot_driver_avg_laptime(driver_data: List[dict], race_name: str) -> str:
    """Bar chart: Pilot bazlı ortalama tur süresi."""
    if not driver_data:
        return ""
    df = pd.DataFrame(driver_data).sort_values("avg_lap_time")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PRIMARY if i == 0 else "#444444" for i in range(len(df))]
    bars = ax.barh(df["Driver"], df["avg_lap_time"], color=colors, height=0.6)
    ax.bar_label(bars, fmt="%.2fs", padding=3, color=TEXT, fontsize=8)
    _apply_theme(ax, f"{race_name} — Ortalama Tur Süresi", "Saniye", "Pilot")
    ax.invert_yaxis()
    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    return b64


# ─────────────────────────────────────────────────────────────
# 2. CONSISTENCY SCORE
# ─────────────────────────────────────────────────────────────

def plot_consistency_score(driver_data: List[dict], race_name: str) -> str:
    """Bar chart: Pilot istikrar skoru (yüksek = iyi)."""
    if not driver_data:
        return ""
    df = pd.DataFrame(driver_data).sort_values("consistency_score", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PRIMARY if i == 0 else "#3A5A8A" for i in range(len(df))]
    bars = ax.bar(df["Driver"], df["consistency_score"], color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, color=TEXT, fontsize=8)
    _apply_theme(ax, f"{race_name} — Consistency Score (Yüksek = İstikrarlı)", "Pilot", "Score")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# 3. LASTİK TİPİ BAZLI ORTALAMA TUR SÜRESİ
# ─────────────────────────────────────────────────────────────

def plot_compound_avg_laptime(compound_stats: List[dict], race_name: str) -> str:
    """Bar chart: Compound bazlı ortalama tur süresi."""
    if not compound_stats:
        return ""
    df = pd.DataFrame(compound_stats).sort_values("avg_lap_time")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [COMPOUND_COLORS.get(c, "#888888") for c in df["Compound"]]
    bars = ax.bar(df["Compound"], df["avg_lap_time"], color=colors, width=0.5,
                  edgecolor=GRID, linewidth=0.5)
    ax.bar_label(bars, fmt="%.2fs", padding=3, color=TEXT, fontsize=9)
    _apply_theme(ax, f"{race_name} — Lastik Tipi Ortalama Tur Süresi", "Lastik", "Saniye")
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# 4. LASTİK BOZULMA LINE CHART
# ─────────────────────────────────────────────────────────────

def plot_tire_degradation(tyre_age_data: Dict[str, List[dict]], race_name: str) -> str:
    """Line chart: Lastik yaşı vs ortalama tur zamanı (her compound için ayrı çizgi)."""
    if not tyre_age_data:
        return ""
    fig, ax = plt.subplots(figsize=(10, 5))
    for compound, records in tyre_age_data.items():
        if not records:
            continue
        age_df = pd.DataFrame(records).sort_values("TyreLife")
        # Key adı 'mean' veya 'LapTime' olabilir
        y_col = "mean" if "mean" in age_df.columns else ("LapTime" if "LapTime" in age_df.columns else None)
        if y_col is None:
            continue
        color = COMPOUND_COLORS.get(compound, "#888888")
        ax.plot(age_df["TyreLife"], age_df[y_col],
                color=color, linewidth=2.5, label=compound, marker="o", markersize=4)
        if "std" in age_df.columns:
            ax.fill_between(age_df["TyreLife"],
                            age_df[y_col] - age_df["std"],
                            age_df[y_col] + age_df["std"],
                            alpha=0.15, color=color)
    _apply_theme(ax, f"{race_name} — Lastik Bozulma Eğrisi", "Lastik Yaşı (Tur)", "Ort. Tur Süresi (s)")
    ax.legend()
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# 5. TYRELİFE vs LAP TIME SCATTER
# ─────────────────────────────────────────────────────────────

def plot_tyre_life_scatter(df: pd.DataFrame, race_name: str) -> str:
    """Scatter: Lastik yaşı vs tur zamanı, lastik tipine göre renklendirilmiş."""
    if df.empty or "TyreLife" not in df.columns or "LapTime" not in df.columns:
        return ""
    clean = df.copy()
    for col in ["IsPitLap", "IsOutlier", "IsSCLap"]:
        if col in clean.columns:
            clean = clean[~clean[col]]

    fig, ax = plt.subplots(figsize=(10, 5))
    if "Compound" in clean.columns:
        for compound in clean["Compound"].unique():
            cdf = clean[clean["Compound"] == compound]
            color = COMPOUND_COLORS.get(compound, "#888888")
            ax.scatter(cdf["TyreLife"], cdf["LapTime"],
                       color=color, alpha=0.5, s=20, label=compound)
        ax.legend(title="Lastik")
    else:
        ax.scatter(clean["TyreLife"], clean["LapTime"], color=PRIMARY, alpha=0.4, s=20)
    _apply_theme(ax, f"{race_name} — Lastik Yaşı vs Tur Zamanı",
                 "Lastik Yaşı (Tur)", "Tur Zamanı (s)")
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# 6. DRIVER COMPARISON
# ─────────────────────────────────────────────────────────────

def plot_driver_comparison(comparison: dict, race_name: str) -> str:
    """Radar/bar chart: İki pilot karşılaştırması."""
    d1_name = comparison.get("driver1", "D1")
    d2_name = comparison.get("driver2", "D2")
    d1 = comparison.get("d1", {})
    d2 = comparison.get("d2", {})

    metrics = {
        "Ort. Tur (s)":        ("avg_lap_time",    True),   # True = küçük iyi
        "En Hızlı Tur (s)":   ("best_lap_time",   True),
        "Std Sapma (s)":       ("lap_time_std",    True),
        "Consistency":         ("consistency",     False),  # False = büyük iyi
        "Pit Etkisi (s)":      ("pit_impact",      False),
    }

    labels, v1, v2 = [], [], []
    for label, (key, lower_better) in metrics.items():
        val1 = d1.get(key)
        val2 = d2.get(key)
        if val1 is not None and val2 is not None:
            labels.append(label)
            v1.append(float(val1))
            v2.append(float(val2))

    if not labels:
        return ""

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, v1, w, label=d1_name, color=PRIMARY, alpha=0.85)
    b2 = ax.bar(x + w/2, v2, w, label=d2_name, color="#3A8AFF", alpha=0.85)
    ax.bar_label(b1, fmt="%.2f", padding=3, color=TEXT, fontsize=7)
    ax.bar_label(b2, fmt="%.2f", padding=3, color=TEXT, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    _apply_theme(ax, f"{race_name} — {d1_name} vs {d2_name}", "", "Değer")
    ax.legend()
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# 7. PİT ÖNCESİ / SONRASI PACE
# ─────────────────────────────────────────────────────────────

def plot_pit_impact(driver_analysis: List[dict], race_name: str) -> str:
    """Bar chart: Pit öncesi vs sonrası ortalama pace."""
    if not driver_analysis:
        return ""
    rows = []
    for d in driver_analysis:
        for detail in d.get("details", []):
            rows.append({
                "Driver":  d["Driver"],
                "before":  detail.get("before_avg", 0),
                "after":   detail.get("after_avg", 0),
                "impact":  detail.get("impact", 0),
            })
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    avg = df.groupby("Driver")[["before", "after", "impact"]].mean().reset_index()
    avg = avg.sort_values("impact", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(avg))
    w = 0.35
    ax.bar(x - w/2, avg["before"], w, label="Pit Öncesi", color="#888888", alpha=0.8)
    ax.bar(x + w/2, avg["after"],  w, label="Pit Sonrası", color=PRIMARY,  alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(avg["Driver"], rotation=30, ha="right")
    _apply_theme(ax, f"{race_name} — Pit Stop Etkisi (Öncesi / Sonrası)",
                 "Pilot", "Ortalama Tur Süresi (s)")
    ax.legend()
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# 8. SICAKLIK vs LAP TIME
# ─────────────────────────────────────────────────────────────

def plot_temp_vs_laptime(df: pd.DataFrame, race_name: str) -> str:
    """Scatter: AirTemp ve TrackTemp vs LapTime."""
    if df.empty or "LapTime" not in df.columns:
        return ""
    clean = df.copy()
    for col in ["IsPitLap", "IsOutlier"]:
        if col in clean.columns:
            clean = clean[~clean[col]]

    cols = [c for c in ["AirTemp", "TrackTemp"] if c in clean.columns]
    if not cols:
        return ""

    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]

    labels_map = {"AirTemp": "Hava Sıcaklığı (°C)", "TrackTemp": "Pist Sıcaklığı (°C)"}
    for ax, col in zip(axes, cols):
        valid = clean[[col, "LapTime"]].dropna()
        ax.scatter(valid[col], valid["LapTime"], color=PRIMARY, alpha=0.35, s=15)
        # Trend çizgisi
        if len(valid) > 5:
            z = np.polyfit(valid[col], valid["LapTime"], 1)
            p = np.poly1d(z)
            xs = np.linspace(valid[col].min(), valid[col].max(), 100)
            ax.plot(xs, p(xs), color=ACCENT, linewidth=1.5, linestyle="--", label="Trend")
            ax.legend()
        _apply_theme(ax, f"{labels_map[col]} vs Tur Zamanı",
                     labels_map[col], "Tur Zamanı (s)")

    fig.suptitle(f"{race_name} — Sıcaklık Etkisi",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# 9. KORELASYON HEATMAP
# ─────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, race_name: str) -> str:
    """Seaborn heatmap: Sayısal değişkenler arası korelasyon."""
    numeric_cols = ["LapTime", "TyreLife", "AirTemp", "TrackTemp",
                    "Sector1Time", "Sector2Time", "Sector3Time",
                    "consistency_score", "tire_degradation_rate"]
    avail = [c for c in numeric_cols if c in df.columns]
    if len(avail) < 3:
        return ""

    corr = df[avail].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                mask=mask, ax=ax, linewidths=0.5, linecolor=GRID,
                annot_kws={"size": 8}, vmin=-1, vmax=1)
    _apply_theme(ax, f"{race_name} — Korelasyon Matrisi")
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# 10. K-MEANS CLUSTER
# ─────────────────────────────────────────────────────────────

def plot_kmeans_clusters(kmeans_result: dict, race_name: str) -> str:
    """Scatter: K-Means küme görselleştirmesi (ilk 2 boyut PCA ile)."""
    X_scaled = kmeans_result.get("X_scaled")
    labels   = kmeans_result.get("labels")
    label_map = kmeans_result.get("label_map", {})
    driver_clusters = kmeans_result.get("driver_clusters", [])

    if not X_scaled or not labels:
        return ""

    X = np.array(X_scaled)
    y = np.array(labels)

    # PCA ile 2D'ye indir (sklearn'e bağımlı)
    try:
        from sklearn.decomposition import PCA
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X2d = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_
            axis_labels = (
                f"PC1 ({explained[0]*100:.1f}%)",
                f"PC2 ({explained[1]*100:.1f}%)",
            )
        else:
            X2d = X
            axis_labels = ("Özellik 1", "Özellik 2")
    except Exception:
        X2d = X[:, :2] if X.shape[1] >= 2 else np.column_stack([X[:, 0], np.zeros(len(X))])
        axis_labels = ("Boyut 1", "Boyut 2")

    cluster_colors = [PRIMARY, "#3A8AFF", "#39B54A", "#FFF200", "#FF6B6B"]
    n_clusters = len(set(y))

    fig, ax = plt.subplots(figsize=(9, 6))
    for c_id in range(n_clusters):
        mask = y == c_id
        color = cluster_colors[c_id % len(cluster_colors)]
        c_label = label_map.get(c_id, f"Küme {c_id}")
        ax.scatter(X2d[mask, 0], X2d[mask, 1], c=color, s=100,
                   label=c_label, alpha=0.85, edgecolors="white", linewidth=0.5)

    # Pilot isimlerini ekle
    driver_names = [d["Driver"] for d in driver_clusters]
    for i, name in enumerate(driver_names):
        if i < len(X2d):
            ax.annotate(name, (X2d[i, 0], X2d[i, 1]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=SUBTEXT)

    _apply_theme(ax, f"{race_name} — K-Means Performans Kümeleri",
                 axis_labels[0], axis_labels[1])
    ax.legend(title="Küme")
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# 11. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

def plot_feature_importance(feat_imp: List[Tuple[str, float]], race_name: str) -> str:
    """Horizontal bar: Decision Tree feature importance."""
    if not feat_imp:
        return ""
    labels = [f[0] for f in feat_imp]
    values = [f[1] for f in feat_imp]
    colors = [PRIMARY if i == 0 else "#444444" for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, color=TEXT, fontsize=8)
    _apply_theme(ax, f"{race_name} — Feature Importance (Decision Tree)",
                 "Önem Skoru", "Özellik")
    ax.set_xlim(0, max(values) * 1.2)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# 12. MODEL KARŞILAŞTIRMA
# ─────────────────────────────────────────────────────────────

def plot_model_comparison(comparison: dict, race_name: str) -> str:
    """Bar chart: DT vs kNN accuracy karşılaştırması."""
    if not comparison:
        return ""
    models = ["Decision Tree", "kNN"]
    accs = [
        comparison.get("decision_tree_accuracy", 0) * 100,
        comparison.get("knn_accuracy", 0) * 100,
    ]
    colors = [PRIMARY if accs[0] >= accs[1] else "#444444",
              "#3A8AFF" if accs[1] > accs[0] else "#444444"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, accs, color=colors, width=0.4)
    ax.bar_label(bars, fmt="%.1f%%", padding=5, color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    _apply_theme(ax, f"{race_name} — Model Karşılaştırması", "Model", "Doğruluk (%)")
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# 13. ELBOW CURVE
# ─────────────────────────────────────────────────────────────

def plot_elbow_curve(elbow_data: list, race_name: str) -> str:
    """Elbow method görselleştirmesi."""
    if not elbow_data:
        return ""
    ks = [d[0] for d in elbow_data]
    inertias = [d[1] for d in elbow_data]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, inertias, color=PRIMARY, marker="o", linewidth=2, markersize=7)
    ax.fill_between(ks, inertias, alpha=0.1, color=PRIMARY)
    _apply_theme(ax, f"{race_name} — Elbow Method (K-Means)",
                 "Küme Sayısı (k)", "Inertia")
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────
# ANA: TÜM GRAFİKLERİ ÜRET
# ─────────────────────────────────────────────────────────────

def generate_all_charts(df: pd.DataFrame,
                         analysis_results: dict,
                         ml_results: dict,
                         race_name: str) -> Dict[str, str]:
    """
    Tüm grafikleri üretir ve base64 sözlüğü döndürür.
    Dashboard bu sözlüğü kullanır.
    """
    charts = {}
    logger.info(f"Grafik üretimi başlıyor: {race_name}")

    try:
        # Pilot performansı
        stab = analysis_results.get("driver_stability", {})
        if stab.get("data"):
            charts["driver_avg_laptime"] = plot_driver_avg_laptime(stab["data"], race_name)
            charts["consistency_score"]  = plot_consistency_score(stab["data"], race_name)

        # Lastik stratejisi
        tire = analysis_results.get("tire_strategy", {})
        if tire.get("compound_stats"):
            charts["compound_avg_laptime"] = plot_compound_avg_laptime(tire["compound_stats"], race_name)
        if tire.get("tyre_age_data"):
            charts["tire_degradation"]     = plot_tire_degradation(tire["tyre_age_data"], race_name)

        # Lastik scatter
        charts["tyre_life_scatter"] = plot_tyre_life_scatter(df, race_name)

        # Pit stop
        pit = analysis_results.get("pit_stop", {})
        if pit.get("driver_analysis"):
            charts["pit_impact"] = plot_pit_impact(pit["driver_analysis"], race_name)

        # Sıcaklık
        charts["temp_vs_laptime"]     = plot_temp_vs_laptime(df, race_name)
        charts["correlation_heatmap"] = plot_correlation_heatmap(df, race_name)

        # ML
        kmeans = ml_results.get("kmeans", {})
        if kmeans and not kmeans.get("error"):
            charts["kmeans_clusters"] = plot_kmeans_clusters(kmeans, race_name)
            charts["elbow_curve"]     = plot_elbow_curve(kmeans.get("elbow_inertias", []), race_name)

        dt = ml_results.get("decision_tree", {})
        if dt and not dt.get("error"):
            charts["feature_importance"] = plot_feature_importance(dt.get("feature_importance", []), race_name)

        comp = ml_results.get("comparison", {})
        if comp and not comp.get("error"):
            charts["model_comparison"] = plot_model_comparison(comp, race_name)

    except Exception as e:
        logger.error(f"Grafik üretim hatası: {e}")

    # Boş stringleri filtrele
    charts = {k: v for k, v in charts.items() if v}
    logger.info(f"Üretilen grafik sayısı: {len(charts)}")
    return charts
