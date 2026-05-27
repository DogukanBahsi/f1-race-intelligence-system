"""
config.py
---------
F1 Race Intelligence System - Merkezi Konfigürasyon Dosyası
Tüm sabitler, path'ler ve model ayarları buradan yönetilir.
Kod içinde sabit string path kullanılmaz.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PROJE KÖKÜ
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────
# SEZON
# ─────────────────────────────────────────────────────────────
SEASON = 2025
SESSION_TYPE = "R"  # R = Race

# ─────────────────────────────────────────────────────────────
# 2025 SEZONU TÜM YARIŞLAR
# ─────────────────────────────────────────────────────────────
ALL_2025_RACES = [
    "Australia",
    "China",
    "Japan",
    "Bahrain",
    "Saudi Arabia",
    "Miami",
    "Emilia-Romagna",
    "Monaco",
    "Spain",
    "Canada",
    "Austria",
    "Great Britain",
    "Belgium",
    "Hungary",
    "Netherlands",
    "Italy",
    "Azerbaijan",
    "Singapore",
    "United States",
    "Mexico",
    "Brazil",
    "Las Vegas",
    "Qatar",
    "Abu Dhabi",
]

# ─────────────────────────────────────────────────────────────
# VARSAYILAN 5 YARIŞ (İlk açılışta işlenecek)
# ─────────────────────────────────────────────────────────────
DEFAULT_RACES = [
    "Bahrain",
    "Monaco",
    "Great Britain",
    "Belgium",
    "Italy",
]

# ─────────────────────────────────────────────────────────────
# PATH TANIMLARI
# ─────────────────────────────────────────────────────────────
CACHE_DIR          = BASE_DIR / "cache"
DATA_DIR           = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORTS_DIR        = DATA_DIR / "exports"
REPORTS_DIR        = BASE_DIR / "reports"
FIGURES_DIR        = REPORTS_DIR / "figures"
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated_reports"
NOTEBOOKS_DIR      = BASE_DIR / "notebooks"
DASHBOARD_DIR      = BASE_DIR / "dashboard"

# SQLite veritabanı yolu
DATABASE_PATH = DATA_DIR / "f1_intelligence.db"

# ─────────────────────────────────────────────────────────────
# LOGGING AYARLARI
# ─────────────────────────────────────────────────────────────
LOG_FILE  = BASE_DIR / "f1_system.log"
LOG_LEVEL = "INFO"  # DEBUG | INFO | WARNING | ERROR

# ─────────────────────────────────────────────────────────────
# MODEL AYARLARI
# ─────────────────────────────────────────────────────────────
KMEANS_CONFIG = {
    "n_clusters": 3,
    "random_state": 42,
    "max_iter": 300,
    "n_init": 10,
}

DECISION_TREE_CONFIG = {
    "max_depth": 5,
    "random_state": 42,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
}

KNN_CONFIG = {
    "n_neighbors": 5,
    "metric": "euclidean",
}

# K-Means için test edilecek küme sayısı aralığı (Elbow method)
KMEANS_K_RANGE = range(2, 8)

# ─────────────────────────────────────────────────────────────
# DASHBOARD VARSAYILAN AYARLAR
# ─────────────────────────────────────────────────────────────
DASHBOARD_CONFIG = {
    "default_race": "Bahrain",
    "page_title": "F1 Race Intelligence System",
    "page_icon": "🏎️",
    "layout": "wide",
    "theme": "dark",
}

# ─────────────────────────────────────────────────────────────
# VERİ İŞLEME AYARLARI
# ─────────────────────────────────────────────────────────────
PREPROCESSING_CONFIG = {
    # Aykırı tur zamanı filtreleme: ortalamanın kaç std sapmasının dışı aykırı?
    "outlier_std_threshold": 3.0,
    # Pit stop turunu etiketlemek için maksimum tur zamanı katı
    "pit_lap_time_multiplier": 1.5,
    # Minimum stint uzunluğu (analiz için)
    "min_stint_laps": 3,
}

# Lastik bileşikleri renk paleti (Plotly uyumlu)
COMPOUND_COLORS = {
    "SOFT":       "#E8002D",   # F1 kırmızı
    "MEDIUM":     "#FFF200",   # F1 sarı
    "HARD":       "#EBEBEB",   # Beyaz/gri
    "INTERMEDIATE": "#39B54A", # Yeşil
    "WET":        "#0067FF",   # Mavi
    "UNKNOWN":    "#888888",
}

# Dashboard renk paleti
DASHBOARD_COLORS = {
    "background":  "#0D0D0D",
    "surface":     "#1A1A1A",
    "primary":     "#E8002D",
    "accent":      "#FF6B6B",
    "text":        "#FFFFFF",
    "subtext":     "#AAAAAA",
    "grid":        "#2A2A2A",
}

# ─────────────────────────────────────────────────────────────
# FastF1 RETRY AYARLARI
# ─────────────────────────────────────────────────────────────
FASTF1_RETRY_CONFIG = {
    "max_retries": 3,
    "retry_delay": 5,  # saniye
}

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────
# CROSS-VALIDATION AYARLARI
# ─────────────────────────────────────────────────────────────
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": RANDOM_STATE,
}

# ─────────────────────────────────────────────────────────────
# GRIDSEARCHCV PARAMETRELERİ
# ─────────────────────────────────────────────────────────────
GRIDSEARCH_DT_PARAMS = {
    "max_depth": [3, 4, 5, 7, None],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3],
    "criterion": ["gini", "entropy"],
}

GRIDSEARCH_KNN_PARAMS = {
    "n_neighbors": [3, 5, 7, 9],
    "metric": ["euclidean", "manhattan"],
    "weights": ["uniform", "distance"],
}

KMEANS_GRID_PARAMS = {
    "n_clusters": [2, 3, 4, 5],
    "init": ["k-means++", "random"],
}

# ─────────────────────────────────────────────────────────────
# AKADEMİK ÇIKTI DİZİNLERİ
# ─────────────────────────────────────────────────────────────
FIGURES_XAI_DIR   = FIGURES_DIR / "xai"
FIGURES_EVAL_DIR  = FIGURES_DIR / "evaluation"
TABLES_DIR        = REPORTS_DIR / "tables"
EXPERIMENTS_DIR   = BASE_DIR / "experiments"


def ensure_directories():
    """Tüm gerekli klasörlerin var olduğundan emin ol."""
    dirs = [
        CACHE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        EXPORTS_DIR, FIGURES_DIR, GENERATED_REPORTS_DIR,
        NOTEBOOKS_DIR, DASHBOARD_DIR,
        FIGURES_XAI_DIR, FIGURES_EVAL_DIR,
        TABLES_DIR, EXPERIMENTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("✅ Tüm klasörler oluşturuldu.")
    print(f"   Proje kökü : {BASE_DIR}")
    print(f"   Database   : {DATABASE_PATH}")
    print(f"   Season     : {SEASON}")
    print(f"   Races      : {len(ALL_2025_RACES)} yarış tanımlı")
    print(f"   Defaults   : {DEFAULT_RACES}")
