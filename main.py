"""
main.py
-------
F1 Race Intelligence System — Ana Pipeline Orchestrator
Versiyon 2: Cache + Insight Engine + PDF + Gerçekçi Simulated Data
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ensure_directories, DEFAULT_RACES, ALL_2025_RACES, SEASON,
    PROCESSED_DATA_DIR, DATABASE_PATH,
)
from src.logger import get_logger, log_success
from src.database import initialize_database, get_all_processed_races
from src.data_loader import load_or_fetch_race_data, is_race_processed
from src.eda import run_full_eda
from src.analysis import run_all_analyses
from src.models import run_all_models
from src.visualization import generate_all_charts
from src.report_generator import generate_race_report
from src.insight_engine import generate_race_insights
from src.cache_manager import load_analysis_cache, save_analysis_cache
from src.pdf_report import generate_pdf_report

logger = get_logger("Main")
SEP = "═" * 62


def banner():
    print(f"""
{SEP}
  🏎️  F1 RACE INTELLIGENCE & STRATEGY ANALYSIS SYSTEM v2
  📅  Formula 1 — {SEASON} Sezonu
  🔧  Veri Madenciliği / Veri Analizi Projesi
  ⚠️   Gerçek veri için: pip install fastf1  (internet gerekli)
{SEP}
""")


def step(n, title):
    print(f"\n[ADIM {n}] {title}")
    print("─" * 50)


def process_single_race(race_name: str, force: bool = False) -> bool:
    """Tek yarış için tam pipeline. Lazy loading + cache destekli."""
    print(f"\n  📍 {race_name} GP işleniyor...")

    if not force and is_race_processed(race_name):
        # Cache'den analiz var mı?
        cached = load_analysis_cache(race_name)
        if cached:
            print(f"  ✅ {race_name}: Cache'den yüklendi. (Analiz atlanıyor)")
            return True
        print(f"  ✅ {race_name}: İşlenmiş veri mevcut, analiz yenileniyor...")

    start = time.time()

    # 1. Veri yükle
    print(f"  📥 Veri yükleniyor...")
    df = load_or_fetch_race_data(race_name, use_sample_on_failure=True)
    if df is None or df.empty:
        logger.error(f"{race_name}: Veri yüklenemedi.")
        return False

    is_sim = "IsSimulated" in df.columns and df["IsSimulated"].any()
    data_src = "[SIMULATED]" if is_sim else "[REAL FastF1]"
    print(f"  {data_src} {len(df)} satır, {df['Driver'].nunique()} pilot")

    # 2. EDA
    try:
        run_full_eda(df, race_name)
    except Exception as e:
        logger.warning(f"EDA hatası: {e}")

    # 3. Analizler
    print(f"  🔍 Analizler yapılıyor...")
    try:
        analysis = run_all_analyses(df, race_name)
    except Exception as e:
        logger.warning(f"Analiz hatası: {e}")
        analysis = {}

    # 4. ML
    print(f"  🤖 ML modelleri eğitiliyor...")
    try:
        ml = run_all_models(df, race_name)
        dt  = ml.get("decision_tree", {})
        knn = ml.get("knn", {})
        if not dt.get("error"):
            print(f"  ✅ Decision Tree: %{dt.get('accuracy',0)*100:.1f} | "
                  f"kNN: %{knn.get('accuracy',0)*100:.1f}")
    except Exception as e:
        logger.warning(f"ML hatası: {e}")
        ml = {}

    # 5. Insight Engine
    print(f"  💡 Insight engine çalışıyor...")
    try:
        insights = generate_race_insights(race_name, df, analysis, ml)
        print(f"  ✅ {len(insights.get('top5_summary', []))} kritik insight üretildi")
    except Exception as e:
        logger.warning(f"Insight hatası: {e}")
        insights = {}

    # 6. Grafikler
    print(f"  📈 Grafikler üretiliyor...")
    try:
        charts = generate_all_charts(df, analysis, ml, race_name)
        print(f"  ✅ {len(charts)} grafik üretildi")
    except Exception as e:
        logger.warning(f"Grafik hatası: {e}")
        charts = {}

    # 7. Cache kaydet
    try:
        save_analysis_cache(race_name, analysis, ml, charts)
    except Exception as e:
        logger.warning(f"Cache kaydedilemedi: {e}")

    # 8. MD + TXT rapor
    try:
        paths = generate_race_report(race_name, analysis, ml)
        for fmt, p in paths.items():
            print(f"  ✅ Rapor ({fmt}): {p.name}")
    except Exception as e:
        logger.warning(f"Rapor hatası: {e}")

    # 9. PDF rapor
    try:
        pdf_path = generate_pdf_report(race_name, analysis, ml, insights, charts)
        if pdf_path:
            print(f"  ✅ PDF raporu: {pdf_path.name}")
    except Exception as e:
        logger.warning(f"PDF hatası: {e}")

    # 10. Akademik özet rapor
    _print_academic_summary(race_name, ml)

    elapsed = time.time() - start
    log_success(f"Dashboard data is ready: {race_name} ({elapsed:.1f}s)")
    return True


def _print_academic_summary(race_name: str, ml: dict) -> None:
    """Akademik değerlendirme sonuçlarını terminale özetle."""
    ev = ml.get("evaluation", {})
    if not ev or ev.get("error"):
        return

    cv = ev.get("cross_validation", {})
    gs = ev.get("grid_search", {})
    roc = ev.get("roc_auc", {})
    mc = ml.get("statistical_tests", {}).get("mcnemar", {})

    print(f"\n  📐 AKADEMİK ÖZET — {race_name}")
    print(f"  {'─'*48}")

    dt_cv = cv.get("decision_tree", {})
    knn_cv = cv.get("knn", {})
    if dt_cv:
        print(f"  CV ({cv.get('n_folds',5)}-Fold) │ DT: {dt_cv.get('mean_accuracy',0):.3f}±{dt_cv.get('std_accuracy',0):.3f}"
              f" │ kNN: {knn_cv.get('mean_accuracy',0):.3f}±{knn_cv.get('std_accuracy',0):.3f}")

    dt_gs = gs.get("decision_tree", {})
    knn_gs = gs.get("knn", {})
    if dt_gs:
        print(f"  GridSearch   │ DT best: {dt_gs.get('best_score',0):.3f}"
              f" │ kNN best: {knn_gs.get('best_score',0):.3f}")

    dt_roc = roc.get("Decision Tree", {})
    knn_roc = roc.get("kNN", {})
    if dt_roc:
        print(f"  ROC AUC      │ DT: {dt_roc.get('auc',0):.3f}"
              f" │ kNN: {knn_roc.get('auc',0):.3f}")

    if mc:
        sig = "Anlamlı ✓" if mc.get("significant") else "Anlamlı değil"
        print(f"  McNemar      │ p={mc.get('p_value',1):.4f} → {sig}")

    comp = ev.get("model_comparison", {})
    if comp.get("best_model"):
        print(f"  En iyi model │ {comp['best_model']}")
    print(f"  {'─'*48}")


def main():
    banner()

    step(1, "Klasörler ve veritabanı hazırlanıyor...")
    ensure_directories()
    Path("data/analysis_cache").mkdir(parents=True, exist_ok=True)
    initialize_database()
    print("  ✅ Tüm dizinler hazır")
    print(f"  ✅ SQLite: {DATABASE_PATH}")

    step(2, "Mevcut durum kontrol ediliyor...")
    processed = get_all_processed_races()
    missing   = [r for r in DEFAULT_RACES if r not in processed]
    print(f"  İşlenmiş : {processed or 'Yok'}")
    print(f"  Varsayılan: {DEFAULT_RACES}")
    print(f"  Eksik     : {missing or 'Yok'}")

    step(3, f"Varsayılan {len(DEFAULT_RACES)} yarış işleniyor (Lazy Loading + Cache)...")
    ok_count = fail_count = 0
    for race in DEFAULT_RACES:
        ok = process_single_race(race)
        if ok: ok_count += 1
        else:  fail_count += 1

    # Özet
    print(f"\n{SEP}")
    print(f"  📊 PIPELINE TAMAMLANDI")
    print(f"{SEP}")
    print(f"  ✅ Başarılı  : {ok_count}/{len(DEFAULT_RACES)}")
    if fail_count: print(f"  ❌ Başarısız : {fail_count}")
    print(f"  🗄️  Database  : {DATABASE_PATH}")
    print(f"  💾 Processed : {PROCESSED_DATA_DIR}")
    print(f"  📄 Raporlar  : reports/generated_reports/")
    print(f"\n  Hazır yarışlar: {get_all_processed_races()}")
    print(f"\n{SEP}")
    print("  🚀 Dashboard'ları başlatmak için:\n")
    print("     Flask   : python dashboard/app.py")
    print("     Streamlit: streamlit run dashboard/streamlit_app.py")
    print(f"\n  Tarayıcıda: http://localhost:5050 (Flask)")
    print(f"              http://localhost:8501 (Streamlit)")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
