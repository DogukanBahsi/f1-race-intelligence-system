"""
dashboard/streamlit_app.py
--------------------------
F1 Race Intelligence System — Streamlit Dashboard
Komut: streamlit run dashboard/streamlit_app.py
"""

import sys
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from config import ALL_2025_RACES, DEFAULT_RACES, SEASON, GENERATED_REPORTS_DIR
from src.data_loader import load_or_fetch_race_data, is_race_processed
from src.analysis import run_all_analyses, compare_drivers
from src.models import run_all_models
from src.visualization import (
    plot_driver_avg_laptime, plot_consistency_score,
    plot_compound_avg_laptime, plot_tire_degradation,
    plot_tyre_life_scatter, plot_driver_comparison,
    plot_pit_impact, plot_temp_vs_laptime,
    plot_correlation_heatmap, plot_kmeans_clusters,
    plot_feature_importance, plot_model_comparison,
    plot_elbow_curve, generate_all_charts,
)
from src.insight_engine import generate_race_insights
from src.cache_manager import load_analysis_cache, save_analysis_cache
from src.report_generator import generate_race_report
from src.pdf_report import generate_pdf_report
from src.database import get_all_processed_races
from src.utils import seconds_to_mmss

# ─────────────────────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Intelligence System",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# TEMA / CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Ana arka plan */
    .stApp { background-color: #0D0D0D; color: #FFFFFF; }
    .stSidebar { background-color: #111111; border-right: 1px solid #2A2A2A; }

    /* Başlık çubuğu */
    .f1-header {
        background: linear-gradient(90deg, #E8002D 0%, #1a1a1a 60%);
        padding: 16px 24px; border-radius: 6px; margin-bottom: 20px;
        border-left: 6px solid #E8002D;
    }
    .f1-title { font-size: 28px; font-weight: 900; color: #FFFFFF;
                letter-spacing: 3px; font-family: monospace; margin: 0; }
    .f1-sub   { font-size: 12px; color: #AAAAAA; letter-spacing: 4px;
                font-family: monospace; margin: 0; }

    /* Metrik kartlar */
    div[data-testid="metric-container"] {
        background: #1A1A1A; border: 1px solid #2A2A2A;
        border-radius: 6px; padding: 12px;
    }
    div[data-testid="metric-container"] label { color: #888 !important; font-size: 11px !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #E8002D !important; font-weight: 900 !important; font-size: 24px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #111; border-bottom: 2px solid #2A2A2A; }
    .stTabs [data-baseweb="tab"]      { color: #888; font-family: monospace; font-size: 12px; }
    .stTabs [aria-selected="true"]    { color: #E8002D !important; border-bottom: 2px solid #E8002D !important; }

    /* Selectbox */
    .stSelectbox label { color: #888; font-size: 11px; letter-spacing: 1px; }

    /* Simulated uyarı */
    .sim-warning {
        background: rgba(255,179,0,0.1); border: 1px solid rgba(255,179,0,0.4);
        border-left: 4px solid #FFB300; padding: 10px 14px; border-radius: 4px;
        font-size: 12px; color: #FFE082; font-family: monospace;
    }
    /* Insight card */
    .insight-card {
        background: #1A1A1A; border: 1px solid #2A2A2A; border-radius: 6px;
        padding: 14px; margin-bottom: 10px; border-left: 3px solid #E8002D;
    }
    .insight-title { color: #E8002D; font-weight: bold; font-size: 13px; }
    .insight-value { color: #FFFFFF; font-size: 18px; font-weight: 900; font-family: monospace; }
    .insight-text  { color: #AAAAAA; font-size: 12px; line-height: 1.6; }

    /* Table */
    .stDataFrame { background: #1A1A1A; }

    /* Button */
    .stButton button {
        background: #E8002D; color: white; border: none;
        font-family: monospace; font-weight: bold; letter-spacing: 1px;
    }
    .stButton button:hover { background: #FF4D6D; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# YARDIMCI: BASE64 GRAFİK GÖSTER
# ─────────────────────────────────────────────────────────────
def show_chart(b64: str, caption: str = ""):
    if b64 and b64.startswith("data:image"):
        st.image(b64, use_container_width=True, caption=caption)
    else:
        st.caption("📊 Grafik mevcut değil")


def show_insight_card(ins: dict):
    level_colors = {"positive": "#00C851", "warning": "#FFB300", "neutral": "#888", "info": "#2196F3"}
    color = level_colors.get(ins.get("level", "neutral"), "#888")
    st.markdown(f"""
    <div class="insight-card" style="border-left-color:{color}">
        <div class="insight-title">{ins.get('icon','')} {ins.get('title','')}</div>
        <div class="insight-value">{ins.get('driver','')}</div>
        <div style="color:{color};font-size:12px;font-family:monospace;margin:4px 0">{ins.get('value','')}</div>
        <div class="insight-text">{ins.get('text','')}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE & VERİ YÜKLEYİCİ
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def load_race_data(race_name: str):
    """Yarış verisini yükle ve tüm analizleri çalıştır. 5dk cache."""
    df = load_or_fetch_race_data(race_name, use_sample_on_failure=True)
    if df is None or df.empty:
        return None, {}, {}, {}, {}
    analysis = run_all_analyses(df, race_name)
    ml       = run_all_models(df, race_name)
    insights = generate_race_insights(race_name, df, analysis, ml)
    charts   = generate_all_charts(df, analysis, ml, race_name)
    # Cache kaydet
    try:
        save_analysis_cache(race_name, analysis, ml, charts)
    except Exception:
        pass
    return df, analysis, ml, insights, charts


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="f1-header">
    <div class="f1-title">🏎 F1 RACE INTELLIGENCE SYSTEM</div>
    <div class="f1-sub">STRATEGY ANALYSIS DASHBOARD — 2025 SEASON</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏁 YARIŞ SEÇİMİ")
    processed = get_all_processed_races()

    # İşlenmiş yarışları önce göster
    race_options = (
        [f"✓ {r}" for r in ALL_2025_RACES if r in processed] +
        [r for r in ALL_2025_RACES if r not in processed]
    )
    raw_races = [r.replace("✓ ", "") for r in race_options]

    selected_display = st.selectbox("Yarış", race_options, index=0)
    selected_race    = selected_display.replace("✓ ", "")

    st.divider()

    st.markdown("### 👤 PİLOT KARŞILAŞTIRMA")
    driver1 = st.selectbox("Pilot 1", ["—"] + ["VER","NOR","LEC","PIA","HAM","RUS","ALO","STR","GAS","TSU","LAW","ANT","BEA","DOO","HUL","SAI","ALB","BOR","HAD","OCO"], key="d1")
    driver2 = st.selectbox("Pilot 2", ["—"] + ["VER","NOR","LEC","PIA","HAM","RUS","ALO","STR","GAS","TSU","LAW","ANT","BEA","DOO","HUL","SAI","ALB","BOR","HAD","OCO"], key="d2", index=2)
    run_compare = st.button("🔄 KARŞILAŞTIR", use_container_width=True)

    st.divider()

    st.markdown("### 📊 SİSTEM")
    st.caption(f"Sezon: {SEASON}")
    st.caption(f"Toplam yarış: {len(ALL_2025_RACES)}")
    st.caption(f"İşlenmiş: {len(processed)}")
    st.caption("FastF1 + SQLite + Cache")

    st.divider()
    st.markdown("### ✓ HAZIR YARIŞLAR")
    for r in processed:
        st.caption(f"✅ {r}")


# ─────────────────────────────────────────────────────────────
# VERİ YÜKLEMESİ
# ─────────────────────────────────────────────────────────────
with st.spinner(f"⏳ {selected_race} GP yükleniyor..."):
    df, analysis, ml, insights, charts = load_race_data(selected_race)

if df is None:
    st.error(f"❌ {selected_race} verisi yüklenemedi.")
    st.stop()

is_sim = "IsSimulated" in df.columns and bool(df["IsSimulated"].any())
if is_sim:
    st.markdown("""
    <div class="sim-warning">
    ⚠️ <b>SİMÜLE VERİ</b> — Bu dashboard simüle edilmiş verilerle çalışmaktadır.
    Gerçek 2025 FastF1 verisi için: <code>pip install fastf1</code> (internet gerekli)
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

# ─────────────────────────────────────────────────────────────
# SEKMELER
# ─────────────────────────────────────────────────────────────
tab_overview, tab_driver, tab_compare, tab_tire, tab_pit, \
tab_weather, tab_ml, tab_cv, tab_roc, tab_xai, tab_error, \
tab_insight, tab_report, tab_pres = st.tabs([
    "🏠 Overview",
    "👤 Driver Perf.",
    "⚔️ Comparison",
    "🔴 Tire Strategy",
    "🔧 Pit Stop",
    "🌡️ Weather",
    "🤖 ML Models",
    "📊 CV & GridSearch",
    "📈 ROC / AUC",
    "🔍 XAI (SHAP)",
    "⚠️ Error Analysis",
    "💡 AI Insights",
    "📄 Report",
    "🎯 Sunum",
])


# ══════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════
with tab_overview:
    ov = analysis.get("race_overview", {})
    stab = analysis.get("driver_stability", {})

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("🏎️ YARIŞ", selected_race)
    with c2:
        st.metric("👤 PİLOT", ov.get("total_drivers", "—"))
    with c3:
        st.metric("🔄 TUR", ov.get("total_laps", "—"))
    with c4:
        fl = ov.get("fastest_lap")
        st.metric("⚡ EN HIZLI TUR", seconds_to_mmss(fl) if fl else "—")
    with c5:
        st.metric("🏆 EN HIZLI", ov.get("fastest_driver", "—"))
    with c6:
        st.metric("🎯 EN İSTİKRARLI", ov.get("most_stable_driver", "—"))

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Ortalama Tur Süresi")
        show_chart(charts.get("driver_avg_laptime"))
    with col_b:
        st.markdown("#### Consistency Score")
        show_chart(charts.get("consistency_score"))

    # Sıcaklık bilgisi
    if ov.get("avg_air_temp"):
        st.markdown(
            f"🌡️ Ort. Hava: **{ov['avg_air_temp']:.1f}°C** | "
            f"Pist: **{ov.get('avg_track_temp', 0):.1f}°C** | "
            f"Compound: {ov.get('compound_usage', {})}"
        )


# ══════════════════════════════════════════════════
# 2. DRIVER PERFORMANCE
# ══════════════════════════════════════════════════
with tab_driver:
    stab = analysis.get("driver_stability", {})
    data = stab.get("data", [])

    if data:
        st.markdown("#### 🏁 Pilot İstikrar Sıralaması")
        df_stab = pd.DataFrame(data)
        cols_show = [c for c in ["stability_rank","Driver","Team","avg_lap_time",
                                  "best_lap_time","lap_time_std","consistency_score"]
                     if c in df_stab.columns]
        st.dataframe(
            df_stab[cols_show].style.format({
                "avg_lap_time": "{:.3f}",
                "best_lap_time": "{:.3f}",
                "lap_time_std": "{:.3f}",
                "consistency_score": "{:.4f}",
            }).background_gradient(subset=["consistency_score"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )
        if stab.get("comment"):
            st.info(stab["comment"])
    else:
        st.warning("Pilot istikrar verisi mevcut değil.")


# ══════════════════════════════════════════════════
# 3. DRIVER COMPARISON
# ══════════════════════════════════════════════════
with tab_compare:
    st.markdown("#### ⚔️ Pilot Karşılaştırma")

    stab_data = analysis.get("driver_stability", {}).get("data", [])
    available_drivers = [d["Driver"] for d in stab_data] if stab_data else []

    if available_drivers:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            sel_d1 = st.selectbox("Pilot 1", available_drivers, key="cmp_d1")
        with col2:
            d2_opts = [d for d in available_drivers if d != sel_d1]
            sel_d2  = st.selectbox("Pilot 2", d2_opts, key="cmp_d2")
        with col3:
            st.write("")
            st.write("")
            do_compare = st.button("KARŞILAŞTIR", use_container_width=True)

        if do_compare or run_compare:
            d1_use = sel_d1
            d2_use = sel_d2
            if run_compare and driver1 != "—" and driver2 != "—":
                d1_use, d2_use = driver1, driver2

            comp = compare_drivers(df, d1_use, d2_use, selected_race)

            if comp.get("comment"):
                st.success(comp["comment"])

            # Metrikler
            d1d = comp.get("d1", {})
            d2d = comp.get("d2", {})
            st.markdown(f"##### {d1_use} vs {d2_use}")

            metrics_compare = {
                "Ort. Tur (s)": ("avg_lap_time", True),
                "En Hızlı (s)": ("best_lap_time", True),
                "Std Sapma":    ("lap_time_std", True),
                "Consistency":  ("consistency", False),
            }
            c_cols = st.columns(len(metrics_compare))
            for col, (label, (key, lower_better)) in zip(c_cols, metrics_compare.items()):
                v1 = d1d.get(key)
                v2 = d2d.get(key)
                if v1 is not None and v2 is not None:
                    delta = v1 - v2
                    better = delta < 0 if lower_better else delta > 0
                    with col:
                        st.metric(f"{d1_use} — {label}", f"{v1:.3f}",
                                  delta=f"{delta:+.3f}",
                                  delta_color="normal" if better else "inverse")

            # Grafik
            chart_b64 = plot_driver_comparison(comp, selected_race)
            show_chart(chart_b64, f"{d1_use} vs {d2_use} Karşılaştırması")

            # Sektör karşılaştırma
            st.markdown("##### Sektör Bazlı Karşılaştırma")
            sec_cols = ["Sector1Time", "Sector2Time", "Sector3Time"]
            sec_data = []
            for sc in sec_cols:
                v1 = d1d.get(sc)
                v2 = d2d.get(sc)
                if v1 and v2:
                    winner = d1_use if v1 < v2 else d2_use
                    sec_data.append({
                        "Sektör": sc.replace("Time", ""),
                        d1_use:   f"{v1:.3f}s",
                        d2_use:   f"{v2:.3f}s",
                        "Avantaj": f"🏆 {winner}",
                    })
            if sec_data:
                st.dataframe(pd.DataFrame(sec_data), use_container_width=True, hide_index=True)

            # Lastik karşılaştırma
            d1_cmp = d1d.get("compounds_used", [])
            d2_cmp = d2d.get("compounds_used", [])
            if d1_cmp or d2_cmp:
                st.markdown(f"**{d1_use} lastik stratejisi:** {' → '.join(d1_cmp)}")
                st.markdown(f"**{d2_use} lastik stratejisi:** {' → '.join(d2_cmp)}")

            # Head-to-head insight
            deg1 = d1d.get("tire_degradation", 0) or 0
            deg2 = d2d.get("tire_degradation", 0) or 0
            if deg1 and deg2:
                more_stable = d1_use if deg1 < deg2 else d2_use
                st.info(f"💡 Lastik bozulmasında {more_stable} daha stabil "
                        f"({min(deg1,deg2):.4f} vs {max(deg1,deg2):.4f} s/tur).")
    else:
        st.warning("Önce bir yarış yükleyin.")


# ══════════════════════════════════════════════════
# 4. TIRE STRATEGY
# ══════════════════════════════════════════════════
with tab_tire:
    tire = analysis.get("tire_strategy", {})
    deg  = analysis.get("tire_degradation", {})

    if tire.get("compound_stats"):
        st.markdown(f"#### 🔴 Lastik Karşılaştırması — En Avantajlı: **{tire.get('best_compound','?')}**")
        df_tire = pd.DataFrame(tire["compound_stats"])
        cols = [c for c in ["Compound","avg_lap_time","best_lap_time","std","count"] if c in df_tire.columns]
        st.dataframe(df_tire[cols].style.format({
            "avg_lap_time": "{:.3f}", "best_lap_time": "{:.3f}", "std": "{:.3f}",
        }), use_container_width=True, hide_index=True)

        if tire.get("comment"):
            st.success(tire["comment"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Compound Ort. Tur Süresi")
        show_chart(charts.get("compound_avg_laptime"))
    with col_b:
        st.markdown("#### Lastik Bozulma Eğrisi")
        show_chart(charts.get("tire_degradation"))

    st.markdown("#### Lastik Yaşı vs Tur Zamanı")
    show_chart(charts.get("tyre_life_scatter"))

    # Degradation tablosu
    if deg.get("by_compound"):
        st.markdown("#### 📉 Lastik Bozulma Analizi")
        deg_rows = []
        for compound, info in deg["by_compound"].items():
            rate = info.get("degradation_rate")
            r2   = info.get("r_squared")
            deg_rows.append({
                "Lastik":      compound,
                "Bozulma (s/tur)": f"{rate:.4f}" if rate else "N/A",
                "R²":          f"{r2:.3f}" if r2 else "N/A",
                "Değerlendirme": info.get("comment", ""),
            })
        st.dataframe(pd.DataFrame(deg_rows), use_container_width=True, hide_index=True)
        if deg.get("comment"):
            st.info(deg["comment"])


# ══════════════════════════════════════════════════
# 5. PIT STOP
# ══════════════════════════════════════════════════
with tab_pit:
    pit = analysis.get("pit_stop", {})
    avg_impact = pit.get("race_avg_impact")

    if avg_impact is not None:
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            sign = "+" if avg_impact >= 0 else ""
            st.metric("⚡ Ort. Pit Stop Faydası", f"{sign}{avg_impact:.3f}s")
        with col_m2:
            da = pit.get("driver_analysis", [])
            if da:
                best = max(da, key=lambda x: x.get("avg_impact", -999))
                st.metric("🏆 En İyi Pit Stratejisi", best["Driver"],
                          f"+{best['avg_impact']:.3f}s")
        with col_m3:
            total_pits = sum(d.get("pit_count", 0) for d in pit.get("driver_analysis", []))
            st.metric("🔧 Toplam Pit Stop", total_pits)

        st.markdown("#### Pit Stop Etkisi (Öncesi / Sonrası)")
        show_chart(charts.get("pit_impact"))

        da = pit.get("driver_analysis", [])
        if da:
            st.markdown("#### Pilot Bazlı Pit Stop Analizi")
            pit_rows = []
            for d in sorted(da, key=lambda x: x.get("avg_impact", 0), reverse=True):
                impact = d.get("avg_impact", 0)
                pit_rows.append({
                    "Pilot": d["Driver"],
                    "Pit Sayısı": d["pit_count"],
                    "Ort. Etki (s)": f"{impact:+.3f}",
                    "Değerlendirme": "✅ Undercut avantajı" if impact > 0.5 else
                                     "✓ Normal" if impact > 0 else "⚠️ Sınırlı etki",
                })
            st.dataframe(pd.DataFrame(pit_rows), use_container_width=True, hide_index=True)

        if pit.get("comment"):
            st.info(pit["comment"])
    else:
        st.warning("🔧 Pit stop verisi mevcut değil.")
        st.caption("Gerçek FastF1 verisiyle pit stop analizi otomatik olarak dolacak.")


# ══════════════════════════════════════════════════
# 6. WEATHER IMPACT
# ══════════════════════════════════════════════════
with tab_weather:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🌡️ Sıcaklık vs Tur Zamanı")
        show_chart(charts.get("temp_vs_laptime"))
    with col_b:
        st.markdown("#### 📊 Korelasyon Matrisi")
        show_chart(charts.get("correlation_heatmap"))

    weather = analysis.get("weather_impact", {})
    if weather:
        st.markdown("#### Pearson Korelasyon Analizi")
        w_rows = []
        for col, info in weather.items():
            if isinstance(info, dict) and "pearson_r" in info:
                w_rows.append({
                    "Değişken": info.get("label", col),
                    "Pearson r": f"{info['pearson_r']:.4f}",
                    "p-değeri": f"{info['p_value']:.4f}",
                    "Anlamlı?": "✅ Evet" if info.get("significant") else "❌ Hayır",
                    "Yorum": info.get("interpretation", ""),
                })
        if w_rows:
            st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)

        for col, info in weather.items():
            if isinstance(info, dict) and info.get("comment"):
                st.info(info["comment"])
                break


# ══════════════════════════════════════════════════
# 7. ML MODELS
# ══════════════════════════════════════════════════
with tab_ml:
    st.markdown("#### 🤖 Makine Öğrenmesi Sonuçları")

    comp_ml = ml.get("comparison", {})
    dt      = ml.get("decision_tree", {})
    knn_res = ml.get("knn", {})
    km      = ml.get("kmeans", {})

    if comp_ml:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Decision Tree", f"%{comp_ml.get('decision_tree_accuracy',0)*100:.1f}")
        with c2:
            st.metric(f"kNN (k={knn_res.get('k',5)})", f"%{comp_ml.get('knn_accuracy',0)*100:.1f}")
        with c3:
            st.metric("🏆 Kazanan Model", comp_ml.get("winner", "—"))

        if comp_ml.get("comment"):
            st.info(comp_ml["comment"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Model Karşılaştırması")
        show_chart(charts.get("model_comparison"))
    with col_b:
        st.markdown("#### Feature Importance")
        show_chart(charts.get("feature_importance"))

    # Feature importance tablosu
    feat_imp = dt.get("feature_importance", [])
    if feat_imp:
        st.markdown("#### 🔑 Feature Importance Detayı")
        max_imp = max(f[1] for f in feat_imp) if feat_imp else 1
        fi_rows = [{"Özellik": f, "Önem": f"{v:.4f}",
                    "Görsel": "█" * max(1, int(v/max_imp*20))}
                   for f, v in feat_imp]
        st.dataframe(pd.DataFrame(fi_rows), use_container_width=True, hide_index=True)
        st.success(f"🔑 En kritik faktör: **{dt.get('most_important_feature', '—')}**")

    # K-Means
    st.divider()
    st.markdown("#### K-Means Performans Kümeleri")
    col_a, col_b = st.columns(2)
    with col_a:
        show_chart(charts.get("kmeans_clusters"))
    with col_b:
        show_chart(charts.get("elbow_curve"))

    cluster_summary = km.get("cluster_summary", {})
    if cluster_summary:
        colors = {"Stabil & Hızlı": "🟢", "Agresif": "🟡", "Düzensiz": "⚫"}
        for cluster, members in cluster_summary.items():
            icon = colors.get(cluster, "⚪")
            st.markdown(f"**{icon} {cluster}:** {', '.join(members)}")

        if km.get("comment"):
            st.info(km["comment"])


# ══════════════════════════════════════════════════
# 8. CROSS-VALIDATION & GRIDSEARCH
# ══════════════════════════════════════════════════
with tab_cv:
    st.markdown("#### 📊 Cross-Validation & GridSearch Sonuçları")
    eval_res = ml.get("evaluation", {})

    if eval_res.get("error"):
        st.warning(f"⚠️ {eval_res['error']}")
    elif not eval_res:
        st.info("Akademik değerlendirme verisi yükleniyor (ilk çalıştırmada hesaplanır).")
    else:
        # ── CV Özeti ─────────────────────────────────────────
        cv_data = eval_res.get("cross_validation", {})
        if cv_data:
            st.markdown(f"**{cv_data.get('n_folds', 5)}-Fold Stratified Cross-Validation**")
            if cv_data.get("comment"):
                st.info(cv_data["comment"])

            dt_cv  = cv_data.get("decision_tree", {})
            knn_cv = cv_data.get("knn", {})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("DT CV Mean Acc",
                          f"%{dt_cv.get('mean_accuracy', 0)*100:.1f}",
                          delta=f"±{dt_cv.get('std_accuracy', 0)*100:.1f}%")
            with col2:
                st.metric("kNN CV Mean Acc",
                          f"%{knn_cv.get('mean_accuracy', 0)*100:.1f}",
                          delta=f"±{knn_cv.get('std_accuracy', 0)*100:.1f}%")
            with col3:
                st.metric("DT Overfit Gap",
                          f"{dt_cv.get('overfit_gap', 0):.3f}",
                          delta_color="inverse")
            with col4:
                st.metric("kNN Overfit Gap",
                          f"{knn_cv.get('overfit_gap', 0):.3f}",
                          delta_color="inverse")

            show_chart(cv_data.get("chart_b64", ""), "Fold Bazlı CV Skoru")

            # Fold detay tablosu
            st.markdown("##### Fold Bazlı Detay")
            fold_rows = []
            for fold_i, (dt_acc, knn_acc) in enumerate(
                zip(dt_cv.get("fold_accuracies", []), knn_cv.get("fold_accuracies", [])), 1
            ):
                fold_rows.append({
                    "Fold": fold_i,
                    "DT Accuracy": f"{dt_acc:.4f}",
                    "kNN Accuracy": f"{knn_acc:.4f}",
                    "Fark": f"{dt_acc - knn_acc:+.4f}",
                    "Üstün Model": "DT" if dt_acc > knn_acc else ("kNN" if knn_acc > dt_acc else "Eşit"),
                })
            if fold_rows:
                st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)

            # Metrik tablosu
            st.markdown("##### Kapsamlı Metrik Karşılaştırması")
            metric_rows = [
                {
                    "Metrik": "Ortalama Accuracy",
                    "Decision Tree": f"{dt_cv.get('mean_accuracy', 0):.4f}",
                    "kNN": f"{knn_cv.get('mean_accuracy', 0):.4f}",
                },
                {
                    "Metrik": "Std Sapma",
                    "Decision Tree": f"{dt_cv.get('std_accuracy', 0):.4f}",
                    "kNN": f"{knn_cv.get('std_accuracy', 0):.4f}",
                },
                {
                    "Metrik": "Precision (macro)",
                    "Decision Tree": f"{dt_cv.get('mean_precision', 0):.4f}",
                    "kNN": f"{knn_cv.get('mean_precision', 0):.4f}",
                },
                {
                    "Metrik": "Recall (macro)",
                    "Decision Tree": f"{dt_cv.get('mean_recall', 0):.4f}",
                    "kNN": f"{knn_cv.get('mean_recall', 0):.4f}",
                },
                {
                    "Metrik": "F1-Score (macro)",
                    "Decision Tree": f"{dt_cv.get('mean_f1', 0):.4f}",
                    "kNN": f"{knn_cv.get('mean_f1', 0):.4f}",
                },
                {
                    "Metrik": "Overfit Gap (Train-Test)",
                    "Decision Tree": f"{dt_cv.get('overfit_gap', 0):.4f}",
                    "kNN": f"{knn_cv.get('overfit_gap', 0):.4f}",
                },
            ]
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── GridSearch ────────────────────────────────────────
        gs_data = eval_res.get("grid_search", {})
        if gs_data and not gs_data.get("error"):
            st.markdown("**GridSearchCV Sonuçları**")

            col_gs1, col_gs2 = st.columns(2)
            with col_gs1:
                dt_gs = gs_data.get("decision_tree", {})
                st.markdown("**Decision Tree — En İyi Parametreler**")
                st.json(dt_gs.get("best_params", {}))
                st.metric("DT Best CV Score", f"%{dt_gs.get('best_score', 0)*100:.1f}")
                if dt_gs.get("comment"):
                    st.caption(dt_gs["comment"])

            with col_gs2:
                knn_gs = gs_data.get("knn", {})
                st.markdown("**kNN — En İyi Parametreler**")
                st.json(knn_gs.get("best_params", {}))
                st.metric("kNN Best CV Score", f"%{knn_gs.get('best_score', 0)*100:.1f}")
                if knn_gs.get("comment"):
                    st.caption(knn_gs["comment"])

            show_chart(gs_data.get("chart_b64", ""), "GridSearch Sonuçları")

        st.divider()

        # ── K-Means Optimizasyonu ─────────────────────────────
        km_opt = eval_res.get("kmeans_optimization", {})
        if km_opt:
            st.markdown("**K-Means Hiperparametre Optimizasyonu (Silhouette Score)**")
            if km_opt.get("comment"):
                st.success(km_opt["comment"])
            show_chart(km_opt.get("chart_b64", ""), "K-Means Optimizasyonu")

            km_rows = km_opt.get("all_results", [])
            if km_rows:
                st.dataframe(pd.DataFrame(km_rows), use_container_width=True, hide_index=True)

        # ── İstatistiksel Testler ─────────────────────────────
        st.divider()
        st.markdown("**İstatistiksel Anlamlılık Testleri**")
        stat_tests = ml.get("statistical_tests", {})
        if stat_tests and not stat_tests.get("error"):
            show_chart(stat_tests.get("chart_b64", ""), "İstatistiksel Test Sonuçları")

            for test_key, test_name in [
                ("mcnemar", "McNemar Testi"),
                ("paired_ttest", "Paired t-test"),
                ("wilcoxon", "Wilcoxon Testi"),
            ]:
                t = stat_tests.get(test_key, {})
                if t.get("conclusion"):
                    st.info(f"**{test_name}:** {t['conclusion']}")

            for line in stat_tests.get("interpretation", []):
                st.caption(f"• {line}")
        else:
            st.caption("İstatistiksel test verisi mevcut değil.")


# ══════════════════════════════════════════════════
# 9. ROC / AUC ANALİZİ
# ══════════════════════════════════════════════════
with tab_roc:
    st.markdown("#### 📈 ROC Eğrisi & AUC Analizi")
    eval_res = ml.get("evaluation", {})
    roc_data = eval_res.get("roc_auc", {})

    if not roc_data or roc_data.get("error"):
        st.warning("ROC/AUC verisi mevcut değil.")
        st.caption("Model değerlendirme pipeline'ı çalıştırıldığında hesaplanacak.")
    else:
        col_r1, col_r2, col_r3 = st.columns(3)
        dt_roc  = roc_data.get("Decision Tree", {})
        knn_roc = roc_data.get("kNN", {})

        with col_r1:
            st.metric("Decision Tree AUC", f"{dt_roc.get('auc', 0):.4f}")
        with col_r2:
            st.metric("kNN AUC", f"{knn_roc.get('auc', 0):.4f}")
        with col_r3:
            better = "Decision Tree" if dt_roc.get("auc", 0) >= knn_roc.get("auc", 0) else "kNN"
            st.metric("🏆 Üstün Model (AUC)", better)

        if roc_data.get("comment"):
            st.info(roc_data["comment"])

        show_chart(roc_data.get("chart_b64", ""), "ROC Eğrisi")

        st.markdown("#### ROC Detayları")
        roc_rows = []
        for model_name, data in roc_data.items():
            if model_name in ("chart_b64", "comment") or "comment" in str(model_name):
                continue
            if not isinstance(data, dict) or "auc" not in data:
                continue
            roc_rows.append({
                "Model":           model_name,
                "AUC":             f"{data.get('auc', 0):.4f}",
                "Optimal Eşik":    f"{data.get('best_threshold', 0):.4f}",
                "TPR @ Eşik":      f"{data.get('best_tpr', 0):.4f}",
                "FPR @ Eşik":      f"{data.get('best_fpr', 0):.4f}",
                "Değerlendirme": (
                    "Mükemmel (≥0.90)" if data.get("auc", 0) >= 0.9 else
                    "İyi (0.80-0.90)" if data.get("auc", 0) >= 0.8 else
                    "Orta (0.70-0.80)" if data.get("auc", 0) >= 0.7 else
                    "Zayıf (≤0.70)"
                ),
            })
        if roc_rows:
            st.dataframe(pd.DataFrame(roc_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("""
        **ROC Eğrisi Yorumu:**
        - **AUC = 1.0**: Mükemmel sınıflandırıcı
        - **AUC = 0.5**: Rastgele sınıflandırıcı (faydasız)
        - **AUC > 0.8**: Akademik açıdan kabul edilebilir performans
        - **Optimal Eşik**: Youden J istatistiğini (TPR - FPR) maksimize eden eşik
        """)

    # Model karşılaştırma tablosu
    st.divider()
    st.markdown("#### Model Karşılaştırma Tablosu")
    comp_data = eval_res.get("model_comparison", {})
    if comp_data and comp_data.get("dataframe"):
        df_comp = pd.DataFrame(comp_data["dataframe"])
        numeric_cols = ["CV Mean Acc", "CV Std", "CV Precision", "CV Recall",
                        "CV F1", "AUC", "GS Best Score", "Overfit Gap"]
        st.dataframe(
            df_comp.style.format({
                col: "{:.4f}" for col in numeric_cols if col in df_comp.columns
            }).background_gradient(subset=["AUC", "CV Mean Acc"] if "AUC" in df_comp.columns else [],
                                   cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )
        st.success(f"🏆 En iyi model: **{comp_data.get('best_model', '—')}**")
        show_chart(comp_data.get("chart_b64", ""), "Model Karşılaştırması")
        for line in comp_data.get("interpretation", []):
            st.info(f"💡 {line}")


# ══════════════════════════════════════════════════
# 10. XAI (SHAP / PERMUTATION IMPORTANCE)
# ══════════════════════════════════════════════════
with tab_xai:
    st.markdown("#### 🔍 Açıklanabilir Yapay Zeka (XAI)")
    xai_data = ml.get("xai", {})

    if not xai_data or xai_data.get("error"):
        st.warning(xai_data.get("error", "XAI verisi mevcut değil."))
    else:
        has_shap = xai_data.get("has_shap", False)
        if has_shap:
            st.success("✅ SHAP TreeExplainer aktif")
        else:
            st.info("ℹ️ SHAP kurulu değil — Permutation Importance kullanılıyor. (`pip install shap`)")

        # SHAP / Fallback
        shap_dt = xai_data.get("shap_dt", {})
        if shap_dt:
            st.markdown(f"**Yöntem:** `{shap_dt.get('method', '—')}`")
            if shap_dt.get("comment"):
                st.success(shap_dt["comment"])
            if shap_dt.get("top_feature"):
                st.metric("En Kritik Özellik", shap_dt["top_feature"])

            col_xa, col_xb = st.columns(2)
            with col_xa:
                show_chart(shap_dt.get("chart_b64", ""), "SHAP Özellik Önemi")
            with col_xb:
                if shap_dt.get("beeswarm_b64"):
                    show_chart(shap_dt["beeswarm_b64"], "SHAP Beeswarm")

        st.divider()

        # Permutation Importance karşılaştırması
        st.markdown("#### Permutation Importance — DT vs kNN")
        show_chart(xai_data.get("comparison_chart_b64", ""), "Karşılaştırmalı Permutation Importance")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            perm_dt = xai_data.get("perm_importance_dt", {})
            if perm_dt.get("importance"):
                st.markdown("**Decision Tree — Permutation Importance**")
                df_pi = pd.DataFrame(perm_dt["importance"]).sort_values("PI Mean", ascending=False)
                st.dataframe(df_pi.style.format({"PI Mean": "{:.5f}", "PI Std": "{:.5f}"}),
                             use_container_width=True, hide_index=True)
                if perm_dt.get("comment"):
                    st.caption(perm_dt["comment"])

        with col_p2:
            perm_knn = xai_data.get("perm_importance_knn", {})
            if perm_knn.get("importance"):
                st.markdown("**kNN — Permutation Importance**")
                df_pi2 = pd.DataFrame(perm_knn["importance"]).sort_values("PI Mean", ascending=False)
                st.dataframe(df_pi2.style.format({"PI Mean": "{:.5f}", "PI Std": "{:.5f}"}),
                             use_container_width=True, hide_index=True)
                if perm_knn.get("comment"):
                    st.caption(perm_knn["comment"])

        # Yerel açıklamalar
        st.divider()
        st.markdown("#### Yerel Açıklamalar — Sürücü Bazlı Tahmin Analizi")
        local_exp = xai_data.get("local_explanations", [])
        if local_exp:
            le_rows = []
            for exp in local_exp:
                row = {
                    "Sürücü": exp["driver"],
                    "Gerçek": "Güçlü" if exp["actual"] == 1 else "Normal",
                    "Tahmin": exp["prediction_label"],
                    "Doğru?": "✅" if exp["correct"] else "❌",
                }
                if exp.get("confidence") is not None:
                    row["Güven"] = f"{exp['confidence']:.2%}"
                for feat, val in (exp.get("features") or {}).items():
                    row[feat] = f"{val:.3f}"
                le_rows.append(row)

            st.dataframe(
                pd.DataFrame(le_rows).style.apply(
                    lambda col: ["background: #1A4D1A" if v == "✅" else
                                 "background: #4D1A1A" if v == "❌" else ""
                                 for v in col] if col.name == "Doğru?" else [""] * len(col),
                    axis=0,
                ),
                use_container_width=True, hide_index=True,
            )


# ══════════════════════════════════════════════════
# 11. HATA ANALİZİ
# ══════════════════════════════════════════════════
with tab_error:
    st.markdown("#### ⚠️ Hata Analizi — Yanlış Tahmin İncelemesi")
    ea_data = ml.get("error_analysis", {})

    if not ea_data or ea_data.get("error"):
        st.warning(ea_data.get("error", "Hata analizi verisi mevcut değil."))
    else:
        # Genel özet
        dt_err  = ea_data.get("dt_errors", {})
        knn_err = ea_data.get("knn_errors", {})

        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        with col_e1:
            st.metric("DT Hata Oranı", f"%{dt_err.get('error_rate', 0)*100:.1f}")
        with col_e2:
            st.metric("kNN Hata Oranı", f"%{knn_err.get('error_rate', 0)*100:.1f}")
        with col_e3:
            st.metric("DT FP", dt_err.get("false_positives", 0))
        with col_e4:
            st.metric("DT FN", dt_err.get("false_negatives", 0))

        if dt_err.get("comment"):
            st.info(f"**DT:** {dt_err['comment']}")
        if knn_err.get("comment"):
            st.info(f"**kNN:** {knn_err['comment']}")

        # Confusion Matrix
        show_chart(ea_data.get("cm_chart_b64", ""), "Confusion Matrix Karşılaştırması")

        # Ortak hatalar
        common = ea_data.get("common_errors", [])
        if common:
            st.markdown(f"#### Her İki Model Tarafından da Yanlış Tahmin Edilen Sürücüler ({len(common)} adet)")
            st.dataframe(pd.DataFrame(common), use_container_width=True, hide_index=True)

        # Sürücü bazlı rapor
        st.divider()
        st.markdown("#### Sürücü Bazlı Tahmin Sonuçları")
        driver_report = ea_data.get("driver_error_report", [])
        if driver_report:
            st.dataframe(pd.DataFrame(driver_report), use_container_width=True, hide_index=True)
        show_chart(ea_data.get("driver_error_chart_b64", ""), "Sürücü Bazlı Hata Dağılımı")

        # Yüksek güvenli hatalar
        hce_dt = dt_err.get("high_conf_errors", [])
        if hce_dt:
            st.markdown(f"#### ⚠️ Yüksek Güvenli Yanlış Tahminler — Decision Tree ({len(hce_dt)} adet)")
            st.dataframe(pd.DataFrame(hce_dt), use_container_width=True, hide_index=True)
            st.caption("Bu sürücüler için model yüksek güven (>70%) ile yanlış tahmin yaptı.")

        # Hata dağılım grafiği
        show_chart(ea_data.get("feature_error_chart_b64", ""), "Hata Dağılımı (Özellik Uzayı)")


# ══════════════════════════════════════════════════
# 12. AI INSIGHTS
# ══════════════════════════════════════════════════
with tab_insight:
    st.markdown("#### 💡 AI Insight Engine — Otomatik Analitik Yorumlar")

    tc = insights.get("track_character", {})
    if tc:
        st.info(f"🏁 **{tc.get('headline','')}**\n\n{tc.get('detail','')}")
        st.caption(f"DRS Bölgesi: {tc.get('drs_zones','?')} | Pit Penceresi: {tc.get('pit_window','?')}")

    st.divider()

    all_insight_sections = [
        ("🏆 Performans", insights.get("performance_insights", [])),
        ("🔴 Lastik",     insights.get("tire_insights", [])),
        ("🔧 Pit Stop",   insights.get("pit_insights", [])),
        ("🌡️ Hava",       insights.get("weather_insights", [])),
        ("🤖 ML",         insights.get("ml_insights", [])),
        ("👤 Pilot",      insights.get("driver_insights", [])),
    ]

    for section_title, section_insights in all_insight_sections:
        if section_insights:
            st.markdown(f"**{section_title}**")
            for ins in section_insights:
                show_insight_card(ins)
            st.markdown("")


# ══════════════════════════════════════════════════
# 9. REPORT
# ══════════════════════════════════════════════════
with tab_report:
    st.markdown("#### 📄 Otomatik Rapor Üretici")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("📄 Markdown Rapor Üret", use_container_width=True):
            with st.spinner("Rapor üretiliyor..."):
                try:
                    paths = generate_race_report(selected_race, analysis, ml)
                    md_content = paths["markdown"].read_text(encoding="utf-8")
                    st.success(f"✅ Rapor kaydedildi: {paths['markdown'].name}")
                    st.download_button(
                        "⬇️ Markdown İndir",
                        data=md_content,
                        file_name=paths["markdown"].name,
                        mime="text/markdown",
                        use_container_width=True,
                    )
                    with st.expander("Rapor Önizleme"):
                        st.markdown(md_content[:3000] + "..." if len(md_content) > 3000 else md_content)
                except Exception as e:
                    st.error(f"Rapor hatası: {e}")

    with col_r2:
        if st.button("📊 PDF Rapor Üret", use_container_width=True):
            with st.spinner("PDF oluşturuluyor..."):
                try:
                    insg = generate_race_insights(selected_race, df, analysis, ml)
                    chrt = generate_all_charts(df, analysis, ml, selected_race)
                    pdf_path = generate_pdf_report(selected_race, analysis, ml, insg, chrt)
                    if pdf_path:
                        st.success(f"✅ PDF kaydedildi: {pdf_path.name}")
                        pdf_bytes = pdf_path.read_bytes()
                        st.download_button(
                            "⬇️ PDF İndir",
                            data=pdf_bytes,
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    else:
                        st.error("PDF oluşturulamadı.")
                except Exception as e:
                    st.error(f"PDF hatası: {e}")

    st.divider()
    # Mevcut raporları listele
    import glob as _glob
    existing_reports = sorted(_glob.glob(str(GENERATED_REPORTS_DIR / "*.md")), reverse=True)
    if existing_reports:
        st.markdown("#### 📁 Mevcut Raporlar")
        for rp in existing_reports[:5]:
            p = Path(rp)
            content_bytes = p.read_bytes()
            st.download_button(
                f"⬇️ {p.name}",
                data=content_bytes,
                file_name=p.name,
                mime="text/markdown",
            )


# ══════════════════════════════════════════════════
# 10. SUNUM MODU
# ══════════════════════════════════════════════════
with tab_pres:
    st.markdown(f"## 🎯 {selected_race} GP — Sunum Özeti")
    st.caption(f"Formula 1 {SEASON} Sezonu | Veri: {'⚠️ Simüle' if is_sim else '✅ Gerçek FastF1'}")
    st.divider()

    top5 = insights.get("top5_summary", [])
    st.markdown("### En Kritik 5 Çıkarım")
    if top5:
        cols_pres = st.columns(min(3, len(top5)))
        for i, ins in enumerate(top5[:3]):
            with cols_pres[i]:
                level_colors = {"positive": "#00C851", "warning": "#FFB300",
                                "neutral": "#888", "info": "#2196F3"}
                color = level_colors.get(ins.get("level", "neutral"), "#888")
                st.markdown(f"""
                <div style="background:#1A1A1A;border:1px solid #2A2A2A;border-top:3px solid {color};
                     border-radius:6px;padding:16px;text-align:center;">
                    <div style="font-size:30px">{ins.get('icon','')}</div>
                    <div style="color:#888;font-size:10px;letter-spacing:2px">{ins.get('title','').upper()}</div>
                    <div style="color:{color};font-size:22px;font-weight:900;font-family:monospace;
                         margin:8px 0">{ins.get('driver','')}</div>
                    <div style="color:#fff;font-size:14px">{ins.get('value','')}</div>
                </div>
                """, unsafe_allow_html=True)

        if len(top5) > 3:
            cols2 = st.columns(len(top5) - 3)
            for i, ins in enumerate(top5[3:]):
                with cols2[i]:
                    st.info(f"{ins.get('icon','')} **{ins.get('title','')}**: "
                            f"{ins.get('driver','')} — {ins.get('value','')}")

    st.divider()
    st.markdown("### 📊 Ana Grafik")
    show_chart(charts.get("driver_avg_laptime"), "Pilot Ortalama Tur Süresi")

    st.divider()
    st.markdown("### 🤖 Model Sonucu")
    comp_ml = ml.get("comparison", {})
    dt      = ml.get("decision_tree", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Kazanan Model", comp_ml.get("winner", "—"))
    with c2:
        st.metric("Doğruluk", f"%{max(comp_ml.get('decision_tree_accuracy',0), comp_ml.get('knn_accuracy',0))*100:.1f}")
    with c3:
        st.metric("En Kritik Faktör", dt.get("most_important_feature", "—"))

    if comp_ml.get("comment"):
        st.info(comp_ml["comment"])
