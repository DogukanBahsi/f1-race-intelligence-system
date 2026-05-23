"""
src/insight_engine.py
---------------------
F1 Race Intelligence System - AI Insight Engine
Analiz sonuçlarından otomatik, okunabilir Türkçe yorumlar üretir.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.logger import get_logger
from src.utils import seconds_to_mmss

logger = get_logger("InsightEngine")


# ─────────────────────────────────────────────────────────────
# PIST KARAKTERİSTİKLERİ (Statik bilgi — veri yoksa kullanılır)
# ─────────────────────────────────────────────────────────────

TRACK_INSIGHTS = {
    "Bahrain": {
        "character": "yüksek sıcaklık ve lastik aşınmasının belirleyici olduğu",
        "notes": "Gece yarışı olmasına rağmen pist sıcaklığı yüksek kalır. "
                 "MEDIUM lastik tercih edilir, son stint HARD ile tamamlanır.",
        "drs_zones": 3, "pit_window": "12-18. turlar",
    },
    "Monaco": {
        "character": "dar sokaklar ve düşük ortalama hızın egemen olduğu",
        "notes": "Overtake neredeyse imkânsız. Pit stop zamanlaması yarışı belirler. "
                 "SC çıkma olasılığı yüksek; undercut stratejisi kritik.",
        "drs_zones": 1, "pit_window": "18-28. turlar",
    },
    "Italy": {
        "character": "yüksek hız ve düşük downforce konfigürasyonunun ön planda olduğu",
        "notes": "Monza'nın uzun düzlükleri en düşük wing açısı gerektirir. "
                 "SOFT + HARD iki durak yaygın. Motor gücü belirleyici.",
        "drs_zones": 2, "pit_window": "16-22. turlar",
    },
    "Belgium": {
        "character": "teknik virajlar ve Raidillon gibi yüksek hızlı bölümlerin bulunduğu",
        "notes": "Spa'nın uzun pisti (~7 km) lastik bozulmasını artırır. "
                 "Yağmur sık; hava durumu strateji değişkenliği yaratır.",
        "drs_zones": 2, "pit_window": "10-18. turlar",
    },
    "Great Britain": {
        "character": "yüksek aerodinamik yük ve hızlı köşelerin belirleyici olduğu",
        "notes": "Silverstone'un yüksek hızlı köşeleri arka lastiklere yük bindirir. "
                 "Copse ve Maggots-Beckets sektörleri differansiyel performans yaratır.",
        "drs_zones": 2, "pit_window": "14-22. turlar",
    },
    "Japan": {
        "character": "teknik S1 ve yüksek hızlı S2'nin zorlu dengesi olduğu",
        "notes": "Suzuka'nın 8 rakamı pisti aerodinamik uzlaşma gerektirir. "
                 "MEDIUM + HARD iki durak optimal. Yağmur olasılığı düşük.",
        "drs_zones": 1, "pit_window": "14-20. turlar",
    },
    "Singapore": {
        "character": "gece yarışı ve yüksek SC olasılığının kritik olduğu",
        "notes": "Marina Bay sokak devresi öngörülemeyen SC dönemleri üretir. "
                 "Tek durak stratejisi SC sayesinde avantajlı olabilir.",
        "drs_zones": 3, "pit_window": "20-30. turlar",
    },
    "Azerbaijan": {
        "character": "uzun ana düzlük ve dar şehir bölümlerinin kontrast oluşturduğu",
        "notes": "Baku'nun 2.2 km düzlüğü en yüksek hızı sunar. "
                 "SC/VSC olasılığı çok yüksek; undercut penceresi kritik.",
        "drs_zones": 2, "pit_window": "12-22. turlar",
    },
}


# ─────────────────────────────────────────────────────────────
# ANA INSIGHT FONKSİYONU
# ─────────────────────────────────────────────────────────────

def generate_race_insights(race_name: str,
                            df: pd.DataFrame,
                            analysis: Dict[str, Any],
                            ml: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tüm insight modüllerini çalıştırır.
    Dashboard AI Insights sekmesi için yapılandırılmış veri döndürür.
    """
    logger.info(f"Insight engine çalışıyor: {race_name}")
    insights = {
        "race_name": race_name,
        "track_character": _track_character(race_name),
        "performance_insights": _performance_insights(analysis, race_name),
        "tire_insights": _tire_insights(analysis, df, race_name),
        "pit_insights": _pit_insights(analysis, race_name),
        "weather_insights": _weather_insights(analysis, df, race_name),
        "ml_insights": _ml_insights(ml, race_name),
        "driver_insights": _driver_insights(analysis, df, race_name),
        "top5_summary": [],
        "is_simulated": "IsSimulated" in df.columns and df["IsSimulated"].any(),
    }

    # Top 5 kritik çıkarım (sunum için)
    insights["top5_summary"] = _build_top5(insights, race_name)
    logger.info(f"Insight engine tamamlandı: {race_name}")
    return insights


# ─────────────────────────────────────────────────────────────
# PİST KARAKTERİ
# ─────────────────────────────────────────────────────────────

def _track_character(race_name: str) -> Dict[str, str]:
    cfg = TRACK_INSIGHTS.get(race_name, {
        "character": "kendine özgü karakteristikleri olan",
        "notes": f"{race_name} için detaylı pist analizi mevcut değil.",
        "drs_zones": "N/A", "pit_window": "N/A",
    })
    return {
        "name": race_name,
        "headline": f"{race_name}, {cfg['character']} bir pist.",
        "detail": cfg["notes"],
        "drs_zones": str(cfg["drs_zones"]),
        "pit_window": cfg.get("pit_window", "N/A"),
    }


# ─────────────────────────────────────────────────────────────
# PERFORMANS INSIGHT
# ─────────────────────────────────────────────────────────────

def _performance_insights(analysis: Dict, race_name: str) -> List[Dict]:
    insights = []
    stab = analysis.get("driver_stability", {})
    data = stab.get("data", [])
    if not data:
        return insights

    # En stabil pilot
    most_stable = stab.get("most_stable", "N/A")
    fastest     = stab.get("fastest", "N/A")

    if data:
        top = data[0]
        cs  = top.get("consistency_score", 0)
        insights.append({
            "icon": "🎯",
            "title": "En İstikrarlı Pilot",
            "driver": most_stable,
            "value": f"Consistency: {cs:.4f}",
            "text": (f"{most_stable} bu yarışta en tutarlı tur zamanlarını sergiledi "
                     f"(std sapma: {top.get('lap_time_std', 0):.3f}s). "
                     f"İstikrarlı bir yarış stratejisi izledi."),
            "level": "positive",
        })

    # En hızlı pilot
    if fastest != "N/A":
        best_row = min(data, key=lambda x: x.get("best_lap_time", 9999))
        insights.append({
            "icon": "⚡",
            "title": "En Hızlı Pilot",
            "driver": fastest,
            "value": seconds_to_mmss(best_row.get("best_lap_time", 0)),
            "text": (f"{fastest} yarışın en hızlı turunu kaydetti "
                     f"({seconds_to_mmss(best_row.get('best_lap_time', 0))}). "
                     f"Ortalama yarış hızı da dikkat çekici düzeyde."),
            "level": "positive",
        })

    # Hız vs istikrar uçurumu
    if fastest != most_stable and fastest != "N/A" and most_stable != "N/A":
        insights.append({
            "icon": "⚖️",
            "title": "Hız - İstikrar Dengesi",
            "driver": f"{fastest} vs {most_stable}",
            "value": "Zıt profiller",
            "text": (f"{fastest} ham hızda öne çıkarken {most_stable} tutarlılıkta üstün. "
                     f"Bu yarışta tutarlılık mı yoksa saf hız mı daha kritik? "
                     f"Pist karakteristiğine göre değişiyor."),
            "level": "neutral",
        })

    return insights


# ─────────────────────────────────────────────────────────────
# LASTİK INSIGHT
# ─────────────────────────────────────────────────────────────

def _tire_insights(analysis: Dict, df: pd.DataFrame, race_name: str) -> List[Dict]:
    insights = []
    tire = analysis.get("tire_strategy", {})
    deg  = analysis.get("tire_degradation", {})

    best_compound = tire.get("best_compound", "N/A")
    compound_stats = tire.get("compound_stats", [])

    if best_compound != "N/A":
        best_data = next((r for r in compound_stats if r.get("Compound") == best_compound), {})
        insights.append({
            "icon": "🔴",
            "title": "En Avantajlı Lastik",
            "driver": best_compound,
            "value": f"{best_data.get('avg_lap_time', 0):.3f}s ort.",
            "text": (f"{race_name}'de {best_compound} lastiği en düşük ortalama tur süresini "
                     f"kaydetmiştir ({best_data.get('avg_lap_time', 0):.3f}s). "
                     f"{int(best_data.get('count', 0))} tur verisi baz alınmıştır."),
            "level": "positive",
        })

    # Bozulma insight
    worst = deg.get("worst_compound")
    by_cmp = deg.get("by_compound", {})
    if worst and worst in by_cmp:
        rate = by_cmp[worst].get("degradation_rate", 0) or 0
        insights.append({
            "icon": "📉",
            "title": "En Hızlı Bozulan Lastik",
            "driver": worst,
            "value": f"{rate:.4f} s/tur",
            "text": (f"{worst} lastiği bu pistte en yüksek bozulma oranını gösterdi "
                     f"({rate:.4f} s/tur artış). "
                     f"{'Erken pit stop avantajlı olabilir.' if rate > 0.1 else 'Bozulma kontrol altında.'}"),
            "level": "warning" if rate > 0.1 else "neutral",
        })

    # Compound fark analizi
    if len(compound_stats) >= 2:
        sorted_c = sorted(compound_stats, key=lambda x: x.get("avg_lap_time", 9999))
        fastest_c = sorted_c[0]
        slowest_c = sorted_c[-1]
        diff = slowest_c.get("avg_lap_time", 0) - fastest_c.get("avg_lap_time", 0)
        if diff > 0.3:
            insights.append({
                "icon": "📊",
                "title": "Lastik Performans Farkı",
                "driver": f"{fastest_c.get('Compound')} > {slowest_c.get('Compound')}",
                "value": f"{diff:.3f}s fark",
                "text": (f"Lastikler arasında {diff:.3f}s anlamlı fark var. "
                         f"{fastest_c.get('Compound')} ile {slowest_c.get('Compound')} "
                         f"arasındaki bu açık strateji kararlarını doğrudan etkiler."),
                "level": "info",
            })

    return insights


# ─────────────────────────────────────────────────────────────
# PIT STOP INSIGHT
# ─────────────────────────────────────────────────────────────

def _pit_insights(analysis: Dict, race_name: str) -> List[Dict]:
    insights = []
    pit = analysis.get("pit_stop", {})

    avg_impact = pit.get("race_avg_impact")
    if avg_impact is None:
        insights.append({
            "icon": "🔧",
            "title": "Pit Stop Verisi",
            "driver": "N/A",
            "value": "Veri yok",
            "text": "Bu yarış için pit stop verisi mevcut değil.",
            "level": "neutral",
        })
        return insights

    level = "positive" if avg_impact > 0 else "warning"
    insights.append({
        "icon": "🔧",
        "title": "Pit Stop Etkisi",
        "driver": "Tüm Pilotlar",
        "value": f"{avg_impact:+.3f}s",
        "text": (f"{race_name} yarışında ortalama pit stop faydası {avg_impact:+.3f}s. "
                 + ("Taze lastik avantajı açıkça görülüyor. Undercut penceresi genişti."
                    if avg_impact > 0.5
                    else "Pit stop sonrası hız artışı sınırlı kaldı.")),
        "level": level,
    })

    # En iyi pit stop pilotu
    driver_analysis = pit.get("driver_analysis", [])
    if driver_analysis:
        best_pit = max(driver_analysis, key=lambda x: x.get("avg_impact", -999))
        if best_pit.get("avg_impact", 0) > 0:
            insights.append({
                "icon": "🏆",
                "title": "En İyi Pit Stop Stratejisi",
                "driver": best_pit["Driver"],
                "value": f"+{best_pit['avg_impact']:.3f}s",
                "text": (f"{best_pit['Driver']} pit stop sonrası en yüksek hız artışını "
                         f"elde etti (+{best_pit['avg_impact']:.3f}s/tur). "
                         f"Strateji kararı doğru zamanlama gösterdi."),
                "level": "positive",
            })

    return insights


# ─────────────────────────────────────────────────────────────
# HAVA INSIGHT
# ─────────────────────────────────────────────────────────────

def _weather_insights(analysis: Dict, df: pd.DataFrame, race_name: str) -> List[Dict]:
    insights = []
    weather = analysis.get("weather_impact", {})

    for col, info in weather.items():
        if not isinstance(info, dict) or "pearson_r" not in info:
            continue
        r   = info.get("pearson_r", 0)
        sig = info.get("significant", False)
        lbl = info.get("label", col)

        if abs(r) > 0.3 and sig:
            direction = "arttıkça tur zamanı uzuyor" if r > 0 else "arttıkça tur zamanı kısalıyor"
            insights.append({
                "icon": "🌡️",
                "title": f"{lbl} Etkisi",
                "driver": race_name,
                "value": f"r = {r:.3f}",
                "text": (f"{race_name}'de {lbl.lower()} {direction} (r={r:.3f}). "
                         f"İstatistiksel olarak anlamlı bu ilişki strateji planlamasında "
                         f"göz önünde bulundurulmalı."),
                "level": "info",
            })
        elif not sig:
            insights.append({
                "icon": "🌡️",
                "title": f"{lbl} Etkisi",
                "driver": race_name,
                "value": f"r = {r:.3f} (p>0.05)",
                "text": (f"{race_name}'de {lbl.lower()} ile tur zamanı arasında "
                         f"istatistiksel olarak anlamlı bir ilişki bulunamadı (r={r:.3f})."),
                "level": "neutral",
            })

    # Ortalama sıcaklıklar
    if "AirTemp" in df.columns:
        avg_air   = df["AirTemp"].mean()
        avg_track = df["TrackTemp"].mean() if "TrackTemp" in df.columns else None
        t_desc = "yüksek" if avg_track and avg_track > 40 else ("orta" if avg_track and avg_track > 30 else "düşük")
        insights.append({
            "icon": "☀️",
            "title": "Yarış Sıcaklık Koşulları",
            "driver": race_name,
            "value": f"{avg_air:.1f}°C hava / {avg_track:.1f}°C pist" if avg_track else f"{avg_air:.1f}°C",
            "text": (f"Yarış boyunca ortalama pist sıcaklığı {t_desc} seviyede ({avg_track:.1f}°C). "
                     f"Bu koşullar lastik bozulması ve performans için "
                     f"{'kritik etkiye sahip.' if avg_track and avg_track > 45 else 'makul düzeyde.'}")
                     if avg_track else f"Hava sıcaklığı: {avg_air:.1f}°C",
            "level": "warning" if avg_track and avg_track > 45 else "neutral",
        })

    return insights


# ─────────────────────────────────────────────────────────────
# ML INSIGHT
# ─────────────────────────────────────────────────────────────

def _ml_insights(ml: Dict, race_name: str) -> List[Dict]:
    insights = []
    dt   = ml.get("decision_tree", {})
    knn  = ml.get("knn", {})
    comp = ml.get("comparison", {})
    km   = ml.get("kmeans", {})

    # Feature importance
    feat_imp = dt.get("feature_importance", [])
    if feat_imp:
        top_feat = feat_imp[0]
        feat_name = top_feat[0]
        feat_val  = top_feat[1]
        readable = {
            "consistency_score":   "tur zamanı tutarlılığı",
            "average_lap_time":    "ortalama tur hızı",
            "tire_degradation_rate": "lastik bozulma hızı",
            "pit_stop_impact":     "pit stop etkisi",
            "sector_consistency":  "sektör tutarlılığı",
            "avg_air_temp":        "ortalama hava sıcaklığı",
            "lap_time_std":        "tur zamanı değişkenliği",
        }.get(feat_name, feat_name)

        insights.append({
            "icon": "🔑",
            "title": "En Kritik Performans Faktörü",
            "driver": feat_name,
            "value": f"Önem: {feat_val:.3f}",
            "text": (f"Decision Tree analizine göre {race_name}'de yarış performansını "
                     f"en çok etkileyen faktör **{readable}** olarak belirlendi "
                     f"(önem skoru: {feat_val:.3f}). "
                     f"Bu faktöre odaklanan pilotlar avantaj elde etti."),
            "level": "positive",
        })

    # Model karşılaştırma
    winner = comp.get("winner", "N/A")
    dt_acc = comp.get("decision_tree_accuracy", 0) * 100
    kn_acc = comp.get("knn_accuracy", 0) * 100
    if winner != "N/A":
        insights.append({
            "icon": "🤖",
            "title": "En İyi Model",
            "driver": winner,
            "value": f"%{max(dt_acc, kn_acc):.1f} doğruluk",
            "text": (f"Decision Tree (%{dt_acc:.1f}) ve kNN (%{kn_acc:.1f}) karşılaştırmasında "
                     f"{winner} daha iyi performans gösterdi. "
                     f"Model, 'güçlü performans' tahmininde bu başarı oranını yakaladı."),
            "level": "positive",
        })

    # K-Means
    cluster_summary = km.get("cluster_summary", {})
    stable_group = cluster_summary.get("Stabil & Hızlı", [])
    if stable_group:
        insights.append({
            "icon": "📊",
            "title": "Sezonun Elit Grubu (K-Means)",
            "driver": ", ".join(stable_group[:3]),
            "value": f"{len(stable_group)} pilot",
            "text": (f"K-Means kümeleme analizi {len(stable_group)} pilotu 'Stabil & Hızlı' "
                     f"kategorisinde grupladı: {', '.join(stable_group)}. "
                     f"Bu pilotlar hem hız hem tutarlılıkta öne çıkıyor."),
            "level": "positive",
        })

    return insights


# ─────────────────────────────────────────────────────────────
# PİLOT BAZLI INSIGHT
# ─────────────────────────────────────────────────────────────

def _driver_insights(analysis: Dict, df: pd.DataFrame, race_name: str) -> List[Dict]:
    insights = []
    stab = analysis.get("driver_stability", {})
    data = stab.get("data", [])

    if len(data) < 3:
        return insights

    # Consistency outlier — en tutarsız
    worst = data[-1]
    insights.append({
        "icon": "⚠️",
        "title": "En Tutarsız Pilot",
        "driver": worst.get("Driver", "N/A"),
        "value": f"std: {worst.get('lap_time_std', 0):.3f}s",
        "text": (f"{worst.get('Driver')} bu yarışta en yüksek tur zamanı değişkenliğini "
                 f"yaşadı (std: {worst.get('lap_time_std', 0):.3f}s). "
                 f"SC, strateji değişikliği veya sürüş hatası etkili olmuş olabilir."),
        "level": "warning",
    })

    # Sektör dominansı
    sector_cols = ["Sector1Time", "Sector2Time", "Sector3Time"]
    avail = [c for c in sector_cols if c in df.columns]
    if avail and "Driver" in df.columns:
        clean = df.copy()
        for col in ["IsPitLap", "IsOutlier", "IsSCLap"]:
            if col in clean.columns:
                clean = clean[~clean[col]]

        for i, sec in enumerate(avail[:3], 1):
            sec_best = clean.groupby("Driver")[sec].mean().idxmin()
            sec_val  = clean.groupby("Driver")[sec].mean().min()
            insights.append({
                "icon": f"S{i}",
                "title": f"Sektör {i} Uzmanı",
                "driver": sec_best,
                "value": f"{sec_val:.3f}s ort.",
                "text": (f"{sec_best}, {race_name}'de Sektör {i}'de rakiplerine üstünlük sağladı "
                         f"(ortalama {sec_val:.3f}s). "
                         f"{'Bu sektördeki üstünlük pit window açısından kritik.' if i == 1 else ''}"),
                "level": "positive",
            })

    return insights


# ─────────────────────────────────────────────────────────────
# TOP 5 ÖZET (Sunum için)
# ─────────────────────────────────────────────────────────────

def _build_top5(insights: Dict, race_name: str) -> List[Dict]:
    """Dashboard Presentation Mode için en kritik 5 çıkarım."""
    top5 = []

    # 1. Performans
    perf = insights.get("performance_insights", [])
    if perf:
        top5.append(perf[0])

    # 2. Lastik
    tire = insights.get("tire_insights", [])
    if tire:
        top5.append(tire[0])

    # 3. Pit stop
    pit = insights.get("pit_insights", [])
    if pit:
        top5.append(pit[0])

    # 4. ML
    ml_ins = insights.get("ml_insights", [])
    if ml_ins:
        top5.append(ml_ins[0])

    # 5. Pist
    tc = insights.get("track_character", {})
    top5.append({
        "icon": "🏁",
        "title": "Pist Karakteri",
        "driver": race_name,
        "value": "Track Analysis",
        "text": tc.get("headline", "") + " " + tc.get("detail", ""),
        "level": "info",
    })

    return top5[:5]


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from src.utils import generate_sample_lap_data
    from src.preprocessing import preprocess_laps
    from src.feature_engineering import engineer_features
    from src.analysis import run_all_analyses
    from src.models import run_all_models

    df  = generate_sample_lap_data("Bahrain")
    df  = preprocess_laps(df, "Bahrain")
    df  = engineer_features(df, "Bahrain")
    an  = run_all_analyses(df, "Bahrain")
    ml  = run_all_models(df, "Bahrain")
    ins = generate_race_insights("Bahrain", df, an, ml)

    print(f"\n=== TOP 5 INSIGHTS: Bahrain ===")
    for i, item in enumerate(ins["top5_summary"], 1):
        print(f"\n{i}. {item['icon']} {item['title']} — {item['driver']}")
        print(f"   {item['value']}")
        print(f"   {item['text'][:100]}...")
