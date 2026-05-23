"""
src/report_generator.py
-----------------------
F1 Race Intelligence System - Otomatik Rapor Üretici
Analiz sonuçlarından Markdown ve TXT rapor üretir.
"""

import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from config import GENERATED_REPORTS_DIR, SEASON
from src.logger import get_logger
from src.utils import seconds_to_mmss

logger = get_logger("ReportGenerator")


# ─────────────────────────────────────────────────────────────
# ANA RAPOR FONKSİYONU
# ─────────────────────────────────────────────────────────────

def generate_race_report(race_name: str,
                          analysis_results: Dict[str, Any],
                          ml_results: Dict[str, Any]) -> Dict[str, Path]:
    """
    Analiz ve ML sonuçlarından tam yarış raporu üretir.
    Hem Markdown hem TXT formatında kaydeder.
    """
    logger.info(f"Rapor üretiliyor: {race_name}")
    GENERATED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    content = _build_report_content(race_name, analysis_results, ml_results)

    slug = race_name.lower().replace(" ", "_")
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Markdown
    md_path = GENERATED_REPORTS_DIR / f"{slug}_report_{ts}.md"
    md_path.write_text(content["markdown"], encoding="utf-8")

    # TXT
    txt_path = GENERATED_REPORTS_DIR / f"{slug}_report_{ts}.txt"
    txt_path.write_text(content["plain"], encoding="utf-8")

    logger.info(f"Raporlar kaydedildi: {md_path.name}, {txt_path.name}")
    return {"markdown": md_path, "txt": txt_path}


# ─────────────────────────────────────────────────────────────
# RAPOR İÇERİĞİ OLUŞTURMA
# ─────────────────────────────────────────────────────────────

def _build_report_content(race_name: str,
                           analysis: Dict[str, Any],
                           ml: Dict[str, Any]) -> Dict[str, str]:
    """Analiz sonuçlarını rapor metni olarak formatla."""
    now = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    overview = analysis.get("race_overview", {})
    stability = analysis.get("driver_stability", {})
    tire = analysis.get("tire_strategy", {})
    tire_deg = analysis.get("tire_degradation", {})
    pit = analysis.get("pit_stop", {})
    weather = analysis.get("weather_impact", {})
    kmeans = ml.get("kmeans", {})
    dt = ml.get("decision_tree", {})
    knn = ml.get("knn", {})
    comparison = ml.get("comparison", {})

    # ── Markdown ─────────────────────────────────────────────
    md = []
    md.append(f"# 🏎️ F1 Race Intelligence Report")
    md.append(f"## {race_name} Grand Prix — {SEASON} Sezonu")
    md.append(f"*Rapor tarihi: {now}*")
    md.append("")
    md.append("---")

    # Genel Bakış
    md.append("## 📊 Yarış Genel Bakış")
    md.append(f"| Metrik | Değer |")
    md.append(f"|--------|-------|")
    md.append(f"| Toplam Tur | {overview.get('total_laps', 'N/A')} |")
    md.append(f"| Toplam Pilot | {overview.get('total_drivers', 'N/A')} |")
    fl = overview.get('fastest_lap')
    md.append(f"| En Hızlı Tur | {seconds_to_mmss(fl) if fl else 'N/A'} |")
    md.append(f"| En Hızlı Pilot | {overview.get('fastest_driver', 'N/A')} |")
    md.append(f"| En İstikrarlı Pilot | {overview.get('most_stable_driver', 'N/A')} |")
    md.append(f"| Ortalama Hava Sıcaklığı | {overview.get('avg_air_temp', 'N/A')}°C |")
    md.append(f"| Ortalama Pist Sıcaklığı | {overview.get('avg_track_temp', 'N/A')}°C |")
    md.append("")

    # Pilot İstikrarı
    md.append("## 🏁 Pilot İstikrar Analizi")
    if stability.get("data"):
        md.append(f"**En İstikrarlı Pilot:** {stability.get('most_stable', 'N/A')}")
        md.append(f"**En Hızlı Pilot:** {stability.get('fastest', 'N/A')}")
        md.append("")
        md.append("| Sıra | Pilot | Ort. Tur | En Hızlı | Std Sapma | Consistency |")
        md.append("|------|-------|----------|----------|-----------|-------------|")
        for row in stability["data"][:10]:
            md.append(
                f"| {row.get('stability_rank','–')} "
                f"| {row.get('Driver','–')} "
                f"| {row.get('avg_lap_time', 0):.3f}s "
                f"| {row.get('best_lap_time', 0):.3f}s "
                f"| {row.get('lap_time_std', 0):.3f}s "
                f"| {row.get('consistency_score', 0):.4f} |"
            )
        md.append("")
    if stability.get("comment"):
        md.append(f"> {stability['comment']}")
    md.append("")

    # Lastik Stratejisi
    md.append("## 🔴 Lastik Stratejisi Analizi")
    if tire.get("compound_stats"):
        md.append(f"**En Avantajlı Lastik:** {tire.get('best_compound', 'N/A')}")
        md.append("")
        md.append("| Lastik | Ort. Tur (s) | En Hızlı (s) | Std Sapma | Tur Sayısı |")
        md.append("|--------|--------------|--------------|-----------|------------|")
        for row in tire["compound_stats"]:
            md.append(
                f"| {row.get('Compound','–')} "
                f"| {row.get('avg_lap_time',0):.3f} "
                f"| {row.get('best_lap_time',0):.3f} "
                f"| {row.get('std',0):.3f} "
                f"| {int(row.get('count',0))} |"
            )
        md.append("")
    if tire.get("comment"):
        md.append(f"> {tire['comment']}")
    md.append("")

    # Tire Degradation
    md.append("## 📉 Lastik Bozulma Analizi")
    if tire_deg.get("by_compound"):
        md.append(f"**En Hızlı Bozulan Lastik:** {tire_deg.get('worst_compound', 'N/A')}")
        md.append("")
        md.append("| Lastik | Bozulma Oranı (s/tur) | R² |")
        md.append("|--------|----------------------|-----|")
        for compound, info in tire_deg["by_compound"].items():
            rate = info.get("degradation_rate")
            r2   = info.get("r_squared")
            rate_str = f"{rate:.4f}" if (rate is not None and not (isinstance(rate, float) and rate != rate)) else "N/A"
            r2_str   = f"{r2:.3f}"   if (r2   is not None and not (isinstance(r2,   float) and r2   != r2  )) else "N/A"
            md.append(
                f"| {compound} "
                f"| {rate_str} "
                f"| {r2_str} |"
            )
        md.append("")
    if tire_deg.get("comment"):
        md.append(f"> {tire_deg['comment']}")
    md.append("")

    # Pit Stop
    md.append("## 🔧 Pit Stop Analizi")
    if pit.get("race_avg_impact") is not None:
        md.append(f"**Ortalama Pit Stop Faydası:** {pit['race_avg_impact']:.3f}s")
        md.append("")
        if pit.get("driver_analysis"):
            md.append("| Pilot | Pit Sayısı | Ortalama Etki (s) |")
            md.append("|-------|------------|-------------------|")
            for d in sorted(pit["driver_analysis"],
                            key=lambda x: x.get("avg_impact", 0), reverse=True):
                md.append(
                    f"| {d['Driver']} "
                    f"| {d['pit_count']} "
                    f"| {d['avg_impact']:.3f} |"
                )
            md.append("")
    if pit.get("comment"):
        md.append(f"> {pit['comment']}")
    md.append("")

    # Hava Etkisi
    md.append("## 🌡️ Hava ve Sıcaklık Analizi")
    for col, info in weather.items():
        if isinstance(info, dict) and "pearson_r" in info:
            sig = "✅ İstatistiksel olarak anlamlı" if info.get("significant") else "❌ Anlamlı değil"
            md.append(f"**{info.get('label', col)}**")
            md.append(f"- Pearson r = {info['pearson_r']}")
            md.append(f"- p-değeri = {info['p_value']} ({sig})")
            md.append(f"- {info.get('interpretation', '')}")
            md.append(f"> {info.get('comment', '')}")
            md.append("")

    # ML Sonuçları
    md.append("## 🤖 Makine Öğrenmesi Sonuçları")

    # K-Means
    md.append("### K-Means Clustering")
    if kmeans and not kmeans.get("error"):
        md.append(f"**Küme Sayısı:** {kmeans.get('n_clusters', 'N/A')}")
        for cluster, members in kmeans.get("cluster_summary", {}).items():
            md.append(f"- **{cluster}:** {', '.join(members)}")
        md.append("")
        if kmeans.get("comment"):
            md.append(f"> {kmeans['comment']}")
    else:
        md.append(f"> Hata: {kmeans.get('error', 'Bilinmeyen hata')}")
    md.append("")

    # Decision Tree
    md.append("### Decision Tree Sınıflandırma")
    if dt and not dt.get("error"):
        md.append(f"**Doğruluk:** %{dt.get('accuracy', 0)*100:.1f}")
        md.append("")
        if dt.get("feature_importance"):
            md.append("**Feature Importance:**")
            for feat, imp in dt["feature_importance"]:
                bar = "█" * int(imp * 20)
                md.append(f"- `{feat}`: {imp:.4f} {bar}")
        md.append("")
        md.append(f"**En Önemli Özellik:** `{dt.get('most_important_feature', 'N/A')}`")
        md.append("")
        if dt.get("comment"):
            md.append(f"> {dt['comment']}")
    md.append("")

    # kNN
    md.append("### kNN Sınıflandırma")
    if knn and not knn.get("error"):
        md.append(f"**Doğruluk:** %{knn.get('accuracy', 0)*100:.1f} (k={knn.get('k', 'N/A')})")
        if knn.get("comment"):
            md.append(f"> {knn['comment']}")
    md.append("")

    # Model karşılaştırması
    md.append("### Model Karşılaştırması")
    if comparison:
        md.append(f"| Model | Doğruluk |")
        md.append(f"|-------|----------|")
        md.append(f"| Decision Tree | %{comparison.get('decision_tree_accuracy', 0)*100:.1f} |")
        md.append(f"| kNN | %{comparison.get('knn_accuracy', 0)*100:.1f} |")
        md.append(f"| **Kazanan** | **{comparison.get('winner', 'N/A')}** |")
        md.append("")
        if comparison.get("comment"):
            md.append(f"> {comparison['comment']}")
    md.append("")

    # Sonuç
    md.append("---")
    md.append("## 📝 Genel Sonuç")
    md.append(_generate_conclusion(race_name, overview, stability, tire, dt, comparison))
    md.append("")
    md.append("---")
    md.append(f"*Bu rapor F1 Race Intelligence System tarafından otomatik üretilmiştir.*")
    md.append(f"*Veri kaynağı: FastF1 / {SEASON} Formula 1 Sezonu*")

    markdown_text = "\n".join(md)
    plain_text    = _md_to_plain(markdown_text)

    return {"markdown": markdown_text, "plain": plain_text}


def _generate_conclusion(race_name, overview, stability, tire, dt, comparison) -> str:
    """Analiz sonuçlarından otomatik genel sonuç paragrafı üret."""
    parts = []

    fastest = overview.get("fastest_driver", "N/A")
    stable  = overview.get("most_stable_driver", stability.get("most_stable", "N/A"))
    best_compound = tire.get("best_compound", "N/A")
    best_feature  = dt.get("most_important_feature", "N/A") if isinstance(dt, dict) else "N/A"
    winner_model  = comparison.get("winner", "N/A") if isinstance(comparison, dict) else "N/A"

    parts.append(
        f"{race_name} Grand Prix analizi tamamlandı. "
        f"Yarışın en hızlı pilotu **{fastest}** olurken, "
        f"en tutarlı performansı **{stable}** sergilemiştir."
    )
    if best_compound and best_compound != "N/A":
        parts.append(
            f"Lastik stratejisi açısından **{best_compound}** bileşiği bu pistte en avantajlı "
            "seçenek olarak öne çıkmıştır."
        )
    if best_feature and best_feature != "N/A":
        parts.append(
            f"Makine öğrenmesi modeli, yarış performansını etkileyen en kritik faktör olarak "
            f"**{best_feature}** özelliğini belirlemiştir."
        )
    if winner_model and winner_model != "N/A":
        parts.append(
            f"Sınıflandırma modelleri karşılaştırmasında **{winner_model}** daha yüksek "
            "doğruluk oranına ulaşmıştır."
        )

    return " ".join(parts)


def _md_to_plain(md_text: str) -> str:
    """Markdown'ı düz metne çevir (basit, kütüphanesiz)."""
    import re
    text = md_text
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\|(.*?)\|", lambda m: m.group(0), text)
    text = re.sub(r"^>+\s*", "  >> ", text, flags=re.MULTILINE)
    text = re.sub(r"---+", "─" * 50, text)
    return text


if __name__ == "__main__":
    # Minimal test
    dummy_analysis = {
        "race_overview": {
            "race_name": "Bahrain", "total_laps": 200, "total_drivers": 10,
            "fastest_lap": 91.5, "fastest_driver": "VER",
            "most_stable_driver": "HAM", "avg_air_temp": 28.5, "avg_track_temp": 42.0,
        },
        "driver_stability": {"data": [], "most_stable": "HAM", "fastest": "VER", "comment": "Test."},
        "tire_strategy": {"compound_stats": [], "best_compound": "MEDIUM", "comment": ""},
        "tire_degradation": {"by_compound": {}, "worst_compound": "SOFT", "comment": ""},
        "pit_stop": {"race_avg_impact": 1.5, "driver_analysis": [], "comment": ""},
        "weather_impact": {},
    }
    dummy_ml = {
        "kmeans": {"n_clusters": 3, "cluster_summary": {"Stabil": ["VER", "HAM"]}, "comment": ""},
        "decision_tree": {"accuracy": 0.8, "feature_importance": [("consistency_score", 0.45)],
                          "most_important_feature": "consistency_score", "comment": ""},
        "knn": {"accuracy": 0.75, "k": 5, "comment": ""},
        "comparison": {"decision_tree_accuracy": 0.8, "knn_accuracy": 0.75, "winner": "Decision Tree", "comment": ""},
    }

    paths = generate_race_report("Bahrain", dummy_analysis, dummy_ml)
    print("✅ Rapor oluşturuldu:")
    for fmt, path in paths.items():
        print(f"   {fmt}: {path}")
