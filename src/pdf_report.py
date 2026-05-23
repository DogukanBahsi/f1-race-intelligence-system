"""
src/pdf_report.py
-----------------
F1 Race Intelligence System - PDF Rapor Üretici
ReportLab ile profesyonel PDF raporu oluşturur.
"""

import io
import base64
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from config import GENERATED_REPORTS_DIR, SEASON
from src.logger import get_logger
from src.utils import seconds_to_mmss

logger = get_logger("PDFReport")

# F1 renk paleti
F1_RED   = (0.91, 0.0,  0.18)   # #E8002D
F1_BLACK = (0.05, 0.05, 0.05)
F1_GRAY  = (0.10, 0.10, 0.10)
F1_LIGHT = (0.85, 0.85, 0.85)
F1_WHITE = (1.0,  1.0,  1.0)
F1_GOLD  = (1.0,  0.71, 0.0)


def generate_pdf_report(race_name: str,
                         analysis: Dict[str, Any],
                         ml: Dict[str, Any],
                         insights: Dict[str, Any],
                         charts: Dict[str, str]) -> Optional[Path]:
    """
    Analiz sonuçlarından profesyonel PDF raporu üretir.
    Başarı durumunda Path döndürür, hata durumunda None.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm, mm
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                         TableStyle, HRFlowable, Image as RLImage,
                                         KeepTogether, PageBreak)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError:
        logger.error("reportlab yüklü değil. 'pip install reportlab' çalıştırın.")
        return None

    GENERATED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    slug     = race_name.lower().replace(" ", "_")
    pdf_path = GENERATED_REPORTS_DIR / f"{slug}_report_{ts}.pdf"

    # ── Stiller ──────────────────────────────────────────────
    from reportlab.lib.colors import Color, HexColor
    RED   = HexColor("#E8002D")
    DARK  = HexColor("#0D0D0D")
    DGRAY = HexColor("#1A1A1A")
    LGRAY = HexColor("#AAAAAA")
    WHITE = HexColor("#FFFFFF")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("Title",
        fontName="Helvetica-Bold", fontSize=28, textColor=RED,
        spaceAfter=6, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("Subtitle",
        fontName="Helvetica", fontSize=13, textColor=LGRAY,
        spaceAfter=2, alignment=TA_CENTER)
    h1_style = ParagraphStyle("H1",
        fontName="Helvetica-Bold", fontSize=16, textColor=RED,
        spaceBefore=16, spaceAfter=8)
    h2_style = ParagraphStyle("H2",
        fontName="Helvetica-Bold", fontSize=12, textColor=DARK,
        spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("Body",
        fontName="Helvetica", fontSize=10, textColor=DARK,
        spaceAfter=4, leading=14)
    small_style = ParagraphStyle("Small",
        fontName="Helvetica", fontSize=8, textColor=LGRAY,
        spaceAfter=2)
    mono_style = ParagraphStyle("Mono",
        fontName="Courier", fontSize=9, textColor=DARK,
        spaceAfter=3, leading=12)

    # ── İçerik listesi ───────────────────────────────────────
    content = []

    # ── Kapak ────────────────────────────────────────────────
    content.append(Spacer(1, 2*cm))
    content.append(Paragraph("🏎 F1 RACE INTELLIGENCE", title_style))
    content.append(Paragraph(f"{race_name} Grand Prix — {SEASON} Sezonu", subtitle_style))
    content.append(Spacer(1, 0.3*cm))
    content.append(HRFlowable(width="100%", thickness=2, color=RED))
    content.append(Spacer(1, 0.2*cm))

    now = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    content.append(Paragraph(f"Rapor tarihi: {now}", small_style))

    is_sim = insights.get("is_simulated", True)
    if is_sim:
        content.append(Spacer(1, 0.3*cm))
        sim_style = ParagraphStyle("SimWarn",
            fontName="Helvetica-Bold", fontSize=9,
            textColor=HexColor("#FFB300"), spaceAfter=4)
        content.append(Paragraph(
            "⚠ UYARI: Bu rapor simüle edilmiş veri üzerinde üretilmiştir. "
            "Gerçek veri için FastF1 ve internet bağlantısı gereklidir.",
            sim_style))

    content.append(Spacer(1, 1*cm))

    # ── Genel Bakış ───────────────────────────────────────────
    ov = analysis.get("race_overview", {})
    content.append(Paragraph("1. YARIŞ GENEL BAKIŞ", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
    content.append(Spacer(1, 0.3*cm))

    ov_data = [
        ["Metrik", "Değer"],
        ["Yarış", race_name],
        ["Sezon", str(SEASON)],
        ["Toplam Tur", str(ov.get("total_laps", "N/A"))],
        ["Toplam Pilot", str(ov.get("total_drivers", "N/A"))],
        ["En Hızlı Tur", seconds_to_mmss(ov.get("fastest_lap")) if ov.get("fastest_lap") else "N/A"],
        ["En Hızlı Pilot", ov.get("fastest_driver", "N/A")],
        ["En İstikrarlı Pilot", ov.get("most_stable_driver", "N/A")],
        ["Ort. Hava Sıcaklığı", f"{ov.get('avg_air_temp', 0):.1f}°C"],
        ["Ort. Pist Sıcaklığı", f"{ov.get('avg_track_temp', 0):.1f}°C"],
    ]

    ov_table = Table(ov_data, colWidths=[7*cm, 10*cm])
    ov_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), RED),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 10),
        ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#F8F8F8"), WHITE]),
        ("GRID",         (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    content.append(ov_table)
    content.append(Spacer(1, 0.5*cm))

    # ── Pilot İstikrar ────────────────────────────────────────
    stab = analysis.get("driver_stability", {})
    driver_data = stab.get("data", [])
    if driver_data:
        content.append(Paragraph("2. PİLOT İSTİKRAR ANALİZİ", h1_style))
        content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
        content.append(Spacer(1, 0.3*cm))

        if stab.get("comment"):
            content.append(Paragraph(stab["comment"], body_style))
            content.append(Spacer(1, 0.2*cm))

        drv_table_data = [["Sıra", "Pilot", "Ort.Tur(s)", "En Hızlı(s)", "Std Sapma", "Consistency"]]
        for i, row in enumerate(driver_data[:10]):
            drv_table_data.append([
                str(row.get("stability_rank", i+1)),
                row.get("Driver", ""),
                f"{row.get('avg_lap_time', 0):.3f}",
                f"{row.get('best_lap_time', 0):.3f}",
                f"{row.get('lap_time_std', 0):.3f}",
                f"{row.get('consistency_score', 0):.4f}",
            ])

        drv_table = Table(drv_table_data,
                          colWidths=[1.5*cm, 3*cm, 3*cm, 3*cm, 3*cm, 3.5*cm])
        drv_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), RED),
            ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#F8F8F8"), WHITE]),
            ("GRID",         (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
            ("ALIGN",        (1, 1), (-1, -1), "CENTER"),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        # İlk sırayı vurgula
        drv_table.setStyle(TableStyle([
            ("TEXTCOLOR",  (0, 1), (-1, 1), RED),
            ("FONTNAME",   (0, 1), (-1, 1), "Helvetica-Bold"),
        ]))
        content.append(drv_table)
        content.append(Spacer(1, 0.5*cm))

    # ── Grafik ekle (varsa) ──────────────────────────────────
    _add_chart_to_pdf(content, charts, "driver_avg_laptime",
                      "Pilot Ortalama Tur Süresi", body_style)
    _add_chart_to_pdf(content, charts, "consistency_score",
                      "Pilot Consistency Score", body_style)

    # ── Lastik Stratejisi ────────────────────────────────────
    tire = analysis.get("tire_strategy", {})
    compound_stats = tire.get("compound_stats", [])
    if compound_stats:
        content.append(Paragraph("3. LASTİK STRATEJİSİ ANALİZİ", h1_style))
        content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
        content.append(Spacer(1, 0.3*cm))

        if tire.get("comment"):
            content.append(Paragraph(tire["comment"], body_style))

        tire_data = [["Lastik", "Ort. Tur (s)", "En Hızlı (s)", "Std Sapma", "Tur Sayısı"]]
        for row in compound_stats:
            tire_data.append([
                row.get("Compound", ""),
                f"{row.get('avg_lap_time', 0):.3f}",
                f"{row.get('best_lap_time', 0):.3f}",
                f"{row.get('std', 0):.3f}",
                str(int(row.get("count", 0))),
            ])

        tire_table = Table(tire_data, colWidths=[3.5*cm, 4*cm, 4*cm, 3.5*cm, 3*cm])
        tire_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), RED),
            ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#F8F8F8"), WHITE]),
            ("GRID",       (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
            ("ALIGN",      (1, 1), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        content.append(tire_table)
        content.append(Spacer(1, 0.4*cm))

    _add_chart_to_pdf(content, charts, "tire_degradation",
                      "Lastik Bozulma Eğrisi", body_style)

    # ── Pit Stop ─────────────────────────────────────────────
    pit = analysis.get("pit_stop", {})
    content.append(Paragraph("4. PİT STOP ANALİZİ", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
    content.append(Spacer(1, 0.3*cm))

    avg_impact = pit.get("race_avg_impact")
    if avg_impact is not None:
        content.append(Paragraph(
            f"Ortalama pit stop faydası: {avg_impact:+.3f}s", h2_style))
        if pit.get("comment"):
            content.append(Paragraph(pit["comment"], body_style))
    else:
        content.append(Paragraph("Pit stop verisi mevcut değil.", body_style))
    content.append(Spacer(1, 0.3*cm))

    # ── ML Sonuçları ─────────────────────────────────────────
    dt   = ml.get("decision_tree", {})
    knn  = ml.get("knn", {})
    comp = ml.get("comparison", {})

    content.append(Paragraph("5. MAKİNE ÖĞRENMESİ SONUÇLARI", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
    content.append(Spacer(1, 0.3*cm))

    ml_data = [
        ["Model", "Doğruluk", "Not"],
        ["Decision Tree", f"%{dt.get('accuracy', 0)*100:.1f}",
         f"max_depth=5, {dt.get('most_important_feature', 'N/A')} en önemli"],
        ["kNN", f"%{knn.get('accuracy', 0)*100:.1f}", f"k={knn.get('k', 5)}"],
        ["Kazanan", comp.get("winner", "N/A"), "Daha yüksek doğruluk"],
    ]
    ml_table = Table(ml_data, colWidths=[4.5*cm, 3.5*cm, 10*cm])
    ml_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), RED),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#F8F8F8"), WHITE]),
        ("GRID",       (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    content.append(ml_table)
    content.append(Spacer(1, 0.3*cm))

    # Feature importance
    feat_imp = dt.get("feature_importance", [])
    if feat_imp:
        content.append(Paragraph("Feature Importance (Decision Tree):", h2_style))
        fi_data = [["Özellik", "Önem Skoru", "Görsel"]]
        max_imp = max(f[1] for f in feat_imp) if feat_imp else 1
        for feat, imp in feat_imp:
            bar = "█" * max(1, int(imp / max_imp * 20))
            fi_data.append([feat, f"{imp:.4f}", bar])
        fi_table = Table(fi_data, colWidths=[6*cm, 3.5*cm, 8.5*cm])
        fi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME",   (2, 1), (2, -1), "Courier"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("TEXTCOLOR",  (2, 1), (2, -1), RED),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#F8F8F8"), WHITE]),
            ("GRID",       (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        content.append(fi_table)
        content.append(Spacer(1, 0.3*cm))

    _add_chart_to_pdf(content, charts, "feature_importance",
                      "Feature Importance Görselleştirme", body_style)

    # ── AI Insights ──────────────────────────────────────────
    top5 = insights.get("top5_summary", [])
    if top5:
        content.append(Paragraph("6. AI INSIGHTS — EN KRİTİK 5 ÇIKARIM", h1_style))
        content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
        content.append(Spacer(1, 0.3*cm))
        for i, ins in enumerate(top5, 1):
            content.append(Paragraph(
                f"{i}. {ins.get('icon','')} <b>{ins.get('title','')}</b> "
                f"— {ins.get('driver','')} ({ins.get('value','')})",
                h2_style))
            content.append(Paragraph(ins.get("text", ""), body_style))
            content.append(Spacer(1, 0.2*cm))

    # ── K-Means ──────────────────────────────────────────────
    km = ml.get("kmeans", {})
    cluster_summary = km.get("cluster_summary", {})
    if cluster_summary:
        content.append(Paragraph("7. K-MEANS KÜMELEME ANALİZİ", h1_style))
        content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
        content.append(Spacer(1, 0.3*cm))

        km_data = [["Küme", "Pilot Sayısı", "Pilotlar"]]
        for cluster, members in cluster_summary.items():
            km_data.append([cluster, str(len(members)), ", ".join(members)])

        km_table = Table(km_data, colWidths=[5*cm, 3*cm, 10*cm])
        km_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#F8F8F8"), WHITE]),
            ("GRID",       (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ]))
        content.append(km_table)
        if km.get("comment"):
            content.append(Spacer(1, 0.2*cm))
            content.append(Paragraph(km["comment"], body_style))

    # ── Footer ───────────────────────────────────────────────
    content.append(Spacer(1, 1*cm))
    content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
    content.append(Spacer(1, 0.2*cm))
    content.append(Paragraph(
        f"F1 Race Intelligence System | FastF1 / {SEASON} Formula 1 Sezonu | "
        f"{'⚠ Simüle veri' if insights.get('is_simulated') else '✅ Gerçek FastF1 verisi'}",
        small_style))

    # ── PDF yaz ──────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    doc.build(content)
    logger.info(f"PDF raporu oluşturuldu: {pdf_path.name}")
    return pdf_path


def _add_chart_to_pdf(content, charts: Dict[str, str],
                       key: str, title: str, style) -> None:
    """Base64 grafiği PDF'e ekle."""
    from reportlab.platypus import Spacer, Paragraph, Image as RLImage
    from reportlab.lib.units import cm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.colors import HexColor

    b64 = charts.get(key, "")
    if not b64 or not b64.startswith("data:image"):
        return

    try:
        img_data = base64.b64decode(b64.split(",")[1])
        img_buf  = io.BytesIO(img_data)
        img      = RLImage(img_buf, width=16*cm, height=7*cm)
        content.append(Spacer(1, 0.3*cm))
        content.append(img)
        content.append(Spacer(1, 0.2*cm))
    except Exception as e:
        logger.warning(f"Grafik PDF'e eklenemedi ({key}): {e}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from src.utils import generate_sample_lap_data
    from src.preprocessing import preprocess_laps
    from src.feature_engineering import engineer_features
    from src.analysis import run_all_analyses
    from src.models import run_all_models
    from src.visualization import generate_all_charts
    from src.insight_engine import generate_race_insights

    df  = generate_sample_lap_data("Bahrain")
    df  = preprocess_laps(df, "Bahrain")
    df  = engineer_features(df, "Bahrain")
    an  = run_all_analyses(df, "Bahrain")
    ml  = run_all_models(df, "Bahrain")
    ch  = generate_all_charts(df, an, ml, "Bahrain")
    ins = generate_race_insights("Bahrain", df, an, ml)
    pdf = generate_pdf_report("Bahrain", an, ml, ins, ch)
    print("PDF:", pdf)
