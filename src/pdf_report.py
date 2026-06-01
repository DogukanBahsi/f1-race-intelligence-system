"""
src/pdf_report.py
-----------------
F1 Race Intelligence System - PDF Rapor Üretici
ReportLab + DejaVuSans (Unicode/Türkçe destekli) ile profesyonel PDF raporu.
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

# F1 renk sabitleri (tuple → HexColor'a çevirme fonksiyondan yapılacak)
_RED_HEX   = "#E8002D"
_DARK_HEX  = "#0D0D0D"
_DGRAY_HEX = "#1A1A1A"
_LGRAY_HEX = "#AAAAAA"
_WHITE_HEX = "#FFFFFF"
_GOLD_HEX  = "#B5860D"


# ─────────────────────────────────────────────────────────────
# FONT KURULUMU — DejaVuSans (matplotlib içinde gelir, Unicode)
# ─────────────────────────────────────────────────────────────

_FONT_NORMAL = "Helvetica"       # fallback
_FONT_BOLD   = "Helvetica-Bold"  # fallback
_FONT_MONO   = "Courier"         # monospace fallback
_UNICODE_OK  = False


def _setup_fonts() -> bool:
    """
    DejaVuSans fontunu ReportLab'a kaydeder.
    Başarılıysa True döner (Türkçe karakter desteği aktif).
    Başarısız olursa Helvetica fallback ile devam edilir.
    """
    global _FONT_NORMAL, _FONT_BOLD, _FONT_MONO, _UNICODE_OK
    try:
        import matplotlib
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        normal_path = font_dir / "DejaVuSans.ttf"
        bold_path   = font_dir / "DejaVuSans-Bold.ttf"
        mono_path   = font_dir / "DejaVuSansMono.ttf"

        if normal_path.exists():
            pdfmetrics.registerFont(TTFont("DejaVuSans",      str(normal_path)))
            pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(bold_path) if bold_path.exists() else str(normal_path)))
            if mono_path.exists():
                pdfmetrics.registerFont(TTFont("DejaVuSansMono", str(mono_path)))
                _FONT_MONO = "DejaVuSansMono"
            else:
                _FONT_MONO = "DejaVuSans"
            _FONT_NORMAL = "DejaVuSans"
            _FONT_BOLD   = "DejaVuSans-Bold"
            _UNICODE_OK  = True
            logger.info("DejaVuSans fontu yüklendi — Türkçe karakter desteği aktif.")
            return True

        # DejaVuSans bulunamadı — Windows sistem fontunu dene
        import os
        win_fonts = [
            Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts" / "arial.ttf",
            Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts" / "calibri.ttf",
        ]
        for wf in win_fonts:
            if wf.exists():
                pdfmetrics.registerFont(TTFont("SystemFont",     str(wf)))
                pdfmetrics.registerFont(TTFont("SystemFont-Bold", str(wf)))
                _FONT_NORMAL = "SystemFont"
                _FONT_BOLD   = "SystemFont-Bold"
                _UNICODE_OK  = True
                logger.info(f"Sistem fontu yüklendi: {wf.name}")
                return True

    except Exception as e:
        logger.warning(f"Font yükleme hatası: {e} — Helvetica fallback kullanılacak.")

    logger.warning("Unicode font bulunamadı — Türkçe karakterler için ASCII dönüşümü uygulanacak.")
    return False


def _safe(text: str) -> str:
    """
    Unicode font yoksa Türkçe özel karakterleri ASCII karşılıklarına dönüştürür.
    Unicode font varsa dokunmaz.
    """
    if _UNICODE_OK or text is None:
        return str(text) if text is not None else ""
    tr_map = str.maketrans(
        "çğıöşüÇĞİÖŞÜ",
        "cgiosüCGIOSU"
    )
    return str(text).translate(tr_map)


def _s(text) -> str:
    """str + _safe kombinasyonu."""
    return _safe(str(text) if text is not None else "")


def generate_pdf_report(race_name: str,
                        analysis:  Dict[str, Any],
                        ml:        Dict[str, Any],
                        insights:  Dict[str, Any],
                        charts:    Dict[str, str]) -> Optional[Path]:
    """
    Analiz sonuçlarından profesyonel PDF raporu üretir.
    Başarı durumunda Path döndürür, hata durumunda None.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, HRFlowable, KeepTogether, PageBreak,
        )
    except ImportError:
        logger.error("reportlab yüklü değil. 'pip install reportlab' çalıştırın.")
        return None

    # Font kur
    _setup_fonts()

    GENERATED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    slug     = race_name.lower().replace(" ", "_")
    pdf_path = GENERATED_REPORTS_DIR / f"{slug}_report_{ts}.pdf"

    # Renkler
    RED   = HexColor(_RED_HEX)
    DARK  = HexColor(_DARK_HEX)
    LGRAY = HexColor(_LGRAY_HEX)
    WHITE = HexColor(_WHITE_HEX)
    GOLD  = HexColor(_GOLD_HEX)
    ROW1  = HexColor("#F5F5F5")

    fn  = _FONT_NORMAL
    fb  = _FONT_BOLD
    fm  = _FONT_MONO

    # ── Stil tanımları ────────────────────────────────────────
    title_style = ParagraphStyle("FTitle",
        fontName=fb, fontSize=26, textColor=RED,
        spaceAfter=6, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("FSubtitle",
        fontName=fn, fontSize=13, textColor=LGRAY,
        spaceAfter=2, alignment=TA_CENTER)
    h1_style = ParagraphStyle("FH1",
        fontName=fb, fontSize=15, textColor=RED,
        spaceBefore=18, spaceAfter=8)
    h2_style = ParagraphStyle("FH2",
        fontName=fb, fontSize=11, textColor=DARK,
        spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle("FBody",
        fontName=fn, fontSize=10, textColor=DARK,
        spaceAfter=4, leading=15)
    small_style = ParagraphStyle("FSmall",
        fontName=fn, fontSize=8, textColor=LGRAY,
        spaceAfter=2)
    mono_style = ParagraphStyle("FMono",
        fontName=fm, fontSize=8, textColor=DARK,
        spaceAfter=3, leading=11)
    warn_style = ParagraphStyle("FWarn",
        fontName=fb, fontSize=9,
        textColor=HexColor("#FFB300"), spaceAfter=4)

    def tbl_style_base(header_bg=RED):
        return TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), fb),
            ("FONTSIZE",      (0, 0), (-1,  0), 9),
            ("FONTNAME",      (0, 1), (-1, -1), fn),
            ("FONTSIZE",      (0, 1), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [ROW1, WHITE]),
            ("GRID",          (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
            ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ])

    # ─────────────────────────────────────────────────────────
    # İÇERİK OLUŞTURMA
    # ─────────────────────────────────────────────────────────
    content = []

    # ── 1. KAPAK ─────────────────────────────────────────────
    content.append(Spacer(1, 2.5 * cm))
    content.append(Paragraph("F1 RACE INTELLIGENCE SYSTEM", title_style))
    content.append(Paragraph(
        f"{_s(race_name)} Grand Prix — {SEASON} Sezonu", subtitle_style))
    content.append(Spacer(1, 0.4 * cm))
    content.append(HRFlowable(width="100%", thickness=3, color=RED))
    content.append(Spacer(1, 0.3 * cm))

    now = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    content.append(Paragraph(f"Rapor tarihi: {now}", small_style))
    content.append(Paragraph("BLM308 Veri Madenciligi — F1 Race Intelligence System", small_style))

    is_sim = insights.get("is_simulated", True)
    if is_sim:
        content.append(Spacer(1, 0.3 * cm))
        content.append(Paragraph(
            "UYARI: Bu rapor simule edilmis veri uzerinde uretilmistir. "
            "Gercek veri icin FastF1 ve internet baglantisi gereklidir.",
            warn_style))

    content.append(Spacer(1, 1.2 * cm))

    # ── 2. YARIŞ GENEL BAKIŞ ─────────────────────────────────
    ov = analysis.get("race_overview", {})
    content.append(Paragraph("1. YARIS GENEL BAKIS", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
    content.append(Spacer(1, 0.3 * cm))

    ov_rows = [
        [_s("Metrik"),               _s("Deger")],
        [_s("Yaris"),                _s(race_name)],
        [_s("Sezon"),                _s(SEASON)],
        [_s("Toplam Tur"),           _s(ov.get("total_laps", "N/A"))],
        [_s("Toplam Pilot"),         _s(ov.get("total_drivers", "N/A"))],
        [_s("En Hizli Tur"),         _s(seconds_to_mmss(ov.get("fastest_lap")) if ov.get("fastest_lap") else "N/A")],
        [_s("En Hizli Pilot"),       _s(ov.get("fastest_driver", "N/A"))],
        [_s("En Istikrarli Pilot"),  _s(ov.get("most_stable_driver", "N/A"))],
        [_s("Ort. Hava Sicakligi"),  f"{ov.get('avg_air_temp', 0):.1f} C"],
        [_s("Ort. Pist Sicakligi"),  f"{ov.get('avg_track_temp', 0):.1f} C"],
    ]
    ov_tbl = Table(ov_rows, colWidths=[7 * cm, 10 * cm])
    ov_tbl.setStyle(tbl_style_base())
    content.append(ov_tbl)
    content.append(Spacer(1, 0.6 * cm))

    # ── 3. PİLOT İSTİKRAR ANALİZİ ───────────────────────────
    stab       = analysis.get("driver_stability", {})
    driver_data = stab.get("data", [])
    if driver_data:
        content.append(Paragraph("2. PILOT ISTIKRAR ANALIZI", h1_style))
        content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
        content.append(Spacer(1, 0.3 * cm))

        if stab.get("comment"):
            content.append(Paragraph(_s(stab["comment"]), body_style))
            content.append(Spacer(1, 0.2 * cm))

        drv_rows = [[_s("Sira"), _s("Pilot"), _s("Ort.Tur(s)"),
                     _s("En Hizli(s)"), _s("Std Sapma"), _s("Consistency")]]
        for i, row in enumerate(driver_data[:12]):
            drv_rows.append([
                _s(row.get("stability_rank", i + 1)),
                _s(row.get("Driver", "")),
                f"{row.get('avg_lap_time', 0):.3f}",
                f"{row.get('best_lap_time', 0):.3f}",
                f"{row.get('lap_time_std', 0):.3f}",
                f"{row.get('consistency_score', 0):.4f}",
            ])

        drv_tbl = Table(drv_rows, colWidths=[1.5*cm, 3*cm, 3*cm, 3*cm, 3*cm, 3.5*cm])
        drv_tbl.setStyle(tbl_style_base())
        # İlk sırayı vurgula
        drv_tbl.setStyle(TableStyle([
            ("TEXTCOLOR", (0, 1), (-1, 1), RED),
            ("FONTNAME",  (0, 1), (-1, 1), fb),
        ]))
        content.append(drv_tbl)
        content.append(Spacer(1, 0.5 * cm))

    _add_chart(content, charts, "driver_avg_laptime",
               _s("Pilot Ortalama Tur Suresi"), body_style)
    _add_chart(content, charts, "consistency_score",
               _s("Pilot Consistency Score"), body_style)

    # ── 4. LASTİK STRATEJİSİ ─────────────────────────────────
    tire          = analysis.get("tire_strategy", {})
    compound_stats = tire.get("compound_stats", [])
    if compound_stats:
        content.append(Paragraph("3. LASTIK STRATEJISI ANALIZI", h1_style))
        content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
        content.append(Spacer(1, 0.3 * cm))
        if tire.get("comment"):
            content.append(Paragraph(_s(tire["comment"]), body_style))
            content.append(Spacer(1, 0.2 * cm))

        tire_rows = [[_s("Lastik"), _s("Ort. Tur (s)"),
                      _s("En Hizli (s)"), _s("Std Sapma"), _s("Tur Sayisi")]]
        for row in compound_stats:
            tire_rows.append([
                _s(row.get("Compound", "")),
                f"{row.get('avg_lap_time', 0):.3f}",
                f"{row.get('best_lap_time', 0):.3f}",
                f"{row.get('std', 0):.3f}",
                str(int(row.get("count", 0))),
            ])

        tire_tbl = Table(tire_rows, colWidths=[3.5*cm, 4*cm, 4*cm, 3.5*cm, 3*cm])
        tire_tbl.setStyle(tbl_style_base())
        content.append(tire_tbl)
        content.append(Spacer(1, 0.4 * cm))

    _add_chart(content, charts, "tire_degradation",
               _s("Lastik Bozulma Egrisi"), body_style)

    # ── 5. PİT STOP ──────────────────────────────────────────
    pit = analysis.get("pit_stop", {})
    content.append(Paragraph("4. PIT STOP ANALIZI", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
    content.append(Spacer(1, 0.3 * cm))

    avg_impact = pit.get("race_avg_impact")
    if avg_impact is not None:
        content.append(Paragraph(
            f"Ortalama pit stop faydasi: {avg_impact:+.3f}s", h2_style))
        if pit.get("comment"):
            content.append(Paragraph(_s(pit["comment"]), body_style))
    else:
        content.append(Paragraph(_s("Pit stop verisi mevcut degil."), body_style))
    content.append(Spacer(1, 0.3 * cm))

    # ── 6. MAKİNE ÖĞRENMESİ SONUÇLARI ───────────────────────
    dt   = ml.get("decision_tree", {})
    knn  = ml.get("knn", {})
    comp = ml.get("comparison", {})
    ev   = ml.get("evaluation", {})

    content.append(PageBreak())
    content.append(Paragraph("5. MAKINE OGRENMESI SONUCLARI", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
    content.append(Spacer(1, 0.3 * cm))

    ml_rows = [
        [_s("Model"), _s("Train Dogruluk"), _s("Test Dogruluk"), _s("Not")],
        [_s("Decision Tree"),
         f"%{dt.get('train_accuracy', dt.get('accuracy', 0)) * 100:.1f}",
         f"%{dt.get('accuracy', 0) * 100:.1f}",
         _s(f"max_depth=5, {dt.get('most_important_feature', 'N/A')} en onemli")],
        [_s("kNN"),
         f"%{knn.get('train_accuracy', knn.get('accuracy', 0)) * 100:.1f}",
         f"%{knn.get('accuracy', 0) * 100:.1f}",
         _s(f"k={knn.get('k', 5)}")],
        [_s("Kazanan"), "", _s(comp.get("winner", "N/A")),
         _s("Daha yuksek test dogrulugu")],
    ]
    ml_tbl = Table(ml_rows, colWidths=[4*cm, 3.5*cm, 3.5*cm, 7*cm])
    ml_tbl.setStyle(tbl_style_base())
    content.append(ml_tbl)
    content.append(Spacer(1, 0.4 * cm))

    # Feature importance
    feat_imp = dt.get("feature_importance", [])
    if feat_imp:
        content.append(Paragraph(_s("Feature Importance (Decision Tree):"), h2_style))
        fi_rows = [[_s("Ozellik"), _s("Onem Skoru"), _s("Gorsel")]]
        max_imp  = max(f[1] for f in feat_imp) if feat_imp else 1.0
        for feat, imp in feat_imp[:8]:
            bar_len = max(1, int(imp / max_imp * 15))
            fi_rows.append([_s(feat), f"{imp:.4f}", "|" * bar_len])
        fi_tbl = Table(fi_rows, colWidths=[6.5*cm, 3.5*cm, 8*cm])
        fi_tbl.setStyle(tbl_style_base(header_bg=DARK))
        fi_tbl.setStyle(TableStyle([("TEXTCOLOR", (2, 1), (2, -1), RED)]))
        content.append(fi_tbl)
        content.append(Spacer(1, 0.4 * cm))

    # ── 7. AKADEMİK DEĞERLENDİRME ───────────────────────────
    content.append(Paragraph("6. AKADEMIK DEGERLENDIRME", h1_style))
    content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
    content.append(Spacer(1, 0.3 * cm))

    # 6a. Cross Validation
    cv_res = ev.get("cross_validation", {})
    if cv_res and not ev.get("error"):
        content.append(Paragraph(_s("6.1 Stratified K-Fold Capraz Dogrulama"), h2_style))
        dt_cv  = cv_res.get("decision_tree", {})
        knn_cv = cv_res.get("knn", {})
        n_f    = cv_res.get("n_folds", 3)

        cv_rows = [
            [_s("Model"), _s(f"CV Acc ({n_f}-fold)"), _s("Std"),
             _s("Precision"), _s("Recall"), _s("F1-macro"), _s("Overfit Gap")],
            [_s("Decision Tree"),
             f"{dt_cv.get('mean_accuracy', 0):.3f}",
             f"+-{dt_cv.get('std_accuracy', 0):.3f}",
             f"{dt_cv.get('mean_precision', 0):.3f}",
             f"{dt_cv.get('mean_recall', 0):.3f}",
             f"{dt_cv.get('mean_f1', 0):.3f}",
             f"{dt_cv.get('overfit_gap', 0):.3f}"],
            [_s("kNN"),
             f"{knn_cv.get('mean_accuracy', 0):.3f}",
             f"+-{knn_cv.get('std_accuracy', 0):.3f}",
             f"{knn_cv.get('mean_precision', 0):.3f}",
             f"{knn_cv.get('mean_recall', 0):.3f}",
             f"{knn_cv.get('mean_f1', 0):.3f}",
             f"{knn_cv.get('overfit_gap', 0):.3f}"],
        ]
        cv_tbl = Table(cv_rows, colWidths=[3.5*cm, 2.7*cm, 1.8*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        cv_tbl.setStyle(tbl_style_base())
        content.append(cv_tbl)
        content.append(Spacer(1, 0.4 * cm))

    # 6b. GridSearch
    gs_res = ev.get("grid_search", {})
    if gs_res and not ev.get("error"):
        content.append(Paragraph(_s("6.2 GridSearchCV Hiperparametre Optimizasyonu"), h2_style))
        dt_gs  = gs_res.get("decision_tree", {})
        knn_gs = gs_res.get("knn", {})
        gs_rows = [
            [_s("Model"), _s("Best Score"), _s("En Iyi Parametreler")],
            [_s("Decision Tree"),
             f"{dt_gs.get('best_score', 0):.3f}",
             _s(str(dt_gs.get("best_params", {})))[:80]],
            [_s("kNN"),
             f"{knn_gs.get('best_score', 0):.3f}",
             _s(str(knn_gs.get("best_params", {})))[:80]],
        ]
        gs_tbl = Table(gs_rows, colWidths=[4*cm, 3*cm, 11*cm])
        gs_tbl.setStyle(tbl_style_base())
        content.append(gs_tbl)
        content.append(Spacer(1, 0.4 * cm))

    # 6c. ROC AUC
    roc_res = ev.get("roc_auc", {})
    if roc_res and not ev.get("error"):
        content.append(Paragraph("6.3 ROC / AUC Analizi", h2_style))
        roc_rows = [[_s("Model"), "AUC", _s("Optimal Esik"), _s("Yontem")]]
        for model_name, roc_d in roc_res.items():
            if isinstance(roc_d, dict):
                roc_rows.append([
                    _s(model_name),
                    f"{roc_d.get('auc', 0):.4f}",
                    f"{roc_d.get('optimal_threshold', 0):.3f}",
                    _s(roc_d.get("method", "cross_val_predict")),
                ])
        roc_tbl = Table(roc_rows, colWidths=[5*cm, 3*cm, 4*cm, 6*cm])
        roc_tbl.setStyle(tbl_style_base())
        content.append(roc_tbl)
        content.append(Spacer(1, 0.4 * cm))

    # 6d. İstatistiksel Testler
    stat_tests = ml.get("statistical_tests", {})
    if stat_tests and not stat_tests.get("error"):
        content.append(Paragraph(_s("6.4 Istatistiksel Hipotez Testleri (alfa=0.05)"), h2_style))
        mc = stat_tests.get("mcnemar", {})
        tt = stat_tests.get("paired_ttest", {})
        wx = stat_tests.get("wilcoxon", {})

        st_rows = [
            [_s("Test"), _s("Istatistik"), "p-degeri", _s("H0 Karari"), _s("Yorum")],
            [_s("McNemar"),
             f"chi2={mc.get('chi2_stat', 0):.3f}",
             f"{mc.get('p_value', 1):.4f}",
             _s("Reddedilemez" if not mc.get("significant") else "Reddedildi"),
             _s(mc.get("conclusion", "—")[:50] if mc.get("conclusion") else "—")],
            [_s("Paired t-test"),
             f"t={tt.get('t_stat', 0):.3f}",
             f"{tt.get('p_value', 1):.4f}",
             _s("Reddedilemez" if not tt.get("significant") else "Reddedildi"),
             _s(tt.get("conclusion", "—")[:50] if tt.get("conclusion") else "—")],
            [_s("Wilcoxon"),
             f"W={wx.get('statistic', 0):.3f}",
             f"{wx.get('p_value', 1):.4f}",
             _s("Reddedilemez" if not wx.get("significant") else "Reddedildi"),
             _s(wx.get("conclusion", "—")[:50] if wx.get("conclusion") else "—")],
        ]
        st_tbl = Table(st_rows, colWidths=[3.5*cm, 3.5*cm, 2.5*cm, 3.5*cm, 5*cm])
        st_tbl.setStyle(tbl_style_base())
        content.append(st_tbl)
        content.append(Spacer(1, 0.4 * cm))

    # 6e. K-Means Silhouette
    ev_km = ev.get("kmeans_optimization", {})
    if ev_km and not ev.get("error"):
        best_k   = ev_km.get("best_n_clusters", "N/A")
        best_sil = ev_km.get("best_silhouette", 0)
        content.append(Paragraph(
            f"6.5 K-Means Optimizasyonu: Optimal k={best_k}, Silhouette={best_sil:.3f}",
            h2_style))
        content.append(Spacer(1, 0.2 * cm))

    # ── 8. AI INSIGHTS ───────────────────────────────────────
    top5 = insights.get("top5_summary", [])
    if top5:
        content.append(Paragraph("7. AI INSIGHTS — EN KRITIK 5 CIKARIM", h1_style))
        content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
        content.append(Spacer(1, 0.3 * cm))
        ins_rows = [[_s("No"), _s("Baslik"), _s("Pilot"), _s("Deger"), _s("Aciklama")]]
        for i, ins in enumerate(top5, 1):
            ins_rows.append([
                str(i),
                _s(ins.get("title", ""))[:30],
                _s(ins.get("driver", "")),
                _s(ins.get("value", ""))[:20],
                _s(ins.get("text",  ""))[:60],
            ])
        ins_tbl = Table(ins_rows, colWidths=[1*cm, 4.5*cm, 2.5*cm, 3*cm, 7*cm])
        ins_tbl.setStyle(tbl_style_base())
        content.append(ins_tbl)
        content.append(Spacer(1, 0.4 * cm))

    # ── 9. K-MEANS KÜMELEME ───────────────────────────────────
    km = ml.get("kmeans", {})
    cluster_summary = km.get("cluster_summary", {})
    if cluster_summary:
        content.append(Paragraph("8. K-MEANS KUMELEME ANALIZI", h1_style))
        content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
        content.append(Spacer(1, 0.3 * cm))

        km_rows = [[_s("Kume"), _s("Pilot Sayisi"), _s("Pilotlar")]]
        for cluster, members in cluster_summary.items():
            km_rows.append([
                _s(cluster),
                str(len(members)),
                _s(", ".join(members)),
            ])
        km_tbl = Table(km_rows, colWidths=[5*cm, 3*cm, 10*cm])
        km_tbl.setStyle(tbl_style_base(header_bg=DARK))
        content.append(km_tbl)
        if km.get("comment"):
            content.append(Spacer(1, 0.2 * cm))
            content.append(Paragraph(_s(km["comment"]), body_style))
        content.append(Spacer(1, 0.4 * cm))

    # ── 10. GRAFİKLER ────────────────────────────────────────
    chart_keys = [
        ("kmeans_clusters",    _s("K-Means Kume Dagilimi")),
        ("feature_importance", _s("Feature Importance")),
        ("model_comparison",   _s("Model Karsilastirmasi")),
        ("elbow_curve",        _s("Elbow + Silhouette Egrisi")),
        ("temp_vs_laptime",    _s("Sicaklik - Tur Zamani Iliskisi")),
    ]
    for key, title in chart_keys:
        _add_chart(content, charts, key, title, body_style)

    # ── FOOTER ───────────────────────────────────────────────
    content.append(Spacer(1, 1 * cm))
    content.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
    content.append(Spacer(1, 0.2 * cm))
    data_src = _s("Simule veri") if insights.get("is_simulated") else _s("Gercek FastF1 verisi")
    content.append(Paragraph(
        f"F1 Race Intelligence System  |  FastF1 {SEASON} Sezonu  |  {data_src}",
        small_style))
    content.append(Paragraph(
        f"BLM308 Veri Madenciligi — Ramazan Dogukan Bahsi  |  {now}",
        small_style))

    # ── PDF YAZ ──────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm,   bottomMargin=2 * cm,
    )

    try:
        doc.build(content)
        logger.info(f"PDF raporu olusturuldu: {pdf_path.name}")
        return pdf_path
    except Exception as e:
        logger.error(f"PDF build hatasi: {e}", exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────
# YARDIMCI: BASE64 GRAFİĞİ PDF'E EKLE
# ─────────────────────────────────────────────────────────────

def _add_chart(content: list, charts: Dict[str, str],
               key: str, title: str, style) -> None:
    """Base64 grafiği PDF'e ekler."""
    try:
        from reportlab.platypus import Spacer, Image as RLImage
        from reportlab.lib.units import cm

        b64 = charts.get(key, "")
        if not b64 or not b64.startswith("data:image"):
            return

        img_data = base64.b64decode(b64.split(",")[1])
        img_buf  = io.BytesIO(img_data)
        img      = RLImage(img_buf, width=16 * cm, height=7 * cm)
        content.append(Spacer(1, 0.3 * cm))
        content.append(img)
        content.append(Spacer(1, 0.2 * cm))
    except Exception as e:
        logger.warning(f"Grafik PDF'e eklenemedi ({key}): {e}")


# ─────────────────────────────────────────────────────────────
# DOĞRUDAN ÇALIŞTIRMA TESTİ
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
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
