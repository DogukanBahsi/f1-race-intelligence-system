"""
src/statistical_tests.py
------------------------
F1 Race Intelligence System - İstatistiksel Test Modulu
McNemar Testi ve Paired t-test ile model karsilastirmasi.
"""

import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel, chi2, wilcoxon
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from config import (
    DECISION_TREE_CONFIG, KNN_CONFIG,
    FIGURES_EVAL_DIR, TABLES_DIR, RANDOM_STATE, CV_CONFIG,
)
from src.logger import get_logger, log_success

logger = get_logger("StatisticalTests")

ALPHA = 0.05  # Anlamlilik esigi

CLF_FEATURES = [
    "average_lap_time",
    "tire_degradation_rate",
    "consistency_score",
    "pit_stop_impact",
    "sector_consistency",
]


# =============================================================
# ANA GIRIS NOKTASI
# =============================================================

def run_statistical_tests(
    feature_matrix: pd.DataFrame,
    df: pd.DataFrame,
    race_name: str,
) -> Dict[str, Any]:
    """
    Iki model arasinda istatistiksel anlamlilik testleri:
    1. McNemar Testi  — tahmin hatasi karsilastirmasi
    2. Paired t-test  — CV skor karsilastirmasi
    3. Wilcoxon Test  — CV skor karsilastirmasi (parametrik olmayan)
    """
    logger.info(f"Istatistiksel testler basliyor: {race_name}")
    FIGURES_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = _prepare_data(feature_matrix, df)
    if X is None or len(np.unique(y)) < 2:
        return {"error": "Istatistiksel testler icin yeterli/dengeli veri yok."}

    results: Dict[str, Any] = {"race_name": race_name, "alpha": ALPHA}

    # CV tahminleri
    n_splits = min(CV_CONFIG["n_splits"], _max_safe_folds(y))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    dt = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        **{k: v for k, v in DECISION_TREE_CONFIG.items() if k != "random_state"}
    )
    knn = KNeighborsClassifier(
        n_neighbors=min(KNN_CONFIG["n_neighbors"], len(X) - 1),
        metric=KNN_CONFIG["metric"],
    )

    # Out-of-fold tahminler (McNemar icin)
    dt_pred  = cross_val_predict(dt,  X, y, cv=cv)
    knn_pred = cross_val_predict(knn, X, y, cv=cv)

    # Fold bazli CV skorlar (t-test / Wilcoxon icin)
    dt_scores  = cross_val_score(dt,  X, y, cv=cv, scoring="accuracy")
    knn_scores = cross_val_score(knn, X, y, cv=cv, scoring="accuracy")

    # 1. McNemar Testi
    results["mcnemar"] = _mcnemar_test(y, dt_pred, knn_pred)

    # 2. Paired t-test
    results["paired_ttest"] = _paired_t_test(dt_scores, knn_scores)

    # 3. Wilcoxon (kucuk orneklem icin parametrik olmayan alternatif)
    results["wilcoxon"] = _wilcoxon_test(dt_scores, knn_scores)

    # 4. Genel yorum
    results["interpretation"] = _interpret_tests(
        results["mcnemar"],
        results["paired_ttest"],
        results["wilcoxon"],
    )

    # 5. Gorsel
    results["chart_b64"] = _plot_test_results(
        dt_scores, knn_scores,
        results["mcnemar"],
        results["paired_ttest"],
        race_name,
    )

    # 6. CSV kaydet
    _save_test_csv(results, race_name)

    log_success(
        f"Istatistiksel testler tamamlandi — "
        f"McNemar p={results['mcnemar']['p_value']:.4f} | "
        f"t-test p={results['paired_ttest']['p_value']:.4f}"
    )
    return results


# =============================================================
# 1. McNEMAR TESTİ
# =============================================================

def _mcnemar_test(
    y_true: np.ndarray,
    y_pred_dt: np.ndarray,
    y_pred_knn: np.ndarray,
) -> Dict[str, Any]:
    """
    McNemar Testi: Iki siniflandirici arasindaki hata farki anlamli mi?

    Olumsallık matrisi:
    - b: DT dogru, kNN yanlis
    - c: DT yanlis, kNN dogru

    H0: Her iki modelin hata dagilimi aynidir (b == c)
    H1: Hata dagilimlari farklidir
    """
    dt_correct  = (y_pred_dt  == y_true)
    knn_correct = (y_pred_knn == y_true)

    b = int(( dt_correct & ~knn_correct).sum())  # DT dogru, kNN yanlis
    c = int((~dt_correct &  knn_correct).sum())  # DT yanlis, kNN dogru

    n_discordant = b + c

    if n_discordant == 0:
        # Iki model ayni tahminleri yapti
        return {
            "b": 0, "c": 0,
            "chi2_stat": 0.0,
            "p_value":   1.0,
            "significant": False,
            "conclusion": "Iki model tamamen ayni tahminleri yapti (McNemar testi uygulanamaz).",
            "note": "b=0, c=0 — discordant pair yok.",
        }

    # Sureklilik duzeltmeli McNemar istatistigi
    chi2_stat = float((abs(b - c) - 1.0) ** 2 / n_discordant)
    p_value   = float(1.0 - chi2.cdf(chi2_stat, df=1))
    significant = p_value < ALPHA

    better = "Decision Tree" if b > c else ("kNN" if c > b else "Esit")

    return {
        "b": b, "c": c,
        "n_discordant": n_discordant,
        "chi2_stat":    round(chi2_stat, 4),
        "p_value":      round(p_value, 4),
        "significant":  significant,
        "better_model": better,
        "conclusion": (
            f"McNemar testi: χ²={chi2_stat:.4f}, p={p_value:.4f}. "
            f"{'H0 reddedildi — modeller arasinda anlamli fark var' if significant else 'H0 reddedilemedi — modeller arasinda anlamli fark yok'} "
            f"(α={ALPHA}). "
            + (f"'{better}' daha az hata yapıyor." if better != "Esit" else "Esit performans.")
        ),
    }


# =============================================================
# 2. PAIRED T-TEST
# =============================================================

def _paired_t_test(
    scores_dt: np.ndarray,
    scores_knn: np.ndarray,
) -> Dict[str, Any]:
    """
    Paired t-test: CV fold skorlari arasindaki fark anlamli mi?

    H0: iki modelin ortalama CV skoru esittir
    H1: ortalamalar farklidir (iki tarafli)
    """
    if len(scores_dt) < 2:
        return {
            "t_stat": 0.0, "p_value": 1.0, "significant": False,
            "conclusion": "t-test icin yeterli fold sayisi yok.",
        }

    t_stat, p_value = ttest_rel(scores_dt, scores_knn)
    significant = float(p_value) < ALPHA

    diff = float(np.mean(scores_dt) - np.mean(scores_knn))
    better = "Decision Tree" if diff > 0 else ("kNN" if diff < 0 else "Esit")

    return {
        "t_stat":      round(float(t_stat),    4),
        "p_value":     round(float(p_value),   4),
        "mean_diff":   round(diff,             4),
        "dt_mean":     round(float(scores_dt.mean()),  4),
        "knn_mean":    round(float(scores_knn.mean()), 4),
        "dt_std":      round(float(scores_dt.std()),   4),
        "knn_std":     round(float(scores_knn.std()),  4),
        "significant": significant,
        "better_model": better,
        "conclusion": (
            f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}. "
            f"{'H0 reddedildi — CV ortalamalar arasinda anlamli fark var' if significant else 'H0 reddedilemedi — CV ortalamalari arasinda anlamli fark yok'} "
            f"(α={ALPHA}). "
            f"DT ort.={scores_dt.mean():.3f} vs kNN ort.={scores_knn.mean():.3f}."
        ),
    }


# =============================================================
# 3. WİLCOXON TESTİ (Parametrik Olmayan Alternatif)
# =============================================================

def _wilcoxon_test(
    scores_dt: np.ndarray,
    scores_knn: np.ndarray,
) -> Dict[str, Any]:
    """
    Wilcoxon Isaretli Sira Testi: kucuk orneklemler icin t-test'e alternatif.
    Normallik varsayimi gerektirmez.
    """
    if len(scores_dt) < 3:
        return {
            "stat": 0.0, "p_value": 1.0, "significant": False,
            "conclusion": "Wilcoxon testi icin en az 3 fold gerekli.",
        }

    try:
        diff = scores_dt - scores_knn
        if np.all(diff == 0):
            return {
                "stat": 0.0, "p_value": 1.0, "significant": False,
                "conclusion": "Tum foldlarda skor farki sifir — Wilcoxon uygulanamaz.",
            }

        stat, p_value = wilcoxon(scores_dt, scores_knn, alternative="two-sided")
        significant = float(p_value) < ALPHA

        return {
            "stat":        round(float(stat),    4),
            "p_value":     round(float(p_value), 4),
            "significant": significant,
            "conclusion": (
                f"Wilcoxon testi: W={stat:.4f}, p={p_value:.4f}. "
                f"{'Anlamli fark tespit edildi' if significant else 'Anlamli fark tespit edilemedi'} "
                f"(α={ALPHA})."
            ),
        }
    except Exception as e:
        logger.warning(f"Wilcoxon test hatasi: {e}")
        return {"error": str(e), "significant": False, "p_value": 1.0}


# =============================================================
# GENEL YORUM
# =============================================================

def _interpret_tests(
    mcnemar: dict,
    ttest: dict,
    wilcoxon_res: dict,
) -> List[str]:
    """Uc testi birlestiren genel akademik yorum."""
    lines = []

    mc_sig  = mcnemar.get("significant", False)
    tt_sig  = ttest.get("significant", False)
    wil_sig = wilcoxon_res.get("significant", False)

    sig_count = sum([mc_sig, tt_sig, wil_sig])

    if sig_count == 0:
        lines.append(
            "Uc istatistiksel test de modeller arasinda anlamli bir fark bulmadi (p > 0.05). "
            "Bu, her iki modelin bu veri seti uzerinde benzer performans gosterdigi anlamina gelmektedir."
        )
    elif sig_count >= 2:
        better = ttest.get("better_model") or mcnemar.get("better_model") or "N/A"
        lines.append(
            f"Cogunluk testleri ({sig_count}/3) modeller arasinda anlamli fark tespit etti (p < 0.05). "
            f"'{better}' modeli istatistiksel olarak daha iyi performans gostermektedir."
        )
    else:
        lines.append(
            "Testler karisik sonuclar verdi. Bir test anlamli fark bulurken, "
            "diger(ler)i bulmadi. Kucuk orneklem boyutu (15-20 surucu) bu duruma "
            "yol acmis olabilir — bulgular dikkatli yorumlanmalidir."
        )

    lines.append(
        f"Not: F1 verisi pilot basina 1 satir icerdigindan (n~15-20), "
        f"istatistiksel guc dusuktur. Bulgular oncu bulgular olarak degerlendirilmelidir."
    )

    return lines


# =============================================================
# GORSELLEŞTİRME
# =============================================================

def _plot_test_results(
    scores_dt: np.ndarray,
    scores_knn: np.ndarray,
    mcnemar: dict,
    ttest: dict,
    race_name: str,
) -> str:
    """CV skor karsilastirmasi ve test sonuclari gorseli."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0D0D0D")

    # Sol: Box/Violin gorseli
    ax = axes[0]
    ax.set_facecolor("#1A1A1A")

    n_folds = len(scores_dt)
    folds   = list(range(1, n_folds + 1))

    ax.plot(folds, scores_dt,  "o-", color="#E8002D", linewidth=2.5,
            markersize=8, label=f"Decision Tree ({scores_dt.mean():.3f}±{scores_dt.std():.3f})")
    ax.plot(folds, scores_knn, "s-", color="#1E88E5", linewidth=2.5,
            markersize=8, label=f"kNN ({scores_knn.mean():.3f}±{scores_knn.std():.3f})")
    ax.axhline(scores_dt.mean(),  color="#E8002D", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(scores_knn.mean(), color="#1E88E5", linestyle="--", alpha=0.4, linewidth=1)

    p_val = ttest.get("p_value", 1.0)
    sig_txt = f"p={p_val:.4f} {'*' if p_val < 0.05 else 'ns'}"
    ax.set_title(f"CV Skor Karsilastirmasi\n{sig_txt}", color="white", fontsize=11, pad=8)
    ax.set_xlabel("Fold", color="white")
    ax.set_ylabel("Dogruluk", color="white")
    ax.set_xticks(folds)
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2A2A2A")
    ax.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=8)

    # Sag: Test sonuclari ozet tablosu
    ax2 = axes[1]
    ax2.set_facecolor("#1A1A1A")
    ax2.axis("off")

    test_rows = [
        ["Test",        "Istatistik",          "p-deger",                         "Anlamli?"],
        ["McNemar",     f"χ²={mcnemar.get('chi2_stat',0):.3f}",
                        f"{mcnemar.get('p_value',1):.4f}",
                        "Evet" if mcnemar.get("significant") else "Hayir"],
        ["Paired t",    f"t={ttest.get('t_stat',0):.3f}",
                        f"{ttest.get('p_value',1):.4f}",
                        "Evet" if ttest.get("significant") else "Hayir"],
    ]

    table = ax2.table(
        cellText=test_rows[1:],
        colLabels=test_rows[0],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#1A1A1A" if r > 0 else "#2A2A2A")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#3A3A3A")
        if r > 0 and c == 3:
            cell.set_facecolor("#004D00" if cell.get_text().get_text() == "Evet" else "#4D0000")

    ax2.set_title(f"Istatistiksel Test Ozeti — {race_name}", color="white", fontsize=11, pad=20)

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"stat_tests_{race_name.lower().replace(' ', '_')}.png")
    plt.close(fig)
    return b64


# =============================================================
# KAYDETME VE YARDIMCI
# =============================================================

def _save_test_csv(results: dict, race_name: str) -> None:
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        slug  = race_name.lower().replace(" ", "_")
        path  = TABLES_DIR / f"statistical_tests_{slug}.csv"
        rows = [
            {
                "race":         race_name,
                "test":         "McNemar",
                "statistic":    results.get("mcnemar", {}).get("chi2_stat", ""),
                "p_value":      results.get("mcnemar", {}).get("p_value", ""),
                "significant":  results.get("mcnemar", {}).get("significant", ""),
            },
            {
                "race":         race_name,
                "test":         "Paired t-test",
                "statistic":    results.get("paired_ttest", {}).get("t_stat", ""),
                "p_value":      results.get("paired_ttest", {}).get("p_value", ""),
                "significant":  results.get("paired_ttest", {}).get("significant", ""),
            },
            {
                "race":         race_name,
                "test":         "Wilcoxon",
                "statistic":    results.get("wilcoxon", {}).get("stat", ""),
                "p_value":      results.get("wilcoxon", {}).get("p_value", ""),
                "significant":  results.get("wilcoxon", {}).get("significant", ""),
            },
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info(f"Test sonuclari CSV kaydedildi: {path.name}")
    except Exception as e:
        logger.warning(f"Test CSV kaydetme hatasi: {e}")


def _prepare_data(
    feature_matrix: pd.DataFrame,
    df: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    available = [f for f in CLF_FEATURES if f in feature_matrix.columns]
    if len(available) < 2:
        return None, None, []

    X_raw  = feature_matrix[available].copy()
    imputer  = SimpleImputer(strategy="median")
    X_imp    = imputer.fit_transform(X_raw)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    y = _create_target(feature_matrix)
    if y is None or len(np.unique(y)) < 2:
        return None, None, available

    return X_scaled, y, available


def _create_target(fm: pd.DataFrame) -> Optional[np.ndarray]:
    if "average_lap_time" not in fm.columns:
        return None
    fast   = (fm["average_lap_time"] <= fm["average_lap_time"].median()).astype(int)
    stable = (fm["consistency_score"] >= fm["consistency_score"].median()).astype(int) \
             if "consistency_score" in fm.columns else pd.Series(1, index=fm.index)
    target = (fast & stable).values
    if len(np.unique(target)) < 2:
        target = np.zeros(len(fm), dtype=int)
        target[fm["average_lap_time"].argmin()] = 1
    return target


def _max_safe_folds(y: np.ndarray) -> int:
    return max(2, min(5, int(np.bincount(y).min())))


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _save_fig(fig, filename: str) -> None:
    try:
        FIGURES_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_EVAL_DIR / filename, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    except Exception as e:
        logger.warning(f"Figure kaydetme hatasi ({filename}): {e}")
