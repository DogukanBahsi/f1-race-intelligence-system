"""
src/error_analysis.py
---------------------
F1 Race Intelligence System - Hata Analizi Modulu
Yanlis tahminler, yuksek guvenli hatalar ve surucu bazli hata oranlari.
"""

import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import (
    DECISION_TREE_CONFIG, KNN_CONFIG,
    FIGURES_EVAL_DIR, TABLES_DIR, RANDOM_STATE, CV_CONFIG,
)
from src.logger import get_logger, log_success

logger = get_logger("ErrorAnalysis")

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

def run_error_analysis(
    feature_matrix: pd.DataFrame,
    df: pd.DataFrame,
    race_name: str,
) -> Dict[str, Any]:
    """
    Kapsamli hata analizi pipeline'i.
    Yanlis tahminler, yuksek guvenli hatalar, surucu/ozellik bazli analiz.
    """
    logger.info(f"Hata analizi basliyor: {race_name}")
    FIGURES_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feature_names, drivers = _prepare_data(feature_matrix, df)
    if X is None or len(np.unique(y)) < 2:
        return {"error": "Hata analizi icin yeterli/dengeli veri yok."}

    results: Dict[str, Any] = {"race_name": race_name, "feature_names": feature_names}

    # 1. Cross-val tahminleri al (tum veri uzerinde out-of-fold)
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

    dt_pred  = cross_val_predict(dt,  X, y, cv=cv)
    knn_pred = cross_val_predict(knn, X, y, cv=cv)

    try:
        dt_proba  = cross_val_predict(dt,  X, y, cv=cv, method="predict_proba")
        knn_proba = cross_val_predict(knn, X, y, cv=cv, method="predict_proba")
    except Exception:
        dt_proba  = None
        knn_proba = None

    # 2. Yanlis tahmin analizi
    results["dt_errors"]  = _analyze_errors(
        y, dt_pred,  dt_proba,  feature_matrix, X, feature_names, drivers, "Decision Tree"
    )
    results["knn_errors"] = _analyze_errors(
        y, knn_pred, knn_proba, feature_matrix, X, feature_names, drivers, "kNN"
    )

    # 3. Confusion matrix gorsellestirme
    results["cm_chart_b64"] = _plot_confusion_matrices(
        y, dt_pred, knn_pred, race_name
    )

    # 4. Hata karsilastirmasi (hangi suruculer her iki model tarafindan da yanlis siniflandirildi?)
    results["common_errors"] = _find_common_errors(
        y, dt_pred, knn_pred, drivers
    )

    # 5. Surucu bazli hata raporu
    results["driver_error_report"] = _driver_error_report(
        y, dt_pred, knn_pred, drivers, feature_matrix
    )
    results["driver_error_chart_b64"] = _plot_driver_errors(
        results["driver_error_report"], race_name
    )

    # 6. Ozellik degeri vs hata iliskisi
    results["feature_error_chart_b64"] = _plot_feature_vs_error(
        X, y, dt_pred, feature_names, race_name
    )

    # 7. CSV kaydet
    _save_error_csv(results["driver_error_report"], race_name)

    log_success(f"Hata analizi tamamlandi: {race_name}")
    return results


# =============================================================
# VERI HAZIRLAMA
# =============================================================

def _prepare_data(
    feature_matrix: pd.DataFrame,
    df: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], List[str]]:
    available = [f for f in CLF_FEATURES if f in feature_matrix.columns]
    if len(available) < 2:
        return None, None, [], []

    X_raw  = feature_matrix[available].copy()
    drivers = feature_matrix["Driver"].tolist() if "Driver" in feature_matrix.columns else []

    imputer  = SimpleImputer(strategy="median")
    X_imp    = imputer.fit_transform(X_raw)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    y = _create_target(feature_matrix)
    if y is None or len(np.unique(y)) < 2:
        return None, None, available, drivers

    return X_scaled, y, available, drivers


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


# =============================================================
# HATA ANALİZİ FONKSİYONLARI
# =============================================================

def _analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    feature_matrix: pd.DataFrame,
    X: np.ndarray,
    feature_names: List[str],
    drivers: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """Tek bir model icin hata analizi."""
    wrong_mask = y_pred != y_true
    wrong_idx  = np.where(wrong_mask)[0]

    total     = len(y_true)
    n_wrong   = int(wrong_mask.sum())
    error_rate = round(n_wrong / total, 4) if total > 0 else 0.0

    # Yanlis siniflandirilanlar
    wrong_cases = []
    for i in wrong_idx:
        conf = float(y_proba[i][y_pred[i]]) if y_proba is not None else None
        wrong_cases.append({
            "driver":     drivers[i] if i < len(drivers) else f"Sample_{i}",
            "actual":     "Guclu" if y_true[i] == 1 else "Normal",
            "predicted":  "Guclu" if y_pred[i] == 1 else "Normal",
            "error_type": "FP" if y_pred[i] == 1 else "FN",
            "confidence": round(conf, 4) if conf is not None else None,
        })

    # Yuksek guvenli yanlis tahminler (confidence > 0.7)
    high_conf_errors = [
        c for c in wrong_cases
        if c.get("confidence") is not None and c["confidence"] >= 0.7
    ]

    # En cok karistirilan siniflar
    cm = confusion_matrix(y_true, y_pred)
    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0  # Yanlis Pozitif
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0  # Yanlis Negatif

    return {
        "model":             model_name,
        "total_samples":     total,
        "n_errors":          n_wrong,
        "error_rate":        error_rate,
        "accuracy":          round(1 - error_rate, 4),
        "false_positives":   fp,
        "false_negatives":   fn,
        "wrong_cases":       wrong_cases,
        "high_conf_errors":  high_conf_errors,
        "comment": (
            f"{model_name}: {n_wrong}/{total} ornek yanlis siniflandirildi "
            f"(%{error_rate*100:.1f} hata orani). "
            f"FP={fp}, FN={fn}."
            + (f" {len(high_conf_errors)} yuksek guvenli hata var." if high_conf_errors else "")
        ),
    }


def _find_common_errors(
    y_true: np.ndarray,
    y_pred_dt: np.ndarray,
    y_pred_knn: np.ndarray,
    drivers: List[str],
) -> List[Dict[str, Any]]:
    """Her iki model tarafindan da yanlis siniflandirilan ornekler."""
    both_wrong = (y_pred_dt != y_true) & (y_pred_knn != y_true)
    common = []
    for i in np.where(both_wrong)[0]:
        common.append({
            "driver":  drivers[i] if i < len(drivers) else f"Sample_{i}",
            "actual":  "Guclu" if y_true[i] == 1 else "Normal",
            "dt_pred": "Guclu" if y_pred_dt[i] == 1 else "Normal",
            "knn_pred":"Guclu" if y_pred_knn[i] == 1 else "Normal",
        })
    return common


def _driver_error_report(
    y_true: np.ndarray,
    y_pred_dt: np.ndarray,
    y_pred_knn: np.ndarray,
    drivers: List[str],
    feature_matrix: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Surucu bazli hata ozeti."""
    rows = []
    for i, driver in enumerate(drivers):
        if i >= len(y_true):
            break
        dt_correct  = bool(y_pred_dt[i]  == y_true[i])
        knn_correct = bool(y_pred_knn[i] == y_true[i])
        both_wrong  = (not dt_correct) and (not knn_correct)

        avg_lt = None
        if "average_lap_time" in feature_matrix.columns:
            avg_lt = round(float(feature_matrix.iloc[i]["average_lap_time"]), 3)

        rows.append({
            "Driver":       driver,
            "Gercek":       "Guclu" if y_true[i] == 1 else "Normal",
            "DT Tahmini":   "Guclu" if y_pred_dt[i] == 1 else "Normal",
            "kNN Tahmini":  "Guclu" if y_pred_knn[i] == 1 else "Normal",
            "DT Dogru":     "✓" if dt_correct  else "✗",
            "kNN Dogru":    "✓" if knn_correct else "✗",
            "Her Ikisi Yanlis": "⚠" if both_wrong else "",
            "Ort. Tur (s)": avg_lt,
        })

    return rows


# =============================================================
# GORSELLEŞTİRME
# =============================================================

def _plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred_dt: np.ndarray,
    y_pred_knn: np.ndarray,
    race_name: str,
) -> str:
    """Yan yana confusion matrix gorsellestirmesi."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#0D0D0D")

    class_names = ["Normal", "Guclu"]

    for ax, y_pred, title, color in [
        (axes[0], y_pred_dt,  "Decision Tree", "#E8002D"),
        (axes[1], y_pred_knn, "kNN",           "#1E88E5"),
    ]:
        cm_vals = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ax.set_facecolor("#1A1A1A")

        im = ax.imshow(cm_vals, cmap="Reds" if "Tree" in title else "Blues",
                       alpha=0.8, vmin=0)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, color="white", fontsize=10)
        ax.set_yticklabels(class_names, color="white", fontsize=10)
        ax.set_xlabel("Tahmin Edilen", color="white", fontsize=10)
        ax.set_ylabel("Gercek", color="white", fontsize=10)
        ax.set_title(f"Confusion Matrix — {title}\n{race_name}", color="white", fontsize=11, pad=8)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#2A2A2A")

        for i in range(2):
            for j in range(2):
                val = cm_vals[i, j]
                label = "TP" if i == 1 and j == 1 else \
                        "TN" if i == 0 and j == 0 else \
                        "FP" if i == 0 and j == 1 else "FN"
                ax.text(j, i, f"{val}\n({label})",
                        ha="center", va="center", fontsize=12,
                        color="white", fontweight="bold")

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"confusion_matrix_{race_name.lower().replace(' ', '_')}.png")
    plt.close(fig)
    return b64


def _plot_driver_errors(
    driver_report: List[dict],
    race_name: str,
) -> str:
    """Surucu bazli dogru/yanlis tahmin gorseli."""
    if not driver_report:
        return ""

    df = pd.DataFrame(driver_report)
    drivers   = df["Driver"].tolist()
    dt_wrong  = [1 if r == "✗" else 0 for r in df["DT Dogru"].tolist()]
    knn_wrong = [1 if r == "✗" else 0 for r in df["kNN Dogru"].tolist()]

    fig, ax = plt.subplots(figsize=(max(8, len(drivers) * 0.6), 4))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#1A1A1A")

    x = np.arange(len(drivers))
    w = 0.35
    ax.bar(x - w / 2, dt_wrong,  w, label="DT Hatasi",  color="#E8002D", alpha=0.8)
    ax.bar(x + w / 2, knn_wrong, w, label="kNN Hatasi", color="#1E88E5", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(drivers, rotation=45, ha="right", color="white", fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Dogru", "Yanlis"], color="white")
    ax.set_title(f"Surucu Bazli Siniflandirma Sonucu — {race_name}", color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2A2A2A")
    ax.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=9)

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"driver_errors_{race_name.lower().replace(' ', '_')}.png")
    plt.close(fig)
    return b64


def _plot_feature_vs_error(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    race_name: str,
) -> str:
    """Ozellik degeri vs hata durumu scatter grafigi (ilk 2 ozellik)."""
    if X.shape[1] < 2:
        return ""

    wrong_mask = y_pred != y_true

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#1A1A1A")

    ax.scatter(X[~wrong_mask, 0], X[~wrong_mask, 1],
               c="#00C851", s=80, alpha=0.8, marker="o", label="Dogru Tahmin", zorder=3)
    ax.scatter(X[wrong_mask, 0],  X[wrong_mask, 1],
               c="#E8002D", s=120, alpha=0.9, marker="X", label="Yanlis Tahmin", zorder=4)

    ax.set_xlabel(feature_names[0], color="white", fontsize=11)
    ax.set_ylabel(feature_names[1] if len(feature_names) > 1 else "Feature 2",
                  color="white", fontsize=11)
    ax.set_title(f"Hata Dagilimi — {race_name}\n(Decision Tree, 2D Projeksiyon)",
                 color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2A2A2A")
    ax.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=10)
    ax.grid(color="#2A2A2A", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"error_scatter_{race_name.lower().replace(' ', '_')}.png")
    plt.close(fig)
    return b64


# =============================================================
# KAYDETME VE YARDIMCI
# =============================================================

def _save_error_csv(driver_report: List[dict], race_name: str) -> None:
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        slug = race_name.lower().replace(" ", "_")
        path = TABLES_DIR / f"error_analysis_{slug}.csv"
        pd.DataFrame(driver_report).to_csv(path, index=False)
        logger.info(f"Hata analizi CSV kaydedildi: {path.name}")
    except Exception as e:
        logger.warning(f"Hata analizi CSV hatasi: {e}")


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
