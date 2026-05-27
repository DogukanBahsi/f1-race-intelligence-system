"""
src/xai.py
----------
F1 Race Intelligence System - Aciklanabilir Yapay Zeka (XAI) Modulu
SHAP entegrasyonu ve Permutation Importance ile model kararlarini yorumla.
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
import matplotlib.cm as cm

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from config import DECISION_TREE_CONFIG, KNN_CONFIG, FIGURES_XAI_DIR, RANDOM_STATE
from src.logger import get_logger, log_success

logger = get_logger("XAI")

# SHAP mevcutluk kontrolu
try:
    import shap
    HAS_SHAP = True
    logger.info("SHAP kurulu — TreeExplainer aktif.")
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP kurulu degil. Permutation Importance kullanilacak. (pip install shap)")

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

def run_xai_analysis(
    feature_matrix: pd.DataFrame,
    df: pd.DataFrame,
    race_name: str,
) -> Dict[str, Any]:
    """
    XAI pipeline: SHAP veya Permutation Importance ile ozellik onemliligi.
    Ozet grafikler, force plot destegi (SHAP mevcutsa) ve yerel aciklamalar.
    """
    logger.info(f"XAI analizi basliyor: {race_name}")
    FIGURES_XAI_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {"race_name": race_name, "has_shap": HAS_SHAP}

    X, y, feature_names, drivers = _prepare_data(feature_matrix, df)
    if X is None or len(np.unique(y)) < 2:
        return {"error": "XAI icin yeterli/dengeli veri yok.", "has_shap": HAS_SHAP}

    results["feature_names"] = feature_names
    results["drivers"]       = drivers

    # 1. Decision Tree egit (tum veri uzerinde — XAI icin)
    dt = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        **{k: v for k, v in DECISION_TREE_CONFIG.items() if k != "random_state"}
    )
    dt.fit(X, y)

    # 2. kNN egit
    knn = KNeighborsClassifier(
        n_neighbors=min(KNN_CONFIG["n_neighbors"], len(X) - 1),
        metric=KNN_CONFIG["metric"],
    )
    knn.fit(X, y)

    # 3. SHAP analizi (Decision Tree icin)
    if HAS_SHAP:
        results["shap_dt"] = _shap_tree_analysis(dt, X, y, feature_names, race_name)
    else:
        results["shap_dt"] = _fallback_feature_importance(dt, X, y, feature_names, race_name, "Decision Tree")

    # 4. Permutation Importance (her iki model icin)
    results["perm_importance_dt"]  = _compute_permutation_importance(
        dt, X, y, feature_names, race_name, "Decision Tree"
    )
    results["perm_importance_knn"] = _compute_permutation_importance(
        knn, X, y, feature_names, race_name, "kNN"
    )

    # 5. Yerel aciklama ornekleri (her surucu icin)
    results["local_explanations"] = _local_explanations(dt, X, y, feature_names, drivers)

    # 6. Karsilastirmali ozellik onemi grafigi
    results["comparison_chart_b64"] = _plot_feature_importance_comparison(
        results["perm_importance_dt"],
        results["perm_importance_knn"],
        feature_names,
        race_name,
    )

    log_success(f"XAI analizi tamamlandi: {race_name} (SHAP={'Evet' if HAS_SHAP else 'Hayir'})")
    return results


# =============================================================
# VERI HAZIRLAMA
# =============================================================

def _prepare_data(
    feature_matrix: pd.DataFrame,
    df: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], List[str]]:
    """X, y, ozellik adlari ve surucu listesini hazirla."""
    available = [f for f in CLF_FEATURES if f in feature_matrix.columns]
    if len(available) < 2:
        return None, None, [], []

    X_raw  = feature_matrix[available].copy()
    drivers = feature_matrix["Driver"].tolist() if "Driver" in feature_matrix.columns else []

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)

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
# SHAP ANALİZİ
# =============================================================

def _shap_tree_analysis(
    model: DecisionTreeClassifier,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    race_name: str,
) -> Dict[str, Any]:
    """SHAP TreeExplainer ile karar agaci aciklamasi."""
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Binary siniflandirma: shap_values[1] pozitif sinif icin
        if isinstance(shap_values, list):
            sv = shap_values[1]  # pozitif sinif
        else:
            sv = shap_values

        mean_abs = np.abs(sv).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature":    feature_names,
            "SHAP Mean":  [round(float(v), 5) for v in mean_abs],
        }).sort_values("SHAP Mean", ascending=False)

        # Ozet grafik (beeswarm benzeri bar)
        chart_b64 = _plot_shap_summary(sv, X, feature_names, race_name)
        beeswarm_b64 = _plot_shap_beeswarm(sv, X, feature_names, race_name)

        log_success(f"SHAP analizi tamamlandi. En onemli ozellik: {importance_df.iloc[0]['Feature']}")

        return {
            "method":          "SHAP TreeExplainer",
            "importance":      importance_df.to_dict("records"),
            "top_feature":     importance_df.iloc[0]["Feature"],
            "shap_values":     sv.tolist(),
            "chart_b64":       chart_b64,
            "beeswarm_b64":    beeswarm_b64,
            "comment": (
                f"SHAP analizi: '{importance_df.iloc[0]['Feature']}' ozelligi "
                f"modelin kararlarini en cok etkileyen faktordurc "
                f"(ortalama |SHAP|={importance_df.iloc[0]['SHAP Mean']:.4f})."
            ),
        }
    except Exception as e:
        logger.warning(f"SHAP analizi hatasi: {e}")
        return _fallback_feature_importance(model, X, y, feature_names, race_name, "Decision Tree (SHAP hatali)")


def _plot_shap_summary(shap_vals: np.ndarray, X: np.ndarray,
                        feature_names: List[str], race_name: str) -> str:
    """SHAP ozet bar grafigi."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.6)))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#1A1A1A")

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feature_names)))
    bars = ax.barh(
        [feature_names[i] for i in sorted_idx],
        [mean_abs[i] for i in sorted_idx],
        color=[colors[j] for j in range(len(sorted_idx))],
        alpha=0.85,
    )
    ax.set_xlabel("Ortalama |SHAP Degeri|", color="white", fontsize=11)
    ax.set_title(f"SHAP Ozellik Onemliligi — {race_name}", color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2A2A2A")

    for bar, v in zip(bars, [mean_abs[i] for i in sorted_idx]):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", ha="left", color="white", fontsize=8)

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"shap_summary_{race_name.lower().replace(' ', '_')}.png", xai=True)
    plt.close(fig)
    return b64


def _plot_shap_beeswarm(shap_vals: np.ndarray, X: np.ndarray,
                         feature_names: List[str], race_name: str) -> str:
    """SHAP beeswarm benzeri scatter grafigi."""
    fig, ax = plt.subplots(figsize=(9, max(4, len(feature_names) * 0.7)))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#1A1A1A")

    sorted_idx = np.argsort(np.abs(shap_vals).mean(axis=0))

    for j, feat_idx in enumerate(sorted_idx):
        sv_col   = shap_vals[:, feat_idx]
        x_vals   = sv_col + np.random.uniform(-0.002, 0.002, len(sv_col))
        feat_vals = X[:, feat_idx]
        cmap     = plt.cm.RdBu_r
        colors   = cmap((feat_vals - feat_vals.min()) / (feat_vals.ptp() + 1e-9))
        ax.scatter(x_vals, [j] * len(sv_col), c=colors, s=50, alpha=0.8)

    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], color="white", fontsize=9)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("SHAP Degeri (pozitif = guclu performans etkisi)", color="white", fontsize=10)
    ax.set_title(f"SHAP Beeswarm — {race_name}", color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2A2A2A")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Ozellik Degeri (dusuk → yuksek)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"shap_beeswarm_{race_name.lower().replace(' ', '_')}.png", xai=True)
    plt.close(fig)
    return b64


# =============================================================
# FALLBACK: PERMUTATION IMPORTANCE
# =============================================================

def _fallback_feature_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    race_name: str,
    model_label: str,
) -> Dict[str, Any]:
    """SHAP yoksa sklearn feature_importances_ kullan."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    imp_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": [round(float(v), 5) for v in importances],
    }).sort_values("Importance", ascending=False)

    chart_b64 = _plot_importance_bar(
        imp_df["Feature"].tolist(),
        imp_df["Importance"].tolist(),
        f"Ozellik Onemliligi ({model_label}) — {race_name}",
        race_name,
    )

    top = imp_df.iloc[0]["Feature"]
    return {
        "method":      "sklearn feature_importances_",
        "importance":  imp_df.to_dict("records"),
        "top_feature": top,
        "chart_b64":   chart_b64,
        "comment": (
            f"{model_label}: en onemli ozellik '{top}' "
            f"(onem={imp_df.iloc[0]['Importance']:.4f})."
        ),
    }


def _compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    race_name: str,
    model_label: str,
) -> Dict[str, Any]:
    """sklearn permutation_importance ile stabil ozellik onemliligi."""
    try:
        result = permutation_importance(
            model, X, y,
            n_repeats=30,
            random_state=RANDOM_STATE,
            scoring="accuracy",
        )
        means = result.importances_mean
        stds  = result.importances_std

        imp_df = pd.DataFrame({
            "Feature":  feature_names,
            "PI Mean":  [round(float(v), 5) for v in means],
            "PI Std":   [round(float(v), 5) for v in stds],
        }).sort_values("PI Mean", ascending=False)

        chart_b64 = _plot_permutation_importance(imp_df, model_label, race_name)
        top = imp_df.iloc[0]["Feature"]

        return {
            "model":      model_label,
            "importance": imp_df.to_dict("records"),
            "top_feature": top,
            "chart_b64":  chart_b64,
            "comment": (
                f"{model_label} Permutation Importance: '{top}' ozelligi "
                f"cikarildiginda model dogrulugu en cok dusuyor "
                f"(ortalama dusus: {imp_df.iloc[0]['PI Mean']:.4f})."
            ),
        }
    except Exception as e:
        logger.warning(f"Permutation importance hatasi ({model_label}): {e}")
        return {"model": model_label, "error": str(e), "importance": []}


def _plot_permutation_importance(imp_df: pd.DataFrame, model_label: str, race_name: str) -> str:
    """Permutation Importance bar grafigi (hata cubuklu)."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(imp_df) * 0.55)))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#1A1A1A")

    color = "#E8002D" if "Decision" in model_label else "#1E88E5"
    y_pos = range(len(imp_df))

    ax.barh(
        list(y_pos),
        imp_df["PI Mean"].tolist(),
        xerr=imp_df["PI Std"].tolist(),
        color=color,
        alpha=0.85,
        error_kw={"ecolor": "white", "capsize": 4, "linewidth": 1.2},
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(imp_df["Feature"].tolist(), color="white", fontsize=9)
    ax.set_xlabel("Dogruluk Dususu (ozellik karistirildiginda)", color="white", fontsize=10)
    ax.set_title(f"Permutation Importance — {model_label}\n{race_name}", color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2A2A2A")
    ax.axvline(0, color="#555555", linewidth=0.8)

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    slug = f"perm_imp_{model_label.lower().replace(' ', '_').replace('/', '_')}_{race_name.lower().replace(' ', '_')}"
    _save_fig(fig, f"{slug}.png", xai=True)
    plt.close(fig)
    return b64


def _plot_importance_bar(features: List[str], importances: List[float],
                          title: str, race_name: str) -> str:
    """Genel amacli ozellik onemi bar grafigi."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.55)))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#1A1A1A")

    colors = ["#E8002D" if i == 0 else "#555555" for i in range(len(features))]
    ax.barh(features[::-1], importances[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Ozellik Onemi", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2A2A2A")

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"fi_{race_name.lower().replace(' ', '_')}.png", xai=True)
    plt.close(fig)
    return b64


def _plot_feature_importance_comparison(
    perm_dt: dict,
    perm_knn: dict,
    feature_names: List[str],
    race_name: str,
) -> str:
    """DT ve kNN Permutation Importance karsilastirilmasi."""
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(feature_names) * 0.6)))
    fig.patch.set_facecolor("#0D0D0D")

    configs = [
        (axes[0], perm_dt,  "Decision Tree", "#E8002D"),
        (axes[1], perm_knn, "kNN",           "#1E88E5"),
    ]

    for ax, perm, label, color in configs:
        ax.set_facecolor("#1A1A1A")
        imp_list = perm.get("importance", [])
        if not imp_list:
            ax.text(0.5, 0.5, "Veri yok", color="white", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        imp_df = pd.DataFrame(imp_list).sort_values("PI Mean", ascending=True)
        ax.barh(
            imp_df["Feature"].tolist(),
            imp_df["PI Mean"].tolist(),
            xerr=imp_df["PI Std"].tolist() if "PI Std" in imp_df.columns else None,
            color=color, alpha=0.85,
            error_kw={"ecolor": "white", "capsize": 3},
        )
        ax.set_title(f"{label} — {race_name}", color="white", fontsize=10, pad=8)
        ax.set_xlabel("Dogruluk Dususu", color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#2A2A2A")
        ax.axvline(0, color="#555555", linewidth=0.8)

    plt.suptitle("Permutation Importance Karsilastirmasi", color="white", fontsize=12)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"perm_comparison_{race_name.lower().replace(' ', '_')}.png", xai=True)
    plt.close(fig)
    return b64


# =============================================================
# YEREL ACIKLAMALAR
# =============================================================

def _local_explanations(
    model: DecisionTreeClassifier,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    drivers: List[str],
) -> List[Dict[str, Any]]:
    """
    Her surucu icin yerel aciklama: model tahmin + gercek + ozellik degerleri.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    explanations = []
    for i, (driver, pred, actual) in enumerate(zip(drivers, y_pred, y)):
        conf = float(y_proba[i][pred]) if y_proba is not None else None
        feat_vals = {f: round(float(X[i, j]), 4) for j, f in enumerate(feature_names)}
        explanations.append({
            "driver":       driver,
            "predicted":    int(pred),
            "actual":       int(actual),
            "correct":      bool(pred == actual),
            "confidence":   round(conf, 4) if conf is not None else None,
            "features":     feat_vals,
            "prediction_label": "Guclu Performans" if pred == 1 else "Normal Performans",
        })
    return explanations


# =============================================================
# YARDIMCI
# =============================================================

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _save_fig(fig, filename: str, xai: bool = False) -> None:
    try:
        target = FIGURES_XAI_DIR if xai else FIGURES_XAI_DIR
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / filename, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    except Exception as e:
        logger.warning(f"Figure kaydetme hatasi ({filename}): {e}")
