"""
src/model_evaluation.py
-----------------------
F1 Race Intelligence System - Akademik Model Degerlendirme Modulu
Stratified K-Fold Cross Validation, GridSearchCV, ROC/AUC, Model Karsilastirma
"""

import io
import base64
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
    cross_val_predict,
    cross_validate,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
)

from config import (
    DECISION_TREE_CONFIG,
    KNN_CONFIG,
    KMEANS_CONFIG,
    KMEANS_GRID_PARAMS,
    FIGURES_EVAL_DIR,
    TABLES_DIR,
    RANDOM_STATE,
    CV_CONFIG,
    GRIDSEARCH_DT_PARAMS,
    GRIDSEARCH_KNN_PARAMS,
)
from src.logger import get_logger, log_success

logger = get_logger("ModelEvaluation")

# Siniflandirma icin kullanilacak ozellikler
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

def run_full_evaluation(
    feature_matrix: pd.DataFrame,
    df: pd.DataFrame,
    race_name: str,
) -> Dict[str, Any]:
    """
    Tam akademik degerlendirme pipeline'i.
    CV + GridSearch + ROC/AUC + Model Karsilastirma + Kaydetme
    """
    logger.info(f"Akademik degerlendirme basliyor: {race_name}")
    _ensure_dirs()

    results: Dict[str, Any] = {"race_name": race_name}

    # Veri hazirla
    X, y, feature_names = _prepare_data(feature_matrix, df)
    if X is None or len(np.unique(y)) < 2:
        logger.warning(f"{race_name}: Degerlendirme icin yeterli/dengeli veri yok.")
        return {"error": "Degerlendirme icin yeterli ya da dengeli veri bulunamadi."}

    results["n_samples"] = int(len(y))
    results["n_positive"] = int(y.sum())
    results["feature_names"] = feature_names

    # 1. Cross-Validation
    results["cross_validation"] = run_cross_validation(X, y, race_name, feature_names)

    # 2. GridSearchCV
    results["grid_search"] = run_grid_search(X, y, race_name)

    # 3. K-Means optimizasyonu (silhouette)
    results["kmeans_optimization"] = run_kmeans_optimization(X, race_name)

    # 4. ROC / AUC
    results["roc_auc"] = compute_roc_auc(X, y, race_name)

    # 5. Model karsilastirma tablosu
    results["model_comparison"] = build_model_comparison(
        results["cross_validation"],
        results["grid_search"],
        results["roc_auc"],
        race_name,
    )

    # 6. Dosyalara kaydet
    _save_comparison_csv(results["model_comparison"], race_name)
    _save_comparison_markdown(results["model_comparison"], race_name)
    _save_cv_csv(results["cross_validation"], race_name)

    log_success(f"Akademik degerlendirme tamamlandi: {race_name}")
    return results


# =============================================================
# VERI HAZIRLAMA
# =============================================================

def _prepare_data(
    feature_matrix: pd.DataFrame,
    df: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Feature matrix'ten X ve y hazirla. Olceklendirme dahil."""
    available_features = [f for f in CLF_FEATURES if f in feature_matrix.columns]
    if len(available_features) < 2:
        return None, None, []

    X_raw = feature_matrix[available_features].copy()
    y = _create_target(feature_matrix, df)
    if y is None or len(np.unique(y)) < 2:
        return None, None, available_features

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, y.values, available_features


def _create_target(feature_matrix: pd.DataFrame, df: pd.DataFrame) -> Optional[pd.Series]:
    """
    strong_performance hedef degiskenini olustur.
    Kural: hem hizli (median alt yarisi) hem de istikrarli (consistency_score ust yarisi)
    """
    fm = feature_matrix.copy()
    if "average_lap_time" not in fm.columns:
        return None

    race_median = fm["average_lap_time"].median()
    fast_mask = fm["average_lap_time"] <= race_median

    if "consistency_score" in fm.columns:
        cons_median = fm["consistency_score"].median()
        stable_mask = fm["consistency_score"] >= cons_median
    else:
        stable_mask = pd.Series(True, index=fm.index)

    target = (fast_mask & stable_mask).astype(int)

    # En az bir ornek her siniftan olmali
    if target.nunique() < 2:
        target[:] = 0
        if "average_lap_time" in fm.columns:
            target.iloc[fm["average_lap_time"].argmin()] = 1

    return target


# =============================================================
# 1. CROSS-VALIDATION
# =============================================================

def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    race_name: str,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Stratified K-Fold Cross Validation (5-fold).
    Decision Tree ve kNN icin fold bazli skor raporlamasi.
    """
    n_splits = min(CV_CONFIG["n_splits"], _max_safe_folds(y))
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=CV_CONFIG["shuffle"],
        random_state=CV_CONFIG["random_state"],
    )

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, **{
        k: v for k, v in DECISION_TREE_CONFIG.items() if k != "random_state"
    })
    knn = KNeighborsClassifier(
        n_neighbors=min(KNN_CONFIG["n_neighbors"], len(X) - 1),
        metric=KNN_CONFIG["metric"],
    )

    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    dt_cv  = cross_validate(dt,  X, y, cv=cv, scoring=scoring, return_train_score=True)
    knn_cv = cross_validate(knn, X, y, cv=cv, scoring=scoring, return_train_score=True)

    def _summarize(cv_result: dict, model_name: str) -> dict:
        fold_accs = cv_result["test_accuracy"].tolist()
        return {
            "model": model_name,
            "n_folds": n_splits,
            "fold_accuracies": [round(s, 4) for s in fold_accs],
            "mean_accuracy":   round(float(np.mean(fold_accs)), 4),
            "std_accuracy":    round(float(np.std(fold_accs)),  4),
            "mean_precision":  round(float(np.mean(cv_result["test_precision_macro"])), 4),
            "mean_recall":     round(float(np.mean(cv_result["test_recall_macro"])),    4),
            "mean_f1":         round(float(np.mean(cv_result["test_f1_macro"])),        4),
            "train_mean_acc":  round(float(np.mean(cv_result["train_accuracy"])),       4),
            "overfit_gap":     round(
                float(np.mean(cv_result["train_accuracy"])) -
                float(np.mean(cv_result["test_accuracy"])), 4
            ),
        }

    dt_summary  = _summarize(dt_cv,  "Decision Tree")
    knn_summary = _summarize(knn_cv, "kNN")

    chart_b64 = _plot_cv_comparison(dt_summary, knn_summary, race_name)

    log_success(
        f"CV tamamlandi ({n_splits}-fold) — "
        f"DT: {dt_summary['mean_accuracy']:.3f}±{dt_summary['std_accuracy']:.3f} | "
        f"kNN: {knn_summary['mean_accuracy']:.3f}±{knn_summary['std_accuracy']:.3f}"
    )

    return {
        "n_folds":       n_splits,
        "decision_tree": dt_summary,
        "knn":           knn_summary,
        "chart_b64":     chart_b64,
        "comment": (
            f"{race_name} — {n_splits}-fold Stratified CV: "
            f"Decision Tree ort. dogruluk %{dt_summary['mean_accuracy']*100:.1f} "
            f"(±{dt_summary['std_accuracy']*100:.1f}), "
            f"kNN ort. dogruluk %{knn_summary['mean_accuracy']*100:.1f} "
            f"(±{knn_summary['std_accuracy']*100:.1f})."
        ),
    }


def _plot_cv_comparison(dt: dict, knn: dict, race_name: str) -> str:
    """CV fold skor grafigi — barlar + hata cizgileri."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0D0D0D")

    models   = ["Decision Tree", "kNN"]
    means    = [dt["mean_accuracy"], knn["mean_accuracy"]]
    stds     = [dt["std_accuracy"],  knn["std_accuracy"]]
    colors   = ["#E8002D", "#1E88E5"]

    # Sol: Ortalama dogruluk karsilastirmasi
    ax = axes[0]
    ax.set_facecolor("#1A1A1A")
    bars = ax.bar(models, means, color=colors, width=0.5, alpha=0.85, yerr=stds,
                  error_kw={"ecolor": "white", "capsize": 6, "capthick": 2, "linewidth": 1.5})
    ax.set_title(f"CV Ortalama Dogruluk — {race_name}", color="white", fontsize=12, pad=10)
    ax.set_ylabel("Dogruluk", color="white")
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2A2A2A")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{m:.3f}\n±{s:.3f}", ha="center", color="white", fontsize=9)

    # Sag: Fold bazli skor grafigi
    ax2 = axes[1]
    ax2.set_facecolor("#1A1A1A")
    folds = list(range(1, len(dt["fold_accuracies"]) + 1))
    ax2.plot(folds, dt["fold_accuracies"],  "o-", color="#E8002D", linewidth=2,
             markersize=7, label="Decision Tree")
    ax2.plot(folds, knn["fold_accuracies"], "s-", color="#1E88E5", linewidth=2,
             markersize=7, label="kNN")
    ax2.axhline(dt["mean_accuracy"],  color="#E8002D", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(knn["mean_accuracy"], color="#1E88E5", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_title(f"Fold Bazli Dogruluk — {race_name}", color="white", fontsize=12, pad=10)
    ax2.set_xlabel("Fold", color="white")
    ax2.set_ylabel("Dogruluk", color="white")
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_xticks(folds)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#2A2A2A")
    ax2.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=9)

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"cv_comparison_{race_name.lower().replace(' ', '_')}.png")
    plt.close(fig)
    return b64


# =============================================================
# 2. GRIDSEARCHCV
# =============================================================

def run_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    race_name: str,
) -> Dict[str, Any]:
    """
    GridSearchCV ile Decision Tree ve kNN icin en iyi hiperparametreler.
    Kucuk veri setleri icin cv=3 kullanilir.
    """
    cv_inner = min(3, _max_safe_folds(y))
    results: Dict[str, Any] = {}

    # ── Decision Tree ─────────────────────────────────────────
    dt_base = DecisionTreeClassifier(random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    gs_dt = GridSearchCV(
        dt_base,
        GRIDSEARCH_DT_PARAMS,
        cv=cv_inner,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    gs_dt.fit(X, y)
    dt_time = time.perf_counter() - t0

    results["decision_tree"] = {
        "best_params":  gs_dt.best_params_,
        "best_score":   round(float(gs_dt.best_score_), 4),
        "search_time_s": round(dt_time, 2),
        "cv_results_df": pd.DataFrame(gs_dt.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].sort_values("rank_test_score").head(5).to_dict("records"),
        "comment": (
            f"Decision Tree GridSearch en iyi parametreler: {gs_dt.best_params_} "
            f"=> CV dogrulugu: %{gs_dt.best_score_*100:.1f}"
        ),
    }

    # ── kNN ───────────────────────────────────────────────────
    knn_params = {
        k: [v for v in vals if not (k == "n_neighbors" and v >= len(X))]
        for k, vals in GRIDSEARCH_KNN_PARAMS.items()
    }
    knn_base = KNeighborsClassifier()
    t0 = time.perf_counter()
    gs_knn = GridSearchCV(
        knn_base,
        knn_params,
        cv=cv_inner,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    gs_knn.fit(X, y)
    knn_time = time.perf_counter() - t0

    results["knn"] = {
        "best_params":  gs_knn.best_params_,
        "best_score":   round(float(gs_knn.best_score_), 4),
        "search_time_s": round(knn_time, 2),
        "cv_results_df": pd.DataFrame(gs_knn.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].sort_values("rank_test_score").head(5).to_dict("records"),
        "comment": (
            f"kNN GridSearch en iyi parametreler: {gs_knn.best_params_} "
            f"=> CV dogrulugu: %{gs_knn.best_score_*100:.1f}"
        ),
    }

    chart_b64 = _plot_gridsearch_heatmap(gs_dt, gs_knn, race_name)
    results["chart_b64"] = chart_b64

    log_success(
        f"GridSearch tamamlandi — "
        f"DT best: {gs_dt.best_score_:.3f} | kNN best: {gs_knn.best_score_:.3f}"
    )
    return results


def _plot_gridsearch_heatmap(gs_dt, gs_knn, race_name: str) -> str:
    """GridSearch sonuclarini bar grafik olarak goster."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0D0D0D")

    for ax, gs, title, color in [
        (axes[0], gs_dt,  "Decision Tree GridSearch", "#E8002D"),
        (axes[1], gs_knn, "kNN GridSearch",           "#1E88E5"),
    ]:
        ax.set_facecolor("#1A1A1A")
        scores = gs.cv_results_["mean_test_score"]
        stds   = gs.cv_results_["std_test_score"]
        top_n  = min(10, len(scores))
        idx    = np.argsort(scores)[-top_n:][::-1]

        labels = [str(gs.cv_results_["params"][i]) for i in idx]
        labels = [lbl[:30] + "..." if len(lbl) > 30 else lbl for lbl in labels]

        bars = ax.barh(
            range(len(idx)),
            [scores[i] for i in idx],
            xerr=[stds[i] for i in idx],
            color=color, alpha=0.8,
            error_kw={"ecolor": "white", "capsize": 4},
        )
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(labels, fontsize=7, color="white")
        ax.set_xlabel("CV Dogrulugu", color="white")
        ax.set_title(f"{title}\n{race_name}", color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#2A2A2A")
        ax.set_xlim(0, 1.1)

        # En iyi parametreyi isaretler
        ax.axvline(gs.best_score_, color="white", linestyle="--", alpha=0.6, linewidth=1)
        ax.text(gs.best_score_ + 0.01, 0, f"Best: {gs.best_score_:.3f}",
                color="white", fontsize=8, va="center")

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"gridsearch_{race_name.lower().replace(' ', '_')}.png")
    plt.close(fig)
    return b64


# =============================================================
# 3. K-MEANS OPTIMİZASYONU
# =============================================================

def run_kmeans_optimization(X: np.ndarray, race_name: str) -> Dict[str, Any]:
    """
    Silhouette score ile K-Means parametre taramasi.
    n_clusters ve init kombinasyonlarini dener.
    """
    best_score  = -1.0
    best_params: Dict[str, Any] = {}
    all_results = []

    for n_clusters in KMEANS_GRID_PARAMS["n_clusters"]:
        if n_clusters >= len(X):
            continue
        for init in KMEANS_GRID_PARAMS["init"]:
            km = KMeans(
                n_clusters=n_clusters,
                init=init,
                random_state=RANDOM_STATE,
                n_init=10,
                max_iter=300,
            )
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            sil = float(silhouette_score(X, labels))
            all_results.append({
                "n_clusters": n_clusters,
                "init":       init,
                "silhouette": round(sil, 4),
                "inertia":    round(float(km.inertia_), 2),
            })
            if sil > best_score:
                best_score  = sil
                best_params = {"n_clusters": n_clusters, "init": init}

    all_results.sort(key=lambda r: r["silhouette"], reverse=True)

    chart_b64 = _plot_kmeans_optimization(all_results, race_name)

    log_success(f"K-Means optimizasyonu tamamlandi. En iyi: {best_params}, sil={best_score:.3f}")

    return {
        "best_params":  best_params,
        "best_score":   round(best_score, 4),
        "all_results":  all_results,
        "chart_b64":    chart_b64,
        "comment": (
            f"K-Means icin en iyi parametreler: {best_params} "
            f"(silhouette={best_score:.3f})"
        ),
    }


def _plot_kmeans_optimization(results: List[dict], race_name: str) -> str:
    """K-Means silhouette skorlarini goster."""
    if not results:
        return ""

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#1A1A1A")

    labels = [f"k={r['n_clusters']}, {r['init']}" for r in results]
    scores = [r["silhouette"] for r in results]
    colors = ["#E8002D" if s == max(scores) else "#555555" for s in scores]

    bars = ax.bar(range(len(labels)), scores, color=colors, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8, color="white")
    ax.set_ylabel("Silhouette Score", color="white")
    ax.set_title(f"K-Means Parametre Optimizasyonu — {race_name}", color="white", pad=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2A2A2A")
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0, color="#2A2A2A", linewidth=0.8)

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{s:.3f}", ha="center", fontsize=8, color="white")

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"kmeans_opt_{race_name.lower().replace(' ', '_')}.png")
    plt.close(fig)
    return b64


# =============================================================
# 4. ROC / AUC
# =============================================================

def compute_roc_auc(
    X: np.ndarray,
    y: np.ndarray,
    race_name: str,
) -> Dict[str, Any]:
    """
    Cross_val_predict ile cikan-fold olasilik tahminleri kullanarak
    ROC egrisi ve AUC hesapla. Kucuk veri setleri icin guvercilir yaklasim.
    """
    n_splits = min(CV_CONFIG["n_splits"], _max_safe_folds(y))
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=CV_CONFIG["shuffle"],
        random_state=RANDOM_STATE,
    )

    results: Dict[str, Any] = {}

    for model_name, clf in [
        ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE,
                                                  **{k: v for k, v in DECISION_TREE_CONFIG.items()
                                                     if k != "random_state"})),
        ("kNN",           KNeighborsClassifier(
                              n_neighbors=min(KNN_CONFIG["n_neighbors"], len(X) - 1),
                              metric=KNN_CONFIG["metric"])),
    ]:
        try:
            proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
            fpr, tpr, thresholds = roc_curve(y, proba[:, 1])
            auc_score = float(auc(fpr, tpr))

            # Optimum esik: Youden J (sensitivity + specificity - 1 maksimum)
            j_scores  = tpr - fpr
            best_idx  = int(np.argmax(j_scores))
            best_threshold = float(thresholds[best_idx])

            results[model_name] = {
                "fpr":            fpr.tolist(),
                "tpr":            tpr.tolist(),
                "auc":            round(auc_score, 4),
                "best_threshold": round(best_threshold, 4),
                "best_tpr":       round(float(tpr[best_idx]), 4),
                "best_fpr":       round(float(fpr[best_idx]), 4),
            }
        except Exception as e:
            logger.warning(f"ROC hesaplanamadi ({model_name}): {e}")
            results[model_name] = {"auc": 0.5, "error": str(e)}

    chart_b64 = _plot_roc_curves(results, race_name)
    results["chart_b64"] = chart_b64

    auc_dt  = results.get("Decision Tree", {}).get("auc", 0.5)
    auc_knn = results.get("kNN", {}).get("auc", 0.5)
    results["comment"] = (
        f"{race_name} ROC analizi: Decision Tree AUC={auc_dt:.3f}, "
        f"kNN AUC={auc_knn:.3f}. "
        f"{'Decision Tree' if auc_dt >= auc_knn else 'kNN'} daha iyi ayirt edici guc gostermektedir."
    )

    log_success(f"ROC/AUC tamamlandi — DT AUC={auc_dt:.3f} | kNN AUC={auc_knn:.3f}")
    return results


def _plot_roc_curves(roc_data: dict, race_name: str) -> str:
    """ROC egrisini ciz — her model ayri renk."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#1A1A1A")

    style_map = {
        "Decision Tree": ("#E8002D", "-"),
        "kNN":           ("#1E88E5", "--"),
    }

    for model_name, data in roc_data.items():
        if model_name == "chart_b64" or "comment" in model_name:
            continue
        if "fpr" not in data:
            continue
        color, ls = style_map.get(model_name, ("#888", "-"))
        ax.plot(data["fpr"], data["tpr"],
                color=color, linestyle=ls, linewidth=2.5,
                label=f"{model_name} (AUC = {data['auc']:.3f})")
        # Optimum nokta
        if "best_fpr" in data:
            ax.scatter(data["best_fpr"], data["best_tpr"],
                       color=color, s=100, zorder=5, marker="*")

    ax.plot([0, 1], [0, 1], "k--", color="#555555", linewidth=1.5, label="Rastgele Siniflandirici")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="#555555")

    ax.set_xlabel("False Positive Rate", color="white", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", color="white", fontsize=11)
    ax.set_title(f"ROC Egrisi — {race_name}", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2A2A2A")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=10, loc="lower right")
    ax.grid(color="#2A2A2A", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"roc_{race_name.lower().replace(' ', '_')}.png")
    plt.close(fig)
    return b64


# =============================================================
# 5. MODEL KARSILASTIRMA TABLOSU
# =============================================================

def build_model_comparison(
    cv_results: dict,
    gs_results: dict,
    roc_results: dict,
    race_name: str,
) -> Dict[str, Any]:
    """
    Kapsamli model karsilastirma tablosu:
    Accuracy, Precision, Recall, F1, AUC, CV Mean, CV Std, Train Time
    """
    rows = []

    for model_key, model_name in [("decision_tree", "Decision Tree"), ("knn", "kNN")]:
        cv_data  = cv_results.get(model_key, {})
        gs_data  = gs_results.get(model_key, {})
        roc_data = roc_results.get(model_name, {})

        rows.append({
            "Model":        model_name,
            "CV Mean Acc":  cv_data.get("mean_accuracy", 0),
            "CV Std":       cv_data.get("std_accuracy", 0),
            "CV Precision": cv_data.get("mean_precision", 0),
            "CV Recall":    cv_data.get("mean_recall", 0),
            "CV F1":        cv_data.get("mean_f1", 0),
            "AUC":          roc_data.get("auc", 0.5),
            "GS Best Score":gs_data.get("best_score", 0),
            "Overfit Gap":  cv_data.get("overfit_gap", 0),
            "Best Params":  str(gs_data.get("best_params", {})),
        })

    df = pd.DataFrame(rows)

    # En iyi model: CV Mean Acc'a gore
    best_idx   = df["CV Mean Acc"].idxmax()
    best_model = df.loc[best_idx, "Model"]

    # Yorum olustur
    dt_row  = df[df["Model"] == "Decision Tree"].iloc[0]
    knn_row = df[df["Model"] == "kNN"].iloc[0]

    interpretation = []
    if dt_row["AUC"] > knn_row["AUC"] + 0.05:
        interpretation.append(f"Decision Tree, kNN'e kiyasla belirgin sekilde daha iyi ayirt edici guc gostermektedir (AUC: {dt_row['AUC']:.3f} vs {knn_row['AUC']:.3f}).")
    elif knn_row["AUC"] > dt_row["AUC"] + 0.05:
        interpretation.append(f"kNN, Decision Tree'ye kiyasla daha iyi AUC elde etmistir ({knn_row['AUC']:.3f} vs {dt_row['AUC']:.3f}).")
    else:
        interpretation.append(f"Iki modelin AUC degerleri birbirine yakindir (DT={dt_row['AUC']:.3f}, kNN={knn_row['AUC']:.3f}).")

    if dt_row["Overfit Gap"] > 0.15:
        interpretation.append(f"Decision Tree'de asiri ogrenme riski vardir (egitim-test farki: {dt_row['Overfit Gap']:.3f}).")
    if knn_row["Overfit Gap"] > 0.15:
        interpretation.append(f"kNN'de asiri ogrenme riski vardir (egitim-test farki: {knn_row['Overfit Gap']:.3f}).")

    chart_b64 = _plot_model_comparison_radar(df, race_name)

    log_success(f"Model karsilastirma tablosu olusturuldu. En iyi model: {best_model}")

    return {
        "dataframe":      df.to_dict("records"),
        "best_model":     best_model,
        "interpretation": interpretation,
        "chart_b64":      chart_b64,
        "comment": (
            f"{race_name} — En iyi model: {best_model} "
            f"(CV ort. dogruluk: %{df.loc[best_idx, 'CV Mean Acc']*100:.1f})"
        ),
    }


def _plot_model_comparison_radar(df: pd.DataFrame, race_name: str) -> str:
    """Model karsilastirma bar grafigi."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#1A1A1A")

    metrics = ["CV Mean Acc", "CV Precision", "CV Recall", "CV F1", "AUC"]
    x = np.arange(len(metrics))
    width = 0.35

    colors = ["#E8002D", "#1E88E5"]
    for i, (_, row) in enumerate(df.iterrows()):
        vals   = [float(row[m]) for m in metrics]
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=row["Model"],
                        color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=8, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="white", fontsize=10)
    ax.set_ylabel("Skor", color="white", fontsize=11)
    ax.set_title(f"Model Karsilastirma — {race_name}", color="white", fontsize=13, pad=12)
    ax.set_ylim(0, 1.2)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2A2A2A")
    ax.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=10)
    ax.grid(axis="y", color="#2A2A2A", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _save_fig(fig, f"model_comparison_{race_name.lower().replace(' ', '_')}.png")
    plt.close(fig)
    return b64


# =============================================================
# KAYDETME FONKSİYONLARI
# =============================================================

def _save_comparison_csv(comparison: dict, race_name: str) -> None:
    """Model karsilastirma tablosunu CSV olarak kaydet."""
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        slug = race_name.lower().replace(" ", "_")
        path = TABLES_DIR / f"model_comparison_{slug}.csv"
        rows = comparison.get("dataframe", [])
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            logger.info(f"Karsilastirma CSV kaydedildi: {path.name}")
    except Exception as e:
        logger.warning(f"CSV kaydetme hatasi: {e}")


def _save_comparison_markdown(comparison: dict, race_name: str) -> None:
    """Model karsilastirma tablosunu Markdown olarak kaydet."""
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        slug = race_name.lower().replace(" ", "_")
        path = TABLES_DIR / f"model_comparison_{slug}.md"
        rows = comparison.get("dataframe", [])
        if not rows:
            return

        df = pd.DataFrame(rows)
        numeric_cols = ["CV Mean Acc", "CV Std", "CV Precision", "CV Recall",
                        "CV F1", "AUC", "GS Best Score", "Overfit Gap"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{float(x):.4f}")

        md_lines = [
            f"# Model Karsilastirma — {race_name}",
            "",
            df.to_markdown(index=False),
            "",
            "## Yorum",
            "",
        ]
        for line in comparison.get("interpretation", []):
            md_lines.append(f"- {line}")
        md_lines.append(f"\n**En iyi model:** {comparison.get('best_model', 'N/A')}")

        path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info(f"Karsilastirma Markdown kaydedildi: {path.name}")
    except Exception as e:
        logger.warning(f"Markdown kaydetme hatasi: {e}")


def _save_cv_csv(cv_results: dict, race_name: str) -> None:
    """Cross-validation sonuclarini CSV olarak kaydet."""
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        slug = race_name.lower().replace(" ", "_")
        path = TABLES_DIR / f"cv_results_{slug}.csv"

        rows = []
        for model_key, model_name in [("decision_tree", "Decision Tree"), ("knn", "kNN")]:
            data = cv_results.get(model_key, {})
            fold_accs = data.get("fold_accuracies", [])
            for i, acc in enumerate(fold_accs, 1):
                rows.append({
                    "race":          race_name,
                    "model":         model_name,
                    "fold":          i,
                    "accuracy":      acc,
                    "mean_accuracy": data.get("mean_accuracy"),
                    "std_accuracy":  data.get("std_accuracy"),
                    "mean_f1":       data.get("mean_f1"),
                    "auc":           0,  # ROC'dan ayri hesaplaniyor
                })

        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            logger.info(f"CV sonuclari CSV kaydedildi: {path.name}")
    except Exception as e:
        logger.warning(f"CV CSV kaydetme hatasi: {e}")


# =============================================================
# YARDIMCI FONKSİYONLAR
# =============================================================

def _max_safe_folds(y: np.ndarray) -> int:
    """Stratified split icin guvercil maksimum fold sayisi."""
    min_class = int(np.bincount(y).min())
    return max(2, min(5, min_class))


def _fig_to_b64(fig) -> str:
    """Matplotlib figure'i base64 PNG stringine donustur."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _save_fig(fig, filename: str) -> None:
    """Figure'i disk'e kaydet."""
    try:
        FIGURES_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_EVAL_DIR / filename, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    except Exception as e:
        logger.warning(f"Figure kaydetme hatasi ({filename}): {e}")


def _ensure_dirs() -> None:
    FIGURES_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
