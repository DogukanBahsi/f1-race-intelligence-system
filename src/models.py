"""
src/models.py
-------------
F1 Race Intelligence System - Makine Öğrenmesi Modülü
K-Means Clustering, Decision Tree ve kNN sınıflandırma.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, silhouette_score
)
from sklearn.impute import SimpleImputer

from config import KMEANS_CONFIG, DECISION_TREE_CONFIG, KNN_CONFIG, KMEANS_K_RANGE
from src.logger import get_logger, log_success
from src.feature_engineering import get_driver_feature_matrix

logger = get_logger("Models")

# Modeller için dizin
MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "exports" / "models"


# ─────────────────────────────────────────────────────────────
# ANA ML PIPELINE
# ─────────────────────────────────────────────────────────────

def run_all_models(df: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Tüm ML modellerini eğitir ve sonuçları döndürür.
    """
    logger.info(f"ML pipeline başlıyor: {race_name}")

    results = {}

    # Özellik matrisi oluştur
    feature_matrix = get_driver_feature_matrix(df)
    if feature_matrix.empty or len(feature_matrix) < 3:
        logger.warning("Yeterli pilot yok (minimum 3 gerekli). ML atlanıyor.")
        return {"error": "Yeterli pilot verisi yok.", "feature_matrix": pd.DataFrame()}

    results["feature_matrix"] = feature_matrix

    # ── A) K-Means Clustering ────────────────────────────────
    results["kmeans"] = run_kmeans(feature_matrix, race_name)

    # ── B & C) Sınıflandırma: DT + kNN ───────────────────────
    clf_results = run_classifiers(feature_matrix, df, race_name)
    results["decision_tree"] = clf_results.get("decision_tree", {})
    results["knn"]           = clf_results.get("knn", {})
    results["comparison"]    = clf_results.get("comparison", {})

    # ── D) Akademik Değerlendirme Pipeline ───────────────────
    try:
        from src.model_evaluation import run_full_evaluation
        results["evaluation"] = run_full_evaluation(feature_matrix, df, race_name)
    except Exception as _eval_err:
        logger.warning(f"Akademik değerlendirme hatası: {_eval_err}")
        results["evaluation"] = {}

    # ── E) XAI (SHAP / Permutation Importance) ───────────────
    try:
        from src.xai import run_xai_analysis
        results["xai"] = run_xai_analysis(feature_matrix, df, race_name)
    except Exception as _xai_err:
        logger.warning(f"XAI analizi hatası: {_xai_err}")
        results["xai"] = {}

    # ── F) Hata Analizi ──────────────────────────────────────
    try:
        from src.error_analysis import run_error_analysis
        results["error_analysis"] = run_error_analysis(feature_matrix, df, race_name)
    except Exception as _ea_err:
        logger.warning(f"Hata analizi hatası: {_ea_err}")
        results["error_analysis"] = {}

    # ── G) İstatistiksel Testler ─────────────────────────────
    try:
        from src.statistical_tests import run_statistical_tests
        results["statistical_tests"] = run_statistical_tests(feature_matrix, df, race_name)
    except Exception as _st_err:
        logger.warning(f"İstatistiksel test hatası: {_st_err}")
        results["statistical_tests"] = {}

    # ── H) Gelişmiş Akademik Analiz (Learning Curves, Correlation, Ablation, Overfit)
    try:
        from src.advanced_analysis import run_advanced_analysis
        results["advanced"] = run_advanced_analysis(feature_matrix, df, race_name)
        adv = results["advanced"]
        lc_ok  = "error" not in adv.get("learning_curves", {"error": "?"})
        abl_ok = "error" not in adv.get("ablation_study", {"error": "?"})
        logger.info(f"Gelişmiş analiz: LC={'✓' if lc_ok else '✗'}, Ablation={'✓' if abl_ok else '✗'}")
    except Exception as _adv_err:
        logger.warning(f"Gelişmiş analiz hatası: {_adv_err}")
        results["advanced"] = {}

    log_success(f"ML pipeline tamamlandı: {race_name}")
    return results


# ─────────────────────────────────────────────────────────────
# A) K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────

def run_kmeans(feature_matrix: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Pilotları performans kümelerine ayırır.
    Cluster etiketleri: 0=Stabil, 1=Agresif, 2=Düzensiz (sıralama sonraya bırakılır)
    """
    # Kullanılacak özellikler
    kmeans_features = [
        "average_lap_time",
        "lap_time_std",
        "consistency_score",
        "tire_degradation_rate",
        "sector_consistency",
    ]
    available = [f for f in kmeans_features if f in feature_matrix.columns]

    if len(available) < 2:
        return {"error": "Yeterli özellik yok."}

    X = feature_matrix[available].copy()
    drivers = feature_matrix["Driver"].tolist() if "Driver" in feature_matrix.columns else []

    # Eksik değerleri doldur
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X)

    # Normalize et
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # ── Elbow Method ile optimal k ────────────────────────────
    inertias = []
    sil_scores = []
    k_range = list(KMEANS_K_RANGE)

    for k in k_range:
        if k >= len(feature_matrix):
            break
        km = KMeans(n_clusters=k, random_state=KMEANS_CONFIG["random_state"],
                    n_init=KMEANS_CONFIG["n_init"], max_iter=KMEANS_CONFIG["max_iter"])
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        if k > 1:
            sil = silhouette_score(X_scaled, km.labels_)
            sil_scores.append(float(sil))
        else:
            sil_scores.append(0.0)

    # Belirlenen k ile final model
    n_clusters = min(KMEANS_CONFIG["n_clusters"], len(feature_matrix) - 1)
    km_final = KMeans(
        n_clusters=n_clusters,
        random_state=KMEANS_CONFIG["random_state"],
        n_init=KMEANS_CONFIG["n_init"],
        max_iter=KMEANS_CONFIG["max_iter"],
    )
    km_final.fit(X_scaled)
    labels = km_final.labels_

    # Küme etiketlerini oluştur
    # Kümeleri ortalama tur zamanına göre sırala: en hızlı = 0
    if "average_lap_time" in available:
        lt_idx = available.index("average_lap_time")
        cluster_means = {}
        for c in range(n_clusters):
            mask = labels == c
            cluster_means[c] = X_scaled[mask, lt_idx].mean() if mask.any() else 0

        # Sırala: en düşük ortalama tur zamanı = "Stabil/Hızlı"
        sorted_clusters = sorted(cluster_means, key=cluster_means.get)
        label_map = {
            sorted_clusters[0]: "Stabil & Hızlı",
            sorted_clusters[-1]: "Düzensiz",
        }
        if n_clusters == 3:
            label_map[sorted_clusters[1]] = "Agresif"
        else:
            for i, c in enumerate(sorted_clusters[1:-1]):
                label_map[c] = f"Orta Grup {i+1}"
    else:
        label_map = {i: f"Küme {i+1}" for i in range(n_clusters)}

    cluster_labels = [label_map.get(l, f"Küme {l}") for l in labels]

    # Sürücü-küme atması
    driver_clusters = []
    for i, driver in enumerate(drivers):
        driver_clusters.append({
            "Driver":       driver,
            "cluster_id":   int(labels[i]),
            "cluster_label": cluster_labels[i],
        })

    # Küme özeti
    cluster_summary = {}
    for c_id in range(n_clusters):
        members = [d["Driver"] for d in driver_clusters if d["cluster_id"] == c_id]
        cluster_summary[label_map.get(c_id, f"Küme {c_id}")] = members

    log_success(f"K-Means tamamlandı: {n_clusters} küme, "
                f"{len(drivers)} pilot sınıflandırıldı.")

    return {
        "n_clusters":       n_clusters,
        "driver_clusters":  driver_clusters,
        "cluster_summary":  cluster_summary,
        "features_used":    available,
        "elbow_inertias":   list(zip(k_range[:len(inertias)], inertias)),
        "sil_scores":       list(zip(k_range[:len(sil_scores)], sil_scores)),
        "X_scaled":         X_scaled.tolist(),   # Görselleştirme için
        "labels":           labels.tolist(),
        "label_map":        {int(k): v for k, v in label_map.items()},
        "comment": (
            f"{race_name} yarışında {n_clusters} performans grubu tespit edilmiştir. "
            f"'Stabil & Hızlı' grubundaki pilotlar: "
            f"{', '.join(cluster_summary.get('Stabil & Hızlı', ['N/A']))}."
        ),
    }


# ─────────────────────────────────────────────────────────────
# B) DECISION TREE + C) kNN SINIFLANDIRMA
# ─────────────────────────────────────────────────────────────

def run_classifiers(feature_matrix: pd.DataFrame,
                    df: pd.DataFrame,
                    race_name: str) -> Dict[str, Any]:
    """
    Decision Tree ve kNN modellerini aynı veri üzerinde çalıştırır.
    Target: strong_performance
    Karşılaştırma raporu üretir.
    """
    # ── Target değişkeni üret ─────────────────────────────────
    feature_matrix = _create_target(feature_matrix, df)

    if "strong_performance" not in feature_matrix.columns:
        return {"error": "Target değişkeni oluşturulamadı."}

    # Özellikler
    clf_features = [
        "average_lap_time",
        "tire_degradation_rate",
        "consistency_score",
        "pit_stop_impact",
        "sector_consistency",
    ]
    # AirTemp / TrackTemp ortalama ekle
    for col in ["AirTemp", "TrackTemp"]:
        avg_col = f"avg_{col.lower()}"
        if avg_col in feature_matrix.columns:
            clf_features.append(avg_col)
        elif col in feature_matrix.columns:
            clf_features.append(col)

    available = [f for f in clf_features if f in feature_matrix.columns]
    if len(available) < 2:
        return {"error": "Yeterli özellik yok."}

    X = feature_matrix[available].copy()
    y = feature_matrix["strong_performance"].copy()
    drivers = feature_matrix["Driver"].tolist() if "Driver" in feature_matrix.columns else []

    # Eksik değerleri doldur
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X)

    # Yeterli veri var mı?
    if len(X_imp) < 4:
        return {"error": f"Sınıflandırma için yeterli pilot yok ({len(X_imp)} < 4)."}

    # Normalize
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Train/test split
    test_size = 0.3 if len(X_imp) >= 6 else 0.2
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        # Küçük veri seti için stratify olmadan böl
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

    results = {}

    # ── Decision Tree ─────────────────────────────────────────
    dt_result = _train_decision_tree(
        X_train, X_test, y_train, y_test, available, race_name
    )
    results["decision_tree"] = dt_result

    # ── kNN ──────────────────────────────────────────────────
    knn_result = _train_knn(X_train, X_test, y_train, y_test, race_name)
    results["knn"] = knn_result

    # ── Karşılaştırma ─────────────────────────────────────────
    results["comparison"] = {
        "decision_tree_accuracy": dt_result.get("accuracy", 0),
        "knn_accuracy":           knn_result.get("accuracy", 0),
        "winner": (
            "Decision Tree" if dt_result.get("accuracy", 0) >= knn_result.get("accuracy", 0)
            else "kNN"
        ),
        "features_used": available,
        "n_drivers":     len(X_imp),
        "n_strong":      int(y.sum()),
        "comment": (
            f"{race_name}: Decision Tree doğruluk oranı "
            f"%{dt_result.get('accuracy', 0)*100:.1f}, "
            f"kNN doğruluk oranı %{knn_result.get('accuracy', 0)*100:.1f}. "
            f"Bu yarışta daha başarılı model: "
            f"{'Decision Tree' if dt_result.get('accuracy', 0) >= knn_result.get('accuracy', 0) else 'kNN'}."
        ),
    }

    return results


def _train_decision_tree(X_train, X_test, y_train, y_test,
                          feature_names: List[str], race_name: str) -> dict:
    """Decision Tree modelini eğit, değerlendir ve feature importance üret."""
    dt = DecisionTreeClassifier(**DECISION_TREE_CONFIG)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred).tolist()
    clf_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Feature importance
    importances = dt.feature_importances_
    feat_imp = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1], reverse=True
    )

    most_important = feat_imp[0][0] if feat_imp else "N/A"

    log_success(f"Decision Tree model trained. Accuracy: {acc:.3f}")

    return {
        "accuracy":         round(float(acc), 4),
        "confusion_matrix": cm,
        "classification_report": clf_rep,
        "feature_importance": feat_imp,
        "most_important_feature": most_important,
        "comment": (
            f"Decision Tree modeli {race_name} yarışında %{acc*100:.1f} doğruluk oranı elde etti. "
            f"Yarış performansını en çok etkileyen faktör: {most_important}."
        ),
    }


def _train_knn(X_train, X_test, y_train, y_test, race_name: str) -> dict:
    """kNN modelini eğit ve değerlendir."""
    n_neighbors = min(KNN_CONFIG["n_neighbors"], len(X_train))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=KNN_CONFIG["metric"])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred).tolist()
    clf_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    log_success(f"kNN model trained (k={n_neighbors}). Accuracy: {acc:.3f}")

    return {
        "accuracy":         round(float(acc), 4),
        "k":                n_neighbors,
        "confusion_matrix": cm,
        "classification_report": clf_rep,
        "comment": (
            f"kNN modeli (k={n_neighbors}) {race_name} yarışında "
            f"%{acc*100:.1f} doğruluk oranı elde etti."
        ),
    }


# ─────────────────────────────────────────────────────────────
# TARGET ÜRETME
# ─────────────────────────────────────────────────────────────

def _create_target(feature_matrix: pd.DataFrame,
                    df: pd.DataFrame) -> pd.DataFrame:
    """
    strong_performance target değişkenini oluştur.
    Kural: Ortalama tur zamanı yarış medyanından iyi VE
           consistency_score yüksek (üst %50) ise → 1 (güçlü performans)
    """
    fm = feature_matrix.copy()

    if "average_lap_time" not in fm.columns:
        return fm

    # Yarış medyan tur zamanı
    race_median = fm["average_lap_time"].median()
    # Medyanın üst yarısı
    consistency_median = (
        fm["consistency_score"].median()
        if "consistency_score" in fm.columns else 0
    )

    # İki koşul birden sağlanmalı
    fast_mask    = fm["average_lap_time"] <= race_median
    stable_mask  = (
        fm["consistency_score"] >= consistency_median
        if "consistency_score" in fm.columns
        else pd.Series(True, index=fm.index)
    )

    fm["strong_performance"] = (fast_mask & stable_mask).astype(int)

    n_strong = fm["strong_performance"].sum()
    logger.debug(f"Target: {n_strong}/{len(fm)} pilot 'strong_performance=1'")

    # Dengesiz sınıf kontrolü: en az 1 pozitif ve 1 negatif örnek olmalı
    if fm["strong_performance"].nunique() < 2:
        # En azından en hızlı pilotu 1 yap
        fm["strong_performance"] = 0
        if "average_lap_time" in fm.columns:
            fm.loc[fm["average_lap_time"].idxmin(), "strong_performance"] = 1

    return fm


if __name__ == "__main__":
    from src.utils import generate_sample_lap_data
    from src.preprocessing import preprocess_laps
    from src.feature_engineering import engineer_features

    sample   = generate_sample_lap_data("Bahrain")
    clean    = preprocess_laps(sample, "Bahrain")
    featured = engineer_features(clean, "Bahrain")
    results  = run_all_models(featured, "Bahrain")

    print("\n✅ K-Means:")
    for cluster, members in results.get("kmeans", {}).get("cluster_summary", {}).items():
        print(f"   {cluster}: {members}")

    print("\n✅ Decision Tree Accuracy:", results.get("decision_tree", {}).get("accuracy"))
    print("✅ kNN Accuracy:", results.get("knn", {}).get("accuracy"))
    print("\n✅ En önemli özellik:", results.get("decision_tree", {}).get("most_important_feature"))
