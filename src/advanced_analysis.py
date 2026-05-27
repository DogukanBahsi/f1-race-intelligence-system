"""
src/advanced_analysis.py
------------------------
F1 Race Intelligence System — Gelişmiş Akademik Analiz Modülü
- Learning Curves (bias-variance tradeoff görselleştirme)
- Feature Correlation Heatmap
- Ablation Study (özellik çıkarma etkisi)
- Overfitting Analysis (train vs test accuracy)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from config import RANDOM_STATE, FIGURES_EVAL_DIR, TABLES_DIR
from src.logger import get_logger

logger = get_logger("AdvancedAnalysis")

# ─── Çıktı dizinleri ────────────────────────────────────────────────────────
_EVAL_DIR   = Path(FIGURES_EVAL_DIR)
_TABLES_DIR = Path(TABLES_DIR)

# ─── Yardımcı: slug ─────────────────────────────────────────────────────────
def _slug(race_name: str) -> str:
    return race_name.lower().replace(" ", "_").replace("-", "_")


# ─── Yardımcı: X/y hazırla ──────────────────────────────────────────────────
def _prepare_Xy(feature_matrix: pd.DataFrame):
    """Feature matrix'den X (numpy) ve y (numpy binary) hazırlar."""
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = [c for c in ["strong_performance", "cluster", "Driver"] if c in numeric_cols]
    feat_cols = [c for c in numeric_cols if c not in drop_cols]

    X_df = feature_matrix[feat_cols].copy()
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_df)

    # Hedef: strong_performance kolonu varsa kullan, yoksa median-based binary
    if "strong_performance" in feature_matrix.columns:
        y = feature_matrix["strong_performance"].values.astype(int)
    else:
        thr = feature_matrix["average_lap_time"].median() if "average_lap_time" in feature_matrix.columns else None
        if thr is not None:
            y = (feature_matrix["average_lap_time"] <= thr).astype(int).values
        else:
            y = (np.arange(len(feature_matrix)) < len(feature_matrix) // 2).astype(int)

    return X, y, feat_cols


def _max_safe_folds(y: np.ndarray, max_folds: int = 5) -> int:
    """Küçük veri setlerinde CV kırılmaması için güvenli fold sayısı."""
    counts = np.bincount(y)
    return max(2, min(max_folds, int(counts.min())))


# ─────────────────────────────────────────────────────────────────────────────
# 1. LEARNING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def run_learning_curves(feature_matrix: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Decision Tree ve kNN için learning curve hesaplar ve görselleştirir.

    Learning curve: eğitim seti büyüklüğü artarken train/validation accuracy
    nasıl değişiyor? → Bias-Variance tradeoff görselleştirmesi.
    """
    logger.info(f"Learning Curves hesaplanıyor: {race_name}")

    try:
        X, y, feat_cols = _prepare_Xy(feature_matrix)
        n_samples = len(X)

        if n_samples < 6:
            logger.warning(f"Yeterli örnek yok (n={n_samples}), learning curve atlanıyor.")
            return {"error": f"Yeterli örnek yok (n={n_samples})"}

        n_folds = _max_safe_folds(y, max_folds=3)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

        # Eğitim boyutları: fold-aware sınır içinde (n_samples * (n_folds-1)/n_folds = max train)
        max_train = int(n_samples * (n_folds - 1) / n_folds) - 1
        min_train = max(2 * n_folds, int(n_samples * 0.3))
        min_train = min(min_train, max_train - 1)
        train_sizes = np.linspace(min_train, max_train, 6, dtype=int)
        train_sizes = np.unique(np.clip(train_sizes, 2, max_train))

        models = {
            "Decision Tree": DecisionTreeClassifier(
                max_depth=3, random_state=RANDOM_STATE
            ),
            "kNN": KNeighborsClassifier(n_neighbors=min(3, n_samples - 1)),
        }

        lc_results = {}
        for model_name, model in models.items():
            try:
                sizes, train_scores, val_scores = learning_curve(
                    model, X, y,
                    train_sizes=train_sizes,
                    cv=cv,
                    scoring="accuracy",
                    n_jobs=-1,
                    error_score=0.0,
                )
                lc_results[model_name] = {
                    "train_sizes":      sizes.tolist(),
                    "train_mean":       train_scores.mean(axis=1).tolist(),
                    "train_std":        train_scores.std(axis=1).tolist(),
                    "val_mean":         val_scores.mean(axis=1).tolist(),
                    "val_std":          val_scores.std(axis=1).tolist(),
                }
                logger.info(f"  {model_name} learning curve OK (max_val_acc={max(val_scores.mean(axis=1)):.3f})")
            except Exception as e:
                logger.warning(f"  {model_name} learning curve hatası: {e}")
                lc_results[model_name] = {"error": str(e)}

        # ── Grafik ──────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Learning Curves — {race_name} GP", fontsize=14, fontweight="bold")

        colors = {"Decision Tree": ("#2196F3", "#BBDEFB"), "kNN": ("#FF9800", "#FFE0B2")}

        for ax, (model_name, data) in zip(axes, lc_results.items()):
            if "error" in data:
                ax.text(0.5, 0.5, f"Hata:\n{data['error']}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="red")
                ax.set_title(model_name)
                continue

            sizes     = np.array(data["train_sizes"])
            tm, ts    = np.array(data["train_mean"]), np.array(data["train_std"])
            vm, vs    = np.array(data["val_mean"]),   np.array(data["val_std"])
            c_main, c_light = colors[model_name]

            ax.plot(sizes, tm, "o-", color=c_main, label="Eğitim Doğruluğu", lw=2)
            ax.fill_between(sizes, tm - ts, tm + ts, alpha=0.2, color=c_main)
            ax.plot(sizes, vm, "s--", color="#4CAF50", label="Doğrulama Doğruluğu", lw=2)
            ax.fill_between(sizes, vm - vs, vm + vs, alpha=0.2, color="#4CAF50")

            ax.set_title(f"{model_name}", fontsize=12)
            ax.set_xlabel("Eğitim Seti Boyutu", fontsize=10)
            ax.set_ylabel("Doğruluk", fontsize=10)
            ax.set_ylim(0.0, 1.05)
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

            # Bias/Variance notunu ekle
            if len(vm) > 0 and len(tm) > 0:
                gap = float(tm[-1] - vm[-1])
                if gap > 0.15:
                    note = "⚠ Yüksek Varyans (Overfit)"
                elif vm[-1] < 0.65:
                    note = "⚠ Yüksek Bias (Underfit)"
                else:
                    note = "✓ Dengeli"
                ax.text(0.05, 0.05, note, transform=ax.transAxes,
                        fontsize=9, color="darkblue",
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

        plt.tight_layout()
        slug = _slug(race_name)
        out_path = _EVAL_DIR / f"learning_curves_{slug}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Learning curve kaydedildi: {out_path.name}")

        return {
            "curves":    lc_results,
            "n_folds":   n_folds,
            "n_samples": n_samples,
            "chart":     str(out_path),
        }

    except Exception as e:
        logger.error(f"Learning curve genel hatası: {e}", exc_info=True)
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_correlation(feature_matrix: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Özellikler arası Pearson korelasyon matrisi hesaplar ve ısı haritası üretir.
    Yüksek korelasyon (>0.8): multicollinearity riski.
    """
    logger.info(f"Feature Correlation hesaplanıyor: {race_name}")

    try:
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
        drop_cols    = [c for c in ["strong_performance", "cluster"] if c in numeric_cols]
        feat_cols    = [c for c in numeric_cols if c not in drop_cols]

        if len(feat_cols) < 2:
            return {"error": "Yeterli sayısal özellik yok."}

        X_df = feature_matrix[feat_cols].copy()
        imp  = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(imp.fit_transform(X_df), columns=feat_cols)

        corr_matrix = X_imp.corr(method="pearson")

        # ── Yüksek korelasyon çiftleri ──────────────────────────
        high_corr_pairs = []
        for i, c1 in enumerate(feat_cols):
            for j, c2 in enumerate(feat_cols):
                if j <= i:
                    continue
                val = abs(corr_matrix.loc[c1, c2])
                if val > 0.7:
                    high_corr_pairs.append({
                        "feature_1": c1,
                        "feature_2": c2,
                        "correlation": round(float(val), 4),
                    })

        high_corr_pairs.sort(key=lambda x: -x["correlation"])

        # ── Grafik ──────────────────────────────────────────────
        n_feat = len(feat_cols)
        fig_size = max(8, int(n_feat * 0.8))
        fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size),
                                 gridspec_kw={"width_ratios": [3, 1]})

        # Heatmap
        ax_heat = axes[0]
        import matplotlib.colors as mcolors
        cmap = plt.get_cmap("RdYlGn_r")
        im   = ax_heat.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax_heat, shrink=0.8)
        ax_heat.set_xticks(range(n_feat))
        ax_heat.set_yticks(range(n_feat))

        # Kısa isimler
        short_names = [c.replace("_", "\n") for c in feat_cols]
        ax_heat.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax_heat.set_yticklabels(short_names, fontsize=7)
        ax_heat.set_title(f"Feature Korelasyon Matrisi — {race_name} GP", fontsize=11, pad=10)

        # Değerleri hücrelere yaz
        for i in range(n_feat):
            for j in range(n_feat):
                val = corr_matrix.values[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=6, color=color)

        # Yüksek korelasyon tablosu
        ax_tbl = axes[1]
        ax_tbl.axis("off")
        if high_corr_pairs:
            tbl_data = [[p["feature_1"][:12], p["feature_2"][:12], f"{p['correlation']:.3f}"]
                        for p in high_corr_pairs[:10]]
            tbl = ax_tbl.table(
                cellText=tbl_data,
                colLabels=["Özellik 1", "Özellik 2", "Korelasyon"],
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.4)
            ax_tbl.set_title("Yüksek Korelasyon (>0.7)", fontsize=9, pad=10)

            # Renk kodlama
            for k, pair in enumerate(high_corr_pairs[:10]):
                c = float(pair["correlation"])
                color = "#FF7043" if c > 0.9 else ("#FFB300" if c > 0.8 else "#FFF176")
                for col_idx in range(3):
                    tbl[k + 1, col_idx].set_facecolor(color)
        else:
            ax_tbl.text(0.5, 0.5, "Yüksek korelasyon\nbulunamadı (>0.7)",
                        ha="center", va="center", transform=ax_tbl.transAxes, fontsize=9)

        plt.tight_layout()
        slug = _slug(race_name)
        out_path = _EVAL_DIR / f"feature_correlation_{slug}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Feature correlation kaydedildi: {out_path.name}")

        # CSV kaydet
        corr_csv = _TABLES_DIR / f"feature_correlation_{slug}.csv"
        corr_matrix.to_csv(corr_csv)

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_corr_pairs":    high_corr_pairs,
            "n_features":         n_feat,
            "chart":              str(out_path),
        }

    except Exception as e:
        logger.error(f"Feature correlation hatası: {e}", exc_info=True)
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study(feature_matrix: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Ablation Study: her özelliği tek tek çıkardığımızda model performansı nasıl değişiyor?
    Drop-one-out yaklaşımı ile her özelliğin katkısını ölçer.
    """
    logger.info(f"Ablation Study çalışıyor: {race_name}")

    try:
        X, y, feat_cols = _prepare_Xy(feature_matrix)
        n_samples = len(X)

        if n_samples < 6 or len(feat_cols) < 2:
            return {"error": f"Ablation için yetersiz veri (n={n_samples}, nfeat={len(feat_cols)})"}

        n_folds = _max_safe_folds(y, max_folds=3)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        # Baseline (tüm özellikler)
        dt  = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
        knn = KNeighborsClassifier(n_neighbors=min(3, n_samples - 1))

        baseline_dt  = float(cross_val_score(dt,  X_sc, y, cv=cv, scoring="accuracy").mean())
        baseline_knn = float(cross_val_score(knn, X_sc, y, cv=cv, scoring="accuracy").mean())
        logger.info(f"  Baseline -> DT={baseline_dt:.3f}, kNN={baseline_knn:.3f}")

        # Drop-one-out
        ablation_rows = []
        for i, feat in enumerate(feat_cols):
            X_drop = np.delete(X_sc, i, axis=1)
            try:
                acc_dt  = float(cross_val_score(dt,  X_drop, y, cv=cv, scoring="accuracy").mean())
                acc_knn = float(cross_val_score(knn, X_drop, y, cv=cv, scoring="accuracy").mean())
                delta_dt  = round(acc_dt  - baseline_dt,  4)
                delta_knn = round(acc_knn - baseline_knn, 4)
                ablation_rows.append({
                    "feature":       feat,
                    "dt_without":    round(acc_dt,  4),
                    "knn_without":   round(acc_knn, 4),
                    "delta_dt":      delta_dt,
                    "delta_knn":     delta_knn,
                    "avg_delta":     round((delta_dt + delta_knn) / 2, 4),
                })
            except Exception as e:
                logger.warning(f"  Ablation {feat} hatası: {e}")

        ablation_df = pd.DataFrame(ablation_rows).sort_values("avg_delta")

        # ── Grafik ──────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(feat_cols) * 0.45 + 1)))
        fig.suptitle(f"Ablation Study (Drop-One-Out) — {race_name} GP",
                     fontsize=13, fontweight="bold")

        def _barh_model(ax, col_delta, col_acc, model_name, color_pos, color_neg):
            vals   = ablation_df[col_delta].values
            labels = ablation_df["feature"].values
            accs   = ablation_df[col_acc].values
            colors = [color_neg if v < 0 else color_pos for v in vals]
            bars   = ax.barh(range(len(labels)), vals, color=colors, alpha=0.85, edgecolor="white")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(x=0, color="black", lw=1.2, ls="--")
            ax.set_xlabel("Δ Doğruluk (Baseline'a Göre)", fontsize=9)
            ax.set_title(f"{model_name}\nBaseline = {baseline_dt if 'DT' in model_name else baseline_knn:.3f}",
                         fontsize=10)
            ax.grid(axis="x", alpha=0.3)
            for j, (bar, acc) in enumerate(zip(bars, accs)):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{acc:.3f}", va="center", fontsize=7, color="gray")

        _barh_model(axes[0], "delta_dt",  "dt_without",  "Decision Tree",
                    color_pos="#4CAF50", color_neg="#F44336")
        _barh_model(axes[1], "delta_knn", "knn_without", "kNN",
                    color_pos="#2196F3", color_neg="#FF9800")

        plt.tight_layout()
        slug = _slug(race_name)
        out_path = _EVAL_DIR / f"ablation_study_{slug}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Ablation study kaydedildi: {out_path.name}")

        # CSV kaydet
        abl_csv = _TABLES_DIR / f"ablation_study_{slug}.csv"
        ablation_df.to_csv(abl_csv, index=False)

        return {
            "baseline_dt":    baseline_dt,
            "baseline_knn":   baseline_knn,
            "ablation_table": ablation_df.to_dict(orient="records"),
            "most_important": ablation_df.iloc[0]["feature"] if len(ablation_df) > 0 else None,
            "chart":          str(out_path),
        }

    except Exception as e:
        logger.error(f"Ablation study hatası: {e}", exc_info=True)
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. OVERFITTING ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_overfitting_analysis(feature_matrix: pd.DataFrame, race_name: str) -> Dict[str, Any]:
    """
    Overfitting analizi: max_depth ve n_neighbors hiperparametresi arttıkça
    Train vs Validation doğruluğu nasıl değişiyor?

    DT için: max_depth 1→10
    kNN için: n_neighbors 1→15
    """
    logger.info(f"Overfitting Analysis çalışıyor: {race_name}")

    try:
        X, y, feat_cols = _prepare_Xy(feature_matrix)
        n_samples = len(X)

        if n_samples < 6:
            return {"error": f"Yeterli örnek yok (n={n_samples})"}

        n_folds = _max_safe_folds(y, max_folds=3)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        # ── DT: max_depth sweep ──────────────────────────────────
        dt_depths  = list(range(1, min(11, n_samples)))
        dt_train, dt_val = [], []
        for depth in dt_depths:
            clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
            from sklearn.model_selection import cross_validate as cv_fn
            res = cv_fn(clf, X_sc, y, cv=cv, scoring="accuracy", return_train_score=True)
            dt_train.append(float(res["train_score"].mean()))
            dt_val.append(float(res["test_score"].mean()))

        # ── kNN: n_neighbors sweep ───────────────────────────────
        # Fold başına eğitim boyutu = n_samples * (n_folds-1)/n_folds
        # n_neighbors bu değeri geçmemeli
        max_train_size = int(n_samples * (n_folds - 1) / n_folds) - 1
        max_k = min(15, max_train_size)
        knn_ks = list(range(1, max_k + 1))
        knn_train, knn_val = [], []
        for k in knn_ks:
            clf = KNeighborsClassifier(n_neighbors=k)
            res = cv_fn(clf, X_sc, y, cv=cv, scoring="accuracy", return_train_score=True)
            knn_train.append(float(res["train_score"].mean()))
            knn_val.append(float(res["test_score"].mean()))

        # ── Grafik ──────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Overfitting Analizi — {race_name} GP",
                     fontsize=13, fontweight="bold")

        def _plot_overfit(ax, x_vals, train_scores, val_scores, xlabel, title, opt_mode="max"):
            ax.plot(x_vals, train_scores, "o-", color="#2196F3", lw=2,
                    label="Eğitim Doğruluğu", markersize=5)
            ax.plot(x_vals, val_scores, "s--", color="#FF9800", lw=2,
                    label="Doğrulama Doğruluğu", markersize=5)
            # Gap bölgesi
            ax.fill_between(x_vals, val_scores, train_scores,
                            where=[t > v for t, v in zip(train_scores, val_scores)],
                            alpha=0.15, color="red", label="Overfit Gap")

            if opt_mode == "max":
                best_idx = int(np.argmax(val_scores))
            else:
                best_idx = int(np.argmin(val_scores))
            ax.axvline(x=x_vals[best_idx], color="green", ls=":", lw=1.5,
                       label=f"Opt ({x_vals[best_idx]})")
            ax.scatter([x_vals[best_idx]], [val_scores[best_idx]], s=80,
                       color="green", zorder=5)

            gap_at_opt = train_scores[best_idx] - val_scores[best_idx]
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Doğruluk", fontsize=10)
            ax.set_ylim(0.0, 1.08)
            ax.set_title(f"{title}\n(Overfit Gap @ opt = {gap_at_opt:.3f})", fontsize=10)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, alpha=0.3)

        _plot_overfit(axes[0], dt_depths,  dt_train,  dt_val,
                      "max_depth", "Decision Tree")
        _plot_overfit(axes[1], knn_ks,     knn_train, knn_val,
                      "n_neighbors", "kNN", opt_mode="max")

        plt.tight_layout()
        slug = _slug(race_name)
        out_path = _EVAL_DIR / f"overfitting_analysis_{slug}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Overfitting analysis kaydedildi: {out_path.name}")

        # CSV kaydet
        of_rows = []
        for i, depth in enumerate(dt_depths):
            of_rows.append({"model": "DT", "param": f"max_depth={depth}",
                            "train_acc": dt_train[i], "val_acc": dt_val[i],
                            "gap": dt_train[i] - dt_val[i]})
        for i, k in enumerate(knn_ks):
            of_rows.append({"model": "kNN", "param": f"n_neighbors={k}",
                            "train_acc": knn_train[i], "val_acc": knn_val[i],
                            "gap": knn_train[i] - knn_val[i]})
        of_df = pd.DataFrame(of_rows)
        of_csv = _TABLES_DIR / f"overfitting_analysis_{slug}.csv"
        of_df.to_csv(of_csv, index=False)

        # En iyi parametreler
        best_depth = dt_depths[int(np.argmax(dt_val))]
        best_k     = knn_ks[int(np.argmax(knn_val))]

        return {
            "dt_sweep":  {"depths": dt_depths, "train": dt_train, "val": dt_val},
            "knn_sweep": {"ks": knn_ks, "train": knn_train, "val": knn_val},
            "best_dt_depth":    best_depth,
            "best_knn_k":       best_k,
            "dt_overfit_gap":   round(dt_train[-1] - dt_val[-1], 4),
            "chart":            str(out_path),
        }

    except Exception as e:
        logger.error(f"Overfitting analysis hatası: {e}", exc_info=True)
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# ANA GİRİŞ NOKTASI
# ─────────────────────────────────────────────────────────────────────────────

def run_advanced_analysis(feature_matrix: pd.DataFrame, df: pd.DataFrame,
                          race_name: str) -> Dict[str, Any]:
    """
    Tüm gelişmiş akademik analizleri çalıştırır.
    models.py'deki run_all_models() tarafından H) adımı olarak çağrılır.
    """
    logger.info(f"=== Gelişmiş Akademik Analiz: {race_name} ===")
    results = {}

    try:
        results["learning_curves"]   = run_learning_curves(feature_matrix, race_name)
    except Exception as e:
        logger.warning(f"Learning curves hatası: {e}")
        results["learning_curves"] = {"error": str(e)}

    try:
        results["feature_correlation"] = run_feature_correlation(feature_matrix, race_name)
    except Exception as e:
        logger.warning(f"Feature correlation hatası: {e}")
        results["feature_correlation"] = {"error": str(e)}

    try:
        results["ablation_study"]     = run_ablation_study(feature_matrix, race_name)
    except Exception as e:
        logger.warning(f"Ablation study hatası: {e}")
        results["ablation_study"] = {"error": str(e)}

    try:
        results["overfitting"]        = run_overfitting_analysis(feature_matrix, race_name)
    except Exception as e:
        logger.warning(f"Overfitting analysis hatası: {e}")
        results["overfitting"] = {"error": str(e)}

    logger.info(f"=== Gelişmiş analiz tamamlandı: {race_name} ===")
    return results
