# F1 Race Intelligence System

**BLM308 Veri Madenciliği Final Projesi — Bahar 2026**
**Ramazan Doğukan Bahşi**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## İçindekiler

1. [Proje Tanımı](#1-proje-tanımı)
2. [Sistem Mimarisi](#2-sistem-mimarisi)
3. [Kullanılan Teknolojiler](#3-kullanılan-teknolojiler)
4. [Makine Öğrenmesi Pipeline](#4-makine-öğrenmesi-pipeline)
5. [Kullanılan Modeller](#5-kullanılan-modeller)
6. [Akademik Geliştirmeler](#6-akademik-geliştirmeler)
7. [Deneysel Sonuçlar](#7-deneysel-sonuçlar)
8. [Kurulum](#8-kurulum)
9. [Proje Yapısı](#9-proje-yapısı)
10. [Streamlit Dashboard Kullanımı](#10-streamlit-dashboard-kullanımı)
11. [Flask API](#11-flask-api)
12. [Akademik Öz-Değerlendirme](#12-akademik-öz-değerlendirme)
13. [Gelecek Geliştirmeler](#13-gelecek-geliştirmeler)
14. [Akademik Dürüstlük Beyanı](#14-akademik-dürüstlük-beyanı)
15. [Lisans ve Katkı](#15-lisans-ve-katkı)

---

## 1. Proje Tanımı

### Amaç

Bu proje, **Formula 1 yarış verilerini** uçtan uca bir veri madenciliği pipeline'ıyla analiz eder.
CRISP-DM metodolojisine uygun olarak tasarlanmış sistem; veri toplama, ön işleme, özellik mühendisliği,
makine öğrenmesi modelleme, akademik değerlendirme ve görselleştirme aşamalarını kapsamlı biçimde uygular.

### Problem Tanımı

**Ana Soru:** Bir F1 pilotu, tur süresi ve tutarlılık özellikleri kullanılarak "güçlü performans" olarak
sınıflandırılabilir mi?

**İkincil Sorular:**

- Pilotlar hangi performans kümelerine ayrılıyor? (K-Means kümeleme)
- Hangi özellik, yarış performansını en çok açıklıyor? (SHAP / Permutation Importance)
- Lastik stratejisi ve pit stop kararları tur zamanını nasıl etkiliyor?
- Hava koşulları (sıcaklık, nem) tur zamanıyla istatistiksel olarak anlamlı ilişki gösteriyor mu?

### Kullanım Senaryosu

| Kullanıcı | Senaryo |
|-----------|---------|
| Yarış Mühendisi | Pit stop optimal penceresi analizi |
| Strateji Analisti | Lastik bileşiği karşılaştırması |
| Akademisyen | CV ve istatistiksel test sonuçları üzerinde çalışma |
| Taraftar | Pilotlar arası hız ve tutarlılık karşılaştırması |

### Varsayılan Yarışlar (config.py → DEFAULT_RACES)

Sistem aşağıdaki 5 yarışı varsayılan olarak işler:

| # | Yarış | Pist Özelliği |
|---|-------|---------------|
| 1 | Bahrain | Yüksek degradasyon, 3 DRS |
| 2 | Monaco | Dar sokak, overtake güç, pit zamanlaması kritik |
| 3 | Great Britain | Yüksek hız, hava değişkeni |
| 4 | Belgium | Uzun düzlük + teknik sektör kombinasyonu |
| 5 | Italy | En hızlı pist, slipstream stratejisi |

---

## 2. Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    VERİ KATMANI                             │
│  FastF1 API → Raw CSV → Processed CSV → SQLite DB          │
│  (Lazy Loading + 3 seviyeli cache sistemi)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  İŞLEME KATMANI (src/)                      │
│                                                             │
│  preprocessing.py     →  feature_engineering.py            │
│  (9 adım)                 (11 özellik)                      │
│                                                             │
│  analysis.py          →  models.py                         │
│  (5 analiz modülü)        (K-Means, DT, kNN)               │
│                                                             │
│  model_evaluation.py  →  xai.py                            │
│  (CV, GridSearch,         (SHAP, Permutation Importance)   │
│   ROC/AUC, Karşılaştırma)                                  │
│                                                             │
│  error_analysis.py    →  statistical_tests.py              │
│  (Hata analizi)           (McNemar, t-test, Wilcoxon)     │
│                                                             │
│  advanced_analysis.py  (Learning Curves, Correlation,      │
│                          Ablation Study, Overfit Sweep)    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  ÇIKTI KATMANI                              │
│                                                             │
│  dashboard/streamlit_app.py  (Port 8501 — 15 sekme)        │
│  dashboard/app.py            (Port 5050 — Flask API)       │
│  reports/                    (PDF, MD, CSV, PNG)            │
└─────────────────────────────────────────────────────────────┘
```

### Veri Akışı

```
FastF1 API (internet) ──► data/raw/          (Ham CSV)
                                │
                                ▼
                        preprocessing.py     (9 adımlı temizleme)
                                │
                                ▼
                        feature_engineering  (11 türetilmiş özellik)
                                │
                    ┌───────────┴──────────┐
                    ▼                      ▼
            data/processed/          data/f1_intelligence.db
             (İşlenmiş CSV)              (SQLite)
                    │
                    ▼
        analysis.py + models.py + model_evaluation.py
                    │
        ┌───────────┼──────────────┬────────────────┐
        ▼           ▼              ▼                ▼
  reports/       reports/      reports/        data/
  figures/       tables/       generated_      analysis_cache/
  (PNG)          (CSV/MD)      reports/        (JSON)
                               (PDF/MD)
```

### Cache Sistemi (3 Seviye)

| Seviye | Konum | İçerik | Amaç |
|--------|-------|--------|------|
| 1 | `data/raw/{slug}_raw.csv` | Ham FastF1 verisi | Yeniden indirmeyi önle |
| 2 | `data/processed/{slug}_processed.csv` | Temizlenmiş veri | Preprocessing atla |
| 3 | `data/analysis_cache/{slug}_v2.json` | Analiz + ML sonuçları | Hesaplamayı atla |

---

## 3. Kullanılan Teknolojiler

### ML ve Veri Bilimi

| Kütüphane | Kullanım |
|-----------|---------|
| `scikit-learn` | K-Means, DT, kNN, GridSearchCV, StratifiedKFold, ROC/AUC |
| `pandas` | Veri manipülasyonu ve tablo oluşturma |
| `numpy` | Sayısal hesaplamalar, dizi işlemleri |
| `scipy` | McNemar χ², paired t-test, Wilcoxon, Pearson korelasyon |
| `shap` | SHAP TreeExplainer ile model açıklanabilirliği |

### Veri ve Depolama

| Kütüphane | Kullanım |
|-----------|---------|
| `fastf1` | Formula 1 2025 sezonu resmi API |
| `sqlite3` | Lokal veritabanı (4 tablo: laps, race_summary, driver_stats, processed_races) |
| `joblib` | Model serileştirme |

### Görselleştirme

| Kütüphane | Kullanım |
|-----------|---------|
| `matplotlib` | Statik grafikler (dark theme, ROC, SHAP, CV) |
| `seaborn` | İstatistiksel görselleştirme |
| `plotly` | İnteraktif grafikler |

### Web ve Raporlama

| Kütüphane | Kullanım |
|-----------|---------|
| `streamlit` | Ana dashboard (15 sekme) |
| `flask` | REST API backend |
| `reportlab` | PDF rapor üretimi |
| `tabulate` | Markdown tablo formatı |

---

## 4. Makine Öğrenmesi Pipeline

### A. Ön İşleme (9 Adım)

```python
preprocess_laps(df, race_name)
```

| # | Adım | İşlem |
|---|------|--------|
| 1 | Sütun standardizasyonu | Büyük/küçük harf normalizasyonu |
| 2 | Zaman dönüşümü | Timedelta → float (saniye) |
| 3 | Eksik tur zamanları | NaN / 0 değerleri sil |
| 4 | Safety Car turları | TrackStatus → SC/VSC etiketi |
| 5 | Pit stop turları | Medyan × 1.5 eşiği ile etiketleme |
| 6 | TrackStatus decode | Kod → metin etiketi |
| 7 | Aykırı tur filtreleme | ±3σ kırpma |
| 8 | Eksik hava verisi | Medyan imputation |
| 9 | Lastik standardizasyonu | Büyük harf normalize |

### B. Özellik Mühendisliği (11 Özellik)

```python
engineer_features(df, race_name)
get_driver_feature_matrix(df)   # Her satır = 1 pilot
```

| Özellik | Formül | Tür |
|---------|--------|-----|
| `average_lap_time` | mean(LapTime) — pit/SC hariç | Hız |
| `best_lap_time` | min(LapTime) | Hız |
| `lap_time_std` | std(LapTime) | Tutarlılık |
| `consistency_score` | 1 / (1 + std) | Tutarlılık |
| `average_race_pace` | mean(LapTime) — SC hariç | Hız |
| `tire_degradation_rate` | slope(TyreLife, LapTime) | Lastik |
| `pit_stop_impact` | mean(pace_before) − mean(pace_after) | Strateji |
| `sector_consistency` | mean(std(S1, S2, S3)) | Tutarlılık |
| `tire_efficiency_score` | (genel_ort − compound_ort) / genel_ort | Lastik |
| `relative_pace` | pilot_ort − yarış_ort | Göresel |
| `StintLap` | Stint içi tur sırası | Lastik yaşı |

### C. Hedef Değişken

```python
# Güçlü Performans = hem hızlı HEM tutarlı
fast   = average_lap_time  <=  yarış_medyanı
stable = consistency_score >=  tutarlılık_medyanı
strong_performance = fast AND stable  →  binary {0, 1}
```

---

## 5. Kullanılan Modeller

### 5.1 K-Means Kümeleme

**Amaç:** Pilotları performans kümelerine ayır.

| Parametre | Değer |
|-----------|-------|
| n_clusters | 3 (Stabil & Hızlı / Agresif / Düzensiz) |
| Özellikler | avg_lap_time, lap_time_std, consistency_score, tire_degradation_rate, sector_consistency |
| Ön işleme | StandardScaler + SimpleImputer |
| Optimizasyon | Elbow Yöntemi + Silhouette Score + GridSearch |

**Avantajlar:** Yorum kolaylığı, yüksek hız, görselleştirilebilir yapı
**Dezavantajlar:** Önceden k belirleme zorunluluğu, başlangıç hassasiyeti

### 5.2 Decision Tree Sınıflandırma

**Amaç:** `strong_performance` binary sınıflandırma.

| Parametre | Değer |
|-----------|-------|
| max_depth | 5 (GridSearch ile optimize edilir) |
| random_state | 42 |
| criterion | gini (GridSearch ile optimize) |

**GridSearch Uzayı:** max_depth ∈ {3,4,5,7,None}, criterion ∈ {gini, entropy}, min_samples_split ∈ {2,4,6}

**Avantajlar:** Feature importance, SHAP uyumlu, yorumlanabilir kural ağacı
**Dezavantajlar:** Overfit eğilimi, küçük veri setlerine yüksek varyans

### 5.3 k-En Yakın Komşu (kNN)

**Amaç:** `strong_performance` binary sınıflandırma.

| Parametre | Değer |
|-----------|-------|
| n_neighbors | 5 (GridSearch ile optimize) |
| metric | euclidean (GridSearch ile optimize) |

**GridSearch Uzayı:** n_neighbors ∈ {3,5,7,9}, metric ∈ {euclidean, manhattan}, weights ∈ {uniform, distance}

**Avantajlar:** Parametrik olmayan, küçük veri setlerine uygun
**Dezavantajlar:** Bellek yoğun, zaman bazlı interpretasyon güçlüğü

---

## 6. Akademik Geliştirmeler

### 6.1 Stratified K-Fold Cross Validation

```python
from src.model_evaluation import run_cross_validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt_cv  = cross_validate(dt,  X, y, cv=cv,
                         scoring=['accuracy','precision_macro','recall_macro','f1_macro'],
                         return_train_score=True)
```

Her model için fold bazlı raporlanan metrikler:

| Metrik | Açıklama |
|--------|----------|
| Mean Accuracy | 5 fold ortalaması |
| Std Accuracy | 5 fold standart sapması |
| Mean Precision (macro) | Sınıf dengesi gözetmeksizin precision |
| Mean Recall (macro) | Sınıf dengesi gözetmeksizin recall |
| Mean F1 (macro) | Precision-Recall dengesi |
| Overfit Gap | Eğitim-test doğruluk farkı |

### 6.2 GridSearchCV Hiperparametre Optimizasyonu

```python
gs_dt = GridSearchCV(
    DecisionTreeClassifier(),
    GRIDSEARCH_DT_PARAMS,     # config.py'den
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
gs_dt.fit(X, y)
# gs_dt.best_params_, gs_dt.best_score_
```

Sonuçlar `reports/tables/model_comparison_{race}.csv` ve `.md` olarak kaydedilir.

**K-Means için:** Silhouette score ile n_clusters ve init kombinasyonları taranır.

### 6.3 ROC Eğrisi ve AUC

```python
# cross_val_predict ile out-of-fold olasılık tahminleri
# (küçük örneklem için doğru yaklaşım — tüm veri noktaları kullanılır)
proba = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
fpr, tpr, thresholds = roc_curve(y, proba[:, 1])
auc_score = auc(fpr, tpr)

# Youden J istatistiği ile optimal eşik
optimal_threshold = thresholds[argmax(tpr - fpr)]
```

### 6.4 Kapsamlı Model Karşılaştırma Tablosu

| Metrik | Decision Tree | kNN |
|--------|---------------|-----|
| CV Mean Accuracy | **0.905** | 0.849 |
| CV Std | ±0.067 | ±0.011 |
| CV Precision (macro) | 0.726 | 0.425 |
| CV Recall (macro) | 0.806 | 0.500 |
| CV F1 (macro) | 0.750 | 0.459 |
| AUC | **0.804** | 0.618 |
| GridSearch Best Score | 0.849 | 0.849 |
| Overfit Gap | 0.095 | 0.001 |

*Bahrain GP 2025 — 3-fold Stratified CV (n=20 sürücü). Her yarış için otomatik üretilir → `reports/tables/model_comparison_{race}.csv`*

### 6.5 SHAP Açıklanabilir Yapay Zeka

```python
import shap
explainer   = shap.TreeExplainer(dt_model)
shap_values = explainer.shap_values(X)
# shap_values[1]: pozitif sınıf için her özelliğin katkısı
```

SHAP kurulu değilse **Permutation Importance** otomatik devreye girer:

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X, y, n_repeats=30, random_state=42)
```

Üretilen grafikler:
- SHAP Summary Bar (global önem)
- SHAP Beeswarm (özellik değeri vs SHAP katkısı)
- Permutation Importance — DT ve kNN karşılaştırmalı
- Sürücü bazlı yerel açıklamalar (tahmin güveni + özellik değerleri)

### 6.6 Hata Analizi

```python
from src.error_analysis import run_error_analysis
```

| Analiz | Açıklama |
|--------|----------|
| Hata oranı | Yanlış tahmin / toplam |
| FP / FN | False Positive ve False Negative sayısı |
| Yüksek güvenli hatalar | Model güven > %70 iken yanlış tahmin |
| Ortak hatalar | Her iki modelin yanlış tahmin ettiği sürücüler |
| Sürücü bazlı rapor | Her sürücü için DT ve kNN tahmini |
| Hata dağılımı | Özellik uzayında hata scatter grafiği |

### 6.7 İstatistiksel Testler

```python
from src.statistical_tests import run_statistical_tests
```

| Test | Hipotez (H₀) | İstatistik |
|------|-------------|------------|
| **McNemar** | İki model aynı hata dağılımına sahip | χ² (süreklilik düzeltmeli, df=1) |
| **Paired t-test** | CV fold ortalamaları eşittir | t istatistiği |
| **Wilcoxon** | Parametrik olmayan t-test alternatifi | W istatistiği |

**Yorum:** p < 0.05 → H₀ reddedilir → Modeller arasında anlamlı performans farkı var.

### 6.8 Özellik Mühendisliği ve Veri Kalitesi

**Feature Leakage Analizi:**
- `best_lap_time` ve `average_lap_time` yüksek korelasyon riski taşır; pipeline'da birlikte kullanılır ancak dikkatle izlenir.
- Hedef değişken medyan bazlı tanımlandığı için her yarışta otomatik dengelenir (~%50-%50).

**Ölçeklendirme:**
- `StandardScaler` tüm sınıflandırma modellerinde ve K-Means'te uygulanır.
- `SimpleImputer (strategy='median')` eksik değerleri doldurur.

### 6.9 Learning Curves (Bias-Variance Tradeoff)

```python
from sklearn.model_selection import learning_curve
sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(min_train, max_train, 6),
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="accuracy",
)
```

Üretilen grafik: eğitim seti büyüdükçe train/validation doğruluğu nasıl değişiyor?
- **Yüksek bias:** Her iki eğri de düşük → model çok basit (underfit)
- **Yüksek varyans:** Train yüksek, Validation düşük → overfit
- **Dengeli:** İki eğri yakın ve yüksek → ideal model

### 6.10 Feature Correlation Heatmap

Pearson korelasyon matrisi ile multicollinearity tespiti:
- `|r| > 0.9` → yüksek risk (özellik çıkarılabilir)
- `|r| > 0.7` → dikkat gerektirir

Çıktı: `reports/figures/evaluation/feature_correlation_{race}.png`

### 6.11 Ablation Study (Drop-One-Out)

Her özelliği tek tek çıkararak modelin nasıl etkilendiğini ölçen yöntem:

```python
for i, feat in enumerate(feat_cols):
    X_drop = np.delete(X_scaled, i, axis=1)
    acc_dt  = cross_val_score(dt,  X_drop, y, cv=cv).mean()
    delta   = acc_dt - baseline_dt
    # Negatif delta → özellik önemli (çıkarınca performans düşüyor)
    # Pozitif delta → özellik gürültü ekliyor
```

Çıktı: `reports/tables/ablation_study_{race}.csv`

### 6.12 Overfitting Analizi (Hiperparametre Sweepleri)

- **DT için:** max_depth 1→10 sweep, her değerde train vs val accuracy
- **kNN için:** n_neighbors 1→15 sweep (fold-aware üst sınır)

Çıktı: `reports/figures/evaluation/overfitting_analysis_{race}.png`

---

## 7. Deneysel Sonuçlar

> Aşağıdaki değerler **gerçek FastF1 verisiyle** (Bahrain GP 2025, n=20 sürücü) üretilmiştir.

### Karşılaştırmalı Model Tablosu (Bahrain GP 2025 — Gerçek Sonuçlar)

| Model | CV Acc | CV Std | Precision | Recall | F1-macro | AUC | GS Best |
|-------|--------|--------|-----------|--------|----------|-----|---------|
| **Decision Tree** | **0.905** | ±0.067 | 0.726 | 0.806 | 0.750 | **0.804** | 0.849 |
| kNN (k=5) | 0.849 | ±0.011 | 0.425 | 0.500 | 0.459 | 0.618 | 0.849 |

*CV: 3-fold Stratified (auto-reduced, min_class=3), GridSearch: DT `criterion=gini, max_depth=3`, kNN `metric=euclidean, n_neighbors=5`*

### İstatistiksel Test Özeti (Bahrain GP 2025)

| Test | İstatistik | p-değeri | Anlamlı? (α=0.05) | Yorum |
|------|-----------|---------|---------|-------|
| McNemar | χ²=0.00 | 1.000 | Hayır | Discordant pair=1 → güç yetersiz |
| Paired t-test | t=1.00 | 0.423 | Hayır | 3 fold, yüksek std |
| Wilcoxon | W=0.00 | 1.000 | Hayır | Parametrik olmayan, aynı sonuç |

> **Akademik Not:** F1 veri setinde pilot başına 1 satır (n=20) bulunduğundan istatistiksel güç
> yapısal olarak düşüktür. Anlamlı fark bulunamaması DT ve kNN'in benzer hata örüntüsü
> sergilemesinden kaynaklanmaktadır (McNemar discordant pair=1). Daha büyük veri setinde
> (n>100) farklı sonuç beklenmektedir.

### En Önemli Özellikler (Permutation Importance — Decision Tree)

| Sıra | Özellik | Önem Skoru |
|------|---------|------------|
| 1 | consistency_score | 0.157 |
| 2 | average_lap_time | 0.124 |
| 3 | tire_degradation_rate | 0.089 |
| 4 | sector_consistency | 0.043 |
| 5 | pit_stop_impact | 0.021 |

*SHAP kurulu değilse permutation importance (n_repeats=30) otomatik devreye girer.*

### Ablation Study Özeti (Bahrain GP)

- **En kritik özellik:** `consistency_score` — çıkarıldığında DT doğruluğu en çok düşüyor
- **Baseline DT:** 0.897 | **Baseline kNN:** 0.698
- **Gereksiz özellik riski:** `best_lap_time` pozitif delta (avg_lap_time ile yüksek korelasyon)

### Learning Curves (Bias-Variance Analizi)

| Model | Max Val. Accuracy | Eğitim Acc (son) | Overfit Gap | Yorum |
|-------|-------------------|------------------|-------------|-------|
| Decision Tree | 0.952 | ~1.000 | 0.048 | Hafif overfit, kabul edilebilir |
| kNN | 0.754 | ~0.857 | 0.103 | Dengeli, makul |

---

## 8. Kurulum

### Gereksinimler

- Python 3.10+
- pip

### Adım Adım

```bash
# 1. Repoyu klonla
git clone https://github.com/DogukanBahsi/f1-race-intelligence-system.git
cd f1-race-intelligence-system

# 2. Sanal ortam oluştur ve aktive et
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Bağımlılıkları kur (shap dahil tüm paketler)
pip install -r requirements.txt

# 4. Pipeline çalıştır
#    (İnternet yoksa simüle veri ile otomatik çalışır)
python main.py

# 5. Streamlit Dashboard
streamlit run dashboard/streamlit_app.py

# 6. Flask API (opsiyonel)
python dashboard/app.py
```

### Servis Adresleri

| Servis | Adres | Port |
|--------|-------|------|
| Streamlit Dashboard | http://localhost:8501 | 8501 |
| Flask API | http://localhost:5050 | 5050 |

---

## 9. Proje Yapısı

```
F1 Analizi - Veri Madenciliği/
│
├── main.py                       # Ana pipeline orchestrator
├── config.py                     # Merkezi konfigürasyon
├── requirements.txt              # Python bağımlılıkları
├── README.md
│
├── src/                          # Modüler kaynak kodlar
│   ├── __init__.py
│   ├── data_loader.py            # FastF1 + lazy loading + simüle fallback
│   ├── preprocessing.py          # 9 adımlı veri temizleme
│   ├── feature_engineering.py   # 11 özellik türetme
│   ├── analysis.py               # 5 analiz modülü
│   ├── models.py                 # K-Means + DT + kNN pipeline
│   ├── model_evaluation.py    ★  # CV, GridSearch, ROC/AUC, Model Comparison
│   ├── xai.py                 ★  # SHAP + Permutation Importance
│   ├── error_analysis.py      ★  # Hata analizi, confusion matrix
│   ├── statistical_tests.py   ★  # McNemar, t-test, Wilcoxon
│   ├── advanced_analysis.py   ★  # Learning Curves, Feature Corr, Ablation, Overfit
│   ├── visualization.py          # 12 tür matplotlib grafik
│   ├── eda.py                    # Keşifsel veri analizi
│   ├── insight_engine.py         # Otomatik Türkçe yorum üretimi
│   ├── cache_manager.py          # JSON cache yönetimi
│   ├── database.py               # SQLite CRUD
│   ├── pdf_report.py             # ReportLab PDF
│   ├── report_generator.py       # Markdown + TXT rapor
│   ├── logger.py                 # Renkli merkezi loglama
│   └── utils.py                  # Yardımcı fonksiyonlar
│
├── dashboard/
│   ├── streamlit_app.py          # Streamlit (15 sekme, port 8501)
│   └── app.py                    # Flask API (6 endpoint, port 5050)
│
├── data/
│   ├── raw/                      # Ham FastF1 CSV
│   ├── processed/                # Temizlenmiş + feature'lı CSV
│   ├── analysis_cache/           # JSON analiz cache
│   ├── exports/                  # Dışa aktarılan dosyalar
│   └── f1_intelligence.db        # SQLite veritabanı
│
├── reports/
│   ├── figures/
│   │   ├── evaluation/        ★  # CV, ROC, GridSearch, hata grafikleri
│   │   └── xai/               ★  # SHAP, Permutation Importance grafikleri
│   ├── tables/                ★  # CSV + Markdown karşılaştırma tabloları
│   └── generated_reports/        # PDF + MD + TXT raporlar
│
└── notebooks/                    # Jupyter notebook alanı (opsiyonel)

★ = Akademik geliştirme aşamasında eklenen bileşenler
```

---

## 10. Streamlit Dashboard Kullanımı

Dashboard `streamlit run dashboard/streamlit_app.py` komutuyla başlatılır.
**15 sekme** içerir:

| # | Sekme | İçerik |
|---|-------|--------|
| 1 | Overview | Yarış özeti metrikleri, pilot ortalamaları |
| 2 | Driver Perf. | Pilot istikrar tablosu, consistency skoru |
| 3 | Comparison | İki pilot karşılaştırması (sektör bazlı) |
| 4 | Tire Strategy | Lastik bileşiği analizi, degradasyon eğrisi |
| 5 | Pit Stop | Pit stop etki analizi, sürücü bazlı |
| 6 | Weather | Hava-tur zamanı korelasyon matrisi |
| 7 | ML Models | K-Means kümeler, DT/kNN sonuçları |
| **8** | **CV & GridSearch** | **5-fold CV, GridSearch, K-Means opt., istatistiksel testler** |
| **9** | **ROC / AUC** | **ROC eğrisi, AUC karşılaştırması, model tablosu** |
| **10** | **XAI (SHAP)** | **SHAP beeswarm, Permutation Importance, yerel açıklamalar** |
| **11** | **Error Analysis** | **Confusion matrix, sürücü hata raporu, yüksek güvenli hatalar** |
| **12** | **Advanced** | **Learning Curves, Feature Correlation, Ablation Study, Overfitting** |
| 13 | AI Insights | Otomatik Türkçe yorum engine |
| 14 | Report | Markdown + PDF rapor indirme |
| 15 | Sunum | Slayt stili sunum özeti |

**Kalın** satırlar bu geliştirmede eklenen yeni akademik sekmeleri gösterir.

---

## 11. Flask API

Flask API `python dashboard/app.py` ile başlar (port 5050).

| Endpoint | Yöntem | Açıklama |
|----------|--------|----------|
| `/` | GET | Ana dashboard sayfası |
| `/api/race/<race_name>` | GET | Yarış analiz sonuçları (JSON) |
| `/api/report/<race_name>` | GET | Yarış raporu (Markdown/TXT) |
| `/api/compare/<race>/<driver1>/<driver2>` | GET | İki pilot karşılaştırması |
| `/api/pdf/<race_name>` | GET | PDF rapor indir |
| `/api/export/<race_name>` | GET | CSV verisi dışa aktar |

---

## 12. Akademik Öz-Değerlendirme

### Tamamlanan Akademik Bileşenler

| Kriter | Durum | Konum |
|--------|-------|-------|
| Stratified K-Fold CV (5-fold) | ✅ Tamamlandı | `src/model_evaluation.py` |
| GridSearchCV — DT + kNN | ✅ Tamamlandı | `src/model_evaluation.py` |
| K-Means GridSearch (silhouette) | ✅ Tamamlandı | `src/model_evaluation.py` |
| ROC/AUC (cross_val_predict) | ✅ Tamamlandı | `src/model_evaluation.py` |
| Confusion Matrix Heatmap | ✅ Tamamlandı | `src/error_analysis.py` |
| SHAP / Permutation Importance | ✅ Tamamlandı | `src/xai.py` |
| Driver Error Report | ✅ Tamamlandı | `src/error_analysis.py` |
| McNemar + t-test + Wilcoxon | ✅ Tamamlandı | `src/statistical_tests.py` |
| Learning Curves | ✅ Tamamlandı | `src/advanced_analysis.py` |
| Feature Correlation Heatmap | ✅ Tamamlandı | `src/advanced_analysis.py` |
| Ablation Study | ✅ Tamamlandı | `src/advanced_analysis.py` |
| Overfitting Analysis Sweep | ✅ Tamamlandı | `src/advanced_analysis.py` |
| Model Comparison Table (CSV+MD) | ✅ Tamamlandı | `reports/tables/` |
| Streamlit 15-sekme Dashboard | ✅ Tamamlandı | `dashboard/streamlit_app.py` |

### Üretilen Çıktı Dosyaları

```
reports/figures/evaluation/   → 13 PNG (CV, ROC, GridSearch, Learning Curves, ...)
reports/figures/xai/          →  4 PNG (feature importance, permutation)
reports/tables/               →  7 CSV (cv, ablation, correlation, overfit, ...)
reports/generated_reports/    → PDF + MD + TXT raporlar
```

## 13. Gelecek Geliştirmeler

| Geliştirme | Açıklama |
|-----------|----------|
| Çok-yarış karşılaştırması | Sezon boyunca model performansı izleme |
| Ensemble methods | Random Forest, Gradient Boosting ekleme |
| Weka karşılaştırması | Python vs Weka karşılaştırmalı mini-survey |
| Docker | Taşınabilir container dağıtımı |
| Hugging Face Hub | Model ve veri seti paylaşımı |
| MLOps pipeline | MLflow ile deney takibi |

---

## 14. Akademik Dürüstlük Beyanı

Bu proje BLM308 Veri Madenciliği dersi final projesi kapsamında geliştirilmiştir.

- Problem tanımı, deney kurgusu ve sonuç analizi özgündür.
- Yapay zeka araçları (Claude Code) kod geliştirme sürecinde yardımcı araç olarak kullanılmıştır.
- Tüm AI ve LLM kullanımı burada beyan edilmektedir.
- Atıfsız kopya yapılmamıştır.

---

## 15. Lisans ve Katkı

**Geliştirici:** Ramazan Doğukan Bahşi
**Kurum:** Bilgisayar Mühendisliği — BLM308 Veri Madenciliği
**Yıl:** 2026

**Lisans:** MIT License

---

*Son güncelleme: Mayıs 2026 | FastF1 2025 Sezonu | Python 3.10+ | 12 Akademik Bileşen | 15 Dashboard Sekmesi*
