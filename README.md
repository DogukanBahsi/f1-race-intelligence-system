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
12. [Gelecek Geliştirmeler](#12-gelecek-geliştirmeler)
13. [Akademik Dürüstlük Beyanı](#13-akademik-dürüstlük-beyanı)
14. [Lisans ve Katkı](#14-lisans-ve-katkı)

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
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  ÇIKTI KATMANI                              │
│                                                             │
│  dashboard/streamlit_app.py  (Port 8501 — 14 sekme)        │
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
| `streamlit` | Ana dashboard (14 sekme) |
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
| CV Mean Accuracy | ... | ... |
| CV Std | ... | ... |
| CV Precision | ... | ... |
| CV Recall | ... | ... |
| CV F1 | ... | ... |
| AUC | ... | ... |
| GridSearch Best Score | ... | ... |
| Overfit Gap | ... | ... |

*Her yarış için otomatik üretilir — `reports/tables/` altına kaydedilir.*

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

---

## 7. Deneysel Sonuçlar

> Aşağıdaki değerler simüle veri (n≈20 sürücü) ile üretilmiş örnek sonuçlardır.
> Gerçek FastF1 verisiyle değerler değişecektir.

### Karşılaştırmalı Model Tablosu (Bahrain GP — Örnek)

| Model | CV Acc | CV Std | Precision | Recall | F1 | AUC |
|-------|--------|--------|-----------|--------|----|-----|
| Decision Tree | 0.73 | ±0.12 | 0.74 | 0.72 | 0.73 | 0.81 |
| kNN (k=5) | 0.68 | ±0.15 | 0.69 | 0.67 | 0.68 | 0.76 |

### İstatistiksel Test Özeti

| Test | İstatistik | p-değeri | Anlamlı? (α=0.05) |
|------|-----------|---------|---------|
| McNemar | χ²=1.00 | 0.317 | Hayır |
| Paired t-test | t=1.42 | 0.220 | Hayır |
| Wilcoxon | W=3.00 | 0.250 | Hayır |

> **Not:** F1 verisi pilot başına 1 satır içerdiğinden (n≈15-20), istatistiksel güç
> doğası gereği düşüktür. Bu bulgu, veri setinin yapısal kısıtından kaynaklanmaktadır.

### En Önemli Özellikler (SHAP — Decision Tree)

| Sıra | Özellik | Ortalama SHAP |
|------|---------|---------------|
| 1 | consistency_score | 0.0842 |
| 2 | average_lap_time | 0.0631 |
| 3 | tire_degradation_rate | 0.0287 |
| 4 | sector_consistency | 0.0193 |
| 5 | pit_stop_impact | 0.0124 |

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

# 3. Bağımlılıkları kur
pip install -r requirements.txt

# 4. (Opsiyonel) SHAP kur — XAI için
pip install shap

# 5. Pipeline çalıştır
#    (İnternet yoksa simüle veri ile otomatik çalışır)
python main.py

# 6. Streamlit Dashboard
streamlit run dashboard/streamlit_app.py

# 7. Flask API (opsiyonel)
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
│   ├── streamlit_app.py          # Streamlit (14 sekme, port 8501)
│   └── app.py                    # Flask API (port 5050)
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
└── experiments/               ★  # Deney log dosyaları

★ = Bu geliştirmede eklenen bileşenler
```

---

## 10. Streamlit Dashboard Kullanımı

Dashboard `streamlit run dashboard/streamlit_app.py` komutuyla başlatılır.
**14 sekme** içerir:

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
| 12 | AI Insights | Otomatik Türkçe yorum engine |
| 13 | Report | Markdown + PDF rapor indirme |
| 14 | Sunum | Slayt stili sunum özeti |

**Kalın** satırlar bu geliştirmede eklenen yeni akademik sekmeleri gösterir.

---

## 11. Flask API

Flask API `python dashboard/app.py` ile başlar (port 5050).

| Endpoint | Yöntem | Açıklama |
|----------|--------|----------|
| `/` | GET | Ana sayfa |
| `/api/races` | GET | İşlenmiş yarışlar listesi |
| `/api/race/<name>` | GET | Yarış analiz sonuçları (JSON) |

---

## 12. Gelecek Geliştirmeler

| Geliştirme | Açıklama |
|-----------|----------|
| Çok-yarış karşılaştırması | Sezon boyunca model performansı izleme |
| Ensemble methods | Random Forest, Gradient Boosting ekleme |
| Weka karşılaştırması | Python vs Weka karşılaştırmalı mini-survey |
| Docker | Taşınabilir container dağıtımı |
| Hugging Face Hub | Model ve veri seti paylaşımı |
| MLOps pipeline | MLflow ile deney takibi |

---

## 13. Akademik Dürüstlük Beyanı

Bu proje BLM308 Veri Madenciliği dersi final projesi kapsamında geliştirilmiştir.

- Problem tanımı, deney kurgusu ve sonuç analizi özgündür.
- Yapay zeka araçları (Claude Code) kod geliştirme sürecinde yardımcı araç olarak kullanılmıştır.
- Tüm AI ve LLM kullanımı burada beyan edilmektedir.
- Atıfsız kopya yapılmamıştır.

---

## 14. Lisans ve Katkı

**Geliştirici:** Ramazan Doğukan Bahşi
**Kurum:** Bilgisayar Mühendisliği — BLM308 Veri Madenciliği
**Yıl:** 2026

**Lisans:** MIT License

---

*Son güncelleme: Mayıs 2026 | FastF1 2025 Sezonu | Python 3.10+*
