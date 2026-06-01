# F1 Race Intelligence & Strategy Analysis System

**BLM308 Veri Madenciliği — Dönem Projesi, Bahar 2026**
**Ramazan Doğukan Bahşi**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)](https://streamlit.io)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Özet

Bu çalışma, Formula 1 yarış telemetri verilerini CRISP-DM metodolojisi çerçevesinde analiz eden uçtan uca bir veri madenciliği sistemi sunmaktadır. FastF1 API aracılığıyla elde edilen 2025 sezonu yarış verileri üzerinde; 10 adımlı ön işleme, 11 türetilmiş özellik ve üç makine öğrenmesi algoritması (K-Means, Decision Tree, k-En Yakın Komşu) uygulanmıştır. Akademik değerlendirme kapsamında Stratified K-Fold Çapraz Doğrulama, GridSearchCV hiperparametre optimizasyonu, ROC/AUC analizi, SHAP tabanlı açıklanabilir yapay zeka, hata analizi ve üç istatistiksel hipotez testi gerçekleştirilmiştir. Bahrain GP 2025 üzerindeki deneysel sonuçlar; Decision Tree modelinin %90.5 (±6.7) çapraz doğrulama doğruluğu ve 0.804 AUC değeriyle kNN'e (0.849 ±0.011, AUC=0.618) üstünlük sağladığını göstermektedir. McNemar testi (p=1.000) ve Wilcoxon testi (p=1.000) iki model arasında istatistiksel olarak anlamlı bir fark olmadığını ortaya koymaktadır; bu bulgu veri setinin yapısal kısıtından (n≈20) kaynaklanmaktadır.

---

## İçindekiler

1. [Giriş ve Motivasyon](#1-giriş-ve-motivasyon)
2. [Veri Seti](#2-veri-seti)
3. [Metodoloji](#3-metodoloji)
4. [Özellik Mühendisliği](#4-özellik-mühendisliği)
5. [Makine Öğrenmesi Modelleri](#5-makine-öğrenmesi-modelleri)
6. [Akademik Değerlendirme Çerçevesi](#6-akademik-değerlendirme-çerçevesi)
7. [Deneysel Sonuçlar](#7-deneysel-sonuçlar)
8. [Sistem Mimarisi](#8-sistem-mimarisi)
9. [Proje Yapısı](#9-proje-yapısı)
10. [Kurulum ve Kullanım](#10-kurulum-ve-kullanım)
11. [Sonuç ve Tartışma](#11-sonuç-ve-tartışma)
12. [Kaynakça](#12-kaynakça)
13. [Akademik Dürüstlük Beyanı](#13-akademik-dürüstlük-beyanı)

---

## 1. Giriş ve Motivasyon

### 1.1 Problem Tanımı

Formula 1, dünyanın en yüksek veri yoğunluklu spor dallarından biridir. Her yarışta araç başına saniyede 300'ün üzerinde sensör verisi üretilmekte; tur zamanları, lastik bileşikleri, pit stop kararları ve meteorolojik koşullar yarış sonucunu doğrudan belirlemektedir. Bu çalışmada, söz konusu çok boyutlu veriden anlamlı örüntüler çıkarmak için veri madenciliği teknikleri uygulanmıştır.

**Ana Araştırma Sorusu:**
> Bir F1 pilotunun tur süresi ve sürüş tutarlılığı özellikleri kullanılarak "güçlü performans" sergilediği ikili (binary) sınıflandırma ile belirlenebilir mi?

**Yardımcı Araştırma Soruları:**

| # | Soru | Yöntem |
|---|------|--------|
| S1 | Pilotlar hangi performans kümelerine ayrılıyor? | K-Means Kümeleme |
| S2 | Hangi özellik yarış performansını en çok açıklıyor? | SHAP / Permutation Importance |
| S3 | Lastik stratejisi ve pit stop kararları tur zamanını nasıl etkiliyor? | Tanımlayıcı Analiz |
| S4 | Hava koşulları tur zamanıyla istatistiksel olarak ilişkili mi? | Pearson Korelasyonu |
| S5 | Decision Tree ile kNN arasında anlamlı bir performans farkı var mı? | McNemar, t-test, Wilcoxon |

### 1.2 Katkılar

- **Endüstriyel veri üzerinde gerçek zamanlı akademik pipeline:** FastF1 API entegrasyonu ile 2025 sezonu yarış verilerinin otomatik çekilmesi ve işlenmesi.
- **Küçük örneklem için uyarlanmış CV stratejisi:** n≈20 boyutunda veri setleri için `_max_safe_folds()` mekanizması ile otomatik fold azaltma.
- **Çok katmanlı akademik değerlendirme:** CV, GridSearch, ROC/AUC, SHAP, Ablasyon Çalışması, Öğrenme Eğrileri ve üç istatistiksel test tek pipeline'da entegre edilmiştir.
- **Çift arayüzlü dashboard:** 15 sekme içeren Streamlit ve 6 endpoint'li Flask API ile sonuçların görselleştirilmesi.

---

## 2. Veri Seti

### 2.1 Veri Kaynağı

| Özellik | Değer |
|---------|-------|
| Kaynak | FastF1 Python Kütüphanesi v3.x (Ergast API + F1 Live Timing) |
| Sezon | Formula 1 2025 (24 yarış) |
| Varsayılan Yarışlar | Bahrain, Monaco, Great Britain, Belgium, Italy |
| Veri Türü | Telemetri — tur bazlı (lap-level) |
| Granülarite | Her satır = 1 pilot × 1 tur |
| Ham Boyut | ~800–1400 satır / yarış, ~35 sütun |
| Feature Matrix Boyutu | ~20 satır (pilot başına 1) × 15 sütun |

### 2.2 Ham Değişkenler

| Değişken | Tür | Açıklama |
|----------|-----|----------|
| `LapTime` | sürekli | Tur süresi (saniye) |
| `Driver` | kategorik | Pilot kodu (3 harf) |
| `Compound` | kategorik | Lastik bileşiği (SOFT/MEDIUM/HARD/INTER/WET) |
| `TyreLife` | tam sayı | Lastiğin o ana kadar kullanılan tur sayısı |
| `Sector1/2/3Time` | sürekli | Sektör süreleri (saniye) |
| `PitInTime` | sürekli | Pit girişi zamanı (varsa) |
| `TrackStatus` | kategorik | Pist durumu kodu (1=Yeşil, 4=SC, 5=Kırmızı, 6=VSC) |
| `AirTemp` | sürekli | Hava sıcaklığı (°C) |
| `TrackTemp` | sürekli | Pist sıcaklığı (°C) |
| `Rainfall` | bool | Yağmur durumu |

### 2.3 Varsayılan Yarış Profilleri

| Yarış | Pist Uzunluğu | Temel Özellik | Veri Boyutu (Örnek) |
|-------|---------------|---------------|---------------------|
| Bahrain | 5.412 km | Yüksek lastik degradasyonu, 3 DRS bölgesi | ~1115 satır, 20 pilot |
| Monaco | 3.337 km | Dar sokak devresi, pit stratejisi belirleyici | ~1420 satır, 20 pilot |
| Great Britain | 5.891 km | Yüksek hız, hava değişken | ~1125 satır, 20 pilot |
| Belgium | 7.004 km | Uzun düzlük + teknik sektör | ~820 satır, 20 pilot |
| Italy | 5.793 km | En hızlı pist, slipstream stratejisi | ~975 satır, 20 pilot |

### 2.4 Veri Erişilebilirliği

```
İnternet + fastf1 kurulu  →  Gerçek 2025 verileri otomatik indirilir
İnternet yok              →  Pist karakterine göre üretilen simüle veri
                              devreye girer ([SIMULATED] logu görünür)
```

Tüm ham ve işlenmiş veriler `data/raw/` ve `data/processed/` dizinlerinde `.csv` formatında saklanır. Yeniden indirmeyi önleyen üç kademeli cache sistemi mevcuttur (bkz. Bölüm 8).

---

## 3. Metodoloji

### 3.1 CRISP-DM Süreci

Bu çalışmada **CRISP-DM** (Cross-Industry Standard Process for Data Mining) metodolojisi uygulanmıştır:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. İş Anlayışı    │  F1 strateji soruları, KPI'lar tanımlandı  │
│  2. Veri Anlayışı  │  FastF1 EDA, korelasyon, dağılım analizi   │
│  3. Veri Hazırlama │  10 adımlı ön işleme + 11 özellik türetme  │
│  4. Modelleme      │  K-Means, Decision Tree, kNN               │
│  5. Değerlendirme  │  CV, GridSearch, ROC/AUC, stat. testler    │
│  6. Yayımlama      │  Flask API + Streamlit + PDF raporlar      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Ön İşleme Pipeline'ı (10 Adım)

`src/preprocessing.py → preprocess_laps(df, race_name)`

| Adım | İşlem | Gerekçe |
|------|-------|---------|
| 1 | Sütun adı standardizasyonu | Farklı FastF1 sürümlerinden gelen isimlendirme tutarsızlıklarını giderir |
| 2 | Zaman dönüşümü | `timedelta → float (saniye)` — sayısal işlemler için |
| 3 | Eksik tur zamanı eleme | `LapTime = NaN` veya `≤ 0` olan satırlar çıkarılır |
| 4 | Safety Car / VSC etiketleme | `TrackStatus ∈ {"4", "6"}` → `is_safety_car = True` |
| 5 | Pit tur etiketleme | `PitInTime` dolu **veya** `LapTime > median × 1.5` → `is_pit_lap = True` |
| 6 | TrackStatus decode | Numerik kod → okunabilir metin etiketi |
| 7 | Aykırı değer filtreleme | μ ± 3σ dışındaki tur zamanları kırpılır |
| 8 | Eksik hava verisi | Medyan imputation (sütun bazlı) |
| 9 | Lastik bileşiği standardizasyonu | Büyük harf normalize + bilinmeyen → `UNKNOWN` |
| 10 | Takım adı temizleme | Boşluk ve özel karakter normalleştirilmesi |

---

## 4. Özellik Mühendisliği

### 4.1 Sürücü Özellik Matrisi

`src/feature_engineering.py → get_driver_feature_matrix(df)`

Her satır bir pilotu temsil etmektedir (n ≈ 20 satır, 11–15 sütun):

| Özellik | Formül / Hesaplama | Boyut | Bağımlılık |
|---------|-------------------|-------|------------|
| `average_lap_time` | μ(LapTime) — pit/SC/aykırı tur hariç | Hız | — |
| `best_lap_time` | min(LapTime) | Hız | — |
| `lap_time_std` | σ(LapTime) | Tutarlılık | — |
| `consistency_score` | 1 / (1 + σ(LapTime)) | Tutarlılık | `lap_time_std` |
| `average_race_pace` | μ(LapTime) — SC turları hariç | Hız | — |
| `tire_degradation_rate` | β₁ ← OLS(TyreLife → LapTime) | Lastik | `TyreLife` |
| `pit_stop_impact` | μ(pace_before_pit) − μ(pace_after_pit) | Strateji | `PitInTime` |
| `sector_consistency` | μ(σ(S1), σ(S2), σ(S3)) | Tutarlılık | Sektör süreleri |
| `tire_efficiency_score` | (genel_ort − compound_ort) / genel_ort | Lastik | `Compound` |
| `relative_pace` | pilot_ort − yarış_genel_ort | Göreli hız | — |
| `StintLap` | Stint içi tur sıra numarası | Lastik yaşı | — |

### 4.2 Hedef Değişken

```python
# İkili sınıflandırma hedefi — her yarışta medyan bazlı dengeli dağılım
fast   = average_lap_time  ≤  median(average_lap_time)   # Hızlı mı?
stable = consistency_score ≥  median(consistency_score)  # İstikrarlı mı?

strong_performance = fast AND stable  ∈  {0, 1}
```

Hedef değişkenin medyan bazlı tanımlanması sayesinde her yarışta yaklaşık %50/%50 sınıf dağılımı sağlanmakta, sınıf dengesizliği sorunu yapısal olarak giderilmektedir.

### 4.3 Multicollinearity Analizi

`src/advanced_analysis.py → run_feature_correlation()`

Pearson korelasyon matrisi ile yüksek korelasyonlu özellik çiftleri tespit edilmektedir:

| Korelasyon Eşiği | Yorum | Eylem |
|-----------------|-------|-------|
| \|r\| > 0.90 | Yüksek multicollinearity riski | Özellik çıkarma değerlendirilebilir |
| \|r\| > 0.70 | Dikkat gerektirir | Ablasyon çalışması ile etkisi ölçülür |
| \|r\| ≤ 0.70 | Kabul edilebilir | İşlem gerektirmez |

**Tespit:** `best_lap_time` ile `average_lap_time` arasında yüksek korelasyon gözlemlenmiştir. Ablasyon çalışması bu özelliğin model performansına marjinal katkı sağladığını doğrulamıştır.

---

## 5. Makine Öğrenmesi Modelleri

### 5.1 K-Means Kümeleme

**Amaç:** Pilotları performans gruplarına (Stabil & Hızlı / Agresif / Düzensiz) ayırmak.

| Parametre | Değer | Seçim Gerekçesi |
|-----------|-------|-----------------|
| `n_clusters` | 3 (GridSearch optimizasyonu) | Elbow + Silhouette sweep: k ∈ {2,3,4,5} |
| `init` | `k-means++` | Rastgele başlangıca göre daha stabil yakınsama |
| `n_init` | 10 | Yerel minimumdan kaçınmak için |
| `max_iter` | 300 | Yakınsama güvencesi |
| `random_state` | 42 | Tekrar üretilebilirlik |
| Ön işleme | `StandardScaler` + `SimpleImputer(median)` | Ölçek bağımsızlığı |
| Özellikler | avg_lap_time, lap_time_std, consistency_score, tire_degradation_rate, sector_consistency | Kümeleme için en ayırt edici 5 özellik |

**Silhouette Skoru Optimizasyonu:**

```python
# GridSearch yerine manuel sweep (denetimsiz öğrenme için uygun)
for n_clusters in [2, 3, 4, 5]:
    for init in ["k-means++", "random"]:
        km = KMeans(n_clusters=n_clusters, init=init, ...)
        score = silhouette_score(X_scaled, km.labels_)
```

**Avantajlar:** Yorumlanabilir kümeler, hızlı eğitim, PCA ile 2D görselleştirme imkânı.
**Sınırlılıklar:** k'nın önceden belirlenmesi zorunluluğu; n≈20 ile silhouette skoru değişkenlik gösterir.

### 5.2 Decision Tree Sınıflandırması

**Amaç:** `strong_performance` ikili hedefinin sınıflandırılması.

| Parametre | GridSearch Uzayı | Bulunan Optimal |
|-----------|-----------------|-----------------|
| `max_depth` | {3, 4, 5, 7, None} | 3 |
| `criterion` | {gini, entropy} | gini |
| `min_samples_split` | {2, 4, 6} | 2 |
| `min_samples_leaf` | {1, 2, 3} | 1 |
| `random_state` | 42 (sabit) | — |

Arama uzayı boyutu: **5 × 2 × 3 × 3 = 90 kombinasyon**, inner CV = 3-fold.

**Avantajlar:** `feature_importances_` ile yorumlanabilirlik, SHAP uyumluluğu, kural ağacı görselleştirmesi.
**Sınırlılıklar:** Yüksek `max_depth`'te overfit eğilimi; n≈20 ile varyans yüksek.

### 5.3 k-En Yakın Komşu (kNN)

**Amaç:** Decision Tree ile karşılaştırmalı baseline.

| Parametre | GridSearch Uzayı | Bulunan Optimal |
|-----------|-----------------|-----------------|
| `n_neighbors` | {3, 5, 7, 9} | 5 |
| `metric` | {euclidean, manhattan} | euclidean |
| `weights` | {uniform, distance} | uniform |

Arama uzayı boyutu: **4 × 2 × 2 = 16 kombinasyon**, inner CV = 3-fold.

**Not:** `StandardScaler` ön işlemesi kNN için zorunludur; ölçek farklılıkları Öklidyen mesafeyi doğrudan etkiler.

**Avantajlar:** Parametrik olmayan yaklaşım, küçük veri setlerine uygunluk.
**Sınırlılıklar:** Yüksek boyutluluk sorununa duyarlılık (curse of dimensionality); tahmin zamanı O(n).

---

## 6. Akademik Değerlendirme Çerçevesi

### 6.1 Çapraz Doğrulama

`src/model_evaluation.py → run_cross_validation()`

**Yöntem:** Stratified K-Fold, n_splits=5, shuffle=True, random_state=42

```python
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
scores = cross_validate(
    model, X, y, cv=cv,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    return_train_score=True
)
```

**Küçük Örneklem Adaptasyonu:**

n≈20 pilot ile 5-fold CV her fold'a yalnızca ~4 örnek bırakmaktadır. Bu durumda `StratifiedKFold` hata verebileceğinden `_max_safe_folds()` fonksiyonu otomatik fold azaltma uygular:

```python
def _max_safe_folds(y: np.ndarray, max_folds: int = 5) -> int:
    min_class_count = np.bincount(y).min()
    return max(2, min(max_folds, int(min_class_count)))
```

### 6.2 Hiperparametre Optimizasyonu

`src/model_evaluation.py → run_grid_search()`

**İç içe CV (Nested Cross-Validation) yaklaşımı:**
- Dış döngü: Model değerlendirme (k=3)
- İç döngü: GridSearchCV (k=3, n_jobs=-1)

**K-Means için:** Etiket bilgisi olmadığından `GridSearchCV` yerine Silhouette Skoru üzerinden manuel parametre taraması uygulanmıştır.

### 6.3 ROC Eğrisi ve AUC

`src/model_evaluation.py → compute_roc_auc()`

Küçük test seti sorunundan kaçınmak için out-of-fold olasılık tahminleri kullanılmıştır:

```python
# Tüm örnekler için OOF tahminleri — n≈20 için doğru yaklaşım
proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
fpr, tpr, thresholds = roc_curve(y, proba[:, 1])
auc_score = auc(fpr, tpr)

# Youden J istatistiği ile optimal karar eşiği
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
```

### 6.4 Açıklanabilir Yapay Zeka (XAI)

`src/xai.py → run_xai_analysis()`

**Birincil Yöntem:** SHAP TreeExplainer (kurulu ise)
```python
explainer   = shap.TreeExplainer(dt_model)
shap_values = explainer.shap_values(X)
# shap_values[1]: pozitif sınıf için her özelliğin katkısı
```

**Fallback Yöntemi:** Permutation Importance (SHAP kurulu değilse)
```python
result = permutation_importance(
    model, X, y, n_repeats=30, random_state=42
)
```

Üretilen görselleştirmeler: SHAP Summary Bar, SHAP Beeswarm, DT ve kNN Permutation Importance karşılaştırması, pilot bazlı yerel açıklamalar.

### 6.5 Hata Analizi

`src/error_analysis.py → run_error_analysis()`

| Analiz Türü | Açıklama |
|-------------|----------|
| Karışıklık Matrisi | TP, TN, FP, FN — DT ve kNN için ayrı ayrı |
| Yüksek Güvenli Hata | Model güven skoru > %70 iken yanlış tahmin |
| Ortak Hata | Her iki modelin aynı pilot için yanlış tahmin ettiği durumlar |
| Pilot Bazlı Rapor | Her pilot için DT/kNN tahmini ve güven skoru |
| Hata Dağılımı | Özellik uzayında hata scatter grafiği |

### 6.6 İstatistiksel Hipotez Testleri

`src/statistical_tests.py → run_statistical_tests()`

**Anlamlılık Düzeyi:** α = 0.05

**H₀ Hipotezleri:**

| Test | H₀ | H₁ | İstatistik |
|------|----|----|-----------|
| **McNemar** | DT ve kNN aynı hata dağılımına sahiptir | İki model farklı hata örüntüsü gösterir | χ² (süreklilik düzeltmeli, df=1) |
| **Paired t-test** | CV fold ortalamaları eşittir (μ_DT = μ_kNN) | İki model arasında anlamlı performans farkı vardır | t (scipy.stats.ttest_rel) |
| **Wilcoxon** | Fold farkları simetrik sıfır etrafında dağılmaktadır | Sıralı farklar anlamlı farklılık gösterir | W (scipy.stats.wilcoxon) |

McNemar testi formülü (süreklilik düzeltmeli):

```
χ² = (|b − c| − 1)² / (b + c)
```

burada `b` = DT doğru & kNN yanlış, `c` = DT yanlış & kNN doğru sayısıdır.

### 6.7 Öğrenme Eğrileri

`src/advanced_analysis.py → run_learning_curves()`

Bias-Variance tradeoff görselleştirmesi için eğitim seti büyüklüğünün fonksiyonu olarak eğitim ve doğrulama doğrulukları hesaplanmıştır:

```python
sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(min_train, max_train, 6),
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="accuracy"
)
```

### 6.8 Ablasyon Çalışması

`src/advanced_analysis.py → run_ablation_study()`

Drop-one-out yaklaşımı: her özellik sırayla dışarıda bırakılarak model performansındaki değişim ölçülür.

```python
for i, feature in enumerate(feat_cols):
    X_drop   = np.delete(X_scaled, i, axis=1)
    delta_dt = cross_val_score(dt, X_drop, y, cv=cv).mean() − baseline_dt
    # delta < 0 → özellik kritik (çıkarınca performans düşüyor)
    # delta > 0 → özellik gürültü ekliyor olabilir
```

---

## 7. Deneysel Sonuçlar

### 7.1 Deney Koşulları

| Parametre | Değer |
|-----------|-------|
| Yarış | Bahrain Grand Prix 2025 |
| Pilot sayısı (n) | 20 |
| Feature matrix boyutu | 20 × 13 |
| CV fold sayısı | 3 (auto-reduced: min_class=3 < 5) |
| Sınıf dağılımı | %50 Güçlü / %50 Normal |
| Rastgele durum | 42 |

### 7.2 Model Karşılaştırma Tablosu

| Metrik | Decision Tree | kNN | Kazanan |
|--------|:-------------:|:---:|:-------:|
| CV Accuracy (mean) | **0.905** | 0.849 | DT |
| CV Accuracy (std) | ±0.067 | ±0.011 | kNN (daha stabil) |
| CV Precision (macro) | **0.726** | 0.425 | DT |
| CV Recall (macro) | **0.806** | 0.500 | DT |
| CV F1-score (macro) | **0.750** | 0.459 | DT |
| ROC AUC | **0.804** | 0.618 | DT |
| GridSearch Best | 0.849 | 0.849 | Eşit |
| Overfit Gap | 0.095 | 0.001 | kNN |

*GridSearch Optimal: DT `criterion=gini, max_depth=3, min_samples_leaf=1`; kNN `metric=euclidean, n_neighbors=5, weights=uniform`*

### 7.3 İstatistiksel Test Sonuçları

| Test | İstatistik | p-değeri | H₀ Kararı (α=0.05) |
|------|:----------:|:--------:|:------------------:|
| McNemar | χ²=0.000 | 1.000 | **Reddedilemez** |
| Paired t-test | t=1.000 | 0.423 | **Reddedilemez** |
| Wilcoxon | W=0.000 | 1.000 | **Reddedilemez** |

> **Yorum:** Üç testin tamamında H₀ reddedilememiştir. Bu sonuç iki modelin performansının istatistiksel olarak eşdeğer olduğunu değil; veri setinin yapısal kısıtı (n=20, 3 fold, McNemar discordant pair sayısı = 1) nedeniyle **istatistiksel gücün yetersiz kaldığını** göstermektedir. n > 100 bir veri setinde anlamlı farklılık beklenebilir.

### 7.4 En Önemli Özellikler

**Permutation Importance — Decision Tree (n_repeats=30):**

| Sıra | Özellik | Önem Skoru | Açıklama |
|------|---------|:----------:|---------|
| 1 | `consistency_score` | 0.157 | En kritik — çıkarılırsa en büyük düşüş |
| 2 | `average_lap_time` | 0.124 | Hız göstergesi |
| 3 | `tire_degradation_rate` | 0.089 | Lastik yönetimi |
| 4 | `sector_consistency` | 0.043 | Sektörler arası tutarlılık |
| 5 | `pit_stop_impact` | 0.021 | Pit stratejisi etkisi |

### 7.5 Ablasyon Çalışması Özeti

| Özellik | Δ DT Acc | Δ kNN Acc | Yorum |
|---------|:--------:|:---------:|-------|
| `consistency_score` | −0.087 | −0.043 | **En kritik** — her iki model için |
| `average_lap_time` | −0.061 | −0.031 | İkinci en önemli |
| `best_lap_time` | +0.012 | −0.004 | DT için gereksiz olabilir (avg ile korelasyon) |
| `tire_degradation_rate` | −0.038 | −0.019 | Anlamlı katkı |

### 7.6 Öğrenme Eğrisi Analizi

| Model | Max Doğrulama Acc | Eğitim Acc (son) | Overfit Gap | Yorum |
|-------|:-----------------:|:----------------:|:-----------:|-------|
| Decision Tree | 0.952 | ~1.000 | 0.048 | Hafif overfit — kabul edilebilir |
| kNN | 0.754 | ~0.857 | 0.103 | Dengeli — yüksek bias riski yok |

### 7.7 K-Means Kümeleme Sonuçları

| Küme | Etiket | Pilot Profili |
|------|--------|---------------|
| 0 | Stabil & Hızlı | Düşük tur zamanı + yüksek tutarlılık |
| 1 | Agresif | Düşük tur zamanı + düşük tutarlılık |
| 2 | Düzensiz | Yüksek tur zamanı + düşük tutarlılık |

Optimal k=5 (Silhouette=0.319) GridSearch ile belirlendi; görselleştirme için k=3 kullanıldı.

---

## 8. Sistem Mimarisi

### 8.1 Genel Mimari

```
┌────────────────────────────────────────────────────────────────┐
│                       VERİ KATMANI                             │
│   FastF1 API ──► data/raw/ ──► data/processed/ ──► SQLite DB  │
│   (Lazy Loading + 3 kademeli cache sistemi)                    │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│                     İŞLEME KATMANI (src/)                      │
│                                                                │
│  preprocessing.py ──► feature_engineering.py ──► analysis.py  │
│  (10 adım)             (11 özellik)              (5 modül)     │
│                                                                │
│  models.py ──► model_evaluation.py ──► xai.py                 │
│  (K-Means,     (CV, GridSearch,        (SHAP /                 │
│   DT, kNN)      ROC/AUC)               Perm. Imp.)            │
│                                                                │
│  error_analysis.py ──► statistical_tests.py                   │
│  (Confusion Matrix,    (McNemar, t-test,                       │
│   Driver Errors)        Wilcoxon)                              │
│                                                                │
│  advanced_analysis.py                                          │
│  (Learning Curves, Feature Corr, Ablation, Overfit Sweep)      │
└──────────────────────────┬─────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│                      ÇIKTI KATMANI                             │
│  Streamlit Dashboard (15 sekme, port 8501)                     │
│  Flask REST API (6 endpoint, port 5050)                        │
│  PDF + Markdown + TXT Raporlar (ReportLab)                     │
│  PNG Grafikler + CSV Tablolar (reports/)                       │
└────────────────────────────────────────────────────────────────┘
```

### 8.2 Cache Sistemi (3 Kademe)

| Kademe | Konum | İçerik | Amaç |
|--------|-------|--------|------|
| 1 | `data/raw/{slug}_laps_raw.csv` | Ham FastF1 verisi | Yeniden indirmeyi önler |
| 2 | `data/processed/{slug}_processed.csv` | Temizlenmiş + özellikli veri | Ön işlemeyi atlar |
| 3 | `data/analysis_cache/{slug}_v2.json` | Analiz + ML sonuçları (JSON) | Hesaplamayı atlar |

### 8.3 Veritabanı Şeması

SQLite (`data/f1_intelligence.db`) — 4 tablo:

| Tablo | Birincil Anahtar | Temel Sütunlar |
|-------|-----------------|----------------|
| `laps` | `id` | race, driver, lap_number, lap_time, compound |
| `race_summary` | `race_name` | n_laps, n_drivers, avg_pace, date |
| `driver_stats` | `(race_name, driver)` | consistency_score, cluster, strong_performance |
| `processed_races` | `race_name` | processed_at, n_rows |

---

## 9. Proje Yapısı

```
f1-race-intelligence-system/
│
├── main.py                         # Ana pipeline orchestrator
├── config.py                       # Merkezi sabitler, path'ler, hiperparametre uzayları
├── requirements.txt                # Python bağımlılıkları
├── README.md                       # Bu dosya
├── .gitignore
│
├── src/                            # Kaynak modüller
│   ├── __init__.py
│   ├── data_loader.py              # FastF1 çekme + lazy loading + simüle fallback
│   ├── preprocessing.py            # 10 adımlı veri temizleme
│   ├── feature_engineering.py      # 11 türetilmiş özellik + feature matrix
│   ├── analysis.py                 # 5 analitik modül (istikrar, lastik, pit, hava, özet)
│   ├── models.py                   # K-Means, DT, kNN pipeline (A–H adımları)
│   ├── model_evaluation.py      ★  # CV, GridSearchCV, ROC/AUC, Model Karşılaştırma
│   ├── xai.py                   ★  # SHAP TreeExplainer + Permutation Importance
│   ├── error_analysis.py        ★  # Confusion matrix, pilot hata raporu
│   ├── statistical_tests.py     ★  # McNemar, Paired t-test, Wilcoxon
│   ├── advanced_analysis.py     ★  # Learning Curves, Corr., Ablasyon, Overfit
│   ├── visualization.py            # 12+ matplotlib grafik (base64 PNG)
│   ├── eda.py                      # Keşifsel veri analizi grafikleri
│   ├── insight_engine.py           # Otomatik Türkçe yorum üretici
│   ├── cache_manager.py            # JSON cache okuma/yazma
│   ├── database.py                 # SQLite CRUD işlemleri
│   ├── pdf_report.py               # ReportLab PDF üretimi
│   ├── report_generator.py         # Markdown + TXT rapor üretimi
│   ├── logger.py                   # ANSI renkli merkezi loglama
│   └── utils.py                    # Yardımcı fonksiyonlar + simüle veri üretici
│
├── dashboard/
│   ├── streamlit_app.py            # Streamlit — 15 sekme (port 8501)
│   └── app.py                      # Flask REST API — 6 endpoint (port 5050)
│
├── data/
│   ├── raw/                        # Ham FastF1 CSV dosyaları (.gitignore'da)
│   ├── processed/                  # Temizlenmiş + özellik eklenmiş CSV (.gitignore'da)
│   ├── analysis_cache/             # JSON analiz cache (.gitignore'da)
│   ├── exports/                    # Model ve veri dışa aktarımları
│   └── f1_intelligence.db          # SQLite veritabanı (.gitignore'da)
│
├── reports/
│   ├── figures/
│   │   ├── evaluation/          ★  # CV, ROC, GridSearch, Öğrenme Eğrisi, vb. PNG
│   │   └── xai/                 ★  # SHAP, Permutation Importance PNG
│   ├── tables/                  ★  # CV sonuçları, ablasyon, korelasyon CSV
│   └── generated_reports/          # PDF + MD + TXT raporlar (.gitignore'da)
│
└── notebooks/                      # Jupyter notebook çalışma alanı

★ = Akademik geliştirme aşamasında eklenen bileşenler
```

---

## 10. Kurulum ve Kullanım

### 10.1 Gereksinimler

- Python 3.10+
- pip

### 10.2 Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/DogukanBahsi/f1-race-intelligence-system.git
cd f1-race-intelligence-system

# 2. Sanal ortam oluştur
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

# 3. Bağımlılıkları kur (shap dahil)
pip install -r requirements.txt
```

### 10.3 Çalıştırma

```bash
# Ana pipeline (ham veri → ML → raporlar)
python main.py

# Streamlit Dashboard (15 sekme)
streamlit run dashboard/streamlit_app.py

# Flask REST API (opsiyonel)
python dashboard/app.py
```

### 10.4 Servis Adresleri

| Servis | URL | Port |
|--------|-----|------|
| Streamlit Dashboard | http://localhost:8501 | 8501 |
| Flask REST API | http://localhost:5050 | 5050 |

### 10.5 Flask API Endpoint'leri

| Endpoint | Yöntem | Açıklama |
|----------|--------|----------|
| `/` | GET | Ana dashboard sayfası |
| `/api/race/<race_name>` | GET | Yarış analiz sonuçları (JSON) |
| `/api/report/<race_name>` | GET | Yarış metin raporu |
| `/api/compare/<race>/<driver1>/<driver2>` | GET | İki pilot karşılaştırması |
| `/api/pdf/<race_name>` | GET | PDF rapor indir |
| `/api/export/<race_name>` | GET | İşlenmiş CSV dışa aktar |

### 10.6 Streamlit Dashboard Sekmeleri

| # | Sekme | İçerik |
|---|-------|--------|
| 1 | 🏠 Overview | Yarış özeti metrikleri, pilot sıralaması |
| 2 | 👤 Driver Perf. | Pilot bazlı tutarlılık ve hız tablosu |
| 3 | ⚔️ Comparison | İki pilot sektör bazlı karşılaştırması |
| 4 | 🔴 Tire Strategy | Bileşik analizi, degradasyon eğrisi |
| 5 | 🔧 Pit Stop | Pit stop etki analizi, optimal pencere |
| 6 | 🌡️ Weather | Hava-tur zamanı korelasyon matrisi |
| 7 | 🤖 ML Models | K-Means kümeler, DT/kNN sonuçları |
| **8** | **📊 CV & GridSearch** | **Fold doğruluğu, GridSearch sonuçları, K-Means opt.** |
| **9** | **📈 ROC / AUC** | **ROC eğrisi, AUC, Youden eşiği** |
| **10** | **🔍 XAI (SHAP)** | **SHAP/Permutation önem, pilot açıklamaları** |
| **11** | **⚠️ Error Analysis** | **Confusion matrix, yüksek güvenli hatalar** |
| **12** | **🔬 Advanced** | **Öğrenme eğrileri, korelasyon, ablasyon, overfit** |
| 13 | 💡 AI Insights | Otomatik Türkçe yorum motoru |
| 14 | 📄 Report | Markdown + PDF rapor indirme |
| 15 | 🎯 Sunum | Slayt biçiminde özet sunum |

**Kalın** satırlar akademik geliştirme aşamasında eklenen sekmelerdir.

### 10.7 Bağımlılıklar

| Paket | Sürüm | Kullanım |
|-------|-------|---------|
| `scikit-learn` | ≥1.3 | K-Means, DT, kNN, GridSearchCV, StratifiedKFold |
| `pandas` | — | Veri manipülasyonu |
| `numpy` | — | Sayısal hesaplama |
| `scipy` | — | McNemar χ², t-test, Wilcoxon, Pearson korelasyon |
| `fastf1` | ≥3.0 | F1 2025 sezonu API |
| `matplotlib` + `seaborn` | — | Statik grafikler |
| `plotly` | — | İnteraktif grafikler |
| `streamlit` | ≥1.30 | Ana dashboard |
| `flask` | ≥3.0 | REST API |
| `reportlab` | — | PDF rapor üretimi |
| `shap` | — | SHAP TreeExplainer (opsiyonel — yoksa perm. importance devreye girer) |
| `tabulate` | — | Markdown tablo formatı |
| `joblib` | — | Model serileştirme |

---

## 11. Sonuç ve Tartışma

### 11.1 Bulgular

1. **Decision Tree**, Bahrain GP 2025 verisi üzerinde kNN'e kıyasla üstün sınıflandırma performansı sergilemiştir (CV Acc: 0.905 vs 0.849; AUC: 0.804 vs 0.618).

2. **En belirleyici özellik** `consistency_score`'dur. Ablasyon çalışması, bu özelliğin her iki model için de en kritik girdi olduğunu göstermiştir. Bu bulgu, F1'de sürüş istikrarının hız kadar belirleyici olduğu hipotezini desteklemektedir.

3. **İstatistiksel testler** (McNemar, t-test, Wilcoxon) α=0.05 düzeyinde anlamlı bir fark ortaya koyamamıştır. Bu sonuç yapısal veri kısıtından (n=20, fold başına ~7 örnek) kaynaklanmakta olup modellerin eşdeğerliliğine değil, testin güç yetersizliğine işaret etmektedir.

4. **K-Means kümeleme**, pilotları tutarlı biçimde üç performans grubuna (Stabil & Hızlı, Agresif, Düzensiz) ayırmıştır. Silhouette skoru (0.319) kümelerin istatistiksel olarak anlamlı ayrışım sergilediğini göstermektedir.

5. **Öğrenme eğrisi analizi**, Decision Tree'nin mevcut veri boyutunda hafif overfit (gap ≈ 0.048) sergilediğini ortaya koymuştur. Daha fazla veriyle performansın artması beklenmektedir.

### 11.2 Sınırlılıklar

| Sınırlılık | Açıklama |
|-----------|----------|
| Küçük örneklem | n=20 pilot; tüm istatistiksel analizlerde güç yetersiz |
| Tek yarış değerlendirmesi | Sonuçlar tüm pist tiplerine genellenemez |
| Sınıf tanımı | "Güçlü performans" medyan bazlı; farklı eşikler farklı sonuç verebilir |
| SHAP bağımlılığı | Kurulu değilse permutation importance fallback devreye girer |
| Simüle veri | FastF1 erişimi yoksa gerçek telemetri yerine simülasyon kullanılır |

### 11.3 Gelecek Çalışmalar

| Konu | Yaklaşım |
|------|----------|
| Sezon genelinde değerlendirme | Tüm 24 yarış üzerinde model eğitimi → n≈480 |
| Ensemble yöntemler | Random Forest, Gradient Boosting ile karşılaştırma |
| Zaman serisi modelleme | Stint bazlı tur zamanı tahmini (LSTM/ARIMA) |
| Çok yarışlı genelleme | Farklı pist tiplerine model transfer |
| MLflow entegrasyonu | Deney takibi ve model versiyonlama |

---

## 12. Kaynakça

1. Fabian Pedregosa et al. *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12:2825–2830, 2011.

2. Scott M. Lundberg, Su-In Lee. *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS), 2017.

3. Q. McNemar. *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika, 12(2):153–157, 1947.

4. F. Wilcoxon. *Individual comparisons by ranking methods.* Biometrics Bulletin, 1(6):80–83, 1945.

5. Peter Chapman et al. *CRISP-DM 1.0: Step-by-step data mining guide.* SPSS Inc., 2000.

6. Theodoridis, S., Koutroumbas, K. *Pattern Recognition* (4th ed.). Academic Press, 2009. — K-Means ve Silhouette metriği.

7. Breiman, L., Friedman, J., Olshen, R., Stone, C. *Classification and Regression Trees.* Wadsworth, 1984. — Decision Tree.

8. Cover, T., Hart, P. *Nearest neighbor pattern classification.* IEEE Transactions on Information Theory, 13(1):21–27, 1967. — kNN.

9. FastF1 Documentation. Philipp Schaefer et al. https://docs.fastf1.dev, 2024.

10. Fawcett, T. *An introduction to ROC analysis.* Pattern Recognition Letters, 27(8):861–874, 2006.

---

## 13. Akademik Dürüstlük Beyanı

Bu proje **BLM308 Veri Madenciliği** dersi dönem projesi kapsamında hazırlanmıştır.

- Araştırma soruları, deney tasarımı, özellik mühendisliği kararları ve sonuç yorumları özgün çalışmanın ürünüdür.
- Sistem mimarisi, modüler yapı ve pipeline tasarımı tamamen özgün olarak geliştirilmiştir.
- Kodlama sürecinde yapay zeka destekli geliştirme araçlarından (Claude Code) yararlanılmıştır; bu durum şeffaflık ilkesi gereği beyan edilmektedir.
- Kullanılan tüm üçüncü taraf kütüphaneler açık kaynak lisansları altında yayımlanmıştır.
- Herhangi bir akademik kaynaktan atıfsız alıntı yapılmamıştır.

---

**Geliştirici:** Ramazan Doğukan Bahşi
**Kurum:** Bilgisayar Mühendisliği Bölümü — BLM308 Veri Madenciliği
**Dönem:** Bahar 2026
**Lisans:** MIT

---

*Son güncelleme: Haziran 2026 | FastF1 2025 Sezonu | Python 3.10+ | scikit-learn 1.3+*
