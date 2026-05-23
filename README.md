# 🏎️ F1 Race Intelligence & Strategy Analysis System

> Formula 1 yarış verilerini analiz eden, makine öğrenmesi ile örüntüleri keşfeden ve
> sonuçları iki farklı profesyonel dashboard ile sunan veri madenciliği sistemi.

---

## ⚠️ Veri Kaynağı Durumu

| Durum | Açıklama |
|-------|----------|
| 🌐 **İnternet + FastF1** | `pip install fastf1` → Gerçek 2025 F1 verisi otomatik indirilir |
| 📴 **İnternet yok** | Otomatik simüle fallback aktif olur (**[SIMULATED]** logları görünür) |

**Bu repo şu an simüle veri ile çalışmaktadır.**
Gerçek veriye geçmek: `pip install fastf1` + `data/raw/` klasörünü sil + `python main.py`

---

## 📋 Proje Özeti

| | |
|---|---|
| **Ders** | Veri Madenciliği / Veri Analizi |
| **Sezon** | 2025 Formula 1 (24 yarış) |
| **Veri Kaynağı** | FastF1 API (Ergast + F1 Live Timing) |
| **Fallback** | Gerçekçi simüle veri (pist bazlı, 20 pilot, 2-stop strateji) |
| **Dashboard** | Flask (port 5050) + Streamlit (port 8501) |
| **Modeller** | K-Means, Decision Tree, kNN |
| **Raporlar** | Markdown, TXT, PDF (ReportLab) |

---

## 🎯 Problem Tanımı

Formula 1'de strateji kararları —lastik seçimi, pit stop zamanlaması, sıcaklık yönetimi—
sonucu doğrudan etkiler. Bu sistem:

1. **Pilot istikrarını** ölçer (Consistency Score = 1/(1+std))
2. **Lastik stratejisi** etkisini nicel analiz eder
3. **Pit stop faydasını** pit öncesi/sonrası pace farkıyla hesaplar
4. **Sıcaklık etkisini** Pearson korelasyonu ile test eder
5. **K-Means** ile pilotları performans kümelerine ayırır
6. **Decision Tree + kNN** ile güçlü performansı tahmin eder
7. **AI Insight Engine** ile otomatik Türkçe yorumlar üretir

---

## 🔄 Veri Akış Diyagramı

```
FastF1 API (internet varsa)
    │
    ▼  is_race_processed()? → Evet → Processed CSV'den oku  ─┐
    │                                                          │
    Hayır: ham veriyi indir                                    │
    │                                                          │
    ▼ fetch_race_data_from_fastf1()                            │
Raw CSV (data/raw/) ←── Simüle fallback (internet yoksa)      │
    │                                                          │
    ▼ preprocess_laps()                                        │
Cleaned DataFrame  ←──────────────────────────────────────────┘
    │
    ▼ engineer_features()
Feature-Rich DataFrame (35+ sütun)
    │
    ├──► SQLite  (data/f1_intelligence.db)
    ├──► Processed CSV (data/processed/)
    │
    ▼ run_all_analyses() + run_all_models()
Analysis Results + ML Models
    │
    ▼ generate_race_insights()
AI Insight Engine (otomatik Türkçe yorumlar)
    │
    ▼ generate_all_charts()
Base64 PNG Grafikleri (12 adet)
    │
    ├──► Analysis Cache (data/analysis_cache/*.json)
    ├──► Markdown + TXT Rapor (reports/generated_reports/)
    ├──► PDF Rapor (ReportLab)
    └──► Flask Dashboard (port 5050)
         Streamlit Dashboard (port 8501)
```

---

## ⚡ Lazy Loading Sistemi

```
Kullanıcı yarış seçer
        │
        ▼
processed_races (SQLite) + data/processed/*.csv var mı?
        │
   Evet ──► CSV'den oku → Analiz yap → Dashboard
        │
   Hayır
        │
        ▼
FastF1 erişilebilir mi?
        │
   Evet ──► İndir → raw CSV kaydet → Preprocess → Feature Eng → Kaydet → Dashboard
        │
   Hayır ──► Simüle veri üret → [Aynı pipeline] → Dashboard
```

---

## 💾 Cache Sistemi

```python
# Analiz sonuçları JSON'a kaydedilir
data/analysis_cache/bahrain_v2.json

# Cache hit log:
[INFO] Cached analysis loaded: Bahrain (oluşturma: 2025-05-23 10:00:00)
# Cache miss (ilk kez):
[INFO] Analysis cache created: Bahrain → bahrain_v2.json
```

---

## 📁 Klasör Yapısı

```
f1-race-intelligence-system/
│
├── data/
│   ├── raw/                   ← FastF1 ham CSV'ler
│   ├── processed/             ← Temizlenmiş + feature'lı CSV'ler
│   ├── exports/               ← Kullanıcı export dosyaları
│   ├── analysis_cache/        ← Analiz JSON cache'i (yeni!)
│   └── f1_intelligence.db     ← SQLite veritabanı
│
├── reports/
│   ├── generated_reports/     ← MD, TXT, PDF raporlar
│   └── figures/               ← EDA grafikleri (PNG)
│
├── dashboard/
│   ├── app.py                 ← Flask dashboard (port 5050)
│   └── streamlit_app.py       ← Streamlit dashboard (port 8501)  ← YENİ
│
├── src/
│   ├── data_loader.py         ← FastF1 + lazy loading
│   ├── preprocessing.py       ← 9 adımlı veri temizleme
│   ├── feature_engineering.py ← 11 yeni özellik
│   ├── eda.py                 ← Keşifsel analiz + grafikler
│   ├── analysis.py            ← 5 analiz modülü
│   ├── models.py              ← K-Means, DT, kNN
│   ├── visualization.py       ← 12 grafik türü (base64 PNG)
│   ├── report_generator.py    ← MD + TXT rapor
│   ├── pdf_report.py          ← PDF rapor (ReportLab)           ← YENİ
│   ├── insight_engine.py      ← AI Insight Engine                ← YENİ
│   ├── cache_manager.py       ← Analiz cache yöneticisi          ← YENİ
│   ├── database.py            ← SQLite CRUD
│   ├── logger.py              ← Renkli terminal logging
│   └── utils.py               ← Yardımcı + simüle veri üretici
│
├── config.py                  ← Tüm sabitler ve path'ler
├── main.py                    ← Ana pipeline orchestrator
└── requirements.txt
```

---

## 🔧 Ön İşleme Adımları (preprocessing.py)

| # | Adım | Yöntem |
|---|------|--------|
| 1 | Sütun standardizasyonu | Farklı kaynak isimleri birleştir |
| 2 | Zaman → saniye | Timedelta → float |
| 3 | Eksik LapTime silme | dropna + >0 filtresi |
| 4 | SC/VSC etiketleme | TrackStatus "4"/"6" |
| 5 | Pit tur etiketleme | PitInTime dolu veya süre > medyan × 1.5 |
| 6 | TrackStatus decode | 1=Yeşil, 4=SC, 5=Red, 6=VSC |
| 7 | Outlier tespiti | ±3σ eşiği |
| 8 | Hava verisi doldurma | Medyan imputation |
| 9 | Compound standardize | SOFT/MEDIUM/HARD/INTERMEDIATE/WET |

---

## ⚙️ Feature Engineering (feature_engineering.py)

| Özellik | Formül / Açıklama |
|---------|-------------------|
| `average_lap_time` | Pit/SC/outlier hariç ort. |
| `best_lap_time` | min(LapTime) |
| `lap_time_std` | std(LapTime) |
| `consistency_score` | 1 / (1 + std) — yüksek = istikrarlı |
| `average_race_pace` | SC hariç ort. yarış hızı |
| `tire_degradation_rate` | linreg(TyreLife, LapTime).slope |
| `pit_stop_impact` | mean(pace_before) − mean(pace_after) |
| `sector_consistency` | mean(std(S1), std(S2), std(S3)) |
| `tire_efficiency_score` | (genel_ort − compound_ort) / genel_ort |
| `relative_pace` | pilot_ort − yarış_genel_ort |
| `StintLap` | Stint içi tur sırası |

---

## 🤖 Algoritmalar

### K-Means Clustering
- **Amaç:** Pilotları Stabil & Hızlı / Agresif / Düzensiz gruplarına ayır
- **k seçimi:** Elbow Method + Silhouette Score (2–7 aralığı test)
- **PCA:** 2D görselleştirme için boyut indirgeme

### Decision Tree Classification
- **Target:** `strong_performance` (hız < medyan AND consistency > medyan)
- **max_depth:** 5 (aşırı öğrenme önlemi)
- **Çıktı:** Feature importance ranking

### kNN Classification
- **Amaç:** Decision Tree ile karşılaştırma
- **k:** min(5, n_pilot − 1) — otomatik ayarlama
- **Metrik:** Euclidean mesafe

### Pearson Korelasyonu
- **AirTemp vs LapTime** — hava etkisi
- **TrackTemp vs LapTime** — pist sıcaklığı etkisi
- **p < 0.05:** İstatistiksel anlamlılık eşiği

---

## 📊 Dashboard Karşılaştırması

| Özellik | Flask (app.py) | Streamlit (streamlit_app.py) |
|---------|---------------|------------------------------|
| Port | 5050 | 8501 |
| Tema | Tam özel CSS | Özel CSS enjeksiyonu |
| Sekmeler | 8 sekme | 10 sekme |
| AI Insights | ✅ | ✅ |
| Driver Compare | ✅ | ✅ (sektör + delta) |
| PDF İndir | ✅ | ✅ (download_button) |
| Sunum Modu | ✅ | ✅ |
| Cache göstergesi | Sidebar | Sidebar |

---

## 🚀 Kurulum ve Çalıştırma

### 1. Temel kurulum

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn flask joblib reportlab
# sqlite3 Python standart kütüphanesinde mevcuttur

# Gerçek FastF1 verisi için (internet gerekli):
pip install fastf1
```

### 2. Pipeline çalıştır

```bash
cd f1-race-intelligence-system
python main.py
```

Çıktı:
```
[SIMULATED] Bahrain: 1140 satır, 20 pilot üretildi.   ← internet yoksa
[INFO]      Real FastF1 data loaded successfully.       ← internet varsa
[SUCCESS]   Decision Tree model trained. Accuracy: 0.833
[SUCCESS]   Dashboard data is ready: Bahrain (4.3s)
[INFO]      PDF raporu oluşturuldu: bahrain_report_*.pdf
```

### 3. Flask Dashboard

```bash
python dashboard/app.py
# → http://localhost:5050
```

### 4. Streamlit Dashboard

```bash
pip install streamlit
streamlit run dashboard/streamlit_app.py
# → http://localhost:8501
```

---

## 🌐 FastF1 Internet Gereksinimleri

| İşlem | İnternet gerekli? |
|-------|------------------|
| `pip install fastf1` | ✅ Evet |
| Gerçek F1 verisi çekme | ✅ Evet |
| FastF1 cache'den okuma | ❌ Hayır |
| Simüle veri pipeline | ❌ Hayır |
| Dashboard + analiz | ❌ Hayır |
| PDF rapor üretme | ❌ Hayır |

---

## 📈 Üretilen Çıktılar

| Çıktı | Konum | Format |
|-------|-------|--------|
| Ham veri | `data/raw/` | CSV |
| İşlenmiş veri | `data/processed/` | CSV |
| Analiz cache | `data/analysis_cache/` | JSON |
| Veritabanı | `data/f1_intelligence.db` | SQLite |
| EDA grafikleri | `reports/figures/` | PNG |
| Markdown rapor | `reports/generated_reports/` | .md |
| TXT rapor | `reports/generated_reports/` | .txt |
| PDF rapor | `reports/generated_reports/` | .pdf (224KB+) |

---

## 🔮 Gelecek İyileştirmeler

- [ ] Streamlit `@st.cache_data` ile daha agresif cache
- [ ] Speed trace / telemetry heatmap (FastF1 telemetri)
- [ ] Grafana CSV export entegrasyonu
- [ ] Docker + docker-compose
- [ ] Çok sezonlu karşılaştırma (2023, 2024, 2025)
- [ ] Yarış simülasyonu (pit stop timing optimizer)

---

## 🐳 Grafana Export (Opsiyonel)

```bash
# data/exports/ klasörünü Grafana'ya bağla
docker run -d -p 3000:3000 grafana/grafana
# CSV plugin ile data/exports/*.csv dosyaları okunabilir
```

---

*F1 Race Intelligence System v2 — Veri Madenciliği / Veri Analizi Dönem Projesi*
*Veri: FastF1 API (internet varsa) | Simüle Fallback (internet yoksa)*
