# 🏎️ F1 Race Intelligence & Strategy Analysis System

Formula 1 yarış verilerini analiz eden, makine öğrenmesi destekli yarış strateji ve performans analiz sistemi.

Sistem:
- Gerçek FastF1 verilerini kullanabilir
- İnternet yoksa otomatik simüle veri üretir
- Flask ve Streamlit dashboardları sunar
- ML modelleri ile pilot performans örüntülerini analiz eder
- PDF / Markdown / TXT rapor oluşturabilir

---

## 🚀 Özellikler

- Gerçek FastF1 veri entegrasyonu
- Simulated fallback sistemi
- Premium Flask dashboard
- Streamlit analytics dashboard
- Driver comparison sistemi
- Tire strategy analizi
- Pit stop impact analizi
- Hava sıcaklığı etkisi analizi
- K-Means / Decision Tree / kNN modelleri
- AI Insight Engine
- PDF / Markdown / TXT export
- Cache & lazy loading sistemi
- SQLite veritabanı desteği
- 24 yarış desteği
- Gerçek zamanlı dashboard deneyimi

---


```

---

## 🧠 Makine Öğrenmesi Modelleri

| Model | Amaç |
|---|---|
| K-Means | Pilot performans kümelendirme |
| Decision Tree | Güçlü performans tahmini |
| kNN | Performans sınıflandırması |
| Pearson Correlation | Hava sıcaklığı & pace ilişkisi |

---

## 📊 Analiz Yetenekleri

- Pilot istikrar analizi
- Lastik aşınma analizi
- Pit stop verimlilik analizi
- Sektör bazlı pace karşılaştırması
- Relative race pace hesaplaması
- Hava etkisi analizi
- AI destekli otomatik yorum üretimi
- Feature importance analizi

---

## ⚙️ Kullanılan Teknolojiler

### Backend
- Python
- Flask
- Streamlit
- SQLite

### Veri & ML
- Pandas
- NumPy
- Scikit-learn
- SciPy

### Görselleştirme
- Matplotlib
- Seaborn
- Plotly

### Raporlama
- ReportLab
- Markdown / TXT Export

### Veri Kaynağı
- FastF1 API

---

## 🔄 Sistem Akışı

```text
FastF1 API
   ↓
Veri Toplama
   ↓
Ön İşleme
   ↓
Feature Engineering
   ↓
Makine Öğrenmesi Analizi
   ↓
Insight Generation
   ↓
Grafikler & Dashboard
```

---

## 📁 Proje Yapısı

```text
f1-race-intelligence-system/
│
├── dashboard/
│   ├── app.py
│   └── streamlit_app.py
│
├── src/
│   ├── analysis.py
│   ├── cache_manager.py
│   ├── data_loader.py
│   ├── insight_engine.py
│   ├── models.py
│   ├── pdf_report.py
│   ├── preprocessing.py
│   ├── visualization.py
│   └── ...
│
├── data/
├── reports/
├── config.py
├── main.py
└── requirements.txt
```

---

## 🚀 Kurulum

### Repoyu Klonla

```bash
git clone https://github.com/DogukanBahsi/f1-race-intelligence-system.git
cd f1-race-intelligence-system
```

---

### Virtual Environment Oluştur

```bash
python -m venv venv
```

### Virtual Environment Aktifleştir

#### Windows PowerShell

```bash
.\venv\Scripts\Activate.ps1
```

---

### Gereksinimleri Kur

```bash
pip install -r requirements.txt
```

---

### Gerçek FastF1 Verisi İçin

```bash
pip install fastf1
```

---

## ▶️ Projeyi Çalıştır

### Ana Pipeline

```bash
python main.py
```

---

### Flask Dashboard

```bash
python dashboard/app.py
```

Tarayıcı:
```text
http://localhost:5050
```

---

### Streamlit Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Tarayıcı:
```text
http://localhost:8501
```

---

## 🌐 Gerçek Veri vs Simüle Veri

| Mod | Açıklama |
|---|---|
| REAL FASTF1 | Gerçek yarış verileri kullanılır |
| SIMULATED | İnternet/FastF1 yoksa gerçekçi simüle veri üretilir |

Sistem gerekli durumlarda otomatik olarak simulated mode'a geçer.

---

## 📄 Üretilen Çıktılar

- İşlenmiş CSV dosyaları
- Analiz cache JSON dosyaları
- SQLite veritabanı
- EDA grafikleri
- PDF raporları
- Markdown raporları
- TXT raporları

---

## 🔮 Gelecek Geliştirmeler

- Telemetry heatmaps
- Çok sezonlu analiz
- Docker desteği
- Grafana entegrasyonu
- Pit strategy optimizer
- Live race monitoring

---

## 👨‍💻 Geliştirici

**Doğukan Bahşi**

Bilgisayar Mühendisliği Öğrencisi  
Veri Analizi & Motorsport Teknolojileri

---

## 📌 Proje Hakkında

Bu proje, Veri Madenciliği / Veri Analizi dersi kapsamında geliştirilmiş Formula 1 yarış strateji ve performans analiz sistemidir.
