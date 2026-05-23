# 🏎️ F1 Race Intelligence Report
## Belgium Grand Prix — 2025 Sezonu
*Rapor tarihi: 23 May 2026, 12:35*

---
## 📊 Yarış Genel Bakış
| Metrik | Değer |
|--------|-------|
| Toplam Tur | 819 |
| Toplam Pilot | 20 |
| En Hızlı Tur | 1:44.861 |
| En Hızlı Pilot | ANT |
| En İstikrarlı Pilot | PIA |
| Ortalama Hava Sıcaklığı | 17.0°C |
| Ortalama Pist Sıcaklığı | 24.1°C |

## 🏁 Pilot İstikrar Analizi
**En İstikrarlı Pilot:** PIA
**En Hızlı Pilot:** ANT

| Sıra | Pilot | Ort. Tur | En Hızlı | Std Sapma | Consistency |
|------|-------|----------|----------|-----------|-------------|
| 1 | PIA | 109.281s | 105.706s | 5.509s | 0.1536 |
| 2 | HAM | 109.933s | 106.534s | 5.781s | 0.1475 |
| 3 | RUS | 110.045s | 106.566s | 5.950s | 0.1439 |
| 4 | GAS | 110.784s | 107.177s | 5.966s | 0.1435 |
| 5 | LEC | 109.633s | 106.174s | 5.984s | 0.1432 |
| 6 | VER | 109.663s | 106.096s | 6.004s | 0.1428 |
| 7 | ALB | 110.199s | 106.813s | 6.005s | 0.1428 |
| 8 | NOR | 109.445s | 105.257s | 6.006s | 0.1427 |
| 9 | BEA | 110.933s | 106.709s | 6.120s | 0.1404 |
| 10 | LAW | 110.419s | 106.649s | 6.160s | 0.1397 |

> Belgium yarışında en stabil pilot PIA olmuştur (Consistency Score: 0.154). En hızlı tur ANT adına kayıtlıdır (1:44.861).

## 🔴 Lastik Stratejisi Analizi
**En Avantajlı Lastik:** UNKNOWN

| Lastik | Ort. Tur (s) | En Hızlı (s) | Std Sapma | Tur Sayısı |
|--------|--------------|--------------|-----------|------------|
| UNKNOWN | 106.798 | 104.861 | 0.701 | 53 |
| HARD | 107.124 | 105.257 | 1.086 | 69 |
| MEDIUM | 107.803 | 105.706 | 1.204 | 485 |
| INTERMEDIATE | 123.049 | 119.092 | 1.698 | 140 |

> Belgium yarışında UNKNOWN lastikler en hızlı ortalama süreyi kaydetmiştir (1:46.798).

## 📉 Lastik Bozulma Analizi
**En Hızlı Bozulan Lastik:** INTERMEDIATE

| Lastik | Bozulma Oranı (s/tur) | R² |
|--------|----------------------|-----|
| INTERMEDIATE | 0.3076 | 0.417 |
| MEDIUM | -0.0646 | 0.630 |
| HARD | -0.0673 | 0.451 |
| UNKNOWN | N/A | N/A |

> Belgium yarışında en hızlı bozulan lastik INTERMEDIATE olmuştur (0.308 s/tur artış).

## 🔧 Pit Stop Analizi
**Ortalama Pit Stop Faydası:** 12.371s

| Pilot | Pit Sayısı | Ortalama Etki (s) |
|-------|------------|-------------------|
| STR | 2 | 16.424 |
| LAW | 2 | 15.412 |
| BOR | 2 | 15.205 |
| RUS | 2 | 15.150 |
| ALB | 2 | 15.049 |
| LEC | 2 | 15.004 |
| VER | 2 | 14.881 |
| OCO | 2 | 14.875 |
| TSU | 2 | 14.845 |
| GAS | 2 | 14.062 |
| HAM | 2 | 13.515 |
| NOR | 2 | 13.396 |
| PIA | 2 | 12.823 |
| BEA | 2 | 12.301 |
| SAI | 4 | 8.163 |
| ANT | 4 | 8.127 |
| COL | 4 | 7.649 |
| HUL | 4 | 7.078 |
| HAD | 4 | 7.006 |
| ALO | 4 | 6.452 |

> Belgium yarışında ortalama pit stop faydası 12.37s olarak ölçülmüştür. Pit stop sonrası hız artışı gözlemlenmiştir (undercut avantajı).

## 🌡️ Hava ve Sıcaklık Analizi
**Hava Sıcaklığı**
- Pearson r = nan
- p-değeri = nan (❌ Anlamlı değil)
- Çok güçlü ilişki
> Belgium: Hava Sıcaklığı artarken tur zamanı azalıyor (r=nan, p=>0.05). Çok güçlü ilişki.

**Pist Sıcaklığı**
- Pearson r = nan
- p-değeri = nan (❌ Anlamlı değil)
- Çok güçlü ilişki
> Belgium: Pist Sıcaklığı artarken tur zamanı azalıyor (r=nan, p=>0.05). Çok güçlü ilişki.

## 🤖 Makine Öğrenmesi Sonuçları
### K-Means Clustering
**Küme Sayısı:** 3
- **Agresif:** BEA, HAM
- **Stabil & Hızlı:** ALB, GAS, LAW, LEC, NOR, PIA, RUS, VER
- **Düzensiz:** ALO, ANT, BOR, COL, HAD, HUL, OCO, SAI, STR, TSU

> Belgium yarışında 3 performans grubu tespit edilmiştir. 'Stabil & Hızlı' grubundaki pilotlar: ALB, GAS, LAW, LEC, NOR, PIA, RUS, VER.

### Decision Tree Sınıflandırma
**Doğruluk:** %100.0

**Feature Importance:**
- `average_lap_time`: 1.0000 ████████████████████
- `tire_degradation_rate`: 0.0000 
- `consistency_score`: 0.0000 
- `pit_stop_impact`: 0.0000 
- `sector_consistency`: 0.0000 
- `AirTemp`: 0.0000 
- `TrackTemp`: 0.0000 

**En Önemli Özellik:** `average_lap_time`

> Decision Tree modeli Belgium yarışında %100.0 doğruluk oranı elde etti. Yarış performansını en çok etkileyen faktör: average_lap_time.

### kNN Sınıflandırma
**Doğruluk:** %100.0 (k=5)
> kNN modeli (k=5) Belgium yarışında %100.0 doğruluk oranı elde etti.

### Model Karşılaştırması
| Model | Doğruluk |
|-------|----------|
| Decision Tree | %100.0 |
| kNN | %100.0 |
| **Kazanan** | **Decision Tree** |

> Belgium: Decision Tree doğruluk oranı %100.0, kNN doğruluk oranı %100.0. Bu yarışta daha başarılı model: Decision Tree.

---
## 📝 Genel Sonuç
Belgium Grand Prix analizi tamamlandı. Yarışın en hızlı pilotu **ANT** olurken, en tutarlı performansı **PIA** sergilemiştir. Lastik stratejisi açısından **UNKNOWN** bileşiği bu pistte en avantajlı seçenek olarak öne çıkmıştır. Makine öğrenmesi modeli, yarış performansını etkileyen en kritik faktör olarak **average_lap_time** özelliğini belirlemiştir. Sınıflandırma modelleri karşılaştırmasında **Decision Tree** daha yüksek doğruluk oranına ulaşmıştır.

---
*Bu rapor F1 Race Intelligence System tarafından otomatik üretilmiştir.*
*Veri kaynağı: FastF1 / 2025 Formula 1 Sezonu*