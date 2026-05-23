# 🏎️ F1 Race Intelligence Report
## Belgium Grand Prix — 2025 Sezonu
*Rapor tarihi: 23 May 2026, 08:31*

---
## 📊 Yarış Genel Bakış
| Metrik | Değer |
|--------|-------|
| Toplam Tur | 880 |
| Toplam Pilot | 20 |
| En Hızlı Tur | 1:43.229 |
| En Hızlı Pilot | PIA |
| En İstikrarlı Pilot | HUL |
| Ortalama Hava Sıcaklığı | 18.1°C |
| Ortalama Pist Sıcaklığı | 27.9°C |

## 🏁 Pilot İstikrar Analizi
**En İstikrarlı Pilot:** HUL
**En Hızlı Pilot:** PIA

| Sıra | Pilot | Ort. Tur | En Hızlı | Std Sapma | Consistency |
|------|-------|----------|----------|-----------|-------------|
| 1 | HUL | 106.801s | 105.649s | 0.665s | 0.6006 |
| 2 | ALO | 105.539s | 103.592s | 0.670s | 0.5990 |
| 3 | BOR | 105.979s | 104.673s | 0.672s | 0.5980 |
| 4 | HAD | 106.748s | 105.449s | 0.703s | 0.5870 |
| 5 | VER | 105.413s | 103.439s | 0.706s | 0.5863 |
| 6 | LEC | 105.542s | 104.209s | 0.709s | 0.5853 |
| 7 | OCO | 106.171s | 104.686s | 0.716s | 0.5827 |
| 8 | DOO | 106.785s | 105.343s | 0.720s | 0.5814 |
| 9 | ALB | 106.502s | 104.835s | 0.734s | 0.5765 |
| 10 | ANT | 106.365s | 105.109s | 0.750s | 0.5714 |

> Belgium yarışında en stabil pilot HUL olmuştur (Consistency Score: 0.601). En hızlı tur PIA adına kayıtlıdır (1:43.229).

## 🔴 Lastik Stratejisi Analizi
**En Avantajlı Lastik:** SOFT

| Lastik | Ort. Tur (s) | En Hızlı (s) | Std Sapma | Tur Sayısı |
|--------|--------------|--------------|-----------|------------|
| SOFT | 105.190 | 103.229 | 0.721 | 230 |
| MEDIUM | 106.042 | 104.285 | 0.725 | 285 |
| HARD | 106.679 | 104.608 | 0.749 | 304 |

> Belgium yarışında SOFT lastikler en hızlı ortalama süreyi kaydetmiştir (1:45.190).

## 📉 Lastik Bozulma Analizi
**En Hızlı Bozulan Lastik:** SOFT

| Lastik | Bozulma Oranı (s/tur) | R² |
|--------|----------------------|-----|
| SOFT | 0.0842 | 0.817 |
| MEDIUM | 0.0594 | 0.787 |
| HARD | 0.0623 | 0.897 |

> Belgium yarışında en hızlı bozulan lastik SOFT olmuştur (0.084 s/tur artış).

## 🔧 Pit Stop Analizi
**Ortalama Pit Stop Faydası:** -0.157s

| Pilot | Pit Sayısı | Ortalama Etki (s) |
|-------|------------|-------------------|
| ANT | 2 | 0.220 |
| BOR | 2 | 0.128 |
| HAM | 2 | 0.076 |
| TSU | 2 | 0.065 |
| OCO | 2 | 0.050 |
| HUL | 2 | 0.034 |
| ALO | 2 | -0.015 |
| DOO | 2 | -0.086 |
| NOR | 2 | -0.113 |
| LEC | 2 | -0.166 |
| LAW | 2 | -0.182 |
| HAD | 2 | -0.187 |
| SAI | 2 | -0.189 |
| ALB | 2 | -0.234 |
| STR | 2 | -0.308 |
| BEA | 2 | -0.352 |
| GAS | 2 | -0.369 |
| VER | 2 | -0.389 |
| RUS | 2 | -0.493 |
| PIA | 2 | -0.624 |

> Belgium yarışında ortalama pit stop faydası -0.16s olarak ölçülmüştür. Pit stop sonrası beklenen hız artışı sınırlı kalmıştır.

## 🌡️ Hava ve Sıcaklık Analizi
**Hava Sıcaklığı**
- Pearson r = -0.0616
- p-değeri = 0.078 (❌ Anlamlı değil)
- İhmal edilebilir ilişki
> Belgium: Hava Sıcaklığı artarken tur zamanı azalıyor (r=-0.062, p=>0.05). İhmal edilebilir ilişki.

**Pist Sıcaklığı**
- Pearson r = -0.0211
- p-değeri = 0.5464 (❌ Anlamlı değil)
- İhmal edilebilir ilişki
> Belgium: Pist Sıcaklığı artarken tur zamanı azalıyor (r=-0.021, p=>0.05). İhmal edilebilir ilişki.

## 🤖 Makine Öğrenmesi Sonuçları
### K-Means Clustering
**Küme Sayısı:** 3
- **Agresif:** ALO, BOR, HUL
- **Düzensiz:** ALB, ANT, BEA, DOO, HAD, LAW, LEC, OCO, STR, TSU, VER
- **Stabil & Hızlı:** GAS, HAM, NOR, PIA, RUS, SAI

> Belgium yarışında 3 performans grubu tespit edilmiştir. 'Stabil & Hızlı' grubundaki pilotlar: GAS, HAM, NOR, PIA, RUS, SAI.

### Decision Tree Sınıflandırma
**Doğruluk:** %50.0

**Feature Importance:**
- `consistency_score`: 0.6818 █████████████
- `AirTemp`: 0.3182 ██████
- `average_lap_time`: 0.0000 
- `tire_degradation_rate`: 0.0000 
- `pit_stop_impact`: 0.0000 
- `sector_consistency`: 0.0000 
- `TrackTemp`: 0.0000 

**En Önemli Özellik:** `consistency_score`

> Decision Tree modeli Belgium yarışında %50.0 doğruluk oranı elde etti. Yarış performansını en çok etkileyen faktör: consistency_score.

### kNN Sınıflandırma
**Doğruluk:** %83.3 (k=5)
> kNN modeli (k=5) Belgium yarışında %83.3 doğruluk oranı elde etti.

### Model Karşılaştırması
| Model | Doğruluk |
|-------|----------|
| Decision Tree | %50.0 |
| kNN | %83.3 |
| **Kazanan** | **kNN** |

> Belgium: Decision Tree doğruluk oranı %50.0, kNN doğruluk oranı %83.3. Bu yarışta daha başarılı model: kNN.

---
## 📝 Genel Sonuç
Belgium Grand Prix analizi tamamlandı. Yarışın en hızlı pilotu **PIA** olurken, en tutarlı performansı **HUL** sergilemiştir. Lastik stratejisi açısından **SOFT** bileşiği bu pistte en avantajlı seçenek olarak öne çıkmıştır. Makine öğrenmesi modeli, yarış performansını etkileyen en kritik faktör olarak **consistency_score** özelliğini belirlemiştir. Sınıflandırma modelleri karşılaştırmasında **kNN** daha yüksek doğruluk oranına ulaşmıştır.

---
*Bu rapor F1 Race Intelligence System tarafından otomatik üretilmiştir.*
*Veri kaynağı: FastF1 / 2025 Formula 1 Sezonu*