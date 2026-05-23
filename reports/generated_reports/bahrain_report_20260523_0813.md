# 🏎️ F1 Race Intelligence Report
## Bahrain Grand Prix — 2025 Sezonu
*Rapor tarihi: 23 May 2026, 08:13*

---
## 📊 Yarış Genel Bakış
| Metrik | Değer |
|--------|-------|
| Toplam Tur | 200 |
| Toplam Pilot | 10 |
| En Hızlı Tur | 1:28.667 |
| En Hızlı Pilot | HAM |
| En İstikrarlı Pilot | PIA |
| Ortalama Hava Sıcaklığı | 27.8°C |
| Ortalama Pist Sıcaklığı | 41.9°C |

## 🏁 Pilot İstikrar Analizi
**En İstikrarlı Pilot:** PIA
**En Hızlı Pilot:** HAM

| Sıra | Pilot | Ort. Tur | En Hızlı | Std Sapma | Consistency |
|------|-------|----------|----------|-----------|-------------|
| 1 | PIA | 90.785s | 89.842s | 0.463s | 0.6837 |
| 2 | ALO | 90.546s | 89.340s | 0.608s | 0.6219 |
| 3 | LEC | 90.674s | 89.924s | 0.633s | 0.6123 |
| 4 | VER | 90.586s | 89.376s | 0.638s | 0.6105 |
| 5 | NOR | 90.508s | 88.845s | 0.687s | 0.5928 |
| 6 | OCO | 90.774s | 89.789s | 0.706s | 0.5861 |
| 7 | STR | 90.551s | 89.327s | 0.720s | 0.5816 |
| 8 | HAM | 90.294s | 88.667s | 0.755s | 0.5698 |
| 9 | SAI | 90.447s | 89.473s | 0.797s | 0.5564 |
| 10 | RUS | 90.462s | 88.876s | 0.848s | 0.5413 |

> Bahrain yarışında en stabil pilot PIA olmuştur (Consistency Score: 0.684). En hızlı tur HAM adına kayıtlıdır (1:28.667).

## 🔴 Lastik Stratejisi Analizi
**En Avantajlı Lastik:** SOFT

| Lastik | Ort. Tur (s) | En Hızlı (s) | Std Sapma | Tur Sayısı |
|--------|--------------|--------------|-----------|------------|
| SOFT | 89.949 | 88.667 | 0.592 | 59 |
| MEDIUM | 90.631 | 89.473 | 0.513 | 87 |
| HARD | 91.088 | 89.953 | 0.548 | 54 |

> Bahrain yarışında SOFT lastikler en hızlı ortalama süreyi kaydetmiştir (1:29.949).

## 📉 Lastik Bozulma Analizi
**En Hızlı Bozulan Lastik:** SOFT

| Lastik | Bozulma Oranı (s/tur) | R² |
|--------|----------------------|-----|
| MEDIUM | 0.0520 | 0.835 |
| SOFT | 0.0578 | 0.660 |
| HARD | 0.0458 | 0.469 |

> Bahrain yarışında en hızlı bozulan lastik SOFT olmuştur (0.058 s/tur artış).

## 🔧 Pit Stop Analizi
> Pit stop analizi için yeterli veri yok.

## 🌡️ Hava ve Sıcaklık Analizi
**Hava Sıcaklığı**
- Pearson r = -0.0139
- p-değeri = 0.8454 (❌ Anlamlı değil)
- İhmal edilebilir ilişki
> Bahrain: Hava Sıcaklığı artarken tur zamanı azalıyor (r=-0.014, p=>0.05). İhmal edilebilir ilişki.

**Pist Sıcaklığı**
- Pearson r = 0.0745
- p-değeri = 0.2945 (❌ Anlamlı değil)
- İhmal edilebilir ilişki
> Bahrain: Pist Sıcaklığı artarken tur zamanı da artıyor (r=0.074, p=>0.05). İhmal edilebilir ilişki.

## 🤖 Makine Öğrenmesi Sonuçları
### K-Means Clustering
**Küme Sayısı:** 3
- **Stabil & Hızlı:** HAM, RUS, SAI
- **Düzensiz:** PIA
- **Agresif:** ALO, LEC, NOR, OCO, STR, VER

> Bahrain yarışında 3 performans grubu tespit edilmiştir. 'Stabil & Hızlı' grubundaki pilotlar: HAM, RUS, SAI.

### Decision Tree Sınıflandırma
**Doğruluk:** %66.7

**Feature Importance:**
- `AirTemp`: 0.5833 ███████████
- `TrackTemp`: 0.4167 ████████
- `average_lap_time`: 0.0000 
- `tire_degradation_rate`: 0.0000 
- `consistency_score`: 0.0000 
- `pit_stop_impact`: 0.0000 
- `sector_consistency`: 0.0000 

**En Önemli Özellik:** `AirTemp`

> Decision Tree modeli Bahrain yarışında %66.7 doğruluk oranı elde etti. Yarış performansını en çok etkileyen faktör: AirTemp.

### kNN Sınıflandırma
**Doğruluk:** %66.7 (k=5)
> kNN modeli (k=5) Bahrain yarışında %66.7 doğruluk oranı elde etti.

### Model Karşılaştırması
| Model | Doğruluk |
|-------|----------|
| Decision Tree | %66.7 |
| kNN | %66.7 |
| **Kazanan** | **Decision Tree** |

> Bahrain: Decision Tree doğruluk oranı %66.7, kNN doğruluk oranı %66.7. Bu yarışta daha başarılı model: Decision Tree.

---
## 📝 Genel Sonuç
Bahrain Grand Prix analizi tamamlandı. Yarışın en hızlı pilotu **HAM** olurken, en tutarlı performansı **PIA** sergilemiştir. Lastik stratejisi açısından **SOFT** bileşiği bu pistte en avantajlı seçenek olarak öne çıkmıştır. Makine öğrenmesi modeli, yarış performansını etkileyen en kritik faktör olarak **AirTemp** özelliğini belirlemiştir. Sınıflandırma modelleri karşılaştırmasında **Decision Tree** daha yüksek doğruluk oranına ulaşmıştır.

---
*Bu rapor F1 Race Intelligence System tarafından otomatik üretilmiştir.*
*Veri kaynağı: FastF1 / 2025 Formula 1 Sezonu*