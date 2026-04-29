# 👁️ Voice-Vision: AI-Powered Object Recognition & Spatial Audio

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-green.svg)
![Device](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS-orange.svg)

Bu proje, görme engelli bireylerin çevrelerini daha iyi algılayabilmeleri için tasarlanmış, gerçek zamanlı nesne tespiti ve sesli konumlandırma asistanıdır. Yazılım, kameradan gelen görüntüyü analiz eder, nesneleri tanımlar ve kullanıcının konumuna göre (Sağ, Sol, Ön) sesli geri bildirim sağlar.

## 🚀 Öne Çıkan Özellikler
* **Gerçek Zamanlı Tespit:** YOLOv8 (Small) modeli ile 80 farklı nesne sınıfını anlık olarak tanır.
* **Konumsal Seslendirme:** Nesnenin ekrandaki koordinatlarını hesaplayarak kullanıcının hangi yönünde olduğunu belirtir.
* **Apple Silicon Optimizasyonu:** Mac M serisi işlemciler için `MPS` (Metal Performance Shaders) hızlandırmasını kullanarak yüksek FPS değerlerine ulaşır.
* **Doğal Sesli Geri Bildirim:** `Edge-TTS` (Microsoft Azure tabanlı) kütüphanesi ile gerçekçi ve anlaşılır seslendirme yapar.

## 🛠️ Teknik Mimari
Proje üç ana katmandan oluşmaktadır:
1.  **Görüntü İşleme:** OpenCV ile kameradan alınan kareler YOLOv8 modeline aktarılır.
2.  **Mantık Katmanı:** Nesnelerin merkez koordinatları (x, y) hesaplanarak ekran 3 bölgeye (Sol, Orta, Sağ) ayrılır.
3.  **Çıktı Katmanı:** Tespit edilen ve önceliği olan nesneler `pygame` aracılığıyla ses dosyasına dönüştürülüp oynatılır.

## 📦 Gereksinimler
Projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekir:
```text
ultralytics
opencv-python
edge-tts
pygame
torch
torchvision
