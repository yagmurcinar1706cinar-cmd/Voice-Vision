import cv2
import threading
from collections import deque, defaultdict
import time
import edge_tts
import asyncio
import pygame
import tempfile
import os
import torch
from ultralytics import YOLO

# ─── AYARLAR ──────────────────────────────────────────────────
TURKCE_SES        = "tr-TR-AhmetNeural"
CONFIDENCE_ESIGI  = 0.55
TEKRAR_SURESI     = 6.0
ONAY_TAMPON_BOY   = 5
ONAY_MIN_GORUNUM  = 3
FRAME_ATLAMA      = 2
MAX_KUYRUK        = 5

# ─── APPLE SILICON MPS HIZLANDIRMA ────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✅ Apple Silicon MPS (GPU) aktif!")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("✅ CUDA GPU aktif!")
else:
    DEVICE = "cpu"
    print("⚠️  CPU kullanılıyor.")

# ─── ÖNCELİK SINIFLANDIRMASI ──────────────────────────────────
ONCELIK_YUKSEK = {"person", "car", "motorcycle", "bicycle", "bus", "dog"}
ONCELIK_DUSUK  = {"chair", "couch", "bottle", "cup", "book"}

# ─── NESNE SÖZLÜĞÜ ────────────────────────────────────────────
nesne_sozluk = {
    "person":       "kişi",
    "bicycle":      "bisiklet",
    "car":          "araba",
    "motorcycle":   "motosiklet",
    "bus":          "otobüs",
    "chair":        "sandalye",
    "couch":        "kanepe",
    "bed":          "yatak",
    "dining table": "masa",
    "toilet":       "tuvalet",
    "tv":           "televizyon",
    "laptop":       "dizüstü bilgisayar",
    "mouse":        "fare",
    "keyboard":     "klavye",
    "cell phone":   "telefon",
    "bottle":       "şişe",
    "cup":          "bardak",
    "fork":         "çatal",
    "knife":        "bıçak",
    "spoon":        "kaşık",
    "bowl":         "kase",
    "book":         "kitap",
    "clock":        "saat",
    "vase":         "vazo",
    "scissors":     "makas",
    "backpack":     "sırt çantası",
    "handbag":      "el çantası",
    "umbrella":     "şemsiye",
    "suitcase":     "bavul",
    "cat":          "kedi",
    "dog":          "köpek",
    "bird":         "kuş",
}

# ─── KONUM HESAPLAMA ──────────────────────────────────────────
def konum_hesapla(x1, y1, x2, y2, frame_w, frame_h):
    merkez_x    = (x1 + x2) / 2
    nesne_alani = (x2 - x1) * (y2 - y1)
    frame_alani = frame_w * frame_h

    if merkez_x < frame_w * 0.35:
        yatay = "solunuzda"
    elif merkez_x > frame_w * 0.65:
        yatay = "sağınızda"
    else:
        yatay = "önünüzde"

    oran = nesne_alani / frame_alani
    if oran > 0.25:
        yakinlik = "çok yakın"
    elif oran > 0.08:
        yakinlik = "yakın"
    else:
        yakinlik = None

    return yatay, yakinlik

def mesaj_olustur(tr_isim, yatay, yakinlik):
    if yakinlik:
        return f"{yatay} {yakinlik} bir {tr_isim} var"
    return f"{yatay} bir {tr_isim} var"

# ─── ONAY TAMPONU ─────────────────────────────────────────────
nesne_tamponu = defaultdict(lambda: deque(maxlen=ONAY_TAMPON_BOY))

def tampon_guncelle(anahtar, tespit_edildi: bool) -> bool:
    nesne_tamponu[anahtar].append(1 if tespit_edildi else 0)
    return sum(nesne_tamponu[anahtar]) >= ONAY_MIN_GORUNUM

# ─── TRACKER ──────────────────────────────────────────────────
track_gecmis        = {}
KONUM_DEGISIM_ESIGI = 0.15

def konum_normalize(x1, y1, x2, y2, fw, fh):
    return ((x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh)

def konum_degisti_mi(track_id, yeni_nx, yeni_ny):
    if track_id not in track_gecmis:
        return True
    konum_degeri = track_gecmis[track_id].get("konum")
    if konum_degeri is None:
        return True
    eski_nx, eski_ny = konum_degeri
    mesafe = ((yeni_nx - eski_nx)**2 + (yeni_ny - eski_ny)**2) ** 0.5
    return mesafe > KONUM_DEGISIM_ESIGI

# ─── KONUŞMA SİSTEMİ ──────────────────────────────────────────
konusma_kuyrugu   = deque()
konusuyor         = False
program_calisiyor = True
kuyruk_seti       = set()

def konusma_yoneticisi():
    global konusuyor, program_calisiyor
    pygame.mixer.init()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print(f"[TTS] Edge TTS başlatıldı → {TURKCE_SES}")

    while program_calisiyor:
        if konusma_kuyrugu:
            konusuyor = True
            mesaj = konusma_kuyrugu.popleft()
            kuyruk_seti.discard(mesaj)
            print(f"[SES] {mesaj}")
            try:
                async def _kaydet():
                    kom = edge_tts.Communicate(mesaj, TURKCE_SES)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                        tmp = f.name
                    await kom.save(tmp)
                    return tmp

                tmp_path = loop.run_until_complete(_kaydet())
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                os.unlink(tmp_path)
            except Exception as e:
                print(f"[SES HATA] {e}")
            konusuyor = False
            time.sleep(0.1)
        else:
            time.sleep(0.05)

ses_thread = threading.Thread(target=konusma_yoneticisi, daemon=True)
ses_thread.start()

# ─── KUYRUGA EKLE ─────────────────────────────────────────────
def kuyruga_ekle(mesaj, oncelikli=False):
    if oncelikli:
        konusma_kuyrugu.clear()
        kuyruk_seti.clear()
        konusma_kuyrugu.appendleft(mesaj)
        kuyruk_seti.add(mesaj)
    elif mesaj not in kuyruk_seti and len(konusma_kuyrugu) < MAX_KUYRUK:
        konusma_kuyrugu.append(mesaj)
        kuyruk_seti.add(mesaj)

# ─── MODEL ────────────────────────────────────────────────────
print("YOLOv8 yükleniyor...")
model = YOLO("yolov8s.pt")
model.to(DEVICE)
print("Hazır!\n")

# ─── KAMERA ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_sayaci = 0
son_sonuclar = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_sayaci  += 1
    simdiki_zaman  = time.time()
    frame_h, frame_w = frame.shape[:2]

    if frame_sayaci % FRAME_ATLAMA == 0:
        results      = model.track(frame, verbose=False,
                                   conf=CONFIDENCE_ESIGI,
                                   tracker="botsort.yaml",
                                   persist=True,
                                   device=DEVICE)
        son_sonuclar = results
    else:
        results = son_sonuclar

    aktif_track_idler = set()
    tum_kutular = []

    for result in results:
        if result.boxes.id is None:
            continue
        for box, track_id in zip(result.boxes, result.boxes.id.int().tolist()):
            cls_id = int(box.cls[0])
            eng    = model.names[cls_id]
            conf   = float(box.conf[0])
            if eng in nesne_sozluk:
                oncelik = 0 if eng in ONCELIK_YUKSEK else 1
                tum_kutular.append((oncelik, eng, conf, box, track_id))

    tum_kutular.sort(key=lambda x: x[0])

    for oncelik, eng, conf, box, track_id in tum_kutular:
        aktif_track_idler.add(track_id)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tr = nesne_sozluk[eng]

        renk_kutu = (0, 100, 255) if eng in ONCELIK_YUKSEK else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), renk_kutu, 2)
        cv2.putText(frame, f"[{track_id}] {tr} {conf*100:.0f}%",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk_kutu, 2)

        if not tampon_guncelle(track_id, True):
            continue

        if track_id not in track_gecmis:
            track_gecmis[track_id] = {"eng": eng, "son_ses": 0, "konum": None}

        sure  = TEKRAR_SURESI * 0.5 if eng in ONCELIK_YUKSEK else TEKRAR_SURESI
        gecen = simdiki_zaman - track_gecmis[track_id]["son_ses"]

        nx, ny     = konum_normalize(x1, y1, x2, y2, frame_w, frame_h)
        konum_yeni = konum_degisti_mi(track_id, nx, ny)

        if gecen < sure and not konum_yeni:
            continue

        yatay, yakinlik = konum_hesapla(x1, y1, x2, y2, frame_w, frame_h)
        mesaj_tam = mesaj_olustur(tr, yatay, yakinlik)

        if yatay == "önünüzde":
            # 1. En yüksek öncelik → önde olan her şey
            kuyruga_ekle(mesaj_tam, oncelikli=True)
        elif eng in ONCELIK_YUKSEK:
            # 2. Yüksek öncelikli → önündekinin hemen arkasına
            once_var = any("önünüzde" in k for k in list(konusma_kuyrugu))
            if once_var:
                ilk_indeks = next(
                    (i for i, m in enumerate(konusma_kuyrugu) if "önünüzde" in m), 0
                )
                konusma_kuyrugu.insert(ilk_indeks + 1, mesaj_tam)
                kuyruk_seti.add(mesaj_tam)
            else:
                kuyruga_ekle(mesaj_tam, oncelikli=True)
        else:
            # 3. Düşük öncelikli → sıraya gir
            kuyruga_ekle(mesaj_tam)

        track_gecmis[track_id]["son_ses"] = simdiki_zaman
        track_gecmis[track_id]["konum"]   = (nx, ny)
        print(f"[NESNE] #{track_id} {eng} → '{mesaj_tam}' ({conf*100:.0f}%)")

    for tid in list(nesne_tamponu.keys()):
        if tid not in aktif_track_idler:
            tampon_guncelle(tid, False)

    if frame_sayaci % 300 == 0:
        eski_tidler = [tid for tid in track_gecmis
                       if tid not in aktif_track_idler and
                       simdiki_zaman - track_gecmis[tid]["son_ses"] > 30]
        for tid in eski_tidler:
            del track_gecmis[tid]

    durum = "KONUŞUYOR" if konusuyor else f"DİNLİYOR | {len(aktif_track_idler)} nesne | {DEVICE.upper()}"
    renk  = (0, 140, 255) if konusuyor else (0, 220, 0)
    cv2.putText(frame, durum, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)
    cv2.putText(frame, f"Kuyruk: {len(konusma_kuyrugu)}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Görme Engelli Yardımcı v2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

program_calisiyor = False
time.sleep(0.5)
cap.release()
cv2.destroyAllWindows()