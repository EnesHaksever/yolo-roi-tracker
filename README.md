# ROITracker — Dinamik Pencere Kilidi (YOLO `model.track()` drop-in)

`roi_tracker.py`, hareketli kameradan **tek bir hedefi** (örn. rakip İHA) kararlı
takip etmek için yazılmış, YOLO'nun `model.track()` fonksiyonunun **yerine geçen**
tek sınıflık bir modüldür.

**Mantığı:** Tüm kareyi her frame yüksek çözünürlükte taramak yerine, hedefin
etrafında küçük ve dinamik bir **ROI penceresi** açar. Böylece resize kaynaklı
piksel kaybı ve hesap yükü düşer. İki mod vardır:

- **FULL:** Kilit yokken tüm kareyi tarar (yüksek `imgsz`, seyrek).
- **ROI:** Bir hedefe kilitlenince küçük pencereyle onu takip eder (her frame).

Her iki moddan gelen kutular, **gömülü ve paylaşılan bir ByteTrack**'ten geçer;
ID'ler tutarlı kalır. Çıktı **birebir YOLO `Results` formatındadır**, yani
YOLO çıktısına göre yazılmış kodlar değişmeden çalışır.

---

## Kurulum

```bash
pip install ultralytics opencv-python numpy
```
`torch` ve `ultralytics` zaten kuruluysa ek bir şey gerekmez. Sadece
`roi_tracker.py` dosyasını projene kopyalaman yeterli.

---

## Hızlı Başlangıç (YOLO gibi)

```python
from roi_tracker import ROITracker

# YOLO: model = YOLO("model.pt")
trk = ROITracker("C:/.../yolo11m.pt")

# YOLO: for r in model.track("video.mov", stream=True, show=True): ...
for r in trk.roi_track("video.mov", stream=True, show=True):
    # r ---> ultralytics Results (YOLO ile AYNI)
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        tid  = int(b.id[0]) if b.id is not None else None
        conf = float(b.conf[0])
        cls  = int(b.cls[0])
```

Gerçek zamanlı kamera için sadece kaynağı değiştir:

```python
trk.roi_track(0, show=True)     # 0/1/2 = kamera indeksi
```

Kendi döngünde kare kare beslemek istersen (YOLO `model(frame)` gibi):

```python
res = trk.roi_track(frame)[0]   # tek kare -> tek Results
```

---

## Çıktı Formatı (YOLO `Results`)

`roi_track` her frame için bir ultralytics **`Results`** nesnesi verir. Erişim
YOLO ile **birebir aynıdır**:

| Alan | Açıklama |
|---|---|
| `res.boxes.xyxy` | Kutu köşe koordinatları `[x1,y1,x2,y2]` (global, tam kare) |
| `res.boxes.id` | Track ID'si (yoksa `None`) |
| `res.boxes.conf` | Güven (YOLO conf) |
| `res.boxes.cls` | Sınıf id |
| `res.plot()` / `res.save()` / `res.show()` | YOLO'daki gibi çalışır |

> Sistem **tek hedef** kilitlediği için, kilitliyken `res.boxes` genelde **tek
> kutu** (kilitli hedef) içerir. Tüm trackleri istersen `all_tracks=True` ver.

**Ekstra (opsiyonel) alanlar** — YOLO tüketicilerini bozmaz, isteyen kullanır:

| Alan | Açıklama |
|---|---|
| `res.roi_mode` | `"FULL"` veya `"ROI"` |
| `res.roi_score` | Güven skoru (**saniye**) |
| `res.roi_locked_id` | O an kilitli ID (yoksa `None`) |
| `res.roi_window` | ROI penceresi `(x, y, w, h)` |

---

## 1) Temel Parametreler (çoğu entegrasyon için yeterli)

### `roi_track(...)` fonksiyon seçenekleri (çalışma anında)

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `source` | — | Kaynak: video yolu (`"v.mov"`), **kamera indeksi** (`0`,`1`,`2`), veya tek kare (`np.ndarray`). |
| `show` | `False` | `True` → sonucu pencerede gösterir (ESC ile çıkış). |
| `save` | `False` | `True` → çizimli videoyu `save_path`'e kaydeder. |
| `save_path` | `"roi_output.mp4"` | Kayıt dosyası. |
| `stream` | `False` | `True` → her frame için Results üreten **generator** (uzun/canlı akış için önerilir). `False` → tüm sonuçları liste olarak toplar. |
| `draw_roi` | `True` | `False` → **mavi ROI penceresi** ve kilitli hedef dışındaki **gri kutular** çizilmez; sadece yeşil kilitli hedef çizilir. (Yalnız `show`/`save` için geçerli.) |
| `all_tracks` | `False` | `True` → `res.boxes` penceredeki **tüm** trackleri içerir; `False` → sadece kilitli hedef. |

### Kurulumda (konstrüktör) sık kullanılanlar

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `model_full` | — | Full-frame taraması için YOLO modeli (`.pt` yolu **veya** yüklü `YOLO` nesnesi). |
| `model_roi` | `None` | ROI için ayrı model. `None` → `model_full` ile aynı örnek kullanılır. (Aşağıda ayrıntı var.) |
| `target_class` | `2` | Takip edilecek sınıf id'si (kendi modelindeki hedef sınıf). |
| `conf_thres` | `0.05` | **Tespit/besleme eşiği (DÜŞÜK).** Bu üstündeki her kutu takip için değerlendirilir. Düşük tutulur; ByteTrack conf düşse bile track'i canlı tutar. |
| `lock_conf` | `0.6` | **Kilitlenme eşiği (YÜKSEK).** Yeni bir hedefe *ilk kilitlenmek* için conf bu üstünde olmalı. Çöpe/belirsize kilitlenmeyi önler. |
| `full_imgsz` | `1920` | FULL modda YOLO `imgsz` (aşağıda ayrıntı). |
| `roi_imgsz` | `384` | ROI modda YOLO `imgsz` (aşağıda ayrıntı). |
| `device` | `0` | YOLO cihazı (`0` = `cuda:0`, `"cpu"` de olur). |
| `mask` | `False` | `True` → **OSD/telemetri yazıları** (DISARMED, ALT, SPD, voltaj vb.) tespitten önce gri ile kapatılır; model bu yazıları İHA sanmaz. Çıktı/kayıt karesi bozulmaz — maske sadece modelin gördüğü kopyaya uygulanır. |
| `mask_regions` | `None` | Maskelenecek bölgeler, **normalize** `(x1, y1, x2, y2)` listesi (0–1 oran; her çözünürlükte çalışır). `None` → tipik FPV OSD yerleşimine göre 5 hazır bölge (sol-üst sinyal/LQ, sağ-üst fly/ALT/SPD, orta DISARMED/TRIGGER, sağ-orta Mode, sol-alt akım/voltaj). |

---

## `full_imgsz` ve `roi_imgsz` — neden ikisi ayrı?

`imgsz`, YOLO'nun görüntüyü içeride yeniden boyutlandırdığı kare boyutudur.

- **`full_imgsz` (varsayılan 1920):** FULL modda **tüm kare** taranır. Hedef
  uzak/küçük olabileceği için yüksek tutulur ki küçük nesneler kaçmasın. Ağırdır,
  o yüzden FULL seyrek çalışır (video'da her `full_interval` frame'de bir).
- **`roi_imgsz` (varsayılan 384):** ROI modda sadece **küçük pencere** (örn.
  384×384) işlenir. Pencere zaten hedefe odaklı ve küçük olduğu için düşük `imgsz`
  yeterlidir; çok hızlıdır. Buradaki kazanç işin özü: hedef, resize'dan neredeyse
  hiç piksel kaybetmeden modele girer.

**Jetson / düşük güç ipucu:** Takılma olursa **`full_imgsz`'i düşür** (örn.
1920 → 1280) — asıl yük oradadır. `roi_imgsz` zaten hafiftir.

---

## ROI ve FULL için AYRI model yükleme

İki mod için farklı modeller verebilirsin. Tipik kullanım: FULL için küçük
nesneleri iyi bulan/yüksek çözünürlük modeli, ROI için hız odaklı hafif model —
ya da tam tersi.

```python
from ultralytics import YOLO

# Yol vererek (sınıf içeride yükler):
trk = ROITracker(
    model_full="C:/.../full_model.pt",
    model_roi="C:/.../roi_model.pt",
)

# Ya da önceden yüklenmiş YOLO nesneleri vererek:
m_full = YOLO("C:/.../full_model.pt")
m_roi  = YOLO("C:/.../roi_model.pt")
trk = ROITracker(m_full, m_roi)

# Tek model her iki mod için (varsayılan):
trk = ROITracker("C:/.../model.pt")   # model_roi=None -> aynı model
```

> Not: `model_full` ve `model_roi` **aynı sınıfları** üretmeli (aynı
> `target_class`). Farklı ağırlıklar olabilir ama hedef sınıf id'si tutmalı.

---

## 2) İleri Parametreler (davranış ince ayarı)

Bu grubu, varsayılanlar durumuna uymazsa değiştir. Hepsi **konstrüktörde**
verilir (takip davranışını belirledikleri ve frame'ler arası durum taşıdıkları
için — bu yüzden çalışma ortasında değil, kurulumda sabitlenirler).

### Güven skoru (SANİYE tabanlı — değişken FPS'te doğru çalışır)

Kilitli hedefi takip ettikçe bir "güven skoru" **saniye** cinsinden artar;
kaybolunca azalır. Değerler 60 fps referansıyla verilmiştir.

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `confirm_time` | `1.5` sn | Hedef "doğrulanmış" sayılmak için bu kadar süre takip edilmeli. Bu süreye ulaşmadan kaybolursa kilit **hemen** bırakılır (yanlış şeye kısa kilitlenmeyi önler). |
| `max_score_time` | `3.0` sn | Skor tavanı. Coast (kayıp) süresini de belirler. |
| `init_score_time` | `~0.017` sn | İlk kilitlenmedeki başlangıç skoru. |
| `gain_rate` | `1.0` | Takip edilen her saniye skora eklenen (skor-sn/sn). |
| `decay_rate` | `1.0` | Kaybedilen her saniye skordan düşen. Büyütürsen kaybolunca **daha hızlı** FULL aramaya döner. |
| `video_fps` | `60.0` | Video kaynakta FPS okunamazsa kullanılacak varsayılan (skor `dt`'si için). Kamera'da gerçek zaman (duvar saati) kullanılır. |

> **Coast (kaybolunca ne kadar bekler):** Doğrulanmış güçlü bir kilit kaybolunca,
> skor `confirm_time`'ın altına düşene kadar ROI'de bekler; sonra FULL'e döner.
> Süre ≈ `(mevcut_skor − confirm_time) / decay_rate`. Daha hızlı vazgeçmesi için
> `decay_rate`'i büyüt (örn. 3–6).

### ROI penceresi

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `small_window` | `384` | Küçük ROI penceresi (px). Hedef küçükken kullanılır. |
| `large_window` | `960` | Büyük ROI penceresi (px). Hedef büyüyünce kullanılır. |
| `area_threshold` | `0.07` | Hedef bbox alanı / kare alanı bu oranı geçince `large_window`'a geçilir. |
| `roi_smooth` | `0.4` | ROI merkezinin hedefe yumuşak yaklaşımı (0 = hiç oynama, 1 = anında). Düşürürsen pencere daha yumuşak/yavaş takip eder. |

### FULL arama periyodu

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `full_interval` | `5` | **Video'da** FULL aramanın periyodu (her N frame'de bir tam kare taranır). **Kamera kaynağında (int) otomatik olarak devre dışı** kalır — her frame full taranır. |

### Tracker (ByteTrack) — hareketli kamera / hızlı hedef için

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `box_inflate` | `4.0` | Tracker'a kutular bu oranda **şişirilmiş** verilir (merkez aynı). Hızlı harekette IoU örtüşmesini korur → ID kaçırmayı azaltır. Kabaca **~(k−1)× kutu boyutu/frame** hıza dayanır. Hesap yükü **sıfıra yakın** (Jetson-dostu). Hâlâ kaçırıyorsa 6–8'e çıkar. |
| `track_buffer` | `300` | Kaybolan track kaç frame "lost" tutulsun. Nesne bu sürede geri gelirse **aynı ID** ile bağlanır. Büyük değer = uzun kayıpları kurtarır ama uzun kayıpta yanlış nesneye eski ID'yi bağlama riski artar. (30 fps'de 300 = 10 sn.) |
| `match_thresh` | `0.95` | Eşleştirme gevşekliği (yüksek = düşük IoU'da bile bağla). |
| `fuse_score` | `False` | ByteTrack'in conf-füzyon katılığı; kapalı bırakılır (hızlı harekette daha iyi). |
| `tracker_cfg` | `"bytetrack.yaml"` | Temel tracker konfigürasyonu (ultralytics'ten). |
| `frame_rate` | `30` | Tracker'ın iç zamanlama referansı. |

---

## Örnekler

**Video, göster + kaydet:**
```python
trk = ROITracker("model.pt")
list(trk.roi_track("ucus.mp4", show=True, save=True, save_path="cikti.mp4"))
```

**Kamera (canlı), sadece veri (çizimsiz), kendi işleme:**
```python
trk = ROITracker("model.pt")
for r in trk.roi_track(0, stream=True):        # kamera -> full periyodu yok
    if len(r.boxes):
        tid = int(r.boxes.id[0])
        x1, y1, x2, y2 = r.boxes.xyxy[0].tolist()
        # ... kendi mantığın (yönlendirme, çizim vb.)
```

**ROI penceresini/gri kutuları gizle (sadece hedef çizilsin):**
```python
trk.roi_track("ucus.mp4", show=True, draw_roi=False)
```

**Ayrı modeller + ince ayar:**
```python
trk = ROITracker(
    model_full="full.pt", model_roi="roi.pt",
    target_class=0,
    lock_conf=0.7,           # daha seçici kilitlenme
    box_inflate=6.0,         # daha hızlı hedef toleransı
    track_buffer=120,        # ~4 sn (30 fps)
    decay_rate=3.0,          # kaybolunca daha hızlı FULL'e dön
    full_imgsz=1280,         # Jetson için daha hafif full arama
)
```

**OSD yazılarını maskele (FPV telemetrisi İHA sanılıyorsa):**
```python
trk = ROITracker("model.pt", mask=True)              # hazır FPV yerleşimi

# OSD yerleşimin farklıysa kendi bölgelerini ver (normalize 0-1 oran):
trk = ROITracker("model.pt", mask=True, mask_regions=[
    (0.02, 0.02, 0.18, 0.17),   # sol-üst blok
    (0.84, 0.01, 1.00, 0.16),   # sağ-üst blok
])
```
> Not: Maskelenen bölgede gerçek hedef de görünmez olur. Özellikle ekran
> ortasındaki `DISARMED/TRIGGER` bölgesi uçuşta kaybolan bir yazıysa, o bölgeyi
> `mask_regions`'tan çıkarmak isteyebilirsin.

**Yeni kaynağa geçerken durumu sıfırla:**
```python
trk.reset()
```

---

## Sık Sorulanlar / İpuçları

- **Hedefe geç kilitleniyor:** `lock_conf`'u düşür.
- **Çöpe/yanlış şeye kilitleniyor:** `lock_conf`'u yükselt.
- **Hızlı hedefte ID değişiyor:** `box_inflate`'i artır (6–8).
- **Kısa kaybolmalarda kilidi bırakıyor:** `track_buffer` ve/veya `confirm_time`'ı ayarla.
- **Kaybolunca çok uzun bekliyor:** `decay_rate`'i büyüt.
- **FPS düşük (Jetson):** `full_imgsz`'i düşür, modeli TensorRT/FP16'ya export et. Tracker tarafı zaten çok hafif.
- **Çıktı neden tek kutu?** Sistem tek hedef kilitler; hepsini istersen `all_tracks=True`.
