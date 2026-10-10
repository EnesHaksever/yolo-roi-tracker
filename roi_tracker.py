# -*- coding: utf-8 -*-
"""
roi_tracker.py
================
Dinamik Pencere Kilidi (Dynamic Window Lock) sistemini, YOLO'nun
model.track() fonksiyonunun DROP-IN yerine gececek sekilde tek bir CLASS +
tek fonksiyon (roi_track) halinde modulerlestirir.

Kullanim (YOLO model.track gibi) — IKI MOD:

    from roi_tracker import ROITracker

    trk = ROITracker("C:/Datasets/Yolomodels/yolo11m-480-ensondataset.pt")

    # 1) KAYNAK MODU (ic dongu): video yolu veya kamera indeksi ver,
    #    okuma/gosterme/kaydetme icerde yapilir.
    for res in trk.roi_track("video.mov", show=True, save=True, stream=True):
        for b in res.boxes:                 # YOLO Results ile BIREBIR ayni
            x1, y1, x2, y2 = b.xyxy[0]
            tid  = int(b.id[0]) if b.id is not None else None
            conf = float(b.conf[0])
            cls  = int(b.cls[0])

    # gercek zamanli kamera (0,1,2): 5-frame periyodu OTOMATIK kalkar
    trk.roi_track(0, show=True)

    # 2) FRAME MODU (dis dongu): YOLO'daki model(frame) gibi tek kare ver.
    #    Durum (ByteTrack, kilit, skor) cagrilar arasinda KORUNUR; okuma/
    #    gosterme/kaydetme DIS dongunun isidir (show/save yok sayilir).
    cap = cv2.VideoCapture("video.mov")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = trk.roi_track(frame)                # -> TEK Results (liste degil)
        # canli kamera karesiyse: trk.roi_track(frame, realtime=True)
        disp = trk.render(frame)                  # istege bagli: cizili kare
        cv2.imshow("out", disp)

Ciktinin formati YOLO ile aynidir: her frame icin bir ultralytics Results
nesnesi doner (res.boxes, res.plot(), res.save(), res.boxes.id, .conf, .cls ...).
Bu sistem tek hedef kilidi yaptigi icin, kilitliyken Results TEK kutu (kilitli
hedef) icerir; all_tracks=True verilirse tum trackler doner.

Onemli:
  * ByteTrack GOMULU ve ZORUNLU (paylasilan, global koordinat).
  * Guven skoru SANIYE tabanli -> degisken FPS'li gercek zaman kamerada dogru
    calisir (varsayilanlar 60 fps referansiyla verilmistir).
  * Kaynak kameraysa (int) FULL aramada 5-frame periyodu YOK; video ise korunur.
"""

import time
import numpy as np
import torch

try:
    from ultralytics import YOLO
    from ultralytics.trackers import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml
    from ultralytics.engine.results import Results
except Exception as e:   # pragma: no cover
    raise ImportError("ultralytics gerekli: pip install ultralytics") from e


class _Detections:
    """ByteTracker.update icin minimal sarmalayici (.xywh/.conf/.cls/.xyxy)."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Detections(self.xywh[idx], self.conf[idx], self.cls[idx])

    @property
    def xyxy(self):
        cx, cy, w, h = self.xywh[:, 0], self.xywh[:, 1], self.xywh[:, 2], self.xywh[:, 3]
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def _as_model(m):
    return YOLO(m) if isinstance(m, str) else m


# FPV OSD (telemetri yazilari) icin varsayilan maske bolgeleri.
# NORMALIZE koordinat (x1, y1, x2, y2; 0..1 oran) -> her cozunurlukte calisir.
# Referans goruntu: sol-ust sinyal/LQ, sag-ust fly/ALT/SPD, orta DISARMED/
# TRIGGER, sag-orta Mode, sol-alt akim/voltaj gostergeleri.
DEFAULT_MASK_REGIONS = [
    (0.02, 0.02, 0.18, 0.17),   # sol-ust  : sinyal (-10), LQ 2:100
    (0.84, 0.01, 1.00, 0.16),   # sag-ust  : fly min / ALT / SPD
    (0.30, 0.61, 0.71, 0.74),   # orta     : DISARMED / TRIGGER:NOT READY
    (0.76, 0.68, 1.00, 0.76),   # sag-orta : Mode: ANGL AIR
    (0.02, 0.75, 0.18, 0.97),   # sol-alt  : 8.00A / 4.20v / 25.2v
]


class ROITracker:
    """Dinamik pencere kilidi + gomulu paylasilan ByteTrack; YOLO drop-in.

    KONSTRUKTOR: modeller + kalici ayar/tuning parametreleri (tracker, esikler,
    skor, pencere...). Bunlar takip davranisini belirler ve durum tasir.
    ROI_TRACK (fonksiyon): kosuya ozel G/C secenekleri (source, show, save,
    stream, draw_roi...). Bu ayrim YOLO ile ayni mantiktadir (model kurulurken;
    calisma secenekleri cagride).
    """

    def __init__(
        self,
        model_full,
        model_roi=None,
        *,
        target_class=2,
        conf_thres=0.05,
        lock_conf=0.6,
        small_window=384,
        large_window=960,
        area_threshold=0.07,
        full_imgsz=960,
        roi_imgsz=384,
        full_interval=5,
        video_fps=60.0,
        device=0,
        # --- SANIYE tabanli guven skoru (60 fps referans) ---
        confirm_time=90 / 60.0,       # 1.5 sn  (eski 90 frame)
        max_score_time=180 / 60.0,    # 3.0 sn  (eski 180 frame)
        init_score_time=1 / 60.0,     # ~0.017 sn (eski 1 frame)
        gain_rate=1.0,                # takip edilen her sn +1.0 skor-sn
        decay_rate=1.0,               # kaybedilen her sn -1.0 skor-sn
        # --- ROI ---
        roi_smooth=0.4,
        # --- OSD MASKELEME ---
        mask=False,
        mask_regions=None,
        # --- ByteTrack ---
        tracker_cfg="bytetrack.yaml",
        fuse_score=False,
        match_thresh=0.95,
        track_buffer=300,
        box_inflate=4.0,
        frame_rate=30,
    ):
        self.model_full = _as_model(model_full)
        self.model_roi = self.model_full if model_roi is None else _as_model(model_roi)
        self._names = self.model_full.names

        self.target_class = int(target_class)
        self.conf_thres = float(conf_thres)
        self.lock_conf = float(lock_conf)

        self.small_window = int(small_window)
        self.large_window = int(large_window)
        self.area_threshold = float(area_threshold)

        self.full_imgsz = int(full_imgsz)
        self.roi_imgsz = int(roi_imgsz)
        self.full_interval = max(1, int(full_interval))
        self.video_fps = float(video_fps)
        self.device = device

        self.confirm_time = float(confirm_time)
        self.max_score_time = float(max_score_time)
        self.init_score_time = float(init_score_time)
        self.gain_rate = float(gain_rate)
        self.decay_rate = float(decay_rate)

        self.roi_smooth = float(roi_smooth)

        # OSD maskeleme: mask=True ise tespitten ONCE frame kopyasinda bu
        # bolgeler gri (114) ile kapatilir; yazi/telemetri false-positive
        # olmaz. Cikti/cizim karesi degismez. mask_regions=None -> varsayilan
        # FPV OSD yerlesimi (DEFAULT_MASK_REGIONS, normalize 0..1).
        self.mask = bool(mask)
        self.mask_regions = [tuple(map(float, r)) for r in
                             (mask_regions if mask_regions is not None
                              else DEFAULT_MASK_REGIONS)]
        self._mask_cache = None   # (W, H) -> piksel dikdortgenleri

        self.box_inflate = float(box_inflate)
        self._tracker_args = dict(tracker_cfg=tracker_cfg, fuse_score=fuse_score,
                                  match_thresh=match_thresh, track_buffer=track_buffer,
                                  frame_rate=frame_rate)
        self._build_tracker()

        self._interval = self.full_interval   # kosuya gore guncellenir
        self.reset()

    # -----------------------------------------------------------------
    def _build_tracker(self):
        a = self._tracker_args
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(a["tracker_cfg"])))
        cfg.fuse_score = bool(a["fuse_score"])
        cfg.match_thresh = float(a["match_thresh"])
        cfg.track_buffer = int(a["track_buffer"])
        self.tracker = BYTETracker(cfg, frame_rate=int(a["frame_rate"]))

    def reset(self):
        """Kilidi/durumu ve tracker'i sifirla (yeni kaynak icin)."""
        self.tracking_mode = False
        self.roi = None
        self._frame_idx = 0
        self._last_time = None
        self.conf_score = 0.0
        self.confirmed = False
        self.locked_id = None
        self.track_cx = 0.0
        self.track_cy = 0.0
        self.track_w = 0
        self.track_h = 0
        self._draw_target = None    # (x1,y1,x2,y2,id,conf) cizim icin
        self._draw_others = []      # digerleri (gri)
        self._build_tracker()

    # -----------------------------------------------------------------
    def _mask_px(self, W, H):
        """Normalize maske bolgelerini bu cozunurluk icin piksele cevir (cache'li)."""
        if self._mask_cache is None or self._mask_cache[0] != (W, H):
            rects = []
            for x1, y1, x2, y2 in self.mask_regions:
                rects.append((max(0, int(x1 * W)), max(0, int(y1 * H)),
                              min(W, int(x2 * W)), min(H, int(y2 * H))))
            self._mask_cache = ((W, H), rects)
        return self._mask_cache[1]

    def _apply_mask(self, frame, W, H):
        """Tespit icin kullanilacak kopyada OSD bolgelerini gri (114) kapat."""
        det = frame.copy()
        for x1, y1, x2, y2 in self._mask_px(W, H):
            det[y1:y2, x1:x2] = 114   # YOLO letterbox grisi (notr)
        return det

    # -----------------------------------------------------------------
    def _make_results(self, frame, rows):
        """YOLO Results uret (boxes: Nx7 [x1,y1,x2,y2,id,conf,cls])."""
        if rows:
            data = torch.tensor(rows, dtype=torch.float32)
        else:
            data = torch.zeros((0, 7), dtype=torch.float32)
        res = Results(orig_img=frame, path="", names=self._names, boxes=data)
        # ekstra bilgiler (YOLO tuketicilerini bozmaz)
        res.roi_window = self.roi
        res.roi_score = float(self.conf_score)
        res.roi_confirmed = bool(self.confirmed)
        res.roi_locked_id = self.locked_id
        res.roi_mode = "ROI" if self.tracking_mode else "FULL"
        return res

    # -----------------------------------------------------------------
    def _step(self, frame, dt, all_tracks):
        """Tek frame isle, durumu guncelle, YOLO Results dondur."""
        H, W = frame.shape[:2]
        frame_area = H * W
        self._frame_idx += 1

        # -------- OSD MASKELEME (sadece tespit girdisi; cikti karesi bozulmaz) --------
        det_frame = self._apply_mask(frame, W, H) if self.mask else frame

        # -------- MOD SECIMI --------
        if not self.tracking_mode:
            input_frame = det_frame
            offset_x, offset_y = 0, 0
            mode = "FULL"
        else:
            x, y, rw, rh = self.roi
            rw = min(rw, W)
            rh = min(rh, H)
            x = min(max(0, x), W - rw)
            y = min(max(0, y), H - rh)
            self.roi = (x, y, rw, rh)
            input_frame = det_frame[y:y + rh, x:x + rw]
            offset_x, offset_y = x, y
            mode = "ROI"

        # -------- TESPIT --------
        run_detect = True
        results = None
        if mode == "FULL":
            if self._frame_idx % self._interval == 0:
                results = self.model_full.predict(
                    input_frame, imgsz=self.full_imgsz, conf=self.conf_thres,
                    device=self.device, classes=[self.target_class], verbose=False)[0]
            else:
                run_detect = False
        else:
            results = self.model_roi.predict(
                input_frame, imgsz=self.roi_imgsz, conf=self.conf_thres,
                device=self.device, classes=[self.target_class], verbose=False)[0]

        # -------- GLOBAL koordinata cevir --------
        candidates = []
        if results is not None:
            for box in results.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf < self.conf_thres or cls != self.target_class:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                candidates.append((
                    (x1 + x2) / 2.0 + offset_x, (y1 + y2) / 2.0 + offset_y,
                    float(x2 - x1), float(y2 - y1), conf,
                ))

        # -------- PAYLASILAN TRACKER (sisirilmis kutu) --------
        tracks = []
        if run_detect:
            if candidates:
                k = self.box_inflate
                det = _Detections(
                    [(c[0], c[1], c[2] * k, c[3] * k) for c in candidates],
                    [c[4] for c in candidates],
                    [self.target_class] * len(candidates))
            else:
                det = _Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
            tracks = self.tracker.update(det)

        # -------- HEDEF SECIMI: id + conf --------
        target_row = None
        if len(tracks):
            if self.tracking_mode and self.locked_id is not None:
                for t in tracks:
                    if int(t[4]) == self.locked_id:
                        target_row = t
                        break
            else:
                lockable = [t for t in tracks if t[5] >= self.lock_conf]
                if lockable:
                    target_row = max(lockable, key=lambda t: t[5])
                    self.locked_id = int(target_row[4])

        # -------- SKOR + KONUM (SANIYE tabanli) --------
        target = None
        if target_row is not None:
            x1, y1, x2, y2 = target_row[0], target_row[1], target_row[2], target_row[3]
            tid = int(target_row[4]); conf = float(target_row[5])
            ncx = (x1 + x2) / 2.0; ncy = (y1 + y2) / 2.0
            rw = (x2 - x1) / self.box_inflate
            rh = (y2 - y1) / self.box_inflate

            if self.tracking_mode:
                self.conf_score = min(self.max_score_time,
                                      self.conf_score + dt * self.gain_rate)
                self.track_cx += self.roi_smooth * (ncx - self.track_cx)
                self.track_cy += self.roi_smooth * (ncy - self.track_cy)
            else:
                self.conf_score = self.init_score_time
                self.confirmed = False
                self.track_cx, self.track_cy = ncx, ncy

            if self.conf_score >= self.confirm_time:
                self.confirmed = True

            self.track_w, self.track_h = int(rw), int(rh)
            self.tracking_mode = True
            target = (ncx - rw / 2, ncy - rh / 2, ncx + rw / 2, ncy + rh / 2, tid, conf)

        elif self.tracking_mode:
            if not self.confirmed:
                self.conf_score = 0.0
                self.tracking_mode = False
                self.locked_id = None
            else:
                self.conf_score -= dt * self.decay_rate
                if self.conf_score < self.confirm_time:
                    self.conf_score = 0.0
                    self.tracking_mode = False
                    self.locked_id = None

        # -------- ROI penceresi --------
        if self.tracking_mode:
            bbox_area = self.track_w * self.track_h
            win = self.large_window if (bbox_area / frame_area > self.area_threshold) \
                else self.small_window
            self.roi = (int(self.track_cx - win // 2), int(self.track_cy - win // 2),
                        win, win)

        # -------- cizim verisi + Results --------
        others = []
        for t in tracks:
            cx, cy = (t[0] + t[2]) / 2.0, (t[1] + t[3]) / 2.0
            ww = (t[2] - t[0]) / self.box_inflate
            hh = (t[3] - t[1]) / self.box_inflate
            row = [cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2, int(t[4]), float(t[5])]
            if target is None or int(t[4]) != target[4]:
                others.append(row)
        self._draw_target = target
        self._draw_others = others

        # Results kutulari: all_tracks -> tum trackler, degilse sadece kilitli hedef
        if all_tracks:
            rows = []
            for t in tracks:
                cx, cy = (t[0] + t[2]) / 2.0, (t[1] + t[3]) / 2.0
                ww = (t[2] - t[0]) / self.box_inflate
                hh = (t[3] - t[1]) / self.box_inflate
                rows.append([cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2,
                             int(t[4]), float(t[5]), self.target_class])
        else:
            rows = [[target[0], target[1], target[2], target[3], target[4],
                     target[5], self.target_class]] if target is not None else []

        return self._make_results(frame, rows)

    # -----------------------------------------------------------------
    def _render(self, frame, draw_roi):
        """Cizim (show/save icin). draw_roi=False -> mavi pencere ve gri
        digerleri cizilmez; sadece kilitli yesil hedef cizilir."""
        import cv2
        disp = frame.copy()
        if draw_roi:
            for x1, y1, x2, y2, tid, conf in self._draw_others:
                cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (120, 120, 120), 1)
                cv2.putText(disp, f"id{int(tid)} {conf:.2f}", (int(x1), max(0, int(y1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            if self.tracking_mode and self.roi is not None:
                x, y, w, h = self.roi
                cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if self._draw_target is not None:
            x1, y1, x2, y2, tid, conf = self._draw_target
            cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(disp, f"ID:{tid} {conf:.2f} S:{self.conf_score:.2f}s",
                        (int(x1), max(0, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        return disp

    # -----------------------------------------------------------------
    def render(self, frame, draw_roi=True):
        """Son islenen kareyi cizili dondur (frame modunda show/save yerine).

            res  = trk.roi_track(frame)
            disp = trk.render(frame)   # yesil hedef + mavi ROI + gri digerleri
        """
        return self._render(frame, draw_roi)

    # -----------------------------------------------------------------
    def roi_track(self, source, *, stream=False, show=False, save=False,
                  save_path="roi_output.mp4", draw_roi=True, all_tracks=False,
                  verbose=False, realtime=False, dt=None):
        """YOLO model.track() gibi: kaynagi al, isle, YOLO Results dondur.

        source : video yolu (str), kamera indeksi (int; 0/1/2), veya TEK KARE
                 (np.ndarray).

        --- FRAME MODU (source np.ndarray; dis dongu) ---
        TEK Results doner (liste/generator DEGIL). Durum (ByteTrack, kilit,
        skor) cagrilar arasinda KORUNUR; dis dongu her yeni kareyi ayni
        instance'a verir. show/save/stream/draw_roi YOK SAYILIR — gosterme/
        kaydetme dis dongunun isidir; cizili kare icin trk.render(frame).
        FULL arama periyodu videodaki gibi full_interval'dir.
        realtime : True -> her frame FULL aday + dt duvar saatinden olculur
                 (canli kamera karesi besliyorsan bunu ver).
        dt       : kare suresi (sn). Verilirse aynen kullanilir (orn. 30 fps
                 video okuyorsan dt=1/30). Verilmezse 1/video_fps.
        Durumu sifirlamak icin (yeni video/kaynak): trk.reset().

        --- KAYNAK MODU (str/int; ic dongu) ---
        stream : True -> her frame icin Results ureten generator; False -> liste.
        show   : True -> pencerede goster (ESC ile cikis).
        save   : True -> save_path'e video yaz.
        draw_roi : False -> mavi ROI penceresi ve gri diger-nesne kutulari
                 cizilmez (yalnizca show/save icin gecerli).
        all_tracks : True -> Results tum trackleri icerir; False -> sadece kilitli hedef.
        KAMERA ise FULL aramada 5-frame periyodu OTOMATIK kalkar (her frame full).
        NOT: kaynak modunda durum her cagrida OTOMATIK sifirlanir (yeni kaynak
        temiz baslar); frame modunda asla sifirlanmaz.
        """
        if isinstance(source, np.ndarray):
            return self._step_external(source, realtime=realtime, dt=dt,
                                       all_tracks=all_tracks, verbose=verbose)
        gen = self._run(source, show, save, save_path, draw_roi, all_tracks, verbose)
        return gen if stream else list(gen)

    # -----------------------------------------------------------------
    def _step_external(self, frame, *, realtime, dt, all_tracks, verbose):
        """Dis donguden gelen tek kareyi isle; durum cagrilar arasi korunur."""
        self._interval = 1 if realtime else self.full_interval
        if dt is None:
            if realtime:
                now = time.time()
                dt = (now - self._last_time) if self._last_time else (1.0 / self.video_fps)
                self._last_time = now
                dt = max(1e-4, dt)
            else:
                dt = 1.0 / self.video_fps
        res = self._step(frame, float(dt), all_tracks)
        if verbose:
            print(f"[roi_track] mode={res.roi_mode} score={res.roi_score:.2f}s "
                  f"id={res.roi_locked_id} nbox={len(res.boxes)}")
        return res

    # -----------------------------------------------------------------
    def _run(self, source, show, save, save_path, draw_roi, all_tracks, verbose):
        import cv2

        # yeni kaynak -> temiz durum (tracker/kilit/skor sifirla)
        self.reset()

        # kamera mi?
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        is_cam = isinstance(source, int)
        self._interval = 1 if is_cam else self.full_interval

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Kaynak acilamadi: {source}")
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 0:
            src_fps = self.video_fps
        writer = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # dt: kamera -> duvar saati, video -> 1/fps
                if is_cam:
                    now = time.time()
                    dt = (now - self._last_time) if self._last_time else (1.0 / src_fps)
                    self._last_time = now
                    dt = max(1e-4, dt)
                else:
                    dt = 1.0 / src_fps

                res = self._step(frame, dt, all_tracks)

                if show or save:
                    disp = self._render(frame, draw_roi)
                    if save:
                        if writer is None:
                            h, w = disp.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(save_path, fourcc,
                                                     src_fps if src_fps > 0 else 30, (w, h))
                        writer.write(disp)
                    if show:
                        cv2.imshow("roi_track", disp)
                        if cv2.waitKey(1) == 27:
                            break
                if verbose:
                    print(f"[roi_track] mode={res.roi_mode} score={res.roi_score:.2f}s "
                          f"id={res.roi_locked_id} nbox={len(res.boxes)}")
                yield res
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()


# =====================================================================
# DEMO
# =====================================================================
if __name__ == "__main__":
    import sys
    MODEL = r"C:\\Datasets\\Yolomodels\\yolo11m_yeni2.pt"
    src = sys.argv[1] if len(sys.argv) > 1 else r"C:\Python\QuantumDynamic\OtoLabeling\OyunKayit\2026-06-28-16-13-59.mp4"

    trk = ROITracker(MODEL, mask=True)

    DEMO_FRAME_MODE = False   # True -> dis dongu (frame modu) demosu

    if not DEMO_FRAME_MODE:
        # KAYNAK MODU: video show + save; kamera icin: trk.roi_track(0, show=True)
        for res in trk.roi_track(src, stream=True, show=True, save=True, draw_roi=True):
            # ornek: kilitli hedefin id/konumu (YOLO ile ayni erisim)
            if len(res.boxes):
                b = res.boxes[0]
                # print(int(b.id[0]) if b.id is not None else None, b.xyxy[0].tolist())
                pass
    else:
        # FRAME MODU: okuma/gosterme dis dongude, tracker sadece kare isler.
        import cv2
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = trk.roi_track(frame, dt=1.0 / fps)   # canli kamera: realtime=True
            disp = trk.render(frame)
            cv2.imshow("frame_mode", disp)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
