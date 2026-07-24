"""
zone_monitor.py — Multi-Area Zone Presence Monitoring Engine

Fitur:
- Deteksi person (YOLO nano) secara INDEPENDEN dari mode AI utama
- Mendukung BANYAK ZONA per kamera, masing-masing independen
- Metode Continuous Presence Timer + Grace Period:
    * Timer berjalan selama orang ada di zona
    * Jika orang keluar < grace_period_seconds, timer TIDAK reset (ambil minum sebentar OK)
    * Jika orang keluar >= grace_period_seconds, timer di-reset ke 0 (sesi baru)
- DEBOUNCE: status "tidak hadir" baru ditetapkan setelah 3 detik tidak terdeteksi
  (mengatasi noise YOLO sesaat akibat pose/pencahayaan)
- Evaluasi otomatis di setiap jam bulat berdasarkan Jam Mulai Operasional (start_hour)
- Threshold BERBEDA PER ZONA (default 15 menit)
- Snapshot frame kamera dikirim bersama notifikasi Telegram
- Thread-safe, daemon thread, cleanup otomatis saat shutdown
"""

import os
import cv2
import time
import json
import sqlite3
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ── Lazy import notifier to avoid circular imports
def _get_notifier():
    from telegram_notifier import send_zone_alert
    return send_zone_alert

# ── Database path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zone_config.db")

# ── COCO person class ID
COCO_PERSON = 0

# ── Minimum IoU/containment ratio for "person inside zone"
ZONE_OVERLAP_THRESHOLD = 0.25

# ── Debounce: jumlah detik berturut-turut tanpa deteksi sebelum status ke "tidak hadir"
PRESENCE_DEBOUNCE_SECS = 3.0


# ═══════════════════════════════════════════════════════
#  ZoneConfig — Konfigurasi 1 Zona
# ═══════════════════════════════════════════════════════

class ZoneConfig:
    """Konfigurasi satu zona monitoring."""

    def __init__(self, zone_id: str, cam_id: int, name: str,
                 coords: List[List[float]],
                 threshold_minutes: int = 15,
                 cycle_hours: int = 1,
                 telegram_enabled: bool = True,
                 start_hour: str = "08:00",
                 grace_period_seconds: int = 60):
        self.zone_id = zone_id
        self.cam_id = cam_id
        self.name = name
        # coords: list of [x_norm, y_norm] dalam 0.0–1.0 (normalized ke frame)
        self.coords = coords
        self.threshold_minutes = threshold_minutes
        self.cycle_hours = max(1, int(cycle_hours))
        self.telegram_enabled = telegram_enabled
        # Jam operasional mulai, format "HH:MM" (default "08:00")
        self.start_hour = start_hour
        # Grace period: toleransi jeda keluar dari zona (detik) sebelum timer reset
        self.grace_period_seconds = max(0, int(grace_period_seconds))

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "cam_id": self.cam_id,
            "name": self.name,
            "coords": self.coords,
            "threshold_minutes": self.threshold_minutes,
            "cycle_hours": self.cycle_hours,
            "telegram_enabled": self.telegram_enabled,
            "start_hour": self.start_hour,
            "grace_period_seconds": self.grace_period_seconds,
        }

    @staticmethod
    def from_dict(d: dict) -> "ZoneConfig":
        return ZoneConfig(
            zone_id=d["zone_id"],
            cam_id=d["cam_id"],
            name=d["name"],
            coords=d["coords"],
            threshold_minutes=d.get("threshold_minutes", 15),
            cycle_hours=d.get("cycle_hours", 1),
            telegram_enabled=d.get("telegram_enabled", True),
            start_hour=d.get("start_hour", "08:00"),
            grace_period_seconds=d.get("grace_period_seconds", 60),
        )


# ═══════════════════════════════════════════════════════
#  ZoneContinuousTracker — State Continuous Presence Timer
# ═══════════════════════════════════════════════════════

class ZoneContinuousTracker:
    """
    Melacak kehadiran orang menggunakan metode Continuous Presence Timer.

    Logika:
    - Selama orang HADIR: timer bertambah sesuai waktu nyata.
    - Jika orang KELUAR < grace_period_seconds: timer TIDAK reset, 
      hanya dijeda. Saat orang kembali, timer lanjut dari nilai sebelumnya.
    - Jika orang KELUAR >= grace_period_seconds: timer di-reset ke 0 (sesi baru).
    - DEBOUNCE: status baru berubah ke "tidak hadir" hanya setelah
      PRESENCE_DEBOUNCE_SECS detik berturut-turut tidak terdeteksi.
    """

    def __init__(self, hour_label: str, grace_period_seconds: int = 60):
        self.hour_label = hour_label
        self.grace_period_seconds = grace_period_seconds

        # Total waktu kehadiran kontinyu (detik)
        self._continuous_seconds: float = 0.0
        # Waktu saat sesi hadir terakhir dimulai (None jika tidak hadir)
        self._session_start: Optional[float] = None
        # Waktu terakhir orang terdeteksi (untuk grace period + debounce)
        self._last_seen_time: Optional[float] = None
        # Waktu pertama kali TIDAK terdeteksi (untuk debounce)
        self._first_absent_time: Optional[float] = None
        # Status hadir "efektif" setelah debounce
        self._is_present_debounced: bool = False
        # Lock (RLock to allow re-entrant property access like is_person_present inside snapshot)
        self.lock = threading.RLock()

    def update(self, raw_present: bool, timestamp: float):
        """
        Dipanggil setiap kali ada frame baru dari detector.
        raw_present: apakah orang terdeteksi di frame ini (sebelum debounce).
        """
        with self.lock:
            if raw_present:
                self._last_seen_time = timestamp
                self._first_absent_time = None
                self._is_present_debounced = True

                if self._session_start is None:
                    self._session_start = timestamp
                else:
                    elapsed = timestamp - self._session_start
                    elapsed = min(elapsed, 3.0)  # Cap max 3 detik per frame (anti-lag spike)
                    if elapsed > 0:
                        self._continuous_seconds += elapsed
                    self._session_start = timestamp  # Rolling update

            else:
                # Orang tidak terdeteksi di frame ini
                now = timestamp

                if self._first_absent_time is None:
                    self._first_absent_time = now

                # PENTING: Selama masih dalam Grace Period (misal <= 40s), TETAP HADIR & TETAP AKUMULASI!
                if self._last_seen_time is not None:
                    gap = now - self._last_seen_time
                    if gap < self.grace_period_seconds:
                        # Masih dalam toleransi jeda: tetap akumulasi timer & tetap status HADIR
                        if self._session_start is not None:
                            elapsed = now - self._session_start
                            elapsed = min(elapsed, 3.0)
                            if elapsed > 0:
                                self._continuous_seconds += elapsed
                            self._session_start = now
                        else:
                            self._session_start = now
                        self._is_present_debounced = True
                    else:
                        # Melewati grace period (misal > 40s): status TIDAK HADIR & RESET TIMER KE 0.0m!
                        self._is_present_debounced = False
                        self._session_start = None
                        if self._continuous_seconds > 0:
                            print(
                                f"[ZONE-TRACKER] Grace period exceeded while absent ({gap:.1f}s >= "
                                f"{self.grace_period_seconds}s). Resetting continuous timer to 0.",
                                flush=True
                            )
                            self._continuous_seconds = 0.0
                else:
                    absent_duration = now - self._first_absent_time
                    if absent_duration >= PRESENCE_DEBOUNCE_SECS:
                        self._is_present_debounced = False
                        self._session_start = None

    @property
    def accumulated_minutes(self) -> float:
        with self.lock:
            return self._continuous_seconds / 60.0

    @property
    def accumulated_seconds(self) -> float:
        with self.lock:
            return self._continuous_seconds

    @property
    def is_person_present(self) -> bool:
        with self.lock:
            # 1. Jika terdeteksi / debounced active -> HADIR
            if self._is_present_debounced:
                return True
            # 2. SELAMA masih dalam Grace Period (belum reset ke 0 & akumulasi > 0) -> tetap HADIR!
            if self._last_seen_time is not None:
                gap = time.time() - self._last_seen_time
                if gap < self.grace_period_seconds and self._continuous_seconds > 0:
                    return True
            # 3. Grace period habis -> TIDAK HADIR
            return False

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "hour_label": self.hour_label,
                "accumulated_seconds": round(self._continuous_seconds, 1),
                "accumulated_minutes": round(self._continuous_seconds / 60.0, 2),
                "is_person_present": self.is_person_present,
                "grace_period_seconds": self.grace_period_seconds,
            }


# Alias lama untuk backward compatibility
ZoneCycleTracker = ZoneContinuousTracker


# ═══════════════════════════════════════════════════════
#  ZoneDatabase — Persistensi Konfigurasi & History
# ═══════════════════════════════════════════════════════

class ZoneDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_id TEXT NOT NULL UNIQUE,
                cam_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                coords_json TEXT NOT NULL,
                threshold_minutes INTEGER DEFAULT 15,
                cycle_hours INTEGER DEFAULT 1,
                telegram_enabled INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS zone_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_id TEXT NOT NULL,
                cam_id INTEGER NOT NULL,
                zone_name TEXT NOT NULL,
                cycle_label TEXT NOT NULL,
                accumulated_minutes REAL NOT NULL,
                threshold_minutes INTEGER NOT NULL,
                alert_type TEXT,
                telegram_sent INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Migrasi kolom baru (aman jika sudah ada — akan gagal silent)
        for migration in [
            "ALTER TABLE zones ADD COLUMN cycle_hours INTEGER DEFAULT 1",
            "ALTER TABLE zones ADD COLUMN start_hour TEXT DEFAULT '08:00'",
            "ALTER TABLE zones ADD COLUMN grace_period_seconds INTEGER DEFAULT 60",
        ]:
            try:
                cur.execute(migration)
                conn.commit()
            except Exception:
                pass

        conn.commit()
        conn.close()
        print("[ZONE-DB] Database initialized OK.", flush=True)

    def save_zone(self, zone: ZoneConfig):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO zones
                (zone_id, cam_id, name, coords_json, threshold_minutes, cycle_hours,
                 telegram_enabled, start_hour, grace_period_seconds, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            zone.zone_id,
            zone.cam_id,
            zone.name,
            json.dumps(zone.coords),
            zone.threshold_minutes,
            zone.cycle_hours,
            1 if zone.telegram_enabled else 0,
            zone.start_hour,
            zone.grace_period_seconds,
        ))
        conn.commit()
        conn.close()

    def delete_zone(self, zone_id: str):
        conn = self._connect()
        conn.execute("DELETE FROM zones WHERE zone_id = ?", (zone_id,))
        conn.commit()
        conn.close()

    def _row_to_zone(self, row) -> Optional[ZoneConfig]:
        try:
            coords = json.loads(row["coords_json"])
            keys = row.keys()
            return ZoneConfig(
                zone_id=row["zone_id"],
                cam_id=row["cam_id"],
                name=row["name"],
                coords=coords,
                threshold_minutes=row["threshold_minutes"],
                cycle_hours=row["cycle_hours"] if "cycle_hours" in keys and row["cycle_hours"] else 1,
                telegram_enabled=bool(row["telegram_enabled"]),
                start_hour=row["start_hour"] if "start_hour" in keys and row["start_hour"] else "08:00",
                grace_period_seconds=row["grace_period_seconds"] if "grace_period_seconds" in keys and row["grace_period_seconds"] is not None else 60,
            )
        except Exception as e:
            print(f"[ZONE-DB] Error parsing zone row: {e}", flush=True)
            return None

    def load_all_zones(self) -> List[ZoneConfig]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM zones ORDER BY cam_id, id")
        rows = cur.fetchall()
        conn.close()
        return [z for z in (self._row_to_zone(r) for r in rows) if z is not None]

    def load_zones_for_camera(self, cam_id: int) -> List[ZoneConfig]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM zones WHERE cam_id = ? ORDER BY id", (cam_id,))
        rows = cur.fetchall()
        conn.close()
        return [z for z in (self._row_to_zone(r) for r in rows) if z is not None]

    def log_event(self, zone: ZoneConfig, cycle_label: str,
                  accumulated_minutes: float, alert_type: Optional[str],
                  telegram_sent: bool):
        conn = self._connect()
        conn.execute("""
            INSERT INTO zone_events
                (zone_id, cam_id, zone_name, cycle_label, accumulated_minutes,
                 threshold_minutes, alert_type, telegram_sent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            zone.zone_id, zone.cam_id, zone.name, cycle_label,
            round(accumulated_minutes, 2), zone.threshold_minutes,
            alert_type, 1 if telegram_sent else 0,
        ))
        conn.commit()
        conn.close()

    def get_events(self, cam_id: Optional[int] = None,
                   zone_id: Optional[str] = None,
                   limit: int = 100) -> List[dict]:
        conn = self._connect()
        cur = conn.cursor()
        if zone_id:
            cur.execute(
                "SELECT * FROM zone_events WHERE zone_id = ? ORDER BY id DESC LIMIT ?",
                (zone_id, limit))
        elif cam_id is not None:
            cur.execute(
                "SELECT * FROM zone_events WHERE cam_id = ? ORDER BY id DESC LIMIT ?",
                (cam_id, limit))
        else:
            cur.execute(
                "SELECT * FROM zone_events ORDER BY id DESC LIMIT ?",
                (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════
#  Geometry Helper — Deteksi Person dalam Zona
# ═══════════════════════════════════════════════════════

def is_person_in_zone(bbox: Tuple[int, int, int, int],
                      zone_coords_norm: List[List[float]],
                      frame_w: int, frame_h: int) -> bool:
    """
    Cek apakah person bounding box berada (secara signifikan) di dalam zona.

    Strategi Multi-Anchor + Dual Intersection:
    1. Konversi koordinat zona dari normalized (0-1) ke pixel.
    2. Uji 5 titik jangkar tubuh (Kepala, Dada, Center, Lutut, Kaki).
       Jika SALAH SATU titik ada di dalam poligon zona, return True.
    3. Hitung rasio overlap area irisan terhadap luas person ATAU luas zona.
       Jika irisan >= 15% dari luas person ATAU 15% dari luas zona, return True.
    """
    if len(zone_coords_norm) < 3:
        return False
    if frame_w <= 0 or frame_h <= 0:
        return False

    x1, y1, x2, y2 = bbox
    h_bbox = max(1, y2 - y1)
    center_x = (x1 + x2) // 2

    # Konversi zona ke pixel
    pts = np.array(
        [[int(p[0] * frame_w), int(p[1] * frame_h)] for p in zone_coords_norm],
        dtype=np.int32
    )

    # Grid 9 titik jangkar tubuh (Kiri, Tengah, Kanan x Atas, Tengah, Bawah)
    xs = [
        float(x1 + int((x2 - x1) * 0.25)),
        float(center_x),
        float(x1 + int((x2 - x1) * 0.75)),
    ]
    ys = [
        float(y1 + int(h_bbox * 0.20)),
        float((y1 + y2) / 2),
        float(y1 + int(h_bbox * 0.80)),
    ]

    for px in xs:
        for py in ys:
            if cv2.pointPolygonTest(pts, (px, py), False) >= 0:
                return True

    # Real polygon intersection test using OpenCV contour mask
    person_area = max(1, (x2 - x1) * h_bbox)
    mask_zone = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(mask_zone, [pts], 255)

    mask_person = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.rectangle(mask_person, (x1, y1), (x2, y2), 255, -1)

    inter_mask = cv2.bitwise_and(mask_zone, mask_person)
    overlap_pixels = cv2.countNonZero(inter_mask)

    if (overlap_pixels / person_area >= 0.10):
        return True

    return False


def draw_zones_on_frame(frame: np.ndarray,
                        zones: List[ZoneConfig],
                        trackers: Dict[str, "ZoneContinuousTracker"],
                        person_bboxes: Optional[List[Tuple[int, int, int, int]]] = None) -> np.ndarray:
    """
    Gambar overlay zona di atas frame dengan:
    - Warna berdasarkan status kehadiran & progress threshold
    - Label zona + status (Hadir/Tidak Hadir) + akumulasi menit real-time
    - Bounding box orang yang terdeteksi (opsional)
    """
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Gambar bounding box person (mode AI zone_monitor)
    if person_bboxes:
        for (bx1, by1, bx2, by2) in person_bboxes:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 200), 2)
            cv2.putText(frame, "Person", (bx1 + 4, by1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

    for zone in zones:
        tracker = trackers.get(zone.zone_id)
        accum_min = tracker.accumulated_minutes if tracker else 0.0
        is_present = tracker.is_person_present if tracker else False
        threshold = zone.threshold_minutes
        ratio = accum_min / threshold if threshold > 0 else 1.0

        # Warna berdasarkan status kehadiran
        if is_present:
            if ratio >= 1.0:
                color = (0, 220, 60)     # Hijau terang: hadir & sudah cukup
            else:
                color = (0, 180, 255)    # Orange: hadir, belum cukup
        else:
            if ratio >= 1.0:
                color = (40, 200, 40)    # Hijau redup: sudah cukup tapi sedang absen
            else:
                color = (0, 50, 220)     # Merah: tidak hadir & belum cukup

        # Konversi koordinat zona
        pts = np.array(
            [[int(p[0] * w), int(p[1] * h)] for p in zone.coords],
            dtype=np.int32
        )

        # Gambar filled polygon (transparan)
        cv2.fillPoly(overlay, [pts], color)
        # Border zona
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)

        # Label zona + status
        if len(pts) > 0:
            lx, ly = pts[0]
            status_str = "[HADIR]" if is_present else "[TIDAK HADIR]"
            label = f"{zone.name} | {accum_min:.1f}/{threshold}m | {status_str}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (lx, ly - th - 8), (lx + tw + 8, ly + 2), (10, 10, 30), -1)
            text_color = (80, 255, 120) if is_present else (80, 160, 255)
            cv2.putText(frame, label, (lx + 4, ly - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    # Blend overlay (alpha = 0.20 untuk transparansi zone fill)
    frame = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)
    return frame


# ═══════════════════════════════════════════════════════
#  ZoneMonitor — Engine Utama
# ═══════════════════════════════════════════════════════

class ZoneMonitor:
    """
    Engine utama Zone Monitoring.

    - Singleton global, diinisialisasi sekali saat ai_service.py startup.
    - Thread deteksi YOLO nano terpisah per kamera (independen dari mode AI).
    - Thread scheduler hourly tunggal untuk evaluasi semua zona.
    - Thread-safe untuk semua operasi.
    """

    def __init__(self):
        # Semua zona: key = zone_id
        self._zones: Dict[str, ZoneConfig] = {}
        # Tracker per zona (current cycle): key = zone_id
        self._trackers: Dict[str, ZoneContinuousTracker] = {}
        # YOLO model untuk zone detection (lazy loaded)
        self._yolo_model = None
        self._yolo_lock = threading.Lock()
        # Lock untuk operasi zone config
        self._lock = threading.RLock()
        # Database
        self._db = ZoneDatabase()
        # Frame cache per kamera: key = cam_id → latest frame
        self._frames: Dict[int, np.ndarray] = {}
        self._frames_lock = threading.Lock()
        # Frame terbaru per kamera untuk snapshot (annotated, setelah deteksi)
        self._annotated_frames: Dict[int, np.ndarray] = {}
        self._annotated_lock = threading.Lock()
        # Flag running
        self._running = False
        # Threads
        self._scheduler_thread: Optional[threading.Thread] = None
        self._detector_threads: Dict[int, threading.Thread] = {}
        # Label jam saat ini
        self._current_hour_label: str = self._get_hour_label(datetime.now())

    # ── Label jam: "2026-07-23 09:00"
    @staticmethod
    def _get_hour_label(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:00")

    # ── Start semua thread
    def start(self):
        if self._running:
            return
        self._running = True

        # Muat konfigurasi zona dari DB
        self._load_zones_from_db()

        # Inisialisasi tracker untuk jam saat ini
        self._reset_trackers()

        # Single unified detector thread
        self._main_detector_thread = threading.Thread(
            target=self._main_detector_loop,
            name="ZoneMainDetector",
            daemon=True
        )
        self._main_detector_thread.start()

        # Scheduler thread (cek jam bulat berdasarkan start_hour)
        self._scheduler_thread = threading.Thread(
            target=self._hourly_scheduler_loop,
            name="ZoneScheduler",
            daemon=True
        )
        self._scheduler_thread.start()

        print("[ZONE-MONITOR] Started. Single detector and scheduler threads running.", flush=True)

    def stop(self):
        self._running = False
        print("[ZONE-MONITOR] Stopped.", flush=True)

    # ── Load konfigurasi dari DB
    def _load_zones_from_db(self):
        zones = self._db.load_all_zones()
        with self._lock:
            self._zones = {z.zone_id: z for z in zones}
        print(f"[ZONE-MONITOR] Loaded {len(zones)} zone(s) from DB.", flush=True)

    # ── Reset/buat ulang trackers untuk siklus jam baru
    def _reset_trackers(self, hour_label: Optional[str] = None):
        if hour_label is None:
            hour_label = self._get_hour_label(datetime.now())
        with self._lock:
            self._current_hour_label = hour_label
            for zone_id, zone in self._zones.items():
                self._trackers[zone_id] = ZoneContinuousTracker(
                    hour_label,
                    grace_period_seconds=zone.grace_period_seconds
                )
        print(f"[ZONE-MONITOR] Trackers reset for cycle: {hour_label}", flush=True)

    # ── Feed frame dari camera stream (dipanggil dari luar)
    def feed_frame(self, cam_id: int, frame: np.ndarray):
        """
        Terima frame dari camera reader.
        Dipanggil dari ai_service.py mjpeg_generator atau dedicated loop.
        Thread-safe.
        """
        if frame is None:
            return
        with self._frames_lock:
            self._frames[cam_id] = (frame, time.time())

    # ── Single Thread Detector Loop untuk Semua Kamera dengan Zona Aktif
    def _main_detector_loop(self):
        """
        Thread tunggal hemat CPU: iterasi setiap kamera yang memiliki zona aktif,
        jalankan YOLO nano person detection, dan update tracker.
        """
        print("[ZONE-DETECTOR] Single main detector loop started.", flush=True)
        INTERVAL = 0.8  # ~1.25 FPS - efficient, responsive & keeps CPU free
        _debug_counter = 0

        while self._running:
            try:
                # Ambil daftar unik camera ID yang memiliki zona aktif
                with self._lock:
                    active_cam_ids = sorted(list(set(z.cam_id for z in self._zones.values())))

                if not active_cam_ids:
                    time.sleep(1.0)
                    continue

                _debug_counter += 1
                do_debug = (_debug_counter % 25 == 1)

                now = time.time()

                for cam_id in active_cam_ids:
                    with self._lock:
                        cam_zones = [z for z in self._zones.values() if z.cam_id == cam_id]
                    if not cam_zones:
                        continue

                    # Ambil frame terbaru kamera ini
                    with self._frames_lock:
                        frame_entry = self._frames.get(cam_id)

                    if not frame_entry:
                        continue

                    frame, frame_ts = frame_entry
                    if now - frame_ts > 6.0:
                        # Skip frame basi jika stream kamera terhenti/jeda > 6 detik
                        continue

                    frame = frame.copy()
                    h, w = frame.shape[:2]

                    # Deteksi person
                    person_bboxes = self._detect_persons(frame)

                    # Yield GIL briefly after PyTorch inference so FastAPI threads execute instantly
                    time.sleep(0.05)

                    # Update tracker zona kamera ini
                    with self._lock:
                        for zone in cam_zones:
                            tracker = self._trackers.get(zone.zone_id)
                            if tracker is None:
                                continue

                            is_present = any(
                                is_person_in_zone(bbox, zone.coords, w, h)
                                for bbox in person_bboxes
                            )

                            if do_debug:
                                print(f"  [ZONE-DEBUG] cam_{cam_id} | zone='{zone.name}' | "
                                      f"is_present={is_present} | accum={tracker.accumulated_minutes:.2f}m", flush=True)

                            tracker.update(is_present, now)

                    # Simpan annotated frame untuk snapshot Telegram
                    try:
                        ann_frame = frame.copy()
                        with self._lock:
                            all_zones_cam = [z for z in self._zones.values() if z.cam_id == cam_id]
                            trackers_snap = {k: v for k, v in self._trackers.items()}
                        ann_frame = draw_zones_on_frame(ann_frame, all_zones_cam, trackers_snap, person_bboxes)
                        with self._annotated_lock:
                            self._annotated_frames[cam_id] = ann_frame
                    except Exception:
                        pass

            except Exception as e:
                print(f"[ZONE-DETECTOR] Error in main loop: {e}", flush=True)

            time.sleep(INTERVAL)

        print("[ZONE-DETECTOR] Single main detector loop stopped.", flush=True)

    # ── YOLO nano person detection (lazy load)
    def _detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Deteksi person menggunakan YOLO nano (model ringan).
        Return list of (x1, y1, x2, y2).
        """
        try:
            model = self._get_yolo_model()
            if model is None:
                return []

            import torch
            with self._yolo_lock:
                with torch.no_grad():
                    try:
                        results = model(frame, imgsz=416, verbose=False,
                                       conf=0.22, classes=[COCO_PERSON])
                    except Exception:
                        results = model(frame, imgsz=416, verbose=False,
                                       conf=0.22, classes=[COCO_PERSON])

            bboxes = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls != COCO_PERSON:
                        continue
                    conf = float(box.conf[0])
                    if conf < 0.22:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bboxes.append((x1, y1, x2, y2))
            return bboxes

        except Exception as e:
            print(f"[ZONE-DETECTOR] YOLO error: {e}", flush=True)
            return []

    def _get_yolo_model(self):
        """Lazy load YOLO nano model untuk zone detection (thread-safe)."""
        if self._yolo_model is not None:
            return self._yolo_model
        with self._yolo_lock:
            if self._yolo_model is None:
                try:
                    from analytics_engine import yolo_model, load_yolo_model
                    m = yolo_model
                    if m is None:
                        m = load_yolo_model()
                    self._yolo_model = m
                    print("[ZONE-MONITOR] YOLO nano model loaded for zone detection.", flush=True)
                except Exception as e:
                    print(f"[ZONE-MONITOR] Failed to load YOLO model: {e}", flush=True)
        return self._yolo_model

    # ── Ambil snapshot frame ter-annotated untuk kamera
    def _get_snapshot_jpeg(self, cam_id: int) -> Optional[bytes]:
        """
        Ambil snapshot JPEG dari frame ter-annotated terakhir kamera ini.
        Digunakan untuk dilampirkan ke notifikasi Telegram.
        """
        with self._annotated_lock:
            frame = self._annotated_frames.get(cam_id)
        if frame is None:
            # Fallback: coba ambil frame mentah
            with self._frames_lock:
                frame = self._frames.get(cam_id)
        if frame is None:
            return None
        try:
            # Resize ke 640 lebar maksimal untuk menjaga ukuran file Telegram
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / w
                frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return enc.tobytes() if ok else None
        except Exception as e:
            print(f"[ZONE-MONITOR] Snapshot encode error: {e}", flush=True)
            return None

    # ── Scheduler: tunggu jam bulat berdasarkan start_hour zona
    def _hourly_scheduler_loop(self):
        print("[ZONE-SCHEDULER] Hourly scheduler started.", flush=True)
        while self._running:
            try:
                now = datetime.now()
                # Hitung detik ke jam bulat berikutnya
                next_hour = (now + timedelta(hours=1)).replace(
                    minute=0, second=0, microsecond=0)
                sleep_secs = (next_hour - now).total_seconds()

                print(
                    f"[ZONE-SCHEDULER] Next evaluation at {next_hour.strftime('%H:%M:%S')} "
                    f"({sleep_secs/60:.1f} menit lagi).",
                    flush=True
                )
                # Sleep sampai mendekati jam bulat, cek setiap 10 detik
                while self._running and (datetime.now() < next_hour - timedelta(seconds=2)):
                    time.sleep(10)

                if not self._running:
                    break

                # Tunggu persis di jam bulat
                remaining = (next_hour - datetime.now()).total_seconds()
                if remaining > 0:
                    time.sleep(remaining)

                # Evaluasi siklus yang baru saja selesai
                prev_hour_label = self._get_hour_label(next_hour - timedelta(hours=1))
                print(
                    f"[ZONE-SCHEDULER] Evaluating cycle: {prev_hour_label}",
                    flush=True
                )

                # Hanya evaluasi zona yang start_hour-nya relevan dengan jam ini
                finished_hour = (next_hour - timedelta(hours=1)).hour
                self._evaluate_all_zones(prev_hour_label, finished_hour=finished_hour)

                # Reset tracker untuk siklus baru
                new_hour_label = self._get_hour_label(next_hour)
                self._reset_trackers(new_hour_label)

            except Exception as e:
                print(f"[ZONE-SCHEDULER] Error: {e}", flush=True)
                time.sleep(60)

    # ── Evaluasi semua zona setelah siklus jam selesai
    def _evaluate_all_zones(self, cycle_label: str, finished_hour: Optional[int] = None):
        with self._lock:
            zones_snapshot = list(self._zones.values())
            trackers_snapshot = {k: v for k, v in self._trackers.items()}

        try:
            from telegram_notifier import send_zone_alert
        except Exception:
            send_zone_alert = None

        for zone in zones_snapshot:
            # Cek apakah zona ini aktif di jam yang baru selesai
            # Hanya evaluasi jika jam selesai >= start_hour zona
            if finished_hour is not None:
                try:
                    clean_h = str(zone.start_hour).replace(".", ":").split(":")[0]
                    start_h = int(clean_h)
                    if finished_hour < start_h:
                        # Jam belum dalam jam operasional zona ini, skip
                        continue
                except Exception:
                    pass

            tracker = trackers_snapshot.get(zone.zone_id)
            if tracker is None:
                continue

            accum_min = tracker.accumulated_minutes
            threshold = zone.threshold_minutes
            alert_type = None

            if accum_min == 0.0:
                alert_type = "no_presence"
            elif accum_min < threshold:
                alert_type = "low_presence"
            # else: OK, tidak perlu notifikasi

            # Log event ke DB
            try:
                self._db.log_event(
                    zone=zone,
                    cycle_label=cycle_label,
                    accumulated_minutes=accum_min,
                    alert_type=alert_type,
                    telegram_sent=False
                )
            except Exception as e:
                print(f"[ZONE-MONITOR] DB log error for {zone.zone_id}: {e}", flush=True)

            # Kirim notifikasi jika perlu
            if alert_type and zone.telegram_enabled and send_zone_alert:
                try:
                    # Ambil snapshot frame untuk dilampirkan ke Telegram
                    snapshot_bytes = self._get_snapshot_jpeg(zone.cam_id)

                    sent = send_zone_alert(
                        zone_name=zone.name,
                        cam_id=zone.cam_id,
                        cycle_label=cycle_label,
                        accumulated_minutes=accum_min,
                        threshold_minutes=threshold,
                        alert_type=alert_type,
                        image_bytes=snapshot_bytes,
                    )
                    if sent:
                        try:
                            conn = self._db._connect()
                            conn.execute(
                                """UPDATE zone_events SET telegram_sent = 1
                                   WHERE zone_id = ? AND cycle_label = ?
                                   ORDER BY id DESC LIMIT 1""",
                                (zone.zone_id, cycle_label)
                            )
                            conn.commit()
                            conn.close()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[ZONE-MONITOR] Telegram send error for {zone.zone_id}: {e}", flush=True)

            status = f"{accum_min:.1f}/{threshold}m"
            print(
                f"[ZONE-EVAL] {cycle_label} | Zona '{zone.name}' (cam{zone.cam_id}) | "
                f"Akumulasi: {status} | Alert: {alert_type or 'OK'}",
                flush=True
            )

    # ══════════════════════════════
    #  Public API — Manajemen Zona
    # ══════════════════════════════

    def set_zone(self, zone: ZoneConfig):
        """Tambah atau update zona. Simpan ke DB dan aktifkan tracker."""
        self._db.save_zone(zone)
        with self._lock:
            self._zones[zone.zone_id] = zone
            if zone.zone_id not in self._trackers:
                self._trackers[zone.zone_id] = ZoneContinuousTracker(
                    self._current_hour_label,
                    grace_period_seconds=zone.grace_period_seconds
                )
            else:
                self._trackers[zone.zone_id].grace_period_seconds = zone.grace_period_seconds
        print(f"[ZONE-MONITOR] Zone set: {zone.zone_id} '{zone.name}' cam{zone.cam_id} (grace={zone.grace_period_seconds}s)", flush=True)

    def delete_zone(self, zone_id: str) -> bool:
        self._db.delete_zone(zone_id)
        with self._lock:
            removed = zone_id in self._zones
            self._zones.pop(zone_id, None)
            self._trackers.pop(zone_id, None)
        if removed:
            print(f"[ZONE-MONITOR] Zone deleted: {zone_id}", flush=True)
        return removed

    def get_zones(self, cam_id: Optional[int] = None) -> List[ZoneConfig]:
        with self._lock:
            if cam_id is not None:
                return [z for z in self._zones.values() if z.cam_id == cam_id]
            return list(self._zones.values())

    def get_zone_status(self, cam_id: Optional[int] = None) -> List[dict]:
        """Status real-time semua zona (akumulasi saat ini)."""
        with self._lock:
            zones = [z for z in self._zones.values()
                     if cam_id is None or z.cam_id == cam_id]
            result = []
            for zone in zones:
                tracker = self._trackers.get(zone.zone_id)
                snap = tracker.snapshot() if tracker else {}
                result.append({
                    **zone.to_dict(),
                    "current_cycle": snap,
                    "is_ok": (snap.get("accumulated_minutes", 0) >= zone.threshold_minutes),
                    "is_person_present": snap.get("is_person_present", False),
                })
            return result

    def get_history(self, cam_id: Optional[int] = None,
                    zone_id: Optional[str] = None,
                    limit: int = 100) -> List[dict]:
        return self._db.get_events(cam_id=cam_id, zone_id=zone_id, limit=limit)

    def trigger_test_evaluation(self) -> dict:
        """Trigger evaluasi manual (untuk testing/debug tanpa harus tunggu jam bulat)."""
        cycle_label = self._get_hour_label(datetime.now()) + " [TEST]"
        self._evaluate_all_zones(cycle_label)
        return {"success": True, "cycle_label": cycle_label}

    def get_frame_with_zones(self, cam_id: int) -> Optional[np.ndarray]:
        """
        Return frame terbaru dengan overlay zona digambar di atasnya.
        Digunakan oleh zone monitoring panel di UI atau mode AI zone_monitor.
        """
        with self._annotated_lock:
            frame = self._annotated_frames.get(cam_id)
        if frame is not None:
            return frame.copy()
        # Fallback ke raw frame + draw
        with self._frames_lock:
            frame = self._frames.get(cam_id)
        if frame is None:
            return None
        frame = frame.copy()
        with self._lock:
            cam_zones = [z for z in self._zones.values() if z.cam_id == cam_id]
            trackers = {k: v for k, v in self._trackers.items()}
        return draw_zones_on_frame(frame, cam_zones, trackers)


# ═══════════════════════════════════════════════════════
#  Singleton Global Instance
# ═══════════════════════════════════════════════════════

_zone_monitor_instance: Optional[ZoneMonitor] = None
_instance_lock = threading.Lock()


def get_zone_monitor() -> ZoneMonitor:
    """Dapatkan atau buat singleton ZoneMonitor."""
    global _zone_monitor_instance
    if _zone_monitor_instance is None:
        with _instance_lock:
            if _zone_monitor_instance is None:
                _zone_monitor_instance = ZoneMonitor()
                _zone_monitor_instance.start()
    return _zone_monitor_instance
