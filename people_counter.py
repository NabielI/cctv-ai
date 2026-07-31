# people_counter.py
# Real-time People Counting Engine with configurable IN/OUT line, ByteTrack, snapshot capture, SQLite logging.
# CRITICAL: Import torch BEFORE cv2 to prevent OpenMP conflicts on ARM64!
import torch
import torchvision

import cv2
import numpy as np
import time
import sqlite3
import threading
import os
import json
from datetime import datetime

# ── Constants ──────────────────────────────────────────────────────────────
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "snapshots")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "people_counter.db")
CROSSING_COOLDOWN = 1.5   # seconds per track_id before it can be counted again
MAX_SNAPSHOTS_PER_CAM = 500  # max snapshots stored per camera (oldest deleted)

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ── Database Initialization ─────────────────────────────────────────────────
def init_counter_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crossing_events (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id         INTEGER NOT NULL,
            track_id       INTEGER,
            direction      TEXT NOT NULL,
            timestamp      TEXT NOT NULL,
            snapshot_path  TEXT,
            crossing_count INTEGER DEFAULT 1
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hourly_summary (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id    INTEGER NOT NULL,
            hour      TEXT NOT NULL,
            in_count  INTEGER DEFAULT 0,
            out_count INTEGER DEFAULT 0,
            UNIQUE(cam_id, hour)
        )
    """)
    # Track-level accumulation: total crossings per (cam_id, track_id)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS track_stats (
            cam_id      INTEGER NOT NULL,
            track_id    INTEGER NOT NULL,
            in_count    INTEGER DEFAULT 0,
            out_count   INTEGER DEFAULT 0,
            first_seen  TEXT,
            last_seen   TEXT,
            PRIMARY KEY (cam_id, track_id)
        )
    """)
    conn.commit()
    conn.close()

init_counter_db()


def log_crossing_event(cam_id, track_id, direction, snapshot_path, crossing_count):
    """Log a single crossing event to SQLite. Never deletes any data."""
    try:
        ts = datetime.now().isoformat()
        hour_key = datetime.now().strftime("%Y-%m-%d %H:00")
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # Insert raw event
        cur.execute(
            "INSERT INTO crossing_events (cam_id, track_id, direction, timestamp, snapshot_path, crossing_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cam_id, track_id, direction, ts, snapshot_path, crossing_count)
        )

        # Upsert hourly summary
        if direction == "in":
            cur.execute(
                "INSERT INTO hourly_summary (cam_id, hour, in_count, out_count) VALUES (?, ?, 1, 0) "
                "ON CONFLICT(cam_id, hour) DO UPDATE SET in_count = in_count + 1",
                (cam_id, hour_key)
            )
        else:
            cur.execute(
                "INSERT INTO hourly_summary (cam_id, hour, in_count, out_count) VALUES (?, ?, 0, 1) "
                "ON CONFLICT(cam_id, hour) DO UPDATE SET out_count = out_count + 1",
                (cam_id, hour_key)
            )

        # Upsert track stats
        now_str = datetime.now().isoformat()
        if direction == "in":
            cur.execute(
                "INSERT INTO track_stats (cam_id, track_id, in_count, out_count, first_seen, last_seen) "
                "VALUES (?, ?, 1, 0, ?, ?) "
                "ON CONFLICT(cam_id, track_id) DO UPDATE SET in_count = in_count + 1, last_seen = ?",
                (cam_id, track_id, now_str, now_str, now_str)
            )
        else:
            cur.execute(
                "INSERT INTO track_stats (cam_id, track_id, in_count, out_count, first_seen, last_seen) "
                "VALUES (?, ?, 0, 1, ?, ?) "
                "ON CONFLICT(cam_id, track_id) DO UPDATE SET out_count = out_count + 1, last_seen = ?",
                (cam_id, track_id, now_str, now_str, now_str)
            )

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[PEOPLE-COUNTER] DB log error: {e}", flush=True)


def save_snapshot(cam_id, frame, direction, track_id, crossing_count):
    """Crop and save frame snapshot. Returns relative URL path or None."""
    try:
        cam_snap_dir = os.path.join(SNAPSHOT_DIR, f"cam_{cam_id}")
        os.makedirs(cam_snap_dir, exist_ok=True)

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{direction}_{track_id}_{ts_str}.jpg"
        full_path = os.path.join(cam_snap_dir, filename)

        # Draw info overlay on snapshot copy
        snap = frame.copy()
        h, w = snap.shape[:2]
        label = f"{'MASUK' if direction == 'in' else 'KELUAR'} | ID:{track_id} | x{crossing_count}"
        badge_color = (34, 197, 94) if direction == "in" else (239, 68, 68)
        cv2.rectangle(snap, (0, 0), (w, 30), badge_color, -1)
        cv2.putText(snap, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        ok, enc = cv2.imencode(".jpg", snap, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with open(full_path, "wb") as f:
                f.write(enc.tobytes())

        # Prune oldest snapshots if over limit
        all_snaps = sorted(os.listdir(cam_snap_dir))
        if len(all_snaps) > MAX_SNAPSHOTS_PER_CAM:
            for old in all_snaps[:len(all_snaps) - MAX_SNAPSHOTS_PER_CAM]:
                try:
                    os.remove(os.path.join(cam_snap_dir, old))
                except Exception:
                    pass

        return f"/uploads/snapshots/cam_{cam_id}/{filename}"
    except Exception as e:
        print(f"[PEOPLE-COUNTER] Snapshot save error: {e}", flush=True)
        return None


# ── Line Crossing Math ──────────────────────────────────────────────────────
def cross_sign(x1, y1, x2, y2, px, py):
    """Returns + if point is left of line, - if right. 0 = on line."""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


# ── Per-Camera Counter State ────────────────────────────────────────────────
class PeopleCounter:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.lock = threading.Lock()

        # Line in normalized coords [0,1]
        self.line = None   # {"x1": 0.1, "y1": 0.5, "x2": 0.9, "y2": 0.5}

        # Persistent cumulative counters (never decrease, no limit)
        self.in_count = self._load_total_from_db("in")
        self.out_count = self._load_total_from_db("out")

        # Total unique track IDs ever seen crossing
        self.unique_persons = self._load_unique_persons()

        # Per-track state for crossing detection
        # track_id -> {"last_sign": +1/-1, "last_cross_time": float, "total_crossings": int}
        self.track_states = {}

        # Latest crossing event (for SSE push)
        self.last_event = None          # dict
        self.event_listeners = []       # list of queue.Queue

        # Latest frame with overlay (for MJPEG)
        self.latest_frame_bytes = None

        print(f"[PEOPLE-COUNTER] cam_{cam_id} initialized. IN={self.in_count} OUT={self.out_count}", flush=True)

    def _load_total_from_db(self, direction):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                "SELECT COALESCE(SUM(CASE WHEN direction=? THEN 1 ELSE 0 END), 0) FROM crossing_events WHERE cam_id=?",
                (direction, self.cam_id)
            )
            val = cur.fetchone()[0]
            conn.close()
            return int(val)
        except Exception:
            return 0

    def _load_unique_persons(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(DISTINCT track_id) FROM track_stats WHERE cam_id=? AND track_id IS NOT NULL",
                (self.cam_id,)
            )
            val = cur.fetchone()[0]
            conn.close()
            return int(val)
        except Exception:
            return 0

    def set_line(self, x1, y1, x2, y2):
        """Set counting line in normalized coordinates [0,1]."""
        with self.lock:
            self.line = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
            # Reset per-track crossing state when line changes (don't reset counts!)
            self.track_states = {}
        print(f"[PEOPLE-COUNTER] cam_{self.cam_id} line set: ({x1:.2f},{y1:.2f}) -> ({x2:.2f},{y2:.2f})", flush=True)

    def get_line(self):
        with self.lock:
            return dict(self.line) if self.line else None

    def get_status(self):
        with self.lock:
            return {
                "cam_id": self.cam_id,
                "in_count": self.in_count,
                "out_count": self.out_count,
                "unique_persons": self.unique_persons,
                "line": dict(self.line) if self.line else None,
                "timestamp": datetime.now().isoformat()
            }

    def process_frame(self, frame):
        """
        Run YOLO tracking on frame, detect line crossings, update counters.
        Returns annotated frame (np.ndarray).
        """
        with self.lock:
            line = self.line

        if frame is None:
            return frame

        h, w = frame.shape[:2]
        out = frame.copy()

        # ── Draw line on frame ────────────────────────────────────────────
        if line:
            lx1 = int(line["x1"] * w)
            ly1 = int(line["y1"] * h)
            lx2 = int(line["x2"] * w)
            ly2 = int(line["y2"] * h)

            # Draw thick line with gradient arrow
            cv2.line(out, (lx1, ly1), (lx2, ly2), (255, 200, 0), 3, cv2.LINE_AA)
            # Draw direction indicators (small perpendicular arrows)
            mid_x = (lx1 + lx2) // 2
            mid_y = (ly1 + ly2) // 2
            dx = lx2 - lx1
            dy = ly2 - ly1
            length = max(1, int((dx**2 + dy**2) ** 0.5))
            # Normal vector (left side = IN by convention)
            nx, ny = -dy / length, dx / length
            # IN arrow (left normal)
            in_tip = (int(mid_x + nx * 30), int(mid_y + ny * 30))
            out_tip = (int(mid_x - nx * 30), int(mid_y - ny * 30))
            cv2.arrowedLine(out, (mid_x, mid_y), in_tip, (34, 197, 94), 2, cv2.LINE_AA, tipLength=0.4)
            cv2.arrowedLine(out, (mid_x, mid_y), out_tip, (239, 68, 68), 2, cv2.LINE_AA, tipLength=0.4)
            cv2.putText(out, "IN", in_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (34, 197, 94), 2, cv2.LINE_AA)
            cv2.putText(out, "OUT", out_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (239, 68, 68), 2, cv2.LINE_AA)

            # Draw endpoints
            cv2.circle(out, (lx1, ly1), 6, (255, 200, 0), -1, cv2.LINE_AA)
            cv2.circle(out, (lx2, ly2), 6, (255, 200, 0), -1, cv2.LINE_AA)

        # ── Draw IN/OUT counter overlay ────────────────────────────────────
        cv2.rectangle(out, (0, 0), (220, 58), (15, 23, 42), -1)
        cv2.putText(out, f"IN : {self.in_count}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (34, 197, 94), 2, cv2.LINE_AA)
        cv2.putText(out, f"OUT: {self.out_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (239, 68, 68), 2, cv2.LINE_AA)
        cv2.putText(out, f"UNIK: {self.unique_persons}", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 204, 21), 1, cv2.LINE_AA)

        if line is None:
            cv2.putText(out, "Belum ada garis – atur di UI", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
            return out

        # ── YOLO ByteTrack Inference ────────────────────────────────────────
        from analytics_engine import load_yolo_model, _yolo_lock
        model = load_yolo_model()
        if model is None:
            return out

        tracked_objects = []
        try:
            with _yolo_lock:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],   # person only
                    conf=0.25,
                    imgsz=320,
                    verbose=False
                )
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    if box.id is None:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    track_id = int(box.id[0])
                    conf = float(box.conf[0])
                    tracked_objects.append((x1, y1, x2, y2, track_id, conf))
        except Exception as e:
            print(f"[PEOPLE-COUNTER] Tracking error cam_{self.cam_id}: {e}", flush=True)
            return out

        # ── Crossing Detection ──────────────────────────────────────────────
        lx1_n = line["x1"] * w
        ly1_n = line["y1"] * h
        lx2_n = line["x2"] * w
        ly2_n = line["y2"] * h

        now = time.time()
        new_events = []

        for (x1, y1, x2, y2, track_id, conf) in tracked_objects:
            # Centroid: center-x, bottom-y of bbox
            cx = (x1 + x2) / 2.0
            cy = float(y2)  # bottom of bbox — more stable for crossing

            sign_now = cross_sign(lx1_n, ly1_n, lx2_n, ly2_n, cx, cy)
            sign_class = 1 if sign_now >= 0 else -1

            tstate = self.track_states.get(track_id)

            if tstate is None:
                # First time seeing this track — initialize without counting
                self.track_states[track_id] = {
                    "last_sign": sign_class,
                    "last_cross_time": 0.0,
                    "total_crossings": 0
                }
            else:
                prev_sign = tstate["last_sign"]
                cooldown_ok = (now - tstate["last_cross_time"]) >= CROSSING_COOLDOWN

                if prev_sign != sign_class and cooldown_ok:
                    # Crossing detected!
                    tstate["total_crossings"] += 1
                    tstate["last_cross_time"] = now
                    tstate["last_sign"] = sign_class

                    # Determine IN or OUT direction:
                    # positive→negative = OUT, negative→positive = IN
                    if prev_sign > 0 and sign_class < 0:
                        direction = "out"
                    else:
                        direction = "in"

                    # Snapshot of full frame at crossing moment
                    snap_path = save_snapshot(self.cam_id, out, direction, track_id, tstate["total_crossings"])

                    # Log to DB
                    log_crossing_event(self.cam_id, track_id, direction, snap_path, tstate["total_crossings"])

                    # Update in-memory counters (monotonically increasing, no cap)
                    with self.lock:
                        if direction == "in":
                            self.in_count += 1
                        else:
                            self.out_count += 1

                        # Track unique persons (by track_id)
                        if track_id not in {ts for ts in self.track_states if self.track_states[ts]["total_crossings"] > 0}:
                            self.unique_persons += 1

                        # Recalculate unique from DB periodically (every 20 events)
                        total_events = self.in_count + self.out_count
                        if total_events % 20 == 0:
                            self.unique_persons = self._load_unique_persons()

                    event = {
                        "type": "crossing",
                        "cam_id": self.cam_id,
                        "track_id": track_id,
                        "direction": direction,
                        "in_count": self.in_count,
                        "out_count": self.out_count,
                        "unique_persons": self.unique_persons,
                        "crossing_count": tstate["total_crossings"],
                        "snapshot_path": snap_path,
                        "timestamp": datetime.now().isoformat()
                    }
                    new_events.append(event)

                    print(f"[PEOPLE-COUNTER] cam_{self.cam_id} | {direction.upper()} | track_id={track_id} | "
                          f"crossing_count={tstate['total_crossings']} | IN={self.in_count} OUT={self.out_count}", flush=True)
                else:
                    # Update sign even if not crossing
                    tstate["last_sign"] = sign_class

            # Draw bounding box on frame
            color = (34, 197, 94) if sign_class > 0 else (239, 68, 68)
            ts_info = self.track_states.get(track_id, {})
            total_x = ts_info.get("total_crossings", 0)
            lbl = f"ID:{track_id} ({int(conf*100)}%) x{total_x}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, lbl, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            # Center dot
            cv2.circle(out, (int(cx), int(cy)), 4, (255, 255, 0), -1, cv2.LINE_AA)

        # Broadcast events to SSE listeners
        for ev in new_events:
            with self.lock:
                self.last_event = ev
                dead = []
                for q in self.event_listeners:
                    try:
                        q.put_nowait(ev)
                    except Exception:
                        dead.append(q)
                for q in dead:
                    self.event_listeners.remove(q)

        # Cache latest frame for MJPEG
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            with self.lock:
                self.latest_frame_bytes = enc.tobytes()

        return out

    def subscribe_events(self, q):
        """Register a queue to receive crossing events (SSE)."""
        with self.lock:
            self.event_listeners.append(q)

    def unsubscribe_events(self, q):
        with self.lock:
            if q in self.event_listeners:
                self.event_listeners.remove(q)


# ── Global Registry ─────────────────────────────────────────────────────────
_counters: dict[int, PeopleCounter] = {}
_counters_lock = threading.Lock()


def get_counter(cam_id: int) -> PeopleCounter:
    """Get or create PeopleCounter for given camera."""
    with _counters_lock:
        if cam_id not in _counters:
            _counters[cam_id] = PeopleCounter(cam_id)
        return _counters[cam_id]


# ── Query helpers ─────────────────────────────────────────────────────────────
def get_hourly_data(cam_id: int, date_str: str = None):
    """Returns list of {hour, in_count, out_count} for today (or given date YYYY-MM-DD)."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT hour, in_count, out_count FROM hourly_summary "
            "WHERE cam_id=? AND hour LIKE ? ORDER BY hour ASC",
            (cam_id, f"{date_str}%")
        )
        rows = cur.fetchall()
        conn.close()
        return [{"hour": r[0], "in_count": r[1], "out_count": r[2]} for r in rows]
    except Exception as e:
        print(f"[PEOPLE-COUNTER] Hourly query error: {e}", flush=True)
        return []


def get_snapshots(cam_id: int, limit: int = 50):
    """Returns recent crossing snapshots with per-track crossing_count info."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT e.id, e.track_id, e.direction, e.timestamp, e.snapshot_path, e.crossing_count, "
            "       t.in_count, t.out_count "
            "FROM crossing_events e "
            "LEFT JOIN track_stats t ON e.cam_id=t.cam_id AND e.track_id=t.track_id "
            "WHERE e.cam_id=? AND e.snapshot_path IS NOT NULL "
            "ORDER BY e.id DESC LIMIT ?",
            (cam_id, limit)
        )
        rows = cur.fetchall()
        conn.close()
        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "track_id": r[1],
                "direction": r[2],
                "timestamp": r[3],
                "snapshot_path": r[4],
                "crossing_count": r[5],       # how many times THIS event
                "total_in": r[6] or 0,        # lifetime IN for this track_id
                "total_out": r[7] or 0,       # lifetime OUT for this track_id
                "total_crossings": (r[6] or 0) + (r[7] or 0)  # bolak-balik count
            })
        return result
    except Exception as e:
        print(f"[PEOPLE-COUNTER] Snapshot query error: {e}", flush=True)
        return []


def get_track_stats(cam_id: int, limit: int = 100):
    """Returns track-level statistics (bolak-balik analysis)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT track_id, in_count, out_count, first_seen, last_seen "
            "FROM track_stats WHERE cam_id=? ORDER BY (in_count+out_count) DESC LIMIT ?",
            (cam_id, limit)
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "track_id": r[0],
                "in_count": r[1],
                "out_count": r[2],
                "total_crossings": r[1] + r[2],
                "first_seen": r[3],
                "last_seen": r[4]
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[PEOPLE-COUNTER] Track stats error: {e}", flush=True)
        return []
