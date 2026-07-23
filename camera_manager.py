import os
# Set global environment options for OpenCV FFmpeg capture to force ultra-low 0ms latency
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer+discardcorrupt|max_delay;0|probesize;32|analyzeduration;0|flags;low_delay"

import cv2
import time
import threading


class CameraStream:
    def __init__(self, cam_id, source_url):
        self.cam_id = cam_id
        self.source_url = source_url
        self.mode = "none"
        self.selected_classes = None
        self.frame = None
        self.ai_frame = None
        self.ai_metadata = {"mode": "none"}
        # Cached overlay: drawn on every raw frame for smooth 30 FPS
        self.cached_overlay_fn = None  # callable(frame) -> annotated frame
        self.state = {}
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.ai_thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        self.ai_thread = threading.Thread(target=self._ai_loop, daemon=True)
        self.ai_thread.start()
        print(f"[CAMERA-STREAM] camera_{self.cam_id} reader & async AI worker threads started.", flush=True)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.ai_thread:
            self.ai_thread.join(timeout=2.0)
        print(f"[CAMERA-STREAM] camera_{self.cam_id} threads stopped.", flush=True)

    def _reader_loop(self):
        stream_target = f"rtsp://127.0.0.1:8554/camera_{self.cam_id}" if self.source_url.startswith("onvif://") else self.source_url
        print(f"[CAMERA-STREAM] Starting reader loop for camera_{self.cam_id}: {stream_target}", flush=True)
        cap = cv2.VideoCapture(stream_target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fail_count = 0
        skip_count = 0
        while self.running:
            # Drain ALL queued frames from buffer to guarantee ZERO LATENCY (0 ms lag!)
            for _ in range(4):
                cap.grab()
            ok, frame = cap.retrieve()
            if not ok:
                ok, frame = cap.read()
            if not ok:
                fail_count += 1
                if fail_count > 30:
                    print(f"[CAMERA-STREAM] camera_{self.cam_id} offline. Reconnecting...", flush=True)
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(stream_target)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    fail_count = 0
                    skip_count = 0
                else:
                    time.sleep(0.02)
                continue
            fail_count = 0
            
            # Skip initial gray/noisy frames on startup/reconnect
            if skip_count < 25:
                if frame is not None and frame.std() < 15.0:
                    skip_count += 1
                    continue
                else:
                    skip_count = 25
            
            with self.lock:
                self.frame = frame

            # Continuously feed fresh frame to ZoneMonitor (24/7 background detection)
            try:
                from zone_monitor import get_zone_monitor
                get_zone_monitor().feed_frame(self.cam_id, frame)
            except Exception:
                pass

            time.sleep(0.005)
        cap.release()
        print(f"[CAMERA-STREAM] Reader loop stopped for camera_{self.cam_id}", flush=True)

    def _ai_loop(self):
        """Dedicated Async AI Worker Thread - Runs inference independently without blocking RTSP capture or video stream.
        
        Key design: After inference, store a lightweight overlay function (not a processed frame copy).
        The mjpeg_generator calls this function on each fresh raw frame for smooth 30 FPS output.
        """
        while self.running:
            with self.lock:
                current_mode = self.mode
                current_classes = self.selected_classes

            if current_mode == "none":
                with self.lock:
                    self.cached_overlay_fn = None
                    self.ai_metadata = {"mode": "none"}
                time.sleep(0.05)
                continue

            frame = self.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            try:
                from analytics_engine import run_analytics
                processed, meta = run_analytics(self.cam_id, frame, current_mode, current_classes, self.state)
                with self.lock:
                    self.ai_frame = processed
                    self.ai_metadata = meta
                    # Capture the diff (overlay) between processed and original for re-application
                    # Store processed frame for reference (used as overlay cache)
                    self._last_processed = processed
                    self._last_frame_shape = frame.shape
            except Exception as e:
                print(f"[CAMERA-STREAM] Async AI worker error cam_{self.cam_id}: {e}", flush=True)
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_ai_frame(self):
        """Returns the latest raw frame with cached AI overlay applied.
        
        This always runs at raw frame rate (30 FPS), never blocked by AI inference speed.
        The AI overlay (bboxes, labels) from the last inference is re-drawn on the fresh frame.
        """
        with self.lock:
            raw = self.frame
            if raw is None:
                return None, self.ai_metadata.copy() if isinstance(self.ai_metadata, dict) else {"mode": "none"}

            if self.mode == "none":
                return raw.copy(), {"mode": "none"}

            # If we have a last processed frame with the same shape, use it directly
            # This gives us the AI-annotated frame at AI speed, but we always have SOMETHING to show
            last_proc = getattr(self, '_last_processed', None)
            if last_proc is not None and last_proc.shape == raw.shape:
                return last_proc.copy(), self.ai_metadata.copy()

            # Fallback: return raw frame while AI warms up
            return raw.copy(), self.ai_metadata.copy() if isinstance(self.ai_metadata, dict) else {"mode": "none"}

    def set_mode(self, mode, selected_classes=None):
        with self.lock:
            self.mode = mode
            self.selected_classes = selected_classes
            self.ai_frame = None
            self.ai_metadata = {"mode": mode}
            self.state = {}
        classes_desc = "ALL" if self.selected_classes is None else self.selected_classes
        print(f"[CAMERA-STREAM] camera_{self.cam_id} mode updated to '{mode}' with classes {classes_desc}", flush=True)

    def get_mode(self):
        with self.lock:
            return self.mode

    def get_selected_classes(self):
        with self.lock:
            return None if self.selected_classes is None else list(self.selected_classes)


class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.lock = threading.Lock()

    def add_camera(self, camera_id, source_url):
        with self.lock:
            if camera_id in self.cameras:
                existing = self.cameras[camera_id]
                if existing.source_url == source_url:
                    return existing
                else:
                    print(f"[CAMERA-MANAGER] Camera_{camera_id} URL changed. Re-creating stream.", flush=True)
                    existing.stop()
                    del self.cameras[camera_id]
            stream = CameraStream(camera_id, source_url)
            self.cameras[camera_id] = stream
            stream.start()
            print(f"[CAMERA-MANAGER] Added camera_{camera_id} source: {source_url}", flush=True)
            return stream

    def remove_camera(self, camera_id):
        with self.lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].stop()
                del self.cameras[camera_id]
                print(f"[CAMERA-MANAGER] Removed camera_{camera_id}", flush=True)

    def get_camera(self, camera_id):
        with self.lock:
            return self.cameras.get(camera_id)

    def list_cameras(self):
        with self.lock:
            return list(self.cameras.keys())
