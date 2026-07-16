import os
# Set global environment options for OpenCV FFmpeg capture to force TCP and disable buffering
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;100000"

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
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        print(f"[CAMERA-STREAM] camera_{self.cam_id} thread started.", flush=True)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3.0)
        print(f"[CAMERA-STREAM] camera_{self.cam_id} thread stopped.", flush=True)

    def _reader_loop(self):
        print(f"[CAMERA-STREAM] Starting reader loop for camera_{self.cam_id}: {self.source_url}", flush=True)
        cap = cv2.VideoCapture(self.source_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fail_count = 0
        skip_count = 0
        while self.running:
            ok, frame = cap.read()
            if not ok:
                fail_count += 1
                if fail_count > 30:
                    print(f"[CAMERA-STREAM] camera_{self.cam_id} offline. Reconnecting...", flush=True)
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(self.source_url)
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
                self.frame = frame.copy()
        cap.release()
        print(f"[CAMERA-STREAM] Reader loop stopped for camera_{self.cam_id}", flush=True)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def set_mode(self, mode, selected_classes=None):
        with self.lock:
            self.mode = mode
            self.selected_classes = selected_classes
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
