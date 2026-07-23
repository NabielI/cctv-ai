# CRITICAL: Import torch and torchvision BEFORE cv2 to prevent OpenMP conflict/runtime library errors on ARM64!
import torch
import torchvision
torch.set_num_threads(2)
print(f"[AI-ENGINE] Set PyTorch CPU threads to {torch.get_num_threads()}", flush=True)

import os
os.environ["ULTRALYTICS_TELEMETRY"] = "false"
os.environ["ULTRALYTICS_CHECK"] = "false"
os.environ["OPENVINO_TELEMETRY_OPTOUT"] = "1"
os.environ["OV_TELEMETRY_OPTOUT"] = "1"

import cv2
import numpy as np
import time
import sqlite3
import threading
import urllib.request
import importlib
import ctypes

try:
    import psutil
except ImportError:
    psutil = None

# ── MediaPipe Tasks API Windows compatibility monkeypatch for Python 3.12
orig_cdll = ctypes.CDLL
class PatchedCDLL(orig_cdll):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            if name == 'free' and os.name == 'nt':
                try:
                    msvcrt = orig_cdll('msvcrt')
                    return msvcrt.free
                except Exception:
                    pass
            raise e
ctypes.CDLL = PatchedCDLL

_yolo_lock = threading.RLock()

# Global model pointers and backend tracking status
yolo_model = None
yolo_model_backend = "PyTorch (CPU)"
yolo_model_name_str = "yolo26n"

yolo_model_heavy = None
yolo_model_heavy_backend = "PyTorch (CPU)"
yolo_model_heavy_name_str = "yolo26s"

yolo_pose_model = None
yolo_pose_backend = "PyTorch (CPU)"
yolo_pose_name_str = "yolo26n-pose"

def _find_model_path(name, fallback=None):
    if not name:
        return fallback
    if os.path.exists(name) and (os.path.isdir(name) or os.path.getsize(name) > 0):
        return name
    alt1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.exists(alt1) and (os.path.isdir(alt1) or os.path.getsize(alt1) > 0):
        return alt1
    alt2 = os.path.join(os.path.expanduser("~"), name)
    if os.path.exists(alt2) and (os.path.isdir(alt2) or os.path.getsize(alt2) > 0):
        return alt2
    return fallback

def _find_model_dir_or_pt(ov_primary, ov_secondary, pt_primary, pt_secondary):
    for ov_name in [ov_primary, ov_secondary]:
        p = _find_model_path(ov_name, None)
        if p and os.path.exists(p):
            return p, "OpenVINO"
    for pt_name in [pt_primary, pt_secondary]:
        p = _find_model_path(pt_name, None)
        if p and os.path.exists(p) and (os.path.isdir(p) or os.path.getsize(p) > 0):
            return p, "PyTorch"
    return pt_primary, "PyTorch"

def load_yolo_model():
    global yolo_model, yolo_model_backend, yolo_model_name_str
    if yolo_model is not None:
        return yolo_model
    with _yolo_lock:
        if yolo_model is None:
            ov_path, _ = _find_model_dir_or_pt("yolo26n_openvino_model", "yolov8n_openvino_model", "yolo26n.pt", "yolov8n.pt")
            pt_path = _find_model_path("yolo26n.pt", "yolov8n.pt")
            
            import importlib
            ultralytics = importlib.import_module("ultralytics")
            YOLO = ultralytics.YOLO
            try:
                from ultralytics import settings
                settings.update({'sync': False, 'check': False, 'telemetry': False})
            except Exception:
                pass
            import logging
            logging.getLogger("ultralytics").setLevel(logging.WARNING)

            loaded = False
            # Attempt 1: OpenVINO model folder
            if ov_path and os.path.exists(ov_path) and os.path.isdir(ov_path):
                try:
                    print(f"[AI-ENGINE] Loading OpenVINO Nano model: {ov_path}...", flush=True)
                    m = YOLO(ov_path)
                    _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                    m(_dummy, imgsz=320, verbose=False)
                    yolo_model = m
                    yolo_model_backend = "OpenVINO (CPU)"
                    yolo_model_name_str = os.path.basename(ov_path)
                    loaded = True
                    print(f"[AI-ENGINE] ✅ OpenVINO Nano model loaded successfully.", flush=True)
                except Exception as e:
                    print(f"[AI-ENGINE] ⚠️ OpenVINO Nano load failed ({e}), falling back to PyTorch .pt...", flush=True)

            # Attempt 2: PyTorch .pt fallback
            if not loaded:
                print(f"[AI-ENGINE] Loading PyTorch Nano model: {pt_path}...", flush=True)
                m = YOLO(pt_path)
                try:
                    _device = "cuda" if torch.cuda.is_available() else "cpu"
                    m.to(_device)
                except Exception:
                    pass
                _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                try:
                    m(_dummy, imgsz=320, verbose=False)
                except Exception:
                    pass
                yolo_model = m
                yolo_model_backend = "PyTorch (CPU)"
                yolo_model_name_str = os.path.basename(pt_path)
                print(f"[AI-ENGINE] ✅ PyTorch Nano model loaded successfully.", flush=True)

    return yolo_model

def load_yolo_heavy():
    """Load the 'heavy' detection model (yolo26s Small model).
    Provides high accuracy for vehicle tracking (cars, motorcycles, buses, trucks)
    and multi-class object/attribute detection.
    """
    global yolo_model_heavy, yolo_model_heavy_backend, yolo_model_heavy_name_str
    if yolo_model_heavy is not None:
        return yolo_model_heavy
    with _yolo_lock:
        if yolo_model_heavy is None:
            ov_path, _ = _find_model_dir_or_pt("yolo26s_openvino_model", "yolov8s_openvino_model", "yolo26s.pt", "yolov8s.pt")
            pt_path = _find_model_path("yolo26s.pt", "yolov8s.pt")

            import importlib
            ultralytics = importlib.import_module("ultralytics")
            YOLO = ultralytics.YOLO
            try:
                from ultralytics import settings
                settings.update({'sync': False, 'check': False, 'telemetry': False})
            except Exception:
                pass

            loaded = False
            if ov_path and os.path.exists(ov_path) and os.path.isdir(ov_path):
                try:
                    print(f"[AI-ENGINE] Loading Heavy OpenVINO model (yolo26s): {ov_path}...", flush=True)
                    m = YOLO(ov_path)
                    _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                    m(_dummy, imgsz=320, verbose=False)
                    yolo_model_heavy = m
                    yolo_model_heavy_backend = "OpenVINO (CPU)"
                    yolo_model_heavy_name_str = os.path.basename(ov_path)
                    loaded = True
                    print(f"[AI-ENGINE] ✅ Heavy OpenVINO model (yolo26s) loaded successfully.", flush=True)
                except Exception as e:
                    print(f"[AI-ENGINE] ⚠️ Heavy OpenVINO load failed ({e}), falling back to PyTorch .pt...", flush=True)

            if not loaded:
                print(f"[AI-ENGINE] Loading Heavy PyTorch model (yolo26s): {pt_path}...", flush=True)
                m = YOLO(pt_path)
                try:
                    _device = "cuda" if torch.cuda.is_available() else "cpu"
                    m.to(_device)
                except Exception:
                    pass
                _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                try:
                    m(_dummy, imgsz=320, verbose=False)
                except Exception:
                    pass
                yolo_model_heavy = m
                yolo_model_heavy_backend = "PyTorch (CPU)"
                yolo_model_heavy_name_str = os.path.basename(pt_path)
                print(f"[AI-ENGINE] ✅ Heavy PyTorch model (yolo26s) loaded successfully.", flush=True)

    return yolo_model_heavy

def load_yolo_pose_model():
    global yolo_pose_model, yolo_pose_backend, yolo_pose_name_str
    if yolo_pose_model is not None:
        return yolo_pose_model
    with _yolo_lock:
        if yolo_pose_model is None:
            ov_path, _ = _find_model_dir_or_pt("yolo26n-pose_openvino_model", "yolov8n-pose_openvino_model", "yolo26n-pose.pt", "yolov8n-pose.pt")
            pt_path = _find_model_path("yolo26n-pose.pt", "yolov8n-pose.pt")
            
            import importlib
            ultralytics = importlib.import_module("ultralytics")
            YOLO = ultralytics.YOLO

            loaded = False
            if ov_path and os.path.exists(ov_path) and os.path.isdir(ov_path):
                try:
                    print(f"[AI-ENGINE] Loading Pose OpenVINO model: {ov_path}...", flush=True)
                    m = YOLO(ov_path)
                    _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                    m(_dummy, imgsz=320, verbose=False)
                    yolo_pose_model = m
                    yolo_pose_backend = "OpenVINO (CPU)"
                    yolo_pose_name_str = os.path.basename(ov_path)
                    loaded = True
                    print(f"[AI-ENGINE] ✅ Pose OpenVINO model loaded successfully.", flush=True)
                except Exception as e:
                    print(f"[AI-ENGINE] ⚠️ Pose OpenVINO load failed ({e}), falling back to PyTorch .pt...", flush=True)

            if not loaded:
                print(f"[AI-ENGINE] Loading Pose PyTorch model: {pt_path}...", flush=True)
                m = YOLO(pt_path)
                try:
                    _device = "cuda" if torch.cuda.is_available() else "cpu"
                    m.to(_device)
                except Exception:
                    pass
                _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                try:
                    m(_dummy, imgsz=320, verbose=False)
                except Exception:
                    pass
                yolo_pose_model = m
                yolo_pose_backend = "PyTorch (CPU)"
                yolo_pose_name_str = os.path.basename(pt_path)
                print(f"[AI-ENGINE] ✅ Pose PyTorch model loaded successfully.", flush=True)

    return yolo_pose_model

# ── Cascade & HOG Fallback
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    face_cascade = None

try:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
except Exception:
    hog = None

COCO_PERSON   = 0
COCO_HANDBAG  = 26
COCO_BACKPACK = 24
COCO_SUITCASE = 28
BAG_CLASSES   = {COCO_HANDBAG, COCO_BACKPACK, COCO_SUITCASE}

COCO_CLASS_NAMES = {
    0: "Orang", 1: "Sepeda", 2: "Mobil", 3: "Motor", 5: "Bus", 7: "Truk",
    24: "Tas", 26: "Tas", 28: "Tas", 56: "Kursi", 63: "Laptop"
}

CLASS_COLORS = {
    0: (120, 255, 0),     # Orang
    1: (255, 200, 0),     # Sepeda
    2: (0, 165, 255),     # Mobil
    3: (0, 90, 255),      # Motor
    5: (255, 255, 0),     # Bus
    7: (0, 0, 255),       # Truk
    24: (200, 100, 255),  # Tas
    26: (200, 100, 255),
    28: (200, 100, 255),
    56: (150, 150, 150),  # Kursi
    63: (255, 120, 0)     # Laptop
}

COCO_TRANSLATIONS = {
    "person": "Orang", "bicycle": "Sepeda", "car": "Mobil", "motorcycle": "Motor",
    "airplane": "Pesawat", "bus": "Bus", "train": "Kereta", "truck": "Truk", "boat": "Perahu",
    "backpack": "Tas", "umbrella": "Payung", "handbag": "Tas", "tie": "Dasi", "suitcase": "Tas",
    "chair": "Kursi", "laptop": "Laptop", "cell phone": "HP"
}

def get_class_name(cls_id):
    active_m = yolo_model_heavy if yolo_model_heavy is not None else yolo_model
    if active_m is None or not hasattr(active_m, "names"):
        return COCO_CLASS_NAMES.get(cls_id, "Objek")
    english_name = active_m.names.get(cls_id, "object")
    return COCO_TRANSLATIONS.get(english_name.lower(), english_name.capitalize())

def get_class_color(cls_id):
    if cls_id in CLASS_COLORS:
        return CLASS_COLORS[cls_id]
    state_rand = np.random.RandomState(cls_id)
    return tuple(int(x) for x in state_rand.randint(50, 230, size=3))

def get_dynamic_font_params(frame_height):
    scale = max(0.7, min(1.2, frame_height / 720.0 * 0.85))
    thickness = 2 if scale < 1.0 else 3
    return scale, thickness

def get_dynamic_box_thickness(frame_height):
    return 2 if frame_height < 900 else 3

import re

def clean_ascii_for_cv2(text):
    if not text:
        return ""
    cleaned = re.sub(r'[^\x00-\x7F]+', '', text).strip()
    return cleaned if cleaned else text

def draw_text_with_bg(frame, text, org, color=(255, 255, 255), bg_color=(0, 0, 0)):
    h, w = frame.shape[:2]
    font_scale, thickness = get_dynamic_font_params(h)
    cv2_text = clean_ascii_for_cv2(text)
    (tw, th), baseline = cv2.getTextSize(cv2_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    pad_x, pad_y = 6, 6
    bx1 = max(0, x - pad_x)
    bx2 = min(w, x + tw + pad_x)
    by1 = max(0, y - th - pad_y)
    by2 = min(h, y + baseline + pad_y - 2)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.putText(frame, cv2_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_labeled_box(frame, x1, y1, x2, y2, label, box_color, text_color=(255, 255, 255), box_thickness=None):
    h, w = frame.shape[:2]
    font_scale, thickness = get_dynamic_font_params(h)
    draw_thickness = box_thickness if box_thickness is not None else get_dynamic_box_thickness(h)
    pad_x, pad_y = 6, 6
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, draw_thickness)
    if label:
        cv2_label = clean_ascii_for_cv2(label)
        (tw, th), baseline = cv2.getTextSize(cv2_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if y1 - th - (pad_y * 2) > 0:
            bx1 = max(0, x1)
            bx2 = min(w, x1 + tw + (pad_x * 2))
            by1 = y1 - th - (pad_y * 2)
            by2 = y1
            tx = x1 + pad_x
            ty = y1 - pad_y
        else:
            bx1 = max(0, x1)
            bx2 = min(w, x1 + tw + (pad_x * 2))
            by1 = y1
            by2 = y1 + th + baseline + (pad_y * 2)
            tx = x1 + pad_x
            ty = y1 + th + pad_y - 1
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, -1)
        cv2.putText(frame, cv2_label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

# ── SQLite Setup for Drowsiness Logs
def init_db():
    try:
        conn = sqlite3.connect("drowsiness_logs.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drowsiness_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cam_id INTEGER,
                timestamp TEXT,
                ear REAL
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[AI-ENGINE] Database init error: {e}", flush=True)

init_db()

def log_drowsiness(cam_id, ear):
    try:
        conn = sqlite3.connect("drowsiness_logs.db")
        cursor = conn.cursor()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO drowsiness_logs (cam_id, timestamp, ear) VALUES (?, ?, ?)", (cam_id, timestamp, ear))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[AI-ENGINE] Logging error: {e}", flush=True)

def calculate_ear(eye_landmarks, landmarks, w, h):
    coords = []
    for idx in eye_landmarks:
        lm = landmarks[idx]
        coords.append(np.array([lm.x * w, lm.y * h]))
    p1, p2, p3, p4, p5, p6 = coords
    val_a = np.linalg.norm(p2 - p6)
    val_b = np.linalg.norm(p3 - p5)
    val_c = np.linalg.norm(p1 - p4)
    if val_c == 0:
        return 0.0
    return (val_a + val_b) / (2.0 * val_c)

# ── MediaPipe Tasks API for Face Landmarker (Drowsiness Detection)
HAS_MEDIAPIPE = True
landmarker_instance = None
mp_Image = None
mp_image_format = None

def load_mediapipe():
    global landmarker_instance, mp_Image, mp_image_format, HAS_MEDIAPIPE
    if landmarker_instance is not None:
        return landmarker_instance
    with _yolo_lock:
        if landmarker_instance is None:
            try:
                print(f"[AI-ENGINE] Lazy loading MediaPipe FaceLandmarker...", flush=True)
                model_path = "face_landmarker.task"
                if not os.path.exists(model_path):
                    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                    urllib.request.urlretrieve(url, model_path)
                    
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                BaseOptions = python.BaseOptions
                FaceLandmarker = vision.FaceLandmarker
                FaceLandmarkerOptions = vision.FaceLandmarkerOptions
                RunningMode = vision.RunningMode
                mp_image_format = mp.ImageFormat
                mp_Image = mp.Image
                
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=RunningMode.IMAGE
                )
                landmarker_instance = FaceLandmarker.create_from_options(options)
                HAS_MEDIAPIPE = True
                print("[AI-ENGINE] MediaPipe Tasks FaceLandmarker initialized OK.", flush=True)
            except Exception as e:
                HAS_MEDIAPIPE = False
                print(f"[AI-ENGINE] MediaPipe Face Mesh load error: {e}", flush=True)
    return landmarker_instance

def iou_overlap(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0: return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    denom = area1 + area2 - inter
    return inter / float(denom) if denom > 0 else 0.0

def run_yolo(frame, target_classes, model=None, imgsz=320):
    """Run YOLO inference on frame with target imgsz, with automatic 640 fallback for static OpenVINO models."""
    detections = []
    active_model = model if model is not None else yolo_model
    if active_model is None:
        return detections
    try:
        results = []
        with _yolo_lock:
            with torch.no_grad():
                try:
                    if target_classes is not None:
                        results = active_model(frame, imgsz=imgsz, verbose=False, conf=0.25, classes=list(target_classes))
                    else:
                        results = active_model(frame, imgsz=imgsz, verbose=False, conf=0.25)
                except Exception:
                    if target_classes is not None:
                        results = active_model(frame, imgsz=640, verbose=False, conf=0.25, classes=list(target_classes))
                    else:
                        results = active_model(frame, imgsz=640, verbose=False, conf=0.25)
        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.25: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append((x1, y1, x2, y2, cls, conf))
    except Exception as e:
        print(f"[AI-ENGINE] YOLO error: {e}", flush=True)
    return detections

# ── PyTorch Human Attribute Color Detection
def get_clothing_color_pytorch(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return "Tidak Diketahui", 0
    try:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0].astype(np.float32)
        s_ch = hsv[:, :, 1].astype(np.float32)
        v_ch = hsv[:, :, 2].astype(np.float32)

        total_px = h_ch.size
        if total_px == 0:
            return "Tidak Diketahui", 0

        black_mask = (v_ch < 60)
        white_mask = (s_ch < 35) & (v_ch > 200)
        gray_mask = (s_ch < 40) & (v_ch >= 60) & (v_ch <= 200)
        chromatic_mask = s_ch >= 40

        hue_ranges = {
            "Merah": [(0, 8), (170, 179)],
            "Cokelat": [(9, 20)],
            "Kuning": [(21, 34)],
            "Hijau": [(35, 85)],
            "Biru": [(86, 130)],
            "Ungu": [(131, 169)],
        }

        counts = {}
        counts["Hitam"] = int(np.count_nonzero(black_mask))
        counts["Putih"] = int(np.count_nonzero(white_mask & ~black_mask))
        counts["Abu-abu"] = int(np.count_nonzero(gray_mask & ~black_mask))

        for name, ranges in hue_ranges.items():
            mask = np.zeros_like(h_ch, dtype=bool)
            for (lo, hi) in ranges:
                mask |= (h_ch >= lo) & (h_ch <= hi)
            mask &= chromatic_mask & ~black_mask
            counts[name] = int(np.count_nonzero(mask))

        best_name = max(counts, key=counts.get)
        confidence = int((counts[best_name] / total_px) * 100)
        return best_name, confidence
    except Exception:
        return "Abu-abu", 50

# ── Face Recognition Engine (YuNet + SFace)
YUNET_MODEL_PATH = "face_detection_yunet.onnx"
SFACE_MODEL_PATH = "face_recognition_sface.onnx"

_yunet_detector = None
_sface_recognizer = None
_registered_faces_cache = []

def init_face_recognition():
    global _yunet_detector, _sface_recognizer
    try:
        if not os.path.exists(YUNET_MODEL_PATH):
            yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            urllib.request.urlretrieve(yunet_url, YUNET_MODEL_PATH)
        if not os.path.exists(SFACE_MODEL_PATH):
            sface_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            urllib.request.urlretrieve(sface_url, SFACE_MODEL_PATH)
            
        _yunet_detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (320, 320), 0.6, 0.3, 5000)
        _sface_recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
        print("[AI-ENGINE] YuNet & SFace Face Recognition engine initialized OK.", flush=True)
    except Exception as e:
        print(f"[AI-ENGINE] Warning initializing YuNet/SFace: {e}", flush=True)

    init_face_db()
    reload_registered_faces_cache()

def init_face_db():
    conn = sqlite3.connect("face_database.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS registered_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def reload_registered_faces_cache():
    global _registered_faces_cache
    cache = []
    try:
        conn = sqlite3.connect("face_database.db")
        cur = conn.cursor()
        cur.execute("SELECT id, name, embedding, image_path, created_at FROM registered_faces")
        rows = cur.fetchall()
        for row in rows:
            face_id, name, emb_blob, img_path, created_at = row
            emb_arr = np.frombuffer(emb_blob, dtype=np.float32)
            cache.append({
                "id": face_id,
                "name": name,
                "embedding": emb_arr,
                "image_path": img_path,
                "created_at": str(created_at)
            })
        conn.close()
    except Exception as e:
        print(f"[AI-ENGINE] Error reloading face cache: {e}", flush=True)
    _registered_faces_cache = cache

def extract_face_embedding(img_bgr):
    if img_bgr is None or img_bgr.size == 0 or _sface_recognizer is None:
        return None, None
    h, w = img_bgr.shape[:2]
    if _yunet_detector is not None:
        _yunet_detector.setInputSize((w, h))
        _, faces = _yunet_detector.detect(img_bgr)
        if faces is not None and len(faces) > 0:
            face = faces[0]
            try:
                aligned_face = _sface_recognizer.alignCrop(img_bgr, face)
                feat = _sface_recognizer.feature(aligned_face)
                return feat.flatten(), aligned_face
            except Exception: pass
    if face_cascade is not None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        dets = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        if len(dets) > 0:
            fx, fy, fw, fh = dets[0]
            dummy_face = np.array([[fx, fy, fw, fh, fx+fw/3, fy+fh/3, fx+2*fw/3, fy+fh/3, fx+fw/2, fy+fh/2, fx+fw/3, fy+2*fh/3, fx+2*fw/3, fy+2*fh/3, 1.0]], dtype=np.float32)
            try:
                aligned_face = _sface_recognizer.alignCrop(img_bgr, dummy_face[0])
                feat = _sface_recognizer.feature(aligned_face)
                return feat.flatten(), aligned_face
            except Exception: pass
    return None, None

def register_face(name, image_bytes_list):
    os.makedirs("uploads/faces", exist_ok=True)
    saved_records = []
    conn = sqlite3.connect("face_database.db")
    cur = conn.cursor()
    
    for idx, img_bytes in enumerate(image_bytes_list):
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None: continue
        
        emb, aligned_crop = extract_face_embedding(img_bgr)
        if emb is None: continue
            
        filename = f"face_{int(time.time()*1000)}_{idx}.jpg"
        save_path = os.path.join("uploads", "faces", filename)
        save_img = aligned_crop if aligned_crop is not None else img_bgr
        cv2.imwrite(save_path, save_img)
        
        rel_path = f"/uploads/faces/{filename}"
        emb_bytes = emb.astype(np.float32).tobytes()
        cur.execute("INSERT INTO registered_faces (name, embedding, image_path) VALUES (?, ?, ?)",
                    (name, emb_bytes, rel_path))
        saved_records.append(cur.lastrowid)
        
    conn.commit()
    conn.close()
    reload_registered_faces_cache()
    return len(saved_records)

def get_registered_faces_list():
    conn = sqlite3.connect("face_database.db")
    cur = conn.cursor()
    cur.execute("SELECT id, name, image_path, created_at FROM registered_faces ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "image_path": r[2], "created_at": str(r[3])} for r in rows]

def delete_registered_face(face_id):
    conn = sqlite3.connect("face_database.db")
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM registered_faces WHERE id = ?", (face_id,))
    row = cur.fetchone()
    if row:
        img_path = row[0]
        full_path = os.path.join(".", img_path.lstrip("/"))
        if os.path.exists(full_path):
            try: os.remove(full_path)
            except Exception: pass
    cur.execute("DELETE FROM registered_faces WHERE id = ?", (face_id,))
    conn.commit()
    conn.close()
    reload_registered_faces_cache()
    return True

def match_face_embedding(feat):
    if feat is None or len(_registered_faces_cache) == 0 or _sface_recognizer is None:
        return "Tidak Dikenal", 0.0
    
    best_name = "Tidak Dikenal"
    best_score = 0.0
    feat_norm = np.linalg.norm(feat)
    if feat_norm == 0:
        return "Tidak Dikenal", 0.0
    
    for record in _registered_faces_cache:
        emb = record["embedding"]
        emb_norm = np.linalg.norm(emb)
        if emb_norm == 0: continue
        sim = float(np.dot(feat, emb) / (feat_norm * emb_norm))
        if sim > best_score:
            best_score = sim
            best_name = record["name"]
            
    if best_score >= 0.38:
        return best_name, best_score
    return "Tidak Dikenal", best_score

# Initialize Face Recognition Engine
init_face_recognition()

# ── MODE PROCESSORS

def process_face(frame, state, meta, cached_faces=None):
    """
    Face Recognition with Track-ID & Bounding-Box Caching.
    Runs Face Detection (YuNet/Cascade) every frame, but caches SFace feature recognition
    results per face box to avoid running embedding extraction on every frame for the same face.
    """
    h, w = frame.shape[:2]
    out = frame
    faces = []
    now = time.time()
    
    if cached_faces is not None:
        faces = cached_faces
    else:
        if _yunet_detector is not None:
            _yunet_detector.setInputSize((w, h))
            _, det_faces = _yunet_detector.detect(frame)
            if det_faces is not None:
                for f in det_faces:
                    fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    faces.append((max(0, fx), max(0, fy), max(1, fw), max(1, fh)))
        if not faces and face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                det = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
                for (fx,fy,fw,fh) in det:
                    faces.append((fx,fy,fw,fh))
            except Exception: pass

    meta["faces"] = faces
    meta["face_detected"] = len(faces) > 0
    
    face_rec_cache = state.setdefault("face_recognition_cache", [])
    updated_cache = []
    
    if faces:
        detected_names = []
        for face_tuple in faces:
            fx, fy, fw, fh = face_tuple[0], face_tuple[1], face_tuple[2], face_tuple[3]
            box_curr = [fx, fy, fx + fw, fy + fh]
            
            name = None
            score = 0.0
            
            # Check spatial IoU overlap with previous cached faces (Track-ID caching)
            best_iou = 0.0
            best_match_idx = -1
            for idx, c_item in enumerate(face_rec_cache):
                c_box = c_item["box"]
                overlap = iou_overlap(box_curr, c_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_match_idx = idx
                    
            if best_iou >= 0.35 and best_match_idx != -1 and (now - face_rec_cache[best_match_idx]["time"]) < 3.0:
                # Reuse cached recognition result for this tracked face
                name = face_rec_cache[best_match_idx]["name"]
                score = face_rec_cache[best_match_idx]["score"]
                updated_cache.append({
                    "box": box_curr,
                    "name": name,
                    "score": score,
                    "time": now
                })
            else:
                # Run SFace feature extraction & database matching ONLY for new or un-cached faces
                crop = frame[max(0,fy):min(h,fy+fh), max(0,fx):min(w,fx+fw)]
                if crop.size > 100:
                    emb, _ = extract_face_embedding(crop)
                    if emb is not None:
                        name, score = match_face_embedding(emb)
                if not name:
                    name = "Tidak Dikenal"
                    score = 0.0
                    
                updated_cache.append({
                    "box": box_curr,
                    "name": name,
                    "score": score,
                    "time": now
                })
                
            detected_names.append(name)
            box_color = (0, 220, 0) if name != "Tidak Dikenal" else (0, 120, 255)
            draw_labeled_box(out, fx, fy, fx+fw, fy+fh, name, box_color, (255, 255, 255), 2)
            
        state["face_recognition_cache"] = updated_cache
        meta["face_name"] = ", ".join(set(detected_names))
        meta["people_count"] = len(faces)
    else:
        state["face_recognition_cache"] = []
        meta["face_name"] = "None"
        meta["people_count"] = 0
        draw_text_with_bg(out, "Tidak ada wajah terdeteksi", (10, h - 25), (100, 100, 255), (0, 0, 0))
    
    return out, meta


def process_attribute(frame, persons, detections, bags_raw, meta, cached_colors=None):
    h, w = frame.shape[:2]
    out = frame
    person_color = (0, 165, 255)
    attr_threshold = 40
    object_counts = {}
    detected_objects_list = []
    person_entries = []

    bag_boxes = []
    for (bx1, by1, bw, bh, cls_id, conf) in bags_raw:
        bag_boxes.append({
            "bbox": (bx1, by1, bx1 + bw, by1 + bh),
            "class_id": cls_id,
            "conf": conf,
            "name": get_class_name(cls_id)
        })

    def add_detected_object(class_name, conf, attributes=None):
        attrs = attributes or []
        detected_objects_list.append({
            "class": class_name,
            "conf": round(float(conf), 2),
            "attributes": attrs
        })
        object_counts[class_name] = object_counts.get(class_name, 0) + 1

    current_cached_colors = []
    for idx, (px, py, pw, ph, conf) in enumerate(persons):
        px2 = min(w, px + pw)
        py2 = min(h, py + ph)
        
        aspect_ratio = ph / pw if pw > 0 else 0
        baju_visible = True
        celana_visible = True
        if aspect_ratio < 0.9: celana_visible = False
        if aspect_ratio < 0.4: baju_visible = False

        ux1, ux2 = max(0, px), px2
        uy1, uy2 = max(0, py + int(ph * 0.15)), min(h, py + int(ph * 0.50))
        lx1, lx2 = max(0, px), px2
        ly1, ly2 = max(0, py + int(ph * 0.55)), min(h, py + int(ph * 0.90))

        upper_crop = frame[uy1:uy2, ux1:ux2]
        lower_crop = frame[ly1:ly2, lx1:lx2]

        if cached_colors is not None and idx < len(cached_colors):
            top_color, top_conf, bottom_color, bottom_conf, unique_bags = cached_colors[idx]
        else:
            top_color, top_conf = get_clothing_color_pytorch(upper_crop) if baju_visible and upper_crop.size > 0 else ("Tidak Diketahui", 0)
            bottom_color, bottom_conf = get_clothing_color_pytorch(lower_crop) if celana_visible and lower_crop.size > 0 else ("Tidak Diketahui", 0)

            associated_bags = []
            person_box = [px, py, px2, py2]
            for bag in bag_boxes:
                bx1, by1, bx2, by2 = bag["bbox"]
                bcx = (bx1 + bx2) // 2
                bcy = (by1 + by2) // 2
                if (iou_overlap(person_box, [bx1, by1, bx2, by2]) > 0.05 or (px - 20 <= bcx <= px2 + 20 and py - 20 <= bcy <= py2 + 20)):
                    associated_bags.append(bag["name"])
            unique_bags = list(dict.fromkeys(associated_bags))

        current_cached_colors.append((top_color, top_conf, bottom_color, bottom_conf, unique_bags))

        person_label = f"Orang: {int(conf * 100)}%"
        draw_labeled_box(out, px, py, px2, py2, person_label, person_color, (255, 255, 255))

        top_combined_conf = int(conf * top_conf) if baju_visible else 0
        bottom_combined_conf = int(conf * bottom_conf) if celana_visible else 0

        person_attrs = []
        if baju_visible and upper_crop.size > 0 and top_combined_conf >= attr_threshold:
            top_label = f"Baju {top_color.lower()}: {top_combined_conf}%"
            draw_labeled_box(out, ux1, uy1, ux2, uy2, top_label, person_color, (255, 255, 255))
            person_attrs.append(f"Baju {top_color.lower()} ({top_combined_conf}%)")

        if celana_visible and lower_crop.size > 0 and bottom_combined_conf >= attr_threshold:
            bottom_label = f"Celana {bottom_color.lower()}: {bottom_combined_conf}%"
            draw_labeled_box(out, lx1, ly1, lx2, ly2, bottom_label, person_color, (255, 255, 255))
            person_attrs.append(f"Celana {bottom_color.lower()} ({bottom_combined_conf}%)")

        if unique_bags:
            person_attrs.append(f"Membawa {', '.join(unique_bags)}")

        person_entry = {
            "id": idx + 1, "class": "Orang", "conf": round(float(conf), 2), "attributes": person_attrs
        }
        person_entries.append(person_entry)
        add_detected_object("Orang", conf, person_attrs)

    for (x1, y1, x2, y2, cls, conf) in detections:
        if cls == COCO_PERSON: continue
        disp_name = get_class_name(cls)
        draw_color = person_color if cls in BAG_CLASSES else get_class_color(cls)
        label = f"{disp_name}: {int(conf * 100)}%"
        draw_labeled_box(out, x1, y1, x2, y2, label, draw_color, (255, 255, 255))
        add_detected_object(disp_name, conf)

    meta["cached_colors"] = current_cached_colors
    meta["object_counts"] = object_counts
    meta["attributes"] = person_entries
    meta["detected_objects_list"] = detected_objects_list
    meta["people_count"] = len(persons)
    return out, meta


VEHICLE_CLASS_MAP = {
    0: "Orang", 1: "Sepeda", 2: "Mobil", 3: "Motor", 5: "Bus", 7: "Truk"
}

def process_vehicle_tracking(frame, infer_fr, scale, state, meta, cached_data=None):
    """Single-pass vehicle tracking using YOLO26s OpenVINO / PyTorch fallback with ByteTrack."""
    h, w = frame.shape[:2]
    out = frame
    
    if "seen_vehicle_track_ids" not in state:
        state["seen_vehicle_track_ids"] = {
            "Mobil": set(), "Motor": set(), "Truk": set(), "Bus": set(), "Sepeda": set(), "Orang": set()
        }
    seen_ids = state["seen_vehicle_track_ids"]
    
    current_in_frame = {
        "Mobil": 0, "Motor": 0, "Truk": 0, "Bus": 0, "Sepeda": 0, "Orang": 0
    }
    
    active_model = load_yolo_heavy()
    tracked_objects = []
    
    if cached_data is not None:
        tracked_objects = cached_data
    else:
        if active_model is not None and infer_fr is not None:
            with _yolo_lock:
                results = []
                try:
                    results = active_model.track(
                        infer_fr,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[0, 1, 2, 3, 5, 7],
                        conf=0.20,
                        imgsz=320,
                        verbose=False
                    )
                except Exception:
                    try:
                        results = active_model.track(
                            infer_fr,
                            persist=True,
                            tracker="bytetrack.yaml",
                            classes=[0, 1, 2, 3, 5, 7],
                            conf=0.25,
                            imgsz=640,
                            verbose=False
                        )
                    except Exception:
                        results = []

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        track_id = int(box.id[0].item()) if box.id is not None else None
                        
                        if scale != 1.0:
                            bx1 = int(bx1 / scale)
                            by1 = int(by1 / scale)
                            bx2 = int(bx2 / scale)
                            by2 = int(by2 / scale)
                            
                        tracked_objects.append((bx1, by1, bx2, by2, cls_id, conf, track_id))

    detected_objects_list = []
    for (x1, y1, x2, y2, cls_id, conf, track_id) in tracked_objects:
        class_name = VEHICLE_CLASS_MAP.get(cls_id, "Kendaraan")
        if class_name in current_in_frame:
            current_in_frame[class_name] += 1
            
        if track_id is not None and class_name in seen_ids:
            seen_ids[class_name].add(track_id)
            id_str = f" #{track_id}"
        else:
            id_str = ""
            
        color = CLASS_COLORS.get(cls_id, (0, 255, 255))
        label_txt = f"{class_name}{id_str} ({int(conf*100)}%)"
        draw_labeled_box(out, x1, y1, x2, y2, label_txt, color, (255, 255, 255), 2)
        detected_objects_list.append(label_txt)

    cumulative_counts = {cat: len(seen_ids[cat]) for cat in seen_ids}
    overlay_txt = f"KUMULATIF UNIK: Mobil:{cumulative_counts['Mobil']} | Motor:{cumulative_counts['Motor']} | Truk:{cumulative_counts['Truk']} | Bus:{cumulative_counts['Bus']}"
    draw_text_with_bg(out, overlay_txt, (10, 60), (15, 23, 42), (0, 255, 120))
    
    meta["vehicle_counts"] = cumulative_counts
    meta["current_in_frame"] = current_in_frame
    meta["object_counts"] = cumulative_counts
    meta["people_count"] = current_in_frame.get("Orang", 0)
    meta["detected_objects_list"] = detected_objects_list
    meta["tracked_objects"] = tracked_objects
    
    return out, meta


def process_drowsiness(cam_id, frame, state, meta):
    h, w = frame.shape[:2]
    out = frame
    now = time.time()
    
    load_mediapipe()
    if not HAS_MEDIAPIPE or landmarker_instance is None:
        draw_text_with_bg(out, "ERR: MediaPipe not loaded", (12, h - 18), (255, 255, 255), (0, 0, 255))
        return out, meta

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp_Image(image_format=mp_image_format.SRGB, data=rgb)
    res = landmarker_instance.detect(mp_image)
    
    ear = 0.0
    drowsy = False
    alarm = False
    state_info = state.setdefault("drowsy_timer", {"start_time": None, "current_state": "Normal"})
    
    if res.face_landmarks:
        landmarks = res.face_landmarks[0]
        left_ear = calculate_ear([362, 385, 387, 263, 373, 380], landmarks, w, h)
        right_ear = calculate_ear([33, 160, 158, 133, 153, 144], landmarks, w, h)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < 0.20:
            if state_info["start_time"] is None:
                state_info["start_time"] = now
            else:
                elapsed = now - state_info["start_time"]
                if elapsed > 2.0:
                    drowsy = True
                    alarm = True
                    if state_info["current_state"] != "Mengantuk":
                        state_info["current_state"] = "Mengantuk"
                        log_drowsiness(cam_id, ear)
        else:
            state_info["start_time"] = None
            state_info["current_state"] = "Normal"
            
        drowsy = state_info["current_state"] == "Mengantuk"
        alarm = drowsy
        eye_color = (0, 0, 255) if drowsy else (0, 255, 0)
        
        left_pts = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in [362, 385, 387, 263, 373, 380]], dtype=np.int32)
        right_pts = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in [33, 160, 158, 133, 153, 144]], dtype=np.int32)

        if drowsy:
            overlay = out.copy()
            cv2.fillPoly(overlay, [left_pts], (0, 0, 255))
            cv2.fillPoly(overlay, [right_pts], (0, 0, 255))
            out = cv2.addWeighted(overlay, 0.28, out, 0.72, 0.0)
        
        cv2.polylines(out, [left_pts], True, eye_color, 3 if drowsy else 2, cv2.LINE_AA)
        cv2.polylines(out, [right_pts], True, eye_color, 3 if drowsy else 2, cv2.LINE_AA)
        
        for idx in [362, 385, 387, 263, 373, 380, 33, 160, 158, 133, 153, 144]:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(out, (cx, cy), max(3, h // 320), eye_color, -1, cv2.LINE_AA)
            
    else:
        state_info["start_time"] = None
        state_info["current_state"] = "Normal"
        drowsy = False
        alarm = False

    draw_text_with_bg(out, "Status Kamera: Aktif", (12, 72), (255, 255, 255), (22, 163, 74))
    draw_text_with_bg(out, f"Status Deteksi: {'Mengantuk' if drowsy else 'Normal'}", (12, 114), (255, 255, 255), (0, 0, 255) if drowsy else (22, 163, 74))
    draw_text_with_bg(out, f"EAR: {ear:.2f}", (12, 156), (255, 255, 255), (15, 23, 42))

    if drowsy:
        if int(now * 2.5) % 2 == 0:
            cv2.rectangle(out, (0, 0), (w, h), (0, 0, 255), max(4, get_dynamic_box_thickness(h) + 1))
            font_scale, thickness = get_dynamic_font_params(h)
            alarm_text = "Alarm Berbunyi!"
            (tw, th), _ = cv2.getTextSize(alarm_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            alarm_x = max(12, (w - tw) // 2)
            alarm_y = max(36, int(h * 0.08))
            draw_text_with_bg(out, alarm_text, (alarm_x, alarm_y), (255, 255, 255), (0, 0, 255))

    meta["camera_status"] = "Aktif"
    meta["drowsiness_status"] = "Mengantuk" if drowsy else "Normal"
    meta["ear_value"] = round(ear, 3)
    meta["alarm"] = alarm
    return out, meta


def process_pose(frame, infer_fr, scale, meta, cached_pose_data=None):
    h, w = frame.shape[:2]
    out = frame
    
    if cached_pose_data is None:
        load_yolo_pose_model()
        if yolo_pose_model is None:
            draw_text_with_bg(out, "ERR: YOLO Pose model not loaded", (12, h - 18), (255, 255, 255), (0, 0, 255))
            return out, meta

    box_color = (255, 0, 0)
    head_color = (0, 255, 0)
    torso_color = (255, 0, 255)
    arm_color = (255, 0, 0)
    leg_color = (0, 165, 255)
    line_thickness = get_dynamic_box_thickness(h)
    joint_radius = max(4, h // 260)
    
    connections = [
        ((0, 1), head_color), ((0, 2), head_color), ((1, 3), head_color), ((2, 4), head_color),
        ((5, 6), torso_color), ((5, 11), torso_color), ((6, 12), torso_color), ((11, 12), torso_color),
        ((5, 7), arm_color), ((7, 9), arm_color), ((6, 8), arm_color), ((8, 10), arm_color),
        ((11, 13), leg_color), ((13, 15), leg_color), ((12, 14), leg_color), ((14, 16), leg_color),
    ]

    if cached_pose_data is not None:
        for person in cached_pose_data:
            track_id = person.get("id", -1)
            conf = person.get("conf", 0.0)
            bbox = person.get("bbox")
            xy_list = person.get("keypoints", [])
            
            if bbox:
                x1_n, y1_n, x2_n, y2_n = bbox
                lbl_text = f"id:{track_id} person {conf:.2f}" if track_id != -1 else f"person {conf:.2f}"
                draw_labeled_box(out, x1_n, y1_n, x2_n, y2_n, lbl_text, box_color, (255, 255, 255), line_thickness)
            
            pts = {}
            for idx, pt in enumerate(xy_list):
                px_k_native, py_k_native = pt
                if px_k_native > 0 and py_k_native > 0:
                    pts[idx] = (int(px_k_native), int(py_k_native))
                    
            for (p1, p2), color in connections:
                if p1 in pts and p2 in pts:
                    cv2.line(out, pts[p1], pts[p2], color, line_thickness, cv2.LINE_AA)
                    
            for idx, pt_coords in pts.items():
                c_color = head_color if idx in range(5) else torso_color if idx in [5,6,11,12] else arm_color if idx in [7,8,9,10] else leg_color
                cv2.circle(out, pt_coords, joint_radius, c_color, -1, cv2.LINE_AA)
                
        meta["pose_data"] = cached_pose_data
        meta["people_count"] = len(cached_pose_data)
        return out, meta

    with torch.no_grad():
        try:
            results = yolo_pose_model.track(
                infer_fr, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.20, imgsz=320
            )
        except Exception:
            try:
                results = yolo_pose_model.track(
                    infer_fr, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.20, imgsz=640
                )
            except Exception:
                try:
                    results = yolo_pose_model(infer_fr, verbose=False, conf=0.20, imgsz=640)
                except Exception:
                    results = []
                
    pose_data = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and r.keypoints is not None:
            boxes = r.boxes
            kpts = r.keypoints
            for i in range(len(boxes)):
                box = boxes[i]
                track_id = int(box.id[0]) if box.id is not None else -1
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                if conf < 0.25: continue
                
                box_xy = box.xyxy[0].cpu().numpy().tolist()
                x1_n = max(0, int(box_xy[0] / scale))
                y1_n = max(0, int(box_xy[1] / scale))
                x2_n = min(w, int(box_xy[2] / scale))
                y2_n = min(h, int(box_xy[3] / scale))
                
                lbl_text = f"id:{track_id} person {conf:.2f}" if track_id != -1 else f"person {conf:.2f}"
                draw_labeled_box(out, x1_n, y1_n, x2_n, y2_n, lbl_text, box_color, (255, 255, 255), line_thickness)
                
                if i < len(kpts.xy):
                    xy_list_raw = kpts.xy[i].cpu().numpy()
                    conf_list = kpts.conf[i].cpu().numpy() if kpts.conf is not None else [1.0]*17
                    
                    pts = {}
                    xy_list = []
                    for idx in range(min(17, len(xy_list_raw))):
                        px_k, py_k = xy_list_raw[idx]
                        px_k_native = px_k / scale
                        py_k_native = py_k / scale
                        xy_list.append([round(float(px_k_native), 1), round(float(py_k_native), 1)])
                        
                        if conf_list[idx] > 0.45 and px_k_native > 0 and py_k_native > 0:
                            pts[idx] = (int(px_k_native), int(py_k_native))
                    
                    pose_data.append({
                        "id": track_id, "conf": round(conf, 2), "keypoints": xy_list, "bbox": [x1_n, y1_n, x2_n, y2_n]
                    })
                    
                    for (p1, p2), color in connections:
                        if p1 in pts and p2 in pts:
                            cv2.line(out, pts[p1], pts[p2], color, line_thickness, cv2.LINE_AA)
                            
                    for idx, pt_coords in pts.items():
                        c_color = head_color if idx in range(5) else torso_color if idx in [5,6,11,12] else arm_color if idx in [7,8,9,10] else leg_color
                        cv2.circle(out, pt_coords, joint_radius, c_color, -1, cv2.LINE_AA)

    meta["pose_data"] = pose_data
    meta["people_count"] = len(pose_data)
    return out, meta


# ── Main Dispatcher

MODE_MAPPING = {
    "Matikan Analitik AI": "none",
    "none": "none",
    "Deteksi Wajah (Face Recognition)": "face",
    "face": "face",
    "Atribut Pakaian & Objek Umum": "attribute",
    "attribute": "attribute",
    "Tracking & Penghitungan Kendaraan": "vehicle",
    "vehicle": "vehicle",
    "Deteksi Kantuk (EAR)": "drowsiness",
    "drowsiness": "drowsiness",
    "Pose Estimation & Human Tracking": "pose",
    "pose": "pose"
}

def run_analytics(cam_id, frame, mode_raw, selected_classes, state):
    t_start = time.time()
    mode = MODE_MAPPING.get(mode_raw, "none")
    
    # Pre-calculate resolution downscaling to max 640px wide for pipeline efficiency
    h, w = frame.shape[:2]
    target_w = 640
    if w > target_w:
        scale = target_w / float(w)
        target_h = int(h * scale)
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        h, w = target_h, target_w

    t_now = time.time()
    # Stream FPS Moving Average
    if "last_timestamp" in state:
        time_diff = t_now - state["last_timestamp"]
        if time_diff > 0:
            current_stream_fps = 1.0 / time_diff
            state["stream_fps"] = state.get("stream_fps", current_stream_fps) * 0.85 + current_stream_fps * 0.15
    else:
        state["stream_fps"] = 30.0
    state["last_timestamp"] = t_now

    state["frame_count"] = state.get("frame_count", 0) + 1

    # Mode OFF shortcut
    if mode == 'none':
        draw_text_with_bg(frame, "AI: OFF", (12, 28), (255, 255, 255), (0, 0, 255))
        if "last_processed" in state:
            del state["last_processed"]
        return frame, {
            "engine": "YOLO26", "camera_status": "Aktif", "people_count": 0, "face_detected": False,
            "face_name": "None", "attributes": [], "object_counts": {}, "detected_objects_list": [],
            "drowsiness_status": "Normal", "ear_value": 0.0, "alarm": False, "pose_data": [],
            "infer_time_ms": 0.0, "ai_fps": 0.0, "stream_fps": round(state.get("stream_fps", 30.0), 1),
            "cpu_usage": 0.0, "backend": "N/A", "active_model": "None"
        }

    # Stride interval for non-continuous modes (in seconds)
    MODE_AI_INTERVALS = {
        "face": 0.08, "vehicle": 0.08, "pose": 0.08, "attribute": 0.12, "drowsiness": 0.03
    }
    AI_INTERVAL = MODE_AI_INTERVALS.get(mode, 0.05)
    last_infer = state.get("last_inference_time", 0.0)
    time_since_last = t_now - last_infer
    
    should_skip = False
    if "last_metadata" in state and state.get("last_mode") == mode:
        if mode != 'drowsiness' and time_since_last < AI_INTERVAL:
            should_skip = True

    # ── CPU Usage Query
    cpu_percent = 0.0
    if psutil is not None:
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
        except Exception: pass

    # Get model and backend strings for current active mode
    if mode == 'face':
        act_model_name = "YuNet + SFace"
        act_backend = "OpenCV DNN (CPU)"
    elif mode in ['attribute', 'vehicle']:
        load_yolo_heavy()
        act_model_name = yolo_model_heavy_name_str
        act_backend = yolo_model_heavy_backend
    elif mode == 'pose':
        load_yolo_pose_model()
        act_model_name = yolo_pose_name_str
        act_backend = yolo_pose_backend
    elif mode == 'drowsiness':
        act_model_name = "FaceLandmarker"
        act_backend = "MediaPipe CPU"
    elif mode == 'zone_monitor':
        act_model_name = "Zone Monitor"
        act_backend = "YOLO Nano (CPU)"
    else:
        act_model_name = "YOLO26"
        act_backend = "CPU"

    if should_skip:
        # OVERLAY CACHING: Render previous bounding boxes/metadata over current live frame for smooth non-flickering visual stream
        last_meta = state["last_metadata"].copy()
        draw_text_with_bg(frame, f"AI: {act_model_name} ({act_backend})", (12, 28), (15, 23, 42), (0, 255, 120))
        
        try:
            if mode == 'face':
                processed, meta = process_face(frame, state, last_meta, cached_faces=last_meta.get("faces"))
            elif mode == 'attribute':
                processed, meta = process_attribute(frame, state.get("last_persons_raw", []), state.get("last_detections", []), state.get("last_bags_raw", []), last_meta, cached_colors=last_meta.get("cached_colors"))
            elif mode == 'vehicle':
                processed, meta = process_vehicle_tracking(frame, None, 1.0, state, last_meta, cached_data=last_meta.get("tracked_objects"))
            elif mode == 'pose':
                processed, meta = process_pose(frame, None, 1.0, last_meta, cached_pose_data=last_meta.get("pose_data"))
            elif mode == 'zone_monitor':
                try:
                    from zone_monitor import get_zone_monitor, draw_zones_on_frame
                    zm = get_zone_monitor()
                    cam_zones = zm.get_zones(cam_id=cam_id)
                    with zm._lock:
                        trackers = {k: v for k, v in zm._trackers.items()}
                    processed = draw_zones_on_frame(frame.copy(), cam_zones, trackers)
                    draw_text_with_bg(processed, f"AI: {act_model_name} ({act_backend})", (12, 28), (15, 23, 42), (0, 255, 120))
                    meta = last_meta
                except Exception:
                    processed = frame
                    meta = last_meta
            else:
                processed, meta = frame, last_meta
        except Exception as dispatch_err:
            processed = frame
            meta = last_meta

        meta["infer_time_ms"] = round(state.get("last_infer_time_ms", 0.0), 1)
        meta["ai_fps"] = round(state.get("ai_fps", 0.0), 1)
        meta["stream_fps"] = round(state.get("stream_fps", 0.0), 1)
        meta["cpu_usage"] = round(cpu_percent, 1)
        meta["backend"] = act_backend
        meta["active_model"] = act_model_name
        
        fps_text = f"FPS: {state.get('stream_fps', 0.0):.1f}"
        draw_text_with_bg(processed, fps_text, (max(12, w - 160), 28), (15, 23, 42), (0, 255, 255))
        return processed, meta

    # ── Perform Full Inference
    t_infer_start = time.time()
    draw_text_with_bg(frame, f"AI: {act_model_name} ({act_backend})", (12, 28), (15, 23, 42), (0, 255, 120))

    # Resize for model input (imgsz=320 for nano/pose, imgsz=416 for heavy)
    if mode == 'drowsiness':
        infer_fr = frame
        scale = 1.0
    elif mode in ['attribute', 'vehicle']:
        infer_w = 416
        scale = infer_w / float(w) if w > infer_w else 1.0
        infer_fr = cv2.resize(frame, (infer_w, int(h * scale)), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else frame
    elif mode == 'pose':
        infer_w = 320
        scale = infer_w / float(w) if w > infer_w else 1.0
        infer_fr = cv2.resize(frame, (infer_w, int(h * scale)), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else frame
    else: # face mode
        infer_fr = frame
        scale = 1.0

    persons_raw = []
    detections = []
    bags_raw = []
    
    meta = {
        "engine": act_model_name,
        "backend": act_backend,
        "camera_status": "Aktif",
        "people_count": 0,
        "face_detected": False,
        "face_name": "None",
        "attributes": [],
        "object_counts": {},
        "detected_objects_list": [],
        "drowsiness_status": "Normal",
        "ear_value": 0.0,
        "alarm": False,
        "pose_data": [],
        "stream_fps": round(state.get("stream_fps", 0.0), 1)
    }

    if mode == 'attribute':
        active_m = load_yolo_heavy()
        raw_detections = run_yolo(infer_fr, None, model=active_m, imgsz=416)
        for (x1, y1, x2, y2, cls, conf) in raw_detections:
            if scale != 1.0:
                x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
            detections.append((x1, y1, x2, y2, cls, conf))
            if cls == 0:
                persons_raw.append((x1, y1, x2 - x1, y2 - y1, conf))
            elif cls in BAG_CLASSES:
                bags_raw.append((x1, y1, x2 - x1, y2 - y1, cls, conf))

    # Dispatch to active mode processor
    try:
        if mode == 'face':
            processed, meta = process_face(frame, state, meta)
        elif mode == 'attribute':
            processed, meta = process_attribute(frame, persons_raw, detections, bags_raw, meta)
        elif mode == 'vehicle':
            processed, meta = process_vehicle_tracking(frame, infer_fr, scale, state, meta)
        elif mode == 'drowsiness':
            processed, meta = process_drowsiness(cam_id, frame, state, meta)
        elif mode == 'pose':
            processed, meta = process_pose(frame, infer_fr, scale, meta)
        elif mode == 'zone_monitor':
            try:
                from zone_monitor import get_zone_monitor
                zm = get_zone_monitor()
                # Feed frame to zone monitor background detector
                zm.feed_frame(cam_id, frame)
                # Get annotated frame (drawn by background detector thread)
                annotated = zm.get_frame_with_zones(cam_id)
                processed = annotated if annotated is not None else frame.copy()
                draw_text_with_bg(processed, f"ZONE MONITOR | {act_model_name}", (12, 28), (10, 20, 40), (0, 255, 120))
                # Build metadata for frontend canvas overlay
                cam_zones = zm.get_zones(cam_id=cam_id)
                meta["mode"] = "zone_monitor"
                meta["zones_count"] = len(cam_zones)
                meta["zones"] = [z.to_dict() for z in cam_zones]
                meta["zone_status"] = zm.get_zone_status(cam_id=cam_id)
            except Exception as e:
                print(f"[ANALYTICS] Zone monitor mode error: {e}", flush=True)
                processed = frame
        else:
            processed, meta = frame, meta
    except Exception as dispatch_err:
        import traceback
        print(f"[AI-ENGINE] DISPATCH ERROR cam_{cam_id} mode={mode}: {dispatch_err}", flush=True)
        print(traceback.format_exc(), flush=True)
        processed = frame

    infer_time_ms = (time.time() - t_infer_start) * 1000.0
    state["last_infer_time_ms"] = infer_time_ms

    # AI FPS Moving Average
    if "last_ai_infer_time" in state:
        dt_ai = t_now - state["last_ai_infer_time"]
        if dt_ai > 0:
            current_ai_fps = 1.0 / dt_ai
            state["ai_fps"] = state.get("ai_fps", current_ai_fps) * 0.85 + current_ai_fps * 0.15
    else:
        state["ai_fps"] = 0.0
    state["last_ai_infer_time"] = t_now

    # Benchmark logging every 3.0 seconds
    last_bench_log = state.get("last_benchmark_log_time", 0.0)
    if t_now - last_bench_log >= 3.0:
        state["last_benchmark_log_time"] = t_now
        print(
            f"[BENCHMARK] Cam {cam_id} | Mode: {mode:<10} | Model: {act_model_name:<22} | Backend: {act_backend:<15} | "
            f"Infer Time: {infer_time_ms:5.1f}ms | AI FPS: {state.get('ai_fps', 0.0):4.1f} | Stream FPS: {state.get('stream_fps', 0.0):4.1f} | CPU: {cpu_percent:4.1f}%",
            flush=True
        )

    fps_text = f"FPS: {state.get('stream_fps', 0.0):.1f}"
    draw_text_with_bg(processed, fps_text, (max(12, w - 160), 28), (15, 23, 42), (0, 255, 255))

    meta["infer_time_ms"] = round(infer_time_ms, 1)
    meta["ai_fps"] = round(state.get("ai_fps", 0.0), 1)
    meta["stream_fps"] = round(state.get("stream_fps", 0.0), 1)
    meta["cpu_usage"] = round(cpu_percent, 1)
    meta["backend"] = act_backend
    meta["active_model"] = act_model_name

    # Expose raw person detections untuk ZoneMonitor (list of (x1,y1,x2,y2,cls,conf))
    # ZoneMonitor menggunakan ini untuk cek kehadiran person di zona
    if mode == 'attribute':
        # Attribute mode: person bbox sudah ada di persons_raw (format: x,y,w,h,conf)
        meta["detections_raw"] = [
            (int(px), int(py), int(px+pw), int(py+ph), 0, float(pc))
            for (px, py, pw, ph, pc) in persons_raw
        ]
    elif mode == 'vehicle':
        # Vehicle mode: ambil person dari tracked_objects
        meta["detections_raw"] = [
            (x1, y1, x2, y2, cls, conf)
            for (x1, y1, x2, y2, cls, conf, _track_id) in meta.get("tracked_objects", [])
            if cls == 0  # cls 0 = person (COCO)
        ]
    elif mode == 'pose':
        # Pose mode: ambil bbox dari pose_data
        meta["detections_raw"] = [
            (p["bbox"][0], p["bbox"][1], p["bbox"][2], p["bbox"][3], 0, p["conf"])
            for p in meta.get("pose_data", []) if p.get("bbox")
        ]
    else:
        # Mode lain: zone monitor punya thread YOLO sendiri, detections_raw kosong
        meta["detections_raw"] = []

    # Cache last metadata for overlay persistence
    state["last_metadata"] = meta.copy()
    state["last_mode"] = mode
    state["last_persons_raw"] = persons_raw
    state["last_detections"] = detections
    state["last_bags_raw"] = bags_raw
    state["last_inference_time"] = t_now

    return processed, meta
