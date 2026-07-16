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

# ── YOLO Setup
# yolov8n.pt for light modes (face/sop/contraflow/pose), yolov8s.pt for attribute (needs accuracy)
HAS_YOLO = True
yolo_model = None         # yolov8n — default fast model
yolo_model_heavy = None   # yolov8s — only for attribute mode

# ── Model selection:
# Jika env var FORCE_PT=1 atau OpenVINO kernel.errors.txt ada (corrupt model) → pakai .pt langsung
# OpenVINO model di Raspberry Pi mengalami CISA kernel error (intersecting virtual registers),
# sehingga sementara di-bypass dan diganti dengan PyTorch .pt CPU mode.
_use_openvino = False  # DISABLED: OpenVINO model corrupt (lihat kernel.errors.txt)
if os.environ.get("FORCE_OPENVINO", "0") == "1":
    _use_openvino = True
    print("[AI-ENGINE] FORCE_OPENVINO=1 detected, akan mencoba OpenVINO model.", flush=True)

if _use_openvino:
    YOLO_MODEL_NAME = "yolov8n_openvino_model" if os.path.exists("yolov8n_openvino_model") else "yolov8n.pt"
    YOLO_MODEL_HEAVY = "yolov8s_openvino_model" if os.path.exists("yolov8s_openvino_model") else "yolov8s.pt"
else:
    YOLO_MODEL_NAME = "yolov8n.pt"
    YOLO_MODEL_HEAVY = "yolov8s.pt"
    print(f"[AI-ENGINE] Mode: PyTorch CPU (OpenVINO bypass aktif karena kernel corrupt)", flush=True)

print(f"[AI-ENGINE] YOLO model yang akan dipakai: {YOLO_MODEL_NAME}", flush=True)
print(f"[AI-ENGINE] YOLO heavy model: {YOLO_MODEL_HEAVY}", flush=True)

# Auto-detect GPU
_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[AI-ENGINE] Using device: {_device}", flush=True)

def load_yolo_model():
    global yolo_model, HAS_YOLO
    if yolo_model is not None:
        return yolo_model
    with _yolo_lock:
        if yolo_model is None:
            try:
                print(f"[AI-ENGINE] Lazy loading {YOLO_MODEL_NAME}...", flush=True)
                import importlib
                print(f"[AI-ENGINE] Importing ultralytics...", flush=True)
                ultralytics = importlib.import_module("ultralytics")
                
                # Disable ultralytics telemetry & checks
                try:
                    from ultralytics import settings
                    settings.update({'sync': False, 'check': False, 'telemetry': False})
                    print("[AI-ENGINE] Ultralytics telemetry and sync checks disabled.", flush=True)
                except Exception as se:
                    print(f"[AI-ENGINE] Note: could not update ultralytics settings: {se}", flush=True)

                YOLO = ultralytics.YOLO
                import logging
                logging.getLogger("ultralytics").setLevel(logging.WARNING)

                print(f"[AI-ENGINE] Initializing YOLO with {YOLO_MODEL_NAME}...", flush=True)
                yolo_model = YOLO(YOLO_MODEL_NAME)
                print(f"[AI-ENGINE] Moving YOLO model to {_device}...", flush=True)
                try:
                    yolo_model.to(_device)
                except Exception as te:
                    print(f"[AI-ENGINE] Note: model.to({_device}) skipped: {te}", flush=True)

                # ── Robust warm-up inference dengan fallback bertingkat ──
                # Raspberry Pi: OpenVINO bisa crash di sini, tapi .pt harusnya OK
                is_openvino = "openvino" in str(YOLO_MODEL_NAME).lower()
                warmup_imgsz = 640 if is_openvino else 320
                _dummy = np.zeros((warmup_imgsz, warmup_imgsz, 3), dtype=np.uint8)
                print(f"[AI-ENGINE] Running warm-up inference (imgsz={warmup_imgsz})...", flush=True)
                warmup_ok = False
                # Attempt 1: dengan device argument
                try:
                    yolo_model(_dummy, imgsz=warmup_imgsz, verbose=False, device=_device)
                    warmup_ok = True
                    print(f"[AI-ENGINE] Warm-up attempt 1 (device={_device}) SUCCESS.", flush=True)
                except Exception as we1:
                    print(f"[AI-ENGINE] Warm-up attempt 1 failed: {we1}", flush=True)
                # Attempt 2: tanpa device argument
                if not warmup_ok:
                    try:
                        yolo_model(_dummy, imgsz=warmup_imgsz, verbose=False)
                        warmup_ok = True
                        print(f"[AI-ENGINE] Warm-up attempt 2 (no device) SUCCESS.", flush=True)
                    except Exception as we2:
                        print(f"[AI-ENGINE] Warm-up attempt 2 failed: {we2}", flush=True)
                # Attempt 3: imgsz lebih kecil
                if not warmup_ok:
                    try:
                        _dummy2 = np.zeros((224, 224, 3), dtype=np.uint8)
                        yolo_model(_dummy2, imgsz=224, verbose=False)
                        warmup_ok = True
                        print(f"[AI-ENGINE] Warm-up attempt 3 (imgsz=224) SUCCESS.", flush=True)
                        del _dummy2
                    except Exception as we3:
                        print(f"[AI-ENGINE] Warm-up attempt 3 failed: {we3}", flush=True)
                del _dummy

                if warmup_ok:
                    HAS_YOLO = True
                    print(f"[AI-ENGINE] ✅ {YOLO_MODEL_NAME} loaded & warmed up on {_device}.", flush=True)
                else:
                    # Model ter-load tapi warm-up gagal semua.
                    # Tetap set HAS_YOLO = True karena model mungkin masih bisa inferensi.
                    # Warm-up failure ≠ inference failure.
                    HAS_YOLO = True
                    print(f"[AI-ENGINE] ⚠️  {YOLO_MODEL_NAME} loaded — warm-up gagal tapi model tetap dipakai.", flush=True)
            except Exception as e:
                import traceback
                HAS_YOLO = False
                print(f"[AI-ENGINE] ❌ YOLO load GAGAL - OpenCV HOG fallback aktif.", flush=True)
                print(f"[AI-ENGINE] Error detail: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
    return yolo_model

def load_yolo_heavy():
    global yolo_model_heavy, yolo_model
    if yolo_model_heavy is not None:
        return yolo_model_heavy
    with _yolo_lock:
        if yolo_model_heavy is None:
            try:
                load_yolo_model()
                print(f"[AI-ENGINE] Lazy loading {YOLO_MODEL_HEAVY}...", flush=True)
                import importlib
                ultralytics = importlib.import_module("ultralytics")
                YOLO = ultralytics.YOLO
                import logging
                logging.getLogger("ultralytics").setLevel(logging.WARNING)
                if os.path.exists(YOLO_MODEL_HEAVY):
                    yolo_model_heavy = YOLO(YOLO_MODEL_HEAVY)
                    try:
                        yolo_model_heavy.to(_device)
                    except Exception as te:
                        print(f"[AI-ENGINE] Note: heavy model.to({_device}) skipped: {te}", flush=True)
                    is_openvino = "openvino" in str(YOLO_MODEL_HEAVY).lower()
                    warmup_imgsz = 640 if is_openvino else 416
                    _dummy2 = np.zeros((warmup_imgsz, warmup_imgsz, 3), dtype=np.uint8)
                    try:
                        yolo_model_heavy(_dummy2, imgsz=warmup_imgsz, verbose=False, device=_device)
                    except Exception:
                        yolo_model_heavy(_dummy2, imgsz=warmup_imgsz, verbose=False)
                    del _dummy2
                    print(f"[AI-ENGINE] {YOLO_MODEL_HEAVY} loaded & warmed up on {_device}.", flush=True)
                else:
                    yolo_model_heavy = yolo_model  # fallback to nano if s not found
                    print(f"[AI-ENGINE] {YOLO_MODEL_HEAVY} not found, falling back to {YOLO_MODEL_NAME} for attribute mode.", flush=True)
            except Exception as e:
                print(f"[AI-ENGINE] YOLO heavy load error: {e}", flush=True)
    return yolo_model_heavy

# ── Face Detection & HOG Fallback Setup
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    face_cascade = None

try:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
except Exception:
    hog = None

_yolo_lock = threading.RLock()

# COCO indices
COCO_PERSON   = 0
COCO_HANDBAG  = 26
COCO_BACKPACK = 24
COCO_SUITCASE = 28
BAG_CLASSES   = {COCO_HANDBAG, COCO_BACKPACK, COCO_SUITCASE}

COCO_CLASS_NAMES = {
    0: "Orang",
    1: "Sepeda",
    2: "Mobil",
    3: "Motor",
    5: "Bus",
    7: "Truk",
    24: "Tas",
    26: "Tas",
    28: "Tas",
    56: "Kursi",
    63: "Laptop"
}

CLASS_COLORS = {
    0: (120, 255, 0),     # Orang (bright green)
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
    "person": "Orang",
    "bicycle": "Sepeda",
    "car": "Mobil",
    "motorcycle": "Motor",
    "airplane": "Pesawat",
    "bus": "Bus",
    "train": "Kereta",
    "truck": "Truk",
    "boat": "Perahu",
    "traffic light": "Lampu Lalu Lintas",
    "fire hydrant": "Hydrant Kebakaran",
    "stop sign": "Rambu Stop",
    "parking meter": "Meteran Parkir",
    "bench": "Bangku",
    "bird": "Burung",
    "cat": "Kucing",
    "dog": "Anjing",
    "horse": "Kuda",
    "sheep": "Domba",
    "cow": "Sapi",
    "elephant": "Gajah",
    "bear": "Beruang",
    "zebra": "Zebra",
    "giraffe": "Jerapah",
    "backpack": "Backpack",
    "umbrella": "Payung",
    "handbag": "Tote bag",
    "tie": "Dasi",
    "suitcase": "Koper",
    "frisbee": "Frisbee",
    "skis": "Ski",
    "snowboard": "Papan Salju",
    "sports ball": "Bola Olahraga",
    "kite": "Layang-layang",
    "baseball bat": "Pemukul Bisbol",
    "baseball glove": "Sarung Tangan Bisbol",
    "skateboard": "Skateboard",
    "surfboard": "Papan Seluncur",
    "tennis racket": "Raket Tenis",
    "bottle": "Botol",
    "wine glass": "Gelas Anggur",
    "cup": "Cangkir",
    "fork": "Garpu",
    "knife": "Pisau",
    "spoon": "Sendok",
    "bowl": "Mangkuk",
    "banana": "Pisang",
    "apple": "Apel",
    "sandwich": "Sandwich",
    "orange": "Jeruk",
    "broccoli": "Brokoli",
    "carrot": "Wortel",
    "hot dog": "Hot Dog",
    "pizza": "Pizza",
    "donut": "Donat",
    "cake": "Kue",
    "chair": "Kursi",
    "couch": "Sofa",
    "potted plant": "Tanaman Pot",
    "bed": "Kasur",
    "dining table": "Meja Makan",
    "toilet": "Toilet",
    "tv": "TV",
    "laptop": "Laptop",
    "mouse": "Mouse",
    "remote": "Remote",
    "keyboard": "Keyboard",
    "cell phone": "HP",
    "microwave": "Microwave",
    "oven": "Oven",
    "toaster": "Pemanggang Roti",
    "sink": "Wastafel",
    "refrigerator": "Kulkas",
    "book": "Buku",
    "clock": "Jam",
    "vase": "Vas",
    "scissors": "Gunting",
    "teddy bear": "Boneka Beruang",
    "hair drier": "Pengering Rambut",
    "toothbrush": "Sikat Gigi"
}

def get_class_name(cls_id):
    if not HAS_YOLO or yolo_model is None:
        return COCO_CLASS_NAMES.get(cls_id, "Objek")
    english_name = yolo_model.names.get(cls_id, "object")
    return COCO_TRANSLATIONS.get(english_name.lower(), english_name.capitalize())

def get_class_color(cls_id):
    if cls_id in CLASS_COLORS:
        return CLASS_COLORS[cls_id]
    state_rand = np.random.RandomState(cls_id)
    color = tuple(int(x) for x in state_rand.randint(50, 230, size=3))
    return color

def get_dynamic_font_params(frame_height):
    scale = max(0.8, min(1.25, frame_height / 720.0 * 0.9))
    thickness = 2 if scale < 1.0 else 3
    return scale, thickness

def get_dynamic_box_thickness(frame_height):
    return 2 if frame_height < 900 else 3

def draw_text_with_bg(frame, text, org, color=(255, 255, 255), bg_color=(0, 0, 0)):
    h, w = frame.shape[:2]
    font_scale, thickness = get_dynamic_font_params(h)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    pad_x = 8
    pad_y = 8
    bx1 = max(0, x - pad_x)
    bx2 = min(w, x + tw + pad_x)
    by1 = max(0, y - th - pad_y)
    by2 = min(h, y + baseline + pad_y - 2)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_labeled_box(frame, x1, y1, x2, y2, label, box_color, text_color=(255, 255, 255), box_thickness=None):
    h, w = frame.shape[:2]
    font_scale, thickness = get_dynamic_font_params(h)
    draw_thickness = box_thickness if box_thickness is not None else get_dynamic_box_thickness(h)
    pad_x = 8
    pad_y = 8
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, draw_thickness)
    if label:
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
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
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)


# ── SQLite Database Setup for Drowsiness logs
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
        print("[AI-ENGINE] SQLite database initialized OK.", flush=True)
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
        print(f"[AI-ENGINE] Logged drowsiness for camera_{cam_id} (EAR: {ear:.3f})", flush=True)
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


# ── MediaPipe Setup using new Tasks API
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
                    print("[AI-ENGINE] Downloading face_landmarker.task model...", flush=True)
                    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                    urllib.request.urlretrieve(url, model_path)
                    print("[AI-ENGINE] Download complete.", flush=True)
                    
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


# ── YOLO Pose Setup
HAS_POSE = True
yolo_pose_model = None
POSE_MODEL_NAME = "yolov8n-pose_openvino_model" if os.path.exists("yolov8n-pose_openvino_model") else "yolov8n-pose.pt"

def load_yolo_pose_model():
    global yolo_pose_model, HAS_POSE, yolo_model
    if yolo_pose_model is not None:
        return yolo_pose_model
    with _yolo_lock:
        if yolo_pose_model is None:
            try:
                load_yolo_model()
                print(f"[AI-ENGINE] Lazy loading YOLO Pose model...", flush=True)
                import importlib
                ultralytics = importlib.import_module("ultralytics")
                YOLO = ultralytics.YOLO
                
                yolo_pose_model = YOLO(POSE_MODEL_NAME)
                try:
                    yolo_pose_model.to(_device)
                except Exception as te:
                    print(f"[AI-ENGINE] Note: pose model.to({_device}) skipped: {te}", flush=True)
                is_openvino = "openvino" in str(POSE_MODEL_NAME).lower()
                warmup_imgsz = 640 if is_openvino else 320
                _dummy = np.zeros((warmup_imgsz, warmup_imgsz, 3), dtype=np.uint8)
                try:
                    yolo_pose_model(_dummy, imgsz=warmup_imgsz, verbose=False, device=_device)
                except Exception:
                    yolo_pose_model(_dummy, imgsz=warmup_imgsz, verbose=False)
                del _dummy
                HAS_POSE = True
                print(f"[AI-ENGINE] {POSE_MODEL_NAME} loaded & warmed up on {_device}.", flush=True)
            except Exception as e:
                HAS_POSE = False
                print(f"[AI-ENGINE] Pose model load error: {e}", flush=True)
    return yolo_pose_model


def _nms_rects(rects, overlap_thresh=0.4):
    if len(rects) == 0:
        return []
    boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects], dtype=float)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        iou = np.maximum(0.,xx2-xx1+1)*np.maximum(0.,yy2-yy1+1)
        iou = iou/(areas[i]+areas[order[1:]]-iou)
        order = order[np.where(iou<=overlap_thresh)[0]+1]
    return [rects[i] for i in keep]


def iou_overlap(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1])
    ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
    inter=iw*ih
    if inter==0: return 0.0
    return inter/float((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter)


def run_yolo(frame, target_classes, model=None):
    """Run YOLO inference. Uses yolo_model (nano) by default; pass yolo_model_heavy for attribute mode."""
    detections = []
    active_model = model if model is not None else yolo_model
    if not HAS_YOLO or active_model is None:
        return detections
    try:
        is_openvino = False
        try:
            is_openvino = "openvino" in str(getattr(active_model, "ckpt_path", "") or "").lower()
        except Exception:
            pass
        # Dynamic imgsz on CPU, but OpenVINO models must use 640 to prevent size mismatch
        if is_openvino or _device == "cuda":
            yolo_imgsz = 640
        else:
            yolo_imgsz = 416 if active_model == yolo_model_heavy else 320
        with _yolo_lock:
            with torch.no_grad():
                if target_classes is not None:
                    results = active_model(frame, imgsz=yolo_imgsz, verbose=False, conf=0.35,
                                           classes=list(target_classes), device=_device)
                else:
                    results = active_model(frame, imgsz=yolo_imgsz, verbose=False, conf=0.35,
                                           device=_device)
        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.35: continue
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
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
    except Exception as e:
        print(f"[AI-ENGINE] Error in color classification: {e}", flush=True)
        return "Abu-abu", 50


# ── Mode Handlers

def process_none(frame, meta):
    return frame, meta


def process_face(frame, persons, meta, cached_faces=None):
    h, w = frame.shape[:2]
    out = frame.copy()
    faces = []
    if cached_faces is not None:
        faces = cached_faces
    else:
        if face_cascade is not None:
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            try:
                det = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
                for (fx,fy,fw,fh) in det:
                    faces.append((fx,fy,fw,fh))
            except Exception: pass
            for (px,py,pw,ph,_) in persons:
                crop = gray[max(0,py):max(0,py+int(ph*0.42)), max(0,px):max(0,px+pw)]
                if crop.size < 4: continue
                try:
                    dc = face_cascade.detectMultiScale(crop, scaleFactor=1.1, minNeighbors=4, minSize=(20,20))
                    for (fx,fy,fw,fh) in dc:
                        faces.append((px+fx,py+fy,fw,fh))
                except Exception: pass
            faces = _nms_rects(faces, 0.4)

    meta["faces"] = faces
    meta["face_detected"] = len(faces) > 0
    if faces:
        meta["face_name"] = "Tidak Dikenal"
        for (fx,fy,fw,fh) in faces:
            draw_labeled_box(out, fx, fy, fx+fw, fy+fh, "Tidak Dikenal", (0, 120, 255), (255, 255, 255), 2)
        for (px,py,pw,ph,_) in persons:
            cv2.rectangle(out, (px, py), (px+pw, py+ph), (0, 200, 60), 2)
    else:
        meta["face_name"] = "None"
        draw_text_with_bg(out, "Tidak ada wajah terdeteksi", (10, h - 25), (100, 100, 255), (0, 0, 0))
    
    draw_text_with_bg(out, f"Orang: {len(persons)}", (10, 60), (255, 255, 100), (0, 0, 0))
    return out, meta



def detect_head_attributes_pytorch(head_crop):
    if head_crop is None or head_crop.size == 0:
        return "Tidak", "Tidak"
    try:
        gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        h_h, h_w = gray.shape[:2]
        
        # Sunglasses detection: middle eyes region
        eye_y1 = int(h_h * 0.40)
        eye_y2 = int(h_h * 0.75)
        eye_x1 = int(h_w * 0.25)
        eye_x2 = int(h_w * 0.75)
        
        eye_region = gray[eye_y1:eye_y2, eye_x1:eye_x2]
        has_glasses = "Tidak"
        if eye_region.size > 0:
            avg_val = float(np.mean(eye_region))
            if avg_val < 60:
                has_glasses = "Ya"
                
        # Hat detection: top hair area
        top_y1 = 0
        top_y2 = int(h_h * 0.45)
        top_region = head_crop[top_y1:top_y2, :]
        has_hat = "Tidak"
        if top_region.size > 0:
            hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
            avg_s = float(np.mean(hsv[:,:,1]))
            avg_v = float(np.mean(hsv[:,:,2]))
            if avg_s > 65 or avg_v < 40 or avg_v > 220:
                has_hat = "Ya"
                
        return has_hat, has_glasses
    except Exception:
        return "Tidak", "Tidak"


def process_attribute(frame, persons, detections, bags_raw, meta, cached_colors=None):
    h, w = frame.shape[:2]
    out = frame.copy()
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
        
        # Check aspect ratio for body visibility (detect sitting or occlusion)
        aspect_ratio = ph / pw if pw > 0 else 0
        baju_visible = True
        celana_visible = True

        if aspect_ratio < 0.9:
            celana_visible = False
        if aspect_ratio < 0.4:
            baju_visible = False

        head_y1 = max(0, py)
        head_y2 = min(h, py + int(ph * 0.15))
        head_x1 = max(0, px)
        head_x2 = px2

        ux1 = max(0, px)
        ux2 = px2
        uy1 = max(0, py + int(ph * 0.15))
        uy2 = min(h, py + int(ph * 0.50))

        lx1 = max(0, px)
        lx2 = px2
        ly1 = max(0, py + int(ph * 0.55))
        ly2 = min(h, py + int(ph * 0.90))

        upper_crop = frame[uy1:uy2, ux1:ux2]
        lower_crop = frame[ly1:ly2, lx1:lx2]

        if cached_colors is not None and idx < len(cached_colors):
            top_color, top_conf, bottom_color, bottom_conf, unique_bags = cached_colors[idx]
        else:
            if baju_visible and upper_crop.size > 0:
                top_color, top_conf = get_clothing_color_pytorch(upper_crop)
            else:
                top_color, top_conf = "Tidak Diketahui", 0
                
            if celana_visible and lower_crop.size > 0:
                bottom_color, bottom_conf = get_clothing_color_pytorch(lower_crop)
            else:
                bottom_color, bottom_conf = "Tidak Diketahui", 0

            associated_bags = []
            person_box = [px, py, px2, py2]
            for bag in bag_boxes:
                bx1, by1, bx2, by2 = bag["bbox"]
                bcx = (bx1 + bx2) // 2
                bcy = (by1 + by2) // 2
                if (iou_overlap(person_box, [bx1, by1, bx2, by2]) > 0.05
                        or (px - 20 <= bcx <= px2 + 20 and py - 20 <= bcy <= py2 + 20)):
                    associated_bags.append(bag["name"])
            unique_bags = list(dict.fromkeys(associated_bags))

        current_cached_colors.append((top_color, top_conf, bottom_color, bottom_conf, unique_bags))

        person_label = f"Orang: {int(conf * 100)}%"
        draw_labeled_box(out, px, py, px2, py2, person_label, person_color, (255, 255, 255))

        # Calculate combined confidence (person detection certainty * color similarity)
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
            "id": idx + 1,
            "class": "Orang",
            "conf": round(float(conf), 2),
            "attributes": person_attrs
        }
        person_entries.append(person_entry)
        add_detected_object("Orang", conf, person_attrs)

    for (x1, y1, x2, y2, cls, conf) in detections:
        if cls == COCO_PERSON:
            continue

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


def process_sop(cam_id, frame, persons, state, meta):
    h, w = frame.shape[:2]
    out = frame.copy()
    now = time.time()
    
    rx1,ry1 = int(w*0.3),int(h*0.3)
    rx2,ry2 = int(w*0.7),int(h*0.7)
    
    in_roi  = any(rx1<=px+pw//2<=rx2 and ry1<=py+ph//2<=ry2 for (px,py,pw,ph,_) in persons)
    ts = state.setdefault("sop_timer", {"active":False,"start_time":0.0,"duration":0.0})
    
    if in_roi:
        if not ts["active"]:
            ts.update({"active":True,"start_time":now,"duration":0.0})
        else:
            ts["duration"] = now-ts["start_time"]
    else:
        ts.update({"active":False,"duration":0.0})
        
    dur  = ts["duration"]
    viol = dur > 5.0
    rc   = (0,0,255) if viol else (0,220,220)
    
    draw_labeled_box(out, rx1, ry1, rx2, ry2, "AREA SOP UTAMA", rc, (255, 255, 255))
    draw_text_with_bg(out, f"Operator di ROI: {'YA' if in_roi else 'TIDAK'}", (12, 72), (255, 255, 255), rc)
    draw_text_with_bg(out, f"Durasi: {dur:.1f}s | Batas: 5.0s", (12, 114), (255, 255, 255), (15, 23, 42))
    
    if viol and int(now*2)%2==0:
        alert_y = min(h - 16, ry2 + 34)
        draw_text_with_bg(out, "SOP MELANGGAR!", (rx1, alert_y), (255, 255, 255), (0, 0, 255))
        
    for (px,py,pw,ph,_) in persons:
        c = rc if (rx1<=px+pw//2<=rx2 and ry1<=py+ph//2<=ry2) else (0,200,0)
        draw_labeled_box(out, px, py, px+pw, py+ph, None, c)
        
    if not persons:
        draw_text_with_bg(out, "Tidak ada orang terdeteksi", (12, h - 18), (255, 255, 255), (80, 80, 180))
        
    meta["sop_status"]     = "MELANGGAR" if viol else "AMAN"
    meta["sop_duration_s"] = round(dur, 1)
    return out, meta


def process_contraflow(cam_id, frame, persons, state, meta):
    h, w = frame.shape[:2]
    out = frame.copy()
    now = time.time()
    
    lx = int(w*0.5)
    lc = (255,128,0)
    line_thickness = get_dynamic_box_thickness(h)
    cv2.line(out,(lx,0),(lx,h),lc,line_thickness)
    draw_text_with_bg(out, "JALUR CONTRAFLOW (BATAS)", (min(w - 260, lx + 10), 32), (255, 255, 255), lc)
    
    centroids = []
    for (px,py,pw,ph,_) in persons:
        cx_c,cy_c = px+pw//2, py+ph//2
        centroids.append((cx_c,cy_c))
        draw_labeled_box(out, px, py, px+pw, py+ph, None, (0,200,0))
        
    tracks = state.setdefault("contraflow_tracks", [])
    upd = []
    alarm = False
    
    for cx_c,cy_c in centroids:
        matched = None; md = 80.0
        for t in tracks:
            lhx,lhy = t["history"][-1][:2]
            d = float(np.sqrt((cx_c-lhx)**2+(cy_c-lhy)**2))
            if d < md: md=d; matched=t
        if matched:
            matched["history"].append((cx_c,cy_c,now)); matched["last_seen"]=now
            if len(matched["history"])>=5 and matched["history"][-5][0]>lx and cx_c<lx:
                matched["alert"]=True
            if matched["alert"]: alarm=True
            upd.append(matched)
        else:
            upd.append({"id":len(upd)+1,"history":[(cx_c,cy_c,now)],"last_seen":now,"alert":False})
            
    state["contraflow_tracks"] = [t for t in upd if now-t["last_seen"]<2.5]
    
    if alarm:
        if int(now*2)%2==0:
            draw_text_with_bg(out, "CONTRAFLOW DETECTED!", (max(12, lx - 210), h - 20), (255, 255, 255), (0, 0, 255))
        cv2.arrowedLine(out,(lx+60,h//2),(lx-60,h//2),(0,0,255),max(3, line_thickness + 1),tipLength=0.3)
        
    meta["contraflow_status"] = "MELANGGAR" if alarm else "AMAN"
    meta["contraflow_alarm"]  = alarm
    
    for t in state["contraflow_tracks"]:
        c = (0,0,255) if t["alert"] else (0,200,80)
        for pt in t["history"][-15:]:
            cv2.circle(out,(pt[0],pt[1]),max(4, h // 260),c,-1)
            
    if not persons:
        draw_text_with_bg(out, "Tidak ada orang terdeteksi", (12, h - 18), (255, 255, 255), (80, 80, 180))
        
    return out, meta


def process_drowsiness(cam_id, frame, state, meta):
    h, w = frame.shape[:2]
    out = frame.copy()
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
        
        # Calculate eye aspect ratios (EAR)
        left_ear = calculate_ear([362, 385, 387, 263, 373, 380], landmarks, w, h)
        right_ear = calculate_ear([33, 160, 158, 133, 153, 144], landmarks, w, h)
        ear = (left_ear + right_ear) / 2.0
        
        # Threshold logic
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
    out = frame.copy()
    
    if cached_pose_data is None:
        load_yolo_pose_model()
        if not HAS_POSE or yolo_pose_model is None:
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
        ((0, 1), head_color),
        ((0, 2), head_color),
        ((1, 3), head_color),
        ((2, 4), head_color),
        ((5, 6), torso_color),
        ((5, 11), torso_color),
        ((6, 12), torso_color),
        ((11, 12), torso_color),
        ((5, 7), arm_color),
        ((7, 9), arm_color),
        ((6, 8), arm_color),
        ((8, 10), arm_color),
        ((11, 13), leg_color),
        ((13, 15), leg_color),
        ((12, 14), leg_color),
        ((14, 16), leg_color),
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
                if idx in [0, 1, 2, 3, 4]:
                    c_color = head_color
                elif idx in [5, 6, 11, 12]:
                    c_color = torso_color
                elif idx in [7, 8, 9, 10]:
                    c_color = arm_color
                else:
                    c_color = leg_color
                cv2.circle(out, pt_coords, joint_radius, c_color, -1, cv2.LINE_AA)
                
        meta["pose_data"] = cached_pose_data
        meta["people_count"] = len(cached_pose_data)
        return out, meta

    is_openvino = False
    try:
        is_openvino = "openvino" in str(getattr(yolo_pose_model, "ckpt_path", "") or "").lower()
    except Exception:
        pass
    pose_imgsz = 640 if (_device == "cuda" or is_openvino) else 320
    with torch.no_grad():
        results = yolo_pose_model.track(
            infer_fr,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=0.35,
            iou=0.55,
            device=_device,
            imgsz=pose_imgsz
        )
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
                if conf < 0.35:
                    continue
                
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
                        "id": track_id,
                        "conf": round(conf, 2),
                        "keypoints": xy_list,
                        "bbox": [x1_n, y1_n, x2_n, y2_n]
                    })
                    
                    for (p1, p2), color in connections:
                        if p1 in pts and p2 in pts:
                            cv2.line(out, pts[p1], pts[p2], color, line_thickness, cv2.LINE_AA)
                            
                    for idx, pt_coords in pts.items():
                        if idx < len(xy_list_raw):
                            if idx in [0, 1, 2, 3, 4]:
                                c_color = head_color
                            elif idx in [5, 6, 11, 12]:
                                c_color = torso_color
                            elif idx in [7, 8, 9, 10]:
                                c_color = arm_color
                            else:
                                c_color = leg_color
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
    "Kepatuhan SOP (Timer ROI)": "sop",
    "sop": "sop",
    "Deteksi Arah (Contraflow)": "contraflow",
    "contraflow": "contraflow",
    "Deteksi Kantuk (EAR)": "drowsiness",
    "drowsiness": "drowsiness",
    "Pose Estimation & Human Tracking": "pose",
    "pose": "pose"
}

def run_analytics(cam_id, frame, mode_raw, selected_classes, state):
    t_now = time.time()
    # Ensure we ALWAYS use a copy to prevent modifying the original stream buffer
    frame = frame.copy()
    
    # Downscale the frame immediately to 640px max width to prevent heavy CPU usage on high resolutions
    h, w = frame.shape[:2]
    target_w = 640
    if w > target_w:
        scale = target_w / w
        target_h = int(h * scale)
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        h, w = target_h, target_w

    # ── FPS Calculation
    t_now = time.time()
    if "last_timestamp" in state:
        time_diff = t_now - state["last_timestamp"]
        if time_diff > 0:
            current_fps = 1.0 / time_diff
            state["fps"] = state.get("fps", current_fps) * 0.9 + current_fps * 0.1
    else:
        state["fps"] = 10.0
    state["last_timestamp"] = t_now

    mode = MODE_MAPPING.get(mode_raw, "none")
    
    # Increment frame count
    state["frame_count"] = state.get("frame_count", 0) + 1

    # Log actual real-time FPS per mode to server terminal every 30 frames
    if state["frame_count"] % 30 == 0:
        print(f"[FPS-LOG] Camera {cam_id} | Mode: {mode} | FPS: {state.get('fps', 0.0):.2f}", flush=True)
    
    # If mode is none, just return frame directly (no skip needed since it's light)
    if mode == 'none':
        draw_text_with_bg(frame, "AI: OFF", (12, 28), (255, 255, 255), (0, 0, 255))
        if "last_processed" in state:
            del state["last_processed"]
        return frame, {
            "engine": "YOLOv8s",
            "camera_status": "Aktif",
            "people_count": 0,
            "face_detected": False,
            "face_name": "None",
            "attributes": [],
            "object_counts": {},
            "detected_objects_list": [],
            "sop_status": "AMAN",
            "sop_duration_s": 0.0,
            "contraflow_status": "AMAN", "contraflow_alarm": False,
            "drowsiness_status": "Normal", "ear_value": 0.0, "alarm": False, "pose_data": [],
            "fps": 0.0
        }

    # Target exactly 5 AI inferences per second on CPU (200ms interval)
    AI_INTERVAL = 0.20
    last_infer = state.get("last_inference_time", 0.0)
    time_since_last = t_now - last_infer
    
    should_skip = False
    if "last_metadata" in state and state.get("last_mode") == mode:
        if _device == "cuda" or mode == 'drowsiness':
            should_skip = False  # GPU or Drowsiness (needs continuous EAR) -> always infer
        elif time_since_last < AI_INTERVAL:
            should_skip = True   # CPU-heavy -> skip if under 200ms
            
    if should_skip:
        last_meta = state["last_metadata"].copy()
        eng_lbl = "YOLOv8s (AKTIF)" if HAS_YOLO else "OpenCV HOG Fallback"
        draw_text_with_bg(frame, f"AI: {eng_lbl}", (12, 28), (15, 23, 42), (0, 255, 120))
        
        try:
            if mode == 'face':
                processed, meta = process_face(frame, state.get("last_persons_raw", []), last_meta, cached_faces=last_meta.get("faces"))
            elif mode == 'attribute':
                processed, meta = process_attribute(frame, state.get("last_persons_raw", []), state.get("last_detections", []), state.get("last_bags_raw", []), last_meta, cached_colors=last_meta.get("cached_colors"))
            elif mode == 'sop':
                processed, meta = process_sop(cam_id, frame, state.get("last_persons_raw", []), state, last_meta)
            elif mode == 'contraflow':
                processed, meta = process_contraflow(cam_id, frame, state.get("last_persons_raw", []), state, last_meta)
            elif mode == 'pose':
                processed, meta = process_pose(frame, None, 1.0, last_meta, cached_pose_data=last_meta.get("pose_data"))
            else:
                processed, meta = frame.copy(), last_meta
        except Exception as dispatch_err:
            import traceback
            print(f"[AI-ENGINE] CACHED DISPATCH ERROR cam_{cam_id} mode={mode}: {dispatch_err}", flush=True)
            print(traceback.format_exc(), flush=True)
            processed = frame.copy()
            draw_text_with_bg(processed, f"AI ERR: {mode}", (12, h - 18), (255, 255, 255), (0, 0, 200))
            
        fps_text = f"FPS: {state.get('fps', 0.0):.1f}"
        draw_text_with_bg(processed, fps_text, (max(12, w - 170), 28), (15, 23, 42), (0, 255, 255))
        
        state["last_processed"] = processed.copy()
        state["last_metadata"] = meta.copy()
        return processed, meta

    # Otherwise, run full inference
    eng_lbl = "YOLOv8s (AKTIF)" if HAS_YOLO else "OpenCV HOG Fallback"
    draw_text_with_bg(frame, f"AI: {eng_lbl}", (12, 28), (15, 23, 42), (0, 255, 120))

    # 1. Resize frame for faster AI inference
    # drowsiness uses MediaPipe (processes at original res — it's fast enough)
    # pose/attribute: downscale to 640 wide max
    # face/sop/contraflow: downscale to 480 wide max (very light objects)
    if mode == 'drowsiness':
        infer_fr = frame
        scale = 1.0
    elif mode in ['attribute', 'pose']:
        infer_w = 640
        if w > infer_w:
            scale = infer_w / w
            infer_h = int(h * scale)
            infer_fr = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
        else:
            infer_fr = frame
            scale = 1.0
    else:
        # face/sop/contraflow — use smaller resize
        infer_w = 480
        if w > infer_w:
            scale = infer_w / w
            infer_h = int(h * scale)
            infer_fr = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
        else:
            infer_fr = frame
            scale = 1.0

    # 2. Run object detection base (YOLO or HOG) for relevant modes
    persons_raw = []
    detections = []
    bags_raw = []
    
    meta = {
        "engine": "YOLOv8s" if HAS_YOLO else "OpenCV HOG",
        "camera_status": "Aktif",
        "people_count": 0,
        "face_detected": False,
        "face_name": "None",
        "attributes": [],
        "object_counts": {},
        "detected_objects_list": [],
        "sop_status": "AMAN",
        "sop_duration_s": 0.0,
        "contraflow_status": "AMAN",
        "contraflow_alarm": False,
        "drowsiness_status": "Normal",
        "ear_value": 0.0,
        "alarm": False,
        "pose_data": [],
        "fps": round(state.get("fps", 0.0), 1)
    }

    if mode in ['face', 'attribute', 'sop', 'contraflow']:
        if mode == 'attribute':
            active_model = load_yolo_heavy()
        else:
            active_model = load_yolo_model()
            
        if HAS_YOLO and active_model is not None:
            # attribute uses the heavier model for better multi-class accuracy
            yolo_classes = None if mode == 'attribute' else {0}
            
            raw_detections = run_yolo(infer_fr, yolo_classes, model=active_model)
            for (x1, y1, x2, y2, cls, conf) in raw_detections:
                if scale != 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                detections.append((x1, y1, x2, y2, cls, conf))
                if cls == 0:
                    persons_raw.append((x1, y1, x2 - x1, y2 - y1, conf))
                elif cls in BAG_CLASSES:
                    bags_raw.append((x1, y1, x2 - x1, y2 - y1, cls, conf))
        else:
            if hog is not None:
                try:
                    rects, _ = hog.detectMultiScale(infer_fr, winStride=(8,8), padding=(8,8), scale=1.05)
                    for (rx, ry, rw, rh) in rects:
                        x1, y1, x2, y2 = rx, ry, rx + rw, ry + rh
                        if scale != 1.0:
                            x1 = int(x1 / scale)
                            y1 = int(y1 / scale)
                            x2 = int(x2 / scale)
                            y2 = int(y2 / scale)
                        persons_raw.append((x1, y1, x2 - x1, y2 - y1, 0.6))
                        detections.append((x1, y1, x2, y2, 0, 0.6))
                except Exception:
                    pass

    # Dispatch to specific mode processor — wrapped per-mode to prevent generator crash
    try:
        if mode == 'face':
            processed, meta = process_face(frame, persons_raw, meta)
        elif mode == 'attribute':
            processed, meta = process_attribute(frame, persons_raw, detections, bags_raw, meta)
        elif mode == 'sop':
            processed, meta = process_sop(cam_id, frame, persons_raw, state, meta)
        elif mode == 'contraflow':
            processed, meta = process_contraflow(cam_id, frame, persons_raw, state, meta)
        elif mode == 'drowsiness':
            processed, meta = process_drowsiness(cam_id, frame, state, meta)
        elif mode == 'pose':
            processed, meta = process_pose(frame, infer_fr, scale, meta)
        else:
            processed, meta = frame.copy(), meta
    except Exception as dispatch_err:
        import traceback
        print(f"[AI-ENGINE] DISPATCH ERROR cam_{cam_id} mode={mode}: {dispatch_err}", flush=True)
        print(traceback.format_exc(), flush=True)
        processed = frame.copy()
        draw_text_with_bg(processed, f"AI ERR: {mode}", (12, h - 18), (255, 255, 255), (0, 0, 200))

    # Draw FPS on processed frame
    fps_text = f"FPS: {state.get('fps', 0.0):.1f}"
    draw_text_with_bg(processed, fps_text, (max(12, w - 170), 28), (15, 23, 42), (0, 255, 255))

    # Save to state cache
    state["last_processed"] = processed.copy()
    state["last_metadata"] = meta.copy()
    state["last_mode"] = mode
    state["last_persons_raw"] = persons_raw
    state["last_detections"] = detections
    state["last_bags_raw"] = bags_raw
    state["last_inference_time"] = t_now
    
    return processed, meta
