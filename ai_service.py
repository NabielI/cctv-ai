# CRITICAL: Import torch and torchvision BEFORE cv2 to prevent OpenMP conflict/runtime library errors on ARM64!
import torch
import torchvision

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;100000"
os.environ["ULTRALYTICS_TELEMETRY"] = "false"
os.environ["ULTRALYTICS_CHECK"] = "false"
os.environ["OPENVINO_TELEMETRY_OPTOUT"] = "1"
os.environ["OV_TELEMETRY_OPTOUT"] = "1"

import cv2
import numpy as np
import time
import traceback
import sqlite3
import json
from fastapi import FastAPI, Response, Request, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import modular components
from camera_manager import CameraManager
from analytics_engine import (
    run_analytics, load_yolo_model, load_yolo_heavy, load_mediapipe, load_yolo_pose_model,
    register_face, get_registered_faces_list, delete_registered_face
)
from zone_monitor import get_zone_monitor, ZoneConfig

app = FastAPI(title="NVR AI Analytics Service")

os.makedirs("uploads/faces", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registries
camera_manager = CameraManager()
camera_metadata = {}
camera_states = {}

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Constants
MAX_CONCURRENT_AI_CAMERAS = int(os.environ.get("MAX_CONCURRENT_AI_CAMERAS", 4))

def auto_register_cameras():
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            for cam in config.get("cameras", []):
                cam_id = cam.get("id")
                url = cam.get("url")
                if cam_id is not None and url:
                    print(f"[AI-SERVICE] Auto-registering camera_{cam_id}: {url}", flush=True)
                    camera_manager.add_camera(cam_id, url)
        except Exception as e:
            print(f"[AI-SERVICE] Error in auto-registering cameras: {e}", flush=True)

@app.on_event("startup")
def startup_event():
    auto_register_cameras()
    # Inisialisasi Zone Monitor (thread independen, berjalan terpisah dari AI mode)
    try:
        zm = get_zone_monitor()
        print("[AI-SERVICE] ZoneMonitor started successfully.", flush=True)
    except Exception as e:
        print(f"[AI-SERVICE] Warning: ZoneMonitor failed to start: {e}", flush=True)

# Pydantic schemas
class CameraRegister(BaseModel):
    cam_id: int
    source_url: str

class ModeChange(BaseModel):
    cam_id: Optional[int] = None
    mode: str
    selected_classes: Optional[List[int]] = None

# Helpers
def get_camera_state(cam_id: int) -> dict:
    if cam_id not in camera_states:
        camera_states[cam_id] = {}
    return camera_states[cam_id]


# ── REST API Endpoints

@app.post("/cameras")
def api_add_camera(cam: CameraRegister):
    stream = camera_manager.add_camera(cam.cam_id, cam.source_url)
    if stream:
        return {"success": True, "cam_id": cam.cam_id, "message": "Camera added and started successfully"}
    return JSONResponse(status_code=500, content={"success": False, "message": "Failed to initialize camera"})


@app.delete("/cameras/{camera_id}")
def api_remove_camera(camera_id: int):
    camera_manager.remove_camera(camera_id)
    if camera_id in camera_metadata:
        del camera_metadata[camera_id]
    if camera_id in camera_states:
        del camera_states[camera_id]
    return {"success": True, "message": f"Camera {camera_id} removed"}


@app.post("/cameras/{camera_id}/mode")
def api_set_mode(camera_id: int, mode_data: ModeChange):
    cam = camera_manager.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    if mode_data.mode in ['sop', 'contraflow', 'gesture']:
        raise HTTPException(status_code=400, detail="Mode yang diminta telah dinonaktifkan")

    # Check limit before setting mode
    if mode_data.mode != "none":
        active_ai_count = 0
        for cid, other_cam in camera_manager.cameras.items():
            if cid != camera_id and other_cam.get_mode() != "none":
                active_ai_count += 1
        
        if active_ai_count >= MAX_CONCURRENT_AI_CAMERAS:
            raise HTTPException(
                status_code=400,
                detail=f"Maksimal {MAX_CONCURRENT_AI_CAMERAS} kamera bisa menjalankan mode AI bersamaan sesuai kapasitas hardware, matikan salah satu kamera lain dulu"
            )
            
        # Pre-load model dynamically to prevent stream timeout/freeze in MJPEG thread
        try:
            print(f"[AI-SERVICE] Pre-loading model for mode: {mode_data.mode}", flush=True)
            if mode_data.mode == 'face':
                load_yolo_model()
            elif mode_data.mode in ['attribute', 'vehicle']:
                load_yolo_heavy()
            elif mode_data.mode == 'drowsiness':
                load_mediapipe()
            elif mode_data.mode == 'pose':
                load_yolo_pose_model()
            print(f"[AI-SERVICE] Model pre-loaded successfully for mode: {mode_data.mode}", flush=True)
        except Exception as e:
            print(f"[AI-SERVICE] Error loading model for mode {mode_data.mode}: {e}", flush=True)
            
    cam.set_mode(mode_data.mode, mode_data.selected_classes)
    return {"success": True, "cam_id": camera_id, "mode": mode_data.mode}


# ── Compatibility API Endpoints for NodeJS (server.js)

@app.post("/register")
def compat_register(cam: CameraRegister):
    stream = camera_manager.add_camera(cam.cam_id, cam.source_url)
    if stream:
        return {"success": True, "cam_id": cam.cam_id}
    return JSONResponse(status_code=500, content={"success": False, "error": "Failed to start stream"})


@app.post("/mode")
@app.post("/api/mode")
@app.post("/api/ai/mode")
@app.post("/api/ai/{cam_id}/mode")
def compat_set_mode(mode_data: ModeChange, cam_id: Optional[int] = None):
    target_cam_id = cam_id if cam_id is not None else (mode_data.cam_id if mode_data.cam_id is not None else 0)
    cam = camera_manager.get_camera(target_cam_id)
    if not cam:
        return JSONResponse(status_code=404, content={"success": False, "error": "Camera not registered"})
    
    if mode_data.mode in ['sop', 'contraflow', 'gesture']:
        return JSONResponse(status_code=400, content={"success": False, "error": "Mode yang diminta telah dinonaktifkan"})

    # Check limit before setting mode
    if mode_data.mode != "none":
        active_ai_count = 0
        for cid, other_cam in camera_manager.cameras.items():
            if cid != target_cam_id and other_cam.get_mode() != "none":
                active_ai_count += 1
        
        if active_ai_count >= MAX_CONCURRENT_AI_CAMERAS:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Maksimal {MAX_CONCURRENT_AI_CAMERAS} kamera bisa menjalankan mode AI bersamaan sesuai kapasitas hardware, matikan salah satu kamera lain dulu"
                }
            )
            
        # Pre-load model dynamically to prevent stream timeout/freeze in MJPEG thread
        try:
            print(f"[AI-SERVICE] Pre-loading model for mode: {mode_data.mode}", flush=True)
            if mode_data.mode == 'face':
                load_yolo_model()
            elif mode_data.mode in ['attribute', 'vehicle']:
                load_yolo_heavy()
            elif mode_data.mode == 'drowsiness':
                load_mediapipe()
            elif mode_data.mode == 'pose':
                load_yolo_pose_model()
            print(f"[AI-SERVICE] Model pre-loaded successfully for mode: {mode_data.mode}", flush=True)
        except Exception as e:
            print(f"[AI-SERVICE] Error loading model for mode {mode_data.mode}: {e}", flush=True)
            
    cam.set_mode(mode_data.mode, mode_data.selected_classes)
    return {"success": True, "cam_id": target_cam_id, "mode": mode_data.mode}


@app.get("/active_modes")
@app.get("/api/active_modes")
@app.get("/api/ai/active_modes")
def api_get_active_modes():
    modes = {}
    for cid, cam in camera_manager.cameras.items():
        modes[cid] = cam.get_mode()
    return modes


def sanitize_metadata(obj):
    if isinstance(obj, dict):
        return {k: sanitize_metadata(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_metadata(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return sanitize_metadata(obj.tolist())
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj


@app.get("/metadata")
@app.get("/api/metadata")
@app.get("/api/ai/metadata")
@app.get("/api/ai/{cam_id}/metadata")
def compat_get_metadata(cam_id: int = 0):
    cam = camera_manager.get_camera(cam_id)
    mode = cam.get_mode() if cam else "none"
    sel_classes = cam.get_selected_classes() if cam else None
    
    default_meta = {
        "engine": "YOLO26",
        "backend": "OpenVINO",
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
        "infer_time_ms": 0.0,
        "ai_fps": 0.0,
        "stream_fps": 0.0,
        "cpu_usage": 0.0
    }
    
    meta = camera_metadata.get(cam_id, default_meta).copy()
    meta["mode"] = mode
    meta["selected_classes"] = sel_classes
    return sanitize_metadata(meta)


@app.get("/drowsiness/history")
def api_get_drowsiness_history(cam_id: Optional[int] = Query(None)):
    try:
        conn = sqlite3.connect("drowsiness_logs.db")
        cursor = conn.cursor()
        if cam_id is not None:
            cursor.execute(
                "SELECT id, cam_id, timestamp, ear FROM drowsiness_logs "
                "WHERE cam_id = ? ORDER BY id DESC LIMIT 100", (cam_id,))
        else:
            cursor.execute(
                "SELECT id, cam_id, timestamp, ear FROM drowsiness_logs "
                "ORDER BY id DESC LIMIT 100")
        rows = cursor.fetchall()
        conn.close()
        logs = []
        for r in rows:
            logs.append({"id": r[0], "cam_id": r[1], "timestamp": r[2], "ear": round(r[3], 3)})
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database read error: {e}")


def _make_error_frame(msg: str) -> bytes:
    """Create a small JPEG error frame to keep the stream alive."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (20, 20, 40)
    cv2.putText(img, msg[:80], (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return enc.tobytes() if ok else b''


def mjpeg_generator(cam_id: int):
    print(f"[AI-SERVICE] MJPEG client connected for camera_{cam_id}", flush=True)
    loading_step = 0
    loop_count = 0
    TARGET_FPS = 30
    frame_interval = 1.0 / TARGET_FPS
    last_frame_time = 0.0

    while True:
        try:
            # Strict 30 FPS pacing — never block on AI speed
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

            cam = camera_manager.get_camera(cam_id)

            if not cam:
                raw = _make_error_frame("Camera not registered...")
                if raw:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                           + str(len(raw)).encode() + b'\r\n\r\n' + raw + b'\r\n')
                time.sleep(0.5)
                continue

            frame = cam.get_frame()
            if frame is None:
                img = np.zeros((360, 640, 3), dtype=np.uint8)
                img[:] = (15, 23, 42)
                loading_step = (loading_step + 1) % 4
                dots = "." * loading_step
                cv2.putText(img, f"Connecting to Camera {dots}", (140, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 200), 2, cv2.LINE_AA)
                ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ok:
                    raw = enc.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                           + str(len(raw)).encode() + b'\r\n\r\n' + raw + b'\r\n')
                continue

            # get_ai_frame always returns instantly (cached AI overlay on fresh raw frame)
            processed, meta = cam.get_ai_frame()
            if processed is None:
                processed = frame
            camera_metadata[cam_id] = meta

            # Feed frame ke ZoneMonitor (independen di background, tanpa menimpa overlay di stream AI utama)
            try:
                from zone_monitor import get_zone_monitor
                zm = get_zone_monitor()
                zm.feed_frame(cam_id, frame)
            except Exception:
                pass

            h, w = processed.shape[:2]
            if w > 640:
                scale = 640.0 / w
                new_w, new_h = int(w * scale), int(h * scale)
                processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            ok, enc = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 55])
            if ok:
                raw = enc.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                       + str(len(raw)).encode() + b'\r\n\r\n' + raw + b'\r\n')

            loop_count += 1
            if loop_count % 300 == 0:
                import gc
                gc.collect()

        except Exception as outer_err:
            print(f"[AI-SERVICE] GENERATOR OUTER ERROR cam_{cam_id}: {outer_err}", flush=True)
            print(traceback.format_exc(), flush=True)
            raw = _make_error_frame(f"Stream error: {str(outer_err)[:60]}")
            if raw:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                       + str(len(raw)).encode() + b'\r\n\r\n' + raw + b'\r\n')
            time.sleep(0.5)


@app.get("/video_feed/{camera_id}")
def api_video_feed(camera_id: int):
    return StreamingResponse(
        mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/stream")
@app.get("/api/ai/stream")
def compat_stream(cam_id: int = Query(0)):
    return StreamingResponse(
        mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/api/snapshot/{camera_id}")
def api_camera_snapshot(camera_id: int):
    """Grab latest JPEG frame from CameraManager OpenCV stream."""
    cam = camera_manager.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    frame = cam.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Frame not available yet")
    success, encoded_img = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


# ══════════════════════════════════════════════════
#  Zone Monitoring REST API
# ══════════════════════════════════════════════════

class ZoneConfigRequest(BaseModel):
    zone_id: str
    cam_id: int
    name: str
    coords: list          # [[x_norm, y_norm], ...]
    threshold_minutes: Optional[int] = 15
    cycle_hours: Optional[int] = 1
    telegram_enabled: Optional[bool] = True


@app.get("/api/zones")
@app.get("/api/zones/")
def api_get_zones(cam_id: Optional[int] = Query(None)):
    """Dapatkan semua zona, opsional filter per kamera."""
    zm = get_zone_monitor()
    zones = zm.get_zones(cam_id=cam_id)
    return {"success": True, "data": [z.to_dict() for z in zones]}


@app.post("/api/zones")
@app.post("/api/zones/")
def api_save_zone(req: ZoneConfigRequest):
    """Tambah atau update satu zona."""
    zm = get_zone_monitor()
    zone = ZoneConfig(
        zone_id=req.zone_id,
        cam_id=req.cam_id,
        name=req.name,
        coords=req.coords,
        threshold_minutes=req.threshold_minutes or 15,
        cycle_hours=req.cycle_hours or 1,
        telegram_enabled=req.telegram_enabled if req.telegram_enabled is not None else True,
    )
    zm.set_zone(zone)
    return {"success": True, "zone_id": zone.zone_id, "message": f"Zona '{zone.name}' disimpan."}


@app.delete("/api/zones/{zone_id}")
def api_delete_zone(zone_id: str):
    """Hapus zona berdasarkan zone_id."""
    zm = get_zone_monitor()
    removed = zm.delete_zone(zone_id)
    if removed:
        return {"success": True, "message": f"Zona '{zone_id}' dihapus."}
    raise HTTPException(status_code=404, detail=f"Zona '{zone_id}' tidak ditemukan.")


@app.get("/api/zones/status")
@app.get("/api/zones/status/")
def api_zone_status(cam_id: Optional[int] = Query(None)):
    """Status real-time semua zona (akumulasi menit saat ini dalam siklus jam)."""
    zm = get_zone_monitor()
    status = zm.get_zone_status(cam_id=cam_id)
    return {"success": True, "data": status}


@app.get("/api/zones/history")
@app.get("/api/zones/history/")
def api_zone_history(
    cam_id: Optional[int] = Query(None),
    zone_id: Optional[str] = Query(None),
    limit: int = Query(100)
):
    """Riwayat event/pelanggaran zona."""
    zm = get_zone_monitor()
    history = zm.get_history(cam_id=cam_id, zone_id=zone_id, limit=limit)
    return {"success": True, "data": history}


@app.post("/api/zones/test_evaluate")
def api_test_evaluate():
    """Trigger evaluasi manual untuk testing (tanpa tunggu jam bulat)."""
    zm = get_zone_monitor()
    result = zm.trigger_test_evaluation()
    return result


@app.post("/api/zones/test_telegram")
def api_test_telegram():
    """Kirim pesan test ke Telegram untuk verifikasi konfigurasi."""
    from telegram_notifier import send_test_message, get_bot_info
    bot_info = get_bot_info()
    sent = send_test_message()
    return {
        "success": sent,
        "bot_info": bot_info,
        "message": "Pesan test terkirim!" if sent else "Gagal kirim. Periksa TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID di .env"
    }


try:
    @app.post("/api/faces/register")
    async def api_register_face(name: str = Form(...), photos: List[UploadFile] = File(...)):
        if not name or not name.strip():
            raise HTTPException(status_code=400, detail="Nama wajib diisi")
        if not photos or len(photos) == 0:
            raise HTTPException(status_code=400, detail="Minimal upload 1 foto wajah")
            
        image_bytes_list = []
        for file in photos:
            content = await file.read()
            if content:
                image_bytes_list.append(content)
                
        if not image_bytes_list:
            raise HTTPException(status_code=400, detail="Foto tidak valid")
            
        saved_count = register_face(name.strip(), image_bytes_list)
        if saved_count == 0:
            raise HTTPException(status_code=400, detail="Tidak ada wajah terdeteksi dalam foto. Gunakan foto dengan posisi wajah terlihat jelas.")
            
        return {"status": "success", "count": saved_count, "message": f"Berhasil mendaftarkan {saved_count} foto wajah untuk {name}"}
except Exception as m_err:
    print(f"[AI-SERVICE] Note: Face registration route skipped ({m_err}). Install python-multipart if needed.", flush=True)


@app.get("/api/faces/list")
def api_list_faces():
    faces = get_registered_faces_list()
    return {"status": "success", "data": faces}


@app.delete("/api/faces/{face_id}")
def api_delete_face(face_id: int):
    success = delete_registered_face(face_id)
    if not success:
        raise HTTPException(status_code=404, detail="Data wajah tidak ditemukan")
    return {"status": "success", "message": "Data wajah berhasil dihapus"}


if __name__ == '__main__':
    import uvicorn
    print(f"[AI-SERVICE] Engine starting up with FastAPI...", flush=True)
    uvicorn.run("ai_service:app", host="0.0.0.0", port=5001, reload=False, workers=1)
