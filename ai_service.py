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
from fastapi import FastAPI, Response, Request, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import modular components
from camera_manager import CameraManager
from analytics_engine import run_analytics, load_yolo_model, load_yolo_heavy, load_mediapipe, load_yolo_pose_model

app = FastAPI(title="NVR AI Analytics Service")

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

# Constants
MAX_CONCURRENT_AI_CAMERAS = int(os.environ.get("MAX_CONCURRENT_AI_CAMERAS", 2))

import json
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
    
    if mode_data.mode in ['sop', 'contraflow']:
        raise HTTPException(status_code=400, detail="Mode SOP dan Contraflow telah dinonaktifkan")

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
            elif mode_data.mode == 'attribute':
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
def compat_set_mode(mode_data: ModeChange):
    cam_id = mode_data.cam_id if mode_data.cam_id is not None else 0
    cam = camera_manager.get_camera(cam_id)
    if not cam:
        return JSONResponse(status_code=404, content={"success": False, "error": "Camera not registered"})
    
    if mode_data.mode in ['sop', 'contraflow']:
        return JSONResponse(status_code=400, content={"success": False, "error": "Mode SOP dan Contraflow telah dinonaktifkan"})

    # Check limit before setting mode
    if mode_data.mode != "none":
        active_ai_count = 0
        for cid, other_cam in camera_manager.cameras.items():
            if cid != cam_id and other_cam.get_mode() != "none":
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
            elif mode_data.mode == 'attribute':
                load_yolo_heavy()
            elif mode_data.mode == 'drowsiness':
                load_mediapipe()
            elif mode_data.mode == 'pose':
                load_yolo_pose_model()
            print(f"[AI-SERVICE] Model pre-loaded successfully for mode: {mode_data.mode}", flush=True)
        except Exception as e:
            print(f"[AI-SERVICE] Error loading model for mode {mode_data.mode}: {e}", flush=True)
            
    cam.set_mode(mode_data.mode, mode_data.selected_classes)
    return {"success": True, "cam_id": cam_id, "mode": mode_data.mode}


@app.get("/active_modes")
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
def compat_get_metadata(cam_id: int = Query(0)):
    cam = camera_manager.get_camera(cam_id)
    mode = cam.get_mode() if cam else "none"
    sel_classes = cam.get_selected_classes() if cam else None
    
    default_meta = {
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
        "contraflow_status": "AMAN",
        "contraflow_alarm": False,
        "drowsiness_status": "Normal",
        "ear_value": 0.0,
        "alarm": False,
        "pose_data": [],
        "fps": 0.0
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


# ── MJPEG Video Streaming
# The generator MUST never raise — any exception is caught, logged, and the
# loop continues so the browser keeps receiving frames (no black screen).

def _make_error_frame(msg: str) -> bytes:
    """Create a small JPEG error frame to keep the stream alive."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (20, 20, 40)
    cv2.putText(img, msg[:80], (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return enc.tobytes() if ok else b''


def mjpeg_generator(cam_id: int):
    print(f"[AI-SERVICE] MJPEG client connected for camera_{cam_id}", flush=True)
    loading_step = 0
    loop_count = 0

    while True:
        # ── Outer guard: catches any unexpected error so the generator never dies ──
        try:
            cam = camera_manager.get_camera(cam_id)

            if not cam:
                # Camera not registered — send placeholder and wait
                raw = _make_error_frame("Camera not registered...")
                if raw:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                           + str(len(raw)).encode() + b'\r\n\r\n' + raw + b'\r\n')
                time.sleep(0.5)
                continue

            frame = cam.get_frame()
            if frame is None:
                # Camera registered but not yet streaming — show connecting animation
                img = np.zeros((360, 640, 3), dtype=np.uint8)
                img[:] = (15, 23, 42)
                loading_step = (loading_step + 1) % 4
                dots = "." * loading_step
                cv2.putText(img, f"Connecting to Camera {dots}", (140, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 200), 2, cv2.LINE_AA)
                ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    raw = enc.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                           + str(len(raw)).encode() + b'\r\n\r\n' + raw + b'\r\n')
                time.sleep(0.04)
                continue

            # Retrieve current mode — this is just a dict read, never modifies frame
            mode = cam.get_mode()
            selected_classes = cam.get_selected_classes()
            state = get_camera_state(cam_id)

            # ── Inner guard: analytics exception must NOT kill the stream ──
            try:
                processed, meta = run_analytics(cam_id, frame, mode, selected_classes, state)
                camera_metadata[cam_id] = meta
            except Exception as analytics_err:
                print(f"[AI-SERVICE] Analytics error cam_{cam_id} mode={mode}: {analytics_err}", flush=True)
                print(traceback.format_exc(), flush=True)
                processed = frame.copy()
                cv2.putText(processed, f"AI ERR [{mode}]: {str(analytics_err)[:55]}",
                            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 60, 255), 2, cv2.LINE_AA)

            # Resize aggressively before encoding to reduce JPEG encode time
            # Target: max 960px width for smooth browser display
            h, w = processed.shape[:2]
            if w > 960:
                scale = 960 / w
                new_w, new_h = int(w * scale), int(h * scale)
                processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            ok, enc = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                raw = enc.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                       + str(len(raw)).encode() + b'\r\n\r\n' + raw + b'\r\n')

            loop_count += 1
            if loop_count % 100 == 0:
                import gc
                gc.collect()
            time.sleep(0.066)  # ~15 fps cap; balance responsiveness vs CPU

        except Exception as outer_err:
            # Last-resort catch — log and keep the generator alive
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


if __name__ == '__main__':
    import uvicorn
    print(f"[AI-SERVICE] Engine starting up with FastAPI...", flush=True)
    uvicorn.run("ai_service:app", host="0.0.0.0", port=5001, reload=False, workers=1)
