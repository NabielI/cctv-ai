#!/bin/bash
# fix_raspi.sh
# Script untuk memperbaiki YOLO AI detection di Raspberry Pi 4
# Jalankan di Raspberry Pi dengan: bash /home/nabil/Camera/fix_raspi.sh
# Dibuat oleh Antigravity AI

set -e  # exit on error

CAMERA_DIR="/home/nabil/Camera"
VENV_PYTHON="$CAMERA_DIR/venv/bin/python3"
VENV_PIP="$CAMERA_DIR/venv/bin/pip"

echo "=============================================="
echo " Fix YOLO Detection - Raspberry Pi 4"
echo " Waktu: $(date)"
echo "=============================================="

# ── STEP 1: Verifikasi venv ──
echo ""
echo "[STEP 1] Verifikasi virtual environment..."
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ ERROR: venv tidak ditemukan di $CAMERA_DIR/venv"
    echo "   Buat venv dulu dengan: python3 -m venv $CAMERA_DIR/venv"
    exit 1
fi
echo "✅ venv ditemukan: $VENV_PYTHON"
$VENV_PYTHON --version

# ── STEP 2: Kill ai_service yang lagi jalan ──
echo ""
echo "[STEP 2] Matikan ai_service yang sedang berjalan..."
pkill -f "ai_service.py" 2>/dev/null && echo "   Killed ai_service.py" || echo "   (tidak ada proses ai_service yang berjalan)"
sleep 1

# ── STEP 3: Test import ultralytics ──
echo ""
echo "[STEP 3] Test import ultralytics..."
$VENV_PYTHON -c "import ultralytics; print('ultralytics version:', ultralytics.__version__)" 2>&1 || {
    echo "⚠️  ultralytics tidak terinstall, menginstall..."
    $VENV_PIP install ultralytics
}

# ── STEP 4: Test import torch ──
echo ""
echo "[STEP 4] Test import torch..."
$VENV_PYTHON -c "
import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CPU threads:', torch.get_num_threads())
" 2>&1 || {
    echo "❌ torch tidak terinstall atau error"
    echo "   Akan menginstall torch untuk ARM64..."
    $VENV_PIP install torch --index-url https://download.pytorch.org/whl/cpu
}

# ── STEP 5: Test import torchvision ──
echo ""
echo "[STEP 5] Test import torchvision..."
$VENV_PYTHON -c "import torchvision; print('torchvision version:', torchvision.__version__)" 2>&1 || {
    echo "⚠️  torchvision tidak terinstall atau versi mismatch"
    echo "   Menginstall torchvision yang kompatibel..."
    TORCH_VER=$($VENV_PYTHON -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
    if [ -n "$TORCH_VER" ]; then
        echo "   Torch version: $TORCH_VER"
        $VENV_PIP install torchvision --index-url https://download.pytorch.org/whl/cpu
    else
        $VENV_PIP install torchvision
    fi
}

# ── STEP 6: Test load YOLO26 Model ──
echo ""
echo "[STEP 6] Test load YOLO26 Model (yolov8n.pt)..."
cd "$CAMERA_DIR"
$VENV_PYTHON -c "
import os, sys
os.environ['ULTRALYTICS_TELEMETRY'] = 'false'
os.environ['ULTRALYTICS_CHECK'] = 'false'
import numpy as np
from ultralytics import YOLO, settings
try:
    settings.update({'sync': False, 'check': False, 'telemetry': False})
except Exception:
    pass

print('[TEST] Loading YOLO26 Engine...')
model = YOLO('yolov8n.pt')
print('[TEST] YOLO26 Model loaded OK')
dummy = np.zeros((320, 320, 3), dtype=np.uint8)
print('[TEST] Running YOLO26 inference...')
results = model(dummy, imgsz=320, verbose=False)
print('[TEST] ✅ YOLO26 inference SUCCESS! Detections:', len(results[0].boxes))
" 2>&1
YOLO_STATUS=$?

if [ $YOLO_STATUS -ne 0 ]; then
    echo ""
    echo "❌ YOLO load gagal. Mencoba fix tambahan..."
    
    # Install ulang ultralytics versi stabil
    echo "   Reinstall ultralytics..."
    $VENV_PIP install --upgrade "ultralytics>=8.0.0"
    
    # Coba lagi
    echo "   Retry YOLO test..."
    $VENV_PYTHON -c "
from ultralytics import YOLO
import numpy as np
model = YOLO('yolov8n.pt')
dummy = np.zeros((320, 320, 3), dtype=np.uint8)
r = model(dummy, imgsz=320, verbose=False)
print('✅ YOLO OK setelah reinstall! Detections:', len(r[0].boxes))
" 2>&1 || {
        echo "❌ YOLO masih gagal setelah reinstall"
        echo "   Cek manual: $VENV_PYTHON -c \"from ultralytics import YOLO; print('OK')\""
    }
fi

# ── STEP 7: Test analytics_engine import ──
echo ""
echo "[STEP 7] Test import analytics_engine..."
cd "$CAMERA_DIR"
timeout 30 $VENV_PYTHON -c "
import sys, os
os.environ['ULTRALYTICS_TELEMETRY'] = 'false'
os.environ['ULTRALYTICS_CHECK'] = 'false'
# Test import saja, jangan load model dulu
print('Testing analytics_engine import...')
import importlib.util
spec = importlib.util.spec_from_file_location('analytics_engine', 'analytics_engine.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('HAS_YOLO akan dipakai:', mod.HAS_YOLO)
print('YOLO_MODEL_NAME:', mod.YOLO_MODEL_NAME)
print('✅ analytics_engine import OK')
" 2>&1 || echo "⚠️  analytics_engine import gagal (mungkin normal, cek log)"

# ── STEP 8: Restart ai_service ──
echo ""
echo "[STEP 8] Restart ai_service.py dengan venv..."

# Kill lagi untuk pastikan bersih
pkill -f "ai_service.py" 2>/dev/null || true
sleep 1

# Start ai_service di background dengan log
LOG_FILE="$CAMERA_DIR/ai_service_fix.log"
echo "   Log file: $LOG_FILE"
PYTHONUNBUFFERED=1 nohup $VENV_PYTHON "$CAMERA_DIR/ai_service.py" > "$LOG_FILE" 2>&1 &
AI_PID=$!
echo "   ai_service.py started dengan PID: $AI_PID"

# ── STEP 9: Tunggu dan cek apakah service berhasil start ──
echo ""
echo "[STEP 9] Menunggu ai_service start (15 detik)..."
sleep 15

# Cek apakah process masih jalan
if kill -0 $AI_PID 2>/dev/null; then
    echo "✅ ai_service.py masih berjalan (PID: $AI_PID)"
else
    echo "❌ ai_service.py berhenti! Cek log:"
    tail -30 "$LOG_FILE"
fi

# ── STEP 10: Cek log untuk HAS_YOLO ──
echo ""
echo "[STEP 10] Cek log ai_service..."
sleep 3
if [ -f "$LOG_FILE" ]; then
    echo "--- Tail log (30 baris terakhir) ---"
    tail -30 "$LOG_FILE"
    echo "------------------------------------"
    
    if grep -q "YOLO_MODEL_NAME" "$LOG_FILE"; then
        echo ""
        if grep -q "loaded & warmed up\|loaded — warm-up\|✅" "$LOG_FILE"; then
            echo "✅ YOLO BERHASIL DILOAD!"
        elif grep -q "HOG fallback\|❌ YOLO load GAGAL" "$LOG_FILE"; then
            echo "❌ YOLO masih gagal. Lihat error di log:"
            grep -A5 "Error detail\|HOG fallback\|Traceback" "$LOG_FILE" | head -30
        fi
    fi
fi

# ── STEP 11: Test HTTP endpoint ──
echo ""
echo "[STEP 11] Test endpoint ai_service (port 5001)..."
sleep 3
curl -s "http://127.0.0.1:5001/metadata?cam_id=0" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print('✅ ai_service HTTP OK!')
    print('   engine:', data.get('engine', 'N/A'))
    print('   camera_status:', data.get('camera_status', 'N/A'))
except:
    print('⚠️  Response bukan JSON atau service belum ready')
" 2>/dev/null || echo "⚠️  ai_service port 5001 belum ready (normal jika baru start)"

echo ""
echo "=============================================="
echo " SELESAI - Ringkasan:"
echo "=============================================="
echo " - analytics_engine.py: Force PyTorch .pt (OpenVINO bypass)"
echo " - server.js: Fix Linux commands (pkill, python3 venv)"
echo " - ai_service.py: Restart dengan venv Python"
echo ""
echo " Langkah selanjutnya:"
echo " 1. Buka dashboard di browser"
echo " 2. Pilih mode AI (misal: Deteksi Wajah)"
echo " 3. Verifikasi teks di video: 'YOLOv8s (AKTIF)' bukan 'OpenCV HOG Fallback'"
echo ""
echo " Log ai_service: tail -f $LOG_FILE"
echo "=============================================="
