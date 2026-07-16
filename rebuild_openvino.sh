#!/bin/bash
# rebuild_openvino.sh
# Jalankan SETELAH fix_raspi.sh berhasil dan YOLO PyTorch sudah jalan
# Script ini akan:
# 1. Backup model OpenVINO yang corrupt
# 2. Export ulang model OpenVINO dari .pt yang sudah ter-verifikasi
# 3. Test model OpenVINO baru
# 4. Aktifkan OpenVINO di analytics_engine.py
#
# Jalankan: bash /home/nabil/Camera/rebuild_openvino.sh

CAMERA_DIR="/home/nabil/Camera"
VENV_PYTHON="$CAMERA_DIR/venv/bin/python3"

echo "=============================================="
echo " Rebuild OpenVINO Model - Raspberry Pi 4"
echo " Waktu: $(date)"
echo "=============================================="

cd "$CAMERA_DIR"

# ── STEP 1: Backup model lama yang corrupt ──
echo ""
echo "[STEP 1] Backup model OpenVINO lama (yang corrupt)..."
BACKUP_DIR="$CAMERA_DIR/openvino_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

for dir in yolov8n_openvino_model yolov8s_openvino_model yolov8n-pose_openvino_model; do
    if [ -d "$CAMERA_DIR/$dir" ]; then
        echo "   Backup $dir → $BACKUP_DIR/"
        cp -r "$CAMERA_DIR/$dir" "$BACKUP_DIR/"
        rm -rf "$CAMERA_DIR/$dir"
        echo "   ✅ $dir di-backup dan dihapus"
    fi
done

# Hapus juga kernel.errors.txt yang lama
rm -f "$CAMERA_DIR/kernel.errors.txt"

echo "   Backup tersimpan di: $BACKUP_DIR"

# ── STEP 2: Verifikasi .pt model ada ──
echo ""
echo "[STEP 2] Verifikasi .pt model..."
for pt in yolov8n.pt yolov8s.pt yolov8n-pose.pt; do
    if [ -f "$CAMERA_DIR/$pt" ]; then
        SIZE=$(du -sh "$CAMERA_DIR/$pt" | cut -f1)
        echo "   ✅ $pt ($SIZE)"
    else
        echo "   ❌ $pt tidak ditemukan!"
    fi
done

# ── STEP 3: Export OpenVINO model baru ──
echo ""
echo "[STEP 3] Export ulang model ke OpenVINO format..."
echo "   (Ini akan memakan waktu 5-15 menit...)"

$VENV_PYTHON "$CAMERA_DIR/convert_openvino.py" 2>&1
EXPORT_STATUS=$?

if [ $EXPORT_STATUS -ne 0 ]; then
    echo "❌ Export gagal! Tetap pakai PyTorch .pt mode."
    exit 1
fi

# ── STEP 4: Verifikasi model baru ──
echo ""
echo "[STEP 4] Verifikasi model OpenVINO baru..."
for dir in yolov8n_openvino_model yolov8s_openvino_model; do
    if [ -d "$CAMERA_DIR/$dir" ]; then
        XML_FILE=$(find "$CAMERA_DIR/$dir" -name "*.xml" 2>/dev/null | head -1)
        BIN_FILE=$(find "$CAMERA_DIR/$dir" -name "*.bin" 2>/dev/null | head -1)
        if [ -n "$XML_FILE" ] && [ -n "$BIN_FILE" ]; then
            echo "   ✅ $dir: .xml dan .bin ada"
        else
            echo "   ⚠️  $dir: file tidak lengkap"
        fi
    else
        echo "   ❌ $dir tidak ada setelah export"
    fi
done

# ── STEP 5: Test OpenVINO model baru ──
echo ""
echo "[STEP 5] Test inference dengan model OpenVINO baru..."
$VENV_PYTHON -c "
import os, sys
os.environ['ULTRALYTICS_TELEMETRY'] = 'false'
os.environ['ULTRALYTICS_CHECK'] = 'false'
os.environ['OPENVINO_TELEMETRY_OPTOUT'] = '1'
import numpy as np
from ultralytics import YOLO

dummy = np.zeros((640, 640, 3), dtype=np.uint8)

print('[TEST] Loading yolov8n_openvino_model...')
try:
    model = YOLO('yolov8n_openvino_model')
    print('[TEST] Model loaded')
    results = model(dummy, imgsz=640, verbose=False)
    print('[TEST] ✅ OpenVINO inference OK! Detections:', len(results[0].boxes))
    ov_ok = True
except Exception as e:
    print(f'[TEST] ❌ OpenVINO error: {e}')
    ov_ok = False

sys.exit(0 if ov_ok else 1)
" 2>&1
OV_TEST_STATUS=$?

if [ $OV_TEST_STATUS -eq 0 ]; then
    echo ""
    echo "✅ OpenVINO model baru BERHASIL! Mengaktifkan OpenVINO di analytics_engine.py..."
    
    # Aktifkan OpenVINO dengan mengubah _use_openvino = False → True
    # (Atur env var FORCE_OPENVINO=1 di startup, atau edit langsung)
    sed -i 's/_use_openvino = False  # DISABLED: OpenVINO model corrupt.*/_use_openvino = True  # ENABLED: Model OpenVINO berhasil di-rebuild/' \
        "$CAMERA_DIR/analytics_engine.py"
    
    echo "   analytics_engine.py diupdate: OpenVINO AKTIF"
    
    # Restart ai_service
    echo ""
    echo "[STEP 6] Restart ai_service dengan OpenVINO..."
    pkill -f "ai_service.py" 2>/dev/null || true
    sleep 2
    
    LOG_FILE="$CAMERA_DIR/ai_service_openvino.log"
    PYTHONUNBUFFERED=1 nohup $VENV_PYTHON "$CAMERA_DIR/ai_service.py" > "$LOG_FILE" 2>&1 &
    echo "   ai_service.py restart dengan PID: $!"
    echo "   Log: $LOG_FILE"
    
    echo ""
    echo "=============================================="
    echo " ✅ OpenVINO AKTIF! Performa lebih baik."
    echo " Pantau log: tail -f $LOG_FILE"
    echo "=============================================="
else
    echo ""
    echo "❌ OpenVINO model baru masih bermasalah."
    echo "   Tetap gunakan PyTorch .pt (sudah berjalan)."
    echo "   analytics_engine.py TIDAK diubah."
    echo ""
    echo "   Restore backup jika perlu:"
    echo "   cp -r $BACKUP_DIR/* $CAMERA_DIR/"
fi
