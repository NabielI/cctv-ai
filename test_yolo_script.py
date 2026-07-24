import cv2
import time
from ultralytics import YOLO

model = YOLO('/home/nabil/Camera/yolov8n.pt')
cap = cv2.VideoCapture('rtsp://127.0.0.1:8554/camera_4')

print("Starting live YOLO test on camera_4...")
for i in range(10):
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"Frame {i+1}: read failed")
        continue
    results = model(frame, imgsz=416, verbose=False, conf=0.15, classes=[0])
    boxes = results[0].boxes
    confs = [round(float(b.conf[0]), 2) for b in boxes]
    print(f"Frame {i+1}: detected {len(boxes)} person(s), confs={confs}")
    time.sleep(0.5)

cap.release()
print("Done.")
