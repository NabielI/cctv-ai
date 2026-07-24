import sqlite3, json, requests, cv2, numpy as np
from ultralytics import YOLO
from zone_monitor import is_person_in_zone, ZoneConfig

conn = sqlite3.connect('/home/nabil/Camera/zone_config.db')
cur = conn.cursor()
cur.execute('SELECT zone_id, cam_id, name, coords_json, threshold_minutes, cycle_hours, telegram_enabled, start_hour, grace_period_seconds FROM zones WHERE cam_id=4 AND name="nabil"')
row = cur.fetchone()
if row:
    zone = ZoneConfig(
        zone_id=row[0], cam_id=row[1], name=row[2],
        coords=json.loads(row[3]), threshold_minutes=row[4],
        cycle_hours=row[5], telegram_enabled=bool(row[6]),
        start_hour=row[7], grace_period_seconds=row[8]
    )

    resp = requests.get('http://127.0.0.1:5001/api/snapshot/4')
    img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    model = YOLO('/home/nabil/Camera/yolov8n.pt')
    res = model(img, conf=0.15, classes=[0])

    print(f"=== CHECKING ZONE '{zone.name}' (grace={zone.grace_period_seconds}s) ===")
    print("Zone Polygon Points (normalized):", zone.coords)

    for r in res:
        for b in r.boxes:
            conf = float(b.conf[0])
            bbox = list(map(int, b.xyxy[0].tolist()))
            w_b = max(1, bbox[2] - bbox[0])
            h_b = max(1, bbox[3] - bbox[1])
            ratio = h_b / w_b
            in_zone = is_person_in_zone(bbox, zone.coords, w, h)
            valid_person = in_zone and (ratio >= 0.85)
            print(f"Detection: conf={conf:.2f}, bbox={bbox}, ratio={ratio:.2f}, IN_ZONE={in_zone}, VALID_PERSON={valid_person}")
else:
    print("Zone 'nabil' not found on cam 4!")
