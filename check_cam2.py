import socket
import cv2

ip = '192.168.1.116'
print(f"--- Checking {ip} ---")

for p in [80, 554, 5543, 8000, 8554, 8899, 37777, 5540]:
    s = socket.socket()
    s.settimeout(1.5)
    r = s.connect_ex((ip, p))
    s.close()
    if r == 0:
        print(f"Port {p}: OPEN!")
    else:
        print(f"Port {p}: closed ({r})")

urls = [
    'rtsp://admin:telkomiot123@192.168.1.116:554/cam/realmonitor?channel=1&subtype=1',
    'rtsp://admin:telkomiot123@192.168.1.116:554/cam/realmonitor?channel=1&subtype=0',
    'rtsp://admin:telkomiot123@192.168.1.116:554/h264Preview_01_main',
    'rtsp://admin:telkomiot123@192.168.1.116:554/live/ch0',
    'rtsp://admin:telkomiot123@192.168.1.116:554/onvif1',
    'rtsp://admin:telkomiot123@192.168.1.116:5543/live/channel1',
    'rtsp://admin:telkomiot123@192.168.1.116:8554/live',
]

for u in urls:
    cap = cv2.VideoCapture(u)
    opened = cap.isOpened()
    if opened:
        ret, frame = cap.read()
        print(f"SUCCESS! URL: {u} | read: {ret}")
        cap.release()
        break
    else:
        print(f"Failed: {u}")
