import requests, cv2, numpy as np, time

r1 = requests.get('http://127.0.0.1:5001/api/snapshot/4')
img1 = cv2.imdecode(np.frombuffer(r1.content, np.uint8), cv2.IMREAD_COLOR)
time.sleep(1.0)
r2 = requests.get('http://127.0.0.1:5001/api/snapshot/4')
img2 = cv2.imdecode(np.frombuffer(r2.content, np.uint8), cv2.IMREAD_COLOR)

c1 = cv2.cvtColor(img1[129:194, 339:395], cv2.COLOR_BGR2GRAY)
c2 = cv2.cvtColor(img2[129:194, 339:395], cv2.COLOR_BGR2GRAY)
diff = float(np.mean(cv2.absdiff(c1, c2)))
print(f"STATIC CHAIR MOTION SCORE: {diff:.4f}")
