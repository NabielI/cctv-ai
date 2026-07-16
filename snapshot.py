"""
snapshot.py - Grab a fresh JPEG frame from an RTSP stream via OpenCV.
Usage: python snapshot.py <rtsp_url>
Outputs JPEG bytes to stdout. Exit code 0 = success, 1 = error.

Strategy for HEVC (H.265) cameras:
- HEVC needs many reference frames before the decoder can produce a full image.
- The first ~20 frames often appear gray/noisy. We keep reading until we find
  a frame with sufficient visual variance (std > 30) indicating real content.
- Timeout after 25 seconds to avoid hanging forever.
"""
import sys
import cv2
import time

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: snapshot.py <rtsp_url>\n")
        sys.exit(1)

    rtsp_url = sys.argv[1]

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        sys.stderr.write(f"Failed to open stream: {rtsp_url}\n")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    best_frame = None
    best_std = 0
    start = time.time()
    frame_count = 0
    TIMEOUT_SEC = 25
    GOOD_FRAME_STD = 30  # std > 30 means frame has real image content (not gray/noisy)

    while (time.time() - start) < TIMEOUT_SEC:
        ok, frame = cap.read()
        if not ok:
            # Brief pause then retry if no frame yet
            time.sleep(0.05)
            continue

        frame_count += 1
        current_std = frame.std()

        # Keep track of best frame seen so far
        if current_std > best_std:
            best_std = current_std
            best_frame = frame.copy()

        # Stop as soon as we find a frame with good visual content
        if current_std >= GOOD_FRAME_STD:
            elapsed = time.time() - start
            sys.stderr.write(f"Good frame found at index {frame_count}, std={current_std:.1f}, elapsed={elapsed:.1f}s\n")
            break

    cap.release()

    if best_frame is None:
        sys.stderr.write("Failed to read any frame from stream\n")
        sys.exit(1)

    elapsed = time.time() - start
    sys.stderr.write(f"Using best frame: std={best_std:.1f}, frames_read={frame_count}, elapsed={elapsed:.1f}s\n")

    # Encode as JPEG (quality 90)
    ok, buf = cv2.imencode('.jpg', best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        sys.stderr.write("Failed to encode frame as JPEG\n")
        sys.exit(1)

    sys.stdout.buffer.write(buf.tobytes())
    sys.exit(0)

if __name__ == '__main__':
    main()
