# benchmark_openvino.py
import time
import os
import numpy as np
import torch

# Opt-out of OpenVINO telemetry to prevent network-related hangs
os.environ["OPENVINO_TELEMETRY_OPTOUT"] = "1"
os.environ["OV_TELEMETRY_OPTOUT"] = "1"

# Disable telemetry and check updates to prevent hangs
os.environ["ULTRALYTICS_TELEMETRY"] = "false"
os.environ["ULTRALYTICS_CHECK"] = "false"

from ultralytics import YOLO, settings
try:
    settings.update({'sync': False, 'check': False, 'telemetry': False})
except Exception:
    try:
        settings.update({'sync': False, 'telemetry': False})
    except Exception:
        pass

# Dummy image for inference
dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

print("==================================================")
# Print CPU information
print("Running benchmark on CPU...")
print("==================================================")

# 1. Benchmark PyTorch CPU model
print("\n[1/2] Benchmarking PyTorch CPU Model (yolov8n.pt)...")
torch_model = YOLO("yolov8n.pt")
torch_model.to("cpu")

# Warm-up runs
print("Running warm-up...")
for _ in range(5):
    _ = torch_model(dummy_frame, verbose=False, device="cpu")

# Benchmark runs
print("Running benchmark...")
start_time = time.time()
num_iterations_pytorch = 20  # Keep it lower for slow PyTorch CPU to save time
for _ in range(num_iterations_pytorch):
    _ = torch_model(dummy_frame, verbose=False, device="cpu")
torch_elapsed = time.time() - start_time
torch_fps = num_iterations_pytorch / torch_elapsed
torch_avg_time = (torch_elapsed / num_iterations_pytorch) * 1000

print(f"PyTorch CPU Results:")
print(f"  - Average Inference Time: {torch_avg_time:.2f} ms")
print(f"  - Average FPS: {torch_fps:.2f}")


# 2. Benchmark OpenVINO FP16 model
print("\n[2/2] Benchmarking OpenVINO Model (yolov8n_openvino_model)...")
ov_model_dir = "yolov8n_openvino_model"
if not os.path.exists(ov_model_dir):
    print(f"Error: {ov_model_dir} directory not found! Run convert_openvino.py first.")
    ov_fps = 0.0
    ov_avg_time = 0.0
else:
    ov_model = YOLO(ov_model_dir)
    
    # Warm-up runs
    print("Running warm-up...")
    for _ in range(15):
        _ = ov_model(dummy_frame, verbose=False)

    # Benchmark runs
    print("Running benchmark...")
    start_time = time.time()
    num_iterations_ov = 50  # OpenVINO is fast, we can run more iterations
    for _ in range(num_iterations_ov):
        _ = ov_model(dummy_frame, verbose=False)
    ov_elapsed = time.time() - start_time
    ov_fps = num_iterations_ov / ov_elapsed
    ov_avg_time = (ov_elapsed / num_iterations_ov) * 1000

    print(f"OpenVINO CPU Results:")
    print(f"  - Average Inference Time: {ov_avg_time:.2f} ms")
    print(f"  - Average FPS: {ov_fps:.2f}")

print("\n==================================================")
print("                    SUMMARY                       ")
print("==================================================")
print(f"{'Framework':<15} | {'Avg Latency (ms)':<18} | {'Inference FPS':<12}")
print("-" * 52)
print(f"{'PyTorch (CPU)':<15} | {torch_avg_time:>15.2f} ms | {torch_fps:>9.2f}")
if os.path.exists(ov_model_dir):
    print(f"{'OpenVINO (FP16)':<15} | {ov_avg_time:>15.2f} ms | {ov_fps:>9.2f}")
    speedup = torch_avg_time / ov_avg_time
    print("-" * 52)
    print(f"Speedup: {speedup:.2f}x faster inference using OpenVINO!")
else:
    print(f"{'OpenVINO (FP16)':<15} | {'N/A':>18} | {'N/A':>12}")
print("==================================================")
