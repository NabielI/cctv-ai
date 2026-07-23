# convert_openvino.py
import os
import sys

os.environ["ULTRALYTICS_TELEMETRY"] = "false"
os.environ["ULTRALYTICS_CHECK"] = "false"

try:
    from ultralytics import YOLO, settings
    try:
        settings.update({'sync': False, 'check': False, 'telemetry': False})
    except Exception:
        pass
except ImportError:
    print("Error: ultralytics is not installed. Please install it first.", file=sys.stderr)
    sys.exit(1)

models_to_export = [
    ("yolo26n.pt", "yolo26n_openvino_model"),
    ("yolo26s.pt", "yolo26s_openvino_model"),
    ("yolo26n-pose.pt", "yolo26n-pose_openvino_model")
]

print("=== Starting YOLO26 to OpenVINO IR Export ===")

for pt_model, expected_dir in models_to_export:
    print(f"\n---> Exporting Model: {pt_model} -> {expected_dir}")
    if not os.path.exists(pt_model) or os.path.getsize(pt_model) == 0:
        print(f"Error: {pt_model} is missing or 0 bytes!", file=sys.stderr)
        continue
    
    try:
        print(f"Loading {pt_model}...")
        model = YOLO(pt_model)
        
        print(f"Exporting {pt_model} to OpenVINO IR format (half=True, dynamic=True)...")
        model.export(format="openvino", half=True, dynamic=True)
        
        if os.path.exists(expected_dir):
            print(f"✅ SUCCESS! Created OpenVINO directory: {expected_dir}")
        else:
            print(f"⚠️ Directory {expected_dir} not directly created, searching matching OpenVINO dirs...")
            cwd_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and 'openvino' in d]
            print(f"Found OpenVINO directories: {cwd_dirs}")
            
    except Exception as e:
        print(f"❌ Error exporting {pt_model}: {e}", file=sys.stderr)

print("\n=== Export Process Completed ===")
