# convert_openvino.py
import os
import sys

# Disable telemetry and check updates during export to prevent hangs
os.environ["ULTRALYTICS_TELEMETRY"] = "false"
os.environ["ULTRALYTICS_CHECK"] = "false"

try:
    from ultralytics import YOLO, settings
    try:
        settings.update({'sync': False, 'check': False, 'telemetry': False})
    except Exception as se:
        try:
            settings.update({'sync': False, 'telemetry': False})
        except Exception:
            pass
except ImportError:
    print("Error: ultralytics is not installed. Please install it first.", file=sys.stderr)
    sys.exit(1)

models_to_export = [
    ("yolov8n.pt", "yolov8n_openvino_model"),
    ("yolov8s.pt", "yolov8s_openvino_model"),
    ("yolov8n-pose.pt", "yolov8n-pose_openvino_model")
]

print("=== Starting YOLO to OpenVINO IR Export ===")

for pt_model, expected_dir in models_to_export:
    print(f"\n---> Processing: {pt_model}")
    if not os.path.exists(pt_model):
        print(f"Error: {pt_model} weight file not found in the current directory!", file=sys.stderr)
        continue
    
    try:
        print(f"Loading {pt_model}...")
        model = YOLO(pt_model)
        
        print(f"Exporting {pt_model} to OpenVINO IR format with half=True...")
        # export format openvino creates a folder containing .xml and .bin files
        model.export(format="openvino", half=True)
        
        if os.path.exists(expected_dir):
            print(f"Success! Exported model directory created at: {expected_dir}")
        else:
            # Sometimes name might slightly differ (e.g. replacing hyphen or dot)
            # check directories in CWD that match
            cwd_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and 'openvino' in d]
            print(f"Export completed. Found OpenVINO folders: {cwd_dirs}")
            
    except Exception as e:
        print(f"Error exporting {pt_model}: {e}", file=sys.stderr)

print("\n=== Export Process Completed ===")
