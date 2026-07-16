import os
import json
import time
import subprocess
import atexit
import urllib.request
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# File path to the camera configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

# Process handle for go2rtc
go2rtc_process = None

def load_config():
    if not os.path.exists(CONFIG_PATH):
        # Default fallback config
        return {
            "cameras": [
                {
                    "id": 0,
                    "name": "Kamera Saya (192.168.2.19)",
                    "url": "rtsp://admin:admin123@192.168.2.19:5543/live/channel1"
                },
                {
                    "id": 1,
                    "name": "Kamera Temen (192.168.2.158)",
                    "url": "rtsp://admin:admin123@192.168.2.158:5543/live/channel1"
                }
            ]
        }
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading config: {e}")
        return {"cameras": []}

def save_config(config):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def write_go2rtc_yaml(config):
    go2rtc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'go2rtc')
    yaml_path = os.path.join(go2rtc_dir, 'go2rtc.yaml')
    print(f"Writing go2rtc config to {yaml_path}")
    try:
        os.makedirs(go2rtc_dir, exist_ok=True)
        with open(yaml_path, 'w') as f:
            f.write("streams:\n")
            for cam in config.get("cameras", []):
                cam_id = cam.get("id")
                url = cam.get("url")
                if url:
                    # Write stream name as camera_X
                    f.write(f"  camera_{cam_id}: {url}\n")
    except Exception as e:
        print(f"Error writing go2rtc.yaml: {e}")

def start_go2rtc():
    global go2rtc_process

    go2rtc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "go2rtc")

    import platform

    if platform.system() == "Windows":
        go2rtc_exe = os.path.join(go2rtc_dir, "go2rtc.exe")
    else:
        go2rtc_exe = os.path.join(go2rtc_dir, "go2rtc")


    # Generate the yaml config first
    config = load_config()
    write_go2rtc_yaml(config)
    
    print(f"Starting go2rtc: {go2rtc_exe}")
    try:
        go2rtc_process = subprocess.Popen(
            [go2rtc_exe],
            cwd=go2rtc_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("go2rtc process started.")
    except Exception as e:
        print(f"Error starting go2rtc subprocess: {e}")

def stop_go2rtc():
    global go2rtc_process
    if go2rtc_process and go2rtc_process.poll() is None:
        print("Stopping go2rtc process...")
        go2rtc_process.terminate()
        try:
            go2rtc_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            go2rtc_process.kill()
        print("go2rtc process stopped.")

# Register exit handler
atexit.register(stop_go2rtc)

# Start go2rtc on startup
start_go2rtc()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(load_config())

@app.route('/api/config', methods=['POST'])
def update_config():
    new_config = request.json
    if not new_config or 'cameras' not in new_config:
        return jsonify({"success": False, "message": "Invalid configuration format"}), 400
        
    if save_config(new_config):
        # Update go2rtc yaml
        write_go2rtc_yaml(new_config)
        # Try to hot-reload go2rtc config via API
        try:
            req = urllib.request.Request("http://localhost:1984/api/restart", method="POST")
            with urllib.request.urlopen(req, timeout=3) as resp:
                print("Called go2rtc restart API.")
        except Exception as e:
            print(f"Failed to call restart API, restarting subprocess manually: {e}")
            stop_go2rtc()
            time.sleep(0.5)
            start_go2rtc()
        return jsonify({"success": True, "message": "Configuration saved & streams restarted"})
    else:
        return jsonify({"success": False, "message": "Failed to save configuration"}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    config = load_config()
    status = {}
    
    # Query stream status from go2rtc
    go2rtc_streams = {}
    try:
        with urllib.request.urlopen("http://localhost:1984/api/streams", timeout=2) as resp:
            go2rtc_streams = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error querying go2rtc status: {e}")

    for cam in config.get("cameras", []):
        cam_id = cam.get("id")
        name = cam.get("name")
        url = cam.get("url")
        stream_key = f"camera_{cam_id}"
        
        # Check if go2rtc has active producers for this stream
        connected = False
        if stream_key in go2rtc_streams:
            stream_info = go2rtc_streams[stream_key]
            producers = stream_info.get("producers", [])
            if producers:
                connected = True
                
        status[cam_id] = {
            "name": name,
            "url": url,
            "connected": connected,
            "running": True
        }
    return jsonify(status)

if __name__ == '__main__':
    # Make accessible across the local network by binding to 0.0.0.0
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
