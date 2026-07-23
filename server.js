const express = require('express');
const path = require('path');
const fs = require('fs');
const { spawn, execSync, exec } = require('child_process');
const http = require('http');
const crypto = require('crypto');
const multer = require('multer');

const app = express();
const PORT = 5000;
const CONFIG_PATH = path.join(__dirname, 'config.json');
const UPLOADS_DIR = path.join(__dirname, 'uploads');

// Ensure uploads folder exists
if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// Multer disk storage setup
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, UPLOADS_DIR);
    },
    filename: (req, file, cb) => {
        const uniqueId = crypto.randomUUID();
        const ext = path.extname(file.originalname);
        cb(null, `${uniqueId}${ext}`);
    }
});
const upload = multer({
    storage,
    limits: { fileSize: 50 * 1024 * 1024 } // 50MB limit
});

// Process handle for go2rtc
let go2rtcProcess = null;

// In-memory log buffer for the logs sidebar panel
const systemLogs = [];
function log(msg, source = 'system') {
    const timestamp = new Date().toLocaleTimeString('id-ID');
    const entry = { timestamp, message: msg, source };
    systemLogs.push(entry);
    if (systemLogs.length > 250) systemLogs.shift();
    console.log(`[${source.toUpperCase()}] [${timestamp}] ${msg}`);
}

// Read config.json helper
function loadConfig() {
    if (!fs.existsSync(CONFIG_PATH)) {
        // Fallback default config with ONVIF and Relay support
        return {
            "cameras": [
                {
                    "id": 0,
                    "name": "Kamera Saya (192.168.2.19)",
                    "url": "rtsp://admin:admin123@192.168.2.19:5543/live/channel0",
                    "relayUrl": "",
                    "onvifPort": 80,
                    "onvifUsername": "admin",
                    "onvifPassword": "admin123"
                },
                {
                    "id": 1,
                    "name": "Kamera Temen (192.168.2.158)",
                    "url": "rtsp://admin:admin123@192.168.2.158:5543/live/channel0",
                    "relayUrl": "rtsp://192.168.2.158:8554/camera_0",
                    "onvifPort": 80,
                    "onvifUsername": "admin",
                    "onvifPassword": "admin123"
                }
            ]
        };
    }
    try {
        const raw = fs.readFileSync(CONFIG_PATH, 'utf8');
        return JSON.parse(raw);
    } catch (e) {
        log(`Error reading config.json: ${e.message}`, 'error');
        return { "cameras": [] };
    }
}

// Write config.json helper
function saveConfig(config) {
    try {
        fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), 'utf8');
        log('Config.json successfully written.');
        return true;
    } catch (e) {
        log(`Error writing config.json: ${e.message}`, 'error');
        return false;
    }
}

// Write go2rtc.yaml helper (with low-latency settings and multiple source types)
function writeGo2rtcYaml(config) {
    const go2rtcDir = path.join(__dirname, 'go2rtc');
    const yamlPath = path.join(go2rtcDir, 'go2rtc.yaml');
    log(`Writing go2rtc config to ${yamlPath}`);
    try {
        fs.mkdirSync(go2rtcDir, { recursive: true });

        let yamlContent = "streams:\n";
        (config.cameras || []).forEach(cam => {
            const sourceType = cam.sourceType || 'rtsp';
            
            if (sourceType === 'rtsp') {
                if (cam.url) {
                    const baseUrl = cam.url.split('#')[0];
                    const directEntry = `${baseUrl}#backchannel=0`;

                    if (cam.relayUrl) {
                        yamlContent += `  camera_${cam.id}:\n`;
                        yamlContent += `    - ${cam.relayUrl}\n`;
                        yamlContent += `    - ${directEntry}\n`;
                    } else {
                        yamlContent += `  camera_${cam.id}: ${directEntry}\n`;
                    }
                }
            } else if (sourceType === 'hls' || sourceType === 'youtube') {
                if (cam.url) {
                    const fastStart = cam.fastStart === 1;
                    let ffmpegArgs = '';
                    if (fastStart) {
                        ffmpegArgs += `-c:v libx264 -g 30 -preset ultrafast -tune zerolatency`;
                    } else {
                        ffmpegArgs += `-c:v copy`;
                    }
                    ffmpegArgs += ` -an`; // drop audio by default for security/stability
                    
                    // Mentor's robust reconnect params for go2rtc exec:ffmpeg HLS
                    yamlContent += `  camera_${cam.id}: exec:ffmpeg -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 30 -m3u8_hold_counters 60 -i "${cam.url}" ${ffmpegArgs} -f rtsp {output}\n`;
                }
            } else if (sourceType === 'file') {
                if (cam.filePath) {
                    const fullPath = path.join(__dirname, cam.filePath);
                    yamlContent += `  camera_${cam.id}: exec:ffmpeg -re -stream_loop -1 -i "${fullPath}" -c:v copy -an -f rtsp {output}\n`;
                }
            }
            
            // AI stream redirecting to Python Flask MJPEG stream
            yamlContent += `  camera_${cam.id}_ai: http://127.0.0.1:5001/stream?cam_id=${cam.id}\n`;
        });

        // Low-latency WebRTC + RTSP + API config
        yamlContent += `
rtsp:
  listen: ":8554"

webrtc:
  listen: ":8555/tcp"
  ice_servers:
    - urls: ["stun:stun.l.google.com:19302"]

api:
  listen: ":1984"
  origin: "*"

log:
  level: info
`;
        fs.writeFileSync(yamlPath, yamlContent, 'utf8');
    } catch (e) {
        log(`Error writing go2rtc.yaml: ${e.message}`, 'error');
    }
}

// Register all cameras with the AI service
function registerAllCamerasInAI() {
    const config = loadConfig();
    log(`Registering ${config.cameras.length} camera(s) in Python AI Analytics Service...`);
    for (const cam of config.cameras || []) {
        const postData = JSON.stringify({
            cam_id: cam.id,
            source_url: `rtsp://127.0.0.1:8554/camera_${cam.id}`
        });
        
        const req = http.request({
            hostname: '127.0.0.1',
            port: 5001,
            path: '/register',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData)
            }
        }, (res) => {});
        req.on('error', (e) => {
            // Quietly ignore if AI service is not running yet
        });
        req.write(postData);
        req.end();
    }
}

// Warm-up: force go2rtc to connect to all cameras immediately (no waiting for browser)
function warmupStreams() {
    registerAllCamerasInAI();
    const config = loadConfig();
    const cameras = config.cameras || [];
    if (cameras.length === 0) return;

    log(`Pre-connecting ${cameras.length} camera(s) to go2rtc...`, 'warmup');
    cameras.forEach(cam => {
        const streamKey = `camera_${cam.id}`;
        const options = {
            hostname: '127.0.0.1',
            port: 1984,
            path: `/api/stream?src=${streamKey}`,
            method: 'GET',
            timeout: 5000
        };
        const req = http.request(options, (res) => {
            let raw = '';
            res.on('data', d => { raw += d; });
            res.on('end', () => {
                try {
                    const info = JSON.parse(raw);
                    const state = (info.producers && info.producers[0]) ? info.producers[0].state : 'offline';
                    log(`${streamKey} state check: ${state}`, 'warmup');
                } catch(e) {}
            });
        });
        req.on('error', () => {}); // silently ignore if go2rtc not yet ready
        req.on('timeout', () => { req.destroy(); });
        req.end();
    });
}

// Restart & Watchdog Sync Flags
let keepaliveTimer = null;
let isRestarting = false;
let intentionalStop = false;

// Keepalive watchdog: every 30s re-check all streams and reconnect if disconnected
function startKeepalive() {
    if (keepaliveTimer) clearInterval(keepaliveTimer);
    keepaliveTimer = setInterval(() => {
        if (isRestarting) return; // Skip watchdog while restarting go2rtc config
        const config = loadConfig();
        const cameras = config.cameras || [];
        cameras.forEach(cam => {
            const streamKey = `camera_${cam.id}`;
            const checkOpts = {
                hostname: '127.0.0.1',
                port: 1984,
                path: `/api/stream?src=${streamKey}`,
                method: 'GET',
                timeout: 4000
            };
            const req = http.request(checkOpts, (res) => {
                let raw = '';
                res.on('data', d => { raw += d; });
                res.on('end', () => {
                    try {
                        const info = JSON.parse(raw);
                        const producers = info.producers || [];
                        const connected = producers.some(p => p.state === 'connected');
                        if (!connected) {
                            log(`[Keepalive] ${streamKey} offline — forcing reconnect...`, 'watchdog');
                            // Poke the stream endpoint again to force reconnect
                            const reconnectReq = http.request(checkOpts, () => {});
                            reconnectReq.on('error', () => {});
                            reconnectReq.end();
                        }
                    } catch(e) {}
                });
            });
            req.on('error', () => {
                if (isRestarting) return;
                // go2rtc might have crashed — restart it
                log('[Keepalive] Cannot reach go2rtc API — attempting restart...', 'watchdog');
                startGo2rtc();
                setTimeout(warmupStreams, 3000);
            });
            req.on('timeout', () => { req.destroy(); });
            req.end();
        });
    }, 30000); // check every 30 seconds
}

let aiServiceProcess = null;

// Detect OS to use correct commands (Windows vs Linux/Raspberry Pi)
const isWindows = process.platform === 'win32';
// Path ke Python venv di Raspberry Pi
const VENV_PYTHON = isWindows
    ? 'python'
    : path.join(__dirname, 'venv', 'bin', 'python3');

log(`[PLATFORM] OS: ${process.platform} | Python binary: ${VENV_PYTHON}`);

// Start Python AI Analytics service subprocess helper
function startAIService() {
    // Kill any existing Python AI Service instances first
    try {
        if (isWindows) {
            execSync('taskkill /F /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq ai_service*"', { stdio: 'ignore' });
        } else {
            execSync('pkill -f "ai_service.py" || true', { stdio: 'ignore' });
        }
    } catch(e) {}
    
    log('Starting Python AI Analytics Service (ai_service.py)...');
    try {
        const pythonBin = fs.existsSync(VENV_PYTHON) ? VENV_PYTHON : (isWindows ? 'python' : 'python3');
        log(`[AI-SERVICE] Using Python: ${pythonBin}`);
        aiServiceProcess = spawn(pythonBin, [path.join(__dirname, 'ai_service.py')], {
            cwd: __dirname,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: { ...process.env, PYTHONUNBUFFERED: '1' }
        });

        aiServiceProcess.stdout.on('data', (data) => {
            const lines = data.toString().split('\n');
            lines.forEach(line => {
                const trimmed = line.trim();
                if (trimmed) log(trimmed, 'system');
            });
        });

        aiServiceProcess.stderr.on('data', (data) => {
            const lines = data.toString().split('\n');
            lines.forEach(line => {
                const trimmed = line.trim();
                if (trimmed) log(trimmed, 'system');
            });
        });

        aiServiceProcess.on('error', (err) => {
            log(`Failed to start Python AI Service: ${err.message}`, 'error');
        });

        aiServiceProcess.on('close', (code) => {
            log(`Python AI Service process exited with code ${code}.`, 'warn');
        });
    } catch(e) {
        log(`Error spawning Python AI Service: ${e.message}`, 'error');
    }
}

// Stop Python AI Service subprocess helper
function stopAIService() {
    if (aiServiceProcess && !aiServiceProcess.killed) {
        console.log('Stopping Python AI Service process...');
        aiServiceProcess.kill('SIGKILL');
        aiServiceProcess = null;
    }
}

// Start go2rtc subprocess helper
function startGo2rtc() {
    // Kill any existing orphan go2rtc processes to free port 1984 and load fresh config
    try {
        if (isWindows) {
            execSync('taskkill /F /IM go2rtc.exe', { stdio: 'ignore' });
        } else {
            execSync('pkill -x go2rtc || true', { stdio: 'ignore' });
        }
        log('Terminated existing go2rtc processes.');
    } catch (e) {
        // Ignore if no process is running
    }

    // Wait 500ms to let the OS kernel fully release the TCP socket ports
    setTimeout(() => {
        const go2rtcDir = path.join(__dirname, 'go2rtc');
        // Linux Raspberry Pi pakai binary 'go2rtc' (tanpa .exe), Windows pakai 'go2rtc.exe'
        const go2rtcBin = isWindows ? 'go2rtc.exe' : 'go2rtc';
        const go2rtcExe = path.join(go2rtcDir, go2rtcBin);
        
        // Create config first
        const config = loadConfig();
        writeGo2rtcYaml(config);
        
        log(`Starting go2rtc: ${go2rtcExe}`);
        try {
            go2rtcProcess = spawn(go2rtcExe, [], {
                cwd: go2rtcDir,
                stdio: ['ignore', 'pipe', 'pipe']
            });

            // Capture go2rtc output and send it to our in-memory logs
            go2rtcProcess.stdout.on('data', (data) => {
                const lines = data.toString().split('\n');
                lines.forEach(line => {
                    const trimmed = line.trim();
                    if (trimmed) log(trimmed, 'go2rtc');
                });
            });

            go2rtcProcess.stderr.on('data', (data) => {
                const lines = data.toString().split('\n');
                lines.forEach(line => {
                    const trimmed = line.trim();
                    if (trimmed) log(trimmed, 'go2rtc-err');
                });
            });

            go2rtcProcess.on('error', (err) => {
                log(`Failed to start go2rtc: ${err.message}`, 'error');
            });

            // Auto-restart go2rtc if it crashes unexpectedly
            go2rtcProcess.on('close', (code) => {
                if (intentionalStop) {
                    log('go2rtc process stopped intentionally. Skipping auto-restart.', 'system');
                    intentionalStop = false;
                    return;
                }
                log(`go2rtc process exited with code ${code}. Restarting in 3s...`, 'warn');
                setTimeout(() => {
                    startGo2rtc();
                    setTimeout(warmupStreams, 3000);
                }, 3000);
            });

            log('go2rtc process started.');

            // Warm-up: pre-connect all cameras 2 seconds after go2rtc starts
            setTimeout(warmupStreams, 2000);
        } catch (e) {
            log(`Error spawning go2rtc subprocess: ${e.message}`, 'error');
        }
    }, 500);
}

// Stop go2rtc subprocess helper
function stopGo2rtc() {
    if (go2rtcProcess && !go2rtcProcess.killed) {
        console.log('Stopping go2rtc process...');
        intentionalStop = true;
        go2rtcProcess.kill('SIGKILL');
        go2rtcProcess = null;
    }
}

// Register cleanup listeners
process.on('SIGINT', () => {
    if (keepaliveTimer) clearInterval(keepaliveTimer);
    stopGo2rtc();
    stopAIService();
    process.exit();
});
process.on('SIGTERM', () => {
    if (keepaliveTimer) clearInterval(keepaliveTimer);
    stopGo2rtc();
    stopAIService();
    process.exit();
});
process.on('exit', () => {
    stopGo2rtc();
    stopAIService();
});

// Start AI and media server on boot
startAIService();
startGo2rtc();

// Start keepalive watchdog after 5 seconds (give go2rtc time to init)
setTimeout(startKeepalive, 5000);

// Express Middleware & Routes
app.use(express.json());
app.use(express.static(path.join(__dirname, 'templates')));
app.use('/uploads', express.static(UPLOADS_DIR));

// Proxy /api/faces/* endpoints to Python FastAPI service (port 5001)
app.use('/api/faces', (req, res) => {
    const targetUrl = `http://127.0.0.1:5001/api/faces${req.url}`;
    const proxyReq = http.request(targetUrl, {
        method: req.method,
        headers: {
            ...req.headers,
            host: '127.0.0.1:5001'
        }
    }, (proxyRes) => {
        res.writeHead(proxyRes.statusCode, proxyRes.headers);
        proxyRes.pipe(res);
    });
    proxyReq.on('error', (err) => {
        res.status(500).json({ success: false, message: 'Gagal terhubung ke AI Service' });
    });
    req.pipe(proxyReq);
});

// Proxy /api/zones/* endpoints to Python FastAPI service (Zone Monitoring)
app.use('/api/zones', (req, res) => {
    const targetUrl = `http://127.0.0.1:5001/api/zones${req.url}`;
    const payload = (req.method !== 'GET' && req.method !== 'HEAD' && req.body) ? JSON.stringify(req.body) : '';
    const headers = {
        'host': '127.0.0.1:5001'
    };
    if (payload) {
        headers['content-type'] = 'application/json';
        headers['content-length'] = Buffer.byteLength(payload);
    }
    const proxyReq = http.request(targetUrl, {
        method: req.method,
        headers: headers
    }, (proxyRes) => {
        res.writeHead(proxyRes.statusCode, proxyRes.headers);
        proxyRes.pipe(res);
    });
    proxyReq.on('error', (err) => {
        log(`Zone proxy error: ${err.message}`, 'error');
        if (!res.headersSent) {
            res.status(500).json({ success: false, message: 'Gagal terhubung ke AI Service (Zone Monitor)' });
        }
    });
    if (payload) proxyReq.write(payload);
    proxyReq.end();
});

// Resolve YouTube Live stream URL to direct HLS stream
function resolveYoutubeUrl(youtubeUrl) {
    return new Promise((resolve) => {
        log(`Resolving YouTube Live URL: ${youtubeUrl}...`);
        exec(`yt-dlp -g --format best "${youtubeUrl}"`, (error, stdout, stderr) => {
            if (error) {
                log(`Failed to resolve YouTube Live URL via yt-dlp: ${stderr.trim() || error.message}`, 'error');
                // Resolve with original URL as fallback
                resolve(youtubeUrl);
            } else {
                const resolved = stdout.trim().split('\n')[0]; // take first URL if multiple
                log(`Successfully resolved YouTube Live stream.`);
                resolve(resolved);
            }
        });
    });
}

// File Upload endpoint
app.post('/api/upload', upload.single('file'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ success: false, message: 'Tidak ada berkas yang diunggah.' });
    }
    res.json({
        success: true,
        filePath: `uploads/${req.file.filename}`,
        fileName: req.file.originalname,
        size: req.file.size
    });
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'index.html'));
});

app.get('/api/config', (req, res) => {
    res.json(loadConfig());
});

app.post('/api/config', async (req, res) => {
    const newConfig = req.body;
    if (!newConfig || !newConfig.cameras) {
        return res.status(400).json({ success: false, message: 'Format konfigurasi tidak valid' });
    }

    // Validate: each camera must have a URL OR a filePath (if it is a local file)
    const validCameras = (newConfig.cameras || []).filter(c => (c.url && c.url.trim()) || (c.sourceType === 'file' && c.filePath));
    if (validCameras.length === 0) {
        return res.status(400).json({ success: false, message: 'Minimal 1 kamera dengan URL / berkas valid harus diisi' });
    }

    try {
        // Resolve YouTube streams asynchronously before saving
        const resolvePromises = validCameras.map(async (c, i) => {
            const sourceType = c.sourceType || 'rtsp';
            let resolvedUrl = c.url || '';
            let originalUrl = c.originalUrl || '';

            if (sourceType === 'youtube' && c.url) {
                // If it looks like a YouTube link, resolve it; otherwise keep as is
                if (c.url.includes('youtube.com') || c.url.includes('youtu.be')) {
                    originalUrl = c.url;
                    const resolved = await resolveYoutubeUrl(c.url);
                    if (resolved) {
                        resolvedUrl = resolved;
                    }
                }
            }

            return {
                id: i,
                name: (c.name || `Kamera ${i + 1}`).trim(),
                url: resolvedUrl.trim(),
                sourceType,
                originalUrl: originalUrl.trim(),
                filePath: (c.filePath || '').trim(),
                relayUrl: (c.relayUrl || '').trim(),
                onvifPort: c.onvifPort ? (parseInt(c.onvifPort) || null) : null,
                onvifUsername: (c.onvifUsername || '').trim(),
                onvifPassword: (c.onvifPassword || '').trim(),
                fastStart: c.fastStart ? 1 : 0,
                lowLatency: c.lowLatency ? 1 : 0
            };
        });

        const normalizedCameras = await Promise.all(resolvePromises);
        const normalizedConfig = { cameras: normalizedCameras };

        if (!saveConfig(normalizedConfig)) {
            return res.status(500).json({ success: false, message: 'Gagal menyimpan konfigurasi ke disk' });
        }

        // Write updated go2rtc.yaml with new stream entries
        writeGo2rtcYaml(normalizedConfig);

    // Full go2rtc respawn — /api/restart does NOT reload stream definitions from YAML.
    // Only a kill + restart picks up new camera_N entries.
    log(`Saving ${normalizedConfig.cameras.length} camera(s) — restarting go2rtc...`, 'config');
    isRestarting = true;
    stopGo2rtc();
    setTimeout(() => {
        startGo2rtc();
        // Pre-connect all cameras (including new ones) once go2rtc is ready
        setTimeout(() => {
            warmupStreams();
            isRestarting = false;
            log('go2rtc restarted — all streams warming up.', 'config');
        }, 2500);
        }, 800);

        // Respond immediately so UI is not blocked waiting for go2rtc startup
        res.json({
            success: true,
            message: `${normalizedConfig.cameras.length} kamera disimpan. Stream sedang diinisialisasi...`
        });
    } catch (err) {
        log(`Error saving config: ${err.message}`, 'error');
        res.status(500).json({ success: false, message: `Gagal menyimpan konfigurasi: ${err.message}` });
    }
});

app.get('/api/status', (req, res) => {
    const config = loadConfig();
    
    const options = {
        hostname: '127.0.0.1',
        port: 1984,
        path: '/api/streams',
        method: 'GET',
        timeout: 2000
    };
    
    const reqStatus = http.request(options, (resp) => {
        let rawData = '';
        resp.on('data', (chunk) => { rawData += chunk; });
        resp.on('end', () => {
            let go2rtcStreams = {};
            try {
                go2rtcStreams = JSON.parse(rawData);
            } catch (e) {
                console.error(`Error parsing streams response: ${e.message}`);
            }
            
            const status = {};
            (config.cameras || []).forEach(cam => {
                const streamKey = `camera_${cam.id}`;
                let connected = false;
                
                if (go2rtcStreams[streamKey]) {
                    const producers = go2rtcStreams[streamKey].producers || [];
                    if (producers.length > 0) {
                        connected = true;
                    }
                }
                
                status[cam.id] = {
                    name: cam.name,
                    url: cam.url,
                    connected: connected,
                    running: true
                };
            });
            res.json(status);
        });
    });
    
    reqStatus.on('timeout', () => {
        console.warn('go2rtc status query timed out, destroying request.');
        reqStatus.destroy();
    });
    
    reqStatus.on('error', (err) => {
        console.error(`Error querying go2rtc status: ${err.message}`);
        const status = {};
        (config.cameras || []).forEach(cam => {
            status[cam.id] = {
                name: cam.name,
                url: cam.url,
                connected: false,
                running: false
            };
        });
        res.json(status);
    });
    
    reqStatus.end();
});

// Proxy for go2rtc /api/streams — avoids browser CORS block (different port 1984 vs 5000)
app.get('/api/go2rtc/streams', (req, res) => {
    const options = {
        hostname: '127.0.0.1',
        port: 1984,
        path: '/api/streams',
        method: 'GET',
        timeout: 3000
    };
    let responded = false;
    const sendError = (status, msg) => {
        if (responded) return;
        responded = true;
        res.status(status).json({ error: msg });
    };

    const proxyReq = http.request(options, (proxyRes) => {
        let raw = '';
        proxyRes.on('data', d => { raw += d; });
        proxyRes.on('end', () => {
            if (responded) return;
            responded = true;
            try {
                res.json(JSON.parse(raw));
            } catch(e) {
                res.status(500).json({ error: 'Invalid response from go2rtc' });
            }
        });
    });
    proxyReq.on('error', () => {
        sendError(503, 'go2rtc not available');
    });
    proxyReq.on('timeout', () => {
        proxyReq.destroy();
        sendError(504, 'go2rtc timeout');
    });
    proxyReq.end();
});

// Track stream start times for uptime calculation (keyed by camera id)
const streamStartTimes = {};
const prevBytesRecv = {};

// Snapshot endpoint: grab JPEG frame directly from go2rtc (instant — no Python/FFmpeg needed)
// go2rtc's /api/frame.jpeg uses the already-connected stream, so it's near-instant (<100ms)
// If stream is not yet connected, warmup is triggered and one retry is attempted after 2 seconds.
app.get('/api/snapshot/:id', (req, res) => {
    const camId = parseInt(req.params.id);
    const config = loadConfig();
    const cam = (config.cameras || []).find(c => c.id === camId);

    if (!cam || !cam.url) {
        return res.status(404).send('Camera not found');
    }

    const streamKey = `camera_${camId}`;
    let responded = false;

    function sendError(status, msg) {
        if (responded) return;
        responded = true;
        res.status(status).send(msg);
    }

    function grabFromPythonService() {
        if (responded) return;
        console.log(`Snapshot camera_${camId}: attempting fallback via Python AI Service...`);
        const pyReq = http.get(`http://127.0.0.1:5001/api/snapshot/${camId}`, (pyRes) => {
            if (pyRes.statusCode !== 200) {
                pyRes.resume();
                return sendError(502, 'Failed to capture frame — camera stream offline');
            }
            const chunks = [];
            pyRes.on('data', chunk => chunks.push(chunk));
            pyRes.on('end', () => {
                if (responded) return;
                const buffer = Buffer.concat(chunks);
                if (buffer.length < 3) return sendError(500, 'Empty frame');
                responded = true;
                const camName = cam.name.replace(/\s+/g, '_').replace(/[^\w-]/g, '');
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').split('.')[0];
                const filename = `Snapshot_${camName}_${timestamp}.jpg`;
                res.setHeader('Content-Type', 'image/jpeg');
                res.setHeader('Content-Disposition', `inline; filename="${filename}"`);
                res.setHeader('Content-Length', buffer.length);
                res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
                res.end(buffer);
                console.log(`Snapshot camera_${camId}: OK via Python AI Service, sent ${buffer.length} bytes`);
            });
        });
        pyReq.on('error', (err) => {
            console.error(`Snapshot camera_${camId}: Python AI Service fallback error: ${err.message}`);
            sendError(502, 'Failed to capture frame — stream offline');
        });
        pyReq.setTimeout(4000, () => {
            pyReq.destroy();
            sendError(504, 'Snapshot timeout — stream not responding');
        });
    }

    function grabFrame(isRetry) {
        const frameOptions = {
            hostname: '127.0.0.1',
            port: 1984,
            path: `/api/frame.jpeg?src=${streamKey}&width=0&height=0`,
            method: 'GET',
            timeout: isRetry ? 5000 : 3000
        };

        const frameReq = http.request(frameOptions, (frameRes) => {
            if (frameRes.statusCode !== 200) {
                frameRes.resume();
                console.log(`Snapshot camera_${camId}: go2rtc HTTP ${frameRes.statusCode}, trying Python AI Service fallback...`);
                return grabFromPythonService();
            }

            const chunks = [];
            frameRes.on('data', chunk => chunks.push(chunk));
            frameRes.on('end', () => {
                if (responded) return;
                const buffer = Buffer.concat(chunks);

                if (buffer.length < 3) {
                    console.error(`Snapshot camera_${camId}: empty frame response from go2rtc, trying Python fallback...`);
                    return grabFromPythonService();
                }

                responded = true;
                const camName = cam.name.replace(/\s+/g, '_').replace(/[^\w-]/g, '');
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').split('.')[0];
                const filename = `Snapshot_${camName}_${timestamp}.jpg`;

                res.setHeader('Content-Type', 'image/jpeg');
                res.setHeader('Content-Disposition', `inline; filename="${filename}"`);
                res.setHeader('Content-Length', buffer.length);
                res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
                res.end(buffer);
                console.log(`Snapshot camera_${camId}: OK via go2rtc, sent ${buffer.length} bytes`);
            });
        });

        frameReq.on('timeout', () => {
            frameReq.destroy();
            console.error(`Snapshot camera_${camId}: go2rtc frame request timed out, trying Python fallback...`);
            grabFromPythonService();
        });

        frameReq.on('error', (err) => {
            if (responded) return;
            console.error(`Snapshot camera_${camId}: go2rtc error (${err.message}), trying Python fallback...`);
            grabFromPythonService();
        });

        frameReq.end();
    }

    console.log(`Snapshot camera_${camId}: grabbing frame (stream: ${streamKey})`);
    grabFrame(false);
});

// Uptime endpoint: returns how long each active camera has been streaming
app.get('/api/uptime', async (req, res) => {
    const options = {
        hostname: '127.0.0.1',
        port: 1984,
        path: '/api/streams',
        method: 'GET',
        timeout: 2000
    };
    const reqGo2rtc = http.request(options, (r) => {
        let raw = '';
        r.on('data', d => { raw += d; });
        r.on('end', () => {
            const now = Date.now();
            const result = {};
            let streams = {};
            try { streams = JSON.parse(raw); } catch(e) {}

            const config = loadConfig();
            (config.cameras || []).forEach(cam => {
                const key = `camera_${cam.id}`;
                const stream = streams[key];
                let totalBytes = 0;
                let active = false;
                if (stream && stream.producers) {
                    stream.producers.forEach(p => {
                        if (p.receivers && p.receivers.length > 0) {
                            active = true;
                            p.receivers.forEach(r => { totalBytes += (r.bytes || 0); });
                        }
                    });
                }

                // Track when stream first became active using bytes_recv
                const prev = prevBytesRecv[cam.id] || 0;
                if (active && totalBytes > 0 && prev === 0) {
                    // Stream just came online
                    streamStartTimes[cam.id] = now;
                } else if (!active) {
                    // Stream went offline
                    delete streamStartTimes[cam.id];
                }
                prevBytesRecv[cam.id] = totalBytes;

                const startTime = streamStartTimes[cam.id];
                result[cam.id] = {
                    active,
                    uptimeMs: startTime ? (now - startTime) : 0,
                    startedAt: startTime || null
                };
            });
            res.json(result);
        });
    });
    reqGo2rtc.on('error', () => {
        res.json({});
    });
    reqGo2rtc.on('timeout', () => { reqGo2rtc.destroy(); });
    reqGo2rtc.end();
});

// Logs endpoint
app.get('/api/logs', (req, res) => {
    res.json(systemLogs);
});

// Drowsiness history endpoint
app.get('/api/drowsiness/history', (req, res) => {
    const camId = req.query.cam_id;
    let url = 'http://127.0.0.1:5001/drowsiness/history';
    if (camId !== undefined) {
        url += `?cam_id=${camId}`;
    }
    const http = require('http');
    http.get(url, (resp) => {
        let data = '';
        resp.on('data', chunk => data += chunk);
        resp.on('end', () => {
            res.setHeader('Content-Type', 'application/json');
            res.status(resp.statusCode).send(data);
        });
    }).on('error', (e) => {
        res.status(500).json({ success: false, error: e.message });
    });
});

// Helper: parse IP address from RTSP/HTTP URL
function getCameraIP(url) {
    if (!url) return null;
    const match = url.match(/@([^:\/]+)/);
    if (match) return match[1];
    const hostMatch = url.match(/rtsp:\/\/([^:\/]+)/);
    if (hostMatch) return hostMatch[1];
    return null;
}

// ONVIF PTZ continuous move SOAP builder
function generateWSSecurity(username, password) {
    const nonce = crypto.randomBytes(16);
    const created = new Date().toISOString();
    const nonceBase64 = nonce.toString('base64');
    const sha1 = crypto.createHash('sha1');
    sha1.update(Buffer.concat([nonce, Buffer.from(created), Buffer.from(password)]));
    const digest = sha1.digest('base64');
    return `<Security s:mustUnderstand="1" xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
      <UsernameToken>
        <Username>${username}</Username>
        <Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">${digest}</Password>
        <Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd#Base64Binary">${nonceBase64}</Nonce>
        <Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">${created}</Created>
      </UsernameToken>
    </Security>`;
}

function buildPTZMoveSOAP(username, password, profileToken, pan, tilt, zoom) {
    const sec = generateWSSecurity(username, password);
    return `<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
            xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Header>${sec}</s:Header>
  <s:Body>
    <tptz:ContinuousMove>
      <tptz:ProfileToken>${profileToken}</tptz:ProfileToken>\n      <tptz:Velocity>\n        <tt:PanTilt x="${pan}" y="${tilt}"/>\n        <tt:Zoom x="${zoom}"/>\n      </tptz:Velocity>\n    </tptz:ContinuousMove>\n  </s:Body>\n</s:Envelope>`;
}

function buildPTZStopSOAP(username, password, profileToken) {
    const sec = generateWSSecurity(username, password);
    return `<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
  <s:Header>${sec}</s:Header>
  <s:Body>
    <tptz:Stop>
      <tptz:ProfileToken>${profileToken}</tptz:ProfileToken>
      <tptz:PanTilt>true</tptz:PanTilt>
      <tptz:Zoom>true</tptz:Zoom>
    </tptz:Stop>
  </s:Body>
</s:Envelope>`;
}

function buildGetProfilesSOAP(username, password) {
    const sec = generateWSSecurity(username, password);
    return `<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
  <s:Header>${sec}</s:Header>
  <s:Body><trt:GetProfiles/></s:Body>
</s:Envelope>`;
}

function buildGetDeviceInformationSOAP(username, password) {
    const sec = generateWSSecurity(username, password);
    return `<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <s:Header>${sec}</s:Header>
  <s:Body><tds:GetDeviceInformation/></s:Body>
</s:Envelope>`;
}


async function sendOnvifSOAP(host, port, path, soapAction, soapBody) {
    return new Promise((resolve, reject) => {
        const bodyBuf = Buffer.from(soapBody, 'utf8');
        const options = {
            hostname: host, port: parseInt(port),
            path: path,
            method: 'POST',
            timeout: 5000,
            headers: {
                'Content-Type': 'application/soap+xml; charset=utf-8',
                'SOAPAction': soapAction,
                'Content-Length': bodyBuf.length
            }
        };
        const req = http.request(options, (res) => {
            let data = '';
            res.on('data', d => { data += d; });
            res.on('end', () => resolve({ status: res.statusCode, body: data }));
        });
        req.on('timeout', () => { req.destroy(); reject(new Error('ONVIF request timeout')); });
        req.on('error', reject);
        req.write(bodyBuf);
        req.end();
    });
}

// ONVIF continuous move endpoint
app.post('/api/ptz/:id/move', async (req, res) => {
    const camId = parseInt(req.params.id);
    const { action, speed } = req.body;
    const config = loadConfig();
    const cam = (config.cameras || []).find(c => c.id === camId);
    if (!cam || !cam.url) {
        return res.status(404).json({ success: false, message: 'Kamera tidak ditemukan' });
    }
    const ip = getCameraIP(cam.url);
    const port = cam.onvifPort || 80;
    const username = cam.onvifUsername || 'admin';
    const password = cam.onvifPassword || '';

    if (!ip) {
        return res.status(400).json({ success: false, message: 'Tidak dapat mendeteksi IP Address kamera' });
    }

    log(`PTZ ContinuousMove cam_${camId} (${ip}:${port}) action: ${action}`, 'ptz');

    // Get correct profile token first
    let profileToken = 'Profile_1';
    try {
        const getProfilesSoap = buildGetProfilesSOAP(username, password);
        const onvifRes = await sendOnvifSOAP(ip, port, '/onvif/media_service', 'http://www.onvif.org/ver10/media/wsdl/GetProfiles', getProfilesSoap);
        if (onvifRes.status === 200) {
            const match = onvifRes.body.match(/token="([^"]+)"/);
            if (match) {
                profileToken = match[1];
            }
        } else {
            log(`GetProfiles returned status ${onvifRes.status}, falling back to Profile_1`, 'ptz-warn');
        }
    } catch(e) {
        log(`Failed to fetch ProfileToken, using Profile_1: ${e.message}`, 'ptz-warn');
    }

    let pan = 0;
    let tilt = 0;
    let zoom = 0;
    const ptzSpeed = speed || 0.4;

    switch(action) {
        case 'up': tilt = ptzSpeed; break;
        case 'down': tilt = -ptzSpeed; break;
        case 'left': pan = -ptzSpeed; break;
        case 'right': pan = ptzSpeed; break;
        case 'zoom_in': zoom = ptzSpeed; break;
        case 'zoom_out': zoom = -ptzSpeed; break;
        default:
            return res.status(400).json({ success: false, message: 'Aksi PTZ tidak valid' });
    }

    try {
        const moveSoap = buildPTZMoveSOAP(username, password, profileToken, pan, tilt, zoom);
        const onvifRes = await sendOnvifSOAP(ip, port, '/onvif/ptz_service', 'http://www.onvif.org/ver20/ptz/wsdl/ContinuousMove', moveSoap);
        if (onvifRes.status === 200) {
            res.json({ success: true, message: `ContinuousMove ${action} dikirim` });
        } else if (onvifRes.status === 400) {
            log(`PTZ Move rejected (400) - camera model does not support physical PTZ motors`, 'ptz-warn');
            res.status(400).json({ success: false, message: 'Kamera ini tidak mendukung gerakan PTZ fisik (lensa tetap / non-PTZ)' });
        } else {
            log(`PTZ Move failed with camera status ${onvifRes.status}`, 'ptz-error');
            res.status(500).json({ success: false, message: `Kamera mengembalikan status error ${onvifRes.status}` });
        }
    } catch(e) {
        log(`PTZ Move failed: ${e.message}`, 'ptz-error');
        res.status(500).json({ success: false, message: `Gagal menggerakkan kamera: Kamera tidak terhubung / non-PTZ` });
    }
});

// ONVIF stop endpoint
app.post('/api/ptz/:id/stop', async (req, res) => {
    const camId = parseInt(req.params.id);
    const config = loadConfig();
    const cam = (config.cameras || []).find(c => c.id === camId);
    if (!cam || !cam.url) {
        return res.status(404).json({ success: false, message: 'Kamera tidak ditemukan' });
    }
    const ip = getCameraIP(cam.url);
    const port = cam.onvifPort || 80;
    const username = cam.onvifUsername || 'admin';
    const password = cam.onvifPassword || '';

    if (!ip) {
        return res.status(400).json({ success: false, message: 'Tidak dapat mendeteksi IP Address kamera' });
    }

    log(`PTZ Stop cam_${camId} (${ip}:${port})`, 'ptz');

    let profileToken = 'Profile_1';
    try {
        const getProfilesSoap = buildGetProfilesSOAP(username, password);
        const onvifRes = await sendOnvifSOAP(ip, port, '/onvif/media_service', 'http://www.onvif.org/ver10/media/wsdl/GetProfiles', getProfilesSoap);
        if (onvifRes.status === 200) {
            const match = onvifRes.body.match(/token="([^"]+)"/);
            if (match) {
                profileToken = match[1];
            }
        }
    } catch(e) {}

    try {
        const stopSoap = buildPTZStopSOAP(username, password, profileToken);
        const onvifRes = await sendOnvifSOAP(ip, port, '/onvif/ptz_service', 'http://www.onvif.org/ver20/ptz/wsdl/Stop', stopSoap);
        if (onvifRes.status === 200) {
            res.json({ success: true, message: 'PTZ Stop dikirim' });
        } else {
            res.status(500).json({ success: false, message: `Kamera mengembalikan status error ${onvifRes.status}` });
        }
    } catch(e) {
        log(`PTZ Stop failed: ${e.message}`, 'ptz-error');
        res.status(500).json({ success: false, message: `Gagal menghentikan kamera: ${e.message}` });
    }
});

// AI mode control route
app.post('/api/ai/:id/mode', (req, res) => {
    const camId = parseInt(req.params.id);
    const { mode, selected_classes } = req.body;
    const postData = JSON.stringify({ cam_id: camId, mode, selected_classes });
    const postReq = http.request({
        hostname: '127.0.0.1',
        port: 5001,
        path: '/mode',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(postData)
        }
    }, (postRes) => {
        let data = '';
        postRes.on('data', chunk => data += chunk);
        postRes.on('end', () => {
            res.status(postRes.statusCode).send(data);
        });
    });
    postReq.on('error', (e) => res.status(500).json({ success: false, error: e.message }));
    postReq.write(postData);
    postReq.end();
});

// Proxy to get all camera AI modes
app.get('/api/ai/modes', (req, res) => {
    http.get('http://127.0.0.1:5001/active_modes', (metaRes) => {
        let data = '';
        metaRes.on('data', chunk => data += chunk);
        metaRes.on('end', () => {
            res.setHeader('Content-Type', 'application/json');
            res.status(metaRes.statusCode).send(data);
        });
    }).on('error', (e) => {
        res.status(500).json({ success: false, error: e.message });
    });
});

// AI clear sentence route
app.post('/api/ai/:id/clear_sentence', (req, res) => {
    const camId = parseInt(req.params.id);
    const postData = JSON.stringify({ cam_id: camId });
    const options = {
        hostname: '127.0.0.1',
        port: 5001,
        path: '/clear_sentence',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(postData)
        }
    };
    const proxy = http.request(options, (aiRes) => {
        let data = '';
        aiRes.on('data', chunk => data += chunk);
        aiRes.on('end', () => {
            res.setHeader('Content-Type', 'application/json');
            res.status(aiRes.statusCode).send(data);
        });
    });
    proxy.on('error', (e) => {
        res.status(500).json({ success: false, error: e.message });
    });
    proxy.write(postData);
    proxy.end();
});

// AI metadata retrieval route
app.get('/api/ai/:id/metadata', (req, res) => {
    const camId = parseInt(req.params.id);
    http.get(`http://127.0.0.1:5001/metadata?cam_id=${camId}`, (metaRes) => {
        let data = '';
        metaRes.on('data', chunk => data += chunk);
        metaRes.on('end', () => {
            res.setHeader('Content-Type', 'application/json');
            res.status(metaRes.statusCode).send(data);
        });
    }).on('error', (e) => {
        res.status(500).json({ success: false, error: e.message });
    });
});

// AI stream proxy endpoint (pipes chunked MJPEG stream)
app.get('/api/ai/stream', (req, res) => {
    const camId = req.query.cam_id;
    if (camId === undefined) {
        return res.status(400).send('Missing cam_id query parameter');
    }
    
    const options = {
        hostname: '127.0.0.1',
        port: 5001,
        path: `/stream?cam_id=${camId}`,
        method: 'GET',
        timeout: 10000
    };

    const proxyReq = http.request(options, (proxyRes) => {
        res.writeHead(proxyRes.statusCode, proxyRes.headers);
        proxyRes.pipe(res);
    });

    proxyReq.on('error', (err) => {
        console.error(`AI stream proxy error for cam_${camId}: ${err.message}`);
        if (!res.headersSent) {
            res.status(502).send('AI Service stream not available');
        }
    });

    req.on('close', () => {
        proxyReq.destroy();
    });

    proxyReq.end();
});



// Diagnostic endpoint: Ping camera IP
app.post('/api/diagnose/:id/ping', (req, res) => {
    const camId = parseInt(req.params.id);
    const config = loadConfig();
    const cam = (config.cameras || []).find(c => c.id === camId);
    if (!cam || !cam.url) {
        return res.status(404).json({ error: 'Kamera tidak ditemukan' });
    }
    const ip = getCameraIP(cam.url);
    if (!ip) {
        return res.status(400).json({ error: 'Tidak dapat mendeteksi IP Address kamera' });
    }

    log(`Diagnostics: Pinging camera_${camId} at ${ip}...`, 'system');
    const cmd = process.platform === 'win32' ? `ping -n 3 ${ip}` : `ping -c 3 ${ip}`;
    exec(cmd, (err, stdout, stderr) => {
        res.json({ success: !err, output: stdout || stderr });
    });
});

// Diagnostic endpoint: ONVIF capabilities probe (manufacturer, model, firmware, serial)
app.post('/api/diagnose/:id/onvif', async (req, res) => {
    const camId = parseInt(req.params.id);
    const config = loadConfig();
    const cam = (config.cameras || []).find(c => c.id === camId);
    if (!cam || !cam.url) {
        return res.status(404).json({ error: 'Kamera tidak ditemukan' });
    }
    const ip = getCameraIP(cam.url);
    const port = cam.onvifPort || 8000;
    const username = cam.onvifUsername || 'admin';
    const password = cam.onvifPassword || '';

    if (!ip) {
        return res.status(400).json({ error: 'Tidak dapat mendeteksi IP Address kamera' });
    }

    log(`Diagnostics: Querying ONVIF GetDeviceInformation for camera_${camId} (${ip}:${port})...`, 'system');

    try {
        const soap = buildGetDeviceInformationSOAP(username, password);
        const result = await sendOnvifSOAP(ip, port, '/onvif/device_service', 'http://www.onvif.org/ver10/device/wsdl/GetDeviceInformation', soap);
        if (result.status === 200) {
            const manufacturer = (result.body.match(/<[^:]*:Manufacturer>([^<]+)/) || [])[1] || 'Unknown';
            const model = (result.body.match(/<[^:]*:Model>([^<]+)/) || [])[1] || 'Unknown';
            const firmwareVersion = (result.body.match(/<[^:]*:FirmwareVersion>([^<]+)/) || [])[1] || 'Unknown';
            const serialNumber = (result.body.match(/<[^:]*:SerialNumber>([^<]+)/) || [])[1] || 'Unknown';
            
            res.json({
                success: true,
                output: `=== ONVIF Device Information ===\n` +
                        `Manufacturer     : ${manufacturer}\n` +
                        `Model            : ${model}\n` +
                        `Firmware Version : ${firmwareVersion}\n` +
                        `Serial Number    : ${serialNumber}\n` +
                        `================================`
            });
        } else {
            res.status(500).json({ error: `ONVIF returned status ${result.status}`, output: result.body });
        }
    } catch(e) {
        res.status(500).json({ error: e.message, output: `ONVIF connection error: ${e.message}` });
    }
});


// Run server
app.listen(PORT, '0.0.0.0', () => {
    log(`Server is running at http://localhost:${PORT}`);
});
