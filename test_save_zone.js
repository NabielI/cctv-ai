const http = require('http');

const data = JSON.stringify({
    zone_id: 'cam4_zone_test',
    cam_id: 4,
    name: 'nabil_test',
    coords: [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]],
    threshold_minutes: 5,
    cycle_hours: 1,
    telegram_enabled: true,
    start_hour: '08:00',
    grace_period_seconds: 60
});

const req = http.request('http://127.0.0.1:5000/api/zones', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data)
    }
}, (res) => {
    let body = '';
    res.on('data', d => body += d);
    res.on('end', () => {
        console.log('STATUS:', res.statusCode);
        console.log('BODY:', body);
    });
});

req.on('error', (e) => console.error('REQ ERROR:', e));
req.write(data);
req.end();
