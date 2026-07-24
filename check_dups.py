import sqlite3

conn = sqlite3.connect('/home/nabil/Camera/zone_config.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== CHECKING DUPLICATES ===")
cur.execute('SELECT cam_id, name, COUNT(*) as cnt FROM zones GROUP BY cam_id, name HAVING cnt > 1')
dups = cur.fetchall()
print(f"Duplicate Groups Found: {len(dups)}")
for d in dups:
    print(dict(d))

print("\n=== ALL ZONES IN DATABASE ===")
cur.execute('SELECT id, zone_id, cam_id, name, threshold_minutes, cycle_hours, grace_period_seconds, created_at FROM zones')
rows = cur.fetchall()
for r in rows:
    print(dict(r))
