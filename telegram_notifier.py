"""
telegram_notifier.py — Notifikasi Telegram untuk Zone Monitoring

Mengirim pesan notifikasi profesional menggunakan Telegram Bot API
dengan Markdown formatting (bold, italic, emoji) untuk tampilan yang rapi.
"""

import os
import time
import requests
from datetime import datetime
from typing import Optional

# ── Baca kredensial dari environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# URL API Telegram
TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # detik


def _get_api_url() -> str:
    """Bangun base URL Telegram API dengan token dari env."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
    return f"https://api.telegram.org/bot{token}"


def _is_configured() -> bool:
    """Cek apakah Telegram sudah dikonfigurasi dengan token dan chat_id."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)
    return bool(token and token != "" and chat_id and chat_id != "")


def send_message(text: str, parse_mode: str = "Markdown") -> bool:
    """
    Kirim pesan teks ke Telegram chat yang dikonfigurasi.

    Args:
        text: Isi pesan (mendukung Markdown/HTML formatting Telegram)
        parse_mode: "Markdown" atau "HTML"

    Returns:
        True jika berhasil, False jika gagal
    """
    if not _is_configured():
        print(
            "[TELEGRAM] WARNING: Telegram belum dikonfigurasi. "
            "Set TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID di .env",
            flush=True
        )
        return False

    chat_id = os.environ.get("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)
    url = f"{_get_api_url()}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    print(f"[TELEGRAM] OK Pesan terkirim (message_id={data['result']['message_id']})",
                          flush=True)
                    return True
                else:
                    print(f"[TELEGRAM] ERROR API error: {data.get('description', 'unknown')}", flush=True)
            else:
                print(f"[TELEGRAM] HTTP {resp.status_code}: {resp.text[:200]}", flush=True)

        except requests.exceptions.Timeout:
            print(f"[TELEGRAM] TIMEOUT (attempt {attempt}/{MAX_RETRIES})", flush=True)
        except requests.exceptions.ConnectionError as e:
            print(f"[TELEGRAM] CONNECTION ERROR (attempt {attempt}/{MAX_RETRIES}): {e}", flush=True)
        except Exception as e:
            print(f"[TELEGRAM] ERROR Unexpected error: {e}", flush=True)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    return False


def _build_alert_message(zone_name: str,
                          cam_id: int,
                          cycle_label: str,
                          accumulated_minutes: float,
                          threshold_minutes: int,
                          alert_type: str) -> str:
    """
    Bangun teks pesan notifikasi Telegram dengan Markdown formatting.

    alert_type:
        "low_presence"  → orang terdeteksi tapi kurang dari threshold
        "no_presence"   → tidak ada orang sama sekali dalam 1 jam
    """
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    accum_str = f"{accumulated_minutes:.1f}"
    percent = min(100, int((accumulated_minutes / threshold_minutes * 100))) if threshold_minutes > 0 else 0

    # Progress bar visual (10 blok)
    filled = int(percent / 10)
    bar = "🟥" * filled + "⬛" * (10 - filled)

    if alert_type == "no_presence":
        header = "🚫 *TIDAK ADA AKTIVITAS TERDETEKSI*"
        status_icon = "🔴"
        status_text = "Tidak ada orang terdeteksi"
        detail_line = f"📊 Kehadiran: *0 menit* dari *{threshold_minutes} menit* target"
    else:  # low_presence
        header = "⚠️ *KEHADIRAN DI BAWAH THRESHOLD*"
        status_icon = "🟡"
        status_text = "Orang terdeteksi, namun kurang dari batas minimum"
        detail_line = f"📊 Kehadiran: *{accum_str} menit* dari *{threshold_minutes} menit* target"

    # Parse cycle label untuk tampilan yang lebih ramah
    # "2026-07-23 09:00" → "Senin, 23 Jul 2026 | 09:00 – 10:00"
    try:
        dt = datetime.strptime(cycle_label.replace(" [TEST]", ""), "%Y-%m-%d %H:00")
        day_name = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][dt.weekday()]
        month_name = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"][dt.month - 1]
        cycle_display = f"{day_name}, {dt.day} {month_name} {dt.year} | {dt.hour:02d}:00 – {(dt.hour+1)%24:02d}:00"
    except Exception:
        cycle_display = cycle_label

    is_test = "[TEST]" in cycle_label
    test_badge = " _(TEST MODE)_" if is_test else ""

    message = (
        f"{header}{test_badge}\n"
        f"{'─' * 32}\n"
        f"\n"
        f"📍 *Zona:* {zone_name}\n"
        f"📷 *Kamera:* cam\\_{cam_id}\n"
        f"🕐 *Siklus:* {cycle_display}\n"
        f"\n"
        f"{detail_line}\n"
        f"{bar} {percent}%\n"
        f"\n"
        f"{status_icon} *Status:* {status_text}\n"
        f"\n"
        f"{'─' * 32}\n"
        f"🤖 _CCTV Zone Monitor_ | {now_str}"
    )

    return message


def send_zone_alert(zone_name: str,
                    cam_id: int,
                    cycle_label: str,
                    accumulated_minutes: float,
                    threshold_minutes: int,
                    alert_type: str) -> bool:
    """
    Kirim notifikasi Telegram untuk pelanggaran zone monitoring.

    Args:
        zone_name: Nama zona (misal "Meja Kerja A")
        cam_id: ID kamera
        cycle_label: Label siklus jam (misal "2026-07-23 09:00")
        accumulated_minutes: Total menit kehadiran terakumulasi
        threshold_minutes: Batas minimum menit yang harus terpenuhi
        alert_type: "low_presence" atau "no_presence"

    Returns:
        True jika pesan terkirim berhasil
    """
    print(
        f"[TELEGRAM] Mengirim alert '{alert_type}' untuk zona '{zone_name}' "
        f"(cam{cam_id}) | akumulasi: {accumulated_minutes:.1f}/{threshold_minutes}m",
        flush=True
    )

    msg = _build_alert_message(
        zone_name=zone_name,
        cam_id=cam_id,
        cycle_label=cycle_label,
        accumulated_minutes=accumulated_minutes,
        threshold_minutes=threshold_minutes,
        alert_type=alert_type,
    )

    return send_message(msg, parse_mode="Markdown")


def send_test_message() -> bool:
    """Kirim pesan test untuk memverifikasi konfigurasi Telegram."""
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    msg = (
        f"✅ *Koneksi Telegram Berhasil!*\n"
        f"{'─' * 30}\n"
        f"\n"
        f"🤖 CCTV Zone Monitor siap mengirim notifikasi.\n"
        f"📡 Bot terhubung dengan sukses.\n"
        f"\n"
        f"_Test dikirim pada: {now_str}_"
    )
    return send_message(msg, parse_mode="Markdown")


def get_bot_info() -> Optional[dict]:
    """Dapatkan informasi bot untuk verifikasi konfigurasi."""
    if not _is_configured():
        return None
    try:
        url = f"{_get_api_url()}/getMe"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("ok"):
                return data["result"]
    except Exception as e:
        print(f"[TELEGRAM] getMe error: {e}", flush=True)
    return None
