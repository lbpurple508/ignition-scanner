#!/usr/bin/env python3

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, DecimalException, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / ".scanner_state"
SEEN_FILE = STATE_DIR / "seen_signals.json"
ENTRY_FILE = STATE_DIR / "entry_state.json"
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
TELEGRAM_URL = "https://api.telegram.org/bot{}/sendMessage"
IST = ZoneInfo("Asia/Kolkata")
MAX_AGE_SECONDS = Decimal(12 * 60 * 60)
MAX_TOKEN_PRICE_USDT = Decimal("3")  # Maximum live entry price allowed
STALE_MS = 5 * 60 * 1000
VALID_SYMBOL = re.compile(r"^[A-Z0-9]+USDT$")


def as_decimal(value):
    if isinstance(value, bool):
        raise ValueError("invalid number")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("invalid number") from exc
    if not number.is_finite():
        raise ValueError("non-finite number")
    return number


def parse_key(key):
    if not isinstance(key, str) or key.count("|") != 1:
        raise ValueError("invalid key")
    symbol, day_text = key.split("|", 1)
    if not VALID_SYMBOL.fullmatch(symbol):
        raise ValueError("invalid symbol")
    day = datetime.strptime(day_text, "%Y-%m-%d").date()
    if day.isoformat() != day_text:
        raise ValueError("invalid date")
    return symbol, day


def load_seen():
    try:
        payload = json.loads(SEEN_FILE.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != 2:
        return {}
    seen = payload.get("seen")
    return seen if isinstance(seen, dict) else {}


def load_entry_state():
    try:
        payload = json.loads(ENTRY_FILE.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    clean = {}
    for key, value in payload.items():
        if value not in ("entered", "expired"):
            continue
        try:
            parse_key(key)
        except (TypeError, ValueError):
            continue
        clean[key] = value
    return clean


def save_entry_state(state):
    temporary = ENTRY_FILE.with_name(ENTRY_FILE.name + ".tmp")
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, separators=(",", ":"), allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, ENTRY_FILE)
        return True
    except (OSError, TypeError, ValueError):
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def prune_entry_state(state, seen, today):
    changed = False
    cutoff = today - timedelta(days=7)
    for key in list(state):
        try:
            _, alert_day = parse_key(key)
        except (TypeError, ValueError):
            del state[key]
            changed = True
            continue
        if key not in seen and alert_day < cutoff:
            del state[key]
            changed = True
    return changed


def fetch_candles(symbol):
    try:
        response = requests.get(
            BINANCE_URL,
            params={"symbol": symbol, "interval": "1m", "limit": 60},
            timeout=(5, 15),
            headers={"User-Agent": "data-watcher/1.0"},
        )
        response.raise_for_status()
        rows = response.json()
    except (requests.RequestException, ValueError, TypeError):
        return None
    if not isinstance(rows, list) or len(rows) < 35:
        return None
    return rows


def candle_fields(row):
    if not isinstance(row, (list, tuple)) or len(row) < 7:
        raise ValueError("malformed candle")
    close_time = as_decimal(row[6])
    if (
        close_time != close_time.to_integral_value()
        or close_time <= 0
        or int(close_time) % 60_000 != 59_999
    ):
        raise ValueError("invalid candle time")
    close = as_decimal(row[4])
    volume = as_decimal(row[5])
    if close <= 0 or volume < 0:
        raise ValueError("invalid candle values")
    return int(close_time), close, volume


def evaluate(rows, now_ms):
    if not isinstance(rows, list) or len(rows) < 35:
        return None
    try:
        recent = [candle_fields(row) for row in rows[-32:]]
        for previous, current in zip(recent, recent[1:]):
            if current[0] - previous[0] != 60_000:
                return None

        prior = recent[:30]
        latest_time, latest_close, latest_volume = recent[-2]
        _, reference_close, _ = recent[-7]
        if now_ms < latest_time or now_ms - latest_time > STALE_MS:
            return None
        if reference_close <= 0 or latest_close <= 0 or latest_volume <= 0:
            return None

        avg_volume = sum((item[2] for item in prior), Decimal("0")) / Decimal(30)
        if not avg_volume.is_finite() or avg_volume <= 0:
            return None

        ret_5m = (latest_close - reference_close) / reference_close * Decimal(100)
        vol_ratio = latest_volume / avg_volume
        if not ret_5m.is_finite() or not vol_ratio.is_finite():
            return None
        if (
            ret_5m > Decimal("1.5")
            and vol_ratio > Decimal("3.0")
            and latest_close <= MAX_TOKEN_PRICE_USDT
        ):
            return {
                "P1": latest_close,
                "P2": latest_close * Decimal("0.95"),
                "P3": latest_close * Decimal("1.30"),
                "R1": ret_5m,
                "V1": vol_ratio,
            }
    except (IndexError, TypeError, ValueError, DecimalException, OverflowError):
        return None
    return None


def format_decimal(value):
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def format_two(value):
    return format(value.quantize(Decimal("0.01")), "f")


def send_telegram(message):
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    try:
        response = requests.post(
            TELEGRAM_URL.format(token),
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=(5, 15),
        )
        response.raise_for_status()
        result = response.json()
        return isinstance(result, dict) and result.get("ok") is True
    except (requests.RequestException, ValueError, TypeError):
        return False


def expired_message(symbol, alert_time):
    fired = datetime.fromtimestamp(float(alert_time), tz=timezone.utc).astimezone(IST)
    return (
        "⏰ <b>EXPIRED</b>\n\n"
        f"<b>Pair:</b> {symbol}\n"
        f"<b>Alert fired:</b> {fired:%Y-%m-%d %H:%M} IST\n"
        "<b>No ignition in 12 hours. Skip this one.</b>"
    )


def entry_message(symbol, values):
    return (
        "🔥 <b>ENTRY</b>\n\n"
        f"<b>Pair:</b> {symbol}\n"
        f"<b>Entry:</b> {format_decimal(values['P1'])}\n"
        f"<b>Stop loss:</b> {format_decimal(values['P2'])} (-5.0%)\n"
        f"<b>Take profit:</b> {format_decimal(values['P3'])} (+30.0%)\n"
        f"<b>5min return:</b> {format_two(values['R1'])}%\n"
        f"<b>Volume ratio:</b> {format_two(values['V1'])}x"
    )


def process(now_seconds=None):
    seen = load_seen()
    state = load_entry_state()
    fixed_now = None if now_seconds is None else as_decimal(now_seconds)

    def current_time():
        return as_decimal(time.time()) if fixed_now is None else fixed_now

    initial_now = current_time()
    today = datetime.fromtimestamp(float(initial_now), tz=timezone.utc).astimezone(IST).date()
    changed = prune_entry_state(state, seen, today)
    entered = 0
    expired = 0

    for key, raw_timestamp in seen.items():
        if key in state:
            continue
        try:
            symbol, _ = parse_key(key)
            alert_time = as_decimal(raw_timestamp)
            if alert_time <= 0:
                continue

            now = current_time()
            age = now - alert_time
            if age < 0:
                continue
            if age >= MAX_AGE_SECONDS:
                if send_telegram(expired_message(symbol, alert_time)):
                    state[key] = "expired"
                    if not save_entry_state(state):
                        state.pop(key, None)
                    else:
                        expired += 1
                    changed = True
                continue

            rows = fetch_candles(symbol)
            if rows is None:
                continue

            check_now = current_time()
            age = check_now - alert_time
            if age < 0:
                continue
            if age >= MAX_AGE_SECONDS:
                if send_telegram(expired_message(symbol, alert_time)):
                    state[key] = "expired"
                    if not save_entry_state(state):
                        state.pop(key, None)
                    else:
                        expired += 1
                    changed = True
                continue

            values = evaluate(rows, int(check_now * Decimal(1000)))
            if values is not None and send_telegram(entry_message(symbol, values)):
                state[key] = "entered"
                if not save_entry_state(state):
                    state.pop(key, None)
                else:
                    entered += 1
                changed = True
        except Exception:
            continue

    if changed:
        save_entry_state(state)
    return entered, expired


def main():
    import traceback
    try:
        entered, expired = process()
    except Exception as e:
        print(f"watcher_error: {type(e).__name__}: {e}")
        traceback.print_exc()
        entered, expired = 0, 0
    print(f"entered={entered} expired={expired}")


if __name__ == "__main__":
    main()
