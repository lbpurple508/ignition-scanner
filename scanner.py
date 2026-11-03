#!/usr/bin/env python3
"""Scan Binance Spot USDT pairs for the user's exact 15-minute T candle."""

import fcntl
import html
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

BINANCE = "https://data-api.binance.vision"
INTERVAL = "15m"
INTERVAL_MS = 15 * 60 * 1000
MAX_WORKERS = 12
REQUESTS_PER_SECOND = 8  # 8 kline requests/s ~= 960 request-weight/minute.
BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / ".scanner_state"
STATE_FILE = STATE_DIR / "seen_signals.json"
LOCK_FILE = STATE_DIR / "scanner.lock"
SEEN_TTL_SEC = 24 * 60 * 60
IST = ZoneInfo("Asia/Kolkata")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ignition-scanner")
_thread_local = threading.local()
_request_lock = threading.Lock()
_next_request_at = 0.0


def _pace_request():
    """Keep aggregate Binance request weight comfortably below the public limit."""
    global _next_request_at
    with _request_lock:
        now = time.monotonic()
        delay = max(0.0, _next_request_at - now)
        if delay:
            time.sleep(delay)
        _next_request_at = max(time.monotonic(), _next_request_at) + 1.0 / REQUESTS_PER_SECOND


def _session():
    """Give each worker its own HTTP session with bounded GET retries."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        retry = Retry(
            total=4,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=MAX_WORKERS,
            pool_maxsize=MAX_WORKERS,
        )
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": "ignition-scanner/2.1"})
        _thread_local.session = session
    return session


def get_json(path, params=None):
    _pace_request()
    response = _session().get(f"{BINANCE}{path}", params=params, timeout=(5, 20))
    response.raise_for_status()
    return response.json()


def fetch_server_time_ms():
    payload = get_json("/api/v3/time")
    server_time = int(payload["serverTime"])
    if server_time <= 0:
        raise RuntimeError("Binance returned an invalid server time")
    return server_time


def fetch_universe():
    """Return every active Binance Spot symbol quoted in USDT; apply no market filters."""
    exchange_info = get_json("/api/v3/exchangeInfo")
    entries = exchange_info.get("symbols") if isinstance(exchange_info, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError("Binance returned an invalid exchange-info response")

    symbols = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        if item.get("quoteAsset") != "USDT" or item.get("status") != "TRADING":
            continue
        if item.get("isSpotTradingAllowed") is False:
            continue
        symbol = item.get("symbol")
        if isinstance(symbol, str) and symbol:
            symbols.append(symbol)

    if not symbols:
        raise RuntimeError("Binance returned no active USDT Spot pairs")
    return sorted(set(symbols))


def exact_t_candle(kline):
    """Match ONLY Body 0%, Upper Wick 0%, Lower Wick 100%.

    Exactly equivalent to Open == High == Close and Low < High.
    Decimal avoids approximate floating-point comparisons.
    """
    try:
        open_price = Decimal(str(kline[1]))
        high_price = Decimal(str(kline[2]))
        low_price = Decimal(str(kline[3]))
        close_price = Decimal(str(kline[4]))
        open_time_ms = int(kline[0])
        close_time_ms = int(kline[6])
    except (IndexError, TypeError, ValueError, InvalidOperation):
        return None

    prices = (open_price, high_price, low_price, close_price)
    if not all(price.is_finite() for price in prices):
        return None
    if not (open_price == high_price == close_price and low_price < high_price):
        return None

    return {
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "open_time_ms": open_time_ms,
        "close_time_ms": close_time_ms,
        "body_pct": Decimal("0"),
        "upper_wick_pct": Decimal("0"),
        "lower_wick_pct": Decimal("100"),
    }


def fetch_latest_closed_kline(symbol, server_time_ms):
    """Fetch the latest fully closed 15-minute candle for one pair."""
    klines = get_json(
        "/api/v3/klines",
        {"symbol": symbol, "interval": INTERVAL, "limit": 2},
    )
    if not isinstance(klines, list):
        return None

    for kline in reversed(klines):
        try:
            close_time_ms = int(kline[6])
        except (IndexError, TypeError, ValueError):
            continue
        if close_time_ms < server_time_ms:
            return kline
    return None


def fmt_price(value):
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def format_alert(symbol, candle):
    close_time = datetime.fromtimestamp(
        candle["close_time_ms"] / 1000, tz=timezone.utc
    ).astimezone(IST)
    return (
        "🟢 <b>15m T-CANDLE MATCH</b>\n\n"
        f"<b>Pair:</b> {html.escape(symbol)}\n"
        "<b>Body:</b> 0%\n"
        "<b>Upper Wick:</b> 0%\n"
        "<b>Lower Wick:</b> 100%\n\n"
        f"<b>Open:</b> {fmt_price(candle['open'])}\n"
        f"<b>High:</b> {fmt_price(candle['high'])}\n"
        f"<b>Low:</b> {fmt_price(candle['low'])}\n"
        f"<b>Close:</b> {fmt_price(candle['close'])}\n"
        f"<b>Candle closed:</b> {close_time:%Y-%m-%d %H:%M:%S} IST"
    )


def telegram(message):
    """Send with bounded retry for Telegram rate limits, transient 5xx and network errors."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("Telegram is not configured: set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID secrets")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    for attempt in range(3):
        try:
            response = _session().post(url, json=payload, timeout=(5, 20))
            try:
                result = response.json()
            except ValueError:
                result = {}
            if not isinstance(result, dict):
                result = {}

            if response.status_code == 429:
                parameters = result.get("parameters")
                if not isinstance(parameters, dict):
                    parameters = {}
                retry_after = parameters.get("retry_after", 2)
                try:
                    delay = int(retry_after)
                except (TypeError, ValueError):
                    delay = 2
                if attempt < 2:
                    time.sleep(min(max(delay, 1), 30))
                    continue
                return False

            if response.status_code >= 500:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return False

            return response.status_code == 200 and result.get("ok") is True
        except requests.RequestException as exc:
            # Do not log exception text: request URLs contain the Telegram bot token.
            log.warning("Telegram request failed (attempt %s/3): %s", attempt + 1, type(exc).__name__)
            if attempt < 2:
                time.sleep(2 ** attempt)

    return False


def load_state():
    """Load current state and migrate the previous flat seen-signals JSON format."""
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"seen": {}, "pending": {}}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read scanner state; starting with empty state: %s", exc)
        return {"seen": {}, "pending": {}}

    if not isinstance(data, dict):
        return {"seen": {}, "pending": {}}
    if isinstance(data.get("seen"), dict) and isinstance(data.get("pending"), dict):
        return {"seen": data["seen"], "pending": data["pending"]}
    # Backward compatibility with the original {"SYMBOL|open_time": timestamp} file.
    return {"seen": data, "pending": {}}


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_file = STATE_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(state, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    temporary_file.replace(STATE_FILE)


def serialize_candle(symbol, candle):
    return {
        "symbol": symbol,
        "open": str(candle["open"]),
        "high": str(candle["high"]),
        "low": str(candle["low"]),
        "close": str(candle["close"]),
        "open_time_ms": candle["open_time_ms"],
        "close_time_ms": candle["close_time_ms"],
    }


def deserialize_pending(record):
    if not isinstance(record, dict) or not isinstance(record.get("symbol"), str):
        raise ValueError("Invalid pending alert")
    candle = {
        "open": Decimal(str(record["open"])),
        "high": Decimal(str(record["high"])),
        "low": Decimal(str(record["low"])),
        "close": Decimal(str(record["close"])),
        "open_time_ms": int(record["open_time_ms"]),
        "close_time_ms": int(record["close_time_ms"]),
        "body_pct": Decimal("0"),
        "upper_wick_pct": Decimal("0"),
        "lower_wick_pct": Decimal("100"),
    }
    if not (candle["open"] == candle["high"] == candle["close"] and candle["low"] < candle["high"]):
        raise ValueError("Pending alert no longer satisfies the exact T-candle rule")
    return record["symbol"], candle


def scan_symbol(symbol, server_time_ms):
    try:
        kline = fetch_latest_closed_kline(symbol, server_time_ms)
        if kline is None:
            return None
        candle = exact_t_candle(kline)
        if candle is None:
            return None
        return symbol, candle
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        log.warning("Could not scan %s: %s", symbol, exc)
        return "__SCAN_ERROR__", symbol


def set_github_output(name, value):
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")


@contextmanager
def scanner_lock():
    """Prevent overlapping system-cron invocations on Linux and GitHub runners."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def run_scan():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variable")

    log.info("Scanner started; rule: Body 0%%, Upper Wick 0%%, Lower Wick 100%%")
    server_time_ms = fetch_server_time_ms()
    symbols = fetch_universe()
    log.info("Binance Spot USDT pairs: %s", len(symbols))

    hits = []
    scan_errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(scan_symbol, symbol, server_time_ms) for symbol in symbols]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                scan_errors += 1
                log.warning("Unexpected worker failure: %s", type(exc).__name__)
                continue
            if result is None:
                continue
            if result[0] == "__SCAN_ERROR__":
                scan_errors += 1
            else:
                hits.append(result)

    log.info("Scan complete: %s pairs checked; %s errors; %s exact T matches", len(symbols), scan_errors, len(hits))

    state = load_state()
    now = time.time()
    cutoff = now - SEEN_TTL_SEC
    old_seen = state["seen"]
    state["seen"] = {
        key: timestamp
        for key, timestamp in old_seen.items()
        if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
        and math.isfinite(float(timestamp)) and timestamp > cutoff
    }
    state_changed = state["seen"] != old_seen

    for symbol, candle in hits:
        key = f"{symbol}|{candle['open_time_ms']}"
        if key not in state["seen"] and key not in state["pending"]:
            state["pending"][key] = serialize_candle(symbol, candle)
            state_changed = True

    # Persist matches before sending, so a temporary Telegram failure is retried next cron run.
    if state_changed:
        save_state(state)

    sent_count = 0
    failed_alerts = 0
    for key in sorted(list(state["pending"])):
        if key in state["seen"]:
            del state["pending"][key]
            state_changed = True
            continue
        try:
            symbol, candle = deserialize_pending(state["pending"][key])
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            log.error("Removing invalid pending alert %s: %s", key, exc)
            del state["pending"][key]
            state_changed = True
            continue

        if telegram(format_alert(symbol, candle)):
            state["seen"][key] = time.time()
            del state["pending"][key]
            sent_count += 1
            state_changed = True
            save_state(state)
            log.info("Telegram alert sent: %s", symbol)
        else:
            failed_alerts += 1
            log.error("Telegram alert failed and will retry on next run: %s", symbol)

    if state_changed:
        save_state(state)
    set_github_output("cache_changed", "true" if state_changed else "false")
    log.info("Finished: %s alerts sent; %s alert failures; %s pending", sent_count, failed_alerts, len(state["pending"]))


def main():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variable")
    with scanner_lock() as acquired:
        if not acquired:
            log.warning("Another scanner instance is active; this invocation will exit")
            return
        run_scan()


if __name__ == "__main__":
    main()
