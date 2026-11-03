import html
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
SEEN_FILE = Path(".scanner_state/seen_signals.json")
SEEN_TTL_SEC = 24 * 60 * 60
IST = ZoneInfo("Asia/Kolkata")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ignition-scanner")
_thread_local = threading.local()


def _session():
    """Give each worker its own HTTP session with bounded retries."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        retry = Retry(
            total=4,
            connect=3,
            read=3,
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
        session.headers.update({"User-Agent": "ignition-scanner/2.0"})
        _thread_local.session = session
    return session


def get_json(path, params=None):
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
        if item.get("quoteAsset") != "USDT":
            continue
        if item.get("status") != "TRADING":
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

    This is exactly equivalent to Open == High == Close and Low < High.
    Decimal is used so equality is exact rather than floating-point approximate.
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
                parameters = result.get("parameters") or {}
                retry_after = parameters.get("retry_after", 2)
                try:
                    delay = int(retry_after)
                except (TypeError, ValueError):
                    delay = 2
                time.sleep(min(max(delay, 1), 30))
                continue

            return response.status_code == 200 and result.get("ok") is True
        except requests.RequestException as exc:
            # Do not log the exception string: request URLs contain the Telegram bot token.
            log.warning("Telegram request failed (attempt %s/3): %s", attempt + 1, type(exc).__name__)
            if attempt < 2:
                time.sleep(2 ** attempt)

    return False


def load_seen():
    try:
        data = json.loads(SEEN_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read scanner state; starting with an empty state: %s", exc)
        return {}


def save_seen(seen):
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = SEEN_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(seen, separators=(",", ":")), encoding="utf-8")
    temporary_file.replace(SEEN_FILE)


def set_github_output(name, value):
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")


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


def main():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID GitHub Actions secret")

    log.info("Scanner started; rule: Body 0%%, Upper Wick 0%%, Lower Wick 100%%")
    server_time_ms = fetch_server_time_ms()
    symbols = fetch_universe()
    expected_open_ms = (server_time_ms // INTERVAL_MS) * INTERVAL_MS - INTERVAL_MS
    log.info("Binance Spot USDT pairs: %s", len(symbols))
    log.info("Latest expected closed 15m candle open time (UTC ms): %s", expected_open_ms)

    hits = []
    scan_errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(scan_symbol, symbol, server_time_ms) for symbol in symbols]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # Keep one unexpected pair failure from aborting the full scan.
                scan_errors += 1
                log.warning("Unexpected worker failure: %s", exc)
                continue
            if result is None:
                continue
            if result[0] == "__SCAN_ERROR__":
                scan_errors += 1
            else:
                hits.append(result)

    log.info("Scan complete: %s pairs checked; %s errors; %s exact T matches", len(symbols), scan_errors, len(hits))

    now = time.time()
    cutoff = now - SEEN_TTL_SEC
    seen = {
        key: timestamp
        for key, timestamp in load_seen().items()
        if isinstance(timestamp, (int, float)) and timestamp > cutoff
    }

    sent_count = 0
    failed_alerts = 0
    for symbol, candle in sorted(hits, key=lambda item: item[0]):
        key = f"{symbol}|{candle['open_time_ms']}"
        if key in seen:
            log.info("Duplicate prevented: %s", key)
            continue
        if telegram(format_alert(symbol, candle)):
            seen[key] = time.time()
            sent_count += 1
            log.info("Telegram alert sent: %s", symbol)
        else:
            failed_alerts += 1
            log.error("Telegram alert failed: %s", symbol)

    if sent_count:
        save_seen(seen)
        set_github_output("cache_changed", "true")
    else:
        set_github_output("cache_changed", "false")

    log.info("Finished: %s alerts sent; %s alert failures", sent_count, failed_alerts)


if __name__ == "__main__":
    main()
