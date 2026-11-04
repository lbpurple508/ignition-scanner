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
REQUESTS_PER_SECOND = 8  # Binance kline request weight is 2: about 960 weight/minute.
BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / ".scanner_state"
STATE_FILE = STATE_DIR / "seen_signals.json"
LOCK_FILE = STATE_DIR / "scanner.lock"
SEEN_TTL_SEC = 24 * 60 * 60
IST = ZoneInfo("Asia/Kolkata")

# Binance bStocks/tokenized-security base assets, compiled from Binance Spot listings.
# This is deliberately an explicit list; do NOT exclude every ticker ending in "B",
# because ordinary crypto assets such as BNB also end in B.
STOCK_LINKED_ASSETS = frozenset({
    "AAPLB", "AAOIB", "ADBEB", "ALABB", "AMATB", "AMCB", "AMDB", "AMZNB",
    "APPB", "ARMB", "ASMLB", "ASTSB", "AVGOB", "BABAB", "BBB", "BEB",
    "BMNRB", "BNCB", "BOTB", "BRKBB", "BSPB", "CBRSB", "COHRB", "COINB",
    "CRCLB", "CRMB", "CRWDB", "CRDOB", "CYPHB", "DELLB", "DRAMB", "EWYB",
    "FLNCB", "FWDIB", "GEVB", "GLWB", "GOOGLB", "GPROB", "GSB", "HIMSB",
    "HOODB", "HPEB", "IBMB", "INTCB", "IRENB", "JPMB", "KSTRB", "LLYB",
    "MARVLB", "MDBB", "MRNAB", "MRVLB", "MSTRB", "MUB", "NBISB", "NFLXB",
    "NOKB", "NVDAB", "PDDB", "PENGB", "PYPLB", "QCOMB", "RDDTB", "RKLBB",
    "SECZB", "SHAZB", "SMCIB", "SNDKB", "SMHB", "SOXLB", "SOXSB", "SPCXB",
    "SPYB", "SQQQB", "STXB", "TSLAB", "TMFB", "TSMB", "USARB", "USDEB",
    "UVXYB", "VSTB", "WDCB", "WENB", "ZMB", "SKDDB", "SKUUB",
})

# Fiat-pegged or stable-index base assets to exclude from USDT-quoted scanning.
# USDT is intentionally NOT listed: it remains the quote asset for eligible pairs.
STABLECOIN_ASSETS = frozenset({
    "AEUR", "AUSD", "BIDR", "BFUSD", "BUSD", "CRVUSD", "DAI", "EURI",
    "EURC", "EURS", "EURT", "FDUSD", "FRAX", "GHO", "GUSD", "IDRT",
    "KGST", "LUSD", "MUSD", "OUSD", "PYUSD", "RLUSD", "SUSD", "TRYB",
    "TUSD", "U", "USDB", "USDD", "USDE", "USDF", "USD0", "USD1",
    "USDG", "USDP", "USDQ", "USDS", "USDTB", "USDX", "USDY", "USDC",
    "XUSD",
})
PENDING_TTL_MS = 30 * 60 * 1000

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


def effective_server_time_ms(server_time_ms, server_time_monotonic):
    """Advance the Binance server-time snapshot by elapsed monotonic time."""
    elapsed_ms = max(0, int((time.monotonic() - server_time_monotonic) * 1000))
    return server_time_ms + elapsed_ms


def is_stock_linked_symbol(symbol):
    """Return True for Binance-listed tokenized stock/security base assets."""
    base_asset = symbol[:-4] if isinstance(symbol, str) and symbol.endswith("USDT") else symbol
    return base_asset in STOCK_LINKED_ASSETS


def is_stablecoin_asset(base_asset):
    """Return True for known fiat-pegged base assets; USDT is deliberately allowed as quote."""
    return isinstance(base_asset, str) and base_asset in STABLECOIN_ASSETS and base_asset != "USDT"


def is_stablecoin_symbol(symbol):
    """Return True when a USDT pair's base asset is a known stablecoin."""
    base_asset = symbol[:-4] if isinstance(symbol, str) and symbol.endswith("USDT") else symbol
    return is_stablecoin_asset(base_asset)


def is_excluded_symbol(symbol):
    """Return True for tokenized stocks or stablecoin base assets."""
    return is_stock_linked_symbol(symbol) or is_stablecoin_symbol(symbol)


def fetch_universe():
    """Return active Binance Spot USDT crypto pairs, excluding bStocks and stablecoin bases."""
    exchange_info = get_json("/api/v3/exchangeInfo")
    entries = exchange_info.get("symbols") if isinstance(exchange_info, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError("Binance returned an invalid exchange-info response")

    symbols = []
    excluded_stock_pairs = 0
    excluded_stablecoin_pairs = 0
    for item in entries:
        if not isinstance(item, dict):
            continue
        if item.get("quoteAsset") != "USDT" or item.get("status") != "TRADING":
            continue
        if item.get("isSpotTradingAllowed") is False:
            continue

        symbol = item.get("symbol")
        base_asset = item.get("baseAsset")
        if not isinstance(base_asset, str) and isinstance(symbol, str) and symbol.endswith("USDT"):
            base_asset = symbol[:-4]
        if base_asset in STOCK_LINKED_ASSETS:
            excluded_stock_pairs += 1
            continue
        if is_stablecoin_asset(base_asset):
            excluded_stablecoin_pairs += 1
            continue
        if isinstance(symbol, str) and symbol:
            symbols.append(symbol)

    log.info("Excluded %s tokenized stock/security USDT pairs", excluded_stock_pairs)
    log.info("Excluded %s stablecoin-base USDT pairs (USDT quote retained)", excluded_stablecoin_pairs)
    if not symbols:
        raise RuntimeError("Binance returned no eligible active crypto USDT Spot pairs")
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
    if (
        open_time_ms < 0
        or open_time_ms % INTERVAL_MS != 0
        or close_time_ms != open_time_ms + INTERVAL_MS - 1
    ):
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


def fetch_new_closed_klines(
    symbol, server_time_ms, server_time_monotonic, last_processed_open_time_ms=None
):
    """Fetch only unseen, fully closed candles for this symbol.

    First-ever run: inspect just the latest completed candle and save it as T1
    (if it matches). Later runs: resume strictly after the saved candle cursor.
    """
    if last_processed_open_time_ms is None:
        params = {"symbol": symbol, "interval": INTERVAL, "limit": 2}
    else:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": int(last_processed_open_time_ms) + INTERVAL_MS,
            "limit": 1000,
        }

    klines = get_json("/api/v3/klines", params)
    if not isinstance(klines, list):
        raise ValueError(f"Binance returned a malformed kline response for {symbol}")

    effective_time_ms = effective_server_time_ms(server_time_ms, server_time_monotonic)
    closed_by_open_time = {}
    for kline in klines:
        try:
            open_time_ms = int(kline[0])
            close_time_ms = int(kline[6])
        except (IndexError, TypeError, ValueError):
            continue

        if close_time_ms >= effective_time_ms:
            continue
        if open_time_ms < 0 or open_time_ms % INTERVAL_MS != 0:
            continue
        if close_time_ms != open_time_ms + INTERVAL_MS - 1:
            continue
        if (
            last_processed_open_time_ms is not None
            and open_time_ms <= int(last_processed_open_time_ms)
        ):
            continue
        closed_by_open_time[open_time_ms] = kline

    closed = [closed_by_open_time[key] for key in sorted(closed_by_open_time)]
    if last_processed_open_time_ms is None:
        # Do not back-scan and instantly alert on an old TT at first startup.
        # The newest completed candle is T1; a future closed candle must be T2.
        closed = closed[-1:]
    return closed

def fmt_price(value):
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def format_alert(symbol, first, second):
    first_time = datetime.fromtimestamp(
        first["close_time_ms"] / 1000, tz=timezone.utc
    ).astimezone(IST)
    second_time = datetime.fromtimestamp(
        second["close_time_ms"] / 1000, tz=timezone.utc
    ).astimezone(IST)
    return (
        "🟢 <b>15m TT-CANDLE MATCH</b>\n\n"
        f"<b>Pair:</b> {html.escape(symbol)}\n"
        "<b>Two consecutive T candles detected</b>\n"
        "<b>Each candle:</b> Body 0% | Upper Wick 0% | Lower Wick 100%\n\n"
        f"<b>T1:</b> O/H/C {fmt_price(first['open'])} | L {fmt_price(first['low'])}\n"
        f"<b>T1 closed:</b> {first_time:%Y-%m-%d %H:%M:%S} IST\n"
        f"<b>T2:</b> O/H/C {fmt_price(second['open'])} | L {fmt_price(second['low'])}\n"
        f"<b>T2 closed:</b> {second_time:%Y-%m-%d %H:%M:%S} IST"
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
    """Load seen/pending alerts and per-symbol staged T-candle progress."""
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"seen": {}, "pending": {}, "tracking": {}}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Scanner state is unreadable or corrupt; refusing to send alerts until it is repaired"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("Scanner state has an invalid format; refusing to send duplicate alerts")
    if "seen" in data or "pending" in data or "tracking" in data:
        if not isinstance(data.get("seen"), dict) or not isinstance(data.get("pending"), dict):
            raise RuntimeError("Scanner state is incomplete or invalid; refusing to send duplicate alerts")
        tracking = data.get("tracking", {})
        if not isinstance(tracking, dict):
            raise RuntimeError("Scanner tracking state is invalid; refusing to skip or duplicate candles")
        return {
            "seen": data["seen"],
            "pending": data["pending"],
            "tracking": tracking,
        }
    # Backward compatibility with the original flat seen-signals JSON file.
    return {"seen": data, "pending": {}, "tracking": {}}

def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_file = STATE_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(state, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    temporary_file.replace(STATE_FILE)


def _serialize_t_candle(candle):
    return {
        "open": str(candle["open"]),
        "high": str(candle["high"]),
        "low": str(candle["low"]),
        "close": str(candle["close"]),
        "open_time_ms": candle["open_time_ms"],
        "close_time_ms": candle["close_time_ms"],
    }


def serialize_match(symbol, first, second):
    return {
        "symbol": symbol,
        "first": _serialize_t_candle(first),
        "second": _serialize_t_candle(second),
    }


def prune_pending_alerts(pending, server_time_ms):
    """Discard malformed, stock-linked, or stale queued alerts (keep 30 minutes max)."""
    cutoff_ms = server_time_ms - PENDING_TTL_MS
    kept = {}
    for key, record in pending.items():
        symbol = key.split("|", 1)[0] if isinstance(key, str) else ""
        if is_excluded_symbol(symbol) or not isinstance(record, dict):
            continue
        second = record.get("second")
        if not isinstance(second, dict):
            continue
        try:
            close_time_ms = int(second["close_time_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if cutoff_ms <= close_time_ms < server_time_ms:
            kept[key] = record
    return kept


def _deserialize_t_candle(record):
    if not isinstance(record, dict):
        raise ValueError("Invalid T-candle data")
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
    prices = (candle["open"], candle["high"], candle["low"], candle["close"])
    if not all(price.is_finite() for price in prices):
        raise ValueError("Pending candle contains a non-finite price")
    if (
        candle["open_time_ms"] < 0
        or candle["open_time_ms"] % INTERVAL_MS != 0
        or candle["close_time_ms"] != candle["open_time_ms"] + INTERVAL_MS - 1
    ):
        raise ValueError("Pending candle has invalid 15-minute timestamps")
    if not (candle["open"] == candle["high"] == candle["close"] and candle["low"] < candle["high"]):
        raise ValueError("Pending candle does not satisfy the exact T-candle rule")
    return candle


def deserialize_pending(record):
    if not isinstance(record, dict) or not isinstance(record.get("symbol"), str):
        raise ValueError("Invalid pending TT alert")
    if is_excluded_symbol(record["symbol"]):
        raise ValueError("Pending alert belongs to an excluded stock/stablecoin pair")
    first = _deserialize_t_candle(record["first"])
    second = _deserialize_t_candle(record["second"])
    if second["open_time_ms"] - first["open_time_ms"] != INTERVAL_MS:
        raise ValueError("Pending T candles are not consecutive 15-minute candles")
    return record["symbol"], first, second


def serialize_tracking(last_processed_open_time_ms, first_t):
    """Persist a symbol's candle cursor and, when present, its first T candle."""
    return {
        "last_processed_open_time_ms": last_processed_open_time_ms,
        "first_t": _serialize_t_candle(first_t) if first_t is not None else None,
    }


def deserialize_tracking(record):
    if record is None:
        return None, None
    if not isinstance(record, dict):
        raise ValueError("Invalid per-symbol tracking state")

    last_processed = record.get("last_processed_open_time_ms")
    if (
        isinstance(last_processed, bool)
        or not isinstance(last_processed, int)
        or last_processed < 0
        or last_processed % INTERVAL_MS != 0
    ):
        raise ValueError("Invalid last-processed candle timestamp")

    first_record = record.get("first_t")
    first_t = _deserialize_t_candle(first_record) if first_record is not None else None
    if first_t is not None and first_t["open_time_ms"] != last_processed:
        raise ValueError("Waiting T candle does not match the last-processed candle")
    return last_processed, first_t


def advance_t_sequence(symbol, closed_klines, tracking_record, server_time_ms):
    """Apply each new closed candle in order: T1 -> next candle T2 -> confirm TT.

    A non-T next candle resets the sequence. After a confirmed pair, both candles
    are consumed and the scanner waits for a fresh T1 (non-overlapping pairs).
    """
    last_processed, first_t = deserialize_tracking(tracking_record)
    matches = []

    for kline in sorted(closed_klines, key=lambda item: int(item[0])):
        open_time_ms = int(kline[0])
        if last_processed is not None and open_time_ms <= last_processed:
            continue

        # A missing 15-minute candle breaks consecutiveness.
        if (
            last_processed is not None
            and open_time_ms != last_processed + INTERVAL_MS
        ):
            first_t = None

        current_t = exact_t_candle(kline)
        if current_t is None:
            first_t = None
        else:
            if (
                first_t is not None
                and current_t["open_time_ms"] - first_t["open_time_ms"] == INTERVAL_MS
            ):
                # Do not send an old pair if cron was stopped for too long.
                if current_t["close_time_ms"] >= server_time_ms - PENDING_TTL_MS:
                    matches.append((symbol, first_t, current_t))
                # This T2 may also be T1 of the next adjacent TT pair.
                # The timestamp-based key still prevents the same pair repeating.
                first_t = current_t
            else:
                first_t = current_t

        last_processed = open_time_ms

    if last_processed is None:
        return tracking_record, matches
    return serialize_tracking(last_processed, first_t), matches


def scan_symbol(symbol, server_time_ms, server_time_monotonic, tracking_record=None):
    """Resume this symbol's staged T1/T2 confirmation from persistent state."""
    try:
        last_processed, _ = deserialize_tracking(tracking_record)
        closed_klines = fetch_new_closed_klines(
            symbol,
            server_time_ms,
            server_time_monotonic,
            last_processed,
        )
        sequence_now_ms = effective_server_time_ms(server_time_ms, server_time_monotonic)
        updated_tracking, matches = advance_t_sequence(
            symbol, closed_klines, tracking_record, sequence_now_ms
        )
        return symbol, updated_tracking, matches
    except (
        requests.RequestException,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        InvalidOperation,
    ) as exc:
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

    log.info(
        "Scanner started in staged TT mode: store T1, wait for the next closed 15m candle, "
        "then confirm only if T2 is also an exact T; stocks excluded"
    )
    state = load_state()
    server_time_ms = fetch_server_time_ms()
    server_time_monotonic = time.monotonic()
    symbols = fetch_universe()
    log.info("Binance Spot USDT pairs: %s", len(symbols))

    updated_tracking = dict(state["tracking"])
    hits = []
    scan_errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                scan_symbol,
                symbol,
                server_time_ms,
                server_time_monotonic,
                state["tracking"].get(symbol),
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                scan_errors += 1
                log.warning("Unexpected worker failure for %s: %s", symbol, type(exc).__name__)
                continue
            if not isinstance(result, tuple) or not result:
                scan_errors += 1
                continue
            if result[0] == "__SCAN_ERROR__":
                scan_errors += 1
                continue

            result_symbol, tracking_record, symbol_hits = result
            if tracking_record is not None:
                updated_tracking[result_symbol] = tracking_record
            hits.extend(symbol_hits)

    log.info(
        "Scan complete: %s eligible crypto pairs checked; %s errors; %s TT sequences confirmed",
        len(symbols),
        scan_errors,
        len(hits),
    )

    now = time.time()
    cutoff = now - SEEN_TTL_SEC
    old_seen = state["seen"]
    old_pending = state["pending"]
    old_tracking = state["tracking"]

    state_now_ms = effective_server_time_ms(server_time_ms, server_time_monotonic)
    state["pending"] = prune_pending_alerts(old_pending, state_now_ms)
    state["seen"] = {
        key: timestamp
        for key, timestamp in old_seen.items()
        if isinstance(timestamp, (int, float))
        and not isinstance(timestamp, bool)
        and math.isfinite(float(timestamp))
        and timestamp > cutoff
        and not is_excluded_symbol(key.split("|", 1)[0])
    }
    state["tracking"] = {
        symbol: record
        for symbol, record in updated_tracking.items()
        if not is_excluded_symbol(symbol)
    }
    state_changed = (
        state["seen"] != old_seen
        or state["pending"] != old_pending
        or state["tracking"] != old_tracking
    )

    for symbol, first, second in hits:
        key = f"{symbol}|TT|{first['open_time_ms']}|{second['open_time_ms']}"
        if key not in state["seen"] and key not in state["pending"]:
            state["pending"][key] = serialize_match(symbol, first, second)
            state_changed = True

    # Save T1/cursors and newly confirmed TT alerts before Telegram delivery.
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
            symbol, first, second = deserialize_pending(state["pending"][key])
            expected_key = f"{symbol}|TT|{first['open_time_ms']}|{second['open_time_ms']}"
            if key != expected_key:
                raise ValueError("Pending alert key does not match its candle data")
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            log.error("Removing invalid pending alert %s: %s", key, exc)
            del state["pending"][key]
            state_changed = True
            continue

        if telegram(format_alert(symbol, first, second)):
            state["seen"][key] = time.time()
            del state["pending"][key]
            sent_count += 1
            state_changed = True
            save_state(state)
            log.info("Telegram TT confirmation sent: %s", symbol)
        else:
            failed_alerts += 1
            log.error("Telegram alert failed and will retry on next cron run: %s", symbol)

    if state_changed:
        save_state(state)
    set_github_output("cache_changed", "true" if state_changed else "false")
    log.info(
        "Finished: %s TT confirmations sent; %s alert failures; %s pending",
        sent_count,
        failed_alerts,
        len(state["pending"]),
    )

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
