#!/usr/bin/env python3
"""Data collection tool."""

import fcntl
import html
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, time as clock_time, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

BINANCE = "https://data-api.binance.vision"
ONE_MINUTE_MS = 60_000
ONE_HOUR_MS = 60 * ONE_MINUTE_MS
WINDOW_CANDLES = 30
BASELINE_HOURS = 24
FETCH_1M_LIMIT = WINDOW_CANDLES + 1
FETCH_1H_LIMIT = BASELINE_HOURS + 2
MIN_VALID_CANDLES = 12
MAX_WORKERS = 12
REQUESTS_PER_SECOND = 8
PENDING_TTL_MS = 30 * 60 * 1000
SEEN_TTL_SECONDS = 7 * 24 * 60 * 60
SEEN_FUTURE_SKEW_SECONDS = 5 * 60
MAX_TIMESTAMP_MS = 253402300799999  # Last millisecond representable by datetime.
STATE_VERSION = 2

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / ".scanner_state"
STATE_FILE = STATE_DIR / "seen_signals.json"
LOCK_FILE = STATE_DIR / "scanner.lock"

IST = ZoneInfo("Asia/Kolkata")
SCAN_START_IST = clock_time(5, 30)
SCAN_END_IST = clock_time(18, 0)

MAX_VOL_PEAK_RATIO = Decimal("100")
MAX_VOLUME_ACCELERATION = Decimal("0.50")
MIN_TAKER_BUY_RATIO = Decimal("0.50")  # strict lower bound
MAX_TAKER_BUY_RATIO = Decimal("0.62")  # strict upper bound
MIN_LOWER_WICK_DENSITY = Decimal("0.35")
REQUIRED_G4H_COUNT = 0
MIN_QUOTE_VOLUME_PER_MINUTE = Decimal("1000")
MAX_TOKEN_PRICE_USDT = Decimal("3")  # Maximum latest close allowed for an alert

# Tokenized equities / securities listed on Binance Spot.
# Explicit list: do not blanket-exclude every ticker ending in B (e.g. BNB).
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

# Stablecoin / fiat-pegged base assets to exclude. USDT is intentionally absent:
# it remains the quote asset for normal crypto pairs.
STABLECOIN_ASSETS = frozenset({
    "AEUR", "AUSD", "BIDR", "BFUSD", "BUSD", "CRVUSD", "DAI", "EURI",
    "EURC", "EURS", "EURT", "FDUSD", "FRAX", "GHO", "GUSD", "IDRT",
    "KGST", "LUSD", "MUSD", "OUSD", "PYUSD", "RLUSD", "SUSD", "TRYB",
    "TUSD", "U", "USDB", "USDD", "USDE", "USDF", "USD0", "USD1",
    "USDG", "USDP", "USDQ", "USDS", "USDTB", "USDX", "USDY", "USDC",
    "XUSD",
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ignition-scanner")
_thread_local = threading.local()
_request_lock = threading.Lock()
_next_request_at = 0.0


def _pace_request():
    """Pace aggregate public Binance requests across all scanner threads."""
    global _next_request_at
    with _request_lock:
        now = time.monotonic()
        delay = max(0.0, _next_request_at - now)
        if delay:
            time.sleep(delay)
        _next_request_at = max(time.monotonic(), _next_request_at) + 1.0 / REQUESTS_PER_SECOND


def _session():
    """Create a retrying HTTP session per worker thread."""
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
        session.headers.update({"User-Agent": "ignition-scanner/4.0"})
        _thread_local.session = session
    return session


def get_json(path, params=None):
    _pace_request()
    response = _session().get(f"{BINANCE}{path}", params=params, timeout=(5, 20))
    response.raise_for_status()
    return response.json()


def as_integer(value, name):
    """Parse a supported epoch-millisecond integer without truncation or huge casts."""
    number = as_decimal(value, name)
    if number != number.to_integral_value():
        raise ValueError(f"Non-integer {name}")
    if number < 0 or number > MAX_TIMESTAMP_MS:
        raise ValueError(f"Out-of-range {name}")
    return int(number)


def fetch_server_time_ms():
    payload = get_json("/api/v3/time")
    if not isinstance(payload, dict) or "serverTime" not in payload:
        raise RuntimeError("Binance returned an invalid server-time payload")
    server_time_ms = as_integer(payload["serverTime"], "server time")
    if not (0 < server_time_ms <= MAX_TIMESTAMP_MS):
        raise RuntimeError("Binance returned an invalid server time")
    return server_time_ms


def effective_server_time_ms(server_time_ms, server_time_monotonic):
    elapsed_ms = max(0, int((time.monotonic() - server_time_monotonic) * 1000))
    return server_time_ms + elapsed_ms


def _valid_usdt_symbol(symbol):
    return (
        isinstance(symbol, str)
        and symbol.isascii()
        and symbol == symbol.upper()
        and symbol.endswith("USDT")
        and len(symbol) > 4
        and symbol[:-4].isalnum()
    )


def base_asset_from_symbol(symbol):
    return symbol[:-4] if _valid_usdt_symbol(symbol) else symbol


def is_excluded_symbol(symbol):
    base_asset = base_asset_from_symbol(symbol)
    return base_asset in STOCK_LINKED_ASSETS or base_asset in STABLECOIN_ASSETS


def fetch_universe():
    """Fetch current active Binance Spot USDT pairs, excluding stocks and stablecoins."""
    payload = get_json("/api/v3/exchangeInfo")
    entries = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError("Binance returned invalid exchange-info data")

    symbols = []
    excluded_stocks = 0
    excluded_stablecoins = 0
    for item in entries:
        if not isinstance(item, dict):
            continue
        if item.get("quoteAsset") != "USDT" or item.get("status") != "TRADING":
            continue
        if item.get("isSpotTradingAllowed") is not True:
            continue

        symbol = item.get("symbol")
        base_asset = item.get("baseAsset")
        if not _valid_usdt_symbol(symbol) or not isinstance(base_asset, str):
            continue
        # Exchange metadata must agree with the symbol suffix; fail closed on malformed rows.
        if (
            not base_asset.isascii()
            or not base_asset.isalnum()
            or base_asset != base_asset_from_symbol(symbol)
        ):
            continue

        if base_asset in STOCK_LINKED_ASSETS:
            excluded_stocks += 1
            continue
        if base_asset in STABLECOIN_ASSETS:
            excluded_stablecoins += 1
            continue
        symbols.append(symbol)

    symbols = sorted(set(symbols))
    log.info(
        "Eligible USDT Spot pairs: %s; excluded %s tokenized securities and %s stablecoin bases",
        len(symbols), excluded_stocks, excluded_stablecoins,
    )
    if not symbols:
        raise RuntimeError("Binance returned no eligible USDT Spot crypto pairs")
    return symbols


def as_decimal(value, name):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not number.is_finite():
        raise ValueError(f"Non-finite {name}")
    return number


def parse_kline(row, interval_ms):
    """Parse Binance kline fields and calculate candle morphology from real OHLC."""
    if not isinstance(row, (list, tuple)) or len(row) < 11:
        raise ValueError("Malformed Binance kline row")

    try:
        open_time_ms = as_integer(row[0], "kline open timestamp")
        close_time_ms = as_integer(row[6], "kline close timestamp")
    except (TypeError, ValueError, InvalidOperation, OverflowError) as exc:
        raise ValueError("Invalid kline timestamps") from exc

    if (
        open_time_ms < 0
        or open_time_ms % interval_ms != 0
        or close_time_ms != open_time_ms + interval_ms - 1
    ):
        raise ValueError("Misaligned kline timestamps")

    open_price = as_decimal(row[1], "open")
    high_price = as_decimal(row[2], "high")
    low_price = as_decimal(row[3], "low")
    close_price = as_decimal(row[4], "close")
    base_volume = as_decimal(row[5], "base volume")
    quote_volume = as_decimal(row[7], "quote volume")
    taker_buy_base = as_decimal(row[9], "taker-buy base volume")

    if min(open_price, high_price, low_price, close_price) <= 0:
        raise ValueError("Non-positive OHLC price")
    if high_price < max(open_price, close_price, low_price):
        raise ValueError("Inconsistent high price")
    if low_price > min(open_price, close_price, high_price):
        raise ValueError("Inconsistent low price")
    if min(base_volume, quote_volume, taker_buy_base) < 0:
        raise ValueError("Negative volume")
    if (base_volume == 0) != (quote_volume == 0):
        raise ValueError("Base and quote volumes disagree on zero trades")
    if base_volume == 0 and high_price != low_price:
        raise ValueError("Zero-volume candle has a non-zero price range")
    if taker_buy_base > base_volume:
        raise ValueError("Taker-buy base volume exceeds total base volume")

    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    candle_range = high_price - low_price
    valid = candle_range > 0
    body_ratio = abs(close_price - open_price) / candle_range if valid else Decimal("0")

    return {
        "open_time_ms": open_time_ms,
        "close_time_ms": close_time_ms,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "base_volume": base_volume,
        "quote_volume": quote_volume,
        "taker_buy_base": taker_buy_base,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "body_ratio": body_ratio,
        "valid": valid,
    }


def fetch_completed_1m_window(symbol, server_time_ms, server_time_monotonic):
    """Fetch exactly the latest 30 completed, continuous 1-minute bars."""
    rows = get_json(
        "/api/v3/klines",
        {"symbol": symbol, "interval": "1m", "limit": FETCH_1M_LIMIT},
    )
    if not isinstance(rows, list):
        raise ValueError(f"Malformed 1m kline response for {symbol}")

    now_ms = effective_server_time_ms(server_time_ms, server_time_monotonic)
    completed = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 11:
            raise ValueError(f"Malformed 1m kline row for {symbol}")
        try:
            close_ms = as_integer(row[6], "1m close timestamp")
        except (TypeError, ValueError, IndexError, InvalidOperation, OverflowError) as exc:
            raise ValueError(f"Invalid 1m close time for {symbol}") from exc
        if close_ms >= now_ms:
            continue
        bar = parse_kline(row, ONE_MINUTE_MS)
        if bar["open_time_ms"] in completed:
            log.warning("Duplicate 1m candle timestamp for %s; skipping incomplete window", symbol)
            return None
        completed[bar["open_time_ms"]] = bar

    bars = [completed[key] for key in sorted(completed)]
    if len(bars) < WINDOW_CANDLES:
        return None
    window = bars[-WINDOW_CANDLES:]
    for previous, current in zip(window, window[1:]):
        if current["open_time_ms"] - previous["open_time_ms"] != ONE_MINUTE_MS:
            return None
    return window


def fetch_prior_24h_quote_baseline(symbol, window_start_ms):
    """Average quote volume/minute over 24 completed hourly candles before the scan window.

    The report does not define its VolPeak baseline. We use the prior 24 complete
    hours (ending before this 30m window) so max 1m volume / baseline can exceed
    100 and the stated threshold is meaningful.
    """
    rows = get_json(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": "1h",
            "endTime": int(window_start_ms) - 1,
            "limit": FETCH_1H_LIMIT,
        },
    )
    if not isinstance(rows, list):
        raise ValueError(f"Malformed 1h baseline response for {symbol}")

    completed = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 8:
            raise ValueError(f"Malformed 1h baseline row for {symbol}")
        try:
            open_ms = as_integer(row[0], "1h open timestamp")
            close_ms = as_integer(row[6], "1h close timestamp")
        except (TypeError, ValueError, IndexError, InvalidOperation, OverflowError) as exc:
            raise ValueError(f"Invalid 1h baseline timestamp for {symbol}") from exc
        if close_ms >= window_start_ms:
            continue
        if open_ms < 0 or open_ms % ONE_HOUR_MS != 0:
            raise ValueError(f"Misaligned 1h baseline timestamp for {symbol}")
        if close_ms != open_ms + ONE_HOUR_MS - 1:
            raise ValueError(f"Invalid 1h baseline close time for {symbol}")
        quote_volume = as_decimal(row[7], "1h quote volume")
        if quote_volume < 0:
            raise ValueError("Negative 1h quote volume")
        if open_ms in completed:
            log.warning("Duplicate 1h candle timestamp for %s; skipping incomplete baseline", symbol)
            return None
        completed[open_ms] = quote_volume

    hours = sorted(completed.items())[-BASELINE_HOURS:]
    if len(hours) != BASELINE_HOURS:
        return None
    for (previous_ms, _), (current_ms, _) in zip(hours, hours[1:]):
        if current_ms - previous_ms != ONE_HOUR_MS:
            return None

    total_quote_volume = sum((volume for _, volume in hours), Decimal("0"))
    baseline_per_minute = total_quote_volume / Decimal(BASELINE_HOURS * 60)
    return baseline_per_minute if baseline_per_minute > 0 else None


def compute_features(window, prior_24h_quote_per_minute):
    """Calculate the report's metrics plus the minimum liquidity floor.

    - VolPeak = max 1m USDT quote volume / prior-24h average USDT volume per minute.
    - VolAccel = last 5m average quote volume / preceding 25m average quote volume.
    - TakerBuy = total taker-buy base volume / total base volume across 30m.
    - Lower-wick density = L-only candles / valid (non-zero-range) candles.
    - G4H = green, >75%-body, >2x-30m-average-volume candle (wick suffix is ignored).
    """
    if not isinstance(window, list) or len(window) != WINDOW_CANDLES:
        return None
    baseline = as_decimal(prior_24h_quote_per_minute, "prior-24h quote volume baseline")
    if baseline <= 0:
        return None

    bars = sorted(window, key=lambda item: item["open_time_ms"])
    for previous, current in zip(bars, bars[1:]):
        if current["open_time_ms"] - previous["open_time_ms"] != ONE_MINUTE_MS:
            return None

    valid = [bar for bar in bars if bar["valid"]]
    if len(valid) < MIN_VALID_CANDLES:
        return None

    quote_total = sum((bar["quote_volume"] for bar in bars), Decimal("0"))
    base_total = sum((bar["base_volume"] for bar in bars), Decimal("0"))
    taker_buy_total = sum((bar["taker_buy_base"] for bar in bars), Decimal("0"))
    if quote_total <= 0 or base_total <= 0:
        return None

    quote_per_minute = quote_total / Decimal(WINDOW_CANDLES)
    prior_25_average = sum((bar["quote_volume"] for bar in bars[:25]), Decimal("0")) / Decimal("25")
    last_5_average = sum((bar["quote_volume"] for bar in bars[25:]), Decimal("0")) / Decimal("5")
    if prior_25_average <= 0:
        return None

    vol_peak_ratio = max(bar["quote_volume"] for bar in bars) / baseline
    volume_acceleration = last_5_average / prior_25_average
    taker_buy_ratio = taker_buy_total / base_total

    lower_only_count = sum(
        1 for bar in valid
        if bar["lower_wick"] > 0 and bar["upper_wick"] == 0
    )
    lower_wick_density = Decimal(lower_only_count) / Decimal(len(valid))

    g4h_count = sum(
        1 for bar in valid
        if (
            bar["close"] > bar["open"]
            and bar["body_ratio"] > Decimal("0.75")
            and bar["quote_volume"] > Decimal("2") * quote_per_minute
        )
    )

    return {
        "window_start_ms": int(bars[0]["open_time_ms"]),
        "window_end_ms": int(bars[-1]["close_time_ms"]),
        "last_close": str(bars[-1]["close"]),
        "prior_24h_quote_per_minute": str(baseline),
        "quote_volume_per_minute": str(quote_per_minute),
        "vol_peak_ratio": str(vol_peak_ratio),
        "volume_acceleration": str(volume_acceleration),
        "taker_buy_ratio": str(taker_buy_ratio),
        "lower_wick_density": str(lower_wick_density),
        "g4h_count": int(g4h_count),
        "valid_candles": int(len(valid)),
        "invalid_candles": int(WINDOW_CANDLES - len(valid)),
    }


def passes_pump_filter(features):
    """Pass only when every stated research threshold is satisfied."""
    if not isinstance(features, dict):
        return False
    if any(
        type(features.get(field)) is not int
        for field in (
            "window_start_ms", "window_end_ms", "g4h_count",
            "valid_candles", "invalid_candles",
        )
    ):
        return False
    try:
        vol_peak = as_decimal(features["vol_peak_ratio"], "vol_peak_ratio")
        volume_acceleration = as_decimal(features["volume_acceleration"], "volume_acceleration")
        taker_buy = as_decimal(features["taker_buy_ratio"], "taker_buy_ratio")
        lower_density = as_decimal(features["lower_wick_density"], "lower_wick_density")
        quote_per_minute = as_decimal(features["quote_volume_per_minute"], "quote_volume_per_minute")
        g4h_count = int(features["g4h_count"])
        valid_count = int(features["valid_candles"])
        last_close = as_decimal(features["last_close"], "last_close")
        prior_baseline = as_decimal(
            features["prior_24h_quote_per_minute"], "prior_24h_quote_per_minute"
        )
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
        return False

    metrics = (
        vol_peak, volume_acceleration, taker_buy, lower_density,
        quote_per_minute, last_close, prior_baseline,
    )
    if not all(metric.is_finite() for metric in metrics):
        return False
    if g4h_count < 0 or valid_count < 0:
        return False

    # Reject malformed persisted or externally supplied metrics before applying thresholds.
    try:
        start_ms = int(features["window_start_ms"])
        end_ms = int(features["window_end_ms"])
        invalid_count = int(features["invalid_candles"])
        if (
            isinstance(features["window_start_ms"], bool)
            or isinstance(features["window_end_ms"], bool)
            or Decimal(str(features["window_start_ms"])) != Decimal(start_ms)
            or Decimal(str(features["window_end_ms"])) != Decimal(end_ms)
            or Decimal(str(features["g4h_count"])) != Decimal(g4h_count)
            or Decimal(str(features["valid_candles"])) != Decimal(valid_count)
            or Decimal(str(features["invalid_candles"])) != Decimal(invalid_count)
        ):
            return False
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
        return False

    if (
        start_ms < 0
        or end_ms > MAX_TIMESTAMP_MS
        or start_ms % ONE_MINUTE_MS != 0
        or end_ms - start_ms != WINDOW_CANDLES * ONE_MINUTE_MS - 1
        or last_close <= 0
        or prior_baseline <= 0
    ):
        return False
    if not (MIN_VALID_CANDLES <= valid_count <= WINDOW_CANDLES):
        return False
    if invalid_count != WINDOW_CANDLES - valid_count or g4h_count < 0:
        return False
    if (
        vol_peak < 0
        or volume_acceleration < 0
        or not (Decimal("0") <= taker_buy <= Decimal("1"))
        or not (Decimal("0") <= lower_density <= Decimal("1"))
        or quote_per_minute < 0
    ):
        return False

    return (
        vol_peak < MAX_VOL_PEAK_RATIO
        and volume_acceleration < MAX_VOLUME_ACCELERATION
        and MIN_TAKER_BUY_RATIO < taker_buy < MAX_TAKER_BUY_RATIO
        and lower_density >= MIN_LOWER_WICK_DENSITY
        and g4h_count == REQUIRED_G4H_COUNT
        and quote_per_minute >= MIN_QUOTE_VOLUME_PER_MINUTE
        and last_close <= MAX_TOKEN_PRICE_USDT
    )


def datetime_from_ms(timestamp_ms):
    """Convert epoch milliseconds without a float round-trip."""
    timestamp_ms = as_integer(timestamp_ms, "timestamp")
    whole_seconds, milliseconds = divmod(timestamp_ms, 1000)
    return datetime.fromtimestamp(
        whole_seconds, tz=timezone.utc
    ).replace(microsecond=milliseconds * 1000)


def in_scan_window(timestamp_ms):
    local_time = datetime_from_ms(timestamp_ms).astimezone(IST).time()
    return SCAN_START_IST <= local_time < SCAN_END_IST


def ist_date_for(timestamp_ms):
    return datetime_from_ms(timestamp_ms).astimezone(IST).date().isoformat()


def signal_key(symbol, signal_time_ms):
    return f"{symbol}|{ist_date_for(signal_time_ms)}"


def prune_seen(seen, now_seconds=None):
    """Discard malformed, stale, or implausibly future-dated deduplication entries."""
    if not isinstance(seen, dict):
        return {}
    now = Decimal(str(time.time() if now_seconds is None else now_seconds))
    cutoff = now - Decimal(SEEN_TTL_SECONDS)
    future_limit = now + Decimal(SEEN_FUTURE_SKEW_SECONDS)
    kept = {}
    for key, sent_at in seen.items():
        if not isinstance(key, str) or not isinstance(sent_at, (int, float)) or isinstance(sent_at, bool):
            continue
        symbol, separator, day = key.partition("|")
        if (
            not separator
            or "|" in day
            or not _valid_usdt_symbol(symbol)
            or is_excluded_symbol(symbol)
        ):
            continue
        try:
            if datetime.strptime(day, "%Y-%m-%d").date().isoformat() != day:
                continue
            timestamp = Decimal(str(sent_at))
        except (ValueError, TypeError, InvalidOperation):
            continue
        if timestamp.is_finite() and cutoff <= timestamp <= future_limit:
            kept[key] = sent_at
    return kept


def fmt_decimal(value, places=3):
    return f"{Decimal(str(value)):.{places}f}"


def format_alert(symbol, features, signal_time_ms):
    signal_time_ms = as_integer(signal_time_ms, "alert signal timestamp")
    window_start_ms = as_integer(features["window_start_ms"], "alert window start")
    window_end_ms = as_integer(features["window_end_ms"], "alert window end")
    signal_time = datetime_from_ms(signal_time_ms).astimezone(IST)
    start_time = datetime_from_ms(window_start_ms).astimezone(IST)
    end_time = datetime_from_ms(window_end_ms + 1).astimezone(IST)
    return (
        "🟡 <b>MATCH</b>\n\n"
        f"<b>Pair:</b> {html.escape(symbol)}\n"
        f"<b>Signal time:</b> {signal_time:%Y-%m-%d %H:%M:%S} IST\n"
        f"<b>30m window:</b> {start_time:%H:%M}–{end_time:%H:%M} IST\n\n"
        f"<b>VolPeak / prior 24h avg:</b> {fmt_decimal(features['vol_peak_ratio'])}x (limit &lt;100)\n"
        f"<b>Volume acceleration:</b> {fmt_decimal(features['volume_acceleration'])}x (limit &lt;0.50)\n"
        f"<b>Taker-buy ratio:</b> {fmt_decimal(Decimal(features['taker_buy_ratio']) * 100, 2)}% (strictly 50–62%)\n"
        f"<b>Lower-only wick density:</b> {fmt_decimal(Decimal(features['lower_wick_density']) * 100, 1)}% (limit ≥35%)\n"
        f"<b>G4H candles:</b> {features['g4h_count']} (must be 0)\n"
        f"<b>Average quote volume:</b> {fmt_decimal(features['quote_volume_per_minute'], 0)} USDT/min (minimum 1,000)\n"
        f"<b>Valid candles:</b> {features['valid_candles']}/30 (minimum 12)\n"
        f"<b>Last close:</b> {html.escape(str(features['last_close']))} USDT"
    )


def telegram(message):
    """Send Telegram message with bounded retry and no token leakage in logs."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("Telegram is not configured: set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID")
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
                retry_after = parameters.get("retry_after", 2) if isinstance(parameters, dict) else 2
                try:
                    if isinstance(retry_after, bool):
                        raise ValueError("Boolean retry_after")
                    delay_number = as_decimal(retry_after, "Telegram retry_after")
                    if delay_number != delay_number.to_integral_value() or delay_number < 0:
                        raise ValueError("Invalid retry_after")
                except (TypeError, ValueError, InvalidOperation, OverflowError):
                    log.warning("Telegram returned an invalid retry_after value; deferring candidate")
                    return False
                if attempt < 2:
                    if delay_number > 30:
                        log.warning(
                            "Telegram requested a %s-second cooldown; deferring candidate to the next scan",
                            delay_number,
                        )
                        return False
                    time.sleep(max(int(delay_number), 1))
                    continue
                return False

            if response.status_code >= 500:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return False

            return response.status_code == 200 and result.get("ok") is True
        except requests.RequestException as exc:
            log.warning("Telegram request failed (attempt %s/3): %s", attempt + 1, type(exc).__name__)
            if attempt < 2:
                time.sleep(2 ** attempt)
    return False


def initial_state():
    return {"version": STATE_VERSION, "seen": {}, "pending": {}}


def load_state():
    """Load v2 pre-pump state; deliberately discard old TT pending alerts."""
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return initial_state()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Scanner state is unreadable/corrupt; refusing to send until repaired"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("Scanner state has invalid format")
    if data.get("version") != STATE_VERSION:
        log.warning(
            "Migrating old TT scanner state: clearing legacy TT seen/pending data "
            "so no stale TT alert can be sent by the new pre-pump scanner"
        )
        return initial_state()
    if not isinstance(data.get("seen"), dict) or not isinstance(data.get("pending"), dict):
        raise RuntimeError("Pre-pump scanner state is incomplete; refusing to send alerts")
    return {
        "version": STATE_VERSION,
        "seen": data["seen"],
        "pending": data["pending"],
    }


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_file = STATE_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(state, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    temporary_file.replace(STATE_FILE)


def serialize_candidate(symbol, features):
    return {
        "symbol": symbol,
        "features": features,
        "signal_time_ms": int(features["window_end_ms"]),
    }


def deserialize_candidate(record):
    if not isinstance(record, dict) or not isinstance(record.get("symbol"), str):
        raise ValueError("Invalid pending pre-pump candidate")
    symbol = record["symbol"]
    if not _valid_usdt_symbol(symbol):
        raise ValueError("Pending candidate has an invalid Binance symbol")
    if is_excluded_symbol(symbol):
        raise ValueError("Candidate belongs to an excluded stock/stablecoin pair")
    features = record.get("features")
    if not passes_pump_filter(features):
        raise ValueError("Pending candidate no longer satisfies the pre-pump filter")
    raw_signal_time = record["signal_time_ms"]
    if type(raw_signal_time) is not int or not (0 <= raw_signal_time <= MAX_TIMESTAMP_MS):
        raise ValueError("Pending signal timestamp is not a supported integer")
    signal_time_ms = raw_signal_time
    if signal_time_ms != int(features["window_end_ms"]):
        raise ValueError("Pending signal timestamp does not match feature window")
    if not in_scan_window(signal_time_ms):
        raise ValueError("Pending signal timestamp is outside the scan window")
    return symbol, features, signal_time_ms


def prune_pending(pending, now_ms):
    cutoff = now_ms - PENDING_TTL_MS
    kept = {}
    for key, record in pending.items():
        try:
            symbol, features, signal_time_ms = deserialize_candidate(record)
            if key != signal_key(symbol, signal_time_ms):
                continue
            if cutoff <= signal_time_ms < now_ms:
                kept[key] = record
        except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
            continue
    return kept


def scan_symbol(symbol, server_time_ms, server_time_monotonic):
    """Fetch live 1m OHLCV and pre-window 24h volume baseline for one pair."""
    try:
        window = fetch_completed_1m_window(symbol, server_time_ms, server_time_monotonic)
        if window is None:
            return None
        baseline = fetch_prior_24h_quote_baseline(symbol, window[0]["open_time_ms"])
        if baseline is None:
            return None
        features = compute_features(window, baseline)
        if features is None or not passes_pump_filter(features):
            return None
        signal_time_ms = features["window_end_ms"]
        if not in_scan_window(signal_time_ms):
            return None
        return serialize_candidate(symbol, features)
    except (
        requests.RequestException, KeyError, IndexError, TypeError,
        ValueError, InvalidOperation, OverflowError,
    ) as exc:
        log.warning("Could not analyse %s: %s", symbol, type(exc).__name__)
        return {"_scan_error": symbol}


def set_github_output(name, value):
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")


@contextmanager
def scanner_lock():
    """Prevent overlapping system-cron or manually dispatched runs on Linux."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_scan():
    server_time_ms = fetch_server_time_ms()
    server_time_monotonic = time.monotonic()
    now_ms = effective_server_time_ms(server_time_ms, server_time_monotonic)

    if not in_scan_window(now_ms):
        local_now = datetime_from_ms(now_ms).astimezone(IST)
        log.info(
            "Outside the research scan window (%s–%s IST); no market scan or alerts",
            SCAN_START_IST.strftime("%H:%M"),
            SCAN_END_IST.strftime("%H:%M"),
        )
        log.info("Current time: %s IST", local_now.strftime("%Y-%m-%d %H:%M:%S"))
        return

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variable")

    state = load_state()
    symbols = fetch_universe()
    log.info(
        "Starting live pre-pump scan: %s pairs; 30 closed 1m candles; prior 24h baseline",
        len(symbols),
    )

    candidates = []
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                scan_symbol, symbol, server_time_ms, server_time_monotonic
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                errors += 1
                log.warning("Unexpected analysis error for %s: %s", symbol, type(exc).__name__)
                continue
            if isinstance(result, dict) and result.get("_scan_error"):
                errors += 1
            elif result is not None:
                candidates.append(result)

    log.info(
        "Market analysis complete: %s pairs; %s data errors; %s filter matches",
        len(symbols), errors, len(candidates),
    )

    current_now_ms = effective_server_time_ms(server_time_ms, server_time_monotonic)
    previous_seen = state["seen"]
    previous_pending = state["pending"]
    state["seen"] = prune_seen(previous_seen, now_seconds=current_now_ms / 1000.0)
    state["pending"] = prune_pending(previous_pending, current_now_ms)
    changed = state["seen"] != previous_seen or state["pending"] != previous_pending

    for candidate in candidates:
        key = signal_key(candidate["symbol"], candidate["signal_time_ms"])
        if key not in state["seen"] and key not in state["pending"]:
            state["pending"][key] = candidate
            changed = True

    if changed:
        save_state(state)

    sent_count = 0
    failed_count = 0
    for key in sorted(list(state["pending"])):
        if key in state["seen"]:
            del state["pending"][key]
            changed = True
            continue
        try:
            symbol, features, signal_time_ms = deserialize_candidate(state["pending"][key])
            if key != signal_key(symbol, signal_time_ms):
                raise ValueError("Pending candidate key mismatch")
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            log.error("Dropping invalid pending candidate %s: %s", key, type(exc).__name__)
            del state["pending"][key]
            changed = True
            continue

        if telegram(format_alert(symbol, features, signal_time_ms)):
            sent_at_ms = effective_server_time_ms(server_time_ms, server_time_monotonic)
            state["seen"][key] = sent_at_ms / 1000.0
            del state["pending"][key]
            sent_count += 1
            changed = True
            save_state(state)
            log.info("Pre-pump candidate alert sent: %s", symbol)
        else:
            failed_count += 1
            log.error("Telegram send failed; candidate will retry until stale: %s", symbol)

    if changed:
        save_state(state)
    set_github_output("cache_changed", "true" if changed else "false")
    log.info(
        "Finished: %s candidate alerts sent; %s failures; %s pending",
        sent_count, failed_count, len(state["pending"]),
    )


def main():
    with scanner_lock() as acquired:
        if not acquired:
            log.warning("Another scanner process is active; exiting")
            return
        run_scan()


if __name__ == "__main__":
    main()
