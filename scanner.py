import os
import time
import html
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

BINANCE = "https://data-api.binance.vision"

# ===== SYSTEM RULES =====
PRICE_CAP     = 2.0
BARS          = 400
BODY_MIN      = 70.0
VOL_X_MIN     = 2.5
TAKER_MIN     = 0.55
BREAK_MAX_PCT = 2.5
RET_4H_MAX    = 4.0
RNG_12H_MIN   = 3.0

BTC_4H_MIN    = 0.3
BTC_24H_KILL  = -2.0

TP_PCT        = 3.0
SL_PCT        = -3.0
MAX_WORKERS   = 16
# =========================

STABLE = {"XUSD","USDE","RLUSD","U","USDC","USDT","DAI","TUSD",
          "FDUSD","BUSD","PYUSD","USDP","EUR","USD1","AEUR","EURI"}


def ist_now():
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)


def get(url, params=None, retries=3):
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            time.sleep(1)
    return None


def telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram env missing")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=15,
        )
        return r.status_code == 200 and r.json().get("ok", False)
    except Exception as exc:
        print(f"Telegram error: {exc}")
        return False


def fmt_price(p):
    p = float(p)
    if p >= 1:      return f"${p:.4f}"
    if p >= 0.01:   return f"${p:.5f}"
    if p >= 0.0001: return f"${p:.6f}"
    return f"${p:.8f}".rstrip("0").rstrip(".")


def fetch_universe():
    ex = get(f"{BINANCE}/api/v3/exchangeInfo")
    tk = get(f"{BINANCE}/api/v3/ticker/price")
    tk24 = get(f"{BINANCE}/api/v3/ticker/24hr")
    if not ex or not tk or not tk24:
        raise RuntimeError("Binance universe fetch failed")

    px = {x["symbol"]: float(x["price"]) for x in tk}
    vol = {x["symbol"]: float(x["quoteVolume"]) for x in tk24}

    syms = []
    for s in ex["symbols"]:
        sym = s["symbol"]
        if not sym.endswith("USDT"): continue
        if s.get("status") != "TRADING": continue
        if s.get("isSpotTradingAllowed") is False: continue
        base = sym.removesuffix("USDT")
        if base in STABLE: continue
        p = px.get(sym, 0)
        if not (0 < p < PRICE_CAP): continue
        if vol.get(sym, 0) < 100000: continue
        syms.append(sym)
    return sorted(syms)


def btc_regime():
    k = get(f"{BINANCE}/api/v3/klines",
            {"symbol": "BTCUSDT", "interval": "15m", "limit": 100})
    if not k or len(k) < 97:
        return None
    try:
        c = [float(x[4]) for x in k]
        return {
            "1h":  (c[-1] / c[-5]  - 1) * 100,
            "4h":  (c[-1] / c[-17] - 1) * 100,
            "24h": (c[-1] / c[-97] - 1) * 100,
        }
    except (ValueError, IndexError):
        return None


def fetch_klines(sym):
    k = get(f"{BINANCE}/api/v3/klines",
            {"symbol": sym, "interval": "15m", "limit": BARS})
    if not k or len(k) < 100:
        return None
    try:
        o  = np.array([float(x[1]) for x in k], dtype=float)
        h  = np.array([float(x[2]) for x in k], dtype=float)
        l  = np.array([float(x[3]) for x in k], dtype=float)
        c  = np.array([float(x[4]) for x in k], dtype=float)
        v  = np.array([float(x[5]) for x in k], dtype=float)
        tb = np.array([float(x[9]) for x in k], dtype=float)
        if int(k[-1][6]) >= time.time() * 1000:
            o = o[:-1]; h = h[:-1]; l = l[:-1]
            c = c[:-1]; v = v[:-1]; tb = tb[:-1]
        return {"o": o, "h": h, "l": l, "c": c, "v": v, "tb": tb}
    except (ValueError, IndexError):
        return None


def ema(values, period):
    out = np.zeros_like(values)
    k = 2 / (period + 1)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = values[i] * k + out[i-1] * (1 - k)
    return out


def check_signal(d):
    o, h, l, c, v, tb = d["o"], d["h"], d["l"], d["c"], d["v"], d["tb"]
    n = len(c)
    if n < 60: return None

    i = n - 1
    resistance = h[max(0, i-48):i].max()
    if resistance <= 0: return None
    if c[i] <= resistance: return None

    break_pct = (c[i] / resistance - 1) * 100
    if break_pct > BREAK_MAX_PCT: return None

    rng = h[i] - l[i]
    if rng <= 0: return None
    body = abs(c[i] - o[i]) / rng * 100
    if body < BODY_MIN: return None
    if c[i] < o[i]: return None

    v_avg = v[max(0, i-8):i].mean()
    if v_avg <= 0: return None
    vol_x = v[i] / v_avg
    if vol_x < VOL_X_MIN: return None

    if v[i] <= 0: return None
    taker = tb[i] / v[i]
    if taker < TAKER_MIN: return None

    e20 = ema(c, 20)
    if c[i] <= e20[i]: return None

    if i >= 16:
        ret_4h = (c[i] / c[i-16] - 1) * 100
        if ret_4h > RET_4H_MAX: return None
    else:
        return None

    rng12 = (h[max(0, i-48):i].max() - l[max(0, i-48):i].min()) / l[max(0, i-48):i].min() * 100
    if rng12 < RNG_12H_MIN: return None

    return {
        "entry": float(c[i]),
        "break_pct": break_pct,
        "body": body,
        "vol_x": vol_x,
        "taker": taker,
    }


def format_alert(sym, sig, btc):
    entry = sig["entry"]
    sl = entry * (1 + SL_PCT / 100)
    tp = entry * (1 + TP_PCT / 100)
    name = sym.removesuffix("USDT")
    return (
        f"🟢 <b>ENTRY</b>\n\n"
        f"<b>{html.escape(name)}</b>\n"
        f"<b>ENTRY:</b> {fmt_price(entry)}\n"
        f"<b>BREAK:</b> +{sig['break_pct']:.2f}%\n"
        f"<b>BODY:</b> {sig['body']:.0f}%  <b>VOL:</b> {sig['vol_x']:.1f}x  <b>TAKER:</b> {sig['taker']:.2f}\n"
        f"🛑 <b>SL:</b> {fmt_price(sl)} ({SL_PCT:+.0f}%)\n"
        f"🎯 <b>TP:</b> {fmt_price(tp)} ({TP_PCT:+.0f}%)\n\n"
        f"<i>BTC 4h {btc['4h']:+.2f}%  |  IST {ist_now().strftime('%H:%M')}</i>"
    )


def main():
    print("=" * 60)
    print("SCANNER STARTING — IST", ist_now().strftime("%Y-%m-%d %H:%M"))
    print("=" * 60)

    btc = btc_regime()
    if btc is None:
        print("BTC fetch failed — abort")
        return
    print(f"BTC 1h {btc['1h']:+.2f}%  4h {btc['4h']:+.2f}%  24h {btc['24h']:+.2f}%")

    if btc["24h"] < BTC_24H_KILL:
        print(f"KILL SWITCH: BTC 24h {btc['24h']:+.2f}% < {BTC_24H_KILL}% — no signals")
        return
    if btc["4h"] < BTC_4H_MIN:
        print(f"BTC 4h {btc['4h']:+.2f}% < {BTC_4H_MIN}% — no signals")
        return

    try:
        syms = fetch_universe()
    except Exception as exc:
        print(f"Universe error: {exc}")
        return
    print(f"Universe: {len(syms)} pairs < ${PRICE_CAP}")

    hits = []
    def worker(sym):
        d = fetch_klines(sym)
        if d is None: return None
        sig = check_signal(d)
        if sig is None: return None
        return (sym, sig)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(worker, s) for s in syms]
        for f in as_completed(futs):
            r = f.result()
            if r: hits.append(r)

    print(f"Signals found: {len(hits)}")

    hits.sort(key=lambda x: x[1]["vol_x"], reverse=True)
    for sym, sig in hits:
        msg = format_alert(sym, sig, btc)
        ok = telegram(msg)
        print(f"  {sym}: {'sent' if ok else 'FAILED'}  "
              f"break {sig['break_pct']:+.2f}%  vol {sig['vol_x']:.1f}x")

    print("Done.")


if __name__ == "__main__":
    main()
