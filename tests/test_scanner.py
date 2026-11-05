import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import scanner


def make_kline(
    minute,
    open_price="100",
    high_price="100",
    low_price="99",
    close_price="100",
    base_volume="100",
    quote_volume="1200",
    taker_buy_base="55",
    interval_ms=scanner.ONE_MINUTE_MS,
):
    open_ms = minute * interval_ms
    return [
        open_ms,
        str(open_price),
        str(high_price),
        str(low_price),
        str(close_price),
        str(base_volume),
        open_ms + interval_ms - 1,
        str(quote_volume),
        10,
        str(taker_buy_base),
        "0",
        "0",
    ]


def make_passing_window():
    rows = []
    for minute in range(scanner.WINDOW_CANDLES):
        quote_volume = "1200" if minute < 25 else "480"
        if minute < 12:
            rows.append(make_kline(minute, "100", "100", "99", "100",
                                   "100", quote_volume, "55"))
        else:
            rows.append(make_kline(minute, "100", "101", "100", "101",
                                   "100", quote_volume, "55"))
    return [scanner.parse_kline(row, scanner.ONE_MINUTE_MS) for row in rows]


def make_features(**changes):
    features = {
        "window_start_ms": 0,
        "window_end_ms": 30 * scanner.ONE_MINUTE_MS - 1,
        "last_close": "100",
        "prior_24h_quote_per_minute": "100",
        "quote_volume_per_minute": "1080",
        "vol_peak_ratio": "12",
        "volume_acceleration": "0.4",
        "taker_buy_ratio": "0.55",
        "lower_wick_density": "0.4",
        "g4h_count": 0,
        "valid_candles": 20,
        "invalid_candles": 10,
    }
    features.update(changes)
    return features


class KlineParsingTests(unittest.TestCase):
    def test_parses_binance_field_positions(self):
        bar = scanner.parse_kline(make_kline(0), scanner.ONE_MINUTE_MS)
        self.assertEqual(bar["quote_volume"], Decimal("1200"))
        self.assertEqual(bar["taker_buy_base"], Decimal("55"))
        self.assertEqual(bar["base_volume"], Decimal("100"))
        self.assertTrue(bar["valid"])

    def test_zero_range_is_invalid_for_wicks_not_parse_error(self):
        row = make_kline(0, "100", "100", "100", "100", "0", "0", "0")
        bar = scanner.parse_kline(row, scanner.ONE_MINUTE_MS)
        self.assertFalse(bar["valid"])
        self.assertEqual(bar["quote_volume"], Decimal("0"))

    def test_rejects_bad_timestamp(self):
        row = make_kline(0)
        row[6] += 1
        with self.assertRaises(ValueError):
            scanner.parse_kline(row, scanner.ONE_MINUTE_MS)

    def test_rejects_fractional_exchange_timestamps(self):
        fractional_open = make_kline(1)
        fractional_open[0] = str(Decimal(fractional_open[0]) + Decimal("0.5"))
        with self.assertRaises(ValueError):
            scanner.parse_kline(fractional_open, scanner.ONE_MINUTE_MS)

        fractional_close = make_kline(1)
        fractional_close[6] = str(Decimal(fractional_close[6]) + Decimal("0.5"))
        with self.assertRaises(ValueError):
            scanner.parse_kline(fractional_close, scanner.ONE_MINUTE_MS)

    def test_rejects_inconsistent_ohlc(self):
        with self.assertRaises(ValueError):
            scanner.parse_kline(make_kline(0, "100", "99", "98", "100"),
                                scanner.ONE_MINUTE_MS)

    def test_rejects_taker_volume_above_total(self):
        with self.assertRaises(ValueError):
            scanner.parse_kline(
                make_kline(0, base_volume="10", taker_buy_base="11"),
                scanner.ONE_MINUTE_MS,
            )

    def test_rejects_inconsistent_zero_base_and_quote_volume(self):
        with self.assertRaises(ValueError):
            scanner.parse_kline(
                make_kline(0, base_volume="0", quote_volume="1200", taker_buy_base="0"),
                scanner.ONE_MINUTE_MS,
            )
        with self.assertRaises(ValueError):
            scanner.parse_kline(
                make_kline(0, base_volume="100", quote_volume="0", taker_buy_base="55"),
                scanner.ONE_MINUTE_MS,
            )

    def test_rejects_zero_volume_candle_with_price_range(self):
        with self.assertRaises(ValueError):
            scanner.parse_kline(
                make_kline(0, "100", "101", "99", "100", "0", "0", "0"),
                scanner.ONE_MINUTE_MS,
            )


class FeatureCalculationTests(unittest.TestCase):
    def test_calculates_report_features_from_30_bars(self):
        features = scanner.compute_features(make_passing_window(), Decimal("100"))
        self.assertIsNotNone(features)
        self.assertEqual(Decimal(features["volume_acceleration"]), Decimal("0.4"))
        self.assertEqual(Decimal(features["taker_buy_ratio"]), Decimal("0.55"))
        self.assertEqual(Decimal(features["lower_wick_density"]), Decimal("0.4"))
        self.assertEqual(Decimal(features["quote_volume_per_minute"]), Decimal("1080"))
        self.assertEqual(features["valid_candles"], 30)

    def test_peak_volume_uses_independent_prior_24h_baseline(self):
        features = scanner.compute_features(make_passing_window(), Decimal("10"))
        self.assertEqual(Decimal(features["vol_peak_ratio"]), Decimal("120"))

    def test_insufficient_valid_candles_is_skipped(self):
        window = make_passing_window()
        for index in range(11, 30):
            row = make_kline(index, "100", "100", "100", "100", "0", "0", "0")
            window[index] = scanner.parse_kline(row, scanner.ONE_MINUTE_MS)
        self.assertIsNone(scanner.compute_features(window, Decimal("100")))

    def test_gap_in_minute_sequence_is_skipped(self):
        window = make_passing_window()
        window[-1]["open_time_ms"] += scanner.ONE_MINUTE_MS
        self.assertIsNone(scanner.compute_features(window, Decimal("100")))

    def test_zero_baseline_is_skipped(self):
        self.assertIsNone(scanner.compute_features(make_passing_window(), Decimal("0")))


class FilterThresholdTests(unittest.TestCase):
    def test_g4h0_prefix_is_counted_even_without_a_wick(self):
        window = make_passing_window()
        row = make_kline(12, "100", "103", "100", "103", "100", "5000", "55")
        window[12] = scanner.parse_kline(row, scanner.ONE_MINUTE_MS)
        features = scanner.compute_features(window, Decimal("100"))
        self.assertEqual(features["g4h_count"], 1)
        self.assertFalse(scanner.passes_pump_filter(features))

    def test_all_thresholds_pass(self):
        self.assertTrue(scanner.passes_pump_filter(make_features()))

    def test_peak_volume_at_100_is_rejected(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(vol_peak_ratio="100")))

    def test_volume_acceleration_at_half_is_rejected(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(volume_acceleration="0.5")))

    def test_taker_buy_lower_boundary_is_rejected(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(taker_buy_ratio="0.50")))

    def test_taker_buy_upper_boundary_is_rejected(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(taker_buy_ratio="0.62")))

    def test_lower_wick_density_below_35_is_rejected(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(lower_wick_density="0.3499")))

    def test_any_g4h_rejection_candle_fails(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(g4h_count=1)))

    def test_quote_volume_below_1000_fails(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(quote_volume_per_minute="999.99")))

    def test_fewer_than_12_valid_candles_fails(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(valid_candles=11)))

    def test_nonfinite_feature_fails_closed(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(vol_peak_ratio="NaN")))

    def test_negative_metrics_fail_closed(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(vol_peak_ratio="-1")))
        self.assertFalse(scanner.passes_pump_filter(make_features(volume_acceleration="-0.1")))

    def test_wick_density_above_one_fails_closed(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(lower_wick_density="1.01")))

    def test_fractional_counts_fail_closed(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(g4h_count="0.5")))
        self.assertFalse(scanner.passes_pump_filter(make_features(valid_candles="20.5")))

    def test_inconsistent_window_duration_fails_closed(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(window_end_ms=1)))

    def test_missing_alert_fields_fail_closed(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(last_close=None)))
        self.assertFalse(scanner.passes_pump_filter(make_features(prior_24h_quote_per_minute="0")))

    def test_string_timestamps_in_persisted_features_fail_closed(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(window_start_ms="0")))
        self.assertFalse(scanner.passes_pump_filter(make_features(window_end_ms=str(30 * scanner.ONE_MINUTE_MS - 1))))

    def test_overflowing_counts_fail_closed(self):
        self.assertFalse(scanner.passes_pump_filter(make_features(g4h_count=float("inf"))))


class UniverseTests(unittest.TestCase):
    def test_stablecoin_and_stock_assets_are_excluded(self):
        for symbol in ("UUSDT", "USDEUSDT", "USDSUSDT", "USDCUSDT", "BFUSDUSDT", "KGSTUSDT"):
            self.assertTrue(scanner.is_excluded_symbol(symbol), symbol)
        self.assertTrue(scanner.is_excluded_symbol("GPROBUSDT"))
        self.assertFalse(scanner.is_excluded_symbol("BTCUSDT"))
        self.assertFalse(scanner.is_excluded_symbol("BNBUSDT"))

    def test_usdt_quote_is_allowed(self):
        self.assertFalse(scanner.is_excluded_symbol("BTCUSDT"))
        self.assertNotIn("USDT", scanner.STABLECOIN_ASSETS)

    def test_universe_uses_current_exchange_info(self):
        payload = {"symbols": [
            {"symbol":"BTCUSDT","baseAsset":"BTC","quoteAsset":"USDT","status":"TRADING","isSpotTradingAllowed":True},
            {"symbol":"BNBUSDT","baseAsset":"BNB","quoteAsset":"USDT","status":"TRADING","isSpotTradingAllowed":True},
            {"symbol":"UUSDT","baseAsset":"U","quoteAsset":"USDT","status":"TRADING","isSpotTradingAllowed":True},
            {"symbol":"USDEUSDT","baseAsset":"USDE","quoteAsset":"USDT","status":"TRADING","isSpotTradingAllowed":True},
            {"symbol":"GPROBUSDT","baseAsset":"GPROB","quoteAsset":"USDT","status":"TRADING","isSpotTradingAllowed":True},
            {"symbol":"OLDUSDT","baseAsset":"OLD","quoteAsset":"USDT","status":"BREAK","isSpotTradingAllowed":True},
            {"symbol":"MISSINGFLAGUSDT","baseAsset":"MISSINGFLAG","quoteAsset":"USDT","status":"TRADING"},
            {"symbol":"MISSINGBASEUSDT","quoteAsset":"USDT","status":"TRADING","isSpotTradingAllowed":True},
        ]}
        with patch.object(scanner, "get_json", return_value=payload):
            self.assertEqual(scanner.fetch_universe(), ["BNBUSDT", "BTCUSDT"])


class BaselineAndWindowFetchTests(unittest.TestCase):
    def test_server_time_rejects_fractional_timestamp(self):
        with patch.object(scanner, "get_json", return_value={"serverTime": "1700000000000.5"}):
            with self.assertRaises(ValueError):
                scanner.fetch_server_time_ms()

    def test_server_time_rejects_huge_timestamp_before_integer_cast(self):
        with patch.object(scanner, "get_json", return_value={"serverTime": "1e1000000000"}):
            with self.assertRaises(ValueError):
                scanner.fetch_server_time_ms()

    def test_baseline_uses_24_closed_hours_before_window(self):
        start = 25 * scanner.ONE_HOUR_MS
        rows = []
        for index in range(24):
            open_ms = (index + 1) * scanner.ONE_HOUR_MS
            rows.append([
                open_ms, "100", "101", "99", "100", "10",
                open_ms + scanner.ONE_HOUR_MS - 1, "6000",
                100, "5", "30000", "0",
            ])
        with patch.object(scanner, "get_json", return_value=rows) as get:
            baseline = scanner.fetch_prior_24h_quote_baseline("BTCUSDT", start)
        self.assertEqual(baseline, Decimal("100"))
        self.assertEqual(get.call_args.args[1]["endTime"], start - 1)
        self.assertEqual(get.call_args.args[1]["limit"], scanner.FETCH_1H_LIMIT)

    def test_duplicate_hourly_candle_makes_baseline_unusable(self):
        start = 25 * scanner.ONE_HOUR_MS
        rows = []
        for index in range(24):
            open_ms = (index + 1) * scanner.ONE_HOUR_MS
            rows.append([
                open_ms, "100", "101", "99", "100", "10",
                open_ms + scanner.ONE_HOUR_MS - 1, "6000",
                100, "5", "30000", "0",
            ])
        rows.append(rows[-1])
        with patch.object(scanner, "get_json", return_value=rows):
            self.assertIsNone(scanner.fetch_prior_24h_quote_baseline("BTCUSDT", start))

    def test_incomplete_hourly_baseline_is_skipped(self):
        with patch.object(scanner, "get_json", return_value=[]):
            self.assertIsNone(
                scanner.fetch_prior_24h_quote_baseline("BTCUSDT", 25 * scanner.ONE_HOUR_MS)
            )

    def test_only_last_30_closed_minutes_returned(self):
        rows = [make_kline(index) for index in range(31)]
        now = 31 * scanner.ONE_MINUTE_MS + 100
        with patch.object(scanner, "get_json", return_value=rows):
            with patch.object(scanner.time, "monotonic", return_value=10.0):
                bars = scanner.fetch_completed_1m_window("BTCUSDT", now, 10.0)
        self.assertEqual(len(bars), 30)
        self.assertEqual(bars[0]["open_time_ms"], scanner.ONE_MINUTE_MS)
        self.assertEqual(bars[-1]["open_time_ms"], 30 * scanner.ONE_MINUTE_MS)

    def test_missing_minute_makes_window_unusable(self):
        rows = [make_kline(index) for index in range(31) if index != 15]
        now = 31 * scanner.ONE_MINUTE_MS + 100
        with patch.object(scanner, "get_json", return_value=rows):
            with patch.object(scanner.time, "monotonic", return_value=10.0):
                self.assertIsNone(scanner.fetch_completed_1m_window("BTCUSDT", now, 10.0))

    def test_duplicate_minute_makes_window_unusable(self):
        rows = [make_kline(index) for index in range(31)]
        rows.append(rows[-1])
        now = 31 * scanner.ONE_MINUTE_MS + 100
        with patch.object(scanner, "get_json", return_value=rows):
            with patch.object(scanner.time, "monotonic", return_value=10.0):
                self.assertIsNone(scanner.fetch_completed_1m_window("BTCUSDT", now, 10.0))


class ScheduleAndStateTests(unittest.TestCase):
    def test_millisecond_datetime_conversion_is_exact(self):
        stamp = int(datetime(2026, 11, 5, 0, 0, tzinfo=timezone.utc).timestamp()) * 1000 - 1
        self.assertEqual(
            scanner.datetime_from_ms(stamp),
            datetime(2026, 11, 4, 23, 59, 59, 999000, tzinfo=timezone.utc),
        )
        self.assertEqual(
            scanner.datetime_from_ms(scanner.MAX_TIMESTAMP_MS),
            datetime(9999, 12, 31, 23, 59, 59, 999000, tzinfo=timezone.utc),
        )

    def test_research_window_is_0530_to_1800_ist(self):
        before_start = int(datetime(2026, 11, 4, 23, 59, tzinfo=timezone.utc).timestamp() * 1000)
        start = int(datetime(2026, 11, 5, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        before_end = int(datetime(2026, 11, 5, 12, 29, tzinfo=timezone.utc).timestamp() * 1000)
        end = int(datetime(2026, 11, 5, 12, 30, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertFalse(scanner.in_scan_window(before_start))
        self.assertTrue(scanner.in_scan_window(start))
        self.assertTrue(scanner.in_scan_window(before_end))
        self.assertFalse(scanner.in_scan_window(end))

    def test_once_per_symbol_per_ist_day(self):
        stamp = int(datetime(2026, 11, 5, 1, 0, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertEqual(scanner.signal_key("BTCUSDT", stamp), "BTCUSDT|2026-11-05")

    def test_legacy_tt_state_is_migrated_without_old_pending_alerts(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_text(
                '{"seen":{"BTCUSDT|TT|1|2":123},"pending":{"old TT":{}},"tracking":{}}',
                encoding="utf-8",
            )
            with patch.object(scanner, "STATE_FILE", state_file):
                state = scanner.load_state()
        self.assertEqual(state, {"version": scanner.STATE_VERSION, "seen": {}, "pending": {}})

    def test_corrupt_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_text("{bad json", encoding="utf-8")
            with patch.object(scanner, "STATE_FILE", state_file):
                with self.assertRaises(RuntimeError):
                    scanner.load_state()

    def test_non_utf8_state_fails_closed_with_runtime_error(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_bytes(b"\\xff\\xfe")
            with patch.object(scanner, "STATE_FILE", state_file):
                with self.assertRaises(RuntimeError):
                    scanner.load_state()

    def test_pending_candidate_rejects_failed_filter(self):
        candidate = scanner.serialize_candidate("BTCUSDT", make_features(g4h_count=1))
        with self.assertRaises(ValueError):
            scanner.deserialize_candidate(candidate)

    def test_pending_candidate_rejects_malformed_symbol(self):
        candidate = scanner.serialize_candidate("BTC/USDT", make_features())
        with self.assertRaises(ValueError):
            scanner.deserialize_candidate(candidate)

    def test_pending_candidate_rejects_overflow_timestamp(self):
        candidate = scanner.serialize_candidate("BTCUSDT", make_features())
        candidate["signal_time_ms"] = float("inf")
        with self.assertRaises(ValueError):
            scanner.deserialize_candidate(candidate)

    def test_old_pending_candidate_expires(self):
        old_time = 1_000_000
        features = make_features(
            window_end_ms=old_time,
            window_start_ms=old_time - 30 * scanner.ONE_MINUTE_MS,
        )
        record = scanner.serialize_candidate("BTCUSDT", features)
        key = scanner.signal_key("BTCUSDT", old_time)
        self.assertEqual(
            scanner.prune_pending({key: record}, old_time + scanner.PENDING_TTL_MS + 1),
            {},
        )

    def test_seen_state_pruning_ignores_malformed_entries(self):
        now = 2_000_000
        valid_key = "BTCUSDT|1970-01-01"
        malformed = {
            valid_key: now,
            "BTCUSDT|not-a-date": now,
            "BTC/USDT|1970-01-01": now,
            "UUSDT|1970-01-01": now,
            "BTCUSDT|1970-01-01|extra": now,
            "XRPUSDT|1970-01-01": 10 ** 10000,
            "ETHUSDT|1970-01-01": float("nan"),
        }
        cleaned = scanner.prune_seen(malformed, now_seconds=now)
        self.assertEqual(cleaned, {valid_key: now})


if __name__ == "__main__":
    unittest.main()
