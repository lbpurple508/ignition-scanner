import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import scanner


def make_kline(open_ms, open_price, high_price, low_price, close_price):
    return [
        open_ms,
        str(open_price),
        str(high_price),
        str(low_price),
        str(close_price),
        "100",
        open_ms + scanner.INTERVAL_MS - 1,
        "1000",
        10,
        "50",
        "500",
        "0",
    ]


def make_t(open_ms, price="100", low="95"):
    return make_kline(open_ms, price, price, low, price)


class ExactTDefinitionTests(unittest.TestCase):
    def test_exact_t_matches(self):
        candle = scanner.exact_t_candle(make_t(0))
        self.assertIsNotNone(candle)
        self.assertEqual(candle["body_pct"], Decimal("0"))
        self.assertEqual(candle["upper_wick_pct"], Decimal("0"))
        self.assertEqual(candle["lower_wick_pct"], Decimal("100"))

    def test_body_must_be_zero(self):
        self.assertIsNone(scanner.exact_t_candle(make_kline(0, "99", "100", "95", "100")))

    def test_upper_wick_must_be_zero(self):
        self.assertIsNone(scanner.exact_t_candle(make_kline(0, "100", "101", "95", "100")))

    def test_lower_wick_must_be_nonzero(self):
        self.assertIsNone(scanner.exact_t_candle(make_kline(0, "100", "100", "100", "100")))

    def test_nonfinite_values_are_rejected(self):
        self.assertIsNone(scanner.exact_t_candle(make_kline(0, "NaN", "NaN", "1", "NaN")))


class StagedTTSequenceTests(unittest.TestCase):
    def setUp(self):
        self.t1 = make_t(0, "100", "95")
        self.t2 = make_t(scanner.INTERVAL_MS, "98", "90")
        self.t3 = make_t(2 * scanner.INTERVAL_MS, "97", "89")
        self.t4 = make_t(3 * scanner.INTERVAL_MS, "96", "88")
        self.non_t = make_kline(
            scanner.INTERVAL_MS, "98", "99", "90", "98"
        )
        self.now = self.t2[6] + 1000

    def test_first_t_is_saved_and_does_not_alert(self):
        tracking, matches = scanner.advance_t_sequence(
            "BTCUSDT", [self.t1], None, self.now
        )
        self.assertEqual(matches, [])
        last_processed, first_t = scanner.deserialize_tracking(tracking)
        self.assertEqual(last_processed, 0)
        self.assertEqual(first_t["open_time_ms"], 0)

    def test_second_consecutive_t_confirms_once(self):
        initial = scanner.serialize_tracking(0, scanner.exact_t_candle(self.t1))
        tracking, matches = scanner.advance_t_sequence(
            "BTCUSDT", [self.t2], initial, self.now
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], "BTCUSDT")
        self.assertEqual(matches[0][1]["open_time_ms"], 0)
        self.assertEqual(matches[0][2]["open_time_ms"], scanner.INTERVAL_MS)
        last_processed, first_t = scanner.deserialize_tracking(tracking)
        self.assertEqual(last_processed, scanner.INTERVAL_MS)
        self.assertIsNone(first_t)

    def test_non_t_second_candle_resets_sequence(self):
        initial = scanner.serialize_tracking(0, scanner.exact_t_candle(self.t1))
        tracking, matches = scanner.advance_t_sequence(
            "BTCUSDT", [self.non_t], initial, self.now
        )
        self.assertEqual(matches, [])
        last_processed, first_t = scanner.deserialize_tracking(tracking)
        self.assertEqual(last_processed, scanner.INTERVAL_MS)
        self.assertIsNone(first_t)

    def test_t_after_reset_becomes_new_first_t(self):
        initial = scanner.serialize_tracking(0, scanner.exact_t_candle(self.t1))
        tracking, matches = scanner.advance_t_sequence(
            "BTCUSDT", [self.non_t, self.t3], initial, self.now
        )
        self.assertEqual(matches, [])
        last_processed, first_t = scanner.deserialize_tracking(tracking)
        self.assertEqual(last_processed, 2 * scanner.INTERVAL_MS)
        self.assertEqual(first_t["open_time_ms"], 2 * scanner.INTERVAL_MS)

    def test_next_t_after_reset_confirms_new_pair(self):
        initial = scanner.serialize_tracking(
            2 * scanner.INTERVAL_MS,
            scanner.exact_t_candle(self.t3),
        )
        tracking, matches = scanner.advance_t_sequence(
            "BTCUSDT", [self.t4], initial, self.t4[6] + 1000
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][1]["open_time_ms"], 2 * scanner.INTERVAL_MS)
        self.assertEqual(matches[0][2]["open_time_ms"], 3 * scanner.INTERVAL_MS)
        self.assertIsNone(scanner.deserialize_tracking(tracking)[1])

    def test_gap_breaks_sequence_but_current_t_can_start_again(self):
        gap_t = make_t(2 * scanner.INTERVAL_MS, "97", "89")
        initial = scanner.serialize_tracking(0, scanner.exact_t_candle(self.t1))
        tracking, matches = scanner.advance_t_sequence(
            "BTCUSDT", [gap_t], initial, self.now
        )
        self.assertEqual(matches, [])
        self.assertEqual(
            scanner.deserialize_tracking(tracking)[1]["open_time_ms"],
            2 * scanner.INTERVAL_MS,
        )

    def test_reprocessing_same_candle_does_not_confirm_or_repeat(self):
        initial = scanner.serialize_tracking(0, scanner.exact_t_candle(self.t1))
        tracking, matches = scanner.advance_t_sequence(
            "BTCUSDT", [self.t1], initial, self.now
        )
        self.assertEqual(matches, [])
        self.assertEqual(tracking, initial)

    def test_pairs_are_non_overlapping_after_confirmation(self):
        initial = scanner.serialize_tracking(0, scanner.exact_t_candle(self.t1))
        with patch.object(scanner, "PENDING_TTL_MS", 3 * scanner.INTERVAL_MS):
            tracking, matches = scanner.advance_t_sequence(
                "BTCUSDT", [self.t2, self.t3, self.t4], initial, self.t4[6] + 1000
            )
        self.assertEqual(len(matches), 2)
        self.assertEqual(
            [(m[1]["open_time_ms"], m[2]["open_time_ms"]) for m in matches],
            [
                (0, scanner.INTERVAL_MS),
                (2 * scanner.INTERVAL_MS, 3 * scanner.INTERVAL_MS),
            ],
        )
        self.assertIsNone(scanner.deserialize_tracking(tracking)[1])

    def test_stale_second_t_is_not_alerted(self):
        initial = scanner.serialize_tracking(0, scanner.exact_t_candle(self.t1))
        late_now = self.t2[6] + scanner.PENDING_TTL_MS + 1
        tracking, matches = scanner.advance_t_sequence(
            "BTCUSDT", [self.t2], initial, late_now
        )
        self.assertEqual(matches, [])
        self.assertIsNone(scanner.deserialize_tracking(tracking)[1])

    def test_tracking_round_trip(self):
        first = scanner.exact_t_candle(self.t1)
        value = scanner.serialize_tracking(0, first)
        last_processed, restored_first = scanner.deserialize_tracking(value)
        self.assertEqual(last_processed, 0)
        self.assertEqual(restored_first["open"], Decimal("100"))

    def test_invalid_tracking_is_rejected(self):
        with self.assertRaises(ValueError):
            scanner.deserialize_tracking({"last_processed_open_time_ms": "bad", "first_t": None})


class CandleFetchTests(unittest.TestCase):
    def setUp(self):
        self.t1 = make_t(0)
        self.t2 = make_t(scanner.INTERVAL_MS)
        self.t3 = make_t(2 * scanner.INTERVAL_MS)

    def test_first_run_only_bootstraps_latest_closed_candle(self):
        now = 2 * scanner.INTERVAL_MS + 100
        with patch.object(scanner, "get_json", return_value=[self.t2, self.t3]) as get:
            with patch.object(scanner.time, "monotonic", return_value=50.0):
                candles = scanner.fetch_new_closed_klines("BTCUSDT", now, 50.0, None)
        self.assertEqual(candles, [self.t2])
        self.assertEqual(get.call_args.args[1]["limit"], 2)

    def test_later_run_fetches_only_candles_after_saved_cursor(self):
        now = 3 * scanner.INTERVAL_MS + 100
        with patch.object(scanner, "get_json", return_value=[self.t2, self.t3]) as get:
            with patch.object(scanner.time, "monotonic", return_value=50.0):
                candles = scanner.fetch_new_closed_klines(
                    "BTCUSDT", now, 50.0, 0
                )
        self.assertEqual(candles, [self.t2, self.t3])
        self.assertEqual(get.call_args.args[1]["startTime"], scanner.INTERVAL_MS)

    def test_server_clock_advances_across_candle_close_boundary(self):
        now = 2 * scanner.INTERVAL_MS - 10
        with patch.object(scanner, "get_json", return_value=[self.t1, self.t2]):
            with patch.object(scanner.time, "monotonic", return_value=50.020):
                candles = scanner.fetch_new_closed_klines(
                    "BTCUSDT", now, 50.0, 0
                )
        self.assertEqual(candles, [self.t2])

    def test_incomplete_candle_is_never_processed(self):
        now = 2 * scanner.INTERVAL_MS - 10
        with patch.object(scanner, "get_json", return_value=[self.t2]):
            with patch.object(scanner.time, "monotonic", return_value=50.0):
                candles = scanner.fetch_new_closed_klines(
                    "BTCUSDT", now, 50.0, 0
                )
        self.assertEqual(candles, [])


class AssetUniverseTests(unittest.TestCase):
    def test_stocks_excluded_but_crypto_assets_ending_in_b_remain(self):
        self.assertTrue(scanner.is_stock_linked_symbol("GPROBUSDT"))
        self.assertTrue(scanner.is_stock_linked_symbol("TSLABUSDT"))
        self.assertFalse(scanner.is_stock_linked_symbol("BNBUSDT"))
        self.assertFalse(scanner.is_stock_linked_symbol("BTCUSDT"))

    def test_stablecoin_bases_are_excluded_but_usdt_quote_is_kept(self):
        for symbol in ("UUSDT", "USDEUSDT", "USDSUSDT", "USDCUSDT", "FDUSDUSDT"):
            self.assertTrue(scanner.is_stablecoin_symbol(symbol), symbol)
        self.assertFalse(scanner.is_stablecoin_asset("USDT"))
        self.assertFalse(scanner.is_stablecoin_symbol("BTCUSDT"))
        self.assertFalse(scanner.is_stablecoin_symbol("BNBUSDT"))

    def test_universe_keeps_active_crypto_and_excludes_stock_and_stablecoin_tokens(self):
        payload = {
            "symbols": [
                {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "BNBUSDT", "baseAsset": "BNB", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "GPROBUSDT", "baseAsset": "GPROB", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "UUSDT", "baseAsset": "U", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "USDEUSDT", "baseAsset": "USDE", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "USDSUSDT", "baseAsset": "USDS", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "USDTUSDT", "baseAsset": "USDT", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "ETHBTC", "baseAsset": "ETH", "quoteAsset": "BTC", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "OLDUSDT", "baseAsset": "OLD", "quoteAsset": "USDT", "status": "BREAK", "isSpotTradingAllowed": True},
            ]
        }
        with patch.object(scanner, "get_json", return_value=payload):
            self.assertEqual(scanner.fetch_universe(), ["BNBUSDT", "BTCUSDT"])


class AlertStateTests(unittest.TestCase):
    def setUp(self):
        self.first = scanner.exact_t_candle(make_t(0))
        self.second = scanner.exact_t_candle(make_t(scanner.INTERVAL_MS))

    def test_pending_state_round_trip(self):
        record = scanner.serialize_match("BTCUSDT", self.first, self.second)
        symbol, first, second = scanner.deserialize_pending(record)
        self.assertEqual(symbol, "BTCUSDT")
        self.assertEqual(first["open_time_ms"], 0)
        self.assertEqual(second["open_time_ms"], scanner.INTERVAL_MS)

    def test_pending_state_rejects_non_t_candle(self):
        record = scanner.serialize_match("BTCUSDT", self.first, self.second)
        record["second"]["high"] = "99"
        with self.assertRaises(ValueError):
            scanner.deserialize_pending(record)

    def test_pending_state_rejects_stablecoin_base(self):
        record = scanner.serialize_match("USDEUSDT", self.first, self.second)
        with self.assertRaises(ValueError):
            scanner.deserialize_pending(record)

    def test_stale_and_stock_pending_alerts_are_removed(self):
        now_ms = 10 * scanner.INTERVAL_MS
        recent = scanner.serialize_match("BTCUSDT", self.first, self.second)
        recent["second"]["close_time_ms"] = now_ms - 60_000
        stale = scanner.serialize_match("ETHUSDT", self.first, self.second)
        stale["second"]["close_time_ms"] = now_ms - scanner.PENDING_TTL_MS - 1
        stock = scanner.serialize_match("GPROBUSDT", self.first, self.second)
        stock["second"]["close_time_ms"] = now_ms - 60_000
        stable = scanner.serialize_match("USDEUSDT", self.first, self.second)
        stable["second"]["close_time_ms"] = now_ms - 60_000
        kept = scanner.prune_pending_alerts(
            {
                "BTCUSDT|TT|0|900000": recent,
                "ETHUSDT|TT|0|900000": stale,
                "GPROBUSDT|TT|0|900000": stock,
                "USDEUSDT|TT|0|900000": stable,
            },
            now_ms,
        )
        self.assertEqual(list(kept), ["BTCUSDT|TT|0|900000"])

    def test_corrupt_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_text("{invalid json", encoding="utf-8")
            with patch.object(scanner, "STATE_FILE", state_file):
                with self.assertRaises(RuntimeError):
                    scanner.load_state()

    def test_legacy_seen_file_migrates_with_empty_tracking(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_text(json.dumps({"BTCUSDT|123": 456.0}), encoding="utf-8")
            with patch.object(scanner, "STATE_FILE", state_file):
                state = scanner.load_state()
        self.assertIn("BTCUSDT|123", state["seen"])
        self.assertEqual(state["pending"], {})
        self.assertEqual(state["tracking"], {})


if __name__ == "__main__":
    unittest.main()
