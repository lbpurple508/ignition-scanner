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


class ExactTDefinitionTests(unittest.TestCase):
    def test_exact_t_matches(self):
        candle = scanner.exact_t_candle(make_kline(0, "100", "100", "95", "100"))
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


class LatestTTLogicTests(unittest.TestCase):
    def setUp(self):
        self.t1 = make_kline(0, "100", "100", "95", "100")
        self.t2 = make_kline(scanner.INTERVAL_MS, "98", "98", "90", "98")
        self.non_t = make_kline(2 * scanner.INTERVAL_MS, "97", "98", "90", "98")
        self.t3 = make_kline(2 * scanner.INTERVAL_MS, "97", "97", "90", "97")

    def test_fetch_uses_only_latest_two_closed_candles(self):
        now = 2 * scanner.INTERVAL_MS + 100
        with patch.object(scanner, "get_json", return_value=[self.t1, self.t2, self.t3]):
            with patch.object(scanner.time, "monotonic", return_value=50.0):
                pair = scanner.fetch_latest_two_closed_klines("BTCUSDT", now, 50.0)
        self.assertEqual(pair, (self.t2, self.t3))

    def test_server_clock_advances_during_scan_boundary(self):
        # At initial server time T2 has not quite closed. It closes while this
        # request is in flight; monotonic elapsed time must account for that.
        now = 2 * scanner.INTERVAL_MS - 10
        with patch.object(scanner, "get_json", return_value=[self.t1, self.t2, self.t3]):
            with patch.object(scanner.time, "monotonic", return_value=50.020):
                pair = scanner.fetch_latest_two_closed_klines("BTCUSDT", now, 50.0)
        self.assertEqual(pair, (self.t1, self.t2))

    def test_no_pair_when_two_latest_candles_are_not_consecutive(self):
        gap = make_kline(3 * scanner.INTERVAL_MS, "98", "98", "90", "98")
        now = 4 * scanner.INTERVAL_MS + 100
        with patch.object(scanner, "get_json", return_value=[self.t1, self.t2, gap]):
            with patch.object(scanner.time, "monotonic", return_value=5.0):
                self.assertIsNone(scanner.fetch_latest_two_closed_klines("BTCUSDT", now, 5.0))

    def test_only_latest_pair_is_considered_not_older_tt_pair(self):
        # An older pair is TT, but the latest pair contains a non-T candle.
        old_t = make_kline(0, "100", "100", "95", "100")
        old_t2 = make_kline(scanner.INTERVAL_MS, "98", "98", "90", "98")
        latest_non_t = make_kline(2 * scanner.INTERVAL_MS, "97", "98", "90", "97")
        with patch.object(
            scanner,
            "fetch_latest_two_closed_klines",
            return_value=(old_t2, latest_non_t),
        ):
            result = scanner.scan_symbol("BTCUSDT", 3 * scanner.INTERVAL_MS, 100.0)
        self.assertIsNone(result)

    def test_both_latest_candles_must_match_for_one_signal(self):
        with patch.object(scanner, "fetch_latest_two_closed_klines", return_value=(self.t1, self.t2)):
            result = scanner.scan_symbol("BTCUSDT", 2 * scanner.INTERVAL_MS + 100, 10.0)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "BTCUSDT")
        self.assertEqual(result[2]["open_time_ms"] - result[1]["open_time_ms"], scanner.INTERVAL_MS)

    def test_non_t_second_candle_rejects_signal(self):
        with patch.object(scanner, "fetch_latest_two_closed_klines", return_value=(self.t1, self.non_t)):
            result = scanner.scan_symbol("BTCUSDT", 3 * scanner.INTERVAL_MS, 10.0)
        self.assertIsNone(result)


class AssetUniverseTests(unittest.TestCase):
    def test_stocks_excluded_but_crypto_assets_ending_in_b_remain(self):
        self.assertTrue(scanner.is_stock_linked_symbol("GPROBUSDT"))
        self.assertTrue(scanner.is_stock_linked_symbol("TSLABUSDT"))
        self.assertFalse(scanner.is_stock_linked_symbol("BNBUSDT"))
        self.assertFalse(scanner.is_stock_linked_symbol("BTCUSDT"))

    def test_universe_keeps_active_crypto_and_excludes_stock_tokens(self):
        payload = {
            "symbols": [
                {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "BNBUSDT", "baseAsset": "BNB", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "GPROBUSDT", "baseAsset": "GPROB", "quoteAsset": "USDT", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "ETHBTC", "baseAsset": "ETH", "quoteAsset": "BTC", "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "OLDUSDT", "baseAsset": "OLD", "quoteAsset": "USDT", "status": "BREAK", "isSpotTradingAllowed": True},
            ]
        }
        with patch.object(scanner, "get_json", return_value=payload):
            self.assertEqual(scanner.fetch_universe(), ["BNBUSDT", "BTCUSDT"])


class StateAndDeduplicationTests(unittest.TestCase):
    def setUp(self):
        self.first = scanner.exact_t_candle(make_kline(0, "100", "100", "95", "100"))
        self.second = scanner.exact_t_candle(
            make_kline(scanner.INTERVAL_MS, "98", "98", "90", "98")
        )

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

    def test_stale_and_stock_pending_alerts_are_removed(self):
        now_ms = 10 * scanner.INTERVAL_MS
        recent = scanner.serialize_match("BTCUSDT", self.first, self.second)
        recent["second"]["close_time_ms"] = now_ms - 60_000
        stale = scanner.serialize_match("ETHUSDT", self.first, self.second)
        stale["second"]["close_time_ms"] = now_ms - scanner.PENDING_TTL_MS - 1
        stock = scanner.serialize_match("GPROBUSDT", self.first, self.second)
        stock["second"]["close_time_ms"] = now_ms - 60_000
        kept = scanner.prune_pending_alerts(
            {"recent": recent, "stale": stale, "stock": stock}, now_ms
        )
        self.assertEqual(list(kept), ["recent"])

    def test_corrupt_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_text("{invalid json", encoding="utf-8")
            with patch.object(scanner, "STATE_FILE", state_file):
                with self.assertRaises(RuntimeError):
                    scanner.load_state()

    def test_legacy_seen_file_migrates(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_text(json.dumps({"BTCUSDT|123": 456.0}), encoding="utf-8")
            with patch.object(scanner, "STATE_FILE", state_file):
                state = scanner.load_state()
        self.assertIn("BTCUSDT|123", state["seen"])
        self.assertEqual(state["pending"], {})

    def test_same_pair_alert_is_only_sent_once_across_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            state_file = state_dir / "seen.json"
            result = ("BTCUSDT", self.first, self.second)
            with (
                patch.object(scanner, "STATE_DIR", state_dir),
                patch.object(scanner, "STATE_FILE", state_file),
                patch.object(scanner, "TELEGRAM_TOKEN", "test-token"),
                patch.object(scanner, "TELEGRAM_CHAT_ID", "test-chat"),
                patch.object(scanner, "fetch_server_time_ms", return_value=2 * scanner.INTERVAL_MS + 100),
                patch.object(scanner, "fetch_universe", return_value=["BTCUSDT"]),
                patch.object(scanner, "scan_symbol", return_value=result),
                patch.object(scanner, "telegram", return_value=True) as send,
                patch.object(scanner, "set_github_output"),
            ):
                scanner.run_scan()
                scanner.run_scan()
            self.assertEqual(send.call_count, 1)


if __name__ == "__main__":
    unittest.main()
