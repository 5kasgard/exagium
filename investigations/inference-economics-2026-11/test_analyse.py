"""Tests for the offline inference-economics analyser (#258).

Run: python -m pytest test_analyse.py -q   (or: python test_analyse.py)
No network, no paid calls, no secrets.
"""
import csv
import io
import os
import tempfile
import unittest

import analyse


def _write(dirpath, name, text):
    with open(os.path.join(dirpath, name), "w", newline="", encoding="utf-8") as fh:
        fh.write(text)


CONSOLE_FIXTURE = (
    "usage_date_utc,model,context_window,input_no_cache_usd,output_usd,"
    "input_cache_read_usd,input_cache_write_5m_usd,input_cache_write_1h_usd,total_usd\n"
    '"2026-11-25","Claude Opus 5.5","<=200k",0.000344,0.659340,0.492738,4.532380,0.0,5.684802\n'
    '"2026-11-25","Claude Sonnet 5.5","<=200k",0.000072,0.076050,0.040470,0.298298,0.0,0.414890\n'
)

OPENROUTER_FIXTURE = (
    "created_at__day,model,session_id,total_usage\n"
    "2026-11-25,Claude Opus 5.5,none,6.57274\n"
    "2026-11-25,DeepSeek V4.1 Flash 1017,none,0.323751778\n"
)


class AnalyserTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_console_totals_and_cache_share(self):
        rows = list(csv.DictReader(io.StringIO(CONSOLE_FIXTURE)))
        res = analyse.analyse_console(rows)
        self.assertAlmostEqual(res["total_usd"], 6.099692, places=6)
        # cache lifecycle = 5m write + 1h write + read
        self.assertAlmostEqual(res["cache_lifecycle_usd"], 5.363886, places=6)
        self.assertGreater(res["cache_lifecycle_pct"], 80.0)
        self.assertEqual(res["worst_day"], "2026-11-25")

    def test_openrouter_top_model(self):
        rows = list(csv.DictReader(io.StringIO(OPENROUTER_FIXTURE)))
        res = analyse.analyse_openrouter(rows)
        self.assertAlmostEqual(res["total_usage_sum"], 6.896491778, places=6)
        self.assertEqual(res["top_model"], "Claude Opus 5.5")
        self.assertGreater(res["top_model_pct"], 90.0)

    def test_reconcile_opt_in(self):
        console = {"total_usd": 9.013842}
        openrouter = {"total_usage_sum": 8.73274}
        self.assertIsNone(analyse.reconcile(console, openrouter, None))
        recon = analyse.reconcile(console, openrouter, {"observed_spend_usd": 18.64})
        self.assertAlmostEqual(recon["export_sum_usd"], 17.746582, places=6)
        self.assertAlmostEqual(recon["residual_usd"], 0.893418, places=6)

    def test_end_to_end_from_files(self):
        _write(self.tmp, analyse.CONSOLE, CONSOLE_FIXTURE)
        _write(self.tmp, analyse.OPENROUTER, OPENROUTER_FIXTURE)
        console = analyse.analyse_console(analyse._rows(os.path.join(self.tmp, analyse.CONSOLE)))
        openrouter = analyse.analyse_openrouter(analyse._rows(os.path.join(self.tmp, analyse.OPENROUTER)))
        self.assertAlmostEqual(console["total_usd"], 6.099692, places=6)
        self.assertAlmostEqual(openrouter["total_usage_sum"], 6.896491778, places=6)


if __name__ == "__main__":
    unittest.main()
