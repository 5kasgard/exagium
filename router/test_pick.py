"""router/pick.py: least-used pools first, one reviewer per pool, pools over their cap skipped."""
import json, os, subprocess, sys, tempfile, unittest

PICK = os.path.join(os.path.dirname(__file__), "pick.py")
sys.path.insert(0, os.path.dirname(__file__))
import pick  # noqa: E402


def run(led, *a):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(led, f)
    return subprocess.run([sys.executable, "-I", PICK, f.name, *a], capture_output=True, text=True).stdout.split()


def cap_calls(pool):
    return pick.CAPS[pool]["calls"]   # the real caps; a test must not add pools to the module's CAPS


def at(pool, frac):
    return {pool: {"calls": max(0, round(frac * cap_calls(pool)) - 1)}}


class Pick(unittest.TestCase):
    def test_spreads_across_pools(self):
        got = run({"cloudflare": {"calls": 10, "neurons": 4900}, "nvidia": {"calls": 390}, "groq": {"calls": 5}, "openrouter": {"calls": 1}}, "2")
        self.assertEqual([g.split(":")[0] for g in got], ["openrouter", "groq"])   # openrouter 2/500 used < groq 6/500

    def test_full_pool_is_skipped_and_dropped(self):
        led = {"cloudflare": {"calls": 10, "neurons": 4900}}
        self.assertNotIn("cloudflare", [g.split(":")[0] for g in run(led, "5")])
        self.assertEqual(run(led, "check", "cloudflare:@cf/x", "groq:y"), ["groq:y"])

    def test_f037_reserved_calls_count(self):
        """GLM F037: a pool whose headroom runs in flight have reserved is full, though few calls are recorded."""
        led = {"groq": {"calls": 5, "pending_calls": {"r1": 495}}}
        self.assertEqual(run(led, "check", "groq:y", "nvidia:z"), ["nvidia:z"])

    def test_best_pool_picks_pool_with_most_room(self):
        """K58: nvidia (NIM) at 90% and openrouter at 10% -> openrouter for nemotron-120b."""
        led = {}
        led.update(at("nvidia", 0.9))
        led.update(at("openrouter", 0.1))
        self.assertEqual(pick.best_pool(led, "nemotron-120b"), "openrouter")

    def test_best_pool_reads_nim_usage_under_its_ledger_name(self):
        """Review 2026-11-24: NIM usage is recorded under 'nvidia'; a full nvidia entry must not look empty."""
        led = {"nvidia": {"calls": cap_calls("nvidia")}}
        led.update(at("openrouter", 0.5))
        self.assertEqual(pick.best_pool(led, "nemotron-120b"), "openrouter")

    def test_best_pool_none_when_all_full(self):
        """K58: no equivalent pool has room -> None."""
        full = {p: {"calls": cap_calls(p)} for p in pick.EQUIVALENT["gpt-oss-120b"]}
        self.assertIsNone(pick.best_pool(full, "gpt-oss-120b"))


if __name__ == "__main__":
    unittest.main()
