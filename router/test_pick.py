"""router/pick.py: least-used pools first, one reviewer per pool, pools over their cap skipped."""
import json, os, subprocess, sys, tempfile, unittest

PICK = os.path.join(os.path.dirname(__file__), "pick.py")


def run(led, *a):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(led, f)
    return subprocess.run([sys.executable, "-I", PICK, f.name, *a], capture_output=True, text=True).stdout.split()


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


if __name__ == "__main__":
    unittest.main()
