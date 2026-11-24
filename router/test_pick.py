"""router/pick.py: least-used pools first, one reviewer per pool, pools over their cap skipped."""
import importlib.util, json, os, subprocess, sys, tempfile, unittest

PICK = os.path.join(os.path.dirname(__file__), "pick.py")


def run(led, *a):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(led, f)
    return subprocess.run([sys.executable, "-I", PICK, f.name, *a], capture_output=True, text=True).stdout.split()


def load_pick():
    spec = importlib.util.spec_from_file_location("pick_under_test", PICK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    def test_best_pool_picks_most_room(self):
        p = load_pick()
        led = {"nvidia": {"calls": 449}, "openrouter": {"calls": 49}}
        self.assertEqual(p.best_pool(led, "nemotron-120b"), "openrouter")

    def test_best_pool_none_when_all_full(self):
        p = load_pick()
        led = {"nvidia": {"calls": 10000}, "openrouter": {"calls": 10000}}
        self.assertIsNone(p.best_pool(led, "nemotron-120b"))


if __name__ == "__main__":
    unittest.main()
