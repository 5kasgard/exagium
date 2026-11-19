import json, os, subprocess, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(__file__))
import indicators as I

QP = "audit/2026-11-17/queue.json"


def commit(repo, items, t):
    p = os.path.join(repo, QP)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({"items": items}, open(p, "w"))
    env = dict(os.environ, GIT_AUTHOR_DATE=f"@{t} +0000", GIT_COMMITTER_DATE=f"@{t} +0000")
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True)
    subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "q"], check=True, env=env)


class Flow(unittest.TestCase):
    def test_arrivals_closures_cycle_and_littles_law(self):
        d = tempfile.mkdtemp()
        subprocess.run(["git", "init", "-q", d], check=True)
        now = 1_800_000_000
        old = now - 30 * 86400
        commit(d, [{"id": "A", "status": "open"}], old)                                   # arrived before the window
        commit(d, [{"id": "A", "status": "open"}, {"id": "B", "status": "open", "origin": "new"}], now - 5 * 86400)
        commit(d, [{"id": "A", "status": "done"}, {"id": "B", "status": "open", "origin": "new"},
                   {"id": "C", "status": "open", "origin": "follow-up"}], now - 2 * 86400)
        q = {"items": [{"id": "A", "status": "done"}, {"id": "B", "status": "open", "origin": "new"},
                       {"id": "C", "status": "open", "origin": "follow-up"}]}
        f = I.flow(q, d, now=now)
        self.assertEqual(f["arrived"], 2)
        self.assertEqual(f["arrived_by_origin"], {"new": 1, "follow-up": 1})
        self.assertEqual(f["closed"], 1)
        self.assertEqual(f["median_hours_to_done"], 28 * 24.0)                            # A: first seen 30 days ago, done 2 days ago
        self.assertEqual(f["open_now"], 2)
        self.assertEqual(f["littles_wait_days"], round(2 / (1 / 7), 1))

    def test_no_closures_gives_no_wait_estimate(self):
        d = tempfile.mkdtemp()
        subprocess.run(["git", "init", "-q", d], check=True)
        commit(d, [{"id": "A", "status": "open"}], 1_800_000_000 - 86400)
        f = I.flow({"items": [{"id": "A", "status": "open"}]}, d, now=1_800_000_000)
        self.assertEqual((f["closed"], f["littles_wait_days"]), (0, None))


class Connectors(unittest.TestCase):
    def test_used_only_when_every_path_exists_and_is_not_empty(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "audit/periodic"))
        open(os.path.join(d, "real.json"), "w").write("{}")
        open(os.path.join(d, "empty.json"), "w").close()
        json.dump({"connectors": {"a": {"evidence": ["real.json"]}, "b": {"evidence": ["real.json", "missing.json"]},
                                  "c": {"evidence": ["empty.json"]}, "d": {"evidence": []}}},
                  open(os.path.join(d, "audit/periodic/connectors.json"), "w"))
        c = I.connectors(d)
        self.assertEqual(c["by_connector"], {"a": "used", "b": "not yet", "c": "not yet", "d": "not yet"})
        self.assertEqual((c["used"], c["total"]), (1, 4))


if __name__ == "__main__":
    unittest.main()
