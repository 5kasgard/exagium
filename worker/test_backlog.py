import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backlog as B  # noqa: E402
import transitions as T  # noqa: E402


def item(i, status, **k):
    return dict({"id": i, "title": i, "status": status, "kind": "test", "needs": "free", "touches": ["x"],
                 "filed_at": "2026-11-20T00:00:00Z"}, **k)


class Lanes(unittest.TestCase):
    def test_every_state_has_a_lane(self):
        """No status may fall through to 'unknown': a new state must get a consumer before it can hold items."""
        for st in T.STATES:
            if st == "done":
                continue
            L = B.lanes({"items": [item("A", st, next="owner-only: repo-settings" if st == "waiting on the owner" else "x",
                                        runs=[{"at": "2026-11-24T00:00:00Z", "outcome": "park", "result": "superseded"}])]})
            self.assertNotEqual(L["A"]["lane"], "orphan", st)

    def test_orphans(self):
        q = {"items": [item("A", "waiting on the owner", next="unsure what to do"),
                       item("B", "parked", next=" "),
                       item("C", "open", blocked_by=["B"]),
                       item("D", "open", next="Owner, 2026-11-19: add to the queue only, no work yet.")]}
        L = B.lanes(q)
        self.assertEqual([L[k]["lane"] for k in "ABCD"], ["orphan", "orphan", "orphan", "owner-only"])
        text, _, orphans, _ = B.report(q)
        self.assertEqual(sorted(orphans), ["A", "B", "C"])
        self.assertTrue(text.splitlines()[3].lstrip().startswith("ORPHAN"))

    def test_open_without_a_job_is_claude_work_not_hidden(self):
        L = B.lanes({"items": [item("A", "open", kind="decision", split_tried="2026-11-24T00:00:00Z")]})
        self.assertEqual(L["A"]["lane"], "claude")

    def test_open_experiment_is_worker_work(self):
        L = B.lanes({"items": [item("A", "open", kind="test", experiment={"name": "queue_drift"}),
                               item("B", "open", kind="test", experiment={"name": "no_such_experiment"})]})
        self.assertEqual((L["A"]["lane"], L["A"]["why"]), ("worker", "experiment"))
        self.assertEqual(L["B"]["lane"], "claude")

    def test_stale_and_undated(self):
        import datetime as dt
        now = dt.datetime(2026, 12, 1, tzinfo=dt.timezone.utc)
        L = B.lanes({"items": [item("A", "open", kind="decision"), item("B", "open", kind="decision", filed_at=None)]}, now)
        self.assertTrue(L["A"]["stale"])            # 11 days in the claude lane, limit 7
        self.assertFalse(L["B"]["stale"])           # undated: counted apart in the report, not called fresh or stale
        self.assertIsNone(L["B"]["age_days"])


class RunItems(unittest.TestCase):
    def test_run_items_say_what_they_touch(self):
        """K71 was born with touches []; render.py refused opening it and the splitter crashed on every run (2026-11-25)."""
        import generate as G
        self.assertEqual(G.run_touches("packet_pilot"), "router-runs:experiments/packet_pilot/")
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.py")).read()
        self.assertNotIn('"touches": []', src)


if __name__ == "__main__":
    unittest.main()
