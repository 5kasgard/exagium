"""Offline tests for facts/check.py (queue P30)."""
import datetime as dt, json, os, sys, tempfile, unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import check  # noqa: E402

REG = {"facts": [{"id": "a", "probe": {"kind": "x"}, "value": 1, "recheck_days": 1, "cited_by": ["thing A"]},
                 {"id": "b", "probe": {"kind": "y"}, "value": "v", "recheck_days": 7, "cited_by": ["thing B"]}]}


class Check(unittest.TestCase):
    def go(self, values, status=None, now=None):
        with tempfile.TemporaryDirectory() as d:
            r, s = os.path.join(d, "r.json"), os.path.join(d, "s.json")
            json.dump(REG, open(r, "w"))
            if status:
                json.dump(status, open(s, "w"))

            def probe(p):
                v = values[p["kind"]]
                if isinstance(v, Exception):
                    raise v
                return v
            with mock.patch.object(check, "probe", probe):
                rc = check.run(r, s, now)
            return rc, json.load(open(s))

    def test_same_passes(self):
        rc, st = self.go({"x": 1, "y": "v"})
        self.assertEqual(rc, 0)
        self.assertTrue(st["a"]["same"])

    def test_change_fails_and_names_dependants(self):
        rc, st = self.go({"x": 2, "y": "v"})
        self.assertEqual(rc, 2)
        self.assertFalse(st["a"]["same"])

    def test_change_record_carries_dependants(self):
        reg = {"facts": [{"id": "a", "probe": {"kind": "x"}, "value": 1, "recheck_days": 1,
                          "cited_by": ["thing A"], "dependants": ["worker/research_job.py", "card:Q12"]}]}
        with tempfile.TemporaryDirectory() as d:
            r, s = os.path.join(d, "r.json"), os.path.join(d, "s.json")
            json.dump(reg, open(r, "w"))
            with mock.patch.object(check, "probe", lambda p: 2), \
                 mock.patch("builtins.print") as pr:
                self.assertEqual(check.run(r, s), 2)
            recs = [c.args[1] for c in pr.call_args_list if c.args and c.args[0] == "CHANGED"]
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["dependants"], ["worker/research_job.py", "card:Q12"])
            with mock.patch.object(check, "probe", lambda p: 1), \
                 mock.patch("builtins.print") as pr:
                self.assertEqual(check.run(r, os.path.join(d, "s2.json")), 0)
            self.assertEqual([c for c in pr.call_args_list if c.args and c.args[0] == "CHANGED"], [])

    def test_probe_error_is_not_a_change(self):
        rc, _ = self.go({"x": RuntimeError("down"), "y": "v"})
        self.assertEqual(rc, 3)

    def test_not_due_is_skipped(self):
        now = dt.datetime(2026, 11, 20, tzinfo=dt.timezone.utc)
        recent = {"at": (now - dt.timedelta(hours=1)).isoformat(), "value": 1, "same": True}
        rc, _ = self.go({"x": 999, "y": 999}, status={"a": recent, "b": recent}, now=now)
        self.assertEqual(rc, 0)   # neither probed: both checked within their recheck window

    def test_keyed_facts_only_with_keyed(self):
        reg = {"facts": REG["facts"] + [{"id": "k", "keyed": True, "probe": {"kind": "z"}, "value": 5, "recheck_days": 1, "cited_by": ["K"]}]}
        seen = []
        with tempfile.TemporaryDirectory() as d, mock.patch.object(check, "probe", lambda p: seen.append(p["kind"]) or {"x": 1, "y": "v", "z": 6}[p["kind"]]):
            r = os.path.join(d, "r.json"); json.dump(reg, open(r, "w"))
            self.assertEqual(check.run(r, os.path.join(d, "s1.json")), 0)                 # keyless run: k untouched
            self.assertEqual(check.run(r, os.path.join(d, "s2.json"), keyed=True), 2)     # keyed run: only k, and it changed
        self.assertEqual(seen, ["x", "y", "z"])

    def test_field(self):
        self.assertEqual(check.field({"a": {"b": 3}}, "a.b"), 3)


if __name__ == "__main__":
    unittest.main()
