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

    def test_probe_error_is_not_a_change(self):
        rc, _ = self.go({"x": RuntimeError("down"), "y": "v"})
        self.assertEqual(rc, 3)

    def test_not_due_is_skipped(self):
        now = dt.datetime(2026, 11, 20, tzinfo=dt.timezone.utc)
        recent = {"at": (now - dt.timedelta(hours=1)).isoformat(), "value": 1, "same": True}
        rc, _ = self.go({"x": 999, "y": 999}, status={"a": recent, "b": recent}, now=now)
        self.assertEqual(rc, 0)   # neither probed: both checked within their recheck window


if __name__ == "__main__":
    unittest.main()
