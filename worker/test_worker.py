"""Tests for the queue worker's decisions: python3 -m unittest worker/test_worker.py (stdlib only)."""
import datetime as dt, os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digest  # noqa: E402
import research_job as R  # noqa: E402
import doc_job as D  # noqa: E402
import queue_ops as Q  # noqa: E402

ITEM = {"id": "Q99", "title": "t", "next": "n", "targets": [{"path": "a.md", "lines": [2, 3]}]}
FILES = {"a.md": "one\ntwo\nthree\nfour"}


class Edits(unittest.TestCase):
    def check(self, edits):
        return D.check_edits(ITEM, FILES, edits)

    def test_accepts_a_target_line(self):
        ok, why = self.check([{"path": "a.md", "line": 2, "old": "two", "new": "TWO"}])
        self.assertEqual((len(ok), why), (1, []))
        self.assertEqual(D.apply_edits(FILES, ok)["a.md"], "one\nTWO\nthree\nfour")

    def test_rejects(self):
        cases = [
            {"path": "a.md", "line": 4, "old": "four", "new": "x"},          # outside the range
            {"path": "b.md", "line": 2, "old": "two", "new": "x"},           # not a target
            {"path": "a.md", "line": 2, "old": "tw", "new": "x"},            # old doesn't match
            {"path": "a.md", "line": 2, "old": "two", "new": "a\nb"},        # adds a line
            {"path": "a.md", "line": 2, "old": "two", "new": "two"},         # no change
            {"path": "a.md", "line": "2", "old": "two", "new": "x"},         # line not an int
            {"path": "../a.md", "line": 2, "old": "two", "new": "x"},        # path games
            "edit",
        ]
        for e in cases:
            ok, why = self.check([e])
            self.assertEqual(ok, [], e)
            self.assertTrue(why, e)
        self.assertEqual(self.check("not a list"), ([], ["edits is not a list"]))

    def test_only_markdown(self):
        item = dict(ITEM, targets=[{"path": "a.py", "lines": [1, 1]}])
        ok, why = D.check_edits(item, {"a.py": "x"}, [{"path": "a.py", "line": 1, "old": "x", "new": "y"}])
        self.assertEqual(ok, [])

    def test_parse_json(self):
        self.assertEqual(D.parse_json('```json\n{"edits": []}\n```'), {"edits": []})
        self.assertIsNone(D.parse_json("no json"))
        self.assertIsNone(D.parse_json(None))


class Claims(unittest.TestCase):
    t = dt.datetime(2026, 11, 18, 12, tzinfo=dt.timezone.utc)

    def q(self, **extra):
        a = {"id": "A", "status": "open", "targets": [1], "touches": ["x.md"], "blocked_by": []}
        b = {"id": "B", "status": "open", "targets": [1], "touches": ["y/"], "blocked_by": ["C"]}
        c = {"id": "C", "status": "open", "touches": ["z"], "blocked_by": []}
        a.update(extra)
        return {"items": [a, b, c]}

    def test_ready_needs_targets_and_blockers_done(self):
        self.assertEqual([i["id"] for i in Q.ready(self.q(), self.t)], ["A"])

    def test_live_worker_lease_blocks_overlap(self):
        q = self.q()
        q["items"].append({"id": "D", "status": "in progress", "claimed_by": "worker-1", "touches": ["x.md"],
                           "lease_until": Q.iso(self.t + dt.timedelta(hours=1))})
        self.assertEqual(Q.ready(q, self.t), [])

    def test_chat_claim_always_wins_and_never_expires(self):
        q = {"items": [{"id": "D", "status": "in progress", "claimed_by": "chat-x", "touches": ["x.md"]}]}
        self.assertEqual(Q.expire(q, self.t), [])
        self.assertTrue(Q.live_claim(q["items"][0], self.t))

    def test_expired_worker_lease_reopens(self):
        q = {"items": [{"id": "D", "status": "in progress", "claimed_by": "worker-1", "touches": ["x.md"],
                        "lease_until": Q.iso(self.t - dt.timedelta(minutes=1))}]}
        self.assertEqual(Q.expire(q, self.t), ["D"])
        self.assertEqual(q["items"][0]["status"], "open")
        self.assertNotIn("claimed_by", q["items"][0])

    def test_overlap(self):
        self.assertTrue(Q._overlap("record/", "record/README.md"))
        self.assertFalse(Q._overlap("record/README.md", "record/READMEx.md"))


class Research(unittest.TestCase):
    src = [{"sid": "S1", "abstract": "We find that logic gate networks reach 98.5% accuracy on MNIST at low cost."}]

    def test_quote_and_numbers_checked(self):
        ok, dropped = R.verified([
            {"source": "S1", "claim": "LGNs reach 98.5% on MNIST", "quote": "logic gate networks reach 98.5% accuracy on MNIST"},
            {"source": "S1", "claim": "LGNs reach 99% on MNIST", "quote": "logic gate networks reach 98.5% accuracy on MNIST"},
            {"source": "S1", "claim": "x", "quote": "words that are not in the abstract"},
            {"source": "S9", "claim": "x", "quote": "logic gate networks reach 98.5% accuracy"},
            {"source": "S1", "claim": "x", "quote": "logic gate"},
            "junk"], self.src)
        self.assertEqual(len(ok), 1)
        self.assertEqual(len(dropped), 5)


class Digest(unittest.TestCase):
    def test_flags_unpointed_numbers(self):
        s = "Q10 is done in run 44263440298. It saved 30 percent. Nothing else changed."
        self.assertEqual(digest.flag(s), ["It saved 30 percent."])


if __name__ == "__main__":
    unittest.main()
