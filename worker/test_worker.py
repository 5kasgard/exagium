"""Tests for the queue worker's decisions: python3 -m unittest worker/test_worker.py (stdlib only)."""
import datetime as dt, json, os, sys, unittest

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
            {"path": ["a.md"], "line": 2, "old": "two", "new": "x"},         # unhashable path (review M4)
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
        a = {"id": "A", "status": "open", "targets": [{"path": "x.md", "lines": [1, 1]}], "touches": ["x.md"], "blocked_by": []}
        b = {"id": "B", "status": "open", "targets": [{"path": "y/a.md", "lines": [1, 1]}], "touches": ["y/a.md"], "blocked_by": ["C"]}
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

    def test_touches_must_cover_targets_and_ids_plain(self):
        q = self.q(touches=["other.md"], targets=[{"path": "x.md", "lines": [1, 1]}])
        self.assertEqual(Q.ready(q, self.t), [])
        q = self.q(id="A\nB=1", targets=[{"path": "x.md", "lines": [1, 1]}])
        self.assertEqual(Q.ready(q, self.t), [])

    def test_failures_park(self):
        self.assertEqual(Q.failures({"runs": [{"result": "failed: x"}, {"result": "not started: cap"}, {"result": "failed: y"}]}), 2)
        self.assertEqual(Q.failures({"runs": [{"result": "failed: x"}, {"result": "merged"}, {"result": "failed: y"}]}), 1)
        self.assertEqual(Q.failures({"runs": [{"result": "not started: cap"}, {"result": "not started: cap"}]}), 0)

    def test_overlap(self):
        self.assertTrue(Q._overlap("record/", "record/README.md"))
        self.assertFalse(Q._overlap("record/README.md", "record/READMEx.md"))


class Research(unittest.TestCase):
    src = [{"sid": "S1", "abstract": "We find that logic gate networks reach 98.5% accuracy on MNIST at low cost."}]

    def test_quote_and_numbers_checked(self):
        ok, dropped = R.verified([
            {"source": "S1", "claim": "LGNs reach 98.5% on MNIST", "quote": "logic gate networks reach 98.5% accuracy on MNIST", "evidence": "measured"},
            {"source": "S1", "claim": "LGNs reach 99% on MNIST", "quote": "logic gate networks reach 98.5% accuracy on MNIST", "evidence": "measured"},
            {"source": "S1", "claim": "LGNs reach 98.5% on MNIST", "quote": "logic gate networks reach 98.5% accuracy on MNIST", "evidence": "proof"},
            {"source": ["S1"], "claim": "x", "quote": "logic gate networks reach 98.5% accuracy"},
            {"source": "S1", "claim": "x", "quote": "words that are not in the abstract"},
            {"source": "S9", "claim": "x", "quote": "logic gate networks reach 98.5% accuracy"},
            {"source": "S1", "claim": "x", "quote": "logic gate"},
            "junk"], self.src)
        self.assertEqual(len(ok), 1)
        self.assertEqual(len(dropped), 7)

    def test_clean(self):
        t = R.clean("see https://evil.example <script>x</script> @someone fixes #12 [a](b)\nnext")
        for bad in ("http", "<", "@someone", "#12", "](", "\n"):
            self.assertNotIn(bad, t)


class DocSafety(unittest.TestCase):
    def test_paths_and_new_lines(self):
        for p in (".github/a.md", "worker/README.md", "../a.md", "/etc/a.md", "a.py", ["a.md"], "router/bench4/PREREG.md"):
            self.assertFalse(D.safe_path(p), p)
        for p in ("AGENTS.md", "x/HANDOFF-a.md", "VERIFY.md", "explore/BRIEF.md", ".claude/x.md", "a/SKILL.md",
                  "tests/canary-BRIEF.md", "record/experiments/anchors/VERIFY-anchors.md"):
            self.assertTrue(D.safe_path(p), p)
            self.assertFalse(D.merge_ok(p), p)
        self.assertTrue(D.merge_ok("record/README.md"))
        self.assertTrue(D.merge_ok("tests/worker-canary-2.md"))   # Q59: the wider rule must not catch ordinary files
        self.assertFalse(D.safe_new("[a](https://good.org/x)", "[a](https://evil.com/y)"))   # review N2
        self.assertFalse(D.safe_new("see https://a.b/c", "see https://a.b/c and https://d.e"))
        self.assertTrue(D.safe_new("see https://a.b/c here", "see https://a.b/c there"))
        self.assertFalse(D.safe_new("see https://a.b/c here", "see https://a.b/c, here"))   # strict: fails closed, to a PR
        self.assertFalse(D.safe_new("plain line", "plain line, see https://x.y"))
        self.assertFalse(D.safe_new("plain", "plain <img src=x>"))
        self.assertFalse(D.safe_new("plain", "ping @owner"))
        self.assertTrue(D.safe_new("the guarantee", "the claim, today, rests on who can push"))


class Digest(unittest.TestCase):
    def test_flags_unpointed_numbers(self):
        s = "Q10 is done in run 44263440298. It saved 30 percent. Nothing else changed."
        self.assertEqual(digest.flag(s), ["It saved 30 percent."])


class Ledger(unittest.TestCase):
    """Reserved calls are held until the run records, then only calls made count (2026-11-19 fallback-cap fault)."""

    def test_unused_fallback_reservation_is_released_on_record(self):
        import ledger as LG
        led = {"cloudflare": {"reserved": 20, "calls": 1, "neurons": 281.0, "runs": [], "pending": {}}}
        self.assertIsNone(LG.headroom(led, LG.parse("cloudflare:1:400")))   # 1 made of 20: room left

    def test_unrecorded_reservations_still_count(self):
        import ledger as LG
        n = LG.CAPS["cloudflare"]["calls"]
        led = {"cloudflare": {"reserved": n, "calls": 0, "neurons": 0.0, "runs": [], "pending": {},
                              "pending_calls": {f"r{i}": 1 for i in range(n)}}}
        self.assertIsNotNone(LG.headroom(led, LG.parse("cloudflare:1:400")))   # 20 held by runs in flight

    def test_cap_on_calls_made(self):
        import ledger as LG
        led = {"nvidia": {"reserved": LG.CAPS["nvidia"]["calls"], "calls": LG.CAPS["nvidia"]["calls"], "runs": []}}
        self.assertIsNotNone(LG.headroom(led, LG.parse("nvidia:1")))


if __name__ == "__main__":
    unittest.main()


class RejectedNotOffered(unittest.TestCase):
    """Q59 (release test 2, Q38): an edit the reviewer rejected is restored, not turned into a PR."""
    def run_apply(self, verdict):
        import tempfile
        with tempfile.TemporaryDirectory() as repo, tempfile.TemporaryDirectory() as out:
            os.makedirs(os.path.join(repo, "docs"))
            text = "one\nResults can never be changed.\nthree\n"
            open(os.path.join(repo, "docs/a.md"), "w").write(text)
            item = {"id": "X1", "title": "t", "next": "n", "targets": [{"path": "docs/a.md", "lines": [2, 2]}]}
            json.dump({"item": item, "files": {"docs/a.md": text}}, open(os.path.join(out, "bundle.json"), "w"))
            edit = {"path": "docs/a.md", "line": 2, "old": "Results can never be changed.", "new": "Results are expected not to change."}
            json.dump({"proposal": {"edits": [edit], "note": "x"}, "review": {"verdict": verdict, "reason": "r"}},
                      open(os.path.join(out, "draft.json"), "w"))
            D.apply(repo, out)
            return json.load(open(os.path.join(out, "result.json"))), open(os.path.join(repo, "docs/a.md")).read(), text

    def test_reject_restores(self):
        res, now, before = self.run_apply("reject")
        self.assertEqual(res["result"], "rejected")
        self.assertEqual(now, before)

    def test_approve_merges(self):
        res, now, before = self.run_apply("approve")
        self.assertEqual(res["result"], "merge")
        self.assertIn("expected not to change", now)


class GuaranteeWords(unittest.TestCase):
    def test_lost(self):
        self.assertEqual(D.lost_guarantee("Results can never be changed, by anyone.", "Results cannot be changed."), ["anyone", "never"])
        self.assertEqual(D.lost_guarantee("The cap is 4,000.", "The cap is 5,000."), [])
        self.assertEqual(D.lost_guarantee("It must run.", "It must always run."), [])
