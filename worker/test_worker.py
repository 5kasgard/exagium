"""Tests for the queue worker's decisions: python3 -m unittest worker/test_worker.py (stdlib only)."""
import datetime as dt, json, os, sys, unittest, unittest.mock, tempfile

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

    def test_chat_claim_holds_for_its_ttl(self):
        """A chat claim wins for CHAT_TTL_H hours, then expires (GLM review F017: it used to hold forever)."""
        q = {"items": [{"id": "D", "status": "in progress", "claimed_by": "chat-x", "touches": ["x.md"],
                        "claimed_at": Q.iso(self.t - dt.timedelta(hours=1))}]}
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

    def test_rolling_24h(self):
        """Cloudflare's allowance is a rolling 24 h (refused at 06:09Z with 2,405 neurons that UTC day)."""
        import ledger as LG, tempfile
        with tempfile.TemporaryDirectory() as wt:
            os.makedirs(os.path.join(wt, "ledger"))
            rows = [("2026-11-19T05:00:00Z", 5000), ("2026-11-19T20:00:00Z", 6000), ("2026-11-20T01:00:00Z", 2400)]
            with open(os.path.join(wt, "ledger", "calls.jsonl"), "w") as f:
                for t, n in rows:
                    f.write(json.dumps({"t": t, "engine": "cloudflare", "usage": {"neurons": n}}) + "\n")
                f.write(json.dumps({"t": "2026-11-20T02:00:00Z", "engine": "nvidia", "usage": {}}) + "\n")
            now = dt.datetime(2026, 11, 20, 6, 9, tzinfo=dt.timezone.utc)
            self.assertEqual(LG.rolling_neurons(wt, now), 8400)   # the 05:00Z call of the day before is out of the window
            self.assertIsNotNone(LG.headroom({}, LG.parse("cloudflare:1:600"), 8500))   # 9,100 > 9,000
            self.assertIsNone(LG.headroom({}, LG.parse("cloudflare:1:600"), 8000))
            self.assertIsNone(LG.headroom({}, LG.parse("nvidia:1"), 99999))   # other pools: no rolling rule




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
            edit = {"path": "docs/a.md", "line": 2, "old": "Results can never be changed.", "new": "Results can never be altered."}
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
        self.assertIn("never be altered", now)


class GuaranteeWords(unittest.TestCase):
    def test_lost(self):
        self.assertEqual(D.lost_guarantee("Results can never be changed, by anyone.", "Results cannot be changed."), ["anyone", "never"])
        self.assertEqual(D.lost_guarantee("The cap is 4,000.", "The cap is 5,000."), [])
        self.assertEqual(D.lost_guarantee("It must run.", "It must always run."), [])


class Rerank(unittest.TestCase):
    """P32: the citation pick is the live behaviour; the rerank arm only changes the sort key."""
    pool = [{"title": f"t{i}", "abstract": "a " * 20, "group": g, "cited_by": c}
            for i, (g, c) in enumerate([("papers", 5), ("papers", 50), ("papers", 1), ("forum", 0), ("forum", 3)])]

    def test_cite_order(self):
        old = R.PER_KIND.copy()
        try:
            R.PER_KIND.update({"papers": 2, "forum": 1})
            p = R.pick(self.pool, ["papers", "forum"])
            self.assertEqual([x["title"] for x in p], ["t1", "t0", "t4"])
            self.assertEqual([x["sid"] for x in p], ["S1", "S2", "S3"])
            q = R.pick(self.pool, ["papers", "forum"], [0.1, 0.2, 0.9, 0.8, 0.0])
            self.assertEqual([x["title"] for x in q], ["t2", "t1", "t3"])
            self.assertNotIn("sid", self.pool[0])   # the pool is not changed
        finally:
            R.PER_KIND.clear(); R.PER_KIND.update(old)

    def test_scores_parse(self):
        import io, unittest.mock as m

        class Resp(io.BytesIO):
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False
        body = {"results": [{"index": 1, "relevance_score": 0.7}, {"index": 0, "relevance_score": 0.2}]}
        calls = []
        with m.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), \
                m.patch("urllib.request.urlopen", return_value=Resp(json.dumps(body).encode())):
            s, err = R.rerank_scores("q", self.pool[:2], calls)
        self.assertEqual((s, err), ([0.2, 0.7], None))
        body["results"].pop()
        with m.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), \
                m.patch("urllib.request.urlopen", return_value=Resp(json.dumps(body).encode())):
            s, err = R.rerank_scores("q", self.pool[:2], calls)
        self.assertIsNone(s); self.assertIn("1 of 2", err)
        self.assertEqual(len(calls), 2)
        one = lambda *a, **k: Resp(json.dumps({"results": [{"index": 0, "relevance_score": 0.5}]}).encode())
        with m.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), m.patch.object(R, "RERANK_BATCH", 1), \
                m.patch("urllib.request.urlopen", side_effect=one):
            s, err = R.rerank_scores("q", self.pool[:3], calls)
        self.assertEqual((s, err), ([0.5, 0.5, 0.5], None))
        self.assertEqual(len(calls), 5)   # one request per batch


class Generate(unittest.TestCase):
    """P27 generator: script checks and the gate's controls (no model call)."""

    def test_check(self):
        import generate as G
        ok, why = G.check({"question": "How well do cascades of cheap and strong language models trade cost for accuracy?",
                           "queries": ["LLM cascade cost accuracy", "FrugalGPT", "model routing benchmark"]})
        self.assertIsNone(why); self.assertEqual(len(ok["queries"]), 3)
        for bad in ({"question": "short?", "queries": ["a b c"] * 3},
                    {"question": "How does the Forge queue worker decide what to claim next in practice?", "queries": ["x y z"] * 3},
                    {"question": "What does https://example.com say about model routing and cascades?", "queries": ["abc"] * 3},
                    {"question": "How well do cascades trade cost for accuracy in practice today?", "queries": ["abc"]},
                    "not a dict"):
            self.assertIsNone(G.check(bad)[0])

    def test_near_dup_and_ids(self):
        import generate as G
        self.assertTrue(G.near_dup("How strong are tuned gradient-boosted trees against MLPs on tabular data?",
                                   ["How strong are tuned gradient-boosted decision trees against MLPs on tabular data?"]))
        self.assertFalse(G.near_dup("How reliable are LLM judges?", ["What limits CPU inference of TabPFN?"]))
        self.assertEqual(G.next_rid({"items": [{"id": "R9"}, {"id": "R19"}, {"id": "P5"}]}), 20)

    def test_gate_controls(self):
        import generate as G
        self.assertEqual(G.gate_outcomes([0.9, 0.1, 0.7, 0.2], 2), ["accept", "reject"])
        self.assertIsNone(G.gate_outcomes([0.4, 0.1, 0.9], 1))   # the useful control failed: decide nothing
        self.assertIsNone(G.gate_outcomes([0.9, 0.6, 0.9], 1))   # the off-topic control passed: decide nothing
        self.assertIsNone(G.gate_outcomes(None, 1))

    def test_candidates(self):
        import generate as G
        q = {"items": [{"id": "P1", "status": "open", "kind": "build", "value": "low"},
                       {"id": "P2", "status": "open", "kind": "build", "value": "high"},
                       {"id": "P3", "status": "open", "kind": "doc", "value": "high"},
                       {"id": "P4", "status": "open", "kind": "build", "value": "high", "needs": "the owner"},
                       {"id": "R9", "status": "open", "kind": "test", "research": {"question": "q"}},
                       {"id": "R10", "status": "proposed", "generated_from": "P1"}]}
        self.assertEqual([i["id"] for i in G.candidates(q)], ["P2"])

    def test_transitions(self):
        import transitions as TR
        self.assertEqual(TR.next_status("proposed", "accept"), "open")
        self.assertEqual(TR.next_status("proposed", "reject"), "parked")
        with self.assertRaises(TR.NotAllowed):
            TR.next_status("proposed", "claim")


class FallbackReader(unittest.TestCase):
    def test_429_reads_on_fallback_pool(self):
        import tempfile
        seen = []

        def ask(engine, model, prompt):
            seen.append(engine)
            if engine == "cloudflare":
                return {"ok": False, "status": 429, "text": None, "error": "spent"}
            return {"ok": True, "status": 200, "text": '{"claims": [], "open_questions": []}'}
        with tempfile.TemporaryDirectory() as d:
            R._read(d, {"sources": [], "item": {}}, "q?", [], ask)
            r = json.load(open(os.path.join(d, "read.json")))
        self.assertEqual(seen, ["cloudflare", "nvidia"])
        self.assertEqual(r["model_fail"], [])


class Slots(unittest.TestCase):
    def test_paid_only_when_reserved(self):
        import slots, unittest.mock as m
        with m.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PAID_FALLBACK", None)
            self.assertTrue(all(p != "openrouter_paid" for p, _ in slots.chain("reader")))
        with m.patch.dict(os.environ, {"PAID_FALLBACK": "1"}):
            self.assertEqual([p for p, _ in slots.chain("reader")][-2:], ["openrouter_paid", "openrouter_paid"])
        self.assertEqual(slots.chain("reader", ("nvidia", "x")), [("nvidia", "x")])

    def test_moves_on_until_a_reply_parses(self):
        import slots, unittest.mock as m
        replies = iter([{"ok": False, "status": 429}, {"ok": True, "status": 200, "text": "not json"},
                        {"ok": True, "status": 200, "text": '{"a": 1}'}])
        with m.patch.dict(os.environ, {"PAID_FALLBACK": "1"}):
            got, last, tries = slots.run("reader", "p", lambda *a: next(replies), lambda t: R.parse_json(t))
        self.assertEqual(got, {"a": 1})
        self.assertEqual([t["status"] for t in tries], [429, 200, 200])
        self.assertEqual(tries[2]["pool"], "openrouter_paid")

    def test_ledger_dollar_cap(self):
        import ledger as LG
        self.assertIsNone(LG.headroom({}, LG.parse("openrouter_paid:2:5")))
        self.assertIsNotNone(LG.headroom({"openrouter_paid": {"calls": 1, "usd": 0.245}}, LG.parse("openrouter_paid:2:5")))


class ClaudeCost(unittest.TestCase):
    def test_prices(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "router"))
        import engines
        self.assertAlmostEqual(engines.claude_cost("claude-haiku-5-5", 10_000, 2_000), 0.002)
        self.assertAlmostEqual(engines.claude_cost("claude-haiku-5-5", 200_000, 0), 0.1)   # over 100,000: five times
        self.assertAlmostEqual(engines.claude_cost("claude-sonnet-5-5", 1_000_000, 100_000), 3.0)
        with self.assertRaises(ValueError):
            engines.anthropic_call("claude-unpriced", "hi")


class Tier(unittest.TestCase):
    def test_decide_checks_the_redo_question(self):
        import tier as TI
        ctx = TI.CONTROLS[0][0]
        d, why = TI.decide(ctx, lambda p: '{"decision": "redo", "question": "How accurate are gradient-boosted trees against MLPs on tabular benchmarks?", "queries": ["xgboost vs mlp tabular", "tabular deep learning benchmark", "tree ensembles neural nets"], "reason": "off-topic sources"}')
        self.assertEqual((d["decision"], why), ("redo", None))
        d, why = TI.decide(ctx, lambda p: '{"decision": "redo", "question": "bad", "queries": []}')
        self.assertIsNone(d); self.assertIn("script check", why)
        d, why = TI.decide(ctx, lambda p: 'no json')
        self.assertIsNone(d)
        d, _ = TI.decide(ctx, lambda p: '{"decision": "escalate", "reason": "needs a key"}')
        self.assertEqual(d["decision"], "escalate")

    def test_controls_gate_moves_nothing(self):
        import tier as TI, tempfile
        with tempfile.TemporaryDirectory() as d:
            rc = TI.run("/nonexistent", d, ask=lambda p: '{"decision": "escalate", "reason": "x"}')   # control 1 wants redo
            rec = json.load(open(os.path.join(d, "tier.json")))
        self.assertEqual(rc, 0)
        self.assertEqual(rec["items"], [])
        self.assertEqual([c["got"] for c in rec["controls"]], ["escalate", "escalate"])

    def test_candidates(self):
        import tier as TI
        q = {"items": [{"id": "R1", "status": "escalated", "research": {"question": "q"}, "runs": [{"outcome": "no_claims"}]},
                       {"id": "R2", "status": "escalated", "research": {"question": "q"}, "runs": [{"outcome": "failed"}]},
                       {"id": "R3", "status": "open", "research": {"question": "q"}, "runs": [{"outcome": "no_claims"}]}]}
        self.assertEqual([i["id"] for i in TI.candidates(q)], ["R1"])

    def test_summarise(self):
        import tier as TI, tempfile
        with tempfile.TemporaryDirectory() as d:
            json.dump({"sources": [{"title": "TabPFN v2"}], "log": [{"api": "arxiv", "n": 3}]}, open(os.path.join(d, "search.json"), "w"))
            json.dump({"script_dropped": [[{"claim": "c1", "quote": "q1"}, "quote not found"]],
                       "script_kept": [{"claim": "c2", "quote": "q2"}], "check": {"keep": [], "decider": {"p_keep": [0.2]}}},
                      open(os.path.join(d, "read.json"), "w"))
            s = TI.summarise({"research": {"question": "Q?", "queries": ["a"]}}, d)
        self.assertIn("DROPPED by script (quote not found): c1", s["claims"])
        self.assertIn("DROPPED by checker (p=0.20): c2", s["claims"])
        self.assertEqual((s["n"], s["log"]), (1, "arxiv:3"))


class LoopMetrics(unittest.TestCase):
    def test_loop(self):
        q = {"items": [{"id": "A", "runs": [{"at": "2026-11-20T01:00:00Z", "worker": "worker-1", "outcome": "merged"},
                                             {"at": "2026-11-20T02:00:00Z", "worker": "tier-claude-haiku-5-5", "outcome": "redo"},
                                             {"at": "2026-11-19T01:00:00Z", "worker": "claude-shift-x", "outcome": "redo"}]}]}
        led = {"2026-11-20": {"anthropic": {"calls": 3, "usd": 0.006}, "nvidia": {"calls": 40}}}
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GH_TOKEN", None)
            out = digest.loop_metrics(q, "2026-11-20T00:00:00Z", led)
        self.assertIn("items landed (merged, confirmed, solved or decided done): 1", out[0])
        self.assertIn("Claude API tier 1, free worker 1", out[1])
        self.assertIn("$0.0060 ($0.0060 per landed item)", out[3])
        self.assertIn("not measured", out[4])


class GenerateControls(unittest.TestCase):
    def test_frozen_controls(self):
        import generate as G
        c = G.controls()
        self.assertEqual(len(c), 2)
        self.assertEqual(c[0][1], c[1][1])   # same item, a useful and an off-topic question
        self.assertNotIn("R5", repr(c))     # no live queue text in a control


class Diagnose(unittest.TestCase):
    """Rules written from the failed runs of 2026-11-19/20 (samples below are their log tails)."""
    def test_classes(self):
        import diagnose as DG
        tb = ('Traceback (most recent call last):\n  File "/home/runner/work/exagium/exagium/runner/proposals/propose.py", line 30, in families\n'
              "    doc = x\nIndexError: list index out of range\n##[error]Process completed with exit code 1.")
        self.assertEqual(DG.classify(tb), "code_bug")
        self.assertEqual(DG.signature(tb), "runner/proposals/propose.py:families IndexError")
        self.assertEqual(DG.classify("openrouter: cap reached (19 calls made or held; cap {'calls': 30})"), "budget")
        self.assertEqual(DG.classify("CHANGED or-decider-free: registered True, now False; cited by: x"), "fact_changed")
        self.assertEqual(DG.classify("particle-ai/particle-1:free is temporarily rate-limited upstream"), "rate_limited")
        self.assertEqual(DG.classify("Ran 3 tests\nFAILED (failures=1)"), "test_failed")
        self.assertEqual(DG.classify("something odd\n##[error]Process completed with exit code 1."), "unknown")

    def test_report(self):
        import diagnose as DG
        rows = {"runs": [{"class": "budget", "workflow": "rerank-arm"}, {"class": "code_bug", "workflow": "propose", "signature": "a.py:f E"},
                         {"class": "code_bug", "workflow": "propose", "signature": "a.py:f E"}]}
        out = DG.report(rows)
        self.assertEqual(out[0], "- 3 failed runs")
        self.assertIn("code_bug 2 (propose) -> escalated item; a.py:f E", out[1])


class DiagnoseMore(unittest.TestCase):
    def test_transient_and_auth(self):
        import diagnose as DG
        self.assertEqual(DG.classify("E: Failed to fetch x 503  Service Unavailable"), "transient")
        self.assertEqual(DG.classify("probe failed: HTTP Error 401: Unauthorized"), "auth")
        self.assertEqual(DG.error_tail("a\nb\n##[error]boom\ncleanup"), ["a", "b", "##[error]boom"])


class CodeJob(unittest.TestCase):
    def test_edits_and_paths(self):
        import code_job as C
        files = {"a.py": "x = 1\ny = 2\n"}
        ok, why = C.check_edits(files, [{"path": "a.py", "old": "y = 2", "new": "y = 3"}])
        self.assertEqual((len(ok), why), (1, []))
        self.assertEqual(C.apply_edits(files, ok)["a.py"], "x = 1\ny = 3\n")
        for bad in ([{"path": "b.py", "old": "x", "new": "y"}], [{"path": "a.py", "old": " = ", "new": "="}],
                    [{"path": "a.py", "old": "x = 1", "new": "x = 1"}], [], "no"):
            self.assertEqual(C.check_edits(files, bad)[0], [])
        for p in (".github/workflows/x.yml", "worker/ledger.py", "worker/code_job.py", "audit/q.json", "../x.py", "/etc/passwd", "my_secret.py"):
            self.assertFalse(C.allowed(p), p)
        self.assertTrue(C.allowed("worker/diagnose.py"))

    def test_job_type_and_cover(self):
        import queue_ops as QO
        i = {"code": {"files": ["worker/diagnose.py"], "test": "worker/test_worker.py"}, "touches": ["worker/diagnose.py", "worker/test_worker.py"]}
        self.assertEqual(QO.job_type(i), "code")
        self.assertTrue(QO.covered(i))
        self.assertFalse(QO.covered(dict(i, touches=[])))
        self.assertFalse(QO.covered(dict(i, touches=["worker/diagnose.py"])))   # the test file is written too, so it is touched

    def test_tests_failed_is_a_failure(self):
        import transitions as TR
        self.assertEqual(TR.classify({"result": "tests_failed"}, True, "none"), "failed")


class TierMerge(unittest.TestCase):
    def test_merge_gate(self):
        import tier as TI
        item = {"code": {"files": ["worker/diagnose.py"]}}
        ok, why = TI.merge_decision(item, [{"filename": "worker/diagnose.py", "additions": 5, "deletions": 0}], "approve")
        self.assertTrue(ok)
        self.assertFalse(TI.merge_decision(item, [{"filename": "worker/diagnose.py", "additions": 90, "deletions": 0}], "approve")[0])
        self.assertFalse(TI.merge_decision(item, [{"filename": "worker/diagnose.py", "additions": 1, "deletions": 0}], "reject")[0])
        for f in ("worker/tier.py", "router/slots.py", ".github/workflows/x.yml", "worker/other.py"):
            self.assertFalse(TI.merge_decision({"code": {"files": [f]}} if f != "worker/other.py" else item,
                                               [{"filename": f, "additions": 1, "deletions": 0}], "approve")[0], f)

    def test_code_prs(self):
        import tier as TI
        q = {"items": [{"id": "K1", "status": "review", "code": {"files": ["a"]}, "runs": [{"outcome": "pr", "where": "https://github.com/x/y/pull/9"}]},
                       {"id": "K2", "status": "review", "runs": [{"outcome": "pr", "where": "https://github.com/x/y/pull/10"}]}]}
        self.assertEqual([(i["id"], n) for i, n in TI.code_prs(q)], [("K1", 9)])


class TierRedoCap(unittest.TestCase):
    def test_count(self):
        import tier as TI
        i = {"runs": [{"worker": "tier-claude-haiku-5-5", "outcome": "redo"}, {"worker": "worker-1", "outcome": "no_claims"},
                      {"worker": "claude-shift-x", "outcome": "redo"}]}
        self.assertEqual(TI.tier_redos(i), 1)


class Split(unittest.TestCase):
    def test_check(self):
        import split as SP
        lines, tests = {"worker/diagnose.py": 160, "worker/ledger.py": 200, "worker/big.py": 900}, ["worker/test_worker.py"]
        ok, why = SP.check({"title": "Add classes() to diagnose", "files": ["worker/diagnose.py"], "test": "worker/test_worker.py",
                            "next": "Add a function that returns the class names."}, lines, tests)
        self.assertIsNone(why)
        for bad in ({"title": "Change the ledger caps now", "files": ["worker/ledger.py"], "test": "worker/test_worker.py", "next": "x" * 30},
                    {"title": "Edit a big file somewhere", "files": ["worker/big.py"], "test": "worker/test_worker.py", "next": "x" * 30},
                    {"title": "Add classes() to diagnose", "files": ["worker/diagnose.py"], "test": "worker/test_nope.py", "next": "x" * 30},
                    {"title": "Too many files here", "files": ["worker/diagnose.py"] * 3, "test": "worker/test_worker.py", "next": "x" * 30}):
            self.assertIsNone(SP.check(bad, lines, tests)[0], bad["title"])

    def test_candidates(self):
        import split as SP
        q = {"items": [{"id": "P1", "status": "open", "kind": "build", "value": "high"},
                       {"id": "P2", "status": "open", "kind": "build", "blocked_by": ["P9"]}, {"id": "P9", "status": "open", "kind": "doc"},
                       {"id": "P3", "status": "open", "kind": "build", "research": {"question": "q"}},
                       {"id": "K2", "status": "proposed", "split_from": "P4"}, {"id": "P4", "status": "open", "kind": "fix"}]}
        self.assertEqual([i["id"] for i in SP.candidates(q)], ["P1"])


class GeminiReview20261120(unittest.TestCase):
    """Reproductions of the Gemini review findings fixed on 2026-11-20 (F009-F012, F002)."""
    def test_f010_paths_and_overlap(self):
        import code_job as C
        for p in ("./.github/x.yml", "worker//ledger.py", "worker/./ledger.py", "a\\\\b.py"):
            self.assertFalse(C.allowed(p), p)
        files = {"a.py": "abc\n"}
        with self.assertRaises(ValueError):
            C.apply_edits(files, [{"path": "a.py", "old": "abc", "new": "xyz"}, {"path": "a.py", "old": "abc", "new": "q"}])

    def test_f009_no_edits_is_a_failure(self):
        import transitions as TR
        self.assertEqual(TR.classify({"result": "no_edits"}, True, "none"), "failed")

    def test_f002_reserve_additive(self):
        import ledger as LG, tempfile, subprocess as sp
        with tempfile.TemporaryDirectory() as wt:
            os.makedirs(os.path.join(wt, "ledger"))
            with unittest.mock.patch.object(LG, "sync", lambda w: None), unittest.mock.patch.object(LG, "push", lambda *a: True):
                LG.reserve(wt, "r1", "openrouter_paid:2:5")
                LG.reserve(wt, "r1", "openrouter_paid:2:5")
                led = LG.load(LG.day_path(wt))
        self.assertAlmostEqual(led["openrouter_paid"]["pending_usd"]["r1"], 0.02)


class CodeReplyFormat(unittest.TestCase):
    def test_blocks(self):
        import code_job as C
        t = ('Here you go\n<<<<<<< SEARCH worker/diagnose.py\ndef classify(log):\n=======\ndef classes():\n    return ["a{b}"]\n\n\ndef classify(log):\n'
             '>>>>>>> REPLACE\nNOTE: added classes()')
        r = C.parse_reply(t)
        self.assertEqual(r["edits"], [{"path": "worker/diagnose.py", "old": "def classify(log):",
                                       "new": 'def classes():\n    return ["a{b}"]\n\n\ndef classify(log):'}])
        self.assertEqual(r["note"], "added classes()")
        self.assertEqual(C.parse_reply('{"edits": [], "note": "x"}')["note"], "x")
        self.assertIsNone(C.parse_reply("nothing here"))


class SandboxRefused(unittest.TestCase):
    def test_sandbox_paths(self):
        import code_job as C
        for p in ("runner/jail.py", "runner/limits.c", "loop/seccomp_filter.h", "runner/count.py"):
            self.assertFalse(C.allowed(p), p)


class GLMReview20261120(unittest.TestCase):
    """Reproductions of the GLM review findings fixed on 2026-11-21 (F016, F017, F024, F036)."""

    def test_f018_repeated_lease_expiry_escalates(self):
        """A run killed mid-item every time reopened it forever; failures() counted it but nothing acted on the count."""
        import transitions as T
        t = dt.datetime(2026, 11, 21, tzinfo=dt.timezone.utc)
        old = "2026-11-20T00:00:00Z"
        i = {"id": "X1", "status": "in progress", "claimed_by": "worker-1", "claimed_at": old, "lease_until": old,
             "runs": [{"result": "lease expired", "outcome": "lease_expired"}] * (T.MAX_FAILURES - 1)}
        Q.expire({"items": [i]}, t)
        self.assertEqual(i["status"], "escalated")
        j = {"id": "X2", "status": "in progress", "claimed_by": "worker-1", "claimed_at": old, "lease_until": old, "runs": []}
        Q.expire({"items": [j]}, t)
        self.assertEqual(j["status"], "open")   # a first expiry still retries

    def test_f023_failed_log_fetch_is_not_evidence(self):
        """A failed log fetch read as an empty log: 'unknown' with no tail, filed as D23-D25."""
        import diagnose as DG
        from unittest import mock
        def fake_gh(*a):
            if "/logs" in a[0]:
                return None
            if "/jobs" in a[0]:
                return "77\n"
            return '[{"id": 1, "name": "worker", "created_at": "2026-11-20T00:00:00Z"}]'
        with tempfile.TemporaryDirectory() as d, mock.patch.object(DG, "gh", fake_gh):
            out = os.path.join(d, "diag.json")
            DG.scan(1, out)
            rows = json.load(open(out))["runs"]
            self.assertEqual([r["class"] for r in rows], ["log_unavailable"])
            self.assertEqual(DG.file(d, out), 0)   # nothing filed, no queue touched
        with mock.patch.object(DG.subprocess, "run", return_value=unittest.mock.Mock(returncode=1, stdout="")):
            self.assertIsNone(DG.gh("x"))

    def test_f025_no_verdict_is_a_failed_call(self):
        """A reviewer reply with no verdict was a 'rejected' change (escalated, no retry); no review happened."""
        import code_job as C
        with tempfile.TemporaryDirectory() as d:
            repo, out = os.path.join(d, "r"), os.path.join(d, "o"); os.makedirs(repo); os.makedirs(out)
            open(os.path.join(repo, "a.py"), "w").write("x = 1\n")
            json.dump({"files": {"a.py": "x = 1\n"}, "test": "t"}, open(os.path.join(out, "bundle.json"), "w"))
            json.dump({"proposal": {"edits": [{"path": "a.py", "old": "x = 1", "new": "x = 2"}]}, "review": {"reason": "?"}},
                      open(os.path.join(out, "draft.json"), "w"))
            json.dump({"passed": True}, open(os.path.join(out, "test.json"), "w"))
            C.apply(repo, out)
            self.assertEqual(json.load(open(os.path.join(out, "result.json")))["result"], "failed_call")

    def test_f033_bake_pins_each_reader(self):
        import reader_bake as RB
        from unittest import mock
        seen = []
        def fake_read(p):
            seen.append(os.environ.get("READER_OVERRIDE"))
            json.dump({}, open(os.path.join(p, "read.json"), "w")); open(os.path.join(p, "calls.jsonl"), "w").close()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(RB.rj, "read", fake_read), \
                mock.patch.object(RB.rj, "READER", RB.rj.READER), mock.patch.object(RB.rj, "CHECK_CALLS", RB.rj.CHECK_CALLS), \
                mock.patch.dict(os.environ, {}):   # main() sets module globals: restore them for the other tests
            b = os.path.join(d, "x", "b1"); os.makedirs(b); open(os.path.join(b, "search.json"), "w").write("{}")
            try:
                RB.main(os.path.join(d, "out"), [b], ["nvidia,m1", "cloudflare,m2"])
            except Exception:
                pass   # only the reader each read ran under matters here
        self.assertEqual(seen[:2], ["nvidia,m1", "cloudflare,m2"])

    def test_f021_missing_calls_log_releases_reservation(self):
        """A run that crashed before writing calls.jsonl held its reservation all UTC day (gen-44728562276: 4)."""
        import ledger as LG
        from unittest import mock
        with tempfile.TemporaryDirectory() as wt, mock.patch.object(LG, "sync"), \
                mock.patch.object(LG, "push", return_value=True) as push:
            p = LG.day_path(wt); os.makedirs(os.path.dirname(p))
            json.dump({"openrouter": {"reserved": 4, "calls": 0, "runs": [], "pending_calls": {"gen-1": 4}}}, open(p, "w"))
            self.assertEqual(LG.record(wt, "gen-1", os.path.join(wt, "missing.jsonl")), 0)
            self.assertEqual(json.load(open(p))["openrouter"]["pending_calls"], {})
            push.reset_mock()
            self.assertEqual(LG.record(wt, "gen-2", os.path.join(wt, "missing.jsonl")), 0)   # nothing held: no commit
            push.assert_not_called()
    def test_f036_exception_moves_down_the_chain(self):
        import slots
        seen = []

        def ask(pool, model, prompt):
            seen.append(pool)
            if pool == "groq":
                raise KeyError("GROQ_API_KEY")
            return {"ok": True, "status": 200, "text": '{"a": 1}'}
        got, last, tries = slots.run("drafter", "p", ask, lambda t: R.parse_json(t))
        self.assertEqual(got, {"a": 1})
        self.assertEqual(seen[:2], ["groq", "cloudflare"])
        self.assertIn("KeyError", tries and str(tries) or "")

    def test_f024_no_files_never_merges(self):
        import tier as TI
        ok, why = TI.merge_decision({"code": {"files": ["a.py"], "test": "t.py"}}, [], "approve")
        self.assertFalse(ok)
        ok, _ = TI.merge_decision({"code": {"files": ["a.py"], "test": "t.py"}}, None, "approve")
        self.assertFalse(ok)

    def test_f016_f017_expire(self):
        import queue_ops as QO
        t = QO.parse("2026-11-21T00:00:00Z")
        q = {"items": [{"id": "Q58", "status": "in progress"},
                       {"id": "P3", "status": "in progress", "claimed_by": "chat-x", "claimed_at": "2026-11-19T07:40:53Z", "touches": ["tasks/"]},
                       {"id": "P9", "status": "in progress", "claimed_by": "chat-y", "claimed_at": "2026-11-20T20:00:00Z", "touches": ["x/"]}]}
        self.assertEqual(sorted(QO.expire(q, t)), ["P3", "Q58"])
        self.assertEqual([i["status"] for i in q["items"]], ["open", "open", "in progress"])
        self.assertTrue(QO.live_claim(q["items"][2], t))



class IntakeReview(unittest.TestCase):
    """worker/intake_review.py: open review findings become code items, by script, once each."""
    def test_filters_and_dedupe(self):
        import intake_review as IR
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            rows = {"F1": "worker/digest.py", "F2": ".github/workflows/x.yml", "F3": "record/record.py", "F4": "worker/digest.py"}
            with open(os.path.join(d, "findings.jsonl"), "w") as f:
                for k, p in rows.items():
                    f.write(json.dumps({"id": k, "file": p, "lines": [1, 2], "body": {"kind": "bug", "claim": "c " + k, "evidence": "e"}}) + "\n")
            for n in ("fixes.jsonl", "rankings.jsonl"):
                open(os.path.join(d, n), "w").write("")
            open(os.path.join(d, "RESPONSE.md"), "w").write("## Fixed\n| F4 | high | abc1234 | |\n## Not changed\n## Open\n")
            items, skipped = IR.candidates(d)
            self.assertEqual([x["review"].split(":")[1] for x in items], ["F1"])
            self.assertEqual(items[0]["test"], "worker/test_worker.py")
            self.assertEqual(skipped, {"F2": "workflow or refused path", "F3": "a person merges this path", "F4": "closed"})
            q = {"items": [{"id": "G1", "review": items[0]["review"], "status": "open"}]}
            with mock.patch.object(IR.Q, "sync"), mock.patch.object(IR.Q, "load", return_value=q):
                self.assertEqual(IR.file_items(d, items), [])   # filed once already: nothing new



class NewFiles(unittest.TestCase):
    """Owner 2026-11-21: the worker builds as well as edits; an item may create one small file in an existing folder."""
    def test_code_job_creates_a_file(self):
        import code_job as C
        with tempfile.TemporaryDirectory() as d:
            repo, out = os.path.join(d, "r"), os.path.join(d, "o")
            os.makedirs(os.path.join(repo, "worker")); open(os.path.join(repo, "worker", "test_x.py"), "w").write("# t\n")
            item = {"id": "K99", "title": "a new module", "next": "write it", "touches": ["worker/nu.py", "worker/test_x.py"],
                    "code": {"files": ["worker/nu.py"], "new": ["worker/nu.py"], "test": "worker/test_x.py"}}
            json.dump({"item": item}, open(os.path.join(d, "item.json"), "w"))
            C.context(os.path.join(d, "item.json"), repo, out)
            b = json.load(open(os.path.join(out, "bundle.json")))
            self.assertEqual(b["files"]["worker/nu.py"], "")
            r = C.parse_reply("<<<<<<< SEARCH worker/nu.py\n=======\ndef f():\n    return 1\n>>>>>>> REPLACE\nNOTE: new")
            ok, why = C.check_edits(b["files"], r["edits"])
            self.assertEqual((len(ok), why), (1, []))
            self.assertEqual(C.apply_edits(b["files"], ok)["worker/nu.py"], "def f():\n    return 1\n")
            # an existing file still needs exactly-once old text, and creation is refused where it is not declared
            self.assertEqual(C.check_edits({"a.py": "x\n"}, [{"path": "a.py", "old": "", "new": "y"}])[0], [])
            for bad in (".github/x.py", "worker/nu.sh", "nofolder/x.py"):
                self.assertFalse(C.new_ok(repo, bad), bad)

    def test_split_allows_one_new_file_in_an_existing_folder(self):
        import split as SP
        lines, tests = {"worker/a.py": 10, "worker/test_worker.py": 50}, ["worker/test_worker.py"]
        base = {"title": "Add a small helper module", "test": "worker/test_worker.py", "next": "x" * 30}
        self.assertIsNotNone(SP.check(dict(base, files=["worker/nu.py"], new=["worker/nu.py"]), lines, tests)[0])
        self.assertIsNotNone(SP.check(dict(base, files=["worker/nu.py"], new=["worker/nu.py", "worker/test_nu.py"],
                                           test="worker/test_nu.py"), lines, tests)[0])   # with its own new tests file
        self.assertIsNone(SP.check(dict(base, files=["worker/a.py"], test="worker/test_nu.py"), lines, tests)[0])   # not declared new
        for files, new in ((["other/nu.py"], ["other/nu.py"]), (["worker/nu.py"], []), (["worker/b.py", "worker/c.py"], ["worker/b.py", "worker/c.py"]),
                           ([".github/x.py"], [".github/x.py"])):
            self.assertIsNone(SP.check(dict(base, files=files, new=new), lines, tests)[0], files)



class CoderRepair(unittest.TestCase):
    """2026-11-21: missing imports were the commonest code rejection; a static check and one repair round catch them."""
    def test_static_problems(self):
        import code_job as C
        f = {"a.py": "x = 1\n"}
        self.assertEqual(C.static_problems(f, {"a.py": "import json\ndef g(a):\n    return json.dumps([a for a in x])\nx = 1\n"}), [])
        self.assertIn("name 're' is used", C.static_problems(f, {"a.py": "def g():\n    return re.sub('a', 'b', 'c')\n"})[0])
        self.assertIn("syntax error", C.static_problems(f, {"a.py": "def g(:\n"})[0])

    def test_repair_round(self):
        import code_job as C, slots
        from unittest import mock
        bad = "<<<<<<< SEARCH a.py\nx = 1\n=======\nx = json.dumps(1)\n>>>>>>> REPLACE\nNOTE: n"
        good = "<<<<<<< SEARCH a.py\nx = 1\n=======\nimport json\nx = json.dumps(1)\n>>>>>>> REPLACE\nNOTE: n"
        replies = iter([bad, good])
        def fake_run(slot, prompt, ask, parse):
            if slot == "coder":
                t = next(replies)
                return parse(t), {"text": t}, [{"pool": "p"}]
            return {"verdict": "approve", "reason": "ok"}, {"text": ""}, []
        with tempfile.TemporaryDirectory() as d:
            json.dump({"item": {"id": "K1", "title": "t", "next": "n"}, "files": {"a.py": "x = 1\n"}, "prompt": "P"},
                      open(os.path.join(d, "bundle.json"), "w"))
            with mock.patch.object(slots, "run", fake_run), mock.patch.object(C, "arm", lambda i: "one"):
                C.draft(d)
            r = json.load(open(os.path.join(d, "draft.json")))
        self.assertEqual(r["static"], [])
        self.assertEqual(r["repair"]["after"], [])
        self.assertIn("import json", r["proposal"]["edits"][0]["new"])


class DigestBudgetWindow(unittest.TestCase):
    """F029: --hours other than 24 must read every day's ledger, not just the endpoints."""
    def test_f029_every_day_in_window(self):
        import digest as D
        from unittest import mock
        t = dt.datetime(2026, 11, 21, 12, 0, tzinfo=dt.timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            runs = os.path.join(d, "runs")
            os.makedirs(os.path.join(runs, "ledger"))
            for day in ("2026-11-19", "2026-11-20", "2026-11-21"):
                json.dump({"pool": {"usd": 1}}, open(os.path.join(runs, "ledger", day + ".json"), "w"))
            out = os.path.join(d, "out")
            with mock.patch.object(D.Q, "now", return_value=t), \
                    mock.patch.object(D.Q, "load", return_value={"items": []}), \
                    mock.patch.object(D, "queue_at", return_value=(None, "none")), \
                    mock.patch.object(D, "git", return_value=""), \
                    mock.patch.object(D, "metrics", return_value=[]), \
                    mock.patch.object(D, "loop_metrics", return_value=[]):
                D.facts(d, d, runs, out, 48)
            text = open(os.path.join(out, "facts.md")).read()
            self.assertIn("2026-11-20 pool", text)



class SecondDraft(unittest.TestCase):
    """Arm "two": the test job keeps an approved second draft that passes when the first does not."""
    def test_second_draft_chosen_when_first_fails(self):
        import code_job as C
        with tempfile.TemporaryDirectory() as d:
            repo, out = os.path.join(d, "r"), os.path.join(d, "o"); os.makedirs(os.path.join(repo, "pkg")); os.makedirs(out)
            open(os.path.join(repo, "pkg", "m.py"), "w").write("X = 1\n")
            open(os.path.join(repo, "pkg", "__init__.py"), "w").write("")
            t = "import unittest\nfrom pkg import m\nclass T(unittest.TestCase):\n    def test(self):\n        self.assertEqual(m.X, 2)\n"
            open(os.path.join(repo, "pkg", "test_m.py"), "w").write(t)
            files = {"pkg/m.py": "X = 1\n", "pkg/test_m.py": t}
            json.dump({"files": files, "test": "pkg/test_m.py"}, open(os.path.join(out, "bundle.json"), "w"))
            bad = {"edits": [{"path": "pkg/m.py", "old": "X = 1", "new": "X = 3"}]}
            good = {"edits": [{"path": "pkg/m.py", "old": "X = 1", "new": "X = 2"}]}
            json.dump({"proposal": bad, "review": {"verdict": "approve"}, "arm": "two",
                       "alt": {"proposal": good, "review": {"verdict": "approve"}}}, open(os.path.join(out, "draft.json"), "w"))
            from unittest import mock
            with mock.patch.object(C, "CI_TESTS", []):
                C.test(repo, out)
            r = json.load(open(os.path.join(out, "test.json")))
            self.assertEqual((r["chosen"], r["passed"]), ("second", True))
            open(os.path.join(repo, "pkg", "m.py"), "w").write("X = 1\n")   # the land job starts from the bundle's files
            C.apply(repo, out)
            res = json.load(open(os.path.join(out, "result.json")))
            self.assertEqual((res["result"], res["chosen"]), ("pr", "second"))
            self.assertEqual(open(os.path.join(repo, "pkg", "m.py")).read(), "X = 2\n")


if __name__ == "__main__":
    unittest.main()
