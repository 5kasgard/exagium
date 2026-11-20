"""Walk the item state machine (queue Q60). Run: python3 -m unittest worker/test_transitions.py"""
import json, os, random, sys, unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import transitions as T  # noqa: E402

INTENDED_TERMINAL = {"done", "parked"}


class Table(unittest.TestCase):
    def test_every_allowed_pair_lands_in_a_state(self):
        for (s, o), to in T.TABLE.items():
            self.assertIn(s, T.STATES)
            self.assertIn(to, T.STATES)
            self.assertEqual(T.next_status(s, o), to)

    def test_forbidden_pairs_refused(self):
        outcomes = {o for _, o in T.TABLE}
        for s in T.STATES:
            for o in outcomes:
                if (s, o) not in T.TABLE:
                    with self.assertRaises(T.NotAllowed, msg=(s, o)):
                        T.next_status(s, o)

    def test_pipelines_from_the_2026_11_19_trace(self):
        # each worker outcome seen in the record, and where it must go
        self.assertEqual(T.next_status("in progress", "merged"), "done")
        self.assertEqual(T.next_status("in progress", "unchanged"), "review")         # E1, D16
        self.assertEqual(T.next_status("in progress", "guard_blocked"), "escalated")  # E5
        self.assertEqual(T.next_status("in progress", "no_claims"), "escalated")      # R4, R10, R11, R13
        self.assertEqual(T.next_status("in progress", "pr"), "review")                # Q38
        self.assertEqual(T.next_status("in progress", "budget"), "open")
        self.assertEqual(T.next_status("in progress", "failed", 1), "open")
        self.assertEqual(T.next_status("in progress", "failed", T.MAX_FAILURES), "escalated")

    def test_worker_never_reaches_the_owner_directly(self):
        for o in T.WORKER:
            for f in range(0, T.MAX_FAILURES + 2):
                self.assertNotEqual(T.next_status("in progress", o, f), "waiting on the owner", o)

    def test_worker_outcomes_only_from_in_progress(self):
        for o in T.WORKER:
            for s in T.STATES:
                if s != "in progress":
                    with self.assertRaises(T.NotAllowed):
                        T.next_status(s, o)


class Classify(unittest.TestCase):
    """Outcomes from what doc_job.apply and research_job.card really write (review of Q60, 11), for each case seen on
    2026-11-19 plus the infrastructure failures the review named."""
    TEXT = "one\n5. A verdict can never be removed by anyone.\nthree\n"

    def doc(self, edits, verdict=None, model_fail=(), files_changed=False):
        import tempfile, doc_job as D
        with tempfile.TemporaryDirectory() as repo, tempfile.TemporaryDirectory() as out:
            os.makedirs(os.path.join(repo, "docs"))
            open(os.path.join(repo, "docs/a.md"), "w").write(self.TEXT + ("x" if files_changed else ""))
            item = {"id": "X1", "title": "t", "next": "n", "targets": [{"path": "docs/a.md", "lines": [2, 2]}]}
            json.dump({"item": item, "files": {"docs/a.md": self.TEXT}}, open(os.path.join(out, "bundle.json"), "w"))
            json.dump({"proposal": {"edits": edits, "note": "n"}, "review": verdict and {"verdict": verdict, "reason": "r"},
                       "pre_check": [], "model_fail": list(model_fail)}, open(os.path.join(out, "draft.json"), "w"))
            D.apply(repo, out)
            res = json.load(open(os.path.join(out, "result.json")))
            changed = open(os.path.join(repo, "docs/a.md")).read() != self.TEXT
            return T.classify(res, not changed, "none")

    def edit(self, line, new):
        return {"path": "docs/a.md", "line": line, "old": self.TEXT.split("\n")[line - 1], "new": new}

    def test_doc_cases(self):
        self.assertEqual(self.doc([]), "unchanged")                                                        # E1
        self.assertEqual(self.doc([self.edit(2, "5. A verdict cannot be removed.")], "approve"), "guard_blocked")   # E5
        self.assertEqual(self.doc([self.edit(3, "THREE")], "approve"), "guard_blocked")                     # D16: outside target
        self.assertEqual(self.doc([self.edit(2, "5. A verdict can never be erased by anyone.")], "reject"), "guard_blocked")
        self.assertEqual(self.doc([], model_fail=["drafter call failed"]), "failed")                         # review 1
        self.assertEqual(self.doc([], files_changed=True), "retry")                                          # review 4
        bad_path = {"path": "no accepted edit", "line": 2, "old": "x", "new": "y"}
        self.assertEqual(self.doc([bad_path], "approve"), "guard_blocked")                                   # review 3

    def card(self, read, exists=False):
        import tempfile, research_job as R
        with tempfile.TemporaryDirectory() as repo, tempfile.TemporaryDirectory() as out:
            item = {"id": "R99", "touches": ["research/cards/R99.md"], "research": {"question": "q"}}
            json.dump({"item": item, "sources": [], "log": []}, open(os.path.join(out, "search.json"), "w"))
            json.dump(dict({"reading": {"claims": []}, "script_kept": [], "script_dropped": [], "check": None}, **read),
                      open(os.path.join(out, "read.json"), "w"))
            if exists:
                os.makedirs(os.path.join(repo, "research/cards")); open(os.path.join(repo, "research/cards/R99.md"), "w").write("x")
            R.card(repo, out)
            res = json.load(open(os.path.join(out, "result.json")))
            return T.classify(res, True, "none")

    def test_card_cases(self):
        self.assertEqual(self.card({"model_fail": []}), "no_claims")                      # R10, R11, R13 if calls were fine
        self.assertEqual(self.card({"model_fail": ["reader call failed"]}), "failed")     # review 1
        self.assertEqual(self.card({"model_fail": []}, exists=True), "unchanged")         # review 4: duplicate -> review

    def test_landed_and_unknown(self):
        self.assertEqual(T.classify({"result": "merge"}, False, "merged"), "merged")
        self.assertEqual(T.classify({"result": "merge", "edits": 2}, True, "none"), "failed")   # review 2
        self.assertEqual(T.classify({"result": "???"}, True, "none"), "failed")


class RandomWalks(unittest.TestCase):
    def test_random_event_sequences_stay_in_states_and_can_always_end(self):
        rnd = random.Random(20261120)
        outcomes = sorted({o for _, o in T.TABLE})
        for _ in range(2000):
            s, fails = "open", 0
            for _ in range(30):
                o = rnd.choice(outcomes)
                fails = fails + 1 if o == "failed" else 0
                try:
                    s = T.next_status(s, o, fails)
                except T.NotAllowed:
                    continue
                self.assertIn(s, T.STATES)
            self.assertIn(T.next_status(s, "decide_done"), INTENDED_TERMINAL)

    def test_automated_tiers_reach_an_end_without_the_owner_settling(self):
        # review of Q60, 11: from every state the worker or a tier can leave, some automated path reaches done or the owner
        auto = set(T.WORKER) | {"claim", "confirm", "redo", "escalate", "solved"}
        ends = {"done", "waiting on the owner"}
        for start in ("open", "in progress", "review", "escalated"):
            seen, todo = set(), [start]
            while todo:
                x = todo.pop()
                if x in seen:
                    continue
                seen.add(x)
                todo += [to for (s, o), to in T.TABLE.items() if s == x and o in auto]
            self.assertTrue(seen & ends, start)


if __name__ == "__main__":
    unittest.main()


class WorkflowUsesOnlyKnownOutcomes(unittest.TestCase):
    def test_outcomes_in_worker_yml(self):
        import re
        wf = open(os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "worker.yml")).read()
        lit = set(re.findall(r'OUTCOME=([a-z_]+)"', wf)) | set(re.findall(r'"\$ITEM" ([a-z_]+) "', wf))
        self.assertTrue(lit, "no outcomes found: the pattern is stale")
        self.assertTrue(lit <= set(T.WORKER), lit - set(T.WORKER))
        self.assertIn("transitions.py classify", wf)


class CheckerMajority(unittest.TestCase):
    """research_job._read keeps a claim only if a strict majority of checker calls keep it (Q60 escalation of R10-R13)."""
    def run_read(self, checker_replies, n=3):
        import tempfile, research_job as R
        abstract = "We measured a forty percent drop in errors when the model was retrained on fresh data each week."
        src = [{"sid": "s1", "kind": "paper", "via": "x", "title": "t", "year": 2025, "abstract": abstract}]
        claims = [{"source": "s1", "claim": f"Retraining weekly cut errors (variant {'abcdefgh'[k]}).", "evidence": "reported",
                   "quote": "a forty percent drop in errors when the model was retrained"} for k in range(n)]
        replies = iter([json.dumps({"claims": claims, "open_questions": []})] + checker_replies)

        def ask(engine, model, prompt):
            t = next(replies)
            return {"ok": t is not None, "text": t or "", "error": None if t else "down"}
        with tempfile.TemporaryDirectory() as out:
            d = {"item": {"research": {"question": "q"}}, "sources": src}
            R._read(out, d, "q", [], ask)
            return json.load(open(os.path.join(out, "read.json")))

    def test_majority(self):
        rd = self.run_read([json.dumps({"keep": []}), json.dumps({"keep": [0, 1, 2]}), json.dumps({"keep": [0, 1]})])
        self.assertEqual(rd["check"]["keep"], [0, 1])     # the R9 pattern: claim 2 kept by 1 of 3 is dropped
        self.assertEqual(rd["model_fail"], [])

    def test_too_few_usable_replies_is_a_failure(self):
        rd = self.run_read([None, "not json", json.dumps({"keep": [0]})])
        self.assertEqual(rd["check"]["keep"], None)
        self.assertTrue(rd["model_fail"])

    def test_chunks_of_three_each_with_its_own_majority(self):
        j = json.dumps
        rd = self.run_read([j({"keep": [0, 2]}), j({"keep": [0]}), j({"keep": [0, 2]}),     # chunk 0-2: keeps 0, 2
                            j({"keep": [1]}), j({"keep": []}), j({"keep": [1]})], n=5)       # chunk 3-4: keeps 4
        self.assertEqual(rd["check"]["keep"], [0, 2, 4])

    def test_at_most_six_claims_checked(self):
        j = json.dumps
        rd = self.run_read([j({"keep": [0]})] * 6, n=8)
        self.assertEqual(len(rd["script_kept"]), 6)
        self.assertEqual(rd["check"]["keep"], [0, 3])

    def test_decider_decides_and_no_shadow_runs(self):
        """2026-11-20: the Nemotron majority runs only when the decision model fails (or with SHADOW=1)."""
        import research_job as R
        j = json.dumps
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "x"}), \
                mock.patch.object(R, "decide_claims", lambda q, kept, calls: {"keep": [1], "p_keep": [0.2, 0.9, 0.1]}):
            rd = self.run_read([j({"keep": [0, 1, 2]})] * 3)
        self.assertEqual(rd["check"]["keep"], [1])
        self.assertNotIn("shadow_nemotron", rd["check"])
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "x", "SHADOW": "1"}), \
                mock.patch.object(R, "decide_claims", lambda q, kept, calls: {"keep": [1], "p_keep": [0.2, 0.9, 0.1]}):
            rd = self.run_read([j({"keep": [0, 1, 2]})] * 3)
        self.assertEqual(rd["check"]["shadow_nemotron"]["keep"], [0, 1, 2])

    def test_decider_failure_falls_back_to_the_majority(self):
        import research_job as R
        j = json.dumps
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "x"}), \
                mock.patch.object(R, "decide_claims", lambda q, kept, calls: {"keep": None, "error": "status 429"}):
            rd = self.run_read([j({"keep": [0, 2]})] * 3)
        self.assertEqual(rd["check"]["keep"], [0, 2])
        self.assertEqual(rd["check"]["decider_error"], "status 429")


class Quota(unittest.TestCase):
    def test_quota_is_a_budget_wait_not_a_failure(self):
        self.assertEqual(T.classify({"result": "quota", "why": ["quota: reader call failed"]}, True, "none"), "budget")
        self.assertEqual(T.next_status("in progress", "budget"), "open")

