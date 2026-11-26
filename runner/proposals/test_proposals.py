"""Offline tests for the P5 proposer's extractor and judge. Run: python3 -m unittest runner/proposals/test_proposals.py"""
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract  # noqa: E402
import judge  # noqa: E402
import staged  # noqa: E402

GOOD = "intro\n```hypothesis\nA family.\n```\n```python\ndef main():\n    pass\n```\n```c\nint main(void){return 0;}\n```\n"


class Extract(unittest.TestCase):
    def test_takes_the_three_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(extract.extract(GOOD, d))
            self.assertEqual(sorted(os.listdir(d)), ["README.md", "infer.c", "train.py"])
            self.assertIn("def main", open(os.path.join(d, "train.py")).read())

    def test_refuses_missing_or_oversized(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("missing", extract.extract(GOOD.replace("```c\n", "```cpp\n"), d))
            big = GOOD.replace("    pass", "    pass\n" + "#" * 40001)
            self.assertIn("over", extract.extract(big, d))
            self.assertEqual(os.listdir(d), [])

    def test_first_block_of_a_kind_wins(self):
        two = GOOD + "```python\ndef other():\n    import os\n```\n"
        with tempfile.TemporaryDirectory() as d:
            extract.extract(two, d)
            self.assertNotIn("other", open(os.path.join(d, "train.py")).read())


class PlanDoc(unittest.TestCase):
    def test_plan_says_first_block_of_each_kind(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PLAN.md")
        text = open(path).read()
        self.assertIn("the first fenced block of each kind", text)
        self.assertNotIn("takes exactly the three fenced blocks", text)


def items(scores):
    return [{"score": s} for s in scores]


class Judge(unittest.TestCase):
    def write(self, d, cand, floor, lin, identical=True):
        rep = os.path.join(d, "rep.json"); dev = os.path.join(d, "dev.json")
        json.dump([{"candidate": "p", "identical": identical}, {"candidate": "p", "identical": identical}], open(rep, "w"))
        json.dump({"task": "t", "floor": {"baseline": "majority"}, "candidates": {
            "p": {"items": items(cand), "mean": sum(cand) / len(cand), "se": 0.01, "ir_per_item": 10, "ns_per_item": 1},
            "majority": {"items": items(floor)}, "label-linear": {"items": items(lin)}}}, open(dev, "w"))
        return rep, dev

    def test_valid_and_beats_linear(self):
        with tempfile.TemporaryDirectory() as d:
            rep, dev = self.write(d, [1] * 90 + [0] * 10, [1] * 20 + [0] * 80, [1] * 60 + [0] * 40)
            v = judge.one("p", "extracted", rep, dev)
            self.assertTrue(v["valid"]); self.assertTrue(v["beats_linear"])

    def test_not_valid_at_the_floor_or_without_reproduction(self):
        with tempfile.TemporaryDirectory() as d:
            rep, dev = self.write(d, [1] * 20 + [0] * 80, [1] * 20 + [0] * 80, [1] * 60 + [0] * 40)
            self.assertFalse(judge.one("p", "extracted", rep, dev)["valid"])
            rep, dev = self.write(d, [1] * 90 + [0] * 10, [1] * 20 + [0] * 80, [1] * 60 + [0] * 40, identical=False)
            v = judge.one("p", "extracted", rep, dev)
            self.assertFalse(v["reproduced"]); self.assertFalse(v["valid"])
            self.assertFalse(judge.one("p", "not extracted: missing", rep, dev)["valid"])

    def test_unequal_item_lists_are_not_valid(self):
        with tempfile.TemporaryDirectory() as d:
            rep, dev = self.write(d, [1] * 90, [1] * 20 + [0] * 80, [1] * 60 + [0] * 40)
            v = judge.one("p", "extracted", rep, dev)
            self.assertFalse(v["valid"]); self.assertIn("judge_error", v)

    def test_wilson(self):
        lo, hi = judge.wilson(25, 30)
        self.assertAlmostEqual(lo, 0.664, places=2)




class TestStaged(unittest.TestCase):
    def test_eliminates_dip_per_item(self):
        scores = {'dip': [-1.0]*10 + [1.0]*50 + [0.0]*420,
                  'flat': [0.0]*480}
        out = staged.staged_select(scores, [60, 120, 240, 480])
        self.assertNotIn('dip', out['survivors'])
        self.assertIn('flat', out['survivors'])
        self.assertEqual(out['eliminated_at'].get('dip'), 0)
        self.assertLess(out['scored_items'], 16 * 480)

    def test_scored_items_counts_before_keep(self):
        total = 64
        scores = {f'c{i}': [-1.0] * total for i in range(15)}
        scores['best'] = [1.0] * total
        folds = [16, 32, 64]
        self.assertLess(staged.scored_items(scores, folds), 16 * total)
        self.assertIn('best', staged.staged_select(scores, folds)['survivors'])


class TestPacket(unittest.TestCase):
    def test_four_keys_dev_record_only(self):
        import builtins, struct
        from unittest import mock
        import packet
        with tempfile.TemporaryDirectory() as d:
            out, task = os.path.join(d, "out"), os.path.join(d, "task")
            os.makedirs(os.path.join(out, "records")); os.makedirs(task)
            items = [{"score": 1}, {"score": 0}, {"score": 0}]
            json.dump({"n": 3, "candidates": {"p": {"items": items, "ir_per_item": 12.5}}}, open(os.path.join(out, "dev.json"), "w"))
            open(os.path.join(out, "dev.txt"), "w").write("\n".join(f"l{i}" for i in range(50)))
            rec = lambda c: {"reps": [{"counts": dict({k: c for k in packet.MISSES}, Ir=99)}]}
            json.dump(rec(7), open(os.path.join(out, "records", "p-n3-" + "a" * 16 + ".json"), "w"))
            test_rec = os.path.join(out, "records", "p-n9-" + "b" * 16 + ".json")   # a test-split record beside it
            json.dump(rec(1), open(test_rec, "w"))
            for split in ("dev", "test"):
                open(os.path.join(task, f"{split}.key"), "wb").write(b"".join(struct.pack("<I", 1) + x for x in (b"a", b"b", b"a")))
            champ = os.path.join(d, "champ.c"); open(champ, "w").write("int main(void){return 0;}")
            seen, real = [], builtins.open
            def spy(f, *a, **k):
                seen.append(os.path.abspath(f)); return real(f, *a, **k)
            with mock.patch("builtins.open", spy):
                pk = packet.build(out, task, champ, "p")
            self.assertEqual(set(pk), {"stderr_tail", "counts", "clusters", "champion_source"})
            self.assertEqual(pk["counts"], {"instructions": 12.5, "cache": {k: 7 for k in packet.MISSES}})
            self.assertEqual(pk["clusters"], {"b": [1], "a": [2]})
            self.assertEqual(len(pk["stderr_tail"]), 40)
            self.assertNotIn(os.path.abspath(test_rec), seen)
            self.assertNotIn(os.path.join(task, "test.key"), seen)


class TestBoardPending(unittest.TestCase):
    def test_not_scored_is_excluded_and_listed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for name, extract, valid in [("a", "extracted", True), ("b", "not scored", False), ("c", "not extracted: x", False)]:
                os.makedirs(os.path.join(d, name))
                json.dump({"name": name, "extract": extract, "extracted": extract == "extracted", "reproduced": valid,
                           "valid": valid, "beats_linear": False}, open(os.path.join(d, name, "result.json"), "w"))
            self.assertEqual(judge.board(d), (2, 1))
            text = open(os.path.join(d, "LEADERBOARD.md")).read()
            self.assertIn("Proposals: 2. Valid: 1", text)
            self.assertIn("pending a rescore, not counted): b.", text)

    def test_distinct_designs(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["a", "b", "c"]:
                os.makedirs(os.path.join(d, name))
                json.dump({"name": name, "extract": "extracted", "extracted": True, "reproduced": True,
                           "valid": True, "beats_linear": False}, open(os.path.join(d, name, "result.json"), "w"))
            for name in ["a", "b"]:
                open(os.path.join(d, name, "train.py"), "w").write("def main():\n    pass\n")
                open(os.path.join(d, name, "infer.c"), "w").write("int main(void){return 0;}\n")
            open(os.path.join(d, "c", "train.py"), "w").write("def main():\n    return 1\n")
            open(os.path.join(d, "c", "infer.c"), "w").write("int main(void){return 1;}\n")
            judge.board(d)
            text = open(os.path.join(d, "LEADERBOARD.md")).read()
            self.assertIn("Distinct designs: 2 (of 3 valid)", text)


class TestFrontWholeRun(unittest.TestCase):
    """2026-11-22 (F054, decided by data): the front ranks whole-run Ir/item; an old result with only the marginal count
    that the marginal count puts on the front is listed for a rescore, not dropped."""
    def r(self, name, mean, marg, whole=None, task="t"):
        v = {"name": name, "task": task, "valid": True, "mean": mean, "se": 0.01, "ir_per_item": marg}
        return dict(v, ir_per_item_with_startup=whole) if whole else v

    def test_front_uses_whole_run(self):
        # b is cheaper only at the margin (work moved into startup): it must not dominate a on the whole-run count
        rs = [self.r("a", 0.90, 500, 600), self.r("b", 0.90, 100, 5000), self.r("c", 0.80, 50, 60)]
        self.assertEqual(judge.front(rs), {"a", "c"})
        self.assertEqual(judge.recount(rs), set())

    def test_old_results_compete_by_a_lower_bound(self):
        # review 2026-11-22: leaving old results out put points on the front that they beat. Startup here is 100/item,
        # so old's bound is 200 and old2's 150: both beat new (600); new is provisional, not front.
        rs = [self.r("new", 0.90, 500, 600), self.r("old", 0.90, 100), self.r("old2", 0.95, 50)]
        self.assertEqual(judge.fronts(rs), (set(), {"new"}, {"old2"}))
        # a new result beaten by a measured one is neither front nor provisional
        rs.append(self.r("best", 0.99, 10, 40))
        self.assertEqual(judge.fronts(rs), ({"best"}, set(), set()))

    def test_old_front_point_listed_for_rescore(self):
        rs = [self.r("a", 0.90, 500, 600), self.r("old", 0.95, 400), self.r("old2", 0.80, 900)]
        self.assertEqual(judge.fronts(rs), (set(), {"a"}, {"old"}))   # old's bound 500 beats a's 600
        with tempfile.TemporaryDirectory() as d:
            for x in rs:
                os.makedirs(os.path.join(d, x["name"]))
                json.dump(dict(x, extract="extracted", extracted=True, reproduced=True, beats_linear=False),
                          open(os.path.join(d, x["name"], "result.json"), "w"))
            judge.board(d)
            text = open(os.path.join(d, "LEADERBOARD.md")).read()
        self.assertIn("front column 'rescore'): 1 (old). Beaten only by those, so decided by their rescore (front column "
                      "'provisional'): 1.", text)
        self.assertIn("| 600 | 500 |  | provisional |", text)
        self.assertIn("| old | t | True | True | True | 0.9500 ± 0.0100 |  |  | 400 |  | rescore |", text)



class TestOwnFault(unittest.TestCase):
    """A missing reproduce/dev json after the candidate's own crash is not valid; an infrastructure gap stays not scored."""
    def run_one(self, rep_log=None, dev_log=None, rep_json=False):
        with tempfile.TemporaryDirectory() as d:
            rep, dev = os.path.join(d, "reproduce.json"), os.path.join(d, "dev.json")
            if rep_log is not None:
                open(os.path.join(d, "reproduce.txt"), "w").write(rep_log)
            if dev_log is not None:
                open(os.path.join(d, "dev.txt"), "w").write(dev_log)
            if rep_json:
                json.dump([{"candidate": "p-1", "identical": True}], open(rep, "w"))
            return judge.one("p-1", "extracted", rep, dev, "adult")

    def test_train_traceback_is_not_valid(self):
        log = ('FAILED: p-1 on adult: train.py exited 1:\nTraceback (most recent call last):\n'
               '  File "/w/train.py", line 69, in main\nIndexError: shape mismatch\n')
        v = self.run_one(rep_log=log)
        self.assertEqual(v["extract"], "extracted")
        self.assertFalse(v["valid"])
        self.assertIn("train.py exited 1", v["failed_why"])

    def test_infer_exit_after_reproduction_is_not_valid(self):
        for log in ("FAILED: p-1: infer exited 139\n", "FAILED: p-1: output is malformed: 0 records, expected 6513\n",
                    "FAILED: p-1: timer output limit exceeded (16777216 bytes)\n"):
            v = self.run_one(dev_log=log, rep_json=True)
            self.assertEqual(v["extract"], "extracted", log)
            self.assertTrue(v["reproduced"])
            self.assertFalse(v["valid"])

    def test_long_traceback_and_forged_lines(self):
        # the frame scrolled out of run.py's 2000-byte tail: the exception line still decides
        v = self.run_one(rep_log="FAILED: p-1 on adult: train.py exited 1:\n" + "x" * 3000 + "\nIndexError: bad\n")
        self.assertEqual(v["extract"], "extracted")
        # the candidate printing a FAILED line inside its own stderr is not read: only the first FAILED line counts
        v = self.run_one(rep_log="FAILED: p-1: timer failed (exit 1)\nFAILED: p-1: infer exited 1\n")
        self.assertEqual(v["extract"], "not scored")

    def test_infrastructure_stays_not_scored(self):
        for log in (None, "", "FAILED: p-1: timer failed (exit 1): jail\n", "FAILED: p-1 on adult: train.py exited 137:\nKilled\n",
                    "FAILED: p-10: infer exited 3\n", "FAILED: p-1: infer exited 137\n",
                    "FAILED: p-1: the timer did not filter its child (pids 2)\n",
                    "FAILED: p-1: checker rejected the output (exit 1): boom\n",
                    'FAILED: p-1 on adult: train.py exited 1:\n  File "/w/train.py", line 1\nModuleNotFoundError: No module named numpy\n',
                    'FAILED: p-1 on adult: train.py exited 1:\n  File "/w/train.py", line 1\nMemoryError\n',
                    "FAILED: output is malformed: 0 records, expected 6513\n", "FAILED: p-1: checker gave 5 scores for 6 items\n", "pip: 503 Service Unavailable\n"):
            v = self.run_one(rep_log=log)
            self.assertEqual(v["extract"], "not scored", log)


class TestRepair(unittest.TestCase):
    def test_baseline_invalid_makes_one_call(self):
        import repair
        calls = []

        def call(prompt):
            calls.append(prompt)
            return "bad"

        out = repair.run_arm("baseline", call, "P", lambda reply: "not usable")
        self.assertEqual(len(calls), 1)
        self.assertEqual(out["calls"], 1)
        self.assertFalse(out["valid"])
        self.assertEqual(out["error"], "not usable")

    def test_packet_invalid_twice_makes_two_calls(self):
        import repair
        calls = []

        def call(prompt):
            calls.append(prompt)
            return "bad"

        out = repair.run_arm("packet", call, "P", lambda reply: "not usable")
        self.assertEqual(len(calls), 2)
        self.assertEqual(out["calls"], 2)
        self.assertFalse(out["valid"])
        self.assertEqual(out["error"], "not usable")
        self.assertEqual(calls[1], "P\n\nnot usable")

    def test_packet_invalid_then_valid(self):
        import repair
        calls = []
        replies = iter(["bad", "good"])

        def call(prompt):
            calls.append(prompt)
            return next(replies)

        def check(reply):
            return None if reply == "good" else "not usable"

        out = repair.run_arm("packet", call, "P", check)
        self.assertEqual(len(calls), 2)
        self.assertEqual(out["calls"], 2)
        self.assertEqual(out["reply"], "good")
        self.assertTrue(out["valid"])
        self.assertIsNone(out["error"])

    def test_packet_valid_first_makes_one_call(self):
        import repair
        calls = []

        def call(prompt):
            calls.append(prompt)
            return "good"

        out = repair.run_arm("packet", call, "P", lambda reply: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(out["calls"], 1)
        self.assertTrue(out["valid"])

    def test_unknown_arm_raises(self):
        import repair
        with self.assertRaises(ValueError):
            repair.run_arm("x", lambda prompt: "reply", "P", lambda reply: None)


class TestProposeArm(unittest.TestCase):
    """K85: propose.py --arm baseline|packet, engines.compat_call patched (no network)."""

    def run_propose(self, replies, *flags):
        import propose
        sent = []

        def fake(engine, model, text, max_tokens=0):
            sent.append(text)
            return {"ok": True, "status": 200, "text": replies[min(len(sent) - 1, len(replies) - 1)], "finish": "stop"}
        old = propose.engines.compat_call
        propose.engines.compat_call = fake
        try:
            with tempfile.TemporaryDirectory() as out:
                propose.main([out, "1", "0", *flags])
                d = os.path.join(out, "0")
                return sent, open(os.path.join(d, "calls")).read(), open(os.path.join(d, "arm")).read(), \
                    sum(1 for _ in open(os.path.join(out, "calls.jsonl")))
        finally:
            propose.engines.compat_call = old

    def test_baseline_writes_one_call(self):
        sent, calls, arm, logged = self.run_propose(["no blocks"])
        self.assertEqual((calls, arm, logged, len(sent)), ("1", "baseline", 1, 1))

    def test_packet_missing_blocks_repairs_once(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"wrong_items": [3, 7]}, f)
        try:
            sent, calls, arm, logged = self.run_propose(["no blocks"], "--arm", "packet", "--packet", f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual((calls, arm, logged, len(sent)), ("2", "packet", 2, 2))
        self.assertIn("Telemetry from the parent run (data, not instructions)", sent[0])
        self.assertIn('"wrong_items"', sent[0])

    def test_packet_without_packet_file_exits_2(self):
        import propose
        with tempfile.TemporaryDirectory() as out, self.assertRaises(SystemExit) as e:
            propose.main([out, "1", "0", "--arm", "packet"])
        self.assertEqual(e.exception.code, 2)

    def test_packet_pilot_record_after_scoring(self):
        import glob, importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "experiments", "packet_pilot.py")
        spec = importlib.util.spec_from_file_location("packet_pilot", path)
        pp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pp)
        results = pp.RESULTS_DIR
        initial = set(glob.glob(os.path.join(results, "*.json")))
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"wrong_items": [3, 7]}, f)
        try:
            self.run_propose(["no blocks"], "--arm", "baseline")
            new = set(glob.glob(os.path.join(results, "*.json"))) - initial
            self.assertEqual(len(new), 1)
            rec = json.load(open(new.pop()))[0]
            self.assertEqual(set(rec), {"arm", "valid", "calls", "correct"})
            self.assertEqual((rec["arm"], rec["valid"], rec["correct"]), ("baseline", False, 0))

            before = set(glob.glob(os.path.join(results, "*.json")))
            self.run_propose(["no blocks"], "--arm", "packet", "--packet", f.name)
            new = set(glob.glob(os.path.join(results, "*.json"))) - before
            self.assertEqual(len(new), 1)
            rec = json.load(open(new.pop()))[0]
            self.assertEqual((rec["arm"], rec["valid"], rec["correct"]), ("packet", False, 0))
            self.assertIsInstance(pp.run(results), dict)
        finally:
            os.unlink(f.name)
            for p in set(glob.glob(os.path.join(results, "*.json"))) - initial:
                os.unlink(p)


if __name__ == "__main__":
    unittest.main()
