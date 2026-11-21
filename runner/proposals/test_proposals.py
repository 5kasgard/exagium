"""Offline tests for the P5 proposer's extractor and judge. Run: python3 -m unittest runner/proposals/test_proposals.py"""
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract  # noqa: E402
import judge  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
