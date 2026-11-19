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

    def test_wilson(self):
        lo, hi = judge.wilson(25, 30)
        self.assertAlmostEqual(lo, 0.664, places=2)


if __name__ == "__main__":
    unittest.main()
