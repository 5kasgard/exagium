import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R


class T(unittest.TestCase):
    def test_judge(self):
        ok = {"new": 0.8, "unrelated": 0.0}
        self.assertTrue(R.judge([ok, ok]))
        self.assertFalse(R.judge([ok, {"new": 0.49, "unrelated": 0.0}]))
        self.assertFalse(R.judge([ok, {"new": 0.9, "unrelated": 0.5}]))
        self.assertFalse(R.judge([ok, {"new": None}]))
        self.assertFalse(R.judge([]))

    def test_body_order(self):
        b, n = R.body("Q?", [{"title": "T", "abstract": "A"}])
        self.assertEqual(n, 4)
        lines = b["state"].split("SOURCES:\n")[1].splitlines()
        self.assertEqual(lines[0], "0. Q?")
        self.assertTrue(lines[1].startswith("1. Q? An empirical study"))
        self.assertEqual(lines[3], "3. T. A")
        self.assertEqual(set(b["questions"]), {"s0", "s1", "s2", "s3"})

    def test_prod_layout_matches_production(self):
        b, n = R.body("Q?", [{"title": "T", "abstract": "A"}], prod=True)
        self.assertEqual(n, 3)
        lines = b["state"].split("SOURCES:\n")[1].splitlines()
        self.assertEqual(lines[0], "0. " + R.RJ.REL_CONTROL_YES.format(q="Q?"))
        self.assertEqual(lines[1], "1. " + R.RJ.REL_CONTROL_NO[0])
        self.assertEqual(lines[2], "2. T. A")

    def test_production_uses_the_tested_text(self):
        self.assertEqual(R.NEW, R.RJ.REL_CONTROL_YES)


if __name__ == "__main__":
    unittest.main()
