import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instr as I


class T(unittest.TestCase):
    L = {"x": {"yes": [0, 1], "no": [2, 3], "unsure": [4]}}

    def test_score_and_decide(self):
        a = {"x": [0.9, 0.0, 0.6, 0.1, 0.1, 0.1, 0.9]}           # A@0.5 keeps 1 of 2 yes, no false keeps
        b = {"x": [0.9, 0.0, 0.6, 0.6, 0.1, 0.1, 0.0]}           # B@0.5 keeps both yes
        sc = I.score({"A": a, "B": b}, self.L)
        self.assertEqual(sc["A@0.5"]["recall"], 0.5)
        self.assertEqual(sc["B@0.5"]["recall"], 1.0)
        self.assertEqual(sc["B@0.5"]["precision"], 1.0)
        self.assertEqual(I.decide(sc), "B@0.5")                   # an unsure source (index 4, p 0.9 in A) is not scored

    def test_precision_floor_and_controls(self):
        b = {"x": [0.9, 0.0, 0.6, 0.6, 0.6, 0.6, 0.0]}           # recall 1.0 but precision 0.5
        a = {"x": [0.9, 0.0, 0.6, 0.1, 0.1, 0.1, 0.0]}
        self.assertIsNone(I.decide(I.score({"A": a, "B": b}, self.L)))
        c = {"x": [0.4, 0.0, 0.6, 0.6, 0.1, 0.1, 0.0]}           # YES control fails at 0.5
        sc = I.score({"A": a, "B": c}, self.L)
        self.assertFalse(sc["B@0.5"]["controls"])

    def test_failed_request(self):
        sc = I.score({"A": {"x": None}}, self.L)
        self.assertFalse(sc["A@0.5"]["requests_ok"])
        self.assertIsNone(I.decide(sc))


if __name__ == "__main__":
    unittest.main()
