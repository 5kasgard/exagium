"""Offline test of P5b's decision rules on synthetic results."""
import io, json, os, sys, tempfile, unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(__file__))
import decide  # noqa: E402


def cand(scores, ir):
    m = sum(scores) / len(scores)
    return {"items": [{"score": s} for s in scores], "mean": m, "se": 0.01, "ir_per_item": ir, "ns_per_item": ir // 10}


class Rules(unittest.TestCase):
    def run_it(self, props, drop=None):
        n = 400
        base = {"label-gbdt": cand([1] * 380 + [0] * 20, 40000), "label-mlp16": cand([1] * 376 + [0] * 24, 8700),
                "label-linear": cand([1] * 360 + [0] * 40, 5000)}
        r = dict(base, **{p: props.get(p, cand([1] * 370 + [0] * 30, 20000)) for p in decide.PROPS})
        if drop:
            r.pop(drop)
        with tempfile.TemporaryDirectory() as d:
            cp, bp = os.path.join(d, "c.json"), os.path.join(d, "b.md")
            json.dump({"candidates": r}, open(cp, "w"))
            open(bp, "w").write("".join(f"| {p} | optdigits | True | True | True | 0.9864 ± 0.0037 | x | 1 | 1 |\n" for p in decide.PROPS))
            out = io.StringIO()
            with redirect_stdout(out):
                decide.main(cp, bp)
            return out.getvalue()

    def test_clear_winner_holds_and_advances(self):
        txt = self.run_it({decide.PROPS[0]: cand([1] * 398 + [0] * 2, 9000)})
        self.assertIn("**1 of 5** hold", txt)
        self.assertIn("**1 of 5** advance", txt)

    def test_worse_and_dearer_is_dominated(self):
        txt = self.run_it({})
        self.assertIn("**0 of 5** hold", txt)
        self.assertIn("**0 of 5** advance", txt)

    def test_f061_missing_baseline_is_undecidable(self):
        """GLM F061: with a baseline absent, all() over the rest held vacuously narrower rules."""
        txt = self.run_it({decide.PROPS[0]: cand([1] * 398 + [0] * 2, 9000)}, drop="label-gbdt")
        self.assertIn("undecidable", txt)
        self.assertIn("**0 of 5** hold", txt)


    def test_valid_rate_verdict(self):
        self.assertEqual(decide.valid_rate_verdict(11, 20), "keep")
        self.assertEqual(decide.valid_rate_verdict(16, 26), "keep")
        self.assertEqual(decide.valid_rate_verdict(9, 20), "switch")


if __name__ == "__main__":
    unittest.main()
