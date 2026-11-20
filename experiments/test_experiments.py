"""Every experiment file keeps the contract (experiments/README.md), and the runner turns failures into 'invalid'."""
import glob, json, os, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_one  # noqa: E402


class Contract(unittest.TestCase):
    def test_every_experiment_has_plan_run_rule(self):
        for p in glob.glob(os.path.join(HERE, "*.py")):
            n = os.path.basename(p)[:-3]
            if n in ("run_one",) or n.startswith("test_"):
                continue
            m, _ = run_one.load(n)
            self.assertTrue(isinstance(m.PLAN, str) and len(m.PLAN) > 40, n)

    def test_runner_verdicts(self):
        with tempfile.TemporaryDirectory() as d:
            for name, body, want, rc in (
                    ("tmp_ok", "PLAN='x'*50\ndef run():\n    return {'a': 1}\ndef rule(r):\n    return {'verdict': 'holds', 'why': 'a'}\n", "holds", 0),
                    ("tmp_bad", "PLAN='x'*50\ndef run():\n    raise ValueError('no')\ndef rule(r):\n    return {}\n", "invalid", 1),
                    ("tmp_rule", "PLAN='x'*50\ndef run():\n    return {}\ndef rule(r):\n    return {'verdict': 'maybe'}\n", "invalid", 1)):
                p = os.path.join(HERE, name + ".py")
                open(p, "w").write(body)
                try:
                    self.assertEqual(run_one.main(name, os.path.join(d, name)), rc)
                    self.assertEqual(json.load(open(os.path.join(d, name, "verdict.json")))["verdict"]["verdict"], want)
                finally:
                    os.remove(p)
        with self.assertRaises(SystemExit):
            run_one.load("../x")


if __name__ == "__main__":
    unittest.main()
