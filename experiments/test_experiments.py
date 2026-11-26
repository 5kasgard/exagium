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
            if n == "claim_roots":
                import claim_roots
                self.assertTrue(isinstance(claim_roots.PLAN, str) and len(claim_roots.PLAN) > 40, n)
                self.assertTrue(callable(claim_roots.rule), n)
                continue
            m, _ = run_one.load(n)
            self.assertTrue(isinstance(m.PLAN, str) and len(m.PLAN) > 40, n)

    def test_claim_roots_rule(self):
        import claim_roots
        self.assertEqual(claim_roots.rule({"claims": 10, "shared_root_claims": 2})["verdict"], "holds")
        self.assertEqual(claim_roots.rule({"claims": 10, "shared_root_claims": 1})["verdict"], "fails")
        self.assertEqual(claim_roots.rule({"claims": 0, "shared_root_claims": 0})["verdict"], "invalid")

    def test_packet_pilot_no_pilot(self):
        import packet_pilot
        self.assertEqual(packet_pilot.rule({"status": "no_pilot"})["verdict"], "invalid")

    def test_packet_pilot_rule(self):
        import packet_pilot
        result = {
            "baseline": {"proposals": 10, "valid": 10, "calls": 10, "correct": 10},
            "repair": {"proposals": 10, "valid": 10, "calls": 10, "correct": 4},
        }
        self.assertEqual(packet_pilot.rule(result)["verdict"], "holds")
        result["repair"]["correct"] = 3
        self.assertEqual(packet_pilot.rule(result)["verdict"], "fails")

    def test_packet_pilot_ignores_unrelated_json(self):
        import packet_pilot
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "unrelated.json"), "w").write(json.dumps({"digests": 554}))
            self.assertEqual(packet_pilot.run(d), {"status": "no_pilot"})

    def test_drift_online_rule(self):
        import importlib.util as u
        spec = u.spec_from_file_location("e_drift", os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_online.py"))
        m = u.module_from_spec(spec); spec.loader.exec_module(m)
        base = {"pinned": True, "test_items": 2000, "online": 0.9, "static": 0.6, "majority": 0.5, "diff": 0.3, "diff_p2_5": 0.27}
        self.assertEqual(m.rule(base)["verdict"], "holds")
        self.assertEqual(m.rule({**base, "diff": 0.08})["verdict"], "fails")
        self.assertEqual(m.rule({**base, "diff_p2_5": 0.04})["verdict"], "fails")
        self.assertEqual(m.rule({**base, "online": 0.55})["verdict"], "fails")
        self.assertEqual(m.rule({**base, "pinned": False})["verdict"], "invalid")

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
