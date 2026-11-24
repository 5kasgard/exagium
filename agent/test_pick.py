import json, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pick as P

CFG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")))


class Pick(unittest.TestCase):
    def test_default_is_free(self):
        p = P.plan(CFG, "/oc what is ready?")
        self.assertEqual((p["model"], p["paid"]), ("opencode/deepseek-v4.1-flash-free", False))
        self.assertIn("what is ready?", p["prompt"])

    def test_alias_and_explicit_id(self):
        self.assertEqual(P.plan(CFG, "/oc model=pickle hi")["model"], "opencode/big-pickle")
        p = P.plan(CFG, "/oc model=anthropic/claude-opus-5-5 review P36")
        self.assertEqual((p["model"], p["paid"]), ("anthropic/claude-opus-5-5", True))

    def test_paid_refused_when_switched_off(self):
        with self.assertRaises(ValueError):
            P.plan(dict(CFG, allow_paid=False), "/oc model=anthropic/claude-opus-5-5 x")
        self.assertFalse(P.plan(dict(CFG, allow_paid=False), "/oc x")["paid"])   # free still runs

    def test_bad_ids_and_empty_requests_refused(self):
        for c in ("/oc model=$(id) x", "/oc model=a/b;rm x", "/oc model=--help x", "/oc", "/oc model=free", "hello /oc x"):
            with self.assertRaises(ValueError, msg=c):
                P.plan(CFG, c)

    def test_models_listing(self):
        self.assertEqual(P.plan(CFG, "/oc models")["mode"], "models")


if __name__ == "__main__":
    unittest.main()
