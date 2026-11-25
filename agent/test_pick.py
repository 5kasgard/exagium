import json, os, re, sys, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oc_pick as P

CFG = json.load(open(os.path.join(HERE, "models.json")))
WF = os.path.join(HERE, "..", ".github", "workflows", "opencode.yml")


class Pick(unittest.TestCase):
    def test_default_is_free_and_keyless(self):
        p = P.plan(CFG, "/oc what is ready?")
        self.assertEqual((p["model"], p["provider"], p["paid"]), ("opencode/deepseek-v4.1-flash-free", "opencode", False))
        self.assertIn("what is ready?", p["prompt"])

    def test_providers_and_paid_rules(self):
        cases = {"/oc model=pickle hi": ("opencode", False),
                 "/oc model=anthropic/claude-opus-5-5 x": ("anthropic", True),
                 "/oc model=openrouter/qwen/qwen3-coder:free x": ("openrouter", True),   # a suffix is not proof

                 "/oc model=openrouter/openai/gpt-6 x": ("openrouter", True),
                 "/oc model=groq/openai/gpt-oss-120b x": ("groq", False),
                 "/oc model=opencode/gpt-6.1-sol x": ("opencode", True)}
        for c, want in cases.items():
            p = P.plan(CFG, c)
            self.assertEqual((p["provider"], p["paid"]), want, c)

    def test_listed_openrouter_model_is_free(self):
        cfg = json.loads(json.dumps(CFG))
        cfg["providers"]["openrouter"]["free_models"] = ["qwen/qwen3-coder:free"]
        self.assertFalse(P.plan(cfg, "/oc model=openrouter/qwen/qwen3-coder:free x")["paid"])

    def test_paid_refused_when_switched_off(self):
        off = dict(CFG, allow_paid=False)
        for c in ("/oc model=anthropic/claude-opus-5-5 x", "/oc model=openrouter/openai/gpt-6 x"):
            with self.assertRaises(ValueError, msg=c):
                P.plan(off, c)
        self.assertFalse(P.plan(off, "/oc model=groq/openai/gpt-oss-120b x")["paid"])   # free still runs

    def test_bad_ids_unknown_providers_and_empty_requests_refused(self):
        for c in ("/oc model=$(id) x", "/oc model=a/b;rm x", "/oc model=--help x", "/oc", "/oc model=free",
                  "hello /oc x", "/oc model=evilcorp/m x", "/oc models evilcorp"):
            with self.assertRaises(ValueError, msg=c):
                P.plan(CFG, c)

    def test_models_listing(self):
        self.assertEqual((P.plan(CFG, "/oc models")["mode"], P.plan(CFG, "/oc models")["provider"]), ("models", "opencode"))
        self.assertEqual(P.plan(CFG, "/oc models anthropic")["provider"], "anthropic")


class Scrub(unittest.TestCase):
    def test_key_values_are_removed(self):
        import subprocess, tempfile
        f = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        f.write("ok sk-ant-SECRETVALUE123 and again sk-ant-SECRETVALUE123\n"); f.close()
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-ant-SECRETVALUE123")
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "scrub.py"), f.name], env=env, capture_output=True, text=True)
        self.assertIn("2 key value", r.stdout)
        self.assertNotIn("SECRETVALUE", open(f.name).read())
        os.unlink(f.name)
