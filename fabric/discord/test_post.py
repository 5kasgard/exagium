import json, os, sys, unittest, unittest.mock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import post as P


class Post(unittest.TestCase):
    def test_personas(self):
        self.assertEqual(P.persona("agent", "free")["name"], "Scout")
        self.assertEqual(P.persona("agent", "nobody")["name"], "Agent nobody")
        self.assertEqual(P.persona("workflow", "cfbox-deploy")["name"], "Deployer")
        self.assertEqual(P.persona("workflow", "unknown")["name"], "Forge")

    def test_parts(self):
        self.assertEqual(P.parts("\x1b[32mhi\x1b[0m"), ["hi"])
        long = "\n".join(["x" * 100] * 100)          # ~10,100 chars
        ps = P.parts(long)
        self.assertLessEqual(len(ps), P.PARTS)
        self.assertTrue(all(len(p) <= 2000 for p in ps))
        self.assertIn("cut", ps[-1])
        self.assertEqual(P.parts(""), ["(empty reply)"])

    def test_no_hook_no_call(self):
        with unittest.mock.patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": ""}), \
             unittest.mock.patch.object(P.urllib.request, "urlopen", side_effect=AssertionError("called")):
            self.assertEqual(P.post("agent", "free", "hi"), [])

    def test_body(self):
        sent = []
        class R:
            status = 204
            def __enter__(s): return s
            def __exit__(s, *a): pass
        with unittest.mock.patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.example/x"}), \
             unittest.mock.patch.object(P.urllib.request, "urlopen", lambda req, timeout: sent.append(json.loads(req.data)) or R()):
            self.assertEqual(P.post("agent", "coder", "@everyone hi"), [204])
        self.assertEqual(sent[0]["username"], "Coder")
        self.assertEqual(sent[0]["allowed_mentions"], {"parse": []})


if __name__ == "__main__":
    unittest.main()
