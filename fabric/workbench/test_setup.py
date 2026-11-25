import json, os, sys, unittest, unittest.mock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setup as S


class Setup(unittest.TestCase):
    def setUp(self):
        S.TEAM = None
        self._rt = unittest.mock.patch.object(S, "resolve_team", lambda t: None)
        self._rt.start()

    def tearDown(self):
        self._rt.stop()

    def test_team_id_is_added_to_vercel_calls(self):
        S.TEAM = "team_x"
        seen = []
        class R:
            status = 200
            def read(self): return b"{}"
            def __enter__(self): return self
            def __exit__(self, *a): return False
        with unittest.mock.patch.object(S.urllib.request, "urlopen", lambda req, timeout=0: seen.append(req.full_url) or R()):
            S.call(S.VERCEL, "/v9/projects/x", "t")
            S.call(S.VERCEL, "/v10/projects/p/env?upsert=true", "t")
            S.call(S.NEON, "/projects", "t")
        self.assertEqual(seen, ["https://api.vercel.com/v9/projects/x?teamId=team_x",
                                "https://api.vercel.com/v10/projects/p/env?upsert=true&teamId=team_x",
                                "https://console.neon.tech/api/v2/projects"])
    def test_jwks_is_a_private_rs256_key_set(self):
        k = json.loads(S.jwks())["keys"][0]
        self.assertEqual((k["kty"], k["alg"], k["use"]), ("RSA", "RS256", "sig"))
        for f in ("n", "e", "d", "p", "q", "kid"):
            self.assertTrue(k.get(f), f)

    def test_generated_secrets_are_kept_once_set(self):
        os.environ["VERCEL_API_KEY"] = "t"
        calls = []
        def fake(base, path, token, method="GET", body=None):
            calls.append((method, path, body and body.get("key")))
            if path.startswith("/v9/projects/"):
                return 200, {"id": "p1"}
            if path.endswith("/env") and method == "GET":
                return 200, {"envs": [{"key": "AUTH_SECRET", "target": ["production"]}]}
            return 200, {}
        with unittest.mock.patch.object(S, "call", fake):
            S.main(["gen", "AUTH_SECRET", "secret32"])
            self.assertFalse(any(m == "POST" for m, _, _ in calls))   # exists: not regenerated
            S.main(["gen", "KEY_VAULTS_SECRET", "secret32"])
            self.assertIn(("POST", "/v10/projects/p1/env?upsert=true", "KEY_VAULTS_SECRET"), calls)

    def test_missing_source_is_skipped_not_blanked(self):
        os.environ["VERCEL_API_KEY"] = "t"; os.environ.pop("NOPE", None)
        calls = []
        def fake(base, path, token, method="GET", body=None):
            calls.append(method)
            return (200, {"id": "p1"}) if path.startswith("/v9/") else (200, {"envs": []})
        with unittest.mock.patch.object(S, "call", fake):
            S.main(["env", "GROQ_API_KEY", "NOPE"])
        self.assertNotIn("POST", calls)


if __name__ == "__main__":
    unittest.main()
