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


    def test_neon_names_the_organization(self):
        seen = []
        def fake(base, path, token, method="GET", body=None):
            seen.append((method, path, body))
            if path == "/users/me/organizations":
                return 200, {"organizations": [{"id": "org-1"}]}
            if path.startswith("/projects?") and method == "GET":
                return 200, {"projects": []}
            if path == "/projects" and method == "POST":
                return 201, {"connection_uris": [{"connection_uri": "postgres://u:p@h/db"}]}
            return 404, {}
        with unittest.mock.patch.object(S, "call", fake):
            uri, made = S.neon_url("t")
        self.assertEqual((uri, made), ("postgres://u:p@h/db", True))
        self.assertIn(("GET", "/projects?org_id=org-1", None), seen)
        self.assertEqual(seen[-1][2]["project"]["org_id"], "org-1")


    def test_build_value_replaces_a_sensitive_entry(self):
        seen = []
        def fake(base, path, token, method="GET", body=None):
            seen.append((method, path, body))
            if method == "GET":
                return 200, {"envs": [{"id": "e1", "key": "S3_ENDPOINT", "type": "sensitive", "target": ["production"]}]}
            return 200, {}
        with unittest.mock.patch.object(S, "call", fake):
            S.put_env("t", "p", "S3_ENDPOINT", "https://x.example", kind="encrypted")
        self.assertIn(("DELETE", "/v9/projects/p/env/e1", None), seen)
        self.assertEqual(seen[-1][2]["type"], "encrypted")


    def test_pg_search_patch(self):
        import tempfile, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        import patch_lobehub as P
        orig = ("DO $$\nBEGIN\n  IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search') THEN\n"
                + P.OLD + "  ELSE\n    RAISE NOTICE 'x';\n  END IF;\nEND $$;\n")
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "packages/database/migrations/0090_enable_pg_search.sql"
            f.parent.mkdir(parents=True); f.write_text(orig)
            self.assertIn("now skipped", P.patch(d))
            self.assertIn("EXCEPTION WHEN OTHERS", f.read_text())
            self.assertIn("already", P.patch(d))
            f.write_text("changed")
            with self.assertRaises(SystemExit):
                P.patch(d)


    def test_dns_upsert_creates_dns_only_cname_and_refuses_other_types(self):
        import pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        import cf_dns as D
        seen = []
        def fake(path, token, method="GET", body=None):
            seen.append((method, path, body))
            if path.startswith("/zones?"):
                return 200, {"result": [{"id": "z"}]}
            if method == "GET":
                return 200, {"result": []}
            return 200, {"result": {}}
        with unittest.mock.patch.object(D, "call", fake), unittest.mock.patch.object(D, "resolves_to_vercel", lambda h: False):
            self.assertIn("created", D.upsert("workbench.exagium.xyz", "t"))
        self.assertEqual(seen[-1][2]["proxied"], False)
        self.assertEqual(seen[-1][2]["content"], "cname.vercel-dns.com")
        def fake_a(path, token, method="GET", body=None):
            if path.startswith("/zones?"):
                return 200, {"result": [{"id": "z"}]}
            return 200, {"result": [{"type": "A", "id": "r"}]}
        with unittest.mock.patch.object(D, "call", fake_a), unittest.mock.patch.object(D, "resolves_to_vercel", lambda h: False), \
                self.assertRaises(SystemExit):
            D.upsert("workbench.exagium.xyz", "t")
        with unittest.mock.patch.object(D, "call", lambda *a, **k: self.fail("no API call when DNS is already right")), \
                unittest.mock.patch.object(D.socket, "gethostbyname_ex", lambda h: ("cname.vercel-dns.com", [h], ["1.2.3.4"])):
            self.assertIn("no change", D.upsert("workbench.exagium.xyz", "t"))


    def test_log_rows_keep_errors_and_redact(self):
        self.assertIsNone(S.log_row({"level": "info", "responseStatusCode": 200, "message": "ok"}))
        line = S.log_row({"level": "error", "requestMethod": "POST", "requestPath": "/trpc/x?input=secret",
                          "responseStatusCode": 500, "timestampInMs": 0,
                          "message": "boom for someone@example.com at forge-abc-team.vercel.app ip 10.1.2.3 "
                                     "tok " + "a" * 40 + " see https://x.y/z"})
        for leak in ("someone@example.com", "vercel.app", "10.1.2.3", "a" * 40, "https://", "secret"):
            self.assertNotIn(leak, line)
        self.assertIn("POST /trpc/x 500 boom", line)


    def test_mobile_build_patch(self):
        import tempfile, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        import patch_lobehub as P
        pkg = '{"scripts": {' + P.BUILD_OLD + ' && bun run build:next:raw", "build:spa:mobile": "vite build"}}'
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "package.json"
            f.write_text(pkg)
            self.assertIn("now compiles", P.patch_mobile(d))
            self.assertIn("build:spa:raw && bun run build:spa:mobile && bun run build:spa:auth", f.read_text())
            self.assertIn("already", P.patch_mobile(d))
            f.write_text('{"scripts": {"build:raw": "other"}}')
            with self.assertRaises(SystemExit):
                P.patch_mobile(d)


    def test_search_settings_only_list_keys_that_exist(self):
        self.assertEqual(S.search_settings({}), ("", "jina,naive"))
        self.assertEqual(S.search_settings({"HAS_EXA": "true", "HAS_TAVILY": "true", "HAS_BRAVE": "false"}),
                         ("tavily,exa", "exa,jina,naive"))


if __name__ == "__main__":
    unittest.main()
