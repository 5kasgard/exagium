"""Offline tests for keys_probe.py: no network; the key never appears in any output, on any path (queue Q84)."""
import io, json, os, re, sys, unittest, urllib.error, urllib.parse
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keys_probe as K

KEY = "s3cret/K+E=Y&123"
FORMS = (KEY, urllib.parse.quote(KEY, safe=""))


class Resp:
    def __init__(self, status=200, headers=None):
        self.status, self.headers = status, headers or {}
    def __enter__(self): return self
    def __exit__(self, *a): return False


class T(unittest.TestCase):
    def run_main(self, name, env, opener=None):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), redirect_stdout(out), redirect_stderr(err):
            if opener:
                with mock.patch.object(K.OPENER, "open", side_effect=opener):
                    self.assertEqual(K.main(name), 0)
            else:
                self.assertEqual(K.main(name), 0)
        text = out.getvalue() + err.getvalue()
        for f in FORMS:
            self.assertNotIn(f, text, name)
        return json.loads(out.getvalue())

    def test_missing_and_unprobed(self):
        self.assertEqual(self.run_main("SCALEWAY_API_KEY", {})["skipped"], "secret not set")
        self.assertIn("UPSTASH_EMAIL", self.run_main("UPSTASH_API_KEY", {"UPSTASH_API_KEY": KEY})["skipped"])

    def test_every_path_hides_the_key(self):
        seen = []
        def ok(req, timeout):
            seen.append(req.full_url)
            return Resp(200, {"X-RateLimit-Remaining": "9", "X-RateLimit-Echo": KEY, "RateLimit-Policy": "a|b\nc", "Set-Cookie": "x"})
        def http_error(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 401, "denied " + req.full_url, {"Retry-After": "5"}, None)
        def redirect(req, timeout):   # what the no-redirect opener raises on a 3xx
            raise urllib.error.HTTPError(req.full_url, 302, "Found", {"Location": "http://evil.example/?k=" + KEY}, None)
        def crash(req, timeout):
            raise ValueError("boom " + req.full_url)
        for name in K.PROBES:
            r = self.run_main(name, {name: KEY}, ok)
            self.assertEqual(r["status"], 200)
            self.assertEqual(r["limits"]["x-ratelimit-remaining"], "9")
            self.assertEqual(r["limits"]["x-ratelimit-echo"], "[dropped: contains the key]")
            self.assertNotRegex(r["limits"]["ratelimit-policy"], r"[|\n]")
            self.assertEqual(self.run_main(name, {name: KEY}, http_error)["status"], 401)
            self.assertEqual(self.run_main(name, {name: KEY}, redirect)["status"], 302)
            self.assertEqual(self.run_main(name, {name: KEY}, crash)["error"], "ValueError")
        self.assertTrue(any(urllib.parse.quote(KEY, safe="") in u for u in seen))   # query probes really send the key

    def test_redirects_are_refused(self):
        self.assertIsNone(K.NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://x"))

    def test_workflow_steps_match_the_probes(self):
        wf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows", "keys-probe.yml")).read()
        steps = set(re.findall(r"python3 -I router/keys_probe.py (\w+)", wf))
        self.assertEqual(steps, set(K.PROBES) | set(K.NOT_PROBED))
        for k in steps:   # each secret appears once, in its own step's env
            self.assertEqual(wf.count("secrets." + k), 1, k)


if __name__ == "__main__":
    unittest.main()
