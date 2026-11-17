"""Offline tests for the router's pass rule (audit Q14) and secret scrub (audit Q52). No network, no keys.

Run: python3 -m unittest -v router/test_router.py   (from the repo root)
"""
import os, sys, unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engines, probe  # noqa: E402


def rec(engine, model, exact=None, ok=True, **kw):
    r = {"engine": engine, "model": model, "ok": ok}
    if exact is not None:
        r["exact"] = exact
    r.update(kw)
    return r


OR = ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free")
CF = ("cloudflare", "@cf/meta/llama-4-scout-17b-16e-instruct")


class Verdict(unittest.TestCase):
    def test_both_required_exact_passes(self):
        self.assertEqual(probe.verdict([rec(*OR, exact=True), rec(*CF, exact=True)]), 0)

    def test_optional_failures_do_not_fail_the_run(self):
        recs = [rec(*OR, exact=True), rec(*CF, exact=True),
                rec("gemini", "models/gemini-4-argon", exact=False, ok=False, error="429"),
                rec("nvidia", "moonshotai/kimi-k2.6", exact=False),
                rec("env", "GEMINI_API_KEY", ok=False, error="secret not set")]
        self.assertEqual(probe.verdict(recs), 0)

    # refusals: each of these must fail the run
    def test_required_wrong_answer_fails(self):
        self.assertEqual(probe.verdict([rec(*OR, exact=True), rec(*CF, exact=False)]), 1)

    def test_required_missing_exact_field_fails(self):
        self.assertEqual(probe.verdict([rec(*OR, exact=True), rec(*CF)]), 1)

    def test_required_secret_missing_fails(self):
        for k in probe.REQUIRED_SECRETS:
            with self.subTest(secret=k):
                self.assertEqual(probe.verdict([rec(*OR, exact=True), rec(*CF, exact=True),
                                                rec("env", k, ok=False, error="secret not set")]), 1)

    def test_no_required_call_fails(self):
        self.assertEqual(probe.verdict([]), 1)
        self.assertEqual(probe.verdict([rec("gemini", "models/gemini-3.8-flash", exact=True)]), 1)

    def test_only_env_records_fail(self):
        self.assertEqual(probe.verdict([rec("env", "OPENROUTER_API_KEY", ok=False, error="secret not set")]), 1)

    def test_required_is_an_allowlist(self):
        self.assertTrue(probe.required(rec(*OR)))
        self.assertTrue(probe.required(rec("env", "CLOUDFLARE_ACCOUNT_ID", ok=False)))
        for r in (rec("gemini", "x"), rec("nvidia", "x"), rec("env", "GEMINI_API_KEY", ok=False), rec("unknown", "x")):
            self.assertFalse(probe.required(r), r)


class Transport(unittest.TestCase):
    def test_every_request_names_its_agent(self):
        # Groq's Cloudflare front refuses the default Python-urllib agent with 403 / 1010 (2026-11-18)
        import urllib.request
        seen = []
        def fake(req, timeout=None):
            seen.append(req.get_header("User-agent"))
            raise OSError("offline")
        with mock.patch.object(urllib.request, "urlopen", fake), mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            engines._get("https://example.invalid/", {})
            engines.compat_models("groq")
        self.assertEqual(seen, [engines.UA, engines.UA])

    def test_xai_key_is_scrubbed(self):
        self.assertIn("XAI_API_KEY", engines.SECRET_ENVS)


class Scrub(unittest.TestCase):
    NAMES = ("CLOUDFLARE_ACCOUNT_ID", "NARA_BASE_URL", "CROWLLM_BASE_URL", "OPENROUTER_API_KEY", "CLOUDFLARE_API_TOKEN")

    def test_q52_names_are_scrubbed(self):
        for k in self.NAMES:
            self.assertIn(k, engines.SECRET_ENVS)

    def test_values_never_reach_the_text(self):
        env = {"CLOUDFLARE_ACCOUNT_ID": "0123abcd4567ef", "NARA_BASE_URL": "https://nara.example.invalid/v1",
               "CROWLLM_BASE_URL": "https://crow.example.invalid/api", "OPENROUTER_API_KEY": "sk-or-v1-deadbeefcafe"}
        text = ("POST https://api.cloudflare.com/client/v4/accounts/0123abcd4567ef/ai/run failed; "
                "base https://nara.example.invalid/v1 and https://crow.example.invalid/api; key sk-or-v1-deadbeefcafe")
        with mock.patch.dict(os.environ, env):
            out = engines._scrub(text)
        for v in env.values():
            self.assertNotIn(v, out)
        self.assertEqual(out.count("***"), 4)

    def test_empty_and_none_pass_through(self):
        self.assertIsNone(engines._scrub(None))
        self.assertEqual(engines._scrub(""), "")


if __name__ == "__main__":
    unittest.main()
