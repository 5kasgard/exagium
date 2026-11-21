"""Offline tests for the router's pass rule (audit Q14) and secret scrub (audit Q52). No network, no keys.

Run: python3 -m unittest -v router/test_router.py   (from the repo root)
"""
import json, os, sys, unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decisions_route, engines, probe  # noqa: E402


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


class ProbeDocstring(unittest.TestCase):
    def test_f048_cost_mentions_probe_compat_calls(self):
        doc = probe.__doc__ or ""
        self.assertIn("probe_compat()", doc)
        self.assertIn("per keyed OpenAI-compatible pool", doc)
        self.assertIn("up-to-4 candidate calls", doc)


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

    def test_f043_secret_across_the_cut_is_scrubbed(self):
        """GLM F043: truncating to 600 before scrubbing left a key fragment that the replace could not match."""
        import io, urllib.error, urllib.request
        key = "sk-or-v1-" + "a1b2c3d4" * 6
        body = ("x" * 580 + key + " tail").encode()
        def fake(req, timeout=None):
            raise urllib.error.HTTPError("https://e.invalid", 401, "no", {}, io.BytesIO(body))
        with mock.patch.object(urllib.request, "urlopen", fake), mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": key}):
            st, j = engines._send(urllib.request.Request("https://e.invalid"))
        self.assertEqual(st, 401)
        self.assertNotIn(key[:20], j["_error"])

    def test_f042_no_env_switch_for_paid(self):
        with mock.patch.dict(os.environ, {"ROUTER_ALLOW_PAID": "1", "OPENROUTER_API_KEY": "k"}):
            with self.assertRaisesRegex(ValueError, "refusing paid"):
                engines.call("openrouter", "z-ai/glm-5.3-flash", "hi")

    def test_f039_paid_logged_as_paid_pool(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "u.jsonl")
            with mock.patch.object(engines, "_post", return_value=(200, {"choices": [{"message": {"content": "ok"}}]})), \
                    mock.patch.object(engines, "paid_guard"), \
                    mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k", "ROUTER_USAGE_LOG": log}):
                r = engines.call("openrouter", "z-ai/glm-5.3-flash", "hi", allow_paid=True)
            self.assertEqual(r["engine"], "openrouter_paid")
            self.assertEqual(json.loads(open(log).read())["engine"], "openrouter_paid")

    def test_empty_and_none_pass_through(self):
        self.assertIsNone(engines._scrub(None))
        self.assertEqual(engines._scrub(""), "")




class RetryAfter(unittest.TestCase):
    def test_retry_after_seconds_header(self):
        now = 1000.0
        self.assertEqual(engines.retry_after(429, {"Retry-After": "30"}, now), 30)

    def test_retry_after_http_date(self):
        # HTTP date 60 seconds in the future
        from email.utils import formatdate
        now = 1000.0
        future = formatdate(now + 60, usegmt=True)
        self.assertEqual(engines.retry_after(429, {"Retry-After": future}, now), 60)

    def test_non_429_returns_none(self):
        self.assertIsNone(engines.retry_after(200, {"Retry-After": "30"}, 1000.0))
        self.assertIsNone(engines.retry_after(500, {}, 1000.0))

    def test_fallback_when_no_usable_header(self):
        self.assertEqual(engines.retry_after(429, {}, 1000.0), 60)
        self.assertEqual(engines.retry_after(429, {"X-Other": "value"}, 1000.0), 60)

    def test_ratelimit_reset_header(self):
        now = 1000.0
        self.assertEqual(engines.retry_after(429, {"X-RateLimit-Reset": "45"}, now), 45)
        self.assertEqual(engines.retry_after(429, {"x-ratelimit-reset": "45"}, now), 45)
        self.assertEqual(engines.retry_after(429, {"RateLimit-Reset": "45"}, now), 45)


class DecisionsScrub(unittest.TestCase):
    def test_f045_http_error_body_is_scrubbed_in_json_and_stdout(self):
        import contextlib, io, tempfile, urllib.error, urllib.request
        key = "sk-or-v1-" + "f045deadbeef" * 4
        body = ("provider refused: key=" + key + " please retry").encode()

        def fake(req, timeout=None):
            raise urllib.error.HTTPError("https://openrouter.ai/api/alpha/decisions", 401, "no", {}, io.BytesIO(body))

        with tempfile.TemporaryDirectory() as d:
            cases = os.path.join(d, "cases.json")
            out = os.path.join(d, "out.json")
            with open(cases, "w") as f:
                json.dump([{"id": "case-1", "expect": "confirm", "state": "state"}], f)
            buf = io.StringIO()
            with mock.patch.object(decisions_route.urllib.request, "urlopen", fake), \
                    mock.patch.object(decisions_route.time, "sleep", lambda s: None), \
                    mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": key}), \
                    contextlib.redirect_stdout(buf):
                decisions_route.main("test/model", cases, out)
            text = open(out).read()
            printed = buf.getvalue()

        self.assertNotIn(key, text)
        self.assertNotIn(key, printed)
        self.assertIn("***", text)
        self.assertIn("***", printed)
        result = json.loads(text)["results"][0]
        self.assertEqual(result["status"], 401)
        self.assertIn("***", result["raw"]["error"])


class SpendCaps(unittest.TestCase):
    """Queue P4: a paid pool is refused before any request unless listed, within per-call caps, logged and under the daily cap."""

    def setUp(self):
        import tempfile, urllib.request
        self.d = tempfile.mkdtemp()
        self.log = os.path.join(self.d, "usage.jsonl")
        self.sent = []
        def fake(req, timeout=None):
            self.sent.append(req.full_url)
            raise OSError("offline")
        self.p = mock.patch.object(urllib.request, "urlopen", fake)
        self.p.start()

    def tearDown(self):
        self.p.stop()

    def env(self, **kw):
        base = {"CEREBRAS_API_KEY": "k", "XAI_API_KEY": "k", "GEMINI_API_KEY": "k"}
        base.update(kw)
        return mock.patch.dict(os.environ, base, clear=False)

    def test_unlisted_pool_is_refused_before_sending(self):
        with self.env(ROUTER_PAID_POOLS="", ROUTER_USAGE_LOG=self.log):
            for e, m in (("cerebras", "gpt-oss-120b"), ("xai", "grok-mini")):
                with self.assertRaises(engines.Refused):
                    engines.compat_call(e, m, "hi")
            with self.assertRaises(engines.Refused):
                engines.call("gemini", "models/gemini-3.5-flash", "hi")
        self.assertEqual(self.sent, [])

    def test_over_per_call_cap_is_refused(self):
        with self.env(ROUTER_PAID_POOLS="cerebras", ROUTER_USAGE_LOG=self.log):
            with self.assertRaises(engines.Refused):
                engines.compat_call("cerebras", "m", "hi", max_tokens=engines.PAID["cerebras"]["max_tokens"] + 1)
            with self.assertRaises(engines.Refused):
                engines.compat_call("cerebras", "m", "x" * (engines.PAID["cerebras"]["prompt_chars"] + 1))
        self.assertEqual(self.sent, [])

    def test_no_log_means_refused(self):
        with self.env(ROUTER_PAID_POOLS="cerebras"):
            os.environ.pop("ROUTER_USAGE_LOG", None)
            with self.assertRaises(engines.Refused):
                engines.compat_call("cerebras", "m", "hi")
        self.assertEqual(self.sent, [])

    def test_daily_cap_counts_logged_calls_and_refuses_at_the_cap(self):
        import json, time
        day = time.strftime("%Y-%m-%dT00:00:01Z", time.gmtime())
        cap = engines.PAID["xai"]["calls_per_day"]
        with open(self.log, "w") as f:
            for _ in range(cap - 1):
                f.write(json.dumps({"t": day, "engine": "xai"}) + "\n")
            f.write(json.dumps({"t": "2000-01-01T00:00:00Z", "engine": "xai"}) + "\n")   # another day: not counted
            f.write(json.dumps({"t": day, "engine": "cerebras"}) + "\n")                 # another pool: not counted
        with self.env(ROUTER_PAID_POOLS="xai", ROUTER_USAGE_LOG=self.log):
            engines.compat_call("xai", "m", "hi")          # call number cap: allowed, sent, logged
            self.assertEqual(len(self.sent), 1)
            with self.assertRaises(engines.Refused):       # call number cap + 1: refused before sending
                engines.compat_call("xai", "m", "hi")
        self.assertEqual(len(self.sent), 1)
        last = json.loads(open(self.log).read().splitlines()[-1])
        self.assertEqual((last["engine"], last["ok"]), ("xai", False))
        self.assertNotIn("text", last)                     # the log keeps no prompt or reply text

    def test_free_pools_are_not_capped_and_are_logged(self):
        with self.env(ROUTER_PAID_POOLS="", ROUTER_USAGE_LOG=self.log, NVIDIA_API_KEY="k"):
            engines.compat_call("nvidia", "m", "x" * 100000, max_tokens=16000)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(len(open(self.log).read().splitlines()), 1)


class ProbeUnderCaps(unittest.TestCase):
    def test_a_refused_paid_pool_is_a_record_not_a_crash(self):
        import urllib.request
        def fake(req, timeout=None):
            raise OSError("offline")
        env = {"XAI_API_KEY": "k", "ROUTER_PAID_POOLS": ""}
        with mock.patch.object(urllib.request, "urlopen", fake), mock.patch.dict(os.environ, env), \
             mock.patch.object(probe, "compat_models", lambda n: (200, ["grok-mini"], {})), \
             mock.patch.object(probe, "OPENAI_COMPAT", {"xai": engines.OPENAI_COMPAT["xai"]}):
            recs = probe.probe_compat()
        calls = [r for r in recs if r["model"] != "models.list"]
        self.assertTrue(calls and all(not r["ok"] for r in calls), recs)
        self.assertTrue(any("refused by spend cap" in (r.get("error") or "") for r in recs), recs)


if __name__ == "__main__":
    unittest.main()
