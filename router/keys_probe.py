"""One free, read-only call per repository key; records only the HTTP status and rate-limit headers (queue Q84,
Claude chat 2026-11-24). Never prints the key or the response body; the body is untrusted and is not parsed.

python3 -I router/keys_probe.py NAME   (reads the key from the environment variable of the same name)
Exit 0 always: a probe failing is data, recorded in its line, not a CI failure. Endpoints and their sources:
"""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request

# name -> (url, how the key is sent, source). Each is a documented list/whoami call that costs nothing.
PROBES = {
    "SCALEWAY_API_KEY": ("https://api.scaleway.ai/v1/models", "bearer",
                         "scaleway.com/en/docs/generative-apis/api-cli/using-models-api"),
    "ARTIFICIALANALYSIS_API_KEY": ("https://artificialanalysis.ai/api/v2/language/models/free", "x-api-key",
                                   "artificialanalysis.ai/api/v2/openapi (free tier: 10 requests a day)"),
    "DEEPINFRA_API_KEY": ("https://api.deepinfra.com/v1/openai/models", "bearer", "unverified: OpenAI-compatible models list"),
    "FIREWORKS_API_KEY": ("https://api.fireworks.ai/inference/v1/models", "bearer", "unverified: OpenAI-compatible models list"),
    "HUGGINGFACE_API_KEY": ("https://huggingface.co/api/whoami-v2", "bearer", "unverified: Hub whoami"),
    "OPENALEX_API_KEY": ("https://api.openalex.org/works?per-page=1&search=probe&api_key={k}", "query", "unverified: api_key parameter"),
    "CORE_API_KEY": ("https://api.core.ac.uk/v3/search/works?q=probe&limit=1", "bearer", "unverified: CORE API v3"),
    "OPENML_API_KEY": ("https://www.openml.org/api/v1/json/data/list/limit/1?api_key={k}", "query", "unverified: OpenML REST v1"),
    "E2B_API_KEY": ("https://api.e2b.dev/sandboxes", "x-api-key-e2b", "unverified: list running sandboxes (creates none)"),
    "VAST_API_KEY": ("https://console.vast.ai/api/v0/users/current/", "bearer", "unverified: current user"),
    "VULTR_API_KEY": ("https://api.vultr.com/v2/account", "bearer", "unverified: account info"),
}
NOT_PROBED = {"UPSTASH_API_KEY": "the Developer API needs the account email too (HTTP Basic EMAIL:API_KEY): add UPSTASH_EMAIL",
              "BENCHLM_API_KEY": "no documented endpoint found yet"}
KEEP = ("ratelimit", "rate-limit", "remaining", "retry-after", "x-ratelimit")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would resend the key header (or a key in the query) to wherever the server points (review
    2026-11-24, reproduced): refuse it, so a 3xx is recorded as a status."""
    def redirect_request(self, *a, **k):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def clean(v, key):
    """A header value as recorded: never one that contains the key in any form, at most 64 characters, no newlines or
    table pipes (the summary is a Markdown table)."""
    v = str(v)
    if key and (key in v or urllib.parse.quote(key, safe="") in v):
        return "[dropped: contains the key]"
    return v.replace("\n", " ").replace("\r", " ").replace("|", "/")[:64]


def probe(name, key):
    url, how, src = PROBES[name]
    headers = {"User-Agent": "exagium-keys-probe"}
    if how == "bearer":
        headers["Authorization"] = "Bearer " + key
    elif how == "x-api-key":
        headers["x-api-key"] = key
    elif how == "x-api-key-e2b":
        headers["X-API-Key"] = key
    else:
        url = url.format(k=urllib.parse.quote(key, safe=""))
    shown = url.split("?")[0]
    if not url.startswith("https://"):
        return {"key": name, "url": shown, "status": None, "error": "not https", "source": src}
    t0 = time.time()
    try:
        with OPENER.open(urllib.request.Request(url, headers=headers), timeout=20) as r:
            status, h = r.status, r.headers
    except urllib.error.HTTPError as e:
        status, h = e.code, e.headers
    except Exception as e:   # network, TLS, DNS: recorded, not raised
        return {"key": name, "url": shown, "status": None, "error": type(e).__name__, "source": src}
    limits = {k.lower(): clean(v, key) for k, v in (h or {}).items() if any(s in k.lower() for s in KEEP)}
    return {"key": name, "url": shown, "status": status, "ok": 200 <= status < 300, "limits": limits,
            "latency_s": round(time.time() - t0, 2), "source": src}


def main(name):
    if name in NOT_PROBED:
        out = {"key": name, "status": None, "skipped": NOT_PROBED[name]}
    elif name not in PROBES:
        out = {"key": name, "status": None, "skipped": "no probe defined"}
    elif not os.environ.get(name):
        out = {"key": name, "status": None, "skipped": "secret not set"}
    else:
        out = probe(name, os.environ[name])
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
