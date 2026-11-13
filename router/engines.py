"""Free inference engines behind one call. Standard library only.

Keys come from the environment (GitHub secrets) and are never printed or logged.
Every call returns a record: engine, model, ok, status, latency_s, usage, text, error.
"""
import json, os, re, time, urllib.request, urllib.error

TIMEOUT = 180


def _post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    return _send(req)


def _get(url, headers):
    return _send(urllib.request.Request(url, headers=headers))


def _send(req):
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")[:600]
        return e.code, {"_error": raw}
    except Exception as e:  # network, timeout, bad JSON
        return 0, {"_error": f"{type(e).__name__}: {e}"[:600]}


# OpenAI-compatible free pools (added 2026-11-14). name: (base URL or env var holding it, key env var).
# Listed is not the same as usable: the probe measures each before anything is routed to it.
OPENAI_COMPAT = {
    "llm7": ("https://api.llm7.io/v1", "LLM7_API_KEY"),
    "nara": ("env:NARA_BASE_URL", "NARA_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "sambanova": ("https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY"),
    "cohere": ("https://api.cohere.ai/compatibility/v1", "COHERE_API_KEY"),
    "inferbase": ("https://api.inferbase.ai/v1", "INFERBASE_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "crowllm": ("env:CROWLLM_BASE_URL", "CROWLLM_API_KEY"),
}
SECRET_ENVS = ("OPENROUTER_API_KEY", "CLOUDFLARE_API_TOKEN", "GEMINI_API_KEY") + tuple(v[1] for v in OPENAI_COMPAT.values())


def compat_base(name):
    base = OPENAI_COMPAT[name][0]
    return os.environ.get(base[4:], "").rstrip("/") if base.startswith("env:") else base


def _send_h(req):
    """Like _send, but also returns the rate-limit headers (names containing 'ratelimit' or 'retry')."""
    def pick(h):
        return {k.lower(): v for k, v in (h or {}).items() if "ratelimit" in k.lower() or "retry" in k.lower()}
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, json.loads(r.read().decode()), pick(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, {"_error": e.read().decode(errors="replace")[:600]}, pick(e.headers)
    except Exception as e:
        return 0, {"_error": f"{type(e).__name__}: {e}"[:600]}, {}


def compat_models(name):
    """(status, [model ids] or error text, rate-limit headers) for one OpenAI-compatible pool."""
    base, key = compat_base(name), os.environ[OPENAI_COMPAT[name][1]]
    st, j, h = _send_h(urllib.request.Request(base + "/models", headers={"Authorization": "Bearer " + key}))
    if st != 200:
        return st, _scrub(j.get("_error") or json.dumps(j)[:600]), h
    return st, sorted(m.get("id", "") for m in j.get("data", []) if isinstance(m, dict)), h


def compat_call(name, model, prompt, max_tokens=64):
    t0 = time.time()
    base, key = compat_base(name), os.environ[OPENAI_COMPAT[name][1]]
    req = urllib.request.Request(base + "/chat/completions", method="POST",
                                 data=json.dumps({"model": model, "max_tokens": max_tokens,
                                                  "messages": [{"role": "user", "content": prompt}]}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    st, j, h = _send_h(req)
    text = (j.get("choices") or [{}])[0].get("message", {}).get("content") if st == 200 else None
    return {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "engine": name, "model": model,
            "ok": text is not None, "status": st, "latency_s": round(time.time() - t0, 2), "usage": j.get("usage"),
            "text": text, "limits": h, "error": None if text is not None else _scrub(j.get("_error") or json.dumps(j)[:600])}


def _scrub(s):
    for k in SECRET_ENVS:
        v = os.environ.get(k)
        if v and s:
            s = s.replace(v, "***")
    # provider account identifiers are not secrets but they identify the owner; logs are public (2026-11-14)
    return re.sub(r"(account|org|organization|user)(['\"]?\s*[:=]?\s*['\"])[^'\"]{6,}(['\"])", r"\1\2***\3", s, flags=re.I) if s else s


def call(engine, model, prompt, system=None, max_tokens=2000):
    t0 = time.time()
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    text, usage = None, None
    if engine == "openrouter":
        if not model.endswith(":free") and os.environ.get("ROUTER_ALLOW_PAID") != "1":
            raise ValueError(f"refusing paid OpenRouter model {model}: the account holds credit; set ROUTER_ALLOW_PAID=1 only with the owner's approval")
        st, j = _post("https://openrouter.ai/api/v1/chat/completions",
                      {"model": model, "messages": msgs, "max_tokens": max_tokens},
                      {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
        if st == 200 and j.get("choices"):
            text, usage = j["choices"][0]["message"].get("content"), j.get("usage")
    elif engine == "cloudflare":
        acct = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        st, j = _post(f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}",
                      {"messages": msgs, "max_tokens": max_tokens},
                      {"Authorization": "Bearer " + os.environ["CLOUDFLARE_API_TOKEN"]})
        if st == 200 and j.get("success"):
            r = j["result"]
            text = r.get("response") or (r.get("choices") or [{}])[0].get("message", {}).get("content")
            usage = r.get("usage")
    elif engine == "gemini":
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        st, j = _post(f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent", body,
                      {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        if st == 200 and j.get("candidates"):
            parts = j["candidates"][0].get("content", {}).get("parts", [])
            text, usage = "".join(p.get("text", "") for p in parts if not p.get("thought")), j.get("usageMetadata")
    else:
        raise ValueError(engine)
    return {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "engine": engine, "model": model,
            "ok": text is not None, "status": st, "latency_s": round(time.time() - t0, 2), "usage": usage,
            "text": text, "error": None if text is not None else _scrub(j.get("_error") or json.dumps(j)[:600])}


def gemini_models():
    """Text models this key may call with generateContent (name list), or an error record."""
    st, j = _get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                 {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    if st != 200:
        return st, _scrub(j.get("_error", ""))
    return st, [m["name"] for m in j.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
