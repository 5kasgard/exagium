"""Free inference engines behind one call. Standard library only.

Keys come from the environment (GitHub secrets) and are never printed or logged.
Every call returns a record: engine, model, ok, status, latency_s, usage, text, error.
"""
import json, os, time, urllib.request, urllib.error

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


def _scrub(s):
    for k in ("OPENROUTER_API_KEY", "CLOUDFLARE_API_TOKEN", "GEMINI_API_KEY"):
        v = os.environ.get(k)
        if v and s:
            s = s.replace(v, "***")
    return s


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
