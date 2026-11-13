"""One exact-answer call per engine: are the keys in place, and who answers?

Writes one JSON line per call to the file given as argv[1]. Exit 0 if every engine answered
exactly, 1 otherwise (the log says which). Costs: 1 OpenRouter request (of 50/day),
~2 Cloudflare neurons, 1 Gemini request plus a model list.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python3 -I drops the script dir
from engines import call, gemini_models, OPENAI_COMPAT, compat_base, compat_models, compat_call

PROMPT, WANT = "Reply with exactly this text and nothing else: PROBE-OK 7*6=42", "PROBE-OK 7*6=42"


def main(out):
    recs = []
    have = {k: bool(os.environ.get(k)) for k in ("OPENROUTER_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "GEMINI_API_KEY")}
    for k, v in have.items():
        if not v:
            recs.append({"engine": "env", "model": k, "ok": False, "error": "secret not set"})
    plan = []
    if have["OPENROUTER_API_KEY"]:
        plan.append(("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"))
    if have["CLOUDFLARE_API_TOKEN"] and have["CLOUDFLARE_ACCOUNT_ID"]:
        plan.append(("cloudflare", "@cf/meta/llama-4-scout-17b-16e-instruct"))
    if have["GEMINI_API_KEY"]:
        st, models = gemini_models()
        recs.append({"engine": "gemini", "model": "models.list", "ok": st == 200, "status": st,
                     "text": json.dumps(models) if st == 200 else None, "error": None if st == 200 else models})
        want = os.environ.get("PROBE_GEMINI_MODELS", "gemini-3.8-flash,gemini-3.5-flash-lite,gemini-4-argon,gemini-4-carbon").split(",")
        for m in want:  # one request each; listed is not the same as allowed on the free tier
            if st == 200 and "models/" + m in models:
                plan.append(("gemini", "models/" + m))
            else:
                recs.append({"engine": "gemini", "model": m, "ok": False, "error": "not in this key's model list"})
    for engine, model in plan:
        r = call(engine, model, PROMPT, max_tokens=1500)
        r["exact"] = (r["text"] or "").strip() == WANT
        recs.append(r)
    recs += probe_compat()
    with open(out, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    for r in recs:
        print(f"{r['engine']:10} {r['model'][:50]:50} ok={r['ok']} exact={r.get('exact')} {(r.get('error') or '')[:160]}")
    return 0 if all(r.get("exact") or r.get("model") == "models.list" and r["ok"] or r.get("optional") for r in recs) else 1


SKIP = ("embed", "rerank", "whisper", "tts", "speech", "image", "vision-embed", "guard", "safety", "reward", "ocr", "audio", "moderation")
HINTS = {"llm7": ["gpt-4o-mini"], "nvidia": ["nemotron"], "mistral": ["mistral-small", "devstral"], "cerebras": ["gpt-oss"],
         "sambanova": ["DeepSeek"], "cohere": ["command"], "groq": ["gpt-oss"]}


# Measured 2026-11-14 (run 43519222584): listed is not the same as callable (nemotron-51b listed, 404 on call).
# LLM7 also lists closed frontier models (Claude, GPT, Gemini) for free; a free proxy cannot be verified to serve
# them, so only open-weight models are probed there.
CANDIDATES = {
    "nvidia": ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b", "moonshotai/kimi-k2.6"],
    "llm7": ["glm-5.3", "deepseek-v4-pro", "kimi-k2.7-code"],
}


def pick_model(name, ids):
    """PROBE_<NAME>_MODEL if set, else the first listed id matching a hint, else the first chat-like id."""
    env = os.environ.get(f"PROBE_{name.upper()}_MODEL")
    if env:
        return env
    chat = [i for i in ids if not any(x in i.lower() for x in SKIP)]
    for h in HINTS.get(name, []):
        for i in chat:
            if h.lower() in i.lower():
                return i
    return chat[0] if chat else None


def probe_compat():
    """Added 2026-11-14. For each OpenAI-compatible pool: is the key set, which models are listed, and does one
    tiny call answer exactly? A pool whose key is not set is recorded and does not fail the run."""
    recs = []
    for name, (_, key) in OPENAI_COMPAT.items():
        if not os.environ.get(key):
            recs.append({"engine": name, "model": key, "ok": False, "optional": True, "error": "secret not set"})
            continue
        if not compat_base(name):
            recs.append({"engine": name, "model": "base_url", "ok": False, "error": "key set, base URL secret not set"})
            continue
        st, ids, lim = compat_models(name)
        listed = st == 200
        recs.append({"engine": name, "model": "models.list", "ok": listed, "status": st, "limits": lim,
                     "n_models": len(ids) if listed else None, "text": json.dumps(ids[:300]) if listed else None,
                     "error": None if listed else ids})
        cands = [c for c in CANDIDATES.get(name, []) if listed and c in ids][:3] or [pick_model(name, ids if listed else [])]
        for m in cands:
            if m is None:
                recs.append({"engine": name, "model": "none", "ok": False, "error": "no chat model chosen; set PROBE_%s_MODEL" % name.upper()})
                continue
            r = compat_call(name, m, PROMPT, max_tokens=400)   # reasoning models spend tokens before answering
            r["exact"] = (r["text"] or "").strip() == WANT
            recs.append(r)
    return recs


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
