"""One exact-answer call per engine: are the keys in place, and who answers?

Writes one JSON line per call to the file given as argv[1]. Exit 0 if every engine answered
exactly, 1 otherwise (the log says which). Costs: 1 OpenRouter request (of 50/day),
~2 Cloudflare neurons, 1 Gemini request plus a model list.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python3 -I drops the script dir
from engines import call, gemini_models

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
    with open(out, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    for r in recs:
        print(f"{r['engine']:10} {r['model'][:50]:50} ok={r['ok']} exact={r.get('exact')} {(r.get('error') or '')[:160]}")
    return 0 if all(r.get("exact") or r.get("model") == "models.list" and r["ok"] for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
