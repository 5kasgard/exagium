"""One exact-answer call per engine: are the keys in place, and who answers?

Writes one JSON line per call to the file given as argv[1]. Exit 0 if every engine answered
exactly, 1 otherwise (the log says which). Costs: 1 OpenRouter request (of 50/day),
~2 Cloudflare neurons, 1 Gemini request plus a model list.
"""
import json, sys
from engines import call, gemini_models

PROMPT, WANT = "Reply with exactly this text and nothing else: PROBE-OK 7*6=42", "PROBE-OK 7*6=42"


def main(out):
    recs = []
    for missing in [k for k in ("OPENROUTER_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "GEMINI_API_KEY")
                    if not __import__("os").environ.get(k)]:
        recs.append({"engine": "env", "model": missing, "ok": False, "error": "secret not set"})
    if not recs:
        st, models = gemini_models()
        recs.append({"engine": "gemini", "model": "models.list", "ok": st == 200, "status": st,
                     "text": json.dumps(models) if st == 200 else None, "error": None if st == 200 else models})
        flash = [m for m in models if "flash" in m and "image" not in m and "tts" not in m and "live" not in m] if st == 200 else []
        stable = [m for m in flash if "preview" not in m and "exp" not in m] or flash
        for engine, model in [("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
                              ("cloudflare", "@cf/meta/llama-4-scout-17b-16e-instruct")] + \
                             ([("gemini", stable[0])] if stable else []):
            r = call(engine, model, PROMPT, max_tokens=1500)
            r["exact"] = (r["text"] or "").strip() == WANT
            recs.append(r)
    with open(out, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    for r in recs:
        print(f"{r['engine']:10} {r['model'][:50]:50} ok={r['ok']} exact={r.get('exact')} {(r.get('error') or '')[:160]}")
    calls = [r for r in recs if r.get("model") != "models.list"]
    return 0 if calls and all(r.get("exact") for r in calls) and len(calls) == 3 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
