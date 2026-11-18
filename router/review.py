"""Send one prompt file to one or more free engines and save each reply. Standard library only.

usage: review.py PROMPT_FILE OUT_DIR ENGINE:MODEL [ENGINE:MODEL ...]
An outside review that doesn't pass through the author's hands: the prompt is a committed file, the reply
is saved as received. Exit 0 if at least one engine answered.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python3 -I drops the script dir
from engines import call, Refused

prompt, out = open(sys.argv[1]).read(), sys.argv[2]
os.makedirs(out, exist_ok=True)
ok = 0
for i, spec in enumerate(sys.argv[3:]):
    engine, model = spec.split(":", 1)
    try:
        r = call(engine, model, prompt, max_tokens=16000)  # reasoning models spent 4,000 on thinking and cut the reply (2026-11-18)
    except Refused as e:   # queue P4: a paid pool is refused here; record it and go on to the next engine
        r = {"engine": engine, "model": model, "ok": False, "status": None, "latency_s": 0, "text": None, "error": f"refused by spend cap: {e}"}
    ok += r["ok"]
    json.dump(r, open(os.path.join(out, f"reply-{i}.json"), "w"), indent=1)
    open(os.path.join(out, f"reply-{i}.md"), "w").write(f"<!-- {engine} {model} status={r['status']} latency={r['latency_s']}s -->\n\n{r['text'] or r['error']}\n")
    print(f"{engine} {model}: ok={r['ok']} status={r['status']} {r['latency_s']}s {len(r['text'] or '')} chars {(r['error'] or '')[:300]}")
sys.exit(0 if ok else 1)
