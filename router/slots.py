"""Fallback chains per step ("slot"), owner 2026-11-20: if a model is unavailable or a call fails, the step moves to the
next qualified filler instead of stopping. Each entry is (pool, model); "openrouter_paid" entries are used only when the
run reserved paid budget (env PAID_FALLBACK=1) and are capped in dollars by worker/ledger.py.

Order and evidence:
reader   1 cloudflare gpt-oss-120b       live reader since P7 (free; rolling 10,000 neurons a day)
         2 nvidia Nemotron 3 Ultra       reader bake 44720253829: 6/6 reads, 21 decision-kept claims (free)
         3 openrouter_paid GLM-5.3 Flash  $0.04/$0.078 per M tokens; Artificial Analysis Intelligence Index 42, highest of the
                                          flash peers (DeepSeek V4.1 Flash 39, GPT-6 Luna 34; v4.3.2): about $0.0006 a read
         4 openrouter_paid DeepSeek V4.1 Flash 1017  $0.15/$0.60; index 45, leads the peers on AA-LCR (long context) and on
                                          AA-Omniscience (hallucination): about $0.003 a read
Paid entries were chosen from published comparisons, not tested by us (owner: no spend on tests where results exist);
every read they make still passes the same script quote check and decision model, and the ledger records cost per card.

Error policy (any of these moves to the next filler): HTTP 429 (allowance spent or rate-limited upstream), 404/410
(model gone), 5xx or no status (provider or network), a 200 whose reply is empty or not the asked JSON.
"""
import os

SLOTS = {
    "reader": [("cloudflare", "@cf/openai/gpt-oss-120b"), ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
               ("openrouter_paid", "z-ai/glm-5.3-flash"), ("openrouter_paid", "deepseek/deepseek-v4.1-flash-1017")],
}
EST_MILLI_USD = {"reader": 5}   # reservation estimate per paid call, thousandths of a dollar (above both measured-price estimates)


def chain(slot, override=None):
    """The fillers to try, in order. override (pool, model) pins one filler (arm tests)."""
    if override:
        return [tuple(override)]
    paid = os.environ.get("PAID_FALLBACK") == "1"
    return [f for f in SLOTS[slot] if paid or f[0] != "openrouter_paid"]


def run(slot, prompt, ask, parse, override=None):
    """Try each filler until one returns a reply that parses. Returns (parsed or None, last reply, tries list)."""
    tries, last = [], None
    for pool, model in chain(slot, override):
        last = ask(pool, model, prompt)
        got = parse(last.get("text")) if last.get("ok") else None
        tries.append({"pool": pool, "model": model, "status": last.get("status"), "ok": got is not None})
        if got is not None:
            return got, last, tries
    return None, last, tries
