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
    # doc job: drafter on gpt-oss-120b (Groq, then Cloudflare), then NIM Nemotron, then paid GLM-5.3 Flash
    "drafter": [("groq", "openai/gpt-oss-120b"), ("cloudflare", "@cf/openai/gpt-oss-120b"),
                ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"), ("openrouter_paid", "z-ai/glm-5.3-flash")],
    # doc job reviewer: another family than the drafter where possible; Nemotron 3.5 Super free passed P31's known-answer
    # grading 29/30 twice; paid DeepSeek V4.1 Flash 1017 last
    "reviewer": [("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"), ("openrouter", "nvidia/nemotron-3.5-super-vl:free"),
                 ("openrouter_paid", "deepseek/deepseek-v4.1-flash-1017")],
    "reader": [("cloudflare", "@cf/openai/gpt-oss-120b"), ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
               ("openrouter_paid", "z-ai/glm-5.3-flash"), ("openrouter_paid", "deepseek/deepseek-v4.1-flash-1017")],
}
EST_MILLI_USD = {"reader": 5}   # reservation estimate per paid call, thousandths of a dollar (above both measured-price estimates)


def call(engines, pool, model, prompt):
    """One call to a filler, whatever its pool; paid OpenRouter calls are allowed for this call only and recorded as
    their own pool. Reasoning models on NIM get 16,000 tokens (4,000 left Nemotron no answer, release test R1)."""
    if pool == "openrouter_paid":
        os.environ["ROUTER_ALLOW_PAID"] = "1"
        try:
            return dict(engines.call("openrouter", model, prompt, max_tokens=8000), engine="openrouter_paid")
        finally:
            os.environ.pop("ROUTER_ALLOW_PAID", None)
    if pool in ("cloudflare", "openrouter"):
        return engines.call(pool, model, prompt, max_tokens=8000 if pool == "openrouter" else 6000)
    return engines.compat_call(pool, model, prompt, max_tokens=16000)


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
