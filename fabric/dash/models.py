"""Rank models per kind of work from Artificial Analysis evaluations, and say which of our keys reach each one
(owner 2026-11-25: "artificialanalysis.ai/indexes will help us find things, but also tool calling will be important").
Standard library only; daily in .github/workflows/models-rank.yml.

  python3 -I fabric/dash/models.py OUT_JSON        # reads ARTIFICIALANALYSIS_API_KEY

Sources: Artificial Analysis /api/v2/language/models/free (credit: Artificial Analysis, required by its terms) and the
models.dev catalog (which provider serves which model, tool calling, price). Method, stated in the output: each metric
becomes a percentile among models that have it; a job's score is the mean percentile over the job's metrics the model
has (at least MIN_METRICS of them). Percentiles, not raw values, because the metrics have different scales."""
import json, os, re, sys, urllib.request
from datetime import datetime, timezone

AA = "https://artificialanalysis.ai/api/v2/language/models/free"
CATALOG = "https://models.dev/api.json"
# AA's free tier (first live run, 2026-11-25) returns only these indices, no per-benchmark fields: no tau2, Terminal-
# Bench or IFBench. Tool calling therefore is not ranked here; it needs AA Pro or our own probe (see TOOL_NOTE).
JOBS = {   # metric names as AA's API spells them; a job uses the ones a model has
    "overall intelligence": ["artificial_analysis_intelligence_index"],
    "engineering and coding": ["artificial_analysis_engineering_index", "artificial_analysis_intelligence_index"],
    "strategy and operations": ["artificial_analysis_strategy_and_ops_index", "artificial_analysis_intelligence_index"],
    "tool use and agents": ["tau2_telecom", "tau_banking", "automationbench_aa", "aa_briefcase_elo", "terminalbench_hard", "terminalbench_v2_1"],
}
MIN_METRICS = {"overall intelligence": 1, "engineering and coding": 2, "strategy and operations": 2, "tool use and agents": 2}
TOOL_NOTE = ("Tool-use metrics (tau2, Terminal-Bench, AutomationBench) are not in AA's free API tier, so 'tool use and "
             "agents' stays empty until we have AA Pro or measure tool calling ourselves.")
OURS = {   # models.dev provider id -> the secret that reaches it (only providers wired into the workbench or the agent)
    "openrouter": "OPENROUTER_API_KEY", "groq": "GROQ_API_KEY", "google": "GEMINI_API_KEY", "anthropic": "CLAUDE_API_KEY",
    "xai": "XAI_API_KEY", "fireworks-ai": "FIREWORKS_API_KEY", "cerebras": "CEREBRAS_API_KEY", "nvidia": "NVIDIA_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY", "opencode": "OPENCODE_API_KEY"}
FREE_TIER = {"groq", "cerebras", "nvidia"}   # rate-limited free tiers (claude/forge-free-pools.md); others free only at cost 0


def get(url, key=None):
    req = urllib.request.Request(url, headers={"user-agent": "forge-models", **({"x-api-key": key} if key else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def norm(s):
    s = str(s or "").lower().split("/")[-1].split(":")[0]
    s = re.sub(r"-(instruct|chat|preview|latest|it)\b", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def aa_models(key):
    out, page = [], 1
    while page < 20:
        j = get(f"{AA}?page={page}", key)
        rows = j.get("data") if isinstance(j, dict) else j
        out += rows or []
        nxt = isinstance(j, dict) and ((j.get("pagination") or {}).get("next_page") or (j.get("pagination") or {}).get("has_more") or j.get("next"))
        if not rows or not nxt:
            break
        page += 1
    return out


def reach_index(cat):
    """normalized model name -> [(provider, model id, free?, tool_call, $/M blended)]"""
    idx = {}
    for pid, p in cat.items():
        if pid not in OURS:
            continue
        for mid, m in (p.get("models") or {}).items():
            c = m.get("cost") or {}
            free = c.get("input") == 0 and c.get("output") == 0 or pid in FREE_TIER
            price = None if free else round((3 * (c.get("input") or 0) + (c.get("output") or 0)) / 4, 3)
            idx.setdefault(norm(mid), []).append({"provider": pid, "id": mid, "free": free, "tools": bool(m.get("tool_call")), "price": price})
    return idx


def percentiles(models, metric):
    vals = sorted(v for v in (m["ev"].get(metric) for m in models) if isinstance(v, (int, float)))
    if len(vals) < 5:
        return {}
    return {m["name"]: sum(x <= m["ev"][metric] for x in vals) / len(vals) for m in models if isinstance(m["ev"].get(metric), (int, float))}


def main(out):
    key = os.environ.get("ARTIFICIALANALYSIS_API_KEY") or sys.exit("ARTIFICIALANALYSIS_API_KEY is not set")
    raw = aa_models(key)
    cat = get(CATALOG)
    idx = reach_index(cat)
    models = []
    for r in raw:
        ev = r.get("evaluations") or {}
        name = r.get("name") or r.get("slug")
        hits = idx.get(norm(r.get("slug"))) or idx.get(norm(r.get("openrouter_api_id"))) or idx.get(norm(name)) or []
        models.append({"name": name, "slug": r.get("slug"), "creator": (r.get("model_creator") or {}).get("name"),
                       "open": (r.get("licensing") or {}).get("is_open_weights"), "ev": ev, "hits": hits,
                       "aa_price": (r.get("pricing") or {}).get("price_1m_input_tokens")})
    fields = sorted({k for m in models for k, v in m["ev"].items() if isinstance(v, (int, float))})
    by_job = {}
    for job, metrics in JOBS.items():
        pct = {mt: percentiles(models, mt) for mt in metrics}
        rows = []
        for m in models:
            got = [pct[mt][m["name"]] for mt in metrics if m["name"] in pct[mt]]
            if len(got) < MIN_METRICS[job]:
                continue
            free = sorted({h["provider"] for h in m["hits"] if h["free"]})
            paid = sorted({h["provider"] for h in m["hits"] if not h["free"]})
            reach = ("free: " + ", ".join(free) if free else "") + ("; " if free and paid else "") + ("paid: " + ", ".join(paid) if paid else "")
            prices = [h["price"] for h in m["hits"] if h["price"] is not None]
            rows.append({"name": m["name"], "score": round(100 * sum(got) / len(got)), "metrics": len(got),
                         "reach": reach or "not on our keys", "free": bool(free), "tools": any(h["tools"] for h in m["hits"]),
                         "price": min(prices) if prices else None})
        rows.sort(key=lambda x: -x["score"])
        if not rows:
            continue
        by_job[job] = [x for x in rows if x["reach"] != "not on our keys"][:20]
        by_job[job + " (free only)"] = [x for x in rows if x["free"]][:15]
    data = {"generated_at": datetime.now(timezone.utc).isoformat()[:19] + "Z", "models_seen": len(models),
            "reached_by_our_keys": sum(bool(m["hits"]) for m in models), "fields_seen": fields, "by_job": by_job,
            "note": "Data: Artificial Analysis (artificialanalysis.ai), free API tier; reach and price: models.dev catalog. "
                    "Score = mean percentile over the job's metrics the model has. 'free' = cost 0 in the catalog or a "
                    "rate-limited free-tier provider (groq, cerebras, nvidia). Price = catalog $/M tokens, blended 3:1. " + TOOL_NOTE}
    json.dump(data, open(out, "w"), indent=1)
    print(f"models: {len(models)} from AA, {data['reached_by_our_keys']} reachable by our keys; metrics seen: {len(fields)}; "
          + "; ".join(f"{j}: {len(r)}" for j, r in by_job.items() if "free only" not in j))


if __name__ == "__main__":
    main(sys.argv[1])
