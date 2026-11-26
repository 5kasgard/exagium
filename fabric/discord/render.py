"""Turn one `lh agent run --json` event stream into a Discord card (owner feedback 2026-11-27: tool calls looked
like plain messages, steps are noise, stats belong in an overview, replies should continue the same chat).

Event shape measured on run 45880627107 (types and field names only): agent_runtime_init (data.topicId, agentId),
step_start/step_complete, stream_start (data.model, data.provider), stream_chunk (data.content, data.reasoning,
data.toolsCalling), stream_end (data.finalContent, data.usage, data.toolsCalling), tool_start/tool_end
(data.executionTime, data.isSuccess, data.payload), agent_runtime_end (data.reason, data.finalState).

The card: the answer as the body; the tools used as one compact line; a footer with model, time, tokens, cost and
the topic id. The topic id is what lets "Reply to agent" in Discord continue the same LobeHub chat. Steps and
reasoning are not shown. Field names absent from a run are skipped rather than guessed."""
import json

COLOURS = {"lead": 0x5865F2, "free": 0x57F287, "research": 0xFEE75C, "coder": 0xEB459E}


def events(text):
    """`--json` prints pretty-printed JSON objects one after another (not JSON lines)."""
    try:
        v = json.loads(text)
        return v if isinstance(v, list) else [v]
    except ValueError:
        pass
    out, dec, i = [], json.JSONDecoder(), 0
    while i < len(text):
        while i < len(text) and text[i] in " \r\n\t,[]":
            i += 1
        if i >= len(text):
            break
        try:
            v, i = dec.raw_decode(text, i)
            out.append(v)
        except ValueError:
            break
    return [e for e in out if isinstance(e, dict)]


def _first_str(d, names, depth=0):
    if not isinstance(d, dict) or depth > 3:
        return None
    for n in names:
        if isinstance(d.get(n), str) and d[n]:
            return d[n]
    for v in d.values():
        s = _first_str(v, names, depth + 1)
        if s:
            return s
    return None


def summarise(evs):
    s = {"answer": "", "tools": [], "model": None, "provider": None, "topic": None, "tokens": 0, "cost": None,
         "reason": None, "t0": None, "t1": None}
    open_tools = []
    for e in evs:
        typ, d = e.get("type"), e.get("data") if isinstance(e.get("data"), dict) else {}
        ts = e.get("timestamp")
        if isinstance(ts, (int, float)):
            s["t0"] = ts if s["t0"] is None else min(s["t0"], ts)
            s["t1"] = ts if s["t1"] is None else max(s["t1"], ts)
        if typ == "agent_runtime_init":
            s["topic"] = d.get("topicId") or s["topic"]
        elif typ == "stream_start":
            s["model"] = d.get("model") or s["model"]; s["provider"] = d.get("provider") or s["provider"]
        elif typ == "stream_end":
            if isinstance(d.get("finalContent"), str) and d["finalContent"].strip():
                s["answer"] = d["finalContent"]          # the last step's text is the reply
            u = d.get("usage") if isinstance(d.get("usage"), dict) else {}
            s["tokens"] += int(u.get("totalTokens") or (u.get("totalInputTokens") or 0) + (u.get("totalOutputTokens") or 0) or 0)
            if isinstance(u.get("cost"), (int, float)):
                s["cost"] = (s["cost"] or 0) + u["cost"]
        elif typ == "tool_start":
            open_tools.append(_first_str(d, ["apiName", "identifier", "name", "toolName"]) or "tool")
        elif typ == "tool_end":
            name = open_tools.pop(0) if open_tools else (_first_str(d, ["apiName", "identifier", "name"]) or "tool")
            s["tools"].append({"name": name, "ok": d.get("isSuccess") is not False,
                               "ms": d.get("executionTime") if isinstance(d.get("executionTime"), (int, float)) else None})
        elif typ == "agent_runtime_end":
            s["reason"] = d.get("reason")
            fs = d.get("finalState") if isinstance(d.get("finalState"), dict) else {}
            if isinstance(fs.get("cost"), dict) and isinstance(fs["cost"].get("total"), (int, float)):
                s["cost"] = fs["cost"]["total"]
    return s


def card(key, s, seconds, exit_code):
    """One Discord embed (description up to 4,096 characters; Discord's limit)."""
    head = {1: "⚠️ The run failed.\n\n", 2: "⏸️ Waiting for your approval in LobeHub.\n\n", 3: "⏳ Still running; this is what it had so far.\n\n"}
    body = head.get(exit_code, "") + (s["answer"].strip() or "(no text reply)")
    if len(body) > 4000:
        body = body[:3950].rsplit("\n", 1)[0] + "\n\n… (cut; the full reply is in LobeHub)"
    e = {"description": body, "color": COLOURS.get(key, 0x99AAB5)}
    if s["tools"]:
        bits = [f"{'✓' if t['ok'] else '✗'} {t['name']}" + (f" {t['ms'] / 1000:.1f}s" if t["ms"] else "") for t in s["tools"][:12]]
        more = f" +{len(s['tools']) - 12}" if len(s["tools"]) > 12 else ""
        e["fields"] = [{"name": "Tools", "value": (" · ".join(bits) + more)[:1000], "inline": False}]
    foot = [x for x in [s["model"] and f"{s['model']}" + (f" ({s['provider']})" if s["provider"] else ""),
                        f"{seconds:.0f} s",
                        s["tokens"] and f"{s['tokens']:,} tokens",
                        s["cost"] is not None and f"${s['cost']:.4f}",
                        s["topic"] and f"topic {s['topic']}"] if x]
    e["footer"] = {"text": " · ".join(foot)[:2000]}
    return e
