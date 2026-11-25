"""Read-only MCP server over the Forge's shared state (queue P36: the custom connector for any chat client).

Any MCP-capable client (LibreChat, Open WebUI via its MCP support or mcpo, Claude, opencode, ...) can ask the Forge
what it holds. Nothing here writes: no git push, no queue moves, no secrets. It reads two local checkouts:
  FORGE_MAIN   a checkout of main (default: this repository)
  FORGE_STATE  a checkout of pilot/lgn-20261115 (queue, findings, notes, fabric, spine)
Records carry their own provenance (audit/fabric/README.md): answer from them as yourself, not as their recorder.

  python3 fabric/mcp/forge_mcp.py                           # stdio
  python3 fabric/mcp/forge_mcp.py streamable-http           # HTTP on MCP_PORT (default 8765), path /mcp
"""
import datetime as dt, json, os, subprocess, sys

from mcp.server.mcpserver import MCPServer

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.environ.get("FORGE_MAIN") or os.path.dirname(os.path.dirname(HERE))
STATE = os.environ.get("FORGE_STATE", os.path.join(MAIN, "..", "exagium-state"))
QUEUE = "audit/2026-11-17/queue.json"
READABLE = ("audit/notes/", "audit/fabric/", "audit/spine/", "audit/findings/", "audit/2026-11-17/QUEUE.md")

mcp = MCPServer("forge", instructions="Read-only view of the Forge's queue, events, findings and notes. Each record "
                "names who made it; you are not that recorder. Cite item ids and file paths.")


def _queue():
    return json.load(open(os.path.join(STATE, QUEUE)))


def _brief(i):
    return {k: i.get(k) for k in ("id", "status", "kind", "title", "severity", "value", "blocked_by", "touches")}


@mcp.tool()
def refresh() -> str:
    """Pull the latest main and queue branch into the local checkouts (read-only: fetch and fast-forward only)."""
    out = []
    for path in (MAIN, STATE):
        r = subprocess.run(["git", "-C", path, "pull", "-q", "--ff-only"], capture_output=True, text=True)
        out.append(f"{path}: {'ok' if r.returncode == 0 else r.stderr.strip()[:200]}")
    return "\n".join(out)


@mcp.tool()
def queue_summary() -> dict:
    """Counts by status, items the free worker can claim now (worker/queue_ops.ready), and the newest 10 items."""
    q = _queue()
    sys.path.insert(0, os.path.join(MAIN, "worker"))
    try:
        import queue_ops
        ready = [i["id"] for i in queue_ops.ready(q, dt.datetime.now(dt.timezone.utc))]
    except Exception as e:   # main checkout missing or changed: say so, do not guess
        ready = f"unavailable: {e}"
    counts = {}
    for i in q["items"]:
        counts[i["status"]] = counts.get(i["status"], 0) + 1
    newest = sorted(q["items"], key=lambda i: i.get("generated_at", ""), reverse=True)[:10]
    return {"counts": counts, "ready_now": ready, "newest": [_brief(i) for i in newest]}


@mcp.tool()
def queue_item(item_id: str) -> dict:
    """One queue item in full, with its run and event log."""
    for i in _queue()["items"]:
        if i["id"] == item_id:
            return i
    return {"error": f"no item {item_id}"}


@mcp.tool()
def queue_search(text: str = "", status: str = "", limit: int = 25) -> list:
    """Items whose id, title or next step contains text (case-insensitive), optionally of one status."""
    t = text.lower()
    hits = [i for i in _queue()["items"] if (not status or i["status"] == status)
            and (t in i["id"].lower() or t in i.get("title", "").lower() or t in i.get("next", "").lower())]
    return [_brief(i) for i in hits[: max(1, min(limit, 100))]]


@mcp.tool()
def recent_events(hours: float = 24, limit: int = 50) -> list:
    """Queue events (claims, releases, moves; each with actor, outcome, from, to) in the last N hours, newest first."""
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ev = [{"item": i["id"], **{k: r.get(k) for k in ("at", "worker", "outcome", "from", "to")},
           "result": str(r.get("result", ""))[:200]}
          for i in _queue()["items"] for r in i.get("runs", []) if r.get("at", "") >= since]
    return sorted(ev, key=lambda e: e["at"] or "", reverse=True)[: max(1, min(limit, 500))]


@mcp.tool()
def findings(status: str = "", series: str = "", limit: int = 50) -> list:
    """The findings register (audit/findings/register.jsonl), filtered by status and id prefix."""
    out = []
    for line in open(os.path.join(STATE, "audit/findings/register.jsonl")):
        f = json.loads(line)
        if (not status or f.get("status") == status) and (not series or str(f.get("id", "")).startswith(series)):
            out.append(f)
    return out[: max(1, min(limit, 500))]


@mcp.tool()
def list_notes() -> list:
    """Handoffs, notes and fabric files on the queue branch."""
    out = []
    for root in ("audit/notes", "audit/fabric", "audit/spine"):
        for d, _, fs in os.walk(os.path.join(STATE, root)):
            out += [os.path.relpath(os.path.join(d, f), STATE) for f in fs]
    return sorted(out)


@mcp.tool()
def read_note(path: str) -> str:
    """One file under audit/notes, audit/fabric, audit/spine or audit/findings (queue branch), first 60 kB."""
    full = os.path.realpath(os.path.join(STATE, path))
    roots = [os.path.realpath(os.path.join(STATE, r)) for r in READABLE]
    if not any(full == r or full.startswith(r.rstrip(os.sep) + os.sep) for r in roots):
        return "refused: only audit/notes, audit/fabric, audit/spine, audit/findings and QUEUE.md are readable"
    return open(full, encoding="utf-8", errors="replace").read()[:60000]


if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run(transport, port=int(os.environ.get("MCP_PORT", "8765")))
