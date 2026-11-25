// Forge connector on Cloudflare Workers (queue P36): a read-only remote MCP server (Streamable HTTP, JSON responses)
// over the Forge's public state. No secrets: it reads the public repo's raw files. Add the URL ending in /mcp as a
// custom connector in any MCP client (Claude, ChatGPT, LibreChat, Open WebUI, opencode).
// Records carry provenance (audit/fabric/README.md): answer from them as yourself, not as their recorder.
const REPO = "5kasgard/exagium";
const STATE = "pilot/lgn-20261115";
const RAW = (ref, path) => `https://raw.githubusercontent.com/${REPO}/${ref}/${path}`;
const QUEUE = "audit/2026-11-17/queue.json";
const READABLE = ["audit/notes/", "audit/fabric/", "audit/spine/", "audit/findings/", "audit/2026-11-17/QUEUE.md"];

async function get(path, ref = STATE) {
  const r = await fetch(RAW(ref, path), { cf: { cacheTtl: 60 } });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.text();
}
const queue = async () => JSON.parse(await get(QUEUE));
const brief = (i) => ({ id: i.id, status: i.status, kind: i.kind, title: i.title, severity: i.severity, value: i.value,
  blocked_by: i.blocked_by, touches: i.touches });
const clamp = (n, lo, hi, d) => Math.max(lo, Math.min(hi, Number.isFinite(+n) ? +n : d));

const TOOLS = {
  queue_summary: {
    description: "Counts by status, the newest 10 items, and the queue page's 'Can run now' count.",
    schema: { type: "object", properties: {} },
    run: async () => {
      const q = await queue();
      const counts = {};
      for (const i of q.items) counts[i.status] = (counts[i.status] || 0) + 1;
      const md = await get("audit/2026-11-17/QUEUE.md");
      const m = md.match(/## Can run now \((\d+)\)/);
      const newest = [...q.items].sort((a, b) => (b.generated_at || "").localeCompare(a.generated_at || "")).slice(0, 10);
      return { counts, can_run_now: m ? +m[1] : null, newest: newest.map(brief),
        note: "Items a free worker can claim need a job type (targets/research/code); see queue item P38." };
    },
  },
  queue_item: {
    description: "One queue item in full, with its run and event log.",
    schema: { type: "object", properties: { item_id: { type: "string" } }, required: ["item_id"] },
    run: async ({ item_id }) => (await queue()).items.find((i) => i.id === item_id) || { error: `no item ${item_id}` },
  },
  queue_search: {
    description: "Items whose id, title or next step contains text (case-insensitive), optionally of one status.",
    schema: { type: "object", properties: { text: { type: "string" }, status: { type: "string" }, limit: { type: "integer" } } },
    run: async ({ text = "", status = "", limit = 25 }) => {
      const t = String(text).toLowerCase();
      return (await queue()).items.filter((i) => (!status || i.status === status) &&
        [i.id, i.title || "", i.next || ""].some((s) => s.toLowerCase().includes(t))).slice(0, clamp(limit, 1, 100, 25)).map(brief);
    },
  },
  recent_events: {
    description: "Queue events (claims, releases, moves; actor, outcome, from, to) in the last N hours, newest first.",
    schema: { type: "object", properties: { hours: { type: "number" }, limit: { type: "integer" } } },
    run: async ({ hours = 24, limit = 50 }) => {
      const since = new Date(Date.now() - clamp(hours, 0, 24 * 60, 24) * 3600e3).toISOString().slice(0, 19) + "Z";
      const ev = [];
      for (const i of (await queue()).items) for (const r of i.runs || []) if ((r.at || "") >= since)
        ev.push({ item: i.id, at: r.at, worker: r.worker, outcome: r.outcome, from: r.from, to: r.to, result: String(r.result || "").slice(0, 200) });
      return ev.sort((a, b) => (b.at || "").localeCompare(a.at || "")).slice(0, clamp(limit, 1, 500, 50));
    },
  },
  findings: {
    description: "The findings register (audit/findings/register.jsonl), filtered by status and id prefix.",
    schema: { type: "object", properties: { status: { type: "string" }, series: { type: "string" }, limit: { type: "integer" } } },
    run: async ({ status = "", series = "", limit = 50 }) =>
      (await get("audit/findings/register.jsonl")).split("\n").filter(Boolean).map((l) => JSON.parse(l))
        .filter((f) => (!status || f.status === status) && (!series || String(f.id || "").startsWith(series))).slice(0, clamp(limit, 1, 500, 50)),
  },
  read_note: {
    description: "One file under audit/notes, audit/fabric, audit/spine or audit/findings, or QUEUE.md (queue branch), first 60 kB. Start with audit/fabric/README.md.",
    schema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
    run: async ({ path }) => {
      const p = String(path || "");
      if (p.includes("..") || p.includes("\\") || p.startsWith("/") || !READABLE.some((r) => p === r || p.startsWith(r)))
        return "refused: only audit/notes/, audit/fabric/, audit/spine/, audit/findings/ and audit/2026-11-17/QUEUE.md are readable";
      return (await get(p)).slice(0, 60000);
    },
  },
};

const rpc = (id, result) => ({ jsonrpc: "2.0", id, result });
const rpcErr = (id, code, message) => ({ jsonrpc: "2.0", id, error: { code, message } });

async function handle(msg) {
  const { id, method, params = {} } = msg;
  if (method === "initialize")
    return rpc(id, { protocolVersion: params.protocolVersion || "2025-06-18", capabilities: { tools: {} },
      serverInfo: { name: "forge", version: "0.1.0" },
      instructions: "Read-only view of the Forge's queue, events, findings and notes. Each record names who made it; you are not that recorder. Cite item ids and paths." });
  if (method === "notifications/initialized" || (method || "").startsWith("notifications/")) return null;
  if (method === "ping") return rpc(id, {});
  if (method === "tools/list")
    return rpc(id, { tools: Object.entries(TOOLS).map(([name, t]) => ({ name, description: t.description, inputSchema: t.schema })) });
  if (method === "tools/call") {
    const t = TOOLS[params.name];
    if (!t) return rpcErr(id, -32602, `unknown tool ${params.name}`);
    try {
      const out = await t.run(params.arguments || {});
      return rpc(id, { content: [{ type: "text", text: typeof out === "string" ? out : JSON.stringify(out, null, 1) }] });
    } catch (e) {
      return rpc(id, { content: [{ type: "text", text: `error: ${e.message}` }], isError: true });
    }
  }
  return rpcErr(id, -32601, `method not found: ${method}`);
}

export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname !== "/mcp") return new Response("Forge connector: MCP endpoint at /mcp (read-only).\n", { status: 200 });
    if (request.method === "GET") return new Response("method not allowed", { status: 405 });   // no server-sent stream
    if (request.method !== "POST") return new Response("method not allowed", { status: 405 });
    let body;
    try { body = await request.json(); } catch { return Response.json(rpcErr(null, -32700, "parse error"), { status: 400 }); }
    const batch = Array.isArray(body) ? body : [body];
    const out = (await Promise.all(batch.map(handle))).filter(Boolean);
    if (!out.length) return new Response(null, { status: 202 });
    return Response.json(Array.isArray(body) ? out : out[0]);
  },
};
