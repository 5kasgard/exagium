// Forge box: persistent sandboxes as MCP tools (#191; owner 2026-11-26). Any MCP client (the LobeHub workbench, the
// Claude app, workers) opens a box BY NAME and gets the same machine back: same files, same running processes. The
// name lives in the sandbox's own E2B metadata, not in any chat, so a new chat or another agent reconnects to it.
// Idle boxes pause themselves (no compute billing while paused) and wake on the next call.
// Auth: `Authorization: Bearer <BOX_TOKEN>` (its own secret, FORGE_BOX_TOKEN, not shared with forge-act: review). Limits: MAX_BOXES live boxes, a command timeout cap, output capped.
// Every command is appended to /home/user/.forge-box.log in the box with the caller's actor and time (provenance).
import { Sandbox } from "e2b";

const MAX_OUT = 20000;
export const cleanName = (n) => String(n || "").toLowerCase().replace(/[^a-z0-9-]/g, "").slice(0, 40);
export const cleanActor = (a) => String(a || "").replace(/[^\w.\- ]/g, "").slice(0, 80);
const clip = (s) => { s = String(s ?? ""); return s.length > MAX_OUT ? s.slice(0, MAX_OUT / 2) + `\n…[${s.length - MAX_OUT} chars cut]…\n` + s.slice(-MAX_OUT / 2) : s; };
const opts = (env) => ({ apiKey: env.E2B_API_KEY });

const need = (n) => { if (!n) throw new Error("name: letters, digits and - only"); return n; };   // review: "" skipped the filter
async function find(env, name, S) {
  const items = await S.list({ ...opts(env), query: { metadata: { forge: "box", forge_box: need(name) }, state: ["running", "paused"] } }).nextItems();
  return items[0];
}
const IDLE = (env) => Number(env.IDLE_MIN || 10) * 60000;
async function open(env, name, S, create = true) {
  const hit = await find(env, name, S);
  if (hit) return { sb: await S.connect(hit.sandboxId, { ...opts(env), timeoutMs: IDLE(env) }), created: false };   // resumes; a fresh idle window
  if (!create) throw new Error(`no box named ${name}; open it first`);
  const live = await S.list({ ...opts(env), query: { metadata: { forge: "box" }, state: ["running", "paused"] } }).nextItems();
  if (live.length >= Number(env.MAX_BOXES || 3)) throw new Error(`box limit reached (${live.length}); kill one first: ${live.map((x) => x.metadata?.forge_box).join(", ")}`);
  // No autoResume: outside traffic must not wake a box (review); connect() is the only way back in.
  const sb = await S.create({ ...opts(env), timeoutMs: IDLE(env), lifecycle: { onTimeout: "pause" }, metadata: { forge: "box", forge_box: name } });
  const after = await S.list({ ...opts(env), query: { metadata: { forge: "box" }, state: ["running", "paused"] } }).nextItems();
  const same = after.filter((x) => x.metadata?.forge_box === name).map((x) => x.sandboxId).sort();
  if (after.length > Number(env.MAX_BOXES || 3) || (same.length > 1 && same[0] !== sb.sandboxId)) {   // lost a race: undo
    await S.kill(sb.sandboxId, opts(env));
    throw new Error("box limit or a parallel open of the same name; try again");
  }
  return { sb, created: true };
}
const ACTOR = { actor: { type: "string", description: "Who is asking: your model name, e.g. 'gpt-oss-120b in LobeHub'." } };
const NAME = { name: { type: "string", description: "Box name, e.g. 'forge-main'. Same name, same machine, across chats." } };

export const TOOLS = {
  box_open: {
    description: "Open (or create) a persistent Linux box by name: 2 vCPU, ~0.5 GB RAM, Debian with Python, sudo and internet. " +
      "It keeps files and running processes while idle (it pauses itself after ~10 min) and resumes on the next call.",
    schema: { type: "object", properties: { ...NAME, ...ACTOR }, required: ["name", "actor"] },
    run: async (a, env, S) => { const n = need(cleanName(a.name));
      const { sb, created } = await open(env, n, S);
      const u = await sb.commands.run("uname -srm; nproc; df -h /home/user | tail -1", { timeoutMs: 30000 });
      return { box: n, created, machine: u.stdout.trim() }; },
  },
  box_exec: {
    description: "Run a shell command in a box (bash -lc). Returns exit code, stdout and stderr (each capped at 20k chars). " +
      "For long jobs start them in the background (nohup … &) and check later; processes survive pauses.",
    schema: { type: "object", required: ["name", "command", "actor"], properties: { ...NAME, command: { type: "string" },
      timeout_s: { type: "integer", description: "default 120, max 600" }, cwd: { type: "string" }, ...ACTOR } },
    run: async (a, env, S) => { const n = need(cleanName(a.name)); const { sb } = await open(env, n, S, false);
      const t = Math.min(Math.max(Number(a.timeout_s) || 120, 1), 600) * 1000;
      const f = `/home/user/.forge-box/${crypto.randomUUID()}.cmd`;   // one file per call: parallel calls cannot swap commands
      await sb.files.write(f, String(a.command));
      // A note for people in the box, not an audit trail (actor is self-declared and the box can edit it).
      await sb.commands.run(`printf '%s\\t%s\\t%s\\n' "$(date -u +%FT%TZ)" "$ACTOR" "$(head -c 300 "$F" | tr '\\n\\t' '  ')" >> /home/user/.forge-box.log`,
        { envs: { ACTOR: cleanActor(a.actor), F: f }, timeoutMs: 15000 });
      try {   // output capped inside the box, so a huge stdout cannot exhaust the Worker
        const r = await sb.commands.run(`bash -l "$F" > >(head -c ${MAX_OUT * 2}) 2> >(head -c ${MAX_OUT} >&2)`,
          { envs: { F: f }, timeoutMs: t, cwd: a.cwd || "/home/user" });
        return { exit: r.exitCode, stdout: clip(r.stdout), stderr: clip(r.stderr) };
      } catch (e) {   // the SDK throws on a non-zero exit; keep its output
        return { exit: e.exitCode ?? -1, stdout: clip(e.stdout), stderr: clip(e.stderr || e.message) };
      } },
  },
  box_write: {
    description: "Write a text file in a box (whole content).",
    schema: { type: "object", required: ["name", "path", "content", "actor"], properties: { ...NAME, path: { type: "string" }, content: { type: "string" }, ...ACTOR } },
    run: async (a, env, S) => { const { sb } = await open(env, need(cleanName(a.name)), S, false);
      if ( TextEncoder().encode(String(a.content)).length > 2_000_000) throw new Error("content over 2 MB");
      await sb.files.write(String(a.path), String(a.content)); return { written: a.path }; },
  },
  box_read: {
    description: "Read a text file from a box (capped at 20k chars).",
    schema: { type: "object", required: ["name", "path", "actor"], properties: { ...NAME, path: { type: "string" }, ...ACTOR } },
    run: async (a, env, S) => { const { sb } = await open(env, need(cleanName(a.name)), S, false);
      const r = await sb.commands.run(`head -c ${MAX_OUT * 2} -- "$P"`, { envs: { P: String(a.path) }, timeoutMs: 30000 });
      return { path: a.path, content: clip(r.stdout) }; },
  },
  box_list: {
    description: "List boxes with their state (running or paused).",
    schema: { type: "object", properties: { ...ACTOR }, required: ["actor"] },
    run: async (a, env, S) => { const xs = await S.list({ ...opts(env), query: { metadata: { forge: "box" }, state: ["running", "paused"] } }).nextItems();
      return xs.map((x) => ({ box: x.metadata?.forge_box, state: x.state, started: x.startedAt })); },
  },
  box_pause: {
    description: "Pause a box now (keeps files and processes; no compute billing while paused).",
    schema: { type: "object", properties: { ...NAME, ...ACTOR }, required: ["name", "actor"] },
    run: async (a, env, S) => { const hit = await find(env, cleanName(a.name), S); if (!hit) throw new Error("no such box");
      await S.pause(hit.sandboxId, opts(env)); return { paused: a.name }; },
  },
  box_kill: {
    description: "Delete a box and everything in it. Commit or upload anything worth keeping first.",
    schema: { type: "object", properties: { ...NAME, ...ACTOR }, required: ["name", "actor"] },
    run: async (a, env, S) => { const hit = await find(env, cleanName(a.name), S); if (!hit) throw new Error("no such box");
      await S.kill(hit.sandboxId, opts(env)); return { killed: a.name }; },
  },
};

const rpc = (id, result) => ({ jsonrpc: "2.0", id, result });
const rpcErr = (id, code, message) => ({ jsonrpc: "2.0", id, error: { code, message } });

export async function handle(msg, env, S = Sandbox) {
  const { id, method, params = {} } = msg;
  if (method === "initialize")
    return rpc(id, { protocolVersion: params.protocolVersion || "2025-06-18", capabilities: { tools: {} },
      serverInfo: { name: "forge-box", version: "0.1.0" },
      instructions: "Persistent Linux boxes by name. Pass your model name as actor. The repo is the record: commit or upload results; a box is a cache." });
  if ((method || "").startsWith("notifications/")) return null;
  if (method === "ping") return rpc(id, {});
  if (method === "tools/list")
    return rpc(id, { tools: Object.entries(TOOLS).map(([name, t]) => ({ name, description: t.description, inputSchema: t.schema })) });
  if (method === "tools/call") {
    const t = TOOLS[params.name];
    if (!t) return rpcErr(id, -32602, `unknown tool ${params.name}`);
    const a = params.arguments || {};
    if (!cleanActor(a.actor).trim()) return rpc(id, { content: [{ type: "text", text: "error: missing actor" }], isError: true });
    try {
      return rpc(id, { content: [{ type: "text", text: JSON.stringify(await t.run(a, env, S), null, 1) }] });
    } catch (e) {
      return rpc(id, { content: [{ type: "text", text: `error: ${String(e.message || e).slice(0, 400)}` }], isError: true });
    }
  }
  return rpcErr(id, -32601, `method not found: ${method}`);
}

const digest = async (s) => new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)));
export async function authorized(request, env) {
  const want = env.BOX_TOKEN || "";
  if (want.length < 32) return false;
  const got = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
  return crypto.subtle.timingSafeEqual(await digest(got), await digest(want));
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/mcp") return new Response("Forge box: MCP endpoint at /mcp (bearer token required).\n");
    if (request.method !== "POST") return new Response("method not allowed", { status: 405 });
    if (!(await authorized(request, env))) return new Response("unauthorized", { status: 401 });
    if (!env.E2B_API_KEY) return Response.json(rpcErr(null, -32000, "not configured: E2B_API_KEY"), { status: 503 });
    let body;
    try { body = await request.json(); } catch { return Response.json(rpcErr(null, -32700, "parse error"), { status: 400 }); }
    const batch = Array.isArray(body) ? body : [body];
    if (batch.length > 4) return Response.json(rpcErr(null, -32600, "batch of at most 4"), { status: 400 });
    const out = [];
    for (const m of batch) { const r = await handle(m, env); if (r) out.push(r); }
    if (!out.length) return new Response(null, { status: 202 });
    return Response.json(Array.isArray(body) ? out : out[0]);
  },
};
