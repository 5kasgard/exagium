import { handle as discordHandle } from "./discord.js";
// Forge act: the write half of the Forge connector (owner 2026-11-25: "as powerful as this chat"). A remote MCP
// server (Streamable HTTP, JSON responses) that lets a chat the owner uses do things in the repo, never on its own
// authority: it opens issues, comments, opens PRs from workbench/* branches and starts allow-listed workflows.
// It cannot merge, push to main, or touch workflows, agent config or itself; the owner merges.
// Auth: every call needs `Authorization: Bearer <ACT_TOKEN>` (a Worker secret only the owner's chat holds).
// Every write names who asked (`actor`, the calling model) and that it came through forge-act (audit/fabric/README.md).
const REPO = "5kasgard/exagium";
const API = "https://api.github.com";
const BLOCKED = [".github/", "agent/", "fabric/act/", "fabric/worker/", "fabric/chat/"];
const MAX_FILES = 20, MAX_BYTES = 200_000;

export function pathOk(p) {   // review 2026-11-25: "./" segments were normalised away by the URL parser
  p = String(p || "");
  if (!p || p.startsWith("/") || p.includes("\\") || /[\x00-\x1f?#%]/.test(p)) return false;
  if (p.split("/").some((s) => s === "" || s === "." || s === "..")) return false;
  const lp = p.toLowerCase();
  if (BLOCKED.some((b) => lp.startsWith(b))) return false;
  if (/(^|\/)\.env|\.dev\.vars|\.npmrc|id_[rde]+sa|secret|credential|token|\.pem$|\.key$|\.p12$/i.test(p)) return false;
  return true;
}
export const cleanActor = (a) => String(a || "").replace(/[^\w.\- ]/g, "").slice(0, 80);
// The stamp leads every body: GitHub shows these writes as the owner's, so nothing a model writes may start a
// comment (opencode.yml's /oc trigger reads the first characters; review 2026-11-25).
export const stamp = (actor, now = new Date()) =>
  `[forge-act · recorded_by: ${cleanActor(actor)} · ${now.toISOString().slice(0, 19)}Z]\n\n`;
const signed = (a) => stamp(a.actor) + a.body;
const titled = (t) => `forge-act: ${String(t).replace(/[\r\n]/g, " ").slice(0, 190)}`;
const b64 = (s) => btoa(String.fromCharCode(...new TextEncoder().encode(s)));

async function gh(env, path, method = "GET", body) {
  const r = await fetch(API + path, { method, body: body && JSON.stringify(body), headers: {
    authorization: `Bearer ${env.GH_TOKEN}`, accept: "application/vnd.github+json", "user-agent": "forge-act",
    "x-github-api-version": "2022-11-28", ...(body ? { "content-type": "application/json" } : {}) } });
  const j = r.status === 204 ? {} : await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(`GitHub ${method} ${path.split("?")[0]}: HTTP ${r.status} ${String(j.message || "").slice(0, 160)}`);
  return j;
}
const need = (a, ...k) => {
  for (const x of k) if (!a[x] || !String(a[x]).trim()) throw new Error(`missing ${x}`);
  if (!cleanActor(a.actor).trim()) throw new Error("missing actor");
};
const ACTOR = { actor: { type: "string", description: "Who is asking: your model name, e.g. 'claude-sonnet-4.5 in LobeHub'." } };

// Per-workflow limits on inputs (INPUT_RULES in wrangler.toml): {"wf.yml": {"input": ["allowed", ...]}}.
// A listed input must be given and be one of the values (sandbox-eval: only "check", never create or kill).
export function inputProblem(env, workflow, inputs) {
  let rules = {};
  try { rules = JSON.parse(env.INPUT_RULES || "{}"); } catch { return "INPUT_RULES unreadable: refusing"; }
  for (const [k, vals] of Object.entries(rules[workflow] || {})) {
    if (!vals.includes(String(inputs[k] ?? ""))) return `${workflow}: input ${k} must be one of ${vals.join(", ")}`;
  }
  return null;
}

export const TOOLS = {
  create_issue: {
    description: "Open an issue in the Forge repo (a task, finding or question for the owner and workers).",
    schema: { type: "object", properties: { title: { type: "string" }, body: { type: "string" }, ...ACTOR }, required: ["title", "body", "actor"] },
    run: async (a, env) => { need(a, "title", "body", "actor");
      const j = await gh(env, `/repos/${REPO}/issues`, "POST", { title: titled(a.title), body: signed(a) });
      return { issue: j.number, url: j.html_url }; },
  },
  comment: {
    description: "Comment on an issue or pull request by number.",
    schema: { type: "object", properties: { number: { type: "integer" }, body: { type: "string" }, ...ACTOR }, required: ["number", "body", "actor"] },
    run: async (a, env) => { need(a, "number", "body", "actor");
      const j = await gh(env, `/repos/${REPO}/issues/${+a.number}/comments`, "POST", { body: signed(a) });
      return { comment: j.id, url: j.html_url }; },
  },
  open_pr: {
    description: "Propose file changes as a pull request from a new branch workbench/<name> to main. Up to 20 files, 200 kB each; " +
      "whole-file contents. Refused: .github/, agent/, fabric/act|worker|chat/, secret-like paths. The owner reviews and merges.",
    schema: { type: "object", required: ["name", "title", "body", "files", "actor"], properties: {
      name: { type: "string", description: "branch suffix: letters, digits, - and _" }, title: { type: "string" }, body: { type: "string" },
      files: { type: "array", items: { type: "object", required: ["path", "content"], properties: { path: { type: "string" }, content: { type: "string" } } } },
      ...ACTOR } },
    run: async (a, env) => { need(a, "name", "title", "body", "actor");
      const name = String(a.name);
      if (!/^[A-Za-z0-9_-]{1,60}$/.test(name)) throw new Error("name: letters, digits, - and _ only");
      const files = Array.isArray(a.files) ? a.files : [];
      if (!files.length || files.length > MAX_FILES) throw new Error(`files: 1 to ${MAX_FILES}`);
      for (const f of files) {
        if (!pathOk(f.path)) throw new Error(`refused path: ${f.path}`);
        if (new TextEncoder().encode(String(f.content ?? "")).length > MAX_BYTES) throw new Error(`too large: ${f.path}`);
      }
      const branch = `workbench/${name}`;
      const main = await gh(env, `/repos/${REPO}/git/ref/heads/main`);
      await gh(env, `/repos/${REPO}/git/refs`, "POST", { ref: `refs/heads/${branch}`, sha: main.object.sha });
      for (const f of files) {
        let sha;
        try { sha = (await gh(env, `/repos/${REPO}/contents/${encodeURI(f.path)}?ref=${branch}`)).sha; } catch { sha = undefined; }
        await gh(env, `/repos/${REPO}/contents/${encodeURI(f.path)}`, "PUT",
          { message: `${titled(a.title)} (${f.path})\n\nrecorded_by: ${cleanActor(a.actor)} via forge-act`, content: b64(String(f.content ?? "")), branch, ...(sha ? { sha } : {}) });
      }
      const pr = await gh(env, `/repos/${REPO}/pulls`, "POST", { title: titled(a.title), head: branch, base: "main", body: signed(a) });
      return { pr: pr.number, url: pr.html_url, branch }; },
  },
  run_workflow: {
    description: "Start an allow-listed workflow on main (see allowed in the error if refused).",
    schema: { type: "object", properties: { workflow: { type: "string" }, inputs: { type: "object" }, ...ACTOR }, required: ["workflow", "actor"] },
    run: async (a, env) => { need(a, "workflow", "actor");
      const allowed = String(env.ALLOW_WORKFLOWS || "").split(",").map((s) => s.trim()).filter(Boolean);
      if (!allowed.includes(a.workflow)) throw new Error(`not allowed: ${a.workflow}; allowed: ${allowed.join(", ") || "none"}`);
      const bad = inputProblem(env, a.workflow, a.inputs || {});
      if (bad) throw new Error(bad);
      await gh(env, `/repos/${REPO}/actions/workflows/${a.workflow}/dispatches`, "POST", { ref: "main", inputs: a.inputs || {} });
      return { started: a.workflow, note: `runs at https://github.com/${REPO}/actions/workflows/${a.workflow}` }; },
  },
};

const rpc = (id, result) => ({ jsonrpc: "2.0", id, result });
const rpcErr = (id, code, message) => ({ jsonrpc: "2.0", id, error: { code, message } });

export async function handle(msg, env) {
  const { id, method, params = {} } = msg;
  if (method === "initialize")
    return rpc(id, { protocolVersion: params.protocolVersion || "2025-06-18", capabilities: { tools: {} },
      serverInfo: { name: "forge-act", version: "0.1.0" },
      instructions: "Write tools for the Forge repo. Pass your own model name as actor. Changes go through issues and PRs " +
        "that the owner reviews; you cannot merge. Read state with the read-only Forge connector first." });
  if ((method || "").startsWith("notifications/")) return null;
  if (method === "ping") return rpc(id, {});
  if (method === "tools/list")
    return rpc(id, { tools: Object.entries(TOOLS).map(([name, t]) => ({ name, description: t.description, inputSchema: t.schema })) });
  if (method === "tools/call") {
    const t = TOOLS[params.name];
    if (!t) return rpcErr(id, -32602, `unknown tool ${params.name}`);
    try {
      const out = await t.run(params.arguments || {}, env);
      return rpc(id, { content: [{ type: "text", text: JSON.stringify(out, null, 1) }] });
    } catch (e) {
      return rpc(id, { content: [{ type: "text", text: `error: ${e.message}` }], isError: true });
    }
  }
  return rpcErr(id, -32601, `method not found: ${method}`);
}

const digest = async (s) => new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)));
export async function authorized(request, env) {   // digests: equal length, so the token's length is not timed
  const want = env.ACT_TOKEN || "";
  if (want.length < 32) return false;               // a weak or missing token refuses everything
  const got = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
  return crypto.subtle.timingSafeEqual(await digest(got), await digest(want));
}

// The clock: a Cloudflare cron trigger starts the workflows CRON_JOBS lists for that cron string (wrangler.toml says
// why). Only workflow_dispatch on main, with no inputs; nothing else is reachable from here.
export function cronTargets(env, cron) {
  let jobs = {};
  try { jobs = JSON.parse(env.CRON_JOBS || "{}"); } catch { return []; }
  const list = Array.isArray(jobs[cron]) ? jobs[cron] : [];
  return list.filter((w) => /^[a-z0-9][a-z0-9-]*\.yml$/.test(w));
}

export default {
  async scheduled(event, env, ctx) {
    if (!env.GH_TOKEN) { console.log(JSON.stringify({ cron: event.cron, error: "GH_TOKEN missing" })); return; }
    for (const w of cronTargets(env, event.cron)) {
      try {
        await gh(env, `/repos/${REPO}/actions/workflows/${w}/dispatches`, "POST", { ref: "main" });
        console.log(JSON.stringify({ cron: event.cron, dispatched: w }));
      } catch (e) {
        console.log(JSON.stringify({ cron: event.cron, workflow: w, error: String(e && e.message || e).slice(0, 200) }));
      }
    }
  },
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.pathname === "/discord" && request.method === "POST") return discordHandle(request, env, gh, REPO, ctx);
    if (url.pathname !== "/mcp") return new Response("Forge act: MCP endpoint at /mcp (bearer token required).\n");
    if (request.method !== "POST") return new Response("method not allowed", { status: 405 });
    if (!(await authorized(request, env))) return new Response("unauthorized", { status: 401 });
    if (!env.GH_TOKEN) return Response.json(rpcErr(null, -32000, "not configured: GH_TOKEN"), { status: 503 });
    let body;
    try { body = await request.json(); } catch { return Response.json(rpcErr(null, -32700, "parse error"), { status: 400 }); }
    const batch = Array.isArray(body) ? body : [body];
    if (batch.length > 4) return Response.json(rpcErr(null, -32600, "batch of at most 4"), { status: 400 });
    const out = [];
    for (const m of batch) { const r = await handle(m, env); if (r) out.push(r); }   // one at a time
    if (!out.length) return new Response(null, { status: 202 });
    return Response.json(Array.isArray(body) ? out : out[0]);
  },
};
