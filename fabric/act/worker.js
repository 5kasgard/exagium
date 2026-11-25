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

export function pathOk(p) {
  p = String(p || "");
  if (!p || p.startsWith("/") || p.includes("..") || p.includes("\\") || p.includes("//")) return false;
  if (BLOCKED.some((b) => p.startsWith(b))) return false;
  if (/(^|\/)\.env|secret|credential|\.pem$|\.key$/i.test(p)) return false;
  return true;
}
export const stamp = (actor, now = new Date()) =>
  `\n\n---\nrecorded_by: ${String(actor).slice(0, 80)} via forge-act (owner's workbench) · ${now.toISOString().slice(0, 19)}Z`;
const b64 = (s) => btoa(String.fromCharCode(...new TextEncoder().encode(s)));

async function gh(env, path, method = "GET", body) {
  const r = await fetch(API + path, { method, body: body && JSON.stringify(body), headers: {
    authorization: `Bearer ${env.GH_TOKEN}`, accept: "application/vnd.github+json", "user-agent": "forge-act",
    "x-github-api-version": "2022-11-28", ...(body ? { "content-type": "application/json" } : {}) } });
  const j = r.status === 204 ? {} : await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(`GitHub ${method} ${path.split("?")[0]}: HTTP ${r.status} ${String(j.message || "").slice(0, 160)}`);
  return j;
}
const need = (a, ...k) => { for (const x of k) if (!a[x] || !String(a[x]).trim()) throw new Error(`missing ${x}`); };
const ACTOR = { actor: { type: "string", description: "Who is asking: your model name, e.g. 'claude-sonnet-4.5 in LobeHub'." } };

export const TOOLS = {
  create_issue: {
    description: "Open an issue in the Forge repo (a task, finding or question for the owner and workers).",
    schema: { type: "object", properties: { title: { type: "string" }, body: { type: "string" }, ...ACTOR }, required: ["title", "body", "actor"] },
    run: async (a, env) => { need(a, "title", "body", "actor");
      const j = await gh(env, `/repos/${REPO}/issues`, "POST", { title: String(a.title).slice(0, 200), body: a.body + stamp(a.actor) });
      return { issue: j.number, url: j.html_url }; },
  },
  comment: {
    description: "Comment on an issue or pull request by number.",
    schema: { type: "object", properties: { number: { type: "integer" }, body: { type: "string" }, ...ACTOR }, required: ["number", "body", "actor"] },
    run: async (a, env) => { need(a, "number", "body", "actor");
      const j = await gh(env, `/repos/${REPO}/issues/${+a.number}/comments`, "POST", { body: a.body + stamp(a.actor) });
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
          { message: `${a.title} (${f.path})\n\nrecorded_by: ${a.actor} via forge-act`, content: b64(String(f.content ?? "")), branch, ...(sha ? { sha } : {}) });
      }
      const pr = await gh(env, `/repos/${REPO}/pulls`, "POST", { title: String(a.title).slice(0, 200), head: branch, base: "main", body: a.body + stamp(a.actor) });
      return { pr: pr.number, url: pr.html_url, branch }; },
  },
  run_workflow: {
    description: "Start an allow-listed workflow on main (see allowed in the error if refused).",
    schema: { type: "object", properties: { workflow: { type: "string" }, inputs: { type: "object" }, ...ACTOR }, required: ["workflow", "actor"] },
    run: async (a, env) => { need(a, "workflow", "actor");
      const allowed = String(env.ALLOW_WORKFLOWS || "").split(",").map((s) => s.trim()).filter(Boolean);
      if (!allowed.includes(a.workflow)) throw new Error(`not allowed: ${a.workflow}; allowed: ${allowed.join(", ") || "none"}`);
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

export async function authorized(request, env) {
  const got = new TextEncoder().encode((request.headers.get("authorization") || "").replace(/^Bearer\s+/i, ""));
  const want = new TextEncoder().encode(env.ACT_TOKEN || "");
  if (!want.length || got.length !== want.length) return false;
  return crypto.subtle.timingSafeEqual(got, want);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/mcp") return new Response("Forge act: MCP endpoint at /mcp (bearer token required).\n");
    if (request.method !== "POST") return new Response("method not allowed", { status: 405 });
    if (!(await authorized(request, env))) return new Response("unauthorized", { status: 401 });
    if (!env.GH_TOKEN) return Response.json(rpcErr(null, -32000, "not configured: GH_TOKEN"), { status: 503 });
    let body;
    try { body = await request.json(); } catch { return Response.json(rpcErr(null, -32700, "parse error"), { status: 400 }); }
    const batch = Array.isArray(body) ? body : [body];
    const out = (await Promise.all(batch.map((m) => handle(m, env)))).filter(Boolean);
    if (!out.length) return new Response(null, { status: 202 });
    return Response.json(Array.isArray(body) ? out : out[0]);
  },
};
