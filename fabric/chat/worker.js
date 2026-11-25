// Forge chat (queue P36, owner 2026-11-25): the owner's own chat, on his own keys, outside any vendor's chat limits.
// GET / serves the page; POST /api/chat runs a tool-calling loop against the chosen model, with the read-only Forge
// connector (connector.exagium.xyz/mcp) as its tools. Only the owner gets in: Cloudflare Access in front, and every
// API call re-verifies the Access JWT (RS256 against the team's certs, audience ACCESS_AUD, email OWNER_EMAIL).
// Keys are Worker secrets, set by .github/workflows/chat-deploy.yml from the repo's secrets; the browser never sees them.
import PAGE from "./page.html";
import DASH from "./dash.html";   // the Forge dashboard (fabric/dash): owner-only like the chat
const DASH_DATA = "https://raw.githubusercontent.com/5kasgard/exagium/fabric-data/dash.json";

const CONNECTOR = "https://connector.exagium.xyz/mcp";
// provider -> OpenAI-compatible base URL and the secret holding its key
const PROVIDERS = {
  openrouter: { base: "https://openrouter.ai/api/v1", key: "OPENROUTER_API_KEY" },
  groq: { base: "https://api.groq.com/openai/v1", key: "GROQ_API_KEY" },
  nvidia: { base: "https://integrate.api.nvidia.com/v1", key: "NVIDIA_API_KEY" },
  anthropic: { base: "https://api.anthropic.com/v1", key: "ANTHROPIC_API_KEY" },
  google: { base: "https://generativelanguage.googleapis.com/v1beta/openai", key: "GEMINI_API_KEY" },
};
// the model picker; the owner edits this list (paid ones are labelled, and refused while ALLOW_PAID is not "true")
export const MODELS = [
  { id: "openrouter:openai/gpt-oss-120b:free", paid: false },
  { id: "groq:openai/gpt-oss-120b", paid: false },
  { id: "nvidia:nvidia/nemotron-3-ultra-550b-a55b", paid: false },
  { id: "google:gemini-3.5-flash", paid: true },
  { id: "anthropic:claude-sonnet-5-5", paid: true },
  { id: "anthropic:claude-opus-5-5", paid: true },
];
const MAX_STEPS = 8, MAX_TOKENS = 4000, MAX_REQUEST = 200000;   // bytes of conversation per request
const SYSTEM = "You are a model the owner of the Forge (repo 5kasgard/exagium) is chatting with, through his own app. " +
  "Tools give a read-only view of the Forge's queue, events, findings and notes. Records name who made them; you are " +
  "not that recorder, and you act as yourself. Cite item ids and file paths, and say when something is not in the data.";

let certs = null;
const b64u = (s) => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((s.length + 3) % 4)), (c) => c.charCodeAt(0));
export async function verifyAccess(request, env, fetchImpl = fetch) {
  const jwt = request.headers.get("cf-access-jwt-assertion");
  if (!jwt || !env.ACCESS_TEAM || !env.ACCESS_AUD || !env.OWNER_EMAIL) return false;
  const [h, p, s] = jwt.split(".");
  if (!h || !p || !s) return false;
  const head = JSON.parse(new TextDecoder().decode(b64u(h)));
  const body = JSON.parse(new TextDecoder().decode(b64u(p)));
  if (head.alg !== "RS256") return false;
  if (!certs) certs = await (await fetchImpl(`https://${env.ACCESS_TEAM}.cloudflareaccess.com/cdn-cgi/access/certs`)).json();
  const jwk = (certs.keys || []).find((k) => k.kid === head.kid);
  if (!jwk) { certs = null; return false; }
  const key = await crypto.subtle.importKey("jwk", jwk, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]);
  const ok = await crypto.subtle.verify("RSASSA-PKCS1-v1_5", key, b64u(s), new TextEncoder().encode(`${h}.${p}`));
  const aud = Array.isArray(body.aud) ? body.aud : [body.aud];
  const now = Date.now() / 1000;
  return ok && body.iss === `https://${env.ACCESS_TEAM}.cloudflareaccess.com` && aud.includes(env.ACCESS_AUD) &&
    body.exp > now && (body.nbf === undefined || body.nbf <= now + 60) &&
    String(body.email || "").toLowerCase() === String(env.OWNER_EMAIL).toLowerCase();
}

export class ConnectorError extends Error {}
async function mcp(method, params, fetchImpl) {
  // errors surface instead of turning into empty tools or empty results (review in PR #157)
  const r = await fetchImpl(CONNECTOR, { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }) });
  if (r.ok === false) throw new ConnectorError(`connector HTTP ${r.status}`);
  let j;
  try { j = await r.json(); } catch { throw new ConnectorError("connector reply was not JSON"); }
  if (j.error) throw new ConnectorError(`connector error ${j.error.code}: ${String(j.error.message).slice(0, 200)}`);
  if (!j.result) throw new ConnectorError("connector reply had no result");
  return j.result;
}

export async function chat(env, { model, messages }, fetchImpl = fetch) {
  const m = MODELS.find((x) => x.id === model);
  if (!m) return { error: `model not in the list: ${model}` };
  if (m.paid && env.ALLOW_PAID !== "true") return { error: `${model} is paid and ALLOW_PAID is off` };
  const [prov, name] = [model.slice(0, model.indexOf(":")), model.slice(model.indexOf(":") + 1)];
  const p = PROVIDERS[prov], key = env[p.key];
  if (!key) return { error: `no key for ${prov} (${p.key} is not set on the Worker)` };
  if (!Array.isArray(messages) || messages.some((x) => !x || !["user", "assistant"].includes(x.role) || typeof x.content !== "string"))
    return { error: "messages must be a list of {role: user|assistant, content: string}" };
  const listed = (await mcp("tools/list", {}, fetchImpl)).tools || [];
  const allowed = new Set(listed.map((t) => t.name));   // the model may call only what the connector lists
  const tools = listed.map((t) =>
    ({ type: "function", function: { name: t.name, description: t.description, parameters: t.inputSchema } }));
  const convo = [{ role: "system", content: SYSTEM }, ...messages.slice(-40)];
  const trace = [];
  let usage = { prompt_tokens: 0, completion_tokens: 0 };
  for (let step = 0; step < MAX_STEPS; step++) {
    const r = await fetchImpl(`${p.base}/chat/completions`, { method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
      body: JSON.stringify({ model: name, messages: convo, tools, max_tokens: MAX_TOKENS }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) return { error: `${prov} HTTP ${r.status}: ${JSON.stringify(j).slice(0, 300)}`, trace };
    usage.prompt_tokens += j.usage?.prompt_tokens || 0; usage.completion_tokens += j.usage?.completion_tokens || 0;
    const msg = j.choices?.[0]?.message;
    if (!msg) return { error: "no message in the reply", trace };
    convo.push(msg);
    if (!msg.tool_calls?.length) return { reply: msg.content || "", trace, usage, model, paid: m.paid };
    for (const c of msg.tool_calls) {
      const name = c.function?.name;
      let args = null, text;
      try { args = JSON.parse(c.function?.arguments || "{}"); } catch { args = null; }
      if (!allowed.has(name)) text = `refused: ${name} is not a listed tool`;
      else if (args === null || typeof args !== "object" || Array.isArray(args)) text = "refused: tool arguments were not a JSON object";
      else {
        try {
          const res = await mcp("tools/call", { name, arguments: args }, fetchImpl);
          text = (res.content || []).map((x) => x.text).join("\n").slice(0, 30000);
        } catch (e) { text = `tool failed: ${e.message}`; }
      }
      trace.push({ tool: name, args, ok: !text.startsWith("refused") && !text.startsWith("tool failed") });
      convo.push({ role: "tool", tool_call_id: c.id, content: text });
    }
  }
  return { error: `stopped after ${MAX_STEPS} tool steps`, trace, usage };
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (!(await verifyAccess(request, env))) return new Response("forbidden", { status: 403 });
    if (url.pathname === "/" && request.method === "GET")
      return new Response(PAGE, { headers: { "content-type": "text/html; charset=utf-8" } });
    if (url.pathname === "/dash" && request.method === "GET")
      return new Response(DASH, { headers: { "content-type": "text/html; charset=utf-8" } });
    if (url.pathname === "/api/dash") {
      const r = await fetch(DASH_DATA, { cf: { cacheTtl: 120 } });
      return new Response(r.ok ? r.body : JSON.stringify({ error: `dash.json: HTTP ${r.status}` }),
        { status: r.ok ? 200 : 502, headers: { "content-type": "application/json", "cache-control": "no-store" } });
    }
    if (url.pathname === "/api/models") return Response.json({ models: MODELS, allow_paid: env.ALLOW_PAID === "true" });
    if (url.pathname === "/api/chat" && request.method === "POST") {
      const raw = await request.text();
      if (raw.length > MAX_REQUEST) return Response.json({ error: `request over ${MAX_REQUEST} bytes` }, { status: 413 });
      let body;
      try { body = JSON.parse(raw); } catch { return Response.json({ error: "request was not JSON" }, { status: 400 }); }
      try { return Response.json(await chat(env, body)); }
      catch (e) { return Response.json({ error: String(e).slice(0, 300) }, { status: 500 }); }
    }
    return new Response("not found", { status: 404 });
  },
};
