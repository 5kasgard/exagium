// node fabric/act/test_discord.mjs : the Discord gate and commands, with a real Ed25519 key and a fake GitHub
import assert from "node:assert/strict";
import { handle, COMMANDS, RUN } from "./discord.js";

const kp = await crypto.subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"]);
const pub = Buffer.from(await crypto.subtle.exportKey("raw", kp.publicKey)).toString("hex");
const OWNER = "111", calls = [];
const env = { DISCORD_PUBLIC_KEY: pub, DISCORD_OWNER_ID: OWNER };
const gh = async (e, path, method = "GET", body) => {
  calls.push({ path, method, body });
  if (path.includes("/actions/runs")) return { workflow_runs: [{ name: "worker", status: "completed", conclusion: "success", updated_at: "2026-11-26T11:00:00Z" }] };
  if (path.includes("/pulls")) return [{ number: 7, title: "@everyone fix", user: { login: "bot" }, html_url: "https://x/7" }];
  return {};
};
const edits = [];
const post = async (url, init) => { edits.push({ url, body: JSON.parse(init.body) }); return new Response(null, { status: 200 }); };
async function send(obj, opts = {}) {
  const { sign = true, tamper = false } = opts;
  const body = JSON.stringify(obj), ts = String(opts.ts ?? Math.floor(Date.now() / 1000));
  const sig = Buffer.from(await crypto.subtle.sign("Ed25519", kp.privateKey, new TextEncoder().encode(ts + body))).toString("hex");
  const headers = sign ? { "x-signature-ed25519": sig, "x-signature-timestamp": ts } : {};
  const r = await handle(new Request("https://act/discord", { method: "POST", headers, body: tamper ? body + " " : body }), env, gh, "o/r", undefined, post);
  const j = r.status === 200 ? await r.json() : null;
  // a deferred answer (type 5) is followed by an edit carrying the text: expose that as data.content
  if (j && j.type === 5) return { status: r.status, j: { type: 5, data: { flags: j.data.flags, ...edits.at(-1).body } } };
  return { status: r.status, j };
}
const cmd = (name, options = [], uid = OWNER) => ({ type: 2, application_id: "app", token: "tok", member: { user: { id: uid } }, data: { name, options } });

assert.equal((await send({ type: 1 }, { sign: false })).status, 401, "unsigned refused");
assert.equal((await send({ type: 1 }, { tamper: true })).status, 401, "tampered body refused");
assert.deepEqual((await send({ type: 1 })).j, { type: 1 }, "PING answered");
assert.equal((await send({ type: 1 }, { ts: Math.floor(Date.now() / 1000) - 3600 })).status, 401, "an hour-old signed request (replay) refused");
let r = await send(cmd("ask", [{ name: "agent", value: "free" }, { name: "prompt", value: "hi" }], "999"));
assert.match(r.j.data.content, /Not allowed/); assert.equal(calls.length, 0, "a stranger starts nothing");
r = await send(cmd("ask", [{ name: "agent", value: "free" }, { name: "prompt", value: "hi" }]));
assert.match(r.j.data.content, /Scout/); assert.equal(r.j.data.flags, 64);
assert.equal(edits.at(-1).url, "https://discord.com/api/v10/webhooks/app/tok/messages/@original");
assert.deepEqual(calls.pop().body, { ref: "main", inputs: { mode: "run", agent: "free", prompt: "hi" } });
r = await send(cmd("ask", [{ name: "agent", value: "root" }, { name: "prompt", value: "hi" }]));
assert.match(r.j.data.content, /Need an agent/);
r = await send(cmd("ask", [{ name: "agent", value: "free" }, { name: "prompt", value: "x".repeat(1501) }]));
assert.match(r.j.data.content, /at most 1,500/);
r = await send(cmd("run", [{ name: "job", value: "worker.yml" }])); assert.match(r.j.data.content, /Unknown job/);
r = await send(cmd("run", [{ name: "job", value: "workbench-manage.yml" }])); assert.match(r.j.data.content, /Unknown job/);
r = await send(cmd("run", [{ name: "job", value: "schedules.yml" }])); assert.match(r.j.data.content, /Started/);
assert.ok(calls.pop().path.endsWith("/actions/workflows/schedules.yml/dispatches"));
r = await send(cmd("status")); assert.match(r.j.data.content, /worker/);
r = await send(cmd("prs")); assert.ok(!r.j.data.content.includes("@everyone")); assert.deepEqual(r.j.data.allowed_mentions, { parse: [] });
assert.equal((await handle(new Request("https://a", { method: "POST", headers: {}, body: "{}" }), {}, gh, "o/r")).status, 401, "no key configured: refuse");
const runChoices = COMMANDS.find((c) => c.name === "run").options[0].choices.map((c) => c.value);
assert.ok(runChoices.every((v) => RUN.includes(v)) && !runChoices.includes("workbench-manage.yml"));
// "Reply to agent": a form for the owner on an agent card, then the same topic continues
const card = { author: { username: "Scout" }, embeds: [{ footer: { text: "gpt-oss-120b · 52 s · topic tpc_AB12" } }] };
const menu = (uid = OWNER, m = card) => ({ type: 2, application_id: "app", token: "tok", member: { user: { id: uid } },
  data: { type: 3, name: "Reply to agent", target_id: "m1", resolved: { messages: { m1: m } } } });
r = await send(menu()); assert.equal(r.j.type, 9); assert.equal(r.j.data.custom_id, "reply:free:t:tpc_AB12");
r = await send(menu("999")); assert.match(r.j.data.content, /Not allowed/);
r = await send(menu(OWNER, { author: { username: "Forge" }, embeds: [] })); assert.match(r.j.data.content, /not an agent reply/);
const form = (uid = OWNER, cid = "reply:free:t:tpc_AB12", text = "and the open PRs?") => ({ type: 5, application_id: "app", token: "tok",
  member: { user: { id: uid } }, data: { custom_id: cid, components: [{ type: 1, components: [{ type: 4, custom_id: "text", value: text }] }] } });
r = await send(form()); assert.match(r.j.data.content, /same chat/);
assert.deepEqual(calls.pop().body, { ref: "main", inputs: { mode: "run", agent: "free", prompt: "and the open PRs?", topic: "tpc_AB12" } });
const before = calls.length;
r = await send(form("999")); assert.match(r.j.data.content, /Not allowed/);
r = await send(form(OWNER, "reply:root:x;rm")); assert.match(r.j.data.content, /Could not read/);
assert.equal(calls.length, before, "no dispatch for a stranger or a bad form");
// forge-agent path: /ask goes to the AGENT binding by default; its cards carry "chat …" and replies continue it
const agentCalls = [];
env.AGENT = { fetch: async (u, init) => { agentCalls.push(JSON.parse(init.body)); return Response.json({ chat: "c0ffee1234567890", queued: 1 }); } };
r = await send(cmd("ask", [{ name: "agent", value: "coder" }, { name: "prompt", value: "fix it" }]));
assert.match(r.j.data.content, /being written/); assert.deepEqual(agentCalls.pop(), { agent: "coder", prompt: "fix it" });
r = await send(cmd("ask", [{ name: "agent", value: "coder" }, { name: "prompt", value: "fix it" }, { name: "via", value: "lobehub" }]));
assert.match(r.j.data.content, /Asked \*\*Coder/); assert.equal(agentCalls.length, 0);
assert.equal(calls.pop().body.inputs.mode, "run");
const fcard = { webhook_id: "w", author: { username: "Stradivari · Coder" }, embeds: [{ footer: { text: "big-pickle · 9 s · done · chat c0ffee1234567890" } }] };
r = await send(menu(OWNER, fcard)); assert.equal(r.j.data.custom_id, "reply:any:c:c0ffee1234567890");
r = await send(form(OWNER, "reply:any:c:c0ffee1234567890", "now add a test"));
assert.match(r.j.data.content, /same chat/);
assert.deepEqual(agentCalls.pop(), { prompt: "now add a test", chat: "c0ffee1234567890" });
r = await send(form(OWNER, "reply:any:t:tpc_X", "x")); assert.match(r.j.data.content, /Could not read/);
r = await send(cmd("activity")); assert.match(r.j.data.content, /not connected/);
console.log("test_discord: ok");
