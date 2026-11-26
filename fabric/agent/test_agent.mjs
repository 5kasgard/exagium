// node fabric/agent/test_agent.mjs : the loop's pure parts (stream parsing, tool naming, the card), and the gate.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
const CONFIG = JSON.parse(readFileSync(new URL("./agents.json", import.meta.url)));
// worker.js imports agents.json as a module (wrangler bundles it); give node the same thing
const src = readFileSync(new URL("./worker.js", import.meta.url), "utf8").replace('import CONFIG from "./agents.json";', `const CONFIG = ${JSON.stringify(CONFIG)};`);
const m = await import("data:text/javascript," + encodeURIComponent(src));

// SSE: text in pieces, a tool call split across chunks, usage at the end
let st = { text: "", calls: [], tokens: 0 };
for (const l of [
  'data: {"choices":[{"delta":{"content":"Hel"}}]}', 'data: {"choices":[{"delta":{"content":"lo"}}]}',
  'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"exagium__queue_","arguments":"{\\"a"}}]}}]}',
  'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"summary","arguments":"\\":1}"}}]}}]}',
  'data: {"choices":[],"usage":{"total_tokens":321}}', "data: [DONE]", ": keep-alive", "data: {broken"]) m.foldSSE(st, l);
assert.equal(st.text, "Hello");
assert.deepEqual(st.calls[0], { id: "c1", name: "exagium__queue_summary", args: '{"a":1}' });
assert.equal(st.tokens, 321);

// tool names: server prefix, safe characters, 64 max
assert.equal(m.toolName("box", "box_open"), "box__box_open");
assert.equal(m.toolName("act", "a.b c"), "act__a_b_c");
assert.equal(m.toolName("x", "y".repeat(80)).length, 64);
const t = m.openaiTools("exagium", [{ name: "queue_item", description: "d", inputSchema: { type: "object", properties: { item_id: { type: "string" } } } }, { name: "n" }]);
assert.equal(t[0].function.name, "exagium__queue_item");
assert.deepEqual(t[1].function.parameters, { type: "object", properties: {} });

// card: working vs done, tools line, chat id in the footer (what "Reply to agent" reads back)
const a = CONFIG.agents.free;
const s = { text: "Hi", tools: [{ name: "queue_summary", ok: true, ms: 800 }, { name: "box_exec", ok: null }], tokens: 1200, model: a.model, seconds: 3.2, done: false, error: null, chat: "abc12345def" };
let c = m.card(a, s);
assert.ok(c.description.endsWith("▍")); assert.match(c.fields[0].value, /✓ queue_summary 0.8s · … box_exec/);
assert.match(c.footer.text, /working · chat abc12345def/);
c = m.card(a, { ...s, done: true, text: "x".repeat(5000) });
assert.ok(c.description.length <= 4096); assert.match(c.footer.text, /done · chat abc12345def/);

// the entry: only known agents, a prompt, a well-formed chat id; no GET
const fake = { CHAT: { idFromName: (n) => n, get: () => ({ fetch: async () => Response.json({ queued: 1 }) }) } };
const post = (b) => m.default.fetch(new Request("https://agent/", { method: "POST", body: JSON.stringify(b) }), fake);
assert.equal((await post({ agent: "root", prompt: "x" })).status, 400);
assert.equal((await post({ agent: "free", prompt: " " })).status, 400);
assert.equal((await post({ agent: "free", prompt: "x", chat: "../etc" })).status, 400);
const ok = await (await post({ agent: "free", prompt: "hi" })).json();
assert.match(ok.chat, /^[a-z0-9]{16}$/); assert.equal(ok.queued, 1);
assert.equal((await m.default.fetch(new Request("https://agent/"), fake)).status, 404);
for (const [k, v] of Object.entries(CONFIG.agents)) assert.ok(v.servers.length && v.model && v.provider, k);
// parallel calls with reused index 0 and new ids stay separate
st = { text: "", calls: [], tokens: 0 };
for (const l of ['data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"box__box_list","arguments":"{}"}}]}}]}',
  'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"b","function":{"name":"exagium__queue_summary","arguments":"{}"}}]}}]}']) m.foldSSE(st, l);
assert.deepEqual(st.calls.map((c) => c.name), ["box__box_list", "exagium__queue_summary"]);
// history trim: starts at a user turn, within budget
const h = [{ role: "user", content: "a".repeat(50) }, { role: "assistant", content: "", tool_calls: [{ id: "x" }] }, { role: "tool", tool_call_id: "x", content: "t".repeat(50) }, { role: "user", content: "b" }, { role: "assistant", content: "c" }];
assert.equal(m.trimHistory(h, 10000)[0].role, "user"); assert.equal(m.trimHistory(h, 10000).length, 5);
const cut = m.trimHistory(h, 120); assert.equal(cut[0].content, "b");
assert.deepEqual(m.trimHistory([{ role: "tool", content: "x" }, { role: "user", content: "y" }]), [{ role: "user", content: "y" }]);
// redaction and args
assert.ok(!m.redact("Authorization: Bearer abcdefghijklmnop1234").includes("abcdefghijklmnop1234"));
assert.match(m.redact("key ghp_ABCDEFGHIJKLMNOPQRSTUV and csk-abcdefghijklmnopqrstu"), /^key <redacted> and <redacted>$/);
assert.match(m.redact("https://discord.com/api/webhooks/1/abc"), /<redacted>/);
assert.equal(m.safeArgs("{bad"), null); assert.deepEqual(m.safeArgs(""), {});
// listener routing: owner only; reply to a forge card continues; "Name, ..." starts; DM goes to the main agent
const O = "42", base = { author: { id: O }, guild_id: "g", content: "hello" };
assert.equal(m.route({ ...base, author: { id: "7" } }, O), null, "a stranger is ignored");
assert.equal(m.route({ ...base, webhook_id: "w" }, O), null, "webhook posts are ignored");
assert.equal(m.route({ ...base, author: { id: O, bot: true } }, O), null);
assert.equal(m.route(base, O), null, "plain server chatter is ignored");
assert.equal(m.route(base, ""), null, "no owner configured: nothing");
assert.deepEqual(m.route({ ...base, guild_id: undefined }, O), { agent: "lead", prompt: "hello" });
assert.deepEqual(m.route({ ...base, content: "Scout, count the queue" }, O), { agent: "free", prompt: "count the queue" });
assert.deepEqual(m.route({ ...base, content: "coder: fix #12" }, O), { agent: "coder", prompt: "fix #12" });
assert.equal(m.route({ ...base, content: "Bob, hi" }, O), null);
const reply = { ...base, content: "and the PRs?", referenced_message: { author: { username: "Scout" }, embeds: [{ footer: { text: "gpt-oss-120b · 9 s · done · chat c0ffee1234567890" } }] } };
assert.deepEqual(m.route(reply, O), { agent: "free", chat: "c0ffee1234567890", prompt: "and the PRs?" });
assert.equal(m.route({ ...reply, referenced_message: { author: { username: "Scheduler" }, embeds: [] } }, O), null);
console.log("test_agent: ok");
