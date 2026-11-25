// node fabric/box/test_box.mjs — no network: E2B is a stub.
import assert from "node:assert/strict";

import worker, { handle, cleanName, authorized } from "./worker.js";
if (!crypto.subtle.timingSafeEqual) crypto.subtle.timingSafeEqual = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);   // Workers has it; Node does not

assert.equal(cleanName("Forge Main!"), "forgemain");
const TOK = "b".repeat(40), env = { BOX_TOKEN: TOK, E2B_API_KEY: "k", MAX_BOXES: "2" };
const req = (auth, body) => new Request("https://box.exagium.xyz/mcp", { method: "POST", headers: { authorization: auth }, body: JSON.stringify(body) });
assert.equal(await authorized(req("Bearer " + TOK, {}), env), true);
assert.equal(await authorized(req("Bearer x", {}), env), false);
assert.equal((await worker.fetch(req("Bearer nope", { jsonrpc: "2.0", id: 1, method: "tools/list" }), env)).status, 401);

const boxes = [], ran = [];
const mkSb = (id) => ({ sandboxId: id, commands: { run: async (c, o) => { ran.push([c, o]); return { exitCode: 0, stdout: "Linux 6.1\n2", stderr: "" }; } },
  files: { write: async () => {}, read: async () => "hello" } });
const S = {
  list: (o) => ({ nextItems: async () => { assert.equal(o.query.metadata.forge, "box"); return boxes.filter((b) => !("forge_box" in o.query.metadata) || b.metadata.forge_box === o.query.metadata.forge_box); } }),
  connect: async (id) => mkSb(id),
  create: async (o) => { const b = { sandboxId: "sb" + boxes.length, metadata: o.metadata, state: "running" }; boxes.push(b); assert.equal(o.lifecycle.onTimeout, "pause"); assert.ok(!o.lifecycle.autoResume, "no autoResume"); return mkSb(b.sandboxId); },
  pause: async () => true, kill: async (id) => { boxes.splice(boxes.findIndex((b) => b.sandboxId === id), 1); return true; },
};
const call = async (name, args) => JSON.parse((await handle({ jsonrpc: "2.0", id: 1, method: "tools/call", params: { name, arguments: args } }, env, S)).result.content[0].text.replace(/^error: (.*)$/s, '"$1"'));

assert.equal((await call("box_open", { name: "main", actor: "m" })).created, true);
assert.equal((await call("box_open", { name: "main", actor: "m" })).created, false, "same name reconnects");
await call("box_open", { name: "two", actor: "m" });
assert.match(await call("box_open", { name: "three", actor: "m" }), /box limit/);
assert.match(await call("box_exec", { name: "nope", command: "ls", actor: "m" }), /no box named/);
assert.match(await call("box_kill", { name: "!!", actor: "m" }), /name: letters/, "empty name never reaches the filter");
const r = await call("box_exec", { name: "main", command: "ls", actor: "gpt\nx", timeout_s: 9999 });
assert.equal(r.exit, 0);
assert.equal(ran.at(-1)[1].timeoutMs, 600000, "timeout capped");
assert.equal(ran.at(-2)[1].envs.ACTOR, "gptx", "actor cleaned, passed by env not shell text");
assert.match(ran.at(-1)[1].envs.F, /^\/home\/user\/\.forge-box\/[0-9a-f-]{36}\.cmd$/, "one command file per call");
const miss = await handle({ jsonrpc: "2.0", id: 1, method: "tools/call", params: { name: "box_list", arguments: {} } }, env, S);
assert.match(miss.result.content[0].text, /missing actor/);
assert.deepEqual((await call("box_kill", { name: "two", actor: "m" })), { killed: "two" });
console.log("forge-box: all tests pass");
