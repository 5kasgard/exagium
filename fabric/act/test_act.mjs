// node fabric/act/test_act.mjs — no network: GitHub calls are stubbed.
import assert from "node:assert/strict";
import worker, { pathOk, stamp, TOOLS, authorized } from "./worker.js";

for (const p of ["README.md", "audit/notes/x.md", "worker/generate.py"]) assert.ok(pathOk(p), p);
for (const p of [".github/workflows/x.yml", "agent/models.json", "fabric/act/worker.js", "../x", "/etc/x", "a//b", ".env", "x/secrets.json", "k.pem"])
  assert.ok(!pathOk(p), p);
assert.match(stamp("glm-4.6", new Date("2026-11-25T00:00:00Z")), /recorded_by: glm-4.6 via forge-act .* 2026-11-25T00:00:00Z/);

const env = { ACT_TOKEN: "t0ken", GH_TOKEN: "gh", ALLOW_WORKFLOWS: "workbench-logs.yml" };
const req = (auth, body) => new Request("https://act.exagium.xyz/mcp", { method: "POST", headers: { authorization: auth, "content-type": "application/json" }, body: JSON.stringify(body) });
if (!crypto.subtle.timingSafeEqual) crypto.subtle.timingSafeEqual = (a, b) => a.every((x, i) => x === b[i]);
assert.equal(await authorized(req("Bearer t0ken", {}), env), true);
assert.equal(await authorized(req("Bearer t0kex", {}), env), false);
assert.equal(await authorized(req("", {}), { ...env, ACT_TOKEN: "" }), false);
assert.equal((await worker.fetch(req("Bearer nope", { jsonrpc: "2.0", id: 1, method: "tools/list" }), env)).status, 401);
const list = await (await worker.fetch(req("Bearer t0ken", { jsonrpc: "2.0", id: 1, method: "tools/list" }), env)).json();
assert.deepEqual(list.result.tools.map((t) => t.name).sort(), ["comment", "create_issue", "open_pr", "run_workflow"]);
assert.ok(!list.result.tools.some((t) => /merge/.test(t.name)));

const calls = [];
globalThis.fetch = async (url, init = {}) => {
  calls.push([init.method || "GET", url.replace("https://api.github.com", "")]);
  if (url.endsWith("/git/ref/heads/main")) return Response.json({ object: { sha: "abc" } });
  if (url.includes("/contents/") && (init.method || "GET") === "GET") return new Response("{}", { status: 404 });
  if (url.endsWith("/pulls")) return Response.json({ number: 9, html_url: "u" });
  return Response.json({});
};
await assert.rejects(TOOLS.open_pr.run({ name: "x", title: "t", body: "b", actor: "m", files: [{ path: ".github/workflows/a.yml", content: "x" }] }, env), /refused path/);
await assert.rejects(TOOLS.open_pr.run({ name: "bad name", title: "t", body: "b", actor: "m", files: [{ path: "a.md", content: "x" }] }, env), /name/);
await assert.rejects(TOOLS.open_pr.run({ name: "x", title: "t", body: "b", files: [{ path: "a.md", content: "x" }] }, env), /missing actor/);
const pr = await TOOLS.open_pr.run({ name: "x", title: "t", body: "b", actor: "m", files: [{ path: "a.md", content: "hé" }] }, env);
assert.equal(pr.branch, "workbench/x");
assert.deepEqual(calls.map((c) => c[0] + " " + c[1].split("?")[0]),
  ["GET /repos/5kasgard/exagium/git/ref/heads/main", "POST /repos/5kasgard/exagium/git/refs",
   "GET /repos/5kasgard/exagium/contents/a.md", "PUT /repos/5kasgard/exagium/contents/a.md", "POST /repos/5kasgard/exagium/pulls"]);
await assert.rejects(TOOLS.run_workflow.run({ workflow: "opencode.yml", actor: "m" }, env), /not allowed/);
console.log("forge-act: all tests pass");
