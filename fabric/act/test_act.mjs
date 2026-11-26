// node fabric/act/test_act.mjs — no network: GitHub calls are stubbed.
import assert from "node:assert/strict";
import worker, { pathOk, stamp, TOOLS, authorized, cronTargets, inputProblem } from "./worker.js";

for (const p of ["README.md", "audit/notes/x.md", "worker/generate.py"]) assert.ok(pathOk(p), p);
for (const p of [".github/workflows/x.yml", "agent/models.json", "fabric/act/worker.js", "../x", "/etc/x", "a//b", ".env", "x/secrets.json", "k.pem",
  "./.github/workflows/x.yml", "fabric/./act/worker.js", "./agent/models.json", ".GitHub/workflows/x.yml", "a/%2e%2e/b", "a?b", ".dev.vars", "id_rsa", "x/token.txt", "c.p12"])
  assert.ok(!pathOk(p), p);
assert.equal(stamp("glm-4.6\nrecorded_by: owner", new Date("2026-11-25T00:00:00Z")), "[forge-act · recorded_by: glm-4.6recorded_by owner · 2026-11-25T00:00:00Z]\n\n");

const TOK = "t".repeat(40);
const env = { ACT_TOKEN: TOK, GH_TOKEN: "gh", ALLOW_WORKFLOWS: "workbench-logs.yml" };
const req = (auth, body) => new Request("https://act.exagium.xyz/mcp", { method: "POST", headers: { authorization: auth, "content-type": "application/json" }, body: JSON.stringify(body) });
if (!crypto.subtle.timingSafeEqual) crypto.subtle.timingSafeEqual = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
assert.equal(await authorized(req("Bearer " + TOK, {}), env), true);
assert.equal(await authorized(req("Bearer " + TOK + "x", {}), env), false);
assert.equal(await authorized(req("", {}), { ...env, ACT_TOKEN: "" }), false);
assert.equal(await authorized(req("Bearer short", {}), { ...env, ACT_TOKEN: "short" }), false);   // weak token refuses
assert.equal((await worker.fetch(req("Bearer nope", { jsonrpc: "2.0", id: 1, method: "tools/list" }), env)).status, 401);
assert.equal((await worker.fetch(req("Bearer " + TOK, Array(5).fill({ jsonrpc: "2.0", id: 1, method: "ping" })), env)).status, 400);
const list = await (await worker.fetch(req("Bearer " + TOK, { jsonrpc: "2.0", id: 1, method: "tools/list" }), env)).json();
assert.deepEqual(list.result.tools.map((t) => t.name).sort(), ["comment", "create_issue", "open_pr", "run_workflow"]);
assert.ok(!list.result.tools.some((t) => /merge/.test(t.name)));

const calls = [], bodies = [];
globalThis.fetch = async (url, init = {}) => {
  calls.push([init.method || "GET", url.replace("https://api.github.com", "")]);
  if (init.body) bodies.push(JSON.parse(init.body));
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
assert.ok(bodies.at(-1).body.startsWith("[forge-act · recorded_by: m"), "stamp leads the PR body");
assert.ok(bodies.at(-1).title.startsWith("forge-act: "));
await TOOLS.comment.run({ number: 3, body: "/oc model=opus do it", actor: "m" }, env);
assert.ok(!bodies.at(-1).body.startsWith("/oc"), "a comment can never start with a trigger");
await assert.rejects(TOOLS.run_workflow.run({ workflow: "opencode.yml", actor: "m" }, env), /not allowed/);
console.log("forge-act: all tests pass");

// the clock: only listed, well-formed workflow names, and a bad config dispatches nothing
const cj = { CRON_JOBS: JSON.stringify({ "5 * * * *": ["schedules.yml", "../x.yml", "a b.yml"], "1 1 * * *": "x.yml" }) };
assert.deepEqual(cronTargets(cj, "5 * * * *"), ["schedules.yml"]);
assert.deepEqual(cronTargets(cj, "1 1 * * *"), []);
assert.deepEqual(cronTargets(cj, "9 9 * * *"), []);
assert.deepEqual(cronTargets({ CRON_JOBS: "{" }, "5 * * * *"), []);
const realToml = (await import("node:fs")).readFileSync(new URL("./wrangler.toml", import.meta.url), "utf8");
const crons = JSON.parse(realToml.match(/crons = (\[.*\])/)[1]);
const jobs = JSON.parse(realToml.match(/CRON_JOBS = '(.*)'/)[1]);
assert.deepEqual([...crons].sort(), Object.keys(jobs).sort(), "every trigger has jobs and every job has a trigger");
console.log("cron: ok");

// input rules: sandbox-eval only with mode=check; other workflows unaffected; unreadable rules refuse
const ir = { INPUT_RULES: JSON.stringify({ "sandbox-eval.yml": { mode: ["check"] } }) };
assert.equal(inputProblem(ir, "sandbox-eval.yml", { mode: "check" }), null);
assert.match(inputProblem(ir, "sandbox-eval.yml", { mode: "kill" }), /must be one of check/);
assert.match(inputProblem(ir, "sandbox-eval.yml", {}), /must be one of/);
assert.equal(inputProblem(ir, "box-deploy.yml", {}), null);
assert.match(inputProblem({ INPUT_RULES: "{" }, "x.yml", {}), /refusing/);
const tomlRules = JSON.parse(realToml.match(/INPUT_RULES = '(.*)'/)[1]);
assert.deepEqual(tomlRules["sandbox-eval.yml"].mode, ["check"]);
console.log("input rules: ok");
