// Node test for forge-cfbox's gate (no Cloudflare runtime needed): node fabric/cfbox/test_cfbox.mjs
import assert from "node:assert/strict";
import worker, { authorized } from "./worker.js";

const T = "x".repeat(40);
const req = (path, tok, method = "POST") =>
  new Request("https://cfbox.exagium.xyz" + path, { method, headers: tok ? { authorization: "Bearer " + tok } : {} });
let called = 0;
const env = { BOX_TOKEN: T, BOX: { idFromName: (n) => n, get: () => ({ fetch: async () => { called++; return new Response("{}"); } }) } };

assert.equal(authorized(req("/create", T), env), true);
assert.equal(authorized(req("/create", T + "y"), env), false);
assert.equal(authorized(req("/create", null), env), false);
assert.equal(authorized(req("/create", "short"), { BOX_TOKEN: "short" }), false, "tokens under 32 chars never pass");
assert.equal(authorized(req("/create", ""), { BOX_TOKEN: "" }), false, "unset secret refuses everything");
assert.equal((await worker.fetch(req("/create?mode=snap", null), env)).status, 401);
assert.equal((await worker.fetch(req("/create?mode=bogus", T), env)).status, 400);
assert.equal((await worker.fetch(req("/exec?mode=snap", T), env)).status, 400);
assert.equal((await worker.fetch(req("/check?mode=snap", T, "GET"), env)).status, 400);
assert.equal(called, 0);
assert.equal((await worker.fetch(req("/check?mode=native", T), env)).status, 200);
assert.equal(called, 1);
console.log("test_cfbox: ok");
