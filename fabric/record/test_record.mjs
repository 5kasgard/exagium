// node fabric/record/test_record.mjs : naming and helpers (D1 itself is exercised live)
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
const MAKERS = JSON.parse(readFileSync(new URL("./makers.json", import.meta.url)));
const src = readFileSync(new URL("./record.js", import.meta.url), "utf8").replace('import MAKERS from "./makers.json" with { type: "json" };', `const MAKERS = ${JSON.stringify(MAKERS)};`);
const m = await import("data:text/javascript," + encodeURIComponent(src));
assert.equal(m.pickName(["A"], ["A", "B"]), "B");
assert.equal(m.pickName(["A", "B"], ["A", "B"], () => 0), "A 2");
assert.equal(m.pickName(["A", "B", "A 2"], ["A", "B"], () => 0), "A 3");
assert.equal(m.slug("Fabergé"), "faberge");
assert.equal(m.clip("a\n  b", 10), "a b");
assert.equal(new Set(MAKERS.names).size, MAKERS.names.length, "no duplicate names");
// a fake D1 that records statements: register writes one row with a name from the pool
const rows = [];
const db = { prepare: (q) => ({ bind: (...b) => ({ run: async () => rows.push([q, b]), all: async () => ({ results: [] }) }), all: async () => ({ results: [] }) }) };
const r = await m.register(db, { class: "coder", model: "big-pickle", interface: "discord", verified: true });
assert.ok(MAKERS.names.includes(r.name)); assert.match(r.id, /^[a-z0-9-]+-[0-9a-f]{4}$/);
const ins = rows.find(([q]) => q.includes("INSERT INTO agents"));
assert.equal(ins[1][2], "coder"); assert.equal(ins[1][9], 1);
assert.ok(rows.some(([q]) => q.includes("SET retired_at")), "idle instances are retired before a new name is picked");
assert.equal(m.redact("token=abcdefghijklmnopqrstu"), "token=<redacted>");
assert.equal(await m.register(null, {}), null, "no database: nothing, no throw");
console.log("test_record: ok");
