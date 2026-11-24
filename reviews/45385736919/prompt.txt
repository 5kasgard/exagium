You are reviewing four proposed new work items for a public research repository. In it, free AI models work a queue on CI, and every result is checked by scripts and recorded. For each item answer KEEP, DEFER, MERGE (with which) or CUT, give one sentence of reason, and name the single cheapest test that would show the item is worth doing. Judge only the evidence given. Reply as a JSON list of {"item", "verdict", "reason", "cheapest_test"}.

Context, measured:
- Jobs that hold API keys run 51 of the repo's 129 Python files.
- Model-written code may auto-merge only into experiments/*.py.
- On 2026-11-21, four auto-merges reached files that keyed jobs run. That is why the rule is narrow now.
- Of the 78 files no keyed job runs, nearly all are tests, graders, scoring or verdict code, which model code must not edit.
- Model calls per run (ledger, 3,459 calls): worker median 2 (p90 6), proposer 3, tier 6.

Items:
A. Key broker. Keyed steps run only a small frozen set of files (the provider call layer and the ledger), checked out from a pinned main commit, in their own job. All prompt building and reply parsing runs in jobs with no keys, and requests and replies pass as hashed artifacts. Expected cost: 2-3 extra job hops per run. Expected gain: auto-merge can widen to worker/ and router/ logic, so a person is needed only for keys, workflows and gates.
B. A Lean 4 task domain. Proof statements (miniF2F-style) where the Lean kernel is the deterministic checker, so it fits the existing task contract (records in and out, a checker per task). Also Lean specifications for a few invariants of the trust core.
C. A connection graph of the whole system: queue items, findings, files, runs and research cards as nodes, with edges marked as either extracted or inferred. Test an existing open-source code-graph tool (Graphify) against the repo's own queue-only graph v0, and keep whichever answers a fixed set of 10 dependency questions correctly at lower cost.
D. An interface for working on the system that does not depend on any single AI chat product. Compare LibreChat, Open WebUI, OpenHands and Goose. Criteria: use the existing free provider keys, host MCP tools, run on free hosting or start and stop on demand, and be able to read and drive the queue.
