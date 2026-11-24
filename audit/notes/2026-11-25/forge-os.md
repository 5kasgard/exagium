# The Forge as an operating system: what it has, what it lacks (2026-11-25)

Owner, 2026-11-24/25: use the Forge from one place, at any time, with any API model, not limited by one AI chat;
build setups and automations visually or through a main chat; see all the data it makes and works through; let
agents and the parts talk to each other directly, not only through the work; it must evolve; keep all the keys.
"It's sort of like an OS too... it lacks a lot that makes it an actual system to work with."

This note is a hypothesis to test, not a design to impose. Every row below was checked against `main` on 2026-11-25.

## The diagnosis

The parts talk only by leaving files for each other: the queue (`pilot/lgn-20261115`), the call ledger
(`worker-runs`), review replies (`router-runs`), Actions logs. Nothing is live, nothing is pushed to a part that needs
it, and a new agent must read four branches to know the state. That one gap explains several open complaints:
work piles up, the proposer never sees the meters, router scores do not feed selection, outside agents
see "a benchmark tool".

## OS map (what exists on main, what is missing)

| OS part | The Forge today (checked) | Gap |
|---|---|---|
| Scheduler | `worker.yml` cron hourly; `keeper.yml` sleeps and dispatches because GitHub cron fired 3 times in ~30 h (measured 2026-11-19); `worker/lanes.py` sets concurrency from headroom | Polling only; no wake on an event |
| Process table | Queue item `status` + `worker/transitions.py` state machine; Actions run list | No single live view of what runs now, on which item, with which model |
| IPC (parts talking) | None. Commits to queue/ledger/review branches | **The spine** |
| File system | The repo, split over `main`, `pilot/lgn-20261115`, `worker-runs`, `router-runs` | No index across branches |
| Permissions | `tests/test_workflows.py` privilege guard; MERGE_ALLOW = `experiments/*.py`; keys per step | No key broker (P33): agents that hold keys cannot run outside Actions safely |
| Drivers (models, pools) | `router/engines.py`, `pick.py`, `slots.py`, `keys_probe.py`, `worker/exhaust.py` (429 resets) | Calibration registry (K48) not wired; scores do not flow back |
| System log | `ledger/` + `worker/ledger.py`: model calls only | Not every event (moves, results, meters, messages) |
| Supervisor | `worker/diagnose.py` classifies failed runs; `digest.yml` daily | Reports, does not restart or reroute |
| Shell | None. Each outside chat clones the repo | A main chat with tools over the spine |
| GUI | `QUEUE.md` (render.py), daily digest | Console: live queue, findings, runs, ledger, threads |
| Package registry | None for agents, skills, flows | Flows and agent cards as files in the repo |
| Identity | Commit author strings (`worker-<run>`, `claude-chat`) | One identity per actor (model + harness) |

## Hypothesis: five layers, each part replaceable

1. **Event spine**: append-only events (queue moves, run results, meter readings, review verdicts, messages). The
   repo stays the archive; a mirror of every event is committed, so the spine adds no lock-in.
2. **Threads and channels on the spine**: one thread per queue item plus open channels; each actor has an identity;
   A2A for agent-to-agent, MCP for tools (complementary standards).
3. **Key broker (P33)**: agents ask the broker for a model call; it holds the keys, routes free-first, keeps
   budgets, logs every call. Keys never leave it.
4. **Executors**: GitHub Actions keeps running code (sandbox, guard). The spine dispatches; results come back as events.
5. **Views**: console, a main chat with any model through the broker, a visual flow editor whose flows are files that
   go through PR review.

Evolution: the system may propose changes to its own flows, agents and routing through the queue. The trust core
(guard, broker, merge rules) stays human-merge only.

Candidate host for 1-3: Cloudflare Agents SDK on Durable Objects (free plan: 100,000 requests/day, 5 GB, 30 s CPU per
request by default; over a free limit, operations fail). Long jobs stay in Actions. Not chosen; to be measured.

## Tests, cheapest first (each must pass before the next)

1. **Replay (P37, zero calls)**: rebuild today's queue from an event stream derived from the existing records. Pass:
   the replayed state equals `queue.json` item for item. Fail means the event model does not fit the Forge.
2. **Minimal spine**: one append/read/subscribe service on the free tier; Actions posts events. Measure real
   requests/day against the limit at the current run rate.
3. **Two agents on one thread**: a worker asks a reviewer a question and acts on the answer.
4. **Broker (P33)**, then the views.

## Out of scope for now

n8n (Git sync is a paid tier; Sustainable Use License), Flowise (archived 2026-08-10). Windmill stays a candidate
for the flow editor only if its flows can live as repo files under review.

## Blind review 45501864070 (groq gpt-oss-120b; nvidia nemotron-3-ultra; replies on router-runs reviews/45501864070)

| Layer | gpt-oss | nemotron | Applied |
|---|---|---|---|
| 1 Spine | CHANGE: committing every event to Git churns | CHANGE: D1/KV hot log, daily snapshot to repo | **Changed**: hot log in a store, batched snapshot to the repo, not one commit per event |
| 2 Threads | CUT: use GitHub issues/PR comments as threads | CHANGE: typed request/response first, protocols when interop needs them | **Changed**: threads start as one GitHub issue per queue item (free, durable, phone, opencode can act there); A2A only when an outside agent needs it |
| 3 Broker | CHANGE: off GitHub | CHANGE: one service with 27 keys is the largest blast radius | **Changed**: broker starts with free-tier keys only; paid keys stay in Actions steps. Not applied: "Vault mints short-lived tokens so the broker never sees raw keys" -- provider API keys are not short-lived tokens, so the broker would still hold them |
| 4 Executors | CHANGE: self-hosted runner | KEEP | Kept; self-hosted runner noted for later |
| 5 Views | CUT editor | CUT editor; read-only console and CLI chat first | **Changed**: read-only console and chat first; flow editor deferred |

Missing parts they named, kept as candidates: decision-level trace and replay of one agent step; agent identity with
a track record (K48 calibration registry); disaster recovery (store loss, broker compromise, key rotation); agent
memory beyond queue items; live messaging.

Their falsifying test, adopted for the store step: 100 real queue items through the spine on the free tier; the
approach is wrong if it uses more than 50% of the daily free quota or p99 latency exceeds 30 s.

## Revised order

1. P37 replay: done. Fits the state machine (1,356/1,360) but run logs rebuild only 230/339 items.
2. P37b: every status write emits a structured event (local, free).
3. Threads as GitHub issues per item, with the owner-gated opencode agent.
4. Read-only console over the repo and events.
5. Hot event store on a free tier, measured on 100 real items.
6. Broker with free keys only. 7. Flow editor last.
