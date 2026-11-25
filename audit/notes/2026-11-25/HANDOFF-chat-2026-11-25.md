# Handoff, chat 2026-11-25 (Claude, near its usage limit)

## Done today
- **Forge as an OS (P36).** OS gap map and hypothesis are in `audit/notes/2026-11-25/forge-os.md`. Blind review 45501864070 was applied. The order: event spine, then issue threads, then console, then hot store, then broker, then editor.
- **Event spine.** P37 (replay) and P37b (every status write is an event, PR #139) are both done. The replay rebuilds 339 of 340 items; the one exception is a claim made before the merge. Render now refuses unlogged claims and births, and baselines are frozen. Files: `audit/spine/`.
- **Issue agent, live** (PRs #146–#149; blind reviews 45508009679, 45509281079 and 45512119627).
  - Use: `/oc [model=<alias|provider/model>] <question>` on any issue, or `/oc models <provider>`.
  - Only the owner can trigger it.
  - The model job is read-only: no shell or web, a system config, and the queue branch checked out as `state/`. A script posts the answer.
  - Keys come from repo secrets, one step per provider. `agent/models.json` holds the aliases (free, pickle, mimo, opus, sonnet, haiku, fable), the paid rules and `allow_paid`.
  - Measured on the free DeepSeek: about 1 min 50 s, $0, every path cited, correct against QUEUE.md.
- **Keys.** OPENCODE_API_KEY is valid (keys-probe 45506695777). The owner removed the provider keys from the OpenCode Console.

## Open, in order
1. **Worker idle.** 0 of 75 open items are claimable, because none has a job type. The splitter last ran on new items on 21 Nov, and 52 of its 72 pieces are parked.
   - Brief and the free agent's first answer: **issue #150**.
   - Next: check that answer against `queue.json`, then make the smallest fix it identifies. The owner can drive this with `/oc` on #150 (`model=opus` for a stronger pass).
2. **Chat interface (owner, 2026-11-25).** The owner wants a chat they control, with a custom connector, that can be extended without restrictions; the issue agent is a first step, not the end.
   - Hypothesis to test: an MCP server over the repo (queue, events, findings, notes, dispatch of workflows) as the custom connector, behind a chat UI that takes any model.
   - Candidates: LibreChat or Open WebUI self-hosted, or a Cloudflare Agents SDK app (free tier).
   - Cheapest first test: a read-only MCP server exposing the `queue.json`/`ready()`/replay data, tried from one client.
   - Keys stay in one store (see P33 broker).
3. P33 key broker; console (P36 step 4); main branch protection before any agent gets write access.

## Where things live
- The queue, findings, notes and spine are on `pilot/lgn-20261115`.
- Reviews are on `router-runs` under `reviews/<run>/`.
- The agent code is `agent/` and `.github/workflows/opencode.yml` on main.
