# Handoff, chat 2026-11-25 (b) — Claude (Opus 5.5), item P38

## Done
- **Measured why the worker was idle** (queue.json + GitHub PR records, 08:40Z): ready() was 0 of 76 open. The worker is
  fine; its inputs are not. 49 of 73 splitter code items reached a PR with tests passing, but only 9 are done: 28 were
  rejected on merits after tests passed (the coder's own tests pass trivially), 9 waited on a person (auto-merge takes
  only `experiments/*.py`). Detail in P38's next step.
- **Queue reconciled with GitHub:** K25 (#40), K53 (#80), K59 (#118) and K50 (#66) were merged but still "waiting on the
  owner" -> done. K20 (#38 closed unmerged) -> parked. All through `queue_ops move` (events logged).
- **Four open PRs reviewed** (merged onto main locally: 232 CI tests pass; blind review by a separate agent, both
  rejections reproduced by me): **#129 (K60) and #132 (K62): merge** — waiting for the owner, this session is not
  allowed to merge to main (#132 needs a trivial test-file conflict fix after #129). **#145 (K72), #151 (K68): reject** —
  comments on the PRs; items reopened with the repro/fix in their next step. The PRs can be closed.
- **Q68 given 10 exact doc targets** with values counted on main today (a free doc job can now take it). A general
  findings->targets converter was not built: its reach is 1 item.
- ready(): **0 -> 3** (Q68, K68, K72).

## Open, in order
1. **Owner:** merge #129 then #132; close #145 and #151.
2. **P38 (a):** a PR-sync step — PR merged -> `decide_done`, closed -> `park` — so items stop rotting as "waiting on the
   owner". Touches a workflow, so a person merges it.
3. **P38 (b):** splitter sub-items carry a known-answer test written before the code (the K72 repro form), so "tests
   pass" means "item done". Measure: merits-rejections after passing tests, before/after.
4. P33 (widen auto-merge safely), P36 (connector, PR #153), P39 (project-doc export).

## Running / pending
Nothing started by this session. Another session was working on the LobeHub workbench (`fabric/workbench-*`) at the
same time; not touched. Spend: no API calls; Claude chat usage only.
