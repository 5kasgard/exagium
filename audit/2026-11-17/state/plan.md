# Plan: whole-project audit with every fitting skill
Goal: apply every skill from the session's synced skill list (51) that fits a code/research repo to the whole exagium project: all 9 branches, 12 workflows, Actions history. Report what it reveals, verified, as one ranked queue.
Owner decision (2026-11-17): "Triage all 51, run the fit."
Evidence (frozen, read-only): /home/claude/audit/repo (full clone, all branches as local branches), /home/claude/audit/evidence/actions_runs.json (139 runs, 2026-11-05..17), actions_failed_jobs.json (24 non-success runs, failed steps).
Queue: findings land in /home/claude/audit/out/<agent>.md; the orchestrator merges them into queue.md (one-queue).

## Triage of all 51 skills
| skill | verdict | where |
|---|---|---|
| security-code-review, untrusted-content-guard, careful-live-changes | run | A |
| test-design, bug-lifecycle, reproduce-before-filing, implicit-state-machines | run | B |
| code-design, earned-parts-audit, moved-not-removed, directional-reasoning | run | C |
| adversarial-review (D1 claim audit), weigh-the-evidence, paper-research, tiered-source-research, honest-views | run | D |
| agent-architecture, agent-instructions, route-and-test, budgeted-free-runs, local-llm-fit, asd-ste100, clear-explaining | run | E |
| model-training-playbook, reverse-engineering | run | F |
| agent-orchestration, session-handoff, one-queue, taste-and-adjust, thinking-tools, adversarial-review (final) | run | orchestrator |
| tokenizers-nlp, model-internals | skip: `git grep` finds no tokenizer, transformer or interpretability code on main or the pilot branch | - |
| plain-core-themes | skip: for early-stage demo/theme design; the project has no UI to design | - |
| doc-coauthoring, docs, docx, pdf, pptx, xlsx, google-workspace | skip: output formats; the audit is delivered as markdown in the repo and the project | - |
| built-in-browser, chrome-browser, computer-use | skip: the repo is public and the Actions history came through the GitHub API; no browser or desktop step needed | - |
| web-artifacts-builder, video-making, morning, import-memory | skip: unrelated to auditing a repo | - |
| skill-creator, skill-creator-copy | deferred: propose a skill only if the audit finds a recurring procedure | orchestrator, end |
| learn | skip: empty description, nothing to apply | - |

## Steps
### S1 dispatch A-F (wave 1, parallel; read-only, independent: separate output files, no shared writable state)   Status: done (reports A-F; F interrupted by a server restart, auto-resumed)
### S2 verify findings (fresh agent, not the authors; exact checks first)   Status: done (15 of 16 confirmed, 1 partly; C1 also reproduced by the orchestrator)
### S3 merge into one queue (one-queue), ranked   Status: done (52 items; checker tested with 6 injected faults)
### S4 final adversarial review of the queue + report (strong tier)   Status: done (ready with fixes; 2 major + 5 minor, all applied; first attempt cut off by the 5-hour usage limit, rerun 2026-11-18)
### S5 commit report to the pilot branch; project handoff (session-handoff)   Status: done (see progress.md)

## Decisions
- Tiers (hypothesis): Opus for A (security) and D (claims): highest cost if wrong. Sonnet for B, C, E, F: bounded, checkable work. Final review: Opus.
- Agents are read-only on the repo; scratch in their own dirs; no pushes, no live API calls with secrets, no workflow triggers.
