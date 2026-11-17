# Whole-project audit, 2026-11-17

**Scope:** all 8 branches, 12 workflows, 139 Actions runs (2026-11-05..17), on the owner's request to use every skill on the whole project.

**Owner's choice:** triage all 51 synced skills and run the ones that fit.

**Method:** the agent-orchestration skill.
1. Six read-only agents ran in parallel, each loading its own skills. Their reports are in `reports/`.
2. A separate agent, which wrote none of the findings, re-checked the 16 highest-impact ones by reproduction (`reports/S2-verification.md`).
3. Everything was merged into one queue (one-queue skill): `queue.json` is the source of truth, and `python3 render.py queue.json > QUEUE.md` checks it and renders it.
4. A fresh agent reviewed the final result (the S4 section below).

**State files:** `state/plan.md`, `state/findings.md` and `state/progress.md`. `plan.md` holds the skill triage.

## Skill triage (all 51)
- **Run (28):**
  - **A:** security-code-review, untrusted-content-guard, careful-live-changes.
  - **B:** test-design, bug-lifecycle, reproduce-before-filing, implicit-state-machines.
  - **C:** code-design, earned-parts-audit, moved-not-removed, directional-reasoning.
  - **D:** adversarial-review, weigh-the-evidence, paper-research, tiered-source-research, honest-views.
  - **E:** agent-architecture, agent-instructions, route-and-test, budgeted-free-runs, local-llm-fit, asd-ste100, clear-explaining.
  - **F:** model-training-playbook, reverse-engineering.
  - **Orchestrator:** agent-orchestration, one-queue, session-handoff, taste-and-adjust, thinking-tools.
- **Skipped, each with a reason in `state/plan.md`:**
  - tokenizers-nlp and model-internals: `git grep` finds no such code.
  - plain-core-themes: there is no UI to design.
  - docs, docx, pdf, pptx, xlsx, google-workspace and doc-coauthoring: output formats.
  - built-in-browser, chrome-browser and computer-use: the API and git covered everything.
  - web-artifacts-builder, video-making, morning and import-memory: unrelated.
  - learn: empty.
  - skill-creator ×2: deferred, used only if a recurring procedure turns up.

## What it revealed (verified unless marked)
1. **Two bench flaws let an agent's code escape the sandbox or forge grades.** Both are real but dormant now, because bench-3 and bench-5 have empty plans. That dormancy is fragile: their crons still fire, and the plan is built from an unprotected branch. Disabling those crons (Q20) is the one-line mitigation, and it is first on the owner's list.
   - **Escape:** the grade step runs host `git` in the agent's workspace, so a planted git filter runs on the runner. Reproduced twice.
   - **Forgery:** the grader trusts the first `@@RESULTS@@` line printed by a process that imports agent code. Reproduced.
   - They must be fixed before any new bench run. bench-4's design, where only a host-side meter holds the key, is the model.
2. **CI can't say "failed" where it matters.**
   - `loop.yml` ends its steps in `|| true`.
   - The record and probe tests run in no workflow.
   - The router probe has failed every run so far (12 of 12): dead or quota-limited models are listed as required. That is empirical, not a logical certainty.
   - bench jobs stayed green while sending the literal text `$B_MODEL` for 21 attempts.
3. **One bad proposal in `ledger/inbox` blocks the ledger:** every later proposal and the 48-hour proof check. Reproduced.
4. **Supply chain:** the ledger writer, the trust root, resolves 35 unpinned packages while able to sign.
5. **No routing yet.** `router/` has no routing, and the free pools have no work queue after bench-4. This is the main gap against the 24/7 goal, and a decision for the owner.
6. **Structure:**
   - Duplication by directory copies: 5.6% of function lines on main, 29% on the pilot branch, against about 0% in control repos.
   - Storage branches hold 94% of the repo's unique bytes.
   - Four stale test branches still carry privileged workflows.
7. **Claims:** no headline verdict changes. 84 claims were checked: 57 match, 5 rounding ok, 5 mismatches, 2 config mismatches, 6 overclaims and 9 without committed evidence.
   - Several were mine, in the pilot and explore work:
     - H2 called "ruled out";
     - H5 and H7 credited to two runs (GLM only);
     - the LILogicNet thresholds wording;
     - BRIEF's mispredict figure;
     - the staging report's fault counts;
     - an unlabelled host change;
     - the root README's "2.4×" and lgn_d 0.45×/0.63× lines without caveats.
   - All are corrected where they were made, each saying what the first version said (Q34, Q51).
8. **ML code:** no test-data leakage; every retrain that was run (6 of 9 candidates; not lgn_a, lgn_b or lgn_c) is byte-identical; all binaries match an independent reimplementation.
   - The pilot weights were trained on the first 75% of rows; this is now noted in each directory.
   - linear and thermo `infer.c` don't validate N. They are frozen by plan 2, so the fix belongs in the next plan.

**Already known before this audit** (the 2026-11-13 audit and handoffs, so not new): the older workflows' zizmor findings, the README's "can't be lied about" wording, the triplicated helpers, and the OpenTimestamps failure from Nov 6 to Nov 10 (fixed in bdd7ec5).

## Done in this audit (pilot branch)
Q33, Q34, Q41 and Q51 were documentation fixes, recomputed or checked before writing, and rechecked by S4. Code and workflow fixes on `main` wait for the owner, per careful-live-changes, since several touch registered benches or the ledger's trust root.

## Limits
- **One verification pass,** on 16 items. The other findings are each auditor's own check: several agents could not start their own reviewers, and their reports say which.
- **Same model family.** The reviewers were Claude models (Opus and Sonnet). By the adversarial-review skill, that makes their verdicts direction, not independent evidence; GLM or Gemini could give the other-family check.
- **Not checked:** the bubblewrap jail running live, CodeQL, repo settings (branch protection) and the full job logs; only failed step names were available.

## S4: final review (fresh Opus agent, same family as the author)
**Verdict: ready with fixes.** The reviewer recomputed every edited number from the raw files and found none wrong. Full text: `reports/S4-final-review.md`. Its 7 findings and what was done:
1. **(Major)** Q34 was marked done while the staging report still showed unsaved fault counts, and two of my own overclaims were unqueued. Fixed: the counts now come from the JSON, Q51 was added and done, and item 7 above lists every one of my corrections.
2. **(Major)** "Dormant" is fragile. Fixed in the queue: Q20 (disable the bench-3/5 crons) is now major/high with priority 1, and Q01 says the CRITICAL rating was lowered only on the dormancy finding, which depends on Q20.
3. **(Minor)** Coverage. bench-2 was added to Q01. E-18 became Q52, E-12 was added to Q14's refs, C F9/F13 became Q53, and E-11/E-20 were added to Q49's refs.
4. **(Minor)** "Every retrain byte-identical" overstated. Fixed (6 of 9). The pilot weight notes now say the 75% recipe was verified for lgn_d only, and point to the right pilot README.
5. **(Minor)** Wording. Fixed: the router probe line is now empirical, and the claim counts are complete.
6. **(Minor)** Edits. BRIEF's H9 line now says linear has more mispredicts; the 2026-11-15 report notes H2's removal; the loop README hedge is gone (the numbers were rechecked); the "points" placement is fixed.
7. **(Minor)** This section was missing. Added.

No second S4 round was run. The fixes were checked by the orchestrator only: the edits were re-read and the queue re-validated.
