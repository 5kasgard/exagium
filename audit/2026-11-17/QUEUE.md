# exagium: one queue after the whole-project audit (2026-11-17)

Built from six skill-group audits (A security, B tests/bugs/state, C structure, D claims, E agents/free compute, F ML/binaries), a separate verification pass (S2: 15 of 16 high-impact items confirmed by reproduction, 1 partly right), and the 2026-11-13 audit and handoffs. Severity is after S2 corrections. Source of truth: queue.json; render with render.py.

## Can run now (18)

| id | severity | value | kind | title | next | refs |
|---|---|---|---|---|---|---|
| Q12 | major | high | fix | loop.yml steps end in `|| true`: CI cannot fail (22 of 23 runs green) | Remove `|| true` from tests/reproduce/evaluate; keep artefact upload with if: always() | B F4; C F5; S2 reproduced |
| Q13 | major | high | fix | One bad proposal in ledger/inbox blocks every proposal and skips the 48 h upgrade check | Quarantine invalid proposals to ledger/rejected/ with the reason, ingest the rest; add a test | B F5; S2 reproduced (worse than reported) |
| Q19 | major | high | fix | Bench jobs are green on instrument faults (21 bench3 attempts sent literal $B_MODEL, 0 calls, success) | Fail the job when run.json shows 0 model calls or a provider error; already fixed for that cause (2026-11-14), add the guard | E E-06; S2 confirmed |
| Q07 | minor | medium | fix | Older workflows: excessive permissions, persisted tokens, unpinned actions@vN (known since 11-13 zizmor) | Pin actions by SHA, set least permissions, persist-credentials: false; non-registered workflows only | A-security m1,m2; project doc repo-audit-2026-11-13 (known) |
| Q10 | minor | medium | fix | Jail modes differ: bwrap mode has no seccomp filter and allows nested user namespaces; neither mode limits memory or disk | Add the seccomp filter and --disable-userns to bwrap mode; rlimits for memory/disk in both; test in CI | A-security m5; main:loop/loop.py:212-227 |
| Q14 | minor | medium | fix | Router probe can never pass: exit predicate counts dead or quota-limited models (12 of 12 runs red) | Mark speculative models optional; fail only when an engine that answered last time stops | B F1; E E-01, E-02; C F6; S2 confirmed; severity medium; E E-12 (failure kinds lumped together) |
| Q15 | minor | medium | test | record (22) and probe (40) tests run in no workflow | Add a test job to record.yml (read-only) and a small probe CI job | B; C F5 |
| Q16 | minor | medium | test | Untested boundaries: record overdue < vs <=; loop require_plan ignores verify/check-anchors result | Add tests that kill mutants M5 and M9 | B mutation table |
| Q21 | minor | medium | fix | OpenRouter allowance guard fails open when the allowance cannot be read | Fail closed (skip the batch) when the allowance read fails | E E-10 |
| Q52 | minor | medium | fix | Secret scrub list misses CLOUDFLARE_ACCOUNT_ID, NARA_BASE_URL and CROWLLM_BASE_URL, which could reach public logs | Add the three names to SECRET_ENVS in router code (1 line) and a test | E E-18; S4 finding 3 |
| Q08 | minor | low | fix | review.yml copies a dispatcher-chosen path onto a public branch | Restrict the prompt path to router/review/** | A-security m3; main:.github/workflows/review.yml:33-34 |
| Q11 | minor | low | fix | Committed unused ELF loop/confine (20,712 B) | Delete it (loop.py builds its own copy) | A-security m6; C-structure F10; S2 confirmed |
| Q18 | minor | low | fix | Single unretried pushes to router-runs / ci-runs | Retry with pull --rebase up to 3 times | B F8; E E-08 |
| Q22 | minor | low | infra | No neuron/request log for Cloudflare and Gemini calls; conflicting neuron-cap evidence | Log usage per call like OpenRouter; resolve the 2,876-neuron vs 11-12 conflict from logs | E E-07 |
| Q36 | minor | low | doc | Claims without committed evidence (ctest 0.862, some gate counts and times, 3.8 GHz clock, fresh-clone repeat, model-error fit) | Label each as unrecorded or commit the raw output | D-claims missing-evidence list |
| Q44 | minor | low | doc | mlp infer.c "no overflow is possible" holds only for training-bounded weights; docstrings say "only elementwise ops" but use .sum/np.round | Reword both comments (mlp: frozen dirs, so in the next plan; LGN: now) | F findings 4, 5 |
| Q47 | minor | low | fix | loop.yml:106 unguarded empty-glob line (same cause as the 11-05 failure) | Guard with nullglob or a test | B |
| Q48 | minor | low | fix | bench2.yml keeps the unfixed git-maintenance / chmod race | Apply the f76f2da fix or retire bench2 | B |

## Waiting on a blocker (1)

| id | severity | value | kind | title | next | refs |
|---|---|---|---|---|---|---|
| Q46 | minor | low | test | rfc8785 oracle test skipped everywhere (package never installed) | Install rfc8785 in the test job from Q15 (blocked by Q15) | B |

## Needs the owner (25)

| id | severity | value | kind | title | next | refs |
|---|---|---|---|---|---|---|
| Q20 | major | high | infra | bench3/5 crons still fire on complete benches: the one-line mitigation for Q01-Q03 (any reopened cell on the unprotected router-runs branch would start unattended attempts that reach the vulnerable grade step) | Owner: approve removing the schedule: lines from bench3.yml and bench5.yml on main (keep workflow_dispatch); then regenerate bench5 STATUS (Q35) | E E-05; S2 confirmed; S4 finding 2 |
| Q01 | major | high | fix | bench2/3/5 grade steps run host git in the agent workspace: a planted filter runs on the runner (bench3/5 dormant: empty plans; bench2 dispatch-only, no hardening) | Disable the bench3/5 crons first (Q20). Then, before any new bench run: build the diff without the agent .git, or run git inside the no-network container; bench-3/5 are registered, so amend or re-register; retire bench2 (Q48). Severity: A rated it CRITICAL; lowered to major only because S2 found the plans empty, and that lowering depends on Q20. | A-security C1; S2 (orchestrator reproduction); main:.github/workflows/bench3.yml:159-161, bench5.yml:174-176; A C1 bench2.yml:79,83; S4 finding 2 |
| Q02 | major | high | fix | Grader trusts the first @@RESULTS@@ line from a process that imports agent code: a print forges passes | Have the test runner write results to a file the agent process cannot predict (per-run nonce) or run tests in a separate process from the import; registered benches need an amendment | A-security M1; S2 A-M1 confirmed; main:router/bench/grade.py:14-38 |
| Q03 | major | high | fix | bench3/5 give the real API key to the agent container; seal allows all Cloudflare ranges and DNS anywhere (dormant) | Adopt bench-4 design (host-side meter holds the key) for any future bench | A-security M2; S2 A-M2 confirmed; bench3.yml:119-126; bench5.yml:131-133; router/bench3/run_agent.py:63,66 |
| Q04 | major | high | fix | Ledger writer installs 35 unpinned transitive packages in a job that can sign attestations | Hash-pinned lock (pip --require-hashes) for record.yml; record.yml is the trust root, so the owner approves the change | A-security M3; S2 confirmed (37 resolved, 2 pinned); main:.github/workflows/record.yml:49 |
| Q24 | none | high | decision | "router" has no routing and the free pools have no work queue after bench-4 (the 24/7 goal) | Pick one task kind to route first (taste-and-adjust), with a held-out judge set (Q25) | E E-03; C F7 |
| Q25 | minor | high | test | No held-out task set: model, harness and pool all chosen on the same 3 record.py tasks | Build a held-out task set before any routing claim (blocked by Q24) | E E-04 |
| Q05 | major | medium | fix | Third-party code by mutable reference in key-holding steps (OpenHands SDK tag, agent-server image tag, uv) | Pin by commit/digest before the next bench run | A-security M4 |
| Q06 | minor | medium | doc | Verdict independence is a free-text `by` string (minor while only the owner can push) | Either bind verdicts to the pushing GitHub actor, or state the limit in README:3 and record/README | A-security M5; S2 confirmed; severity lowered |
| Q09 | minor | medium | infra | Stale branches still hold privileged, push-triggered workflows | Decide with Q28: tag and delete the stale branches, or strip their workflows (blocked by Q28) | A-security m4; C-structure F4 |
| Q26 | minor | medium | decision | Copy-by-directory duplication: 5.6% of function lines on main, 29% on the pilot branch (controls ~0%) | Option: parameterised candidates (one train.py + config) vs frozen copies; keep frozen for registered plans | C F1, F16; S2 confirmed |
| Q28 | minor | medium | decision | Stale test branches (anchor-test, q2-main, q2-shadow, e2e-main) and drifting record test copies (record-q2 lacks 5 security steps) | Tag each, then delete; first test that the monitor still sees deleted refs (on a throwaway repo) | C F3, F4; A m4 |
| Q29 | minor | medium | decision | Storage branches hold 94% of unique bytes (63 MB gzipped cachegrind) and are cited as evidence but unprotected and unanchored | Keep summaries + hashes on main, move raw outputs to release assets or prune; protect the branches | C F14; S2 partly right (93.7%) |
| Q31 | minor | medium | decision | Frozen digests fork history: a fix to record.py/count.py forces old-commit re-measurement | Version the frozen tools separately from their fixes (e.g. a tools tag per plan) | C F12 |
| Q35 | minor | medium | doc | bench-5 STATUS.md stale: arm A 11/15, p=0.694, final (page says 9/11, p=1.0) | Regenerate once with decide.py (needs a write to router-runs; with Q20) (blocked by Q20) | D-claims; S2 confirmed |
| Q38 | minor | medium | doc | README "Results can't be lied about" reads as a guarantee (known 11-13) | Owner wording, e.g. "results can't be changed after the fact without detection" | repo-audit-2026-11-13 (known); A M5 |
| Q39 | minor | medium | doc | Agent-facing texts: VERIFY.md numbering restarts 4x, 800-word cap vs ~21 items, says 16 counter tests (23 exist); BRIEF says ruled out and still open for the same H; 10 ASD-STE100 rewrites proposed | BRIEF part done on pilot branch; VERIFY.md (main) numbering, word cap and test count need owner sign-off | E E-13, E-14; E ASD-STE100 table |
| Q42 | minor | medium | fix | linear/thermo infer.c do not validate N: negative N writes out of bounds, huge N crashes (frozen by plan 2) | Fix in the next plan's baselines (harness supplies N, so not exploitable today) | F finding 2 |
| Q49 | none | medium | decision | Free-tier facts: OpenRouter 1,000/day needs >=10 credits bought; Cloudflare 10k neurons/day fits reviews, not agent runs; 7 pools unkeyed; Gemini 2/9 probe successes | Use as input to Q24 | E local-llm-fit section (limits dated 2026-11-17; OpenRouter source is a mirror); E E-11; E E-20 |
| Q17 | minor | low | fix | bench3/4 MODE free text: a typo falls through to a counted run | Validate MODE as bench5 does (registered benches: amendment) | B F6 |
| Q23 | minor | low | fix | Stalled bench4 attempts run to the 35-min cap with <=25 calls | Add an idle timeout (registered bench: amendment) | E E-09 |
| Q30 | minor | low | decision | probe/ is an unconsumed leaf (nothing reads profile.json; no workflow runs it) | Wire it into the per-image meter work or park it | C F8 |
| Q37 | minor | low | doc | Scaffold Effect (arXiv 2607.22585) is a workshop preprint, called published; its direction differs from bench-4 | Erratum file next to the registered bench-4 plan | D-claims citations |
| Q40 | minor | low | doc | README leads with word history, omits router/ and benches from the status table | Owner taste; propose a status-first README | E E-16; C F7 |
| Q43 | minor | low | fix | linear accepts pixels >16; linear/thermo/lgn accept trailing bytes in weights.bin | Same as Q42 (blocked by Q42) | F finding 3 |

## Parked (4)

| id | severity | value | kind | title | next | refs |
|---|---|---|---|---|---|---|
| Q27 | none | low | decision | Three decide.py copies (deliberate: frozen per registration; known 11-13) | Revisit only with a shared, versioned rules module | C F2; repo-audit-2026-11-13 (known) |
| Q32 | none | low | decision | canonical() and sha256_file triplicated (known 11-13; byte-equal today) | Cross-module equality test, or one module | C F11; repo-audit-2026-11-13 (known) |
| Q50 | none | low | decision | Ledger cost-to-use: ~650 code lines + 3 workflows for 4 entries; pays off once benches are registered in it | Register the next bench plan in the ledger | C F15 |
| Q53 | minor | low | decision | Small seams: counter has 5 of 13 flags for loop's jail; name collisions (two probe.py, review.py vs review/) | Revisit when the per-image meter work touches counter/loop | C F9, F13 |

## Done (4)

| id | severity | value | kind | title | next | refs |
|---|---|---|---|---|---|---|
| Q33 | minor | medium | doc | loop/README numbers: "17-31% faster" is 14.5-33.1%; "3.5-5.4x" mixes startup; two-meters host range 1.7-2.8x on CI; "dependency chains" hypothesis now stale | Done on pilot branch: loop/README.md corrected (14.5-33.1%; 3.7-5.9x like for like; mlp16 1.8-2.5x per instruction recomputed; stale hypothesis marked). | D-claims; S2 confirmed |
| Q34 | minor | medium | doc | My own pilot/explore claims: H2 "ruled out" (paired +3.8 ns, 18/21), LILogicNet thresholds wording (13 of 14 pairs), BRIEF linear mispredicts 0.002 (2.08), H5/H7 attributed to two runs (only GLM), unlabelled host change, page-fault rounding | Done on pilot branch: BRIEF status, reports/claude-2026-11-15.md, lgn_c/lgn_d docstrings, pilot 2 README, each correction saying what the first version got wrong. Staging report fault counts also corrected to the evidence JSON (S4 finding 1). | D-claims; S2 confirmed |
| Q41 | minor | medium | doc | Pilot LGN weights were trained on the first 2,867 rows; train.py docstrings and dirs do not say so (reproduce would fail on them) | Done on pilot branch: README.md in lgn_a/b/c/d stating the 75% training rows and how to reproduce. | F finding 1 |
| Q51 | minor | medium | doc | My own headline lines without caveats: root README "2.4x" (no host/compiler/I-O caveat) and lgn_d 0.45x/0.63x (no one-session, I/O-sensitive note) | Done on pilot branch: both README lines now carry the caveats (S4 finding 1; D M9, M12) | D-claims M9, M12; S4 finding 1 |

