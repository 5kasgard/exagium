# Audit C: code structure, duplication across branches, what is truly shared

Date 2026-11-17. Read-only: no commits, pushes or branch changes. Worktrees (since removed) and tools are under `/home/claude/audit/scratch/C/`.
Skills applied: code-design, earned-parts-audit, moved-not-removed, directional-reasoning.

## 0. Method, and what was NOT done

- Everything below marked **verified** was re-run or recomputed by me in this session. **suspected** means read from code or docs and not confirmed by running it.
- **Independence limit (important).** I had no agent-spawning tool. The earned-parts-audit asks for a different agent to verify each distillation (stage 2) and for two blind groupers (stage 5). I could not do either. Every check here is a **self-pass by one agent, not independent**. As a partial substitute I used mechanical checks that do not depend on my judgement: an AST clone detector and two control repos (section 5). Stage 2 and the blind grouping should be run by the parent session. The claim list in section 3 gives each check a command.
- Not run: `record.py check-anchors` and `monitor` (need network to Rekor, blockstream.info and mempool.space), a full `probe.py` run, loop under bubblewrap (not installed here; the tests ran in the `confine` fallback jail). README claims of "7 independent reviews", "4 review rounds", "3 adversarial rounds" are **unverified** (doc only).
- Directions used (directional-reasoning): (A) bottom-up from git topology and CI history; (B) backward from each module's README claim to a test or run; (C) sideways: a mechanical clone scan with controls. All three were self-passes by me. Disagreements found: my first hypothesis "giant files are the problem" was refuted (largest file 884 lines); my first hypothesis "deleting the side branches weakens the monitor" is only half-resolved (F4).

## 1. Branch table

Divergence is `git rev-list --left-right --count main...<branch>` against `main` at e093a60 (2026-11-14, 107 commits).

| Branch | Purpose | Behind / ahead of main | Last commit | Written by workflow | Unique content | Still used? | If deleted | Verdict |
|---|---|---|---|---|---|---|---|---|
| **main** | the project | 0 / 0 | 2026-11-14 | owner pushes; `record` daily bot | 214 files, 3.6 MB | yes | n/a | live |
| **pilot/lgn-20261115** | unmerged experiments: `pilots/lgn-20261115`, `-2`, `explore/mlp16-speed`, 2 README rows | 0 / 7 | 2026-11-17 21:11 (today) | none (human/agent commits by `exagium-agent`) | 162 files, +23,545 lines; workflows identical to main | yes, actively worked | loses all pilot and explore work (no copy elsewhere) | **live; keep**. Needs a merge-or-archive decision (F16) |
| **ci-runs** | result store for `loop`, `counter`, `record` (orphan branch, 57 commits) | 107 / 57, no merge base | 2026-11-11 19:41 | counter.yml, loop.yml, record.yml (orphans/failure logs), record-e2e.yml, anchor-test.yml | 5,189 files, 77.9 MB. `runs/` 2,856 files 66.5 MB (cachegrind raw), `loop/` 2,188 files, `anchors/` 134, `e2e/` 7, `q1/` 2, `q2-fail/` 2 | collectors last ran 11-11 (path-triggered; no change since) | loses cited evidence: record/README cites `ci-runs` `q1/*.txt`, `e2e/*.txt`; those 11 files total 9.5 KB | **keep the branch; shrink it** (F14). Cited evidence is not anchored and the branch is unprotected |
| **router-runs** | result store for `router` probe, bench1-5, reviews (59 commits) | 107 / 59, no merge base | 2026-11-16 23:47 | bench*.yml, review.yml, router.yml | 1,366 files, 53.6 MB; bench4 17.9 MB, bench2 11.4 MB | yes (bench3 scheduled 2x/day, last run 11-17; bench4/5 active) | loses all bench raw data; bench PREREG and results files cite it | **live; keep** |
| **q2-main** | experiment Q2 (monitor, gh-verify) stand-in "main" | 69 / 9 | 2026-11-05 19:20 UTC | `record-q2.yml` (branch-only file) | 6 files: `ledger/entries` 1-2 and their Sigstore/OTS anchors; 9 commits | no: 5 runs on 11-05, none since | loses a test ledger and its 2 real Rekor anchors. Results already copied to `ci-runs/q1`, `q2-fail` and `verifications/z-20261106/artifacts/monitor-*.json` | **stale**; archive by tag, then delete is the owner's call |
| **q2-shadow** | the "hidden run on a shadow branch" for the monitor's attack test | 70 / 8 | 2026-11-06 05:17 +1000 | `record-q2.yml` | 4 files (entry 1, its anchors, `ledger/inbox/0002-plan.json`); 8 commits | no | loses the hidden-run fixture the monitor test attacks. Z's artifact `monitor-laundering-attack.json` is on main | **stale**; same as q2-main |
| **e2e-main** | end-to-end race tests A-F of the record writer | 64 / 16 | 2026-11-05 19:54 UTC | `record-e2e.yml` (also on main) | 18 files: ledger entries 1-8 and 5 anchors; 16 commits | no: 8 runs on 11-05 only | loses the live, re-checkable ledger the README's e2e claims rest on (`check-anchors --ref refs/heads/e2e-main`). Text results are on `ci-runs/e2e` | **stale but evidence-bearing**; tag before any delete |
| **anchor-test** | anchor-service study (Sigstore, OTS, RFC 3161, gist) | 63 / **0** | 2026-11-06 05:55 +1000 | `anchor-test.yml` (trigger is push to this branch) | **none**: its tip `ba015da` is an ancestor of main | no: 4 runs on 11-05 | **nothing is lost from git history** (all commits are reachable from main). Loses only the push trigger of `anchor-test.yml` | **stale; lowest-risk delete candidate**. `anchor-test.yml` on main is then dead |

Facts behind the table (all verified): workflow run counts and dates come from `gh api .../actions/workflows/*/runs`. GitHub lists **13** workflows as active, but main holds 12 files, because `record-q2.yml` exists only on the q2 branches. No tags exist. `main` and `ci-runs` report `protected: false`.

### Record workflow copies: drift (verified by diff)

| Copy | Where | Relation to `record.yml` on main | Drift |
|---|---|---|---|
| `record-e2e.yml` (135 lines) | main, e2e-main, anchor-test | `append` job is identical after replacing names (`record-e2e`, `e2e-main`). Different trigger, concurrency group, no `schedule`. Its second job is `gh-verify` instead of `upgrade` | **Hand-synced**: touched in the same commit as record.yml on 11-06 (round 3) and 11-11 (bdd7ec5). Nothing checks equality. Today it is in sync |
| `record-q2.yml` (112 lines) | q2-main, q2-shadow only | Earlier generation. Missing 5 steps record.yml now has: step aside if the branch moved, "existing chain must be fully attested" check, attest the *range*, keep the orphaned bundle on ci-runs, 48-hour pending check | **Drifted by design** (frozen 11-05). Harmless while unused; wrong if anyone cites it as "the same as record.yml" |
| `anchor-test.yml` (67 lines) | main, q2/e2e/anchor-test | main vs anchor-test branch: 4 lines (email quoting) | none |
| `record.yml` on anchor-test branch | anchor-test | record.yml as of 11-06, minus the 48-hour check and with quoting differences | stale snapshot; runs only if pushed to main, so inert |

## 2. Part cards

Machine-readable cards (16, schema from earned-parts-audit, validated): `/home/claude/audit/scratch/C/cards.jsonl`.
LOC counts are lines. "Code" excludes tests, experiments and data. Tests were run on `main` e093a60 unless noted.

| Part | Claims (README or docstring) | LOC (code / tests / docs) | Who depends on it | Spot-check of one claim | Earns its place? |
|---|---|---|---|---|---|
| **probe/** | measures memory levels, bandwidth, compute ceilings, meter tiers; 40 tests ~65 s | 1,164 incl. kernels.c / 415 / 146 (+VERIFY.md 98) | **No code consumer** (nothing reads `profile.json`; no workflow runs it or its tests; `router/probe.py` is a different program). Only docs and LLM-review prompts cite it | verified: `test_probe` 40 tests OK (146.9 s here, not 65 s); `--reanalyse profile.json` exits 0 | **Partly.** Well tested, but unused (F8). It is a standalone instrument. Hypothesis: it earns its place only once something consumes its output |
| **counter/** | deterministic cachegrind counts in a fixed sandbox; 23 tests | 615 / 303 / 194 | `loop` (runs `count.py` in its jail), CI `counter.yml` | verified: 23 tests OK (131 s); `count.py -- /usr/bin/true` exit 0, then `--check` exit 0 "identical counts and output" | **Yes.** Used, tested, has a cross-host CI. Debt: 5 of its 13 CLI flags exist for loop's jail (F9) |
| **record/** | append-only hash chain, one CI writer, Sigstore + OpenTimestamps anchors, monitor; 22 tests | 652 / 279 / 807, plus 725 lines of experiments | `loop` (calls `record.py verify`; gate), CI `record.yml`, bench tasks (the benches use `record/` as the task target) | verified: 22 tests OK (1 skipped); `record.py verify` exit 0, head seq 4, sha `9933d676...`. **Not run:** `check-anchors`, `monitor` (network) | **Yes, as the trust root, but heavy for what it holds** (4 entries, F15). Its own tests never run in CI (F5) |
| **ledger/** | the chain's data | 0 code; 4 entries, 124 KB | `record`, `loop` | verified by `verify` (above) | Yes. 2 plans and 2 verdicts; plan 1 is about the ledger itself, plan 2 is `inconclusive` |
| **loop/** | small classifiers vs 2 baselines under a jail, tier-2 meter, test set gated on a ledger plan; 20 tests | 1,945 (loop.py 666; confine.c, timer.c, seccomp_filter.h; 5 candidate/baseline dirs) / 331 / 202 | pilots (`evaluate`), bench tasks, explore (builds candidates' `infer.c`) | verified: 20 tests OK (109 s) in `confine` mode. bubblewrap path **not exercised** | **Yes.** It is the only consumer of probe-class measurements and the only part with a scientific result (plan 2, inconclusive). Debt: frozen digests (F12), copied candidates (F1), committed binary (F10) |
| **loop jail** (confine.c, seccomp_filter.h, bwrap code in loop.py:153-320) | fail-closed isolation of untrusted code, proven on the host before use | ~250 | loop, counter (`--confine`) | verified indirectly: loop tests passed in confine mode, which is selected only after `check_confine()` proves it | Yes, but it is a second sandbox next to counter's fixed-path sandbox and the benches' Docker (M7) |
| **router/engines+probe+review** | "free-engine router. For now: a probe" | 270 / 0 / 0 | only `probe.py`, `review.py`, `router.yml` | verified: **no function routes or selects an engine** (grep `def route\|select\|choose\|pick` finds only `pick_model` and a header picker). The bench harness calls OpenHands/OpenRouter directly | **Name is not earned** (F7). The code is a client and a probe. It works; `router.yml` has failed 12 of 12 runs (F6) |
| **router/bench1-2** | one-off agent trials | ~450 | `bench/grade.py` and `tasks.json` are reused by bench3-5 (sha-pinned) | `bench.yml`, `bench2.yml`: 1 run each, both success, dispatch-only | History. The two workflows are inert; `grade.py` and `tasks.json` are live shared parts |
| **router/bench3-5** | pre-registered agent benchmark; "every rule is code in decide.py"; selftest before each batch | ~1,800 / 0 (selftest inside decide.py) / 694 | CI bench3/4/5 workflows; `record/` as the task target | verified: `decide.py selftest` OK for bench3 (9 labels), bench4 (8), bench5 (9). The 4 sha256 pins in bench5 PREREG match the files on main | **Yes as method; duplicated as code** (F2). Not in the ledger, as the README admits |
| **verifications/** | independent verifier Z's reports and artefacts | 284 scripts, 29,027 data lines | cited by README, ledger entry 4 | `monitor-clean` artifact lists refs checked including the 4 side branches | Yes, as evidence archive. Not code |
| **runs/** (main) | counter records | 4 JSON files | counter default `--out-dir` | present | Marginal: 4 files; the same data class lives in `ci-runs` |
| **.github/workflows** | 12 files on main | 1,472 | all | see section 1 and F3, F5, F6 | Mixed: 3 one-shot experiment workflows (anchor-test, record-e2e, record-q2) are dead since 11-05; bench/bench2 inert; counter/loop collectors idle until a path changes |
| **pilots/lgn-20261115, -2** (pilot branch) | logic-gate network vs mlp16; dev only, test never scored | 3,581 / 0 / 263 | none | verified: recomputed from `scoreboard-dev.json`: lgn_b 0.85x Ir and +72.3 ns; lgn_d 0.66x Ir and -27.9 ns, as the README says. No code path in `pilots/` or `explore/` reads `optdigits.tes`. Split is labelled `dev-in-sample` | Result earns its place (a negative plus a near-miss, honestly labelled). **Code does not**: 6 copies of ~20 training functions (F1) |
| **explore/mlp16-speed** (pilot branch) | open question kit; why mlp16 retires fewer instr/ns | 2,673 / 0 / 551; 10,963 data lines | README row on the pilot branch | verified: `results/staging-evidence.json` page faults: bigbuf 1,662 vs mmap 262 per run (report table says 1,663) | Earns its place as an investigation. Debt: about 20 single-file C variants 27-57 diff-lines apart; generated `.s`/objdump committed |
| **ci-runs / router-runs content** | raw CI outputs | 131 MB | cited by READMEs | see F14 | Partly: raw cachegrind gz for 20 counter runs is 66 MB to carry two numbers per run |

### Dependency edges between modules (verified by grep)

`loop -> counter` (subprocess to `count.py`, path-hardcoded at loop.py:50), `loop -> record` (subprocess `record.py verify`, loop.py:51, 589), `counter -> loop` (flag names and docstrings referring to `loop/confine.c`: inversion, F9), `router/bench* -> record` (as the task a benchmark agent must fix, not as a library), `bench5 -> bench3` (loads `decide.py` by file path with sha pin). `probe` has no inbound code edge. There are no Python imports between bricks; coupling is by subprocess, path and file digest, which is a clean seam but a fragile one (F12).

## 3. Claims re-checkable by an independent agent (for stage 2)

| Claim | Command (from a clone, worktree of the stated commit) | My result |
|---|---|---|
| probe 40 tests | `cd probe && python3 -m unittest test_probe` | OK, 40 |
| counter 23 tests | `cd counter && python3 -m unittest test_count` | OK, 23 |
| record 22 tests | `cd record && python3 -m unittest test_record` | OK, 22, 1 skipped |
| loop 20 tests | `cd loop && python3 -m unittest test_loop` | OK, 20 (confine mode) |
| ledger chain intact | `cd record && python3 -I record.py verify; echo $?` | exit 0, 4 entries |
| bench rules selftest | `python3 -I router/bench{3,4,5}/decide.py selftest` | OK x3 |
| bench5 pins | `sha256sum router/bench3/{decide,run_agent,measure}.py router/bench/grade.py` | match PREREG |
| pilot numbers | read `pilots/*/scoreboard-dev.json` `candidates[]` | match README |
| exact-twin shares | `python3 -I scratch/C/tools/clones.py <root>` | section 5 |
| CI run history | `gh api repos/5kasgard/exagium/actions/workflows/<file>/runs` | section 1, F5, F6 |

## 4. Findings

Severity is my judgement of cost to the owner's stated worry ("complexity debt accruing by itself"). Fixes are options; deletions are the owner's call. "Test" is the cheapest check that would settle which option fits (taste-and-adjust).

| # | Finding | Location | Evidence | Sev | Status | Options |
|---|---|---|---|---|---|---|
| F1 | **Copy-by-directory duplication of training code.** The candidate contract (a dir holds exactly `train.py`, `infer.c`, `weights.bin`; jailed with only the train file) forces self-contained copies. `mlp16/32/64/train.py` differ in one line (`H = 16/32/64`). On the pilot branch ~20 functions of the LGN trainer exist in 6 files (`lgn_a`..`lgn_d`, `sweeps/exp2.py`, `exp3.py`); `infer.c` is byte-identical in all 4 LGN dirs (md5 a665dafb). Of 25 distinct function names across `lgn_b`, `lgn_d`, `exp2`, `exp3`, 20 are identical wherever they appear | `loop/candidates/*/train.py`; `pilots/lgn-20261115*/lgn_*/`, `sweeps/` | clone detector, section 5; md5 and diff in `scratch/C/dup/` | Med | verified | (a) leave: the dir is the frozen-digest unit, copies are the price of a clean freeze. (b) one shared `lib/` dir whose digest is frozen in the plan, copied into the jail next to each candidate, with variants as a small params file. (c) a generator that writes variant dirs from one source and records the parameters. Test: change one shared function and count files touched (now 6) |
| F2 | **Three `decide.py` re-implementations.** `_binom_cdf` identical in bench3 and bench4; `interval`, `classify`, `load`, `cells`, `futile`, `markdown`, `selftest` evolved separately (443 and 573 changed lines between neighbours). bench5 imports bench3's rules by path with a sha pin instead. Two strategies for the same need | `router/bench{3,4,5}/decide.py` | AST and diff in `scratch/C/dup/` | Med | verified | (a) keep copies: "rules frozen as code" is the reason, PREREG says so. (b) extract the shared statistics to one file and pin it by sha in each PREREG, as bench5 already does for bench3. (c) add a test that fails when a helper listed as shared drifts. Hypothesis: (b), because the repo already contains the precedent |
| F3 | **Test copies of the trust-root workflow.** `record-e2e.yml` must equal `record.yml` to be a valid test of it; equality is kept by hand. It matches today (append job identical after name normalisation). `record-q2.yml` lacks 5 later security steps. Three experiment workflows are dead since 2026-11-05. GitHub lists 13 workflows vs 12 files on main | `.github/workflows/record-e2e.yml`, `anchor-test.yml`; q2 branches | diff in `scratch/C/wf/`; run histories | Med | verified | (a) a CI step that diffs `record.yml` against `record-e2e.yml` after name substitution. (b) generate the copy from the original. (c) retire the e2e copy and keep its results (they are in `ci-runs/e2e` and `record/README`). Do not describe the q2 copy as "the same steps" |
| F4 | **Stale test branches.** `anchor-test`: 0 unique commits. `q2-main`, `q2-shadow`, `e2e-main`: unique content is only test ledgers and anchors, last touched 11-05/06. README and Z's artifact mention them. `record.py monitor` builds its ref list from branches, tags, PRs and "every ref in the activity log (which keeps deleted branches while the repo exists)" | branches; `record/record.py:521-531` | section 1 | Med | **partly suspected**: the claim that GitHub's activity log retains deleted refs is the code docstring's; I could not confirm it (proxy blocked pagination; first page lists only 4 recent refs) | (a) `git tag archive/<name> <branch>` for each, then delete the branches (owner's call). (b) keep as is. (c) first test on a throwaway repo that a deleted branch's ref still appears in `/activity`, because the monitor's completeness for those refs depends on it. Lowest risk: `anchor-test` |
| F5 | **CI green does not mean tests passed.** `loop.yml`: `unittest ... | tee ... || true` and the same for reproduce and evaluate; 22 of 23 runs "success". `counter.yml`: unit tests use `pipefail`, but count steps end `|| true`. The 22 `record` tests and 40 `probe` tests run in no workflow; `record/record.py` can change with no check until the next ledger push or daily run. Only counter's unit tests gate | `.github/workflows/loop.yml:103-108`, `counter.yml:28-45` | grep for `unittest` finds loop.yml and counter.yml only | Med | verified | (a) add one workflow that runs the four unit-test suites (record's takes 1.4 s) on push to the code paths. (b) keep collectors as collectors and rename them so green means "ran". (c) have the final `collect` job fail if any recorded `exit N` is nonzero |
| F6 | **Red CI is normal, so red is ignored.** `router.yml`: 12 of 12 runs failed (probe exits 1 if any engine fails, by design). `record` daily schedule failed 5 days running (11-06 to 11-10) before bdd7ec5 ("failed silently"); the repo then added a 48-hour check | `router/probe.py` docstring; `.github/workflows/router.yml`; record run history | gh API | Med | verified | split exit codes: "workflow broken" vs "an engine is down" (write the data, exit 0, put the verdict in the file). Add a notification for the trust-root workflow only |
| F7 | **"router" has no routing.** Nothing selects an engine; `engines.py` is a client used only by `probe.py` and `review.py`; the bench agents call OpenHands/OpenRouter directly. The module is not in the README status table. Its biggest part (bench1-5, ~2,100 lines) is a benchmark of agents on `record/` tasks | `router/` | grep; README table has probe, counter, record, loop only | Low-Med | verified | rename or document as "free-pool measurement" until a router exists; add a status row. Do not wait for the name to become true |
| F8 | **probe/ is an unconsumed leaf.** 1,164 code, 415 test, 146 doc lines plus a 98-line VERIFY brief; no code reads `profile.json`, no workflow runs it; the benches that need machine facts (`explore/bin/clock`) wrote their own | `probe/` | grep on main and pilot | Low-Med | verified | (a) keep as a documented instrument. (b) feed `meter_tiers` into `loop`'s scoreboard header so one consumer exists. (c) move to `tools/` and stop calling it a brick |
| F9 | **counter knows loop's jail.** `--stdout-record`, `--single-process`, `--confine`, `--confine-ro`, `--confine-ls` (5 of 13 flags) and docstrings name `loop/confine.c`. The lower layer's interface is shaped by its consumer | `counter/count.py:217, 521-532` | grep | Low | verified | extract the jail (confine.c, seccomp_filter.h, bwrap logic ~250 lines) to a `jail/` directory with its own README, and let both bricks call it; or just document the confine protocol once |
| F10 | **Committed ELF binary** `loop/confine` (20,712 B) next to `confine.c`. `loop.py` builds its own copy into a temp dir, so the committed binary is never run by the code; a rebuild here differs byte-for-byte (compiler flags). Added in commit d9d756d ("Z's verification") | `loop/confine` | `git ls-tree`, `cmp`, grep | Low | verified | remove and gitignore it. In a repo whose point is verifiability an unexplained binary is cheap to avoid. Owner's call |
| F11 | **Triplicated digest helpers.** `canonical()` is byte-equivalent in `probe.py:623`, `count.py:53`, `record.py:39`; `sha256_file` exists in 3 forms (streaming in counter, read-all in loop and probe). All bricks are stdlib-only and run under `python3 -I`, so sharing means a copy or a vendored file | those lines | read | Low | verified | leave, and add one test that feeds a fixture to all three and compares bytes (the digest contract is the thing that must not drift). Note `record/experiments/.../jcs_demo.py` documents that Python canonical JSON is not RFC 8785 in general; record refuses floats and non-ASCII keys to stay equal |
| F12 | **Fixes to lower bricks fork history.** Plan 2 froze `loop.py`, `counter/count.py`, `record/record.py`, task and candidates. Z's fix to `record.py` after entry 4 means plan 2 must be re-measured at old commit `113a3f6`, not at HEAD | `loop/README.md:13` | quoted | Med | verified (doc) | accept and write it down per plan (done), or vendor the frozen tools into the plan's directory so HEAD can move. This is the cost of freezing by digest (M2) |
| F13 | **Name collisions.** `probe/probe.py` vs `router/probe.py` (different programs); `router/review.py` vs `router/review/` vs `review.yml`; "record" is a brick, a workflow, a ledger and a task. Makes grep and review harder | | | Low | verified | rename the router ones (`pool_probe.py`, `outside_review.py`) |
| F14 | **Storage branches are 97% of the bytes and 85% of git objects (6,796 of 8,031), unprotected, and cited as evidence.** ci-runs 77.9 MB, router-runs 53.6 MB, main 3.6 MB; packed objects 5,307 / 1,489 / 902 of 8,031 (pack 21.3 MB). `ci-runs/runs/` holds about 19 runs x ~152 cachegrind files (2,856 files, 66.5 MB) to establish equal counts. `record/README` cites `ci-runs` files as evidence; the ledger entries contain 0 references to either branch; `ci-runs` and `main` have `protected: false`. Any clone of all branches pays for it | `ci-runs`, `router-runs` | `git ls-tree -l`, `git count-objects`, `gh api branches/*` | Med | verified | (a) keep raw files only for runs that disagree; store a summary and digest for the rest. (b) move cited evidence (11 small files) to main or a tag. (c) anchor a digest of each storage branch tip through the ledger. (d) protect the branches |
| F15 | **Cost-to-use ratio of the ledger.** `record` is 652 code + 279 test lines, 3 workflows (132 + 135 + 67), 725 experiment lines, 807 doc lines and 29k lines of verification data, to protect 4 entries. The 5 benches (the largest data) are not in it ("not in the ledger yet", README). It earns the narrow claim "tamper-evident log" and passes its checks; no bench conclusion yet leans on it | `record/`, `ledger/` | `record.py verify`, counts | Low-Med | verified (counts); the judgement is mine | keep, and register the next bench plan in it so the cost buys something; or state in the README that the ledger is a proof of mechanism so far |
| F16 | **Explore and pilots on an unmerged branch carry duplication and generated files.** about 20 single-file C variants (27-57 diff lines apart); 7 `programs_*.json`; `reference/*.s` and objdump (45 KB each) committed; `lgn_orig.c` == `lgn_d/infer.c` and `mlp_interleave.c` == `infer_x.c` (copies). Fine on a branch, debt if merged as is | `explore/mlp16-speed/` | diff | Low | verified | when merging: keep reports and results, regenerate `reference/` from `build.sh`, pick one copy of each C file |

Things checked and found fine: file sizes are moderate (largest `probe.py` 884 lines, `loop.py` 666, `record.py` 652, `count.py` 580; longest function 161 lines, `describe_levels`); no Python imports between bricks; the pinned actions are mostly SHA-pinned (21 `checkout` uses are SHA-pinned; `upload/download-artifact@v4` and 2 `checkout@v4` are not).

## 5. Duplication scan with controls (mechanical, `scratch/C/tools/clones.py`)

A function of at least 6 lines counts as an exact twin when its AST (docstring removed, names kept) equals one in a different file. Renamed-identifier twins add nothing beyond exact twins here.

| Tree | Functions | Exact-twin groups | Lines beyond the first copy | Share of function lines |
|---|---|---|---|---|
| exagium **main** (tests excluded / included) | 184 / 261 | 9 | 227 | **5.6% / 4.6%** |
| exagium **pilot branch** (tests excluded / included) | 281 / 358 | 27 | 1,710 | **29.0% / 25.2%** |
| control `psf/requests` (excl / incl tests) | 178 / 440 | 0 / 1 | 0 / 21 | 0.0% / 0.3% |
| control `pallets/click` (excl / incl tests) | 428 / 1,014 | 0 / 0 | 0 / 0 | 0.0% / 0.0% |

Caveats: the controls are two mature libraries that are unrelated to each other, not research harnesses, so the true base rate for "research code with experiments" is probably higher than 0.3%. They show that 25-29% is not normal for library code and that 5% on main is high but not extreme. Main's twins are the three `mlp*` trainers, the two baselines, and bench3/bench4 helpers. The 2-4 line `canonical()` copies fall under the 6-line cut-off and were counted by hand (F11). The detector is one automated pass; no second agent reviewed the groups, so mark groups "high confidence" only where I diffed them (all listed ones).

## 6. Moved, not removed

Each block: hard core, move, where it went, cheaper or dearer there, leaks, prediction to test on the original.

**M1. Trust in results (verdict independence).**
- Hard core: a claimant can lie or fool themselves.
- Move: append-only chain, external anchors, and "a verdict from someone other than the claimant".
- Moved to: the anchors fix *when and in what order* (README: "anchoring can't show that a result was genuine"); genuineness moved to a second agent. Ledger entries 2 and 3: proposer `claude-loop1-proposer`, verifier `claude-loop1-verifier`, both Claude sessions; entry 4: `glm-on-z` (different model). Outside LLM reviews run from committed prompts through free engines (`review.yml`).
- Cheaper because: the check is exact and public for order and time. Dearer for truth: independence is by session or model family, not by a party with different incentives.
- Leaks: 1 of 2 verdicts is same-family. Anchors do not cover `ci-runs` or `router-runs`.
- Prediction: a plan whose proposer and verifier share a model family will pass more often than one judged across families. The ledger has too few entries to test; the next 5 plans could.

**M2. Test-set protection.**
- Hard core: nobody may peek at, or tune against, the test set.
- Move: the test set is scored only if the ledger holds a plan that froze exactly the files; `loop` calls `record.py verify` by subprocess.
- Moved to: record.py's exit code and the plan's frozen digests (including `record.py` and `count.py` themselves).
- Leaks: any fix to a frozen tool breaks re-measurement at HEAD (F12). Pilots avoid the problem by never scoring the test set: no code path reads `optdigits.tes`.
- Prediction: after the next fix to `count.py` or `record.py`, `loop evaluate --test test --plan 2` at HEAD will refuse. (Not run: it needs network for `check-anchors`.)

**M3. Cost: compute, quota and storage moved to free tiers and to git.**
- Hard core: measurement and agent runs cost compute, money and attention.
- Move: GitHub free runners (counter 8 jobs per run, loop 6), free LLM allowances ("1,000 requests a day, shared by every bench", bench5 PREREG; OpenRouter 50/day in probe docstring), results stored by committing them.
- Moved to: GitHub's quota and the repo's size (131 MB on two branches, 66% of objects on ci-runs alone, F14).
- Leaks: the shared daily quota couples benches to each other (bench3 and bench4 and bench5 cannot all run); a `router` probe that fails 12 of 12 runs is the quota showing through.
- Prediction: clone time and size grow linearly with each bench batch (about 60-400 files per bench3/4 batch).

**M4. I/O cost hidden from the meter.**
- Hard core: "energy per correct answer" should include the whole run.
- Move: tier 2 = cachegrind instruction counts (user space) plus native ns timing.
- Moved to: kernel work (page faults, `fread` copies) is invisible to the instruction count and appears only in the ns half. Evidence (verified from the report and results file): one big fresh `n x 64 B` buffer (mlp16's way) costs 1,662 page faults per run vs 262 for `mmap`, and **+41.8 ns/img (IQR 34.8-53.5, 21 of 21 rounds)** more than chunked reading. That exceeds pilot 2's best headline (lgn_d -27.9 +/- 14.9 ns vs mlp16).
- Leaks: a candidate's ns score mixes compute with staging. Whether lgn_d and mlp16 stage input the same way is **suspected, not checked**; `explore/` contains `mlp16_chunked.c` for exactly this control.
- Prediction: run lgn_d and mlp16 with identical input staging; if the ns gap changes by about 40 ns, the pilot-2 ranking is a staging effect. (Staging was measured on a different VM from the pilots, so compare only within one run, as the report says.)

**M5. Rules cannot change after seeing data, so code is copied.**
- Hard core: pre-registration needs frozen decision rules.
- Move: each bench gets its own `decide.py`, sealed by sha256 in PREREG and a `REGISTERED` file.
- Moved to: copy proliferation and hand-synced helpers (F2), and for the record workflow a copy kept equal by hand (F3).
- Prediction: a bug in a shared statistic (the Clopper-Pearson `interval`) would need fixing in 2-3 places, and the copies already differ (verified `interval` 16 vs 15 lines).

**M6. Pass/fail judgement moved out of CI status.**
- Hard core: "did it work?"
- Move: workflows collect output, and results are exit-code lines inside files committed to `ci-runs`/`router-runs`.
- Moved to: whoever reads those files; job status then means "ran" (`|| true`, 22/23 loop runs green) or "probe saw a dead engine" (router 0/12 green).
- Prediction: a regression in `loop` or `record` unit tests would not turn any workflow red. Settled by F5's cheapest test: break a test on a scratch branch and see.

**M7. Running untrusted code.**
- Hard core: candidate `train.py` and benchmark agents are code written by models.
- Move: three isolation mechanisms: counter's fixed-path sandbox, loop's bwrap/confine jail, the benches' Docker containers.
- Moved to: kernel features (user namespaces, Landlock ABI 4, seccomp). Cheaper because each is proven on the host before use (fail closed). Leak: hosts without bubblewrap need Linux 6.7+; this audit host fell back to confine.
- Prediction: on a host with neither, `loop` refuses to run (by design).

**M8. Variation moved into directories** (F1): "a candidate is a frozen dir" turned parameters into file copies.

## 7. What earns its place (summary, hypotheses for the owner to taste)

- **Earns, with evidence:** counter, loop (plus its jail), record as the trust root, bench3-5 method (selftest passes, pins match), pilot results as honest negatives, verifications as evidence archive.
- **Earns only conditionally:** probe (no consumer), the ledger (4 entries; pays off when bench plans enter it), `explore/` (an investigation, not a module).
- **Does not earn as it stands:** the name "router" (no routing); the committed `loop/confine` binary; the three experiment workflows and branches after their results are copied (anchor-test, record-q2, record-e2e); bench/bench2 workflows as live workflows (their `grade.py` and `tasks.json` are live shared files); raw cachegrind gz for runs that agree.
- **Debt that grows by itself:** copy-by-directory candidates (F1) and per-bench `decide.py` (F2) grow with every new experiment; storage branches grow with every batch (F14); CI that cannot go red (F5, F6) hides the first regression.
- Smallest cuts that change the trend (options, not recommendations): (1) one workflow running the 4 unit suites; (2) a shared, sha-pinned statistics file for future benches; (3) a shared candidate library or generator before pilot 3; (4) tag then retire the four stale branches; (5) shrink what `ci-runs` stores per run.

## 8. Open items for the parent session

1. Run stage 2 (independent verification of sections 2 and 4) and stage 5 (two blind groupers over `cards.jsonl` with the controls). I could not spawn agents.
2. Confirm F4's activity-log assumption on a throwaway repo before deleting any branch.
3. Run `record.py check-anchors` and `monitor` from a networked clone (not done here).
4. Settle M4's prediction with identical staging for lgn_d and mlp16.

## Files

- This report: `/home/claude/audit/out/C-structure.md`
- Part cards (JSONL, validated): `/home/claude/audit/scratch/C/cards.jsonl`
- Clone detector and outputs: `/home/claude/audit/scratch/C/tools/` (`clones.py`, `out_*.json`)
- Workflow copies and diffs: `/home/claude/audit/scratch/C/wf/`; duplicated-source copies: `/home/claude/audit/scratch/C/dup/`; test logs: `/home/claude/audit/scratch/C/tests/`; control clones: `/home/claude/audit/scratch/C/controls/`
