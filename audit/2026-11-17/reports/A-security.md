# Audit A: security and trust (exagium, all 8 branches)

Date: 2026-11-17. Auditor: Claude (Opus), one agent. Repo clone: /home/claude/audit/repo (read-only; no pushes, no triggers, no API calls with any key).
Scratch and reproductions: /home/claude/audit/scratch/A/

## 1. Scope and threat model (security-code-review step 1)

- **Finding scope:** every branch (main, pilot/lgn-20261115, ci-runs, router-runs, q2-main, q2-shadow, e2e-main, anchor-test), all workflows, loop/ jail, record/ + ledger/, counter/, probe/, router/.
- **Attackers considered:** (a) untrusted *content producers* with no repo access: the free LLMs (and anyone who can steer them) whose outputs drive agents in bench-3/4/5 and whose replies are committed; candidate code run in the jail; (b) *supply chain*: mutable action tags, PyPI/git/container dependencies; (c) *outside readers* of the public repo and Actions logs (secret leaks); (d) *a repo writer* only where the project claims to defend against one (the ledger's "verdict by someone other than the claimant").
- **Entry points:** agent-written workspace (incl. its `.git`), candidate train.py/infer.c, LLM replies, PyPI/git/ghcr downloads, workflow_dispatch inputs, branch names in `github.ref`.
- **Trust boundaries:** jail (bwrap or confine) vs harness; agent container vs runner host (runner user has passwordless sudo; Runner.Worker holds the job's secrets in memory); job token scopes; the writer identity `record.yml@refs/heads/main`.
- **Existing hardening seen:** SHA-pinned checkout/attest in record/bench3-5, `persist-credentials: false` in most jobs, `python3 -I`, network seals, bench-4 meter (key never enters container), grading in `--network none` container, symlink stripping, concurrency groups, ":free" model gate.

## 2. Coverage (what ran, what did not)

| Tool / step | Result |
|---|---|
| Semgrep 1.185.0 (`--metrics=off`; p/security-audit, p/secrets, p/python, p/c, p/github-actions) on main tree | Ran, 259 rules, 213 files, 33 hits. 45 errors: partial parsing of 9 workflow files + seccomp_filter.h, 3 JSON syntax errors. Workflows covered by zizmor instead. |
| zizmor 1.32.0 `--offline --persona=auditor` on main, q2-main, e2e-main, anchor-test workflows | Ran: 442 findings (41 unpinned-uses, 40 excessive-permissions, 9 template-injection errors, 1 unpinned-images, 29 artipacked, 30 secrets-outside-env). Offline: no impostor-commit / known-vuln-action checks. |
| CodeQL | **Not run** (not installed; time budget). Manual by-bug-class review of the C (confine.c, seccomp_filter.h, timer.c) instead. |
| Secret scan, git history | All 257 commits on all branches, `git log --all -p` (75 MB): 14 token formats + generic `key/token/secret = <24+ chars>`. 1,941 committed `.gz` agent transcripts decompressed and scanned. Actions evidence JSON scanned. |
| Hidden-instruction scan (untrusted-content-guard s.3) | 3,503 unique text blobs on all branches + 1,941 gz: zero-width/bidi/tag chars and AI-addressed phrases. |
| Jail live probe | confine.c built and run locally (kernel 6.18, Landlock OK) through loop.py's own `confined()` with a probe script. **bwrap mode not tested** (bwrap not installed). Test ran as root, so uid-based effects (e.g. sudo) are only partly meaningful. |
| Adversarial review of findings | No sub-agent tool available in this session: findings are **self-verified, not independently confirmed**. The owner's preference is a separate judge agent; recommend one re-checks C1, M1, M2 before acting. |

## 3. Findings

Severity scale: critical / major / minor. Status: **verified** = reproduced in scratch or by an exact tool output; **verified (static)** = the code/config says it unambiguously, not executed; **suspected** = plausible, not reproduced.

### CRITICAL

**C1. Agent-controlled `.git/config` runs code on the runner host during grading (bench-3, bench-5; also dormant bench-2)**
- Location: main:.github/workflows/bench3.yml:159-161; main:.github/workflows/bench5.yml:174-176; main:.github/workflows/bench2.yml:79,83 (no hardening at all there).
- Claim: the grade step runs host-side `git -c core.fsmonitor=false -c core.hooksPath=/dev/null -c safe.directory=* -C "$WS" add -A` and `diff --cached` on `$WS`, which was mounted read-write into the agent container (`$WS/.git` included). Disabling hooks and fsmonitor does not disable **filter drivers** (`filter.<x>.clean` + `.gitattributes`) or diff drivers/external diff, which git runs from the repo's own config. An agent (i.e. any free model's output) can plant them and get a shell on the runner as `runner` (passwordless sudo).
- Impact: forge grades (rewrite `$P/fix_test.py`, `measure.py` or the output dir before grading), poison the artifact later committed to public router-runs, and with sudo read Runner.Worker memory, which holds the job's secrets (OPENROUTER_API_KEY in bench-3, NVIDIA_API_KEY in bench-5). Both workflows are live: bench-3 ran on schedule 12x, last 2026-11-17T08:26Z; bench-5 last 2026-11-16 (evidence/actions_runs.json).
- Evidence (verified): scratch/A/gitfilter/: a repo with `filter.planted.clean = sh -c 'touch MARKER; cat'` and `* filter=planted`, then the workflow's exact `$G -C "$WS" add -A -- . ':(exclude)conversations'` -> `MARKER_ran_on_host` created ("VERIFIED: filter driver ran on host despite fsmonitor/hooks off", git 2.43.0, same major as ubuntu-24.04).
- Fix: never run host git against an agent-writable `.git`. Before grading, move the agent's `.git` aside and diff against a harness-owned git dir (`git --git-dir=$P/base.git --work-tree=$WS` with `-c core.attributesFile=/dev/null` and `GIT_ATTR_NOSYSTEM=1`, and delete any `.gitattributes` from consideration), or compute `agent.diff` inside the `--network none` grading container (as bench-4 effectively does: no host git on WS was found there). Variant search: bench4.yml host git touches only `src` and the pre-attempt WS (clean); bench2.yml has the same flaw plus hooks enabled.

### MAJOR

**M1. Any agent edit can forge a full pass: the grader trusts the first `@@RESULTS@@` line on stdout of a process that imports agent code**
- Location: main:router/bench/grade.py:14-38 (RUNNER and `run_tests`), used by router/bench3/measure.py:31,43 (bench-3/5 final grade and "visible green").
- Claim: test_record imports the agent's `record.py`; code at import time can print `@@RESULTS@@{...all "pass"}` before unittest runs, and `run_tests` returns the first such line.
- Evidence (verified): scratch/A/grade_forge/: record.py prints a forged line on import; both real tests fail; `grade.run_tests` returned `{'test_hidden': 'pass', 'test_a': 'pass'}` (result.txt). The `--network none` container (2026-11-13 fix) does not help: this is in-process.
- Impact: bench results and the decisions decide.py draws from them (STATUS on router-runs) can be wrong with no trace except in agent.diff. Reward-hacking by a model is a realistic, not exotic, path.
- Fix: no in-process fix is complete. At least: read results only from a file descriptor the parent opens and passes (not stdout), plus a per-run nonce; fail the attempt if anything precedes the sentinel; add an automated diff check for module-level side effects (prints, monkeypatching `unittest`/`sys`). Best: hidden tests that exercise the code through a separate process boundary.

**M2. bench-3/bench-5: real API key inside the agent container, and the "seal" leaves exfiltration paths**
- Location: main:.github/workflows/bench3.yml:118-127 (allows all of `cloudflare.com/ips-v4` + DNS 53/udp,tcp to any IP); main:.github/workflows/bench5.yml:131-133 (DNS to any IP + NIM IPs); main:router/bench3/run_agent.py:63-66 (key handed to the SDK's remote workspace); router/bench3/PREREG.md:117 (owner acknowledges the key is in the container).
- Claim: the acknowledged limit is "can reach the provider's network", but the rule set admits every Cloudflare-fronted host (any `*.workers.dev`, any Cloudflare customer) and direct DNS to arbitrary resolvers (a tunnel). The seal check only proves github.com is refused. router/engines.py:104-105 says the OpenRouter account holds credit, so a key in an agent's reach can also be spent directly on paid models, bypassing the harness's `:free` gate.
- Evidence: verified (static) from the iptables rules; not live-tested (no keys used, by brief).
- Fix: move bench-3/5 onto bench-4's meter design (key only in the host meter, no DNS, single port). Short of that: restrict DNS to Docker's embedded resolver, allow only openrouter.ai's resolved /32s on 443, and put a hard credit limit on the OpenRouter key (or a dedicated zero-credit key) for agent runs.

**M3. The ledger writer job resolves 35 unpinned transitive packages while holding OIDC + attestation + contents:write**
- Location: main:.github/workflows/record.yml:44-47 and :129-131 (`pip install sigstore==4.5.1 opentimestamps-client==0.8.0`), job permissions :28-31; same in record-e2e.yml and q2/e2e branch copies.
- Claim: only the two top-level packages are pinned; `ACTIONS_ID_TOKEN_REQUEST_*` is available to every step of a job with `id-token: write`, so any compromised dependency running in the `tools`/`check-anchors` steps could mint a genuine `record.yml@refs/heads/main` attestation, which is the ledger's root of trust. README says "pinned by version, not by hash" but not that the tree floats.
- Evidence (verified): `pip install --dry-run --report` resolved 37 packages (cryptography, requests, urllib3, pydantic, GitPython, tuf, python-bitcoinlib, ...); scratch/A/deps.json.
- Fix: `pip install --require-hashes -r record/requirements.lock` (hash-locked full tree); or run tool installation and anchor checking in a job without `id-token`, passing only files to a minimal attest job.

**M4. Third-party code fetched by mutable reference runs with an API key (bench-3, bench-5)**
- Location: main:.github/workflows/bench3.yml:23-24,110-113; bench5.yml:25-26,117-121.
- Claim: OpenHands SDK installed from git tag `v1.61.0` (tags are mutable), `ghcr.io/openhands/agent-server:1.61.0-python` pulled by tag (no digest), `pip install uv` unpinned; the SDK runs in the step holding `LLM_API_KEY`, and the image receives the key (M2).
- Evidence: verified (static); zizmor flags the image class (unpinned-images) on loop.yml:72.
- Fix: pin the SDK by commit SHA, the image by `@sha256:` digest (bench-5 already hashes its own instrument files; extend that), uv by version+hash.

**M5. "Verdict by someone other than the claimant" is not authenticated**
- Location: main:record/record.py:24 (`NAME` regex), :107-111 (`_name`), :205-212 (same-`by` rule); claim at main:record/README.md:3.
- Claim: `by` is a free-text handle in the inbox JSON. Whoever can push to `ledger/inbox/` on main can write a plan as `alice` and a verdict as `bob`; nothing binds `by` to a GitHub identity or signature. The ledger proves *when* and *that the writer workflow ingested it*, not *who* judged.
- Evidence: verified (static): the body schema has no identity proof and the writer records no pusher/actor.
- Fix: have the writer record the GitHub actor and commit author of the commit that added each proposal into the entry (and reject `by` that does not match a mapping of handles to accounts), or require proposals signed with Sigstore (identity = a person's account). Until then, reword README:3 to "a handle other than the claimant's".

### MINOR

**m1. Mutable `actions/*@vN` tags in jobs with write tokens.** main:.github/workflows/anchor-test.yml:20,35,46,55 (contents+id-token+attestations write); counter.yml:21,41,52 (collect: contents write); loop.yml:60,109,120 (collect: contents write); bench.yml:93,107; bench2.yml:85,96. zizmor: 41 unpinned-uses. Verified (tool). Fix: pin by SHA like record.yml.

**m2. Over-broad permissions / persisted tokens.** router.yml:13-14 (workflow-level contents:write; checkout :22 persists the token while probe.py runs with 15 secrets in env), review.yml:13-14,23, bench.yml/bench2.yml workflow-level write, anchor-test.yml:8-11. zizmor: 40 excessive-permissions, 29 artipacked. Verified (tool). Fix: `contents: read` at top, write only in the push step's job, `persist-credentials: false`.

**m3. review.yml copies a dispatcher-chosen path into a public branch.** main:.github/workflows/review.yml:33-34 (`cp "$PROMPT" .../prompt.txt`, then pushed to router-runs). A path such as a /proc environ file of the step would publish the step's secrets (4 keys in env). Needs write access (dispatch), so a footgun, not an escalation. Suspected (not executed). Fix: `realpath` the input and require it under `router/review/` inside the checkout.

**m4. Stale branches keep live, privileged, push-triggered workflows.** q2-main/q2-shadow (record-q2.yml, record.yml), e2e-main, anchor-test: contents/id-token/attestations write, unpinned actions, and `${{ github.ref }}` / step outputs interpolated into `run:` (zizmor template-injection errors: q2-main:record-q2.yml:55,60,87,95; q2-main:record.yml:57,62,79). Only writers can trigger; they add extra signing identities. Verified (tool). Fix: delete the workflow files on those branches (or the branches), or disable the workflows in repo settings.

**m5. Jail gaps between modes.** bwrap mode (main:loop/loop.py:212-227) has no seccomp filter and no `--disable-userns`, so a candidate can create nested user namespaces (kernel attack surface) where confine refuses `unshare`; neither mode sets memory/file-size/disk rlimits (bwrap's `--tmpfs /tmp` and confine's temp dir are unbounded), so a candidate can exhaust runner RAM/disk within the timeout. Suspected (bwrap not available here). Fix: add `--disable-userns` (bwrap >= 0.8) or apply the same seccomp filter via `--seccomp FD`; set RLIMIT_AS/FSIZE and `--size` on the tmpfs.

**m6. Unexplained committed ELF `loop/confine`.** main:loop/confine (x86-64 PIE, added in d9d756d). loop.py builds confine.c fresh (loop.py:276-281) and never uses it; `freeze` digests do not cover it. A verifier following the dir may run an unreviewed binary. Verified (static). Fix: delete it, or record its source commit and sha256.

## 4. Checked and found clean

- **No untrusted-PR triggers anywhere:** no `pull_request`, `pull_request_target`, `issue_comment`, `workflow_run` on any branch. Triggers are push/dispatch/schedule only (writers).
- **No leaked secrets:** 0 hits across all history of all branches, the 1,941 decompressed transcripts, and the Actions evidence files. Committed router/log/calls.jsonl (181 lines) holds no secret base URLs (NARA/CROWLLM errors are generic).
- **No hidden instructions in committed outside content:** 0 zero-width/bidi/tag characters; 5 "system prompt" phrase hits are the project's own prompt/PREREG text, benign.
- **confine jail (live, local):** refused reading /etc/passwd, /proc/self, parent cmdline, /home, the repo README; refused `kill(parent,0)`, `kill(-1,0)`, `unshare(CLONE_NEWUSER)`; sudo could not run. Only `PR_SET_PDEATHSIG` allowed (harmless).
- **seccomp_filter.h review:** arch check kills foreign/x32 ABIs; signals self-only on low 32 bits (pid_t, correct); fork/clone/clone3, sockets, keyring, ptrace, process_vm, pidfd, ns/mount, io_uring, bpf, perf, userfaultfd, SysV+POSIX IPC, metadata, other-process scheduling denied. No gap found in reviewed classes.
- **timer.c / confine.c by bug class:** realloc failure handled; bounds from destination; option parsing checks `i+1`; fails closed (exit 125). count.py makes itself non-dumpable before candidates run.
- **bench-4:** key held only by the host meter, request allowlist rebuild, no DNS/forwarding, IPv6 blocked, no host git on the agent workspace. Clean for C1/M2 classes.
- **record.yml itself:** checkout/attest SHA-pinned; step-aside on moved branch; `python3 -I`; bot email quoted. Paid-model gate in router/engines.py:104 present.
- **Semgrep hits triaged as false positives:** subprocess with env args in bench4 (harness-controlled), dynamic urllib (fixed URLs), `curl | python3 -c` in bench3/4 (JSON parsing, not shell), printf in verifications/*.c (literal formats), test-file chmod.

## 5. Not checked / open risks

- bwrap mode live; CodeQL; zizmor online audits (impostor commits, known-vulnerable actions).
- Whether the OpenHands agent server persists the key to `/workspace/conversations` (excluded from agent.diff; WS not uploaded, so not public either way).
- Rekor/OTS live verification of existing anchors (would call external services; out of brief).
- Repo settings not visible in a clone: branch protection on main/router-runs/ci-runs, Actions "allow GitHub Actions to create PRs", fork-PR approval, secret scanning, environment protection.
- No independent adversarial review (no sub-agent tool here).

## 6. Skills applied

- **security-code-review:** step 1 threat model (s.1); Semgrep with telemetry off and named rulesets, errors reported; CodeQL not available (said so); manual C review by bug class; triage with reproduction (C1, M1) and variant search (C1 across bench2/4/5); severity after triage; rejected list. Not done: adversarial-review by a fresh agent (no tool).
- **untrusted-content-guard:** s.1 plan (read-only, no outbound, no keys); s.3 hidden-instruction scan of all committed third-party content; s.8 secret scan and dependency/supply-chain pinning (M3, M4, m1); s.9 sandbox review (jail, seals); s.10 OWASP LLM "improper output handling" / "excessive agency" lens (C1, M1, M2). s.2 reader-subagent quarantine not possible (no sub-agent tool); I read LLM transcripts only by regex, never as prose.
- **careful-live-changes:** made no live changes, used no keys, triggered nothing. Applied its checks to what the workflows touch: shared branches (no force-push anywhere; retry/rebase loops on router-runs/ci-runs), spend caps (`:free` gate is an allowlist-by-suffix; OpenRouter key holds credit, so M2 needs a gate-level cap), secrets kept in Actions secrets (none committed). One local side effect corrected: importing loop.py created an ignored `loop/__pycache__/` in the clone; I removed it (`git status --ignored` clean afterwards).
