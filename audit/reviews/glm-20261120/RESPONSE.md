# Response to GLM's review of 83f06fd (2026-11-20)

Files here are the review as delivered (manifest.json names the model as 'Z.ai GLM-4.6'). Every fix below was reproduced
first with a test that fails on the old code, or with a known-bad control caught end to end (runner), before the change.

## Fixed

| id | severity | commit | note |
|---|---|---|---|
| F011 | high | 828266e |  |
| F051 | high | b7aad7f |  |
| F052 | high | b7aad7f |  |
| F016 | high | 583894f |  |
| F036 | high | 583894f |  |
| F024 | high | 583894f |  |
| F017 | high | 583894f |  |
| F053 | high | b7aad7f | RATIO_BAND reported, not enforced: honest trivial baselines reach 20-46x in past runner results (runner/results/2026-11-21-glm-guards/README.md). |
| F003 | medium | 57ffd2a |  |
| F005 | medium | 57ffd2a |  |
| F002 | medium | 57ffd2a |  |
| F004 | medium | 57ffd2a |  |
| F007 | medium | 57ffd2a |  |
| F008 | medium | 57ffd2a |  |
| F037 | medium | 4059641 |  |
| F021 | medium | 4059641 |  |
| F042 | medium | 4059641 |  |
| F043 | medium | 4059641 |  |
| F039 | medium | 4059641 |  |
| F010 | medium | 57ffd2a |  |
| F055 | medium | 0563497 |  |
| F058 | medium | 0563497 |  |
| F018 | medium | ed86d19 |  |
| F033 | medium | ed86d19 | fixed for the future; the one bake run so far (44720253829) is unaffected: its calls logs show each job really called its named reader. |
| F023 | medium | ed86d19 |  |
| F034 | medium | ed86d19 |  |
| F025 | medium | ed86d19 |  |
| F061 | medium | 2e8711d |  |
| F062 | medium | 2e8711d |  |
| F056 | low | b7aad7f |  |
| F057 | low | b7aad7f |  |
| F059 | low | b7aad7f |  |
| F063 | low | 2e8711d |  |

## Not changed, with reason

| id | severity | why |
|---|---|---|
| F019 | medium | already fixed after the reviewed commit: split.regate (e249ac8). |
| F012 | medium | not changed: bench-1 is a registered, disabled bench; its arm C is part of the registration. Re-enabling it needs this first. |
| F054 | medium | owner decision: switching the leaderboard's headline from marginal to startup-included measures changes what has counted for 245 proposals; GLM's own risk note says re-score first. Options in the reply. |

## Open (75), in GLM's order

| order | id | severity | file | claim |
|---|---|---|---|---|
| 24 | F068 | medium | runner/test_runner.py | run.py's gate logic has no unit tests at all - the file is read only as text for a task-name scan, while every branch of cmd_evaluate/cmd_re |
| 25 | F076 | medium | record/test_record.py | cmd_monitor - the re-anchoring/equivocation and orphan-laundering defence, and the only check that catches a rewritten-and-re-anchored histo |
| 26 | F073 | medium | record/record.py | check-anchors' anchor times (and the monitor's logIndex keys) are read from the bundle's unsigned JSON wrapper and never cross-checked again |
| 27 | F077 | medium | counter/count.py | The repo's committed counter evidence (runs/count-*.json) uses schema exagium-count/2 while count.py now writes and checks exagium-count/3,  |
| 28 | F075 | medium | record/record.py | cmd_ingest's append is not crash-atomic: the entry file is written directly (no temp+rename/fsync) and the inbox proposal is removed only af |
| 29 | F074 | medium | record/record.py | check_ots reports a lagging Bitcoin explorer as proof of forgery: a 404 on block-height with att.height above the explorer's stale tip raise |
| 35 | F015 | medium | worker/test_worker.py | The Gemini-20261120b response claims all five fixes were 'each reproduced first with a test', but the GeminiReview20261120 class contains on |
| 39 | F070 | medium | .github/workflows/transfer.yml | The registered-commit comparison the transfer plan relies on is advisory only - transfer.yml prints 'AMENDED since registration' and proceed |
| 40 | F091 | medium | loop/loop.py | Every evaluate/scale run pins all timed runs to the lowest-numbered CPU (min(sched_getaffinity)), and nothing prevents two concurrent loop/s |
| 41 | F092 | medium | loop/loop.py | In confine mode a per-run timeout kills only the harness's direct child (confine execv's the timer, same pid), orphaning the timed candidate |
| 42 | F094 | medium | loop/test_loop.py | The suite selects the jail at import time via MODE = jail_mode(), so on hosts where no jail is available (kernel 5.10: no unprivileged-usern |
| 43 | F083 | medium | facts/check.py | The sha256_url probe deliberately falls back to unverified TLS when certificate verification fails, so the content-hash fact can be answered |
| 44 | F084 | medium | counter/count.py | None of count.py's jail-integration and relocation modes - --single-process, --confine (with --confine-ro/-ls), --stdout-record, and --check |
| 45 | F095 | medium | loop/test_scale.py | scale.py's only tests mock the jail, the timer, the compiler and the host, so the tool's real measure() path - jailed compiles, jailed timed |
| 46 | F100 | medium | worker/RESULTS-release-test-2.md | Grading by free models is unreliable: 3 of 5 failed known-answer controls in the release-test grading round, so model-graded acceptance cann |
| 47 | F101 | medium | runner/proposals/judge.py | beats_linear gates solely on paired dev accuracy without checking ir_per_item or ns_per_item - the cost-blind validity gate the repo's own G |
| 48 | F102 | medium | worker/queue_ops.py | Worker queue coordination is hardcoded to a dated audit path and pilot branch - and the same dated path is also read by whole-audit.yml and  |
| 49 | F103 | medium | runner/jail.py | In confine mode, valgrind runs under confine without limits.c rlimits, and /w host disk is bounded only by fsize * nofile (the repo's own Ge |
| 50 | F104 | medium | loop/seccomp_filter.h | exagium_seccomp uses a syscall denylist ending in SECCOMP_RET_ALLOW rather than an allowlist (the repo's own Gemini F005, open). |
| 51 | F045 | medium | router/decisions_route.py | The decisions_* scripts write raw provider error bodies (e.read() decoded, truncated but unscrubbed by any redaction) into their committed o |
| 52 | F041 | medium | router/engines.py | The engines-layer paid guard (PAID table + paid_guard: allowlist, per-call caps, daily cap, usage log) covers only cerebras/xai/gemini; the  |
| 53 | F049 | medium | router/test_pick.py | pick.py has only two tests, covering neither pending-reservation headroom (F037), the missing-ledger-file silent-{} path, the rolling-neuron |
| 55 | F038 | medium | router/engines.py | engines.call() silently supports only nvidia/groq among the compat pools (and only when system is None); every other compat pool falls throu |
| 56 | F014 | medium | .github/workflows/transfer.yml | transfer/test_fetch.py (8 offline TLS-fallback tests) and router/bench6/pilot/test_envcheck.py are never invoked by any workflow: transfer.y |
| 57 | F040 | medium | router/pick.py | pick.py's per-call Cloudflare neuron projection (600, 'max seen 438') contradicts worker/ledger.py's default projection of the same quantity |
| 58 | F022 | medium | worker/generate.py | today_count() counts every item with origin=='generated' created today - including split.py's K sub-items and diagnose.py's D items - so tho |
| 59 | F020 | medium | worker/generate.py | In run()'s write loop the daily-cap break can fire before 'made' is assigned, so the later move_all reference hits an UnboundLocalError on a |
| 60 | F046 | medium | router/review.py | review.py itself reads any path given as argv[1] with no allowlist, traversal or symlink check (line 11: 'prompt, out = open(sys.argv[1]).re |
| 61 | F006 | low | .github/workflows/worker.yml | The idle job's split-step acceptance count writes a malformed two-line output value whenever the splitter accepts nothing: grep -c prints '0 |
| 62 | F013 | low | .github/workflows/anchor-test.yml | anchor-test.yml's collect job pushes to the shared ci-runs branch in a single attempt with no fetch/rebase/retry loop, unlike every other ci |
| 63 | F029 | low | worker/digest.py | The budget-used section reads only the two endpoint day files {(now), (now-hours)} - correct only for the default 24h window; any other --ho |
| 64 | F030 | low | worker/digest.py | The digest parses queue moves and lease expiries from commit subjects (git log --format=%h %s plus regex over the message text), so any subj |
| 65 | F026 | low | worker/tier.py | The [:400] truncation binds only to the last f-string segment, not the whole note, so the merge note stored in the queue's run record is unt |
| 66 | F027 | low | worker/tier.py | redo() replaces the whole research dict with only question+queries, dropping any other keys (e.g. 'kinds' - an item's source-kind restrictio |
| 67 | F028 | low | worker/tier.py | A failure of the research redo/escalate controls aborts the whole tier run before drain_code(), so code PRs waiting for the merge gate stall |
| 68 | F031 | low | worker/research_job.py | URL_OK requires the https:// scheme, but the arXiv Atom API returns http:// entry ids (and github.com/huggingface.co hosts are not allowlist |
| 69 | F032 | low | worker/research_job.py | The script check 'every number in a claim must occur in its quote' uses substring containment, not token matching, so '15' passes on a quote |
| 70 | F035 | low | worker/README.md | The worker spec card predates 2026-11-20 and contradicts the code: 'Paid pools can't be reserved' while worker.yml reserves openrouter_paid: |
| 71 | F047 | low | router/analysis/design_variance.py | A result with calls == 0 is silently treated as missing (excluded from the calls measure) instead of being counted as zero, biasing the per- |
| 72 | F001 | low | router/decisions_probe.py | decisions_probe.py's batch mode reads only the hard-coded 'noul' key for each answer while per-item mode uses the shape-tolerant p_yes() wal |
| 73 | F048 | low | router/probe.py | The probe docstring's cost statement ('1 OpenRouter request ... 1 Gemini request plus a model list') predates probe_compat() and omits the u |
| 74 | F050 | low | worker/research_job.py | Worker phase scripts mutate the router's process-global engines.TIMEOUT (research_job.py:415 and doc_job.py:185 and code_job.py:115 set 600; |
| 78 | F060 | low | transfer/decide.py | decide.py's docstring also promises exit code 3 on failure, but only 0/1 are ever produced. |
| 80 | F064 | low | runner/CONTRACT.md | The contract's failure criterion points to 'the equivalence check below', but no equivalence check is described anywhere below in the docume |
| 81 | F065 | low | runner/CONTRACT.md | The contract says the floor is a baseline 'whose dev score a candidate must beat to count', but run.py never gates anything on it - it recor |
| 82 | F066 | low | runner/README.md | The Parts section names only 5 of the 9 candidate families that exist in runner/candidates/ (label-gbdt, label-knn, label-wisard, label-mlp2 |
| 83 | F067 | low | runner/proposals/PLAN.md | The plan says extract 'takes exactly the three fenced blocks', but extract.py takes the first block of each kind and silently ignores any fu |
| 84 | F069 | low | transfer/meter.py | The measured subprocesses have no timeout, so one hanging C program stalls the meter (and the CI job) indefinitely. |
| 85 | F071 | low | runner/candidates/sat-walksat/infer.c | Literal variable indices are never checked against the instance's n, so an instance naming a variable >= n reads uninitialized bytes of a[]. |
| 86 | F072 | low | runner/candidates/label-knn/infer.c | infer rejects k > n ('bad weights size', exit 3) while train.py's predict tolerates k > n, so a training set smaller than k breaks only at i |
| 87 | F078 | low | audit/periodic/indicators.py | cannot_fail counts comment lines as failure-swallowing workflow lines, inflating the indicator. |
| 88 | F079 | low | audit/periodic/indicators.py | The duplication share counts only duplicate groups spanning more than one distinct file, so identical functions duplicated within a single f |
| 89 | F080 | low | record/README.md | The record spec card's self-description numbers are stale: '22 tests' (26 exist), 'record.py is 630 lines, the tests 235, record.yml 127' (6 |
| 90 | F081 | low | audit/periodic/README.md | The baseline summary misstates the duplication denominator (7,225 vs the baseline file's own 7,228 function lines). |
| 91 | F082 | low | facts/check.py | The due-date gate trusts the status file blindly: a status 'at' in the future (clock skew or a corrupted write) makes now-at negative so the |
| 92 | F085 | low | facts/test_check.py | Every test replaces check.probe with a mock, so none of the real probe kinds (sha256_url and its TLS fallback, the openrouter/nim/groq/anthr |
| 93 | F086 | low | record/experiments/anchors/ots_check.py | ots_check.py accepts a match from any single explorer (no cross-explorer agreement, unlike record.py's check_ots) and reports pending proofs |
| 94 | F093 | low | loop/loop.py | check_confine() creates its confine build directory with tempfile.mkdtemp and never removes it, leaking an exagium-confine-* directory (with |
| 95 | F096 | low | loop/README.md | The README says test_loop has 20 tests but the suite now runs 23 (the RequirePlan class's 3 tests were added by audit fix 05328a44 before th |
| 96 | F097 | low | loop/README.md | The plan-2 native-time claim '17-31% faster than both baselines on every host' does not match the committed scoreboards, which give 14.5%-33 |
| 97 | F098 | low | loop/README.md | The Files section describes verify/ as 'empty now', but the directory does not exist in the working tree at all (the workflow tolerates its  |
| 98 | F044 | low | router/engines.py | ok-semantics are inconsistent across engines - compat/openrouter/cloudflare/gemini use 'ok': text is not None (an empty-string reply counts  |
| 99 | F099 | low | probe/README.md | The README prices the --quick smoke run at '~10 s' while probe.py's own usage docstring says '~30 s' - the two disagree (VERIFY.md repeats t |
| 100 | F087 | low | probe/probe.py | --reanalyse decides 'same host' (meter re-detection, exit 0 vs 2) by comparing only a 12-hex-char hash of the hostname, so a different machi |
| 101 | F105 | low | worker/grade_canary3.py | Canary 3 grading hardcodes fixed line numbers and exact string equality for E2 (the repo's own Gemini F006, open). |
| 102 | F088 | low | record/record.py | verify's overdue boundary uses the local calendar date rather than UTC, so the overdue determination shifts with the verifier's timezone (a  |
| 103 | F106 | low | README.md | README's proposer numbers lag the live proposals branch: '104 proposals, 66 valid ... 18 beat label-linear' (snapshot 2026-11-19 21:00Z) aga |
| 104 | F089 | low | record/record.py | Both check_sigstore and cmd_monitor read only tlogEntries[0] from a bundle, so a bundle with multiple transparency-log entries (sharded or s |
| 105 | F107 | low | README.md | README says 'loop on 6 runner types, counter on 8', but loop.yml and counter.yml each define exactly 2 runner labels (ubuntu-24.04, ubuntu-2 |
| 106 | F090 | low | tasks/sat/check.py | A literal byte of 0 in a SAT input decodes as variable -1 and silently checks the last variable (abs(0)-1 = -1 indexes o[-1]) instead of bei |
| 107 | F108 | low | worker/doc_job.py | Model replies are parsed by a greedy JSON regex (first '{' to last '}') across the model-calling code - one fenced example or stray braces i |
| 108 | F111 | low | worker/research_job.py | When the decision model succeeds, model_fail is cleared wholesale - erasing checker-chunk failures from the run's record and leaving shadow_ |
| 109 | F109 | low | .github/workflows/keeper.yml | The scheduler-dependency workaround machinery (self-chaining with a dead cap, keeper jobs sleeping up to 5h40m on runners, 30-minute idle re |
| 110 | F110 | low | runner/results/2026-11-18/README.md | The SAT task is explicitly easy (planted instances at 4.2 clauses per variable) and the test-split gate has never been exercised end-to-end  |
| 111 | F009 | low | .github/workflows/record-e2e.yml | record-e2e.yml's header comments (copied from record.yml) claim daily-schedule behaviours this workflow does not have: it has no schedule tr |
