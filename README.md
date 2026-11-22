# exagium

An exagium was the official Roman and Byzantine standard weight against which a gold coin was checked. Clipped coins failed and were taken out of circulation. The word is the root of *assay*, *essay* and *examine*.

**What it is:** a lab where agents build, test, optimise and distil neural networks and other AI, and every result is weighed against real hardware.
- Agents propose.
- A separate verifier and a meter decide.
- Plans and verdicts are recorded in an append-only ledger anchored outside the repo, so a change made after anchoring is detectable. Other results so far point to their runs and files; they are not in the ledger yet.

**Where it is now (November 2026):** the measuring and recording machinery works and has been reviewed. Most results still hold for **one problem**, classifying 8×8 handwritten digits. The first test of whether they carry over to other kinds of task says they do not: on next-character text and on tabular census data, the order of the same model families changed, and the logic-gate network that was cheapest on digits was dearer than the small MLP on both ([`transfer/`](transfer/)). A general task contract now exists and held on its first four tasks, including one scored by a verifier with no answer key (SAT) ([`runner/`](runner/)). With stronger baselines on it (Nov 19), gradient-boosted trees lead on text and tabular data at a much higher cost per item, and lookup tables (WiSARD) were neither cheaper nor as accurate on any task, so the "structures that match the hardware are cheapest" hypothesis is weakened for every family of that kind tried so far. Free models now propose candidates; the first model-written candidate to beat every family here on digits (dev) appeared in the second batch. Next: the rest of the mixed suite, and 20 proposals to decide whether the one-call proposer stays (see **Next**).

## How it works
1. **A candidate** is a folder: `train.py` (deterministic numpy) writes `weights.bin`, and `infer.c` classifies images. Anyone, human or agent, can propose one.
2. **[`loop/`](loop/)** retrains it in a jail and checks that the weights come out byte for byte the same. It then scores it on dev data against mandatory baselines. The test set opens only for a plan registered in the ledger beforehand: the plan fixes the claim and its pass rule before anyone sees a test score.
3. **Two meters measure every candidate's cost:**
   - instructions and simulated cache misses, deterministic in any VM ([`counter/`](counter/));
   - native time from a trusted timer.

   [`loop/scale.py`](loop/README.md#cost-per-image-at-scale-scalepy-added-2026-11-18) adds the cost per image at large run sizes, because rankings flip with N.
4. **[`probe/`](probe/)** measures the machine itself: memory levels, bandwidth, compute ceilings, noise and which meter tiers it has.
5. **[`record/`](record/)** keeps plans and verdicts in an append-only, hash-chained [`ledger/`](ledger/). One CI job writes it, and Sigstore and OpenTimestamps anchor it. A verdict is given by someone other than the claimant.
6. **[`router/`](router/)** connects free model pools, measures which ones work, and runs pre-registered agent benches: sealed, timed coding tasks that test which free model and agent harness can do real work on this repo (the `router-runs` branch holds the raw data).

Everything runs on free compute: GitHub Actions runners (2 runner types; loop on 6 job slots, counter on 8), plus free model tiers.

## Progress
| Date (2026) | Step | Evidence |
|---|---|---|
| by Nov 5 | Machine probe, then deterministic counter, done and reviewed | `probe/README.md`, `counter/README.md` |
| Nov 5–6 | Ledger with outside anchors; first loop of small classifiers against baselines. Plan 1 ("the record holds up"): **pass**, judged by an outside verifier. Plan 2 (the loop): **inconclusive** | ledger entries 1–4 (written Nov 5–6) |
| Nov 11–16 | Agent benches on free models: Nemotron 3 Ultra fixed 3 of 3 real bugs (bench-2); a NIM-hosted agent passed 11 of 15 (bench-5) | `router-runs` branch, `router/bench*/` |
| Nov 15 | Logic-gate networks (LGN) with learned wiring: lgn_d, 0.973 dev accuracy (mlp16: 0.980), fewer instructions | `pilots/` on branch `pilot/lgn-20261115` |
| Nov 15–17 | Why mlp16 is slow: 6 causes ruled out by at least two independent runs (two other models' reports among them). The main one found: page faults on a single big input buffer. Reading by block: mlp16 −30%, LGN −46% | `explore/` on the pilot branch |
| Nov 17–18 | Whole-project audit, 52 items; 14 fixed on `main`, every decision ruled by data | `audit/2026-11-17/` on the pilot branch |
| Nov 18 | Per-image meter: at scale, lgn_blockread costs 31 ns/img against mlp16's 104; at the task's 956 images the order flips | `loop/results/scale-2026-11-18/` |
| Nov 18 | Do rankings transfer? Same 5 families on digits, next-character text and UCI Adult, rule fixed before running: **rankings differ**. 3 accuracy flips over 2 family pairs (one per pair survives 4× training), 4 instruction-count flips; the logic-gate network's cost lead holds on digits only | `transfer/results/2026-11-18/`; runs 44237908056, 44238804472 |
| Nov 18 | General task contract (records in and out, a checker per task): optdigits, text, UCI Adult and planted 3-SAT ran through one runner with no task code in it; WalkSAT solved 200 of 200 dev instances, checked without a key. Training reproduces on one host but not across hosts | `runner/results/2026-11-18/`; run 44253054973 |
| Nov 18 | The runner's jail hardened (Q10): every candidate run gets rlimits and the seccomp filter, including CI's bubblewrap mode and the counter's run, and a probe fails closed without them. Cross-host weights (Q55): OpenBLAS picked kernels per CPU; pinning one kernel type made text weights byte-identical across two CPU families | `runner/results/2026-11-18-q10-q55/`; runs 44257075574, 44258892424, 44260596412 |
| Nov 18 | Queue worker (P8 v1, with P7's research cards and P9's first measures): free models claim queue items, draft, check each other and land fixes without a chat; a daily digest writes the handoff. Release test: 0 double claims, 0 changes outside targets, injection canary held; **failed** on acceptance (6 of 8 outputs acceptable as is, below 4 of 5), so it is paused until a re-test | `worker/RESULTS-release-test-1.md`; `worker-runs` branch |
| Nov 18 | Whole-system indicators, the first part of an independent weekly audit: duplication 4.3% of Python function lines, 31 workflow lines that swallow failures, 3 README numbers without evidence. About one fifth of the week's code changes went into the parts that build and test AI; the rest went into measuring and recording | `audit/periodic/2026-11-18-baseline.json` |
| Nov 19 | Safety before more free work: candidate C now compiles inside the jail (a host `#include` fails, checked every run); paid pools (Cerebras, xAI, Gemini) are refused before any request unless the workflow allows them, within per-call and daily caps; every model call is logged with its usage (Cloudflare's neurons match the published rates exactly); workflows pinned by commit and given least privilege; the worker can start its own next run, since GitHub dropped its hourly schedule; the weekly indicators now run on CI and report to the owner as an issue. Each change reviewed by a free model of another family: 3 of 10 points confirmed and fixed, 7 rejected with evidence | runs 44392549096 (runner, all 4 tasks), 44393180102 (probe under caps), 44395236438 (writer), 44397281606 (usage log), 44397643836 (indicators); issue #1; `router-runs` reviews 44392240031, 44393174128 |
| Nov 19 | Stronger baselines on the three label tasks (P3 step 1, plan fixed first): gradient-boosted trees beat linear and mlp16 on text (0.392 vs 0.338, +10.4 SE) and adult (0.866 vs 0.859, +2.4 SE), not on digits, at 3 to 18 times the instructions; WiSARD lookup tables less accurate and dearer on all three: Q54 weakened. All new families reproduce, also across two machines | `runner/results/2026-11-19-p3-step1/`; run 44486290814 |
| Nov 19 | Free models propose candidates (P5, plan fixed first): a free model writes `train.py` and `infer.c`, scored in the jail with no key present. First 6 proposals: 4 valid; one (a two-layer MLP) reached 0.986 on digits dev, the best of over 100 proposals scored on the same dev split (gbdt 0.979, mlp16 0.977 in the P3 runner; mlp16 reads 0.980 in the Nov 15 pilot, a different trainer and split). Best-of-many on one split overstated it: on a held-out 20% of train it ties gbdt (+0.0017 ± 0.0046), and the top five lost 0.005 more than the baselines did; three of them match gbdt within 1 SE at a third or less of its instructions (`runner/results/2026-11-20-p5b/`). One reply was cut off by the model's output limit | `proposals` branch, LEADERBOARD.md; runs 44487103640, 44492370928 |
| Nov 19 | The last host-compile path for candidate code closed (Q56b): the per-image meter `scale.py` now compiles candidates inside the jail, and every loop CI run includes a scale smoke test. Jail-built and host-built machine code identical for all 5 candidates on `main`; a free outside review's 5 points all rejected with evidence (host #includes refused, probe runs every start). This unblocks the free proposer (P5) | runs 44474761141 (6 of 6 smoke runs, x86 and arm); `router-runs` review 44474822245 |
| Nov 19 | Queue worker release test 2 graded: 26 of 30 outputs acceptable (pass at 25), failed on running unattended. The four misses fixed and replayed on the live worker, scored by script: a script guard on guarantee words held where a reviewer prompt did not. Grading lesson: 3 of 5 free models failed known-answer controls, so exact checks go to scripts | `worker/RESULTS-release-test-2.md`; queue Q59 |
| Nov 20 | Items now move through one explicit state machine (`worker/transitions.py`): a pass ends, a check goes to review, a failed check escalates one tier (free worker → Claude session → owner). Before it, 8 of the 10 items waiting on the owner did not need the owner. Checked by an independent agent; its 11 findings fixed | queue Q60; main 66d1322 |
| Nov 20 | Do P5's best proposals hold on data the selection never saw? (P5b, plan fixed first) No proposal is above every family: the dev "best" ties gbdt on a held-out 20% of train, and the top five dropped 0.005 more than the baselines (winner's curse). Three reach gbdt's accuracy within 1 SE at a third or less of its instructions | `runner/results/2026-11-20-p5b/`; run 44628651975 |
| Nov 20 | Does the proposal's extra depth help beyond digits? (P5a, rule fixed first) No: the two-layer MLP beats mlp16 on digits (also on held-out data) but loses on UCI Adult (−0.006) and on next-character text (−0.050, below linear). A digits-specific gain, as P1 found for families | `runner/results/2026-11-20-p5a/`; run 44626963486 |
| Nov 20 | Are fresh SWE-rebench tasks less memorised than SWE-bench Verified? (P28, rule fixed first) Yes: from the issue text alone, the free model named the file to change for 72% of 40 Verified tasks and 35% of 40 fresh ones (gap 37.5 points, 95% interval 17 to 58, rule 15). The fresh set becomes P28's held-out source; it is less recalled, not proven unseen | `router/bench6/illusion/results/2026-11-19/`; run 44632998540 |
| Nov 20 | Can the held-out tasks' own tests grade an agent? (P28 pilot step 1, rule fixed first, no model call) On 8 of 12 seeded fresh SWE-rebench tasks, yes: the unfixed code fails, the gold patch passes every listed test (rule: at least 8). The other 4: malformed test ids in the dataset (2), a flaky test, an environment failure | `router/bench6/pilot/results/2026-11-20-env.md`; run 44706105305 |
| Nov 21 | Can the held-out tasks grade an agent's attempt, not only the gold patch? (P28 pilot step 2, rule fixed first, no model call) On the 8 usable tasks, yes: the gold patch read resolved (also with the test files edited, which the grader undoes), and no patch or a conftest.py that marks every test passed read unresolved, 32 of 32. Not shown: anything about an agent yet; the first model attempts wait for the pilot rule in PLAN.md to be merged | `router/bench6/pilot/results/2026-11-21-attempt-grader.md`; runs 44959195886, 44959198446, 44959200539, 44959202923 |
| Nov 22 | Which cost ranks the proposals leaderboard (audit F054, decided by data): whole-run instructions per item, startup included, not the marginal count. On all 165 valid proposals with both counts the front is the same under either and the order nearly so (Spearman 0.9996 or more per task), and the marginal count can be gamed by moving work into the empty run. Results scored before the whole-run count existed compete at their marginal count plus the task's measured startup floor; points beaten only by them are marked provisional until those results are rescored (no model calls) | `runner/proposals/judge.py`; `proposals` branch LEADERBOARD.md; rescore runs 44957791736, 44957794189, 44957796895, 44958589795, 44959049219 |

## Best result so far
On one task (optdigits), dev data only (the test set is spent), one host, one session: direction, not a claim. **It does not carry over:** with one generic program per family, the logic-gate network costs 0.47× mlp16's instructions per item on digits, 2.1× on next-character text and 1.3× on UCI Adult, and it is last or near last on accuracy on both (`transfer/results/2026-11-18/`).

| | dev accuracy | instructions per image | large-N cost per image |
|---|---|---|---|
| mlp16 (the best small MLP) | 0.980 | 1× | 104 ns (mlp16_chunked: 83) |
| lgn_d (logic-gate network) | 0.973 | 0.45× | 55 ns (read by block: 31) |

## Who does what
Each kind of work goes to the cheapest worker that measurably does it well. The table changes as the data does.

| Work | Worker | Evidence |
|---|---|---|
| Measuring, scoring, training, recording | GitHub Actions CPU (free; 2 runner types, loop on 6 job slots and counter on 8) | loop 23 of 24 runs ok; counter 21 of 21 |
| Agent coding (fixes, candidates) | Nemotron 3 Ultra (OpenRouter free, about 1,000 requests/day); NIM-hosted agents | bench-2 3/3; bench-5 11/15 |
| Outside review of code and claims | gpt-oss-120b (Cloudflare free) | found a real bug in `scale.py` on Nov 18; 3 of 8 points applied in bench-5 |
| Short checks | Groq gpt-oss-120b (free, 8,000 tokens/min) | probe answers exactly |
| Queue work: doc fixes and literature cards | drafter gpt-oss-120b (Groq, Cloudflare fallback), reader gpt-oss-120b (Cloudflare), checker Nemotron 3 Ultra (NIM); script guards decide what lands | release test 2: 26/30 acceptable |
| Candidate programs (P5) | free NIM model, one call per proposal; scored in the jail with no key | 66 of 104 valid |
| Grading exact outcomes | scripts, not models | Nov 19: 3 of 5 free models failed known-answer controls when grading |
| High-volume bursts | Cerebras, xAI (paid credit) | behind spend caps (Nov 19): refused unless a workflow allows them; 50 and 30 calls a day |
| Design, integration, final adversarial checks | Claude (scarce; used least) | — |
| Liveness probes only | Llama-4-Scout | 0 of 8 review points usable |

Open work is in one queue: `audit/2026-11-17/QUEUE.md` on the pilot branch. [AGENTS.md](AGENTS.md) says how any agent working here closes a session.

## Next
"AI" here means more than neural networks: in-context learners (PFNs), trees and boosting, probabilistic models, theorem provers and SAT solvers, planners and search, program synthesis, agents, and hybrids of these. The plan, in order, each step a test of whether the next is worth building:
1. ~~**Do rankings transfer?**~~ Done 2026-11-18: no. The order of families changed across digits, text and tabular data ([`transfer/`](transfer/)). Two lessons carried forward: every task card needs a tuning budget on a split inside training, or capacity effects pass for structural ones; and a model's size rule that grows with the number of classes needs its own test.
2. ~~**A general task contract:**~~ Done 2026-11-18: records in, records out, and a checker program per task; it held on four tasks, one of them checked with no answer key (SAT) ([`runner/`](runner/)).
3. **A mixed first suite on the contract (started Nov 19):** step 1 added outside-strength baselines (boosted trees, nearest neighbours, a lookup-table network) and a tuning split to the three label tasks ([`runner/results/2026-11-19-p3-step1/`](runner/results/2026-11-19-p3-step1/)). Still to add: vision, audio, language, tabular (with a PFN and boosted trees), time series, control, verifier-scored problems (harder SAT, simple formal lemmas), and learning a rule and emitting it as code. It must settle what the first four tasks didn't: outputs larger than one bound (proofs, programs), slow checkers and a tuning budget in every card. (Weights that differed by host were settled by Q55.)
4. **Free workers (running):** a queue worker takes doc fixes and literature cards ([`worker/`](worker/)); release test 2 graded (26/30 acceptable; failed on running unattended, since fixed: idle re-polls, budget-aware claims, outcome pipelines). Next: P28, held-out outside tasks for agent tests: no SWE-rebench task is newer than the free model's training data, but a recall probe found the fresh set far less memorised than SWE-bench Verified (file named from the issue alone: 35% vs 72%, [`router/bench6/illusion/`](router/bench6/illusion/)), so it is used. Its graders were checked first, with no model call: on 8 of 12 seeded fresh tasks the gold patch turns the task's own tests green and the unfixed code does not; 2 of the 4 others have malformed test ids in the dataset itself ([`router/bench6/pilot/results/2026-11-20-env.md`](router/bench6/pilot/results/2026-11-20-env.md)). Then the feasibility pilot on those 8, and the Claude session draining its review and escalation tiers twice a day.
5. **Compartments and connections from data:** artifacts are grouped by which items they win and fail on, not by human labels, and the groups are recomputed as the collection grows. Shared failures across unrelated families, items that only one family solves, and artifacts that compute the same function are found the same way. Step 1 already keeps every item's outcome for this.
6. **Later:** an interface for working on it without a chat, and outside contributors' agents and compute.

## What this is for
The direction this project is meant to take, not a fixed destination: it follows the data.
- **For anyone who builds or uses AI, and for AI agents themselves.** An agent working on someone's problem can come here for what has been proven about models, structures and optimisation, and get an honest answer, including "unknown".
- **Worked on together.** People can contribute their own agents and compute, as volunteers do for protein folding, and contributors benefit directly from what is found. A result from anyone gains authority only when someone else's test reproduces it.
- **Parts, not one model.** Networks, sub-networks, features, distilled code and tools each carry a spec card: what the part does, where it wins and loses, on which scoreboard, on which hardware.
- **Success means** parts and combinations that measurably beat existing alternatives, hardware used near its real limits, and distilled code better than today's code.
- **It must know itself:** what it has proven versus guessed, what each part costs, and what is decaying. Lying, self-growing complexity and compounding errors are the failures it exists to catch.
- **Involvement is a dial:** work hands-on, steer it, or let it run and report back.

## Principles
- **No result may be a lie.** Each result states what it can't prove. The ledger catches a change made after anchoring; it can't show that a result was genuine to begin with. Only independent re-measurement can (`probe/README.md`, "Digests").
- **The core primitive:** a part, a claim about it, a test fixed in advance against a cheap baseline, a verdict from someone other than the claimant, and an append-only record.
- **The meter is tiered, and every number carries its tier:** real meters (RAPL, perf counters, NVML); timing plus deterministic simulated counts in any VM; calibrated estimates, always labelled.
- **No imposed designs.** Explicit choices are local to one hypothesis and recorded.

## Status of each part
| Part | State |
|---|---|
| [`probe/`](probe/) machine probe | done, 7 independent reviews |
| [`counter/`](counter/) deterministic instruction and cache counts, cross-host CI | done |
| [`record/`](record/) ledger, anchors, monitor; dependencies pinned by hash | done; plan 1 **pass** (entry 4) |
| [`loop/`](loop/) jailed training and scoring, two meters, per-image meter | done; plan 2 **inconclusive** (entry 3) |
| [`transfer/`](transfer/) same families across three kinds of task | step 1 done: rankings differ |
| [`runner/`](runner/) + [`tasks/`](tasks/) the general task contract and runner | held on 4 tasks (3 label, SAT); 3 stronger families and a tuning split added (P3 step 1) |
| [`runner/proposals/`](runner/proposals/) free proposer | running 24/7: 104 proposals, 66 valid (0.63, Wilson 0.54–0.72), 18 beat label-linear, 1.6 free calls per valid; first 20 → KEEP (11 valid) (proposals branch `LEADERBOARD.md`, 2026-11-19 21:00Z) |
| [`worker/`](worker/) free queue worker (doc fixes, literature cards) | running 24/7 under an explicit item state machine (`worker/transitions.py`): passes end, checks go to review, failed checks escalate one tier at a time; release test 2: acceptance 26/30 pass, unattended criterion failed (`worker/RESULTS-release-test-2.md`) |
| [`router/`](router/) free pools and agent benches | working: required pools probed daily; benches 2–5 complete |
| real-hardware calibration (tier 3) | needs hardware |

## Test it yourself
[VERIFY.md](VERIFY.md) is a self-contained brief for an independent agent or person to test the probe's claims, ideally on different hardware.
