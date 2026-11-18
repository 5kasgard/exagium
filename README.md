# exagium

An exagium was the official Roman and Byzantine standard weight against which a gold coin was checked. Clipped coins failed and were taken out of circulation. The word is the root of *assay*, *essay* and *examine*.

**What it is:** a lab where agents build, test, optimise and distil neural networks and other AI, and every result is weighed against real hardware.
- Agents propose.
- A separate verifier and a meter decide.
- Every claim is recorded in an append-only ledger anchored outside the repo, so a change made after anchoring is detectable.

**Where it is now (November 2026):** the measuring and recording machinery works and has been reviewed, but it has only been pointed at **one problem**: classifying 8×8 handwritten digits. Every result below holds for that task only. The first architecture found by following the data, a logic-gate network, is about 1.9–3.3× cheaper per image than the best small MLP at scale, at 0.7 percentage points lower dev accuracy. The next steps widen it to AI in general and then hand the routine work to free models (see **Next**).

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

Everything runs on free compute: GitHub Actions runners (loop on 6 runner types, counter on 8), plus free model tiers.

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

## Best result so far
On one task (optdigits), dev data only (the test set is spent), one host, one session: direction, not a claim. Whether it holds on any other kind of task is untested.

| | dev accuracy | instructions per image | large-N cost per image |
|---|---|---|---|
| mlp16 (the best small MLP) | 0.980 | 1× | 104 ns (mlp16_chunked: 83) |
| lgn_d (logic-gate network) | 0.973 | 0.45× | 55 ns (read by block: 31) |

## Who does what
Each kind of work goes to the cheapest worker that measurably does it well. The table changes as the data does.

| Work | Worker | Evidence |
|---|---|---|
| Measuring, scoring, training, recording | GitHub Actions CPU (free; loop on 6 runner types, counter on 8) | loop 23 of 24 runs ok; counter 21 of 21 |
| Agent coding (fixes, candidates) | Nemotron 3 Ultra (OpenRouter free, about 1,000 requests/day); NIM-hosted agents | bench-2 3/3; bench-5 11/15 |
| Outside review of code and claims | gpt-oss-120b (Cloudflare free) | found a real bug in `scale.py` on Nov 18; 3 of 8 points applied in bench-5 |
| Short checks | Groq gpt-oss-120b (free, 8,000 tokens/min) | probe answers exactly |
| High-volume bursts | Cerebras, xAI (paid credit) | only behind a spend cap, not yet built |
| Design, integration, final adversarial checks | Claude (scarce; used least) | — |
| Liveness probes only | Llama-4-Scout | 0 of 8 review points usable |

Open work is in one queue: `audit/2026-11-17/QUEUE.md` on the pilot branch. [AGENTS.md](AGENTS.md) says how any agent working here closes a session.

## Next
"AI" here means more than neural networks: in-context learners (PFNs), trees and boosting, probabilistic models, theorem provers and SAT solvers, planners and search, program synthesis, agents, and hybrids of these. The plan, in order, each step a test of whether the next is worth building:
1. **Do rankings transfer?** Add two tasks as unlike digits as possible: next-character text and a tabular set. Run the same families on all three, keeping every item's outcome. The rule for "the ranking changed" is fixed before running.
2. **A general task contract:** bytes in, bytes out, and a checker program. A label set is one checker; a proof kernel, a SAT check or held-out test cases are others. Then a mixed first suite: vision, audio, language, tabular, time series, control, verifier-scored problems (SAT, simple formal lemmas), and learning a rule and emitting it as code.
3. **Free models propose candidates around the clock** across the suite. The model's key and the generated code never share a job, and the results build up on a public leaderboard.
4. **Compartments and connections from data:** artifacts are grouped by which items they win and fail on, not by human labels, and the groups are recomputed as the collection grows. Shared failures across unrelated families, items that only one family solves, and artifacts that compute the same function are found the same way.
5. **Later:** an interface for working on it without a chat, and outside contributors' agents and compute.

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
| [`router/`](router/) free pools and agent benches | working: required pools probed daily; benches 2–5 complete |
| real-hardware calibration (tier 3) | needs hardware |

## Test it yourself
[VERIFY.md](VERIFY.md) is a self-contained brief for an independent agent or person to test the probe's claims, ideally on different hardware.
