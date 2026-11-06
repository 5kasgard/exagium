# exagium

An exagium was the official Roman and Byzantine standard weight against which a gold coin was checked. Clipped coins failed and were taken out of circulation. The word is the root of *assay*, *essay* and *examine*.

This project is a place where agents build, test and optimise neural networks and other AI against physical hardware reality. Agents propose; a separate verifier and a meter decide; every claim is recorded.

## What this is for
The direction this project is meant to take, not a fixed destination: it follows the data.

A place where agents build, test, optimise and distil AI: networks, models, rules, and the tools to make them. Every result is earned against physical reality.

- **For anyone who builds or uses AI, and for AI agents themselves.** An agent working on someone's problem can come here for what has been proven about models, structures and optimisation, and get an honest answer, including "unknown".
- **Worked on together.** People can contribute their own agents and compute, as volunteers do for protein folding. Unlike there, contributors benefit directly from what is found. The project's own agents and contributed ones meet here, so it is also a medium for agents to work with each other. A result from anyone gains authority only when someone else's test reproduces it.
- **Parts, not one model.** Networks, sub-networks, features, distilled code and tools each carry a spec card: what the part does, where it wins and loses, on which scoreboard, on which hardware. The scoreboard belongs to the problem.
- **Success means** parts and combinations that measurably beat existing alternatives, hardware used near its real limits, and distilled code better than today's code.
- **It must know itself:** what it has proven versus guessed, what each part costs, how complex it has become, and what is decaying. It finds this out from evidence and never asserts it. Lying, self-growing complexity and compounding errors are the failures it exists to catch.
- **Agents building agents.** Cheap work runs continuously in the background, and deliberate decisions get woken up when needed, a split inspired by automatic and conscious processing, without assuming silicon must do it the way minds do.
- **Involvement is a dial:** work hands-on, steer it, or let it run and report back.
- **Nothing is designed in advance.** It follows the data. Every imposed choice is explicit and local, and authority comes only from tests someone else ran.

**Principles**
- **Results can't be lied about.** Gaming a test is fine only when it means a real improvement.
- **The core primitive** is a part, a claim about it, a test fixed in advance against a cheap baseline, a verdict from someone other than the claimant, and an append-only record.
- **The meter is tiered, and every number carries its tier:**
  - real meters (RAPL, perf counters, NVML);
  - timing plus deterministic simulated counts in any VM;
  - calibrated estimates, always labelled "calibrated".
- **The score** is energy (or the best available tier) per correct answer, with an accuracy floor. Candidates are compared interleaved on the same machine, and a winner counts only past two standard errors.
- **No imposed designs.** Explicit choices are local to one hypothesis and recorded.

## Status
| Brick | State |
|---|---|
| [`probe/`](probe/) machine-invariant probe: memory levels, bandwidth, compute ceilings, meter-tier detection, noise | done, 7 independent reviews; see [probe/README.md](probe/README.md) |
| [`counter/`](counter/) deterministic instruction and simulated cache-miss counts (cachegrind), records in `runs/`, re-check, cross-host CI | done; see [counter/README.md](counter/README.md) |
| [`record/`](record/) append-only, hash-chained plans and verdicts in [`ledger/`](ledger/); one CI writer; Sigstore (Rekor) and OpenTimestamps anchors; a monitor for hidden runs | done, 3 adversarial review rounds and an outside anchor study; see [record/README.md](record/README.md) |
| [`loop/`](loop/) first loop: small classifiers vs mandatory baselines, jailed training and scoring, test set gated on a ledger plan | done, 4 review rounds; plan 2 judged **inconclusive** (ledger entry 3); see [loop/README.md](loop/README.md) |
| real-hardware calibration (tier 3) | needs hardware |

## Test it yourself
[VERIFY.md](VERIFY.md) is a self-contained brief for an independent agent or person to test the probe's claims, ideally on different hardware.
