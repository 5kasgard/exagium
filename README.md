# exagium

An exagium was the official Roman and Byzantine standard weight against which a gold coin was checked. Clipped coins failed and were taken out of circulation. The word is the root of *assay*, *essay* and *examine*.

This project is a place where agents build, test and optimise neural networks and other AI against physical hardware reality. Agents propose; a separate verifier and a meter decide; every claim is recorded.

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
| [`probe/`](probe/) machine-invariant probe: memory levels, bandwidth, compute ceilings, meter-tier detection, noise | done, 4 independent reviews; see [probe/README.md](probe/README.md) |
| cachegrind counter | next |
| the record (append-only claims and verdicts, outside anchor) | planned |
| first loop (small networks, mandatory baselines, separate verifier) | planned |
| real-hardware calibration (tier 3) | needs hardware |

## Test it yourself
[VERIFY.md](VERIFY.md) is a self-contained brief for an independent agent or person to test the probe's claims, ideally on different hardware.
