# exagium independent verification — Z (GLM), 2026-11-06

Against main `d801cea` (full clone), VERIFY.md as of that commit including the
new "Loop claims". Plan 1 judged (due 2026-11-20).

## 1. Hardware and software

Intel Xeon (Granite Rapids 173) 2 vCPU KVM VM **inside a kata container**,
4 GB RAM, kernel 5.10.134; Debian 13, gcc 14.2, Python 3.12.16, numpy 2.1.3,
valgrind 3.24.0 and bubblewrap 0.11.0 (built from source into /usr/local),
sigstore 4.5.1 + opentimestamps-client 0.8.0 (venv). No RAPL, NVML or perf.
Details: `machine_and_tools.txt`.

## 2. Verdict table

| Claim | Verdict |
|---|---|
| 1 compute ceilings are real FMA throughput | CONFIRMED |
| 2 bandwidth (L1 load-bound; DRAM real) | CONFIRMED |
| 3 latency curve | CONFIRMED |
| 4 memory levels + adversarial fit | CONFIRMED |
| 5 meter tiers | CONFIRMED |
| 6 --reanalyse | CONFIRMED |
| 7 labels | CONFIRMED |
| 8 reproducibility (3 seeds) | CONFIRMED |
| 9 portability | probe/counter/record fine; the loop's jail cannot run here (below) |
| 10-14 counter | CONFIRMED (2 header-gated tests replicated by hand) |
| record 1-4 | CONFIRMED, one weakness (below) |
| loop 1 re-measure plan 2 | CONFIRMED (unjailed, disclosed) |
| loop 2 read train.py / infer.c | CONFIRMED clean |
| loop 3 attack the scoring | NOT FULLY ATTEMPTED (jail impossible here) |
| **plan 1 (record can't be made to lie undetected)** | **PASS** |

## 3. Evidence per claim

**Probe.** 40/40 tests pass. `full.json`: L1 1.40 ns, bracket 44-64 KiB (L1D
48 KiB); L2 4.49 ns, bracket 1.0-1.4 MiB, upper end `null` (L2 2 MiB; sweep
can't reach 2x the 504 MiB L3 — honestly warned). objdump: `flops_scalar` is scalar xmm FMAs, no vectors; vec kernels full-width
zmm FMAs; FLOP formula right. 72-77% of the 204.8 GFLOP/s peak at 3.2 GHz
measured under load. L1 flat from 4->16 accumulators (load-bound); my
independent scalar-read program: 14.5 GB/s DRAM-1t vs the probe's 15.7-18.5.
Tiers flip when valgrind is hidden. Three seeds: same brackets, compute spread
0.2-1.7%, DRAM 15-24% (shared VM). My own adversarial generator (gamma noise,
AR(1), off-grid capacities, ramps, spikes, soft mixtures, per-rep capacity
jitter; 200 seeds/case; `scripts/adv_levels2.py`): **0% phantom levels, <=0.5%
false boundaries, <=4.5% wrong brackets**; misses are the safe failure —
matching the README's table including its stated limits.

**Counter.** 23 tests OK; the 2 skipped for missing valgrind headers I
replicated by hand: CACHEGRIND_STOP_INSTRUMENTATION work goes uncounted
(113k Ir), is flagged, main exits 1, --check exits 2, and a digest-fixed
forgery of the flag still fails the re-count. My build: Ir(1000)=35,777,865, Ir(0)=244,738 (a different gcc build; reps
identical). --check: genuine 0; raw edit, digest-fixed forgery, three-layer
forgery all 1; foreign software 2. sweep.py: the D1=49152 config (my measured
L1) is the one whose miss-jump aligns with the latency jump.

**Record.** verify/check-anchors/monitor all exit 0 on the untouched ledger
(monitor 2026-11-06 ~10:35 UTC: 6 refs, 3 Rekor entries, none unexplained;
verify and check-anchors re-confirmed exit 0 at 11:02 UTC —
`artifacts/clean-checks-2026-11-06.txt`; monitor then hit the shared-IP rate
limit again, exit 2, the correct refusal).
Attacks: hand-written entry 4 -> exit 1 "never attested"; fully re-referenced
rewrite of entry 2 -> exit 1 (OTS bytes, sigstore subject, coverage gap);
orphan laundering with the real statement and hidden entries -> exit 1
"published and later removed"; corrupted signature with dead network -> exit 1.
Anchor verified without record.py: `sigstore verify github` OK; entry sha256
and chain recomputed in Node.js (JCS matches byte-for-byte); `ots verify`
resolves to bitcoin tx 3a1beaae..., confirmed in block 974799.

**Loop.** The jail cannot run here: the container's /proc has locked read-only
submounts, so kernel 5.10's `mount_too_revealing` refuses every unprivileged
proc mount (`artifacts/jail-analysis.txt`); test_loop fails closed (11F+2E, all
"bwrap cannot make a jail"). Plan 2's decision inputs were re-measured WITHOUT
the jail, code read first (mlp32/64 differ from mlp16 only in H; none of the
five carries test data, sniffs the environment, detects valgrind, or uses
forks/threads/dlopen): all five retrains **byte-identical** (numpy 2.1.3 vs
CI's 2.5.3); predictions_sha256 identical for all five; accuracies
0.9505/0.9599/0.9594/0.9655/0.9610; Ir/img 4,259/3,615/723/1,024/1,675
(linear +10% from my gcc build, others within 0.5% of CI). The decision rule
on my numbers: part A PASS, part B inconclusive (z 1.62/0.33 vs thermo) —
**matches ledger entry 3**. All 21 frozen digests still match `freeze`.
Timer-line forgery and outliving-process tricks analysed as defended.

## 4. Ways it could report something false (ranked)

1. **Missing OTS anchors are not reported** (record; medium). `rm
   ledger/anchors/000003.ots` -> all three checks exit 0, nothing flagged,
   although 000003 is a range head the writer always stamps, so its absence is
   never legitimate. No lie becomes possible (the Rekor attestation still pins
   the bytes), but the Bitcoin-side timestamp is silently lost. Fix: flag a
   missing .ots for any entry that is a range head.
2. **Environment, not the tool**: in Docker/kata-style containers loop.py
   cannot run at all (jail analysis) — worth a stated limit, since such
   containers are "today's infrastructure".
3. **Shared-IP GitHub API rate limits** make monitor exit 2 from this sandbox
   at times (correct refusal; noted for other verifiers).

## Plan 1 verdict

**PASS.** All three checks exit 0 on the untouched ledger, and no attack —
mine above or the card's earlier rounds, re-tested where feasible — produces a
false 0: every tampering either exits 1 or leaves the entries still pinned by
Rekor. The OTS-deletion weakness loses redundant evidence silently but cannot
make the record lie. Per the rule this is a pass, not inconclusive: monitor
can and did exit 0 today, and every attack was attempted. `plan1-verdict-proposal.json`
holds an inbox-ready verdict body: wrapped as `{"kind": "verdict", "body": …}`
and ingested on a copy of the ledger it becomes entry 4, `verify` exits 0 and
`open_plans` goes empty. Loop claim 3 could not be attempted beyond code
reading: it needs the jail.
