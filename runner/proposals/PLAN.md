# P5: free models propose candidates (plan, fixed before the first proposal)

Written 2026-11-19 before any proposal was requested. Commit time is the registration time. Design from the project doc
`forge-next-item-2026-11-18` (step 4), narrowed by the 2026-11-19 reviews.

## Hypothesis
A free model, given the task contract, one task card, one working candidate as an example, and the current dev scores,
can write new candidates (`train.py` + `infer.c`) that run under the contract and beat the task's floor, often enough
to be worth running unattended.

## How (`.github/workflows/propose.yml`)
1. **reserve** (no key; writes the worker ledger): one NVIDIA NIM call per proposal against the worker's daily caps
   (`worker/ledger.py`, shared with the queue worker).
2. **propose** (holds the key; runs no generated code): `runner/proposals/propose.py` builds the prompt and makes one
   call per proposal to Nemotron 3 Ultra (NVIDIA NIM, free). Tasks rotate over optdigits, text and adult.
3. **score** (no secrets; read-only token): `runner/proposals/extract.py` takes the first fenced block of each kind
   (hypothesis, `train.py`, `infer.c`) and ignores any further same-kind blocks, with size limits, into
   `runner/candidates/prop-<run>-<k>/` on the runner only.
   `runner/run.py reproduce` (twice, in the jail) and `evaluate --split dev` with the card's baselines. Dev only: no
   proposal goes to the test gate (Q25: public test labels could be memorised).
4. **record** (no key; writes the `proposals` branch and the ledger): prompt, reply, files, result and a leaderboard.

## Measures, fixed now
Over the first 20 proposals:
- **valid**: extracted, compiled in the jail, trained twice to identical weights, and dev accuracy above the floor
  (majority) by at least 3 SE;
- **valid rate** per task and overall, with a 95% Wilson interval;
- **beats a baseline**: dev accuracy above label-linear by more than 2 paired SE;
- **free calls per valid candidate**;
- instructions and ns per item of every valid candidate, against the baselines.

## Rules, fixed now
- **Switch rule** (from the design): fewer than 2 valid of the first 20 (below 1 in 10) means the one-call proposer
  is dropped for an agent harness (OpenHands with Nemotron, bench-proven) before more runs.
- **Keep rule**: at least 2 valid of the first 20 keeps the one-call proposer running, with the same measures
  reported per batch.
- No proposal enters `runner/candidates/` on `main` without a separate check by Claude or the owner (code from a model).

## Forecasts (Claude's, for later calibration scoring)
- At least 2 valid of 20 (keep rule): p = 0.7.
- Overall valid rate at least 0.4: p = 0.35.
- At least one valid proposal beats label-linear on some task: p = 0.4.

## Not claimed
Anything on the test split; anything about models other than Nemotron 3 Ultra on NIM; agent harnesses.
