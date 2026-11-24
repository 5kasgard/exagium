# Handoff: scheduled Claude shift, 2026-11-24 ~22:40-23:40Z

Follows the owner-attended session of 2026-11-24/25 (P36 rescope, P37 replay; audit/notes/2026-11-25/forge-os.md).

## Health
- worker.yml and propose.yml chains running: 30 runs each in the last ~4 h and ~9 h, all dispatch, all success
  except the one in progress. review.yml 30/30 success. keeper.yml last ran 2026-11-19 (the self-dispatching chains
  have not needed it). tests.yml: pushes 12/12 success; 4 PR runs `action_required` (worker PRs need approval to run
  CI; the worker's own no-secrets test job is what gates them). No stall found; nothing fixed.

## Tier moves (worker/queue_ops.py move, actor claude-shift; reasons on each run entry)
- R36 confirm -> done: PR #130 merged after removing its one forum claim (the worker's rule: a person merges forum
  claims); 3 paper claims remain.
- K67 redo -> open (PR #137 closed): rule() always returned "holds"; run() read a non-existent root queue.json;
  q2-q5 targets had no answers in the real queue. next names P8r, P11, Q07, P5 (9, 6, 5, 5 dependents).
- K70 redo -> open (PR #138 closed): run() on main read unrelated result files as arms "digests" (554) and "reps" (44)
  (run on the branch merged with main). next fixes the file path and schema.
- K68 (escalated) redo -> open: the item's key list was ambiguous; next names the four keys.
- Owner merges recorded, not made: R15 (#8), K18 (#47), K49 (#66), K58 (#119; was parked) -> done.
- P5a -> done: its pre-registered rule re-graded by script (runner/results/2026-11-20-p5a/decide.py reproduces
  decision.txt: depth helps on digits only).
- Q82 -> open (decide_open under the owner's delegation to data): PR #125's seal controls read 28 of 32 (below).

## Results
- **Q82 seal controls: 28 of 32, rule not met** (audit/notes/2026-11-24/q82-seal-controls.md; runs 45504000468,
  45504003403, 45504007010, 45504010365). Misses: gold/goldtest on iceberg (uv fetches build requirements) and aqua
  (Natural Earth download). Redo in Q82's next: warm with network, `docker network disconnect` before the attempt.
  Deviation: the PLAN's one-task dry run was skipped (no cost either way). PR #125 commented; not closed.
- **P25 step 1 done** (PR #140, main): store/gnomon-trial/QUESTIONS.md, answers.py, answers.json: 10 questions with a
  key computed from committed result files, matching DECISION.md and transfer/README.md. Card R4 does not exist.

## README
PR #143 merged: Nov 24 hardening PRs all merged (#104-#106, #108, #110, #112, #114, #115, #126); Q82 seal status in Next.

## Open list, in order
1. Q82 redo by a free worker, then the 32 controls (32 of 32) and a person's merge; P28's pilot waits on it.
2. P25 step 2: load the record into Gnomon (cap one day of glue), answer with Gnomon and grep, blind scoring.
3. From the previous handoff: P33 key broker design step; P37b event log; P35 truth set (K67 redo).
4. Owner: #109, #111; workflow fixes behind the 403; PR #125 can be closed when the redo lands.

## Spend
This Claude shift only; GitHub Actions (bench6-env x4, no model call). No paid API calls, no subagents. No items with
origin "new" were added, so no blind review was due.
