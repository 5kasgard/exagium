# Handoff: Claude session, 2026-11-25 ~09:30–20:45Z (continuation of the 09:00Z shift, owner live at times)

**Owner decisions in chat:** merge PRs #179 and #180 (done); record Q82 as done (decide_done, done); work on P28 step 3 next.

**Done:**
- K68 → done. Claude wrote packet.py (PR #179, merged as 76324ec) after two free-worker runs produced no edits. test_proposals: 19/19.
- Q82 → done (owner merged #177). README now says #177 is merged (#180).
- **P28 step 3 → PR #194 (branch p28/step3-agent; router/ and workflows, so a person merges).**
  - `agent.py` runs bench-4's base harness (OpenHands SDK) inside each task's own image. The workflow is `bench6-env.yml`, modes agent-gold, agent-mock and agent-model. A new workflow file cannot be dispatched until it is on main, so the modes went into the existing file.
  - `statements.json` freezes the issue text for each task. PLAN.md gained a "Step 3" section, written before any model run.
  - Dry runs: agent-gold resolved 8 of 8 (run 45599149346); agent-mock ran 2 of 2 (run 45599152522), sealed, through the meter, ending by itself.
  - Three dry-run fixes: mount the whole uv Python folder; use bench-4's own launcher; add a seal probe inside the container.
  - **Pilot (run 45601064588): 2 of 8 kept; the rule needs at least 6, so it is NOT MET.** Six attempts hit the 80-call cap, and 2 of 8 resolved their task. `grade_pilot.py` graded from the meter logs.
  - A labelling bug turned up: meter.py logs refusals as `refused_cap`, but agent.py counted HTTP 429. It is fixed and tested. Results are in `router/bench6/pilot/results/2026-11-25-pilot.md`.
- README: a Nov 25 Progress row and Next item 4 updated, both in #194.

**Owner decision open (P28):** what next: a larger cap, another model, an easier sample, or a different held-out set? The pilot does not tell a short cap apart from a weak model.

**Open list, in order:** merge #194; the P28 direction; Q85 (labelled replay of the claim decider); P38(b); P33.

**Spend:** about 640 free OpenRouter calls (8 × ≤80) and GitHub CI. Nothing paid.

**No new items with origin "new"**, so no blind review was due.
