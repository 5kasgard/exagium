# P28 step 3: dry runs of the agent path, no model call (2026-11-25, Claude session)

Branch `p28/step3-agent`. Rules in PLAN.md, "Step 3", written before any model attempt.

**agent-gold** (gold patch applied inside the task's container in place of an agent, diff taken from /testbed, graded by
the sealed grader): 8 of 8 resolved (run 45599149346). The extracted diffs touch the same files as the gold patches
(checked on sunpy and iceberg in run 45596327155), so the warm-up leaves nothing in the agent's diff.

**agent-mock** (bench-4's OpenHands SDK inside the task's image, through bench-4's meter to a local mock model):
sunpy and iceberg-python both started, passed the in-container seal check (meter reachable, 1.1.1.1:443 blocked), made
1 metered call, ended by themselves (harness exit 0, status FINISHED) and read not resolved (run 45599152522).

Fixed on the way (each found by a dry run, none by a model run): the venv's Python links through a version alias, so the
whole managed-Python folder is mounted (run 45596331525); OpenHands is launched by bench-4's `run_in_container.sh`
unchanged, which sets its base URL (run 45597372295); a seal check from inside the container was added before the agent.

Not shown: anything about a model's attempt. Next: the 8 model attempts under PLAN.md's pilot rule.
