# P28 step: environment check of the pilot's candidate tasks (rules fixed 2026-11-20 09:10Z, before any run)

**Why first.** P28's pilot runs the base harness on fresh SWE-rebench tasks and grades each attempt with the task's own
tests. A grader that cannot tell a fixed task from an unfixed one would make every later number meaningless, and an
image that does not pull or build would be scored as the agent's failure. This step makes no model call (no cost).

**Candidates.** The fresh pool of `router/bench6/illusion` (`nebius/SWE-rebench-leaderboard`, splits 2025_10..2026_03),
minus the 40 ids that probe showed to the model, sorted by instance id, permuted by `random.Random(20261121)`. The first
12 of the permutation are the candidates, frozen in `candidates.json` (with patches, tests and image names, so the run
does not depend on the dataset staying unchanged). If a later step needs more, it continues down the same permutation.

**Check per task** (one CI job each, `bench6-env.yml`, 30-minute limit, the task's own `docker_image` and `test_cmd`):
1. pull the image; 2. apply `test_patch` and run the FAIL_TO_PASS and PASS_TO_PASS tests: this is the known-answer
negative (unfixed code); 3. apply the gold `patch` and run them again: the known-answer positive.
A task is **usable** if: the image pulls; in step 2 at least one FAIL_TO_PASS test is not reported PASSED; in step 3
every FAIL_TO_PASS and every PASS_TO_PASS test is reported PASSED (pytest `-rA` lines, parsed by script); all inside
the limit. Anything else (pull failure, patch failure, missing tests, timeout) makes the task unusable, with the reason.

**Rule.** Usable tasks go on to the pilot. If fewer than 8 of 12 are usable, the grader or the images are the problem
and are fixed or replaced (next tasks down the permutation) before any model attempt; the count is reported either way.

**Dry run first.** One task (`ids=<first>`) before the 12, since the workflow is new.

**Open point for the pilot's rule (recorded, not changed here).** P28 says "a 6-attempt pilot (6 candidate tasks, 1
each)" and also "if fewer than 6 tasks survive from 12 candidates": the two do not fit (6 attempts cannot keep 6 of 12).
The next step settles it before any attempt.

## Step 2: the attempt grader (rules fixed 2026-11-21 23:10Z, before any run of it)

**Why before any model attempt.** Step 1 showed each usable task's tests tell the gold patch from no patch when the
test_patch is applied first. A model attempt is graded differently: the agent edits the unfixed code without the new
tests, then the test_patch is applied over its edit. That path is new, so it gets its own known-answer check, with no
model call.

**Grading an attempt** (`envcheck.py attempt`, `bench6-env.yml` mode `gold` or `empty`): in the task's own image, apply
the attempt's diff to `/testbed` (a diff that does not apply = not resolved); put every file the test_patch writes back
as at HEAD (removed if new), so an attempt cannot pass by editing the tests; apply the test_patch; run the task's own
FAIL_TO_PASS and PASS_TO_PASS tests once. **Resolved** = every one of them reported PASSED (SWE-bench's rule). A run that
does not reach the tests (pull failure, timeout, test_patch not applying) reads neither, and counts as a miss below.

**Known-answer rule.** On the 8 usable tasks of step 1: the gold patch as the attempt must read resolved on all 8, and an
empty diff must read not resolved on all 8. Any miss means the grader is fixed before any model attempt; the count is
reported either way. Dry run first: one task in each mode on the branch, before the 16.

**The pilot rule's open point, proposed here to settle before any attempt (not after):** one base-harness attempt on
each of the 8 usable tasks; a task is kept if its attempt finishes inside the cap (call cap and wall time of bench-4's
base harness); fewer than 6 kept means the fresh set is too hard for this pilot, and that is the result. Settled when a
person merges this PLAN; until then no model attempt runs.
