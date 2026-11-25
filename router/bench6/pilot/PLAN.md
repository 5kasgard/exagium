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

## Step 2: the attempt grader (rules fixed 2026-11-21 23:10Z, revised 23:40Z after review, before any run of it)

**Why before any model attempt.** Step 1 showed each usable task's tests tell the gold patch from no patch when the
test_patch is applied first. A model attempt is graded differently: the agent edits the unfixed code without the new
tests, then the test_patch is applied over its edit. That path is new, so it gets its own known-answer check, with no
model call.

**Grading an attempt** (`envcheck.py attempt`, `bench6-env.yml` with a step 2 mode): in the task's own image, apply the
attempt's diff to `/testbed` (a diff that does not apply = not resolved); put back as at HEAD (or remove if new) every
file the test_patch touches, every `conftest.py`, `pytest.ini`, `.pytest.ini`, `pytest.toml`, `tox.ini` and `setup.cfg` the attempt changed; apply the
test_patch; run the task's own FAIL_TO_PASS and PASS_TO_PASS tests once. Only the last short test summary block is
read. **Resolved** = every one of those tests reported PASSED there (SWE-bench's rule). A run that does not reach the
tests (pull failure, timeout, test_patch not applying) reads neither, and counts as a miss below. The grader's helpers
run with `python -I` from `/`, so nothing the attempt adds under `/testbed` is imported by them.

**Not ruled out** (an attempt written to fool the grader): code under test that changes how pytest reports (for example
through a plugin loaded from the package itself, `pyproject.toml`, or `uv.lock` for the task that runs `uv run`), or that
prints a whole fake summary block at exit. Model attempts' diffs are kept with their grades
so a person can read any resolved one.

**Known-answer rule.** On the 8 usable tasks of step 1 (`ids=usable`), four modes, 32 runs, no model call:
`gold` (the gold patch) and `goldtest` (the gold patch plus an `assert False` line appended to every test file the
test_patch touches, which the reset must undo) must read resolved; `empty` (no diff) and `cheat` (no fix, a root
`conftest.py` hook that marks every test passed, the test_patch's files emptied) must read not resolved. Any miss, or
any of the 32 missing, means the grader is fixed before any model attempt; the count is reported either way. Dry run
first: one task in each mode on the branch, before the 32.

**The pilot rule's open point, proposed here to settle before any attempt (not after):** one base-harness attempt on
each of the 8 usable tasks; a task is kept if its attempt finishes inside the cap (call cap and wall time of bench-4's
base harness); fewer than 6 kept means the fresh set is too hard for this pilot, and that is the result. Settled when a
person merges this PLAN; until then no model attempt runs.

## Step 3: one base-harness attempt per usable task (written 2026-11-25, before any model attempt)

**How an attempt runs** (`agent_attempt.py`, `bench6-env.yml` modes `agent-*`): bench-4's base harness (OpenHands SDK v1.61.0,
the venv copied out of bench-4's image and mounted read-only) runs inside the task's own image, launched by bench-4's
`run_in_container.sh` unchanged. Task text: the issue (`statements.json`, frozen from the same dataset rows, base commit
checked) plus "The repository is in /workspace." and nothing more. Before the agent: a warm-up runs the task's own tests
on the unfixed code with the test_patch copied in, then reversed and deleted (so build requirements and test data are
cached, and the agent never sees the tests); then bench-4's seal (containers reach only the meter), checked from inside
the container. Through bench-4's meter: model `nvidia/nemotron-3-ultra-550b-a55b:free`, cap 80 calls, 2100 s wall
(bench-4's base settings, `bench4.yml`). The diff of /testbed against the base commit is graded by the sealed grader (Q82).

**Dry runs first, no model call:** `agent-gold` (the gold patch applied in place of the agent) must read resolved on all
8; `agent-mock` (the harness through the meter to a local mock model) must start, reach the meter, end by itself and read
not resolved.

**The pilot rule, applied:** a task is kept if its attempt ends by itself (ending `exited`, no call refused at the cap,
inside the wall time); a run that never starts (`start_failed`, `warm_failed`, `seal_failed`) is fixed and rerun, not
counted. Fewer than 6 kept of 8 means the fresh set is too hard for this pilot, and that is the result. Whether an attempt
resolves its task is reported for every attempt but decides nothing in this pilot.
