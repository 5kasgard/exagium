# P28 step: SWE-Bench Illusion probe (rules fixed 2026-11-19 23:00Z, before any real call)

**Question.** Is SWE-rebench's fresh set (leaderboard splits 2025_10..2026_03, 361 tasks, `nebius/SWE-rebench-leaderboard`)
less memorised by the free model than SWE-bench Verified (`princeton-nlp/SWE-bench_Verified`, 500 tasks), where
memorisation is documented? Why it matters: queue P28 found that the rule "tasks created after the model's release" cannot
be met (Nemotron 3 Ultra post-training data runs to May 2026; the fresh splits end 2026_03).

**Method** (arXiv 2506.12286, "The SWE-Bench Illusion": state-of-the-art models name the buggy file from the issue text alone
up to 76% of the time on Verified, up to 53% on repositories outside it).
- Sample: 40 tasks from each pool, `random.Random(20261120).sample` over rows sorted by instance id; the ids are frozen in
  `frozen-ids.json` and the run refuses a different sample. Raised from the queue's 20 before any result: at 20 a side the
  dry run put the gap's standard error at 15.8 points, as large as the threshold; at 40 it is 11.2.
- Model: Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b`, NVIDIA NIM free), one call per task, no retry, max 16,000
  tokens. It sees the repository name and the issue text (first 8,000 characters; 2 fresh issues are longer, no Verified
  one is) and returns up to 3 paths in a ```paths block.
- Grading, by script: gold = files changed in the task's own `patch`. Parse = lines of the last ```paths block, else path
  tokens in the reply, normalised (backticks, list marks, `./`, `a/`, `b/` stripped). A task is correct (top-1) if the first
  path is exactly one of the gold files. No reply or no path counts as wrong.

**Rule (from queue P28, unchanged).** If Verified's top-1 accuracy minus the fresh set's is under 15 points, the fresh set is
treated as contaminated too and P28 builds its own tasks from issues after 2026-06-04 with SWE-rebench's open pipeline;
otherwise the fresh set is used. Reported, not decisive: any-of-3 accuracy, answered counts, a 95% interval for the gap,
and top-1 on single-file tasks only (the fresh sample has 23 multi-file patches of 40, Verified 8, which makes top-1 easier on
the fresh side, so it leans toward "contaminated").

**Controls.** `--dry` answers even-numbered tasks with their first gold path and odd ones with a wrong path; grading must read
exactly 0.50 in both sets (it did locally, and runs in CI before the real call). `test_illusion.py` holds known-answer tests.

**Cost.** 80 NIM calls, reserved on the worker ledger (cap 400 a day, worker/ledger.py). Free.
