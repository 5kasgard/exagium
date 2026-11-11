# Pre-registration: bench-2, OpenHands Agent Server in a sealed container, free models

Written 2026-11-11 before any run. Author: Claude Opus (chose the setup and the arms; not an arm). It follows `router/bench/RESULTS-1.md`, whose run was void as a model test because the setup leaked the fixed code.

## Questions
- **H1′.** Is the current OpenHands Agent Server (SDK v1.61.0, image `agent-server:1.61.0-python`), with each conversation in its own container, a workable harness base?
- **H2.** Can free models drive it to fix real exagium bugs?
- **Seal.** Does the agent stay inside its workspace?
- **Budget.** How many model calls does an attempt take?

## Same as bench-1
- The tasks T1–T3 (`router/bench/tasks.json`, `tasks/`) and their task statements. The only addition is "which is in /workspace".
- The grader (`router/bench/grade.py`): the fix commit's test file is copied over the agent's own. A pass means the hidden test passes **and** no pass-to-pass test breaks. Each hidden test was validated to fail on its base and pass on its fix.
- The no-agent control Z must fail all 3, or the run is void.

## Changed from bench-1, and why
- **Seal:**
  - The agent runs inside the Agent Server container, which mounts **only** the pre-fix workspace, at `/workspace`.
  - The full checkout is **deleted** from the runner before the agent starts. The grader and the hidden test are first copied to a private directory.
- **Harness:** the current SDK (v1.61.0) replaces the unmaintained CLI (v1.16.0 on SDK 1.28.1), pinned by git tag and by image tag.
- **Iteration cap:** set explicitly to 80. bench-1 stopped at 40, apparently a default; 80 keeps the worst case (720 requests) inside the day's remaining free allowance (about 790 after bench-1).
- **Arms:** Gemini is dropped (it couldn't sustain a loop on its free tier). All arms are OpenRouter free:

  | Arm | Model | Vendor |
  |---|---|---|
  | A | `nemotron-3-ultra-550b-a55b:free` | NVIDIA |
  | B | `flint-v1.0:free` | Recursive |
  | D | `north-mini-code:free` | Cohere, built for coding agents |

  There is one run per task per arm: 9 attempts plus 3 controls. Each attempt is capped at 35 minutes.
- **Measures** (`router/bench2/measure.py`, dry-run on a fake attempt before registering):
  - **Diff:** `git diff --shortstat HEAD`.
  - **Model calls:** from the SDK's own metrics.
  - **Leak:** the agent's actions or observations name a host path (`/home/runner/work…`, `router/bench`) or fetch a URL. README text no longer counts.
  - **Rate limits:** counted from error records only.
  - **How each run ended:** finished, iteration cap, agent error, provider error, or harness crash.

## Decisions, fixed now
- **Seal holds** if no attempt shows a host path in its actions or observations. If any does, the run is void for H2, whatever the passes.
- **H1′:**
  - **holds** if at least 7 of 9 attempts end without a harness crash, where a harness crash means an exception in `run_agent.py` outside the agent loop, or an SDK error event;
  - **fails** at 4 or fewer;
  - **inconclusive** at 5 or 6.
  - Provider errors are counted and reported apart from crashes. A run cut off by a provider error is excluded from the denominator and reported.
- **H2:**
  - **holds** if any arm passes at least 2 of 3 cleanly;
  - **fails** if 0 of 9 pass cleanly with the seal holding;
  - **inconclusive** otherwise.
- **Budget:** report the median model calls per attempt. Above 100, free OpenRouter carries 10 attempts a day or fewer, and that becomes the owner's cost decision.
- 9 attempts give **a direction, not evidence.** No ranking of models is claimed.

## Spend
$0: OpenRouter free models only, and both the workflow and `engines.py` refuse paid models. Expected use is at most about 720 of the 1,000 daily free requests (9 attempts × at most 80 iterations), on GitHub-hosted runners, for this project's own development and testing.

## Outside review before running (Nemotron 3 Ultra, 2026-11-11; reply cut off at 5 of 6 points)
- **Clean is now defined:** an attempt is *clean* when `measure.py` reports `clean: true`, meaning the agent's actions fetched no URL and its actions or observations named no host path. **H2 counts only clean passes.** A clean fail is still a fail, so an agent that downloads the public fixed repo cannot pass. The "fails" rule of H2 uses clean attempts as its denominator.
- **H1′ is now a share,** so excluding provider-cut runs can't make it impossible to meet:
  - **holds** if at least 78% of the attempts not cut by a provider end without a harness crash;
  - **fails** at 44% or less;
  - **inconclusive** in between, or if fewer than 5 attempts remain.
- **The control is now specified:** one Z run per task, 3 in all. It is deterministic (there is no model), and it behaved correctly on the same runner image in bench-1.
- **Rejected: "the pass-to-pass rule can't be enforced."** The pass-to-pass lists were computed before registering, from the code before and after each fix. They are stored in `router/bench/p2p-<fix>.json`.
