<!-- nvidia nvidia/nemotron-3-ultra-550b-a55b status=200 latency=42.43s -->

EXPLOITABLE: none found. The `if` gate (`author_association == 'OWNER' && user.login == repository_owner && !pull_request`) is cryptographically bound to GitHub’s identity; a non-owner cannot satisfy it. The model receives only the owner’s comment text and (iff the owner opened the issue) the issue title/body. The system agent config (`/etc/opencode/opencode.json`) denies bash, edit, webfetch, websearch, task, skill, and `external_directory` (blocks `/proc`, `/sys`, etc.); resolved tools are read/glob/grep/list only. The model cannot execute commands, write files, reach the network, or read its own environment. The prompt instructs it to “name the paths you used,” so it may surface repo file contents in the answer, but the repo is public—no new exposure. The post job runs once per workflow, has `issues: write` only, and posts exactly one comment.

KEY: the paid step’s `OPENCODE_API_KEY` cannot leak. It exists only in that step’s `env`; the model runs with no shell, no `env` access, no filesystem write, no network, and `external_directory` deny blocks `/proc/self/environ`. Stdout/stderr are captured by the shell redirection, not by the model. The artifact (`answer.md`) and the post job’s environment contain no key. Workflow logs show only the tailed `err.txt` (opencode stderr), which does not include the key.

SPEND: controls — owner-only trigger; `models.json` `allow_paid` gate; 15-min (`timeout 900`) hard limit per run; provider/Console spend limits. Missing — no workflow-level concurrency cap across issues (`cancel-in-progress: false` queues unlimited runs on different issues), no per-run token/cost budget, no alerting. A compromised owner account or a mistaken spam of `/oc` comments on many issues could queue dozens of paid runs sequentially.

RESIDUAL: runaway spend via unbounded queued runs across issues. Cheapest control: change `cancel-in-progress: true` (limits to one active run per issue) and document that the owner must enforce monetary caps at each provider/Console.

VERDICT: SHIP WITH CHANGES
1. `concurrency.cancel-in-progress: true`
2. Add a comment in the workflow reminding the owner to set hard spend limits at Anthropic/Google/NVIDIA/HF and in OpenCode Console.
