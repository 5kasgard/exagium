# worker — spec card (queue P8, first version; P7 and P9 wired in)

**What it is.** Free models working the queue without a chat. A scheduled job claims the top ready item, a free model
does it, another family checks it, a script checks both, and the result lands on main or as a PR. A daily digest
writes the handoff. Claude designs and spot-checks; it is not in the loop.

**How it decides (hypotheses, measured by the digest):**
- **Claims:** git is the lock. A claim is a commit to the queue branch, and git refuses the second of two concurrent
  pushes. Leases last 2 h. A chat's claim always wins. An item whose files a live claim touches is skipped.
- **Job types**, by what an item carries:
  - `targets` (exact file and line ranges): a doc fix (`doc_job.py`);
  - `research` (a question and queries): a literature card (`research_job.py`, P7). Sources, by kind: papers and
    preprints (OpenAlex, arXiv, Semantic Scholar), encyclopedia (Wikipedia), Q&A (Stack Exchange), forums (Hacker
    News comments, Reddit). Every claim is graded by evidence: measured > reported > expert view > experience report >
    opinion, and the card groups claims by grade. YouTube comments need an API key; a headless browser for pages with
    no API comes when a question needs one (queue items).
- **Who does what:**
  - search: API scripts (OpenAlex, arXiv, Semantic Scholar);
  - drafting and first reads: gpt-oss-120b on Cloudflare;
  - the check: Nemotron 3 Ultra on NIM, another family;
  - verification by script: edits only on target lines with exact old text, Markdown only; quotes verbatim in the
    source abstract, numbers in the quote.
- **Lands on main** only if every script check passes and the second model approves (doc: it approves the diff;
  research: it keeps the claim and its grade); else a PR, with the item waiting on the owner. Never written:
  workflows, the worker's own files, registered plans. Only by PR: anything agents read as instructions (AGENTS,
  CLAUDE, VERIFY, BRIEF, SKILL, handoffs, .claude/), and cards with claims from forums, Q&A or Wikipedia (text anyone
  can write; cards also carry an untrusted-data banner); a new line may not add a link, HTML or mention; card text is
  cleaned to one line with no links, HTML, mentions or issue references; a card is never overwritten.
- **Failures:** an item that fails twice waits on the owner instead of being retried hourly; a failed push to main falls
  back to a PR.
- **Budget:** a run checks headroom before it claims, then `ledger.py` reserves its calls by commit, against caps of
  half each free pool's daily allowance. Calls are counted exactly; Cloudflare neurons are projected (1,000 per call)
  and checked against recorded use, so one run can overshoot by the difference. Paid pools can't be reserved.
- **Jobs:** the key and the write token are never in one job: claim (write, no keys), prepare (read-only, no keys;
  the literature search parses untrusted API text here), draft (keys, read-only, runs nothing a model writes), land
  (write, no keys).
- **Pause:** a file `worker/PAUSE` on main stops every run before it claims.

**Measured** (`digest.py`, P9's first measures): claims, double claims (target 0), releases, expired leases, results
by kind, budget used per pool. Spot-check acceptance is recorded per item by Claude or the owner.

**Files.** `queue_ops.py` (claims), `ledger.py` (budget), `doc_job.py`, `research_job.py`, `digest.py`,
`test_worker.py`; workflows `worker.yml` (hourly) and `digest.yml` (daily, 06:23 Brisbane). Records: the queue's
items (`runs`), and the `worker-runs` branch (`ledger/`, `digest/`).

**Not yet:** code items (they need P5's jailed proposer and Q56), the P10 auditor, process proposals (P9's weekly
free-model step).

**Reviews before the first run (2026-11-18):** a separate agent found 4 major issues (a research merge broke the release;
a failed push counted as done; unchecked model text could land on main; a failing item burnt budget hourly), all fixed;
gpt-oss-120b (router-runs `reviews/44290632002`): 3 of 8 points held (neuron cap not projected, false double claims
after a lease expiry, a lenient pointer check; the first two fixed), 5 rejected with reasons in the queue's P8 notes.
