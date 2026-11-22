# Search breadth: does one keyless metasearch find more relevant sources than the per-site APIs? (rules fixed 2026-11-22 09:00Z, before any run)

**Why.** Per-site search needs a key or a workaround per site (Semantic Scholar and Reddit skipped on 79 of 79 queries,
Stack Exchange throttled on 24 of 79, GitHub 0 hits on 50 of 79: logs of the 15 frozen runs). SearXNG is one open-source
metasearch that fans a query out to many engines, needs no key, and runs as a container inside the CI job. Hypothesis:
it adds relevant sources, mostly where the current pools are poor.

**Arms, same 15 questions and the same queries each item already used** (`questions.json`):
- A (current): the first 76 sources of each run's frozen pool.
- B (SearXNG, image `searxng/searxng:2026.11.21-c6ee3e9fd`, categories general and science, JSON): the results for those
  queries, deduplicated by URL, taken round-robin across queries up to 76; snippet as the abstract. Per-engine counts and
  errors are logged (some engines may refuse CI addresses; that is part of the measurement).

**Score.** Production's relevance filter (`research_job.relevance`, instruction B at 0.3, controls) on both arms, 2
batches of 38 per arm and question at most. Kept counts per arm. Then a blind precision check: Claude labels a random
sample of up to 40 kept sources per arm, arm labels hidden and order shuffled (seed 20261122), yes = bears directly on some
part of the question.

**Rule.** Estimated relevant = kept x sampled precision, per arm, summed over the 15. Add SearXNG as a source if
(A + B relevant, after removing B's URLs already in A) is at least 1.25 x A's; else record the result and do not add it.
Also reported: the questions where A kept at most 1 and B kept 2 or more (the poor-pool case), and what the engines refused.
A filter failure on an arm counts as 0 kept for that question. Dry run first (`--dry`: SearXNG queried, no model call).
