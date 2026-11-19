<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=14.46s -->

**P28 – KEEP** – adds a truly external, frozen benchmark and can be run within the free‑compute budget, directly addressing the project’s biggest risk of over‑fitting to self‑written tasks.  

**P29 – DEFER** – postpone until P28 produces a stable “best outside harness” and shows that the current default is definitively out‑performed; only then is there justification for spending scarce compute on an internal harness‑evolution loop.  

**Most likely “nice‑to‑talk‑about” idea:** P29. It is driven by the appeal of an internal “self‑optimising” harness rather than any empirical need demonstrated by existing data.  

**Largest flaw in the test rules:** The evaluation hinges almost exclusively on a token‑per‑pass ratio, with pass quality only checked by a non‑significant Fisher test. This permits a cheap harness that produces many low‑quality passes to be accepted.  

**Cheapest fix:** Add a mandatory minimum pass‑success threshold (e.g., ≥95 % of the baseline pass‑success rate) that a harness must meet before its token‑per‑pass statistic is considered.
