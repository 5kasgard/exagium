# Relevance filter: is its positive control the fault? (rules fixed 2026-11-22 07:50Z, before any run)

**Data that prompted this.** Since #70 (2026-11-22 02:00Z to 06:20Z), 29 research runs: 2 merged, 6 no claims, 21 failed.
All 21 failed on the relevance filter's known-answer control (`worker/research_job.py` `relevance()`): source 0 is the
question's own text and must score p >= 0.5; it scored 0.14 to 0.48 in every one (each run's `result.json` "why").
The unrelated control (ridge regression) scored 0.00 every time.

**Hypothesis.** The control contradicts the instruction. Each source is asked "Is source {i} about the QUESTION itself
... so that it could answer part of it?" A bare restatement of the question cannot answer it, so a careful decision
model says no. A positive control written as a source that answers the question should pass.

**Candidate control** (`NEW` in `replay.py`): the question as a title, then "An empirical study that answers this
question directly: we measure it and report the answer, with numbers and methods."

**Test.** `pools.json`: batch 1 (the first 38 sources) of 15 frozen runs, one per item: the 10 items whose filter failed
(R26 R29 R30 R34 R36 R38 R39 R40 R41 R42, each's latest failed run) and 5 whose filter passed (R28 R31 R33 R35 R37). One
Decisions request per pool (15, free model `perplexity/pplx-decider-v1.3:free`), same instruction as production,
sources = [old control, new control, unrelated control, the 38]. Dry run first (`--dry`, no request).

**Rule.** Adopt the new control if, on all 15 pools, it scores p >= 0.5 and the unrelated control scores p < 0.5.
Otherwise the control is not the (whole) fault: record it and look further (the instruction, the model).
Reported either way: the old control's p (replication: below 0.5 on the 10 failing pools is expected), and per pool the
number of the 38 sources at p >= 0.5 (if near 0 everywhere, the filter is also too strict, a separate finding).
A request that fails (HTTP, missing answers) is reported and counts as not meeting the rule.

**Addendum, 2026-11-22 08:15Z (after the first replay, before the second; from review):** the first replay's request
had the bare question at index 0 and the new control at index 1, so production's exact request was not tested. Second
replay, `--prod`: exactly production's layout `[new control, unrelated, the 38]`, the same 15 pools, one request each.
Same rule; the fix is kept only if it holds here too. A second request per pool in the first layout also shows how much
a score moves between requests (variance was not measured by the first run).
