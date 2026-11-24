# Relevance filter: is its instruction too strict? (rules fixed 2026-11-22 08:10Z, before any request)

**Data.** With the control fixed (#90), production's request kept 18 of 570 sources on 15 frozen pools (0 on 5 of 15;
RESULT.md). Reading the titles, some pools hold many on-topic sources the filter dropped (R31: 4 kept, R38: 3 kept),
and others hold almost none (R26, R42: the search returned mostly off-topic sources). Two separate faults are possible:
the filter too strict, and search too broad. This tests the first.

**Known answers.** `labels.json`: 5 pools (R31, R38 rich; R30, R42, R26 poor), labelled by Claude before any request:
yes 26, no 128, unsure 36 (left out). Labels are one judge's reading of titles and abstract openings: a reference, not
ground truth.

**Variants** (same model, same production layout [YES control, NO control, 38 sources], one request per pool each):
- A, the current instruction: "Is source {i} about the QUESTION itself (not merely the same broad field), so that it could answer part of it?"
- B: "Could source {i} give evidence, a method, a result or a counterexample that bears directly on some part of the QUESTION? Being in the same broad field is not enough."
Thresholds scored: 0.5 (production) and 0.3. 10 requests in all.

**Score** per (variant, threshold), over the labelled yes/no sources of all 5 pools: recall = kept yes / all yes;
precision = kept yes / all kept labelled. Controls must hold for that variant on all 5 pools (YES >= threshold, NO < threshold).

**Rule.** Adopt the (variant, threshold) with the highest recall among those with precision >= 0.7 and controls held,
if its recall is at least 0.15 above A at 0.5. If none qualifies, the instruction stays, and the fault is recorded as
not (only) the filter. A failed request counts as the variant failing.
