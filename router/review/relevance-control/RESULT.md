# Result, 2026-11-22 (decisions-probe run 44998563696, branch fix/relevance-control; rules in PLAN.md, fixed first)

**Rule met: the new control scored p >= 0.5 and the unrelated control p < 0.5 on all 15 pools.** Dry run first (no request).

| item | filter had | old control (bare question) | new control | unrelated | sources kept of 38 |
|---|---|---|---|---|---|
| R26 | failed | 0.337 | 0.975 | 0.001 | 1 |
| R29 | failed | 0.378 | 0.655 | 0.001 | 1 |
| R30 | failed | 0.261 | 0.947 | 0.001 | 2 |
| R34 | failed | 0.936 | 0.997 | 0.001 | 0 |
| R36 | failed | 0.883 | 0.989 | 0.001 | 1 |
| R38 | failed | 0.263 | 0.926 | 0.001 | 4 |
| R39 | failed | 0.509 | 0.981 | 0.001 | 1 |
| R40 | failed | 0.281 | 0.885 | 0.001 | 0 |
| R41 | failed | 0.502 | 0.932 | 0.002 | 0 |
| R42 | failed | 0.207 | 0.928 | 0.0 | 0 |
| R31 | passed | 0.978 | 0.997 | 0.0 | 4 |
| R35 | passed | 0.958 | 0.999 | 0.001 | 1 |
| R33 | passed | 0.982 | 0.999 | 0.001 | 0 |
| R28 | passed | 0.977 | 0.999 | 0.001 | 3 |
| R37 | passed | 0.771 | 0.999 | 0.001 | 0 |

Old control on the 10 failing pools: 6 of 10 below 0.5 this time (0.207-0.936): it sits around the bar and moves between requests, consistent with the hypothesis (in the original 21 runs it was below 0.5 every time). New control: 0.655-0.999.

**Separate finding (not acted on here):** the filter keeps few sources: 18 of 570 (1.2 per 38 on average; 0 on 6 of 15 pools). With the control fixed, many runs will now reach the reader with 0-4 sources, so 'no claims' may become the common ending. Whether the instruction is too strict is the next question, to be tested the same way.
