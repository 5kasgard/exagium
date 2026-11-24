# Result, 2026-11-22 (rules in PLAN.md, fixed first; dry runs before each)

**Production layout (`--prod`, run 44998923618): rule met on 15 of 15 pools.** The new control scored 0.717-1.000 and the unrelated control 0.001-0.002. First layout (bare question at index 0, new control at 1): rule met on 15 of 15 twice (runs 44998563696 and 44998923618), new control 0.655-0.999.

| item | filter had | bare question, request 1 / 2 | new, layout 1, request 1 / 2 | **new, production layout** | unrelated (prod) | sources kept of 38 (prod) |
|---|---|---|---|---|---|---|
| R26 | failed | 0.337 / 0.337 | 0.975 / 0.975 | **0.998** | 0.001 | 1 |
| R29 | failed | 0.378 / 0.301 | 0.655 / 0.666 | **0.717** | 0.002 | 1 |
| R30 | failed | 0.261 / 0.261 | 0.947 / 0.947 | **0.974** | 0.002 | 2 |
| R34 | failed | 0.936 / 0.945 | 0.997 / 0.995 | **0.994** | 0.001 | 0 |
| R36 | failed | 0.883 / 0.813 | 0.989 / 0.993 | **0.999** | 0.001 | 1 |
| R38 | failed | 0.263 / 0.241 | 0.926 / 0.837 | **0.971** | 0.002 | 3 |
| R39 | failed | 0.509 / 0.46 | 0.981 / 0.965 | **0.985** | 0.001 | 1 |
| R40 | failed | 0.281 / 0.314 | 0.885 / 0.897 | **0.928** | 0.002 | 1 |
| R41 | failed | 0.502 / 0.502 | 0.932 / 0.932 | **0.981** | 0.001 | 0 |
| R42 | failed | 0.207 / 0.207 | 0.928 / 0.928 | **0.944** | 0.001 | 0 |
| R31 | passed | 0.978 / 0.973 | 0.997 / 0.998 | **0.998** | 0.001 | 4 |
| R35 | passed | 0.958 / 0.958 | 0.999 / 0.999 | **0.999** | 0.001 | 1 |
| R33 | passed | 0.982 / 0.982 | 0.999 / 0.999 | **1.0** | 0.001 | 0 |
| R28 | passed | 0.977 / 0.977 | 0.999 / 0.999 | **0.999** | 0.002 | 3 |
| R37 | passed | 0.771 / 0.771 | 0.999 / 0.999 | **1.0** | 0.001 | 0 |

Between two identical requests a score moved by at most 0.089. The bare question scored below 0.5 on 6 of the 10 failing pools here (in the 21 original runs it was below 0.5 every time), so its score sits around the bar; the new control's lowest score in the production layout is 0.717 (R29), one request per pool.

What this does not show: that the filter is useful. The new control states that it answers the question, so the model can pass it on that statement alone; the controls now catch all-low or all-high output and answers under wrong indices, not a filter that misses relevant sources.

**Separate finding (not acted on here):** the filter keeps few sources: 18 of 570 in the production layout (0 on 5 of 15 pools). With the control fixed, expect 'no claims' to replace 'filter failed' as the common ending. Whether the instruction is too strict is the next question, to be tested the same way.
