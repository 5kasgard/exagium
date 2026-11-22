# Result, 2026-11-22 (decisions-probe run 44999835217, branch fix/relevance-instruction; rules in PLAN-instruction.md, fixed first; dry run before)

**Rule chose B@0.3.** Scored on Claude's labels for 5 pools (26 yes, 128 no; 36 unsure left out).

| variant @ threshold | recall | precision | kept yes / all yes | controls held |
|---|---|---|---|---|
| A@0.5 | 0.385 | 1.0 | 10 / 26 | True |
| A@0.3 | 0.462 | 1.0 | 12 / 26 | True |
| B@0.5 | 0.769 | 0.909 | 20 / 26 | True |
| B@0.3 | 0.808 | 0.913 | 21 / 26 | True |

Per pool, sources kept at 0.5 / 0.3 (YES control p):

- A: R31 5/6 (YES 1.00); R38 3/4 (YES 0.97); R30 2/2 (YES 0.97); R42 0/1 (YES 0.93); R26 1/1 (YES 1.00)
- B: R31 12/12 (YES 0.76); R38 10/12 (YES 0.61); R30 2/2 (YES 0.93); R42 4/6 (YES 0.74); R26 1/1 (YES 0.97)

One request per variant and pool; the labels are one judge's reading, not ground truth. R26 kept 1 source under every variant: its search pool is almost all off-topic (Claude's labels: 0 yes of 38). That is the second fault, in search, not in the filter.
