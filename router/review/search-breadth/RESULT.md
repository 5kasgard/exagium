# Result, 2026-11-22 (decisions-probe runs 45002586528 dry, 45002936286 real; branch test/search-breadth; rules in PLAN.md, fixed first)

**Rule met: SearXNG adds about 1.4 times as many relevant sources as the current search finds (A + new B = 2.39 x A; rule 1.25).**

| item | current kept | SearXNG kept | SearXNG new (URL not in current) |
|---|---|---|---|
| R26 | 4 | 10 | 10 |
| R29 | 25 | 53 | 49 |
| R30 | 12 | 18 | 17 |
| R34 | 15 | 16 | 13 |
| R36 | 8 | 13 | 13 |
| R38 | 19 | 32 | 26 |
| R39 | 21 | 20 | 20 |
| R40 | 12 | 13 | 11 |
| R41 | 21 | 34 | 34 |
| R42 | 6 | 1 | 1 |
| R31 | 18 | 18 | 15 |
| R35 | 7 | 19 | 18 |
| R33 | 3 | 9 | 8 |
| R28 | 15 | 14 | 12 |
| R37 | 0 | 2 | 2 |

Blind precision sample (Claude, 40 per arm, shuffled together, arm hidden; `labels-sample.json`): current 25/40 = 0.625, SearXNG new 26/40 = 0.650. Estimated relevant: current 186 x 0.625 = 116; SearXNG new 249 x 0.650 = 162.

Poor pools: R26 4 -> +10 (the RFC 6585 and RFC 9110 pages themselves, which no current source returned); R37 0 -> +2; R33 3 -> +8; R35 7 -> +18. R42 is the exception: SearXNG kept 1.

Engines that did not answer from CI (queries out of 74): duckduckgo 74, google scholar 74, semantic scholar 68, europepmc 59. The results came from the engines that did; which ones, per result, is in the run's breadth.json ('via').

Limits: one labeller (Claude), 40 per arm; snippets (up to 600 characters) stand in for abstracts, so the reader will need page text (next test: extraction). One run; SearXNG's engines change. Nothing in production changed yet.

Which engines the kept SearXNG sources came from (a source can list several): google cse 202, arxiv 60, semantic scholar 6, pubmed 3, openairepublications 1. **Most of the gain rests on one engine (Google's custom-search endpoint)**; if it starts refusing CI addresses, the gain shrinks to roughly the arXiv share. The integration should log per-engine counts every run so that shows up at once.
