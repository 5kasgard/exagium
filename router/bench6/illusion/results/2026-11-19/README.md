# P28 Illusion probe, 2026-11-19 (run 44632998540; rules in ../../PLAN.md, fixed before the call)
Nemotron 3 Ultra (NIM free) named the file to change from the repository name and issue text alone. 80 calls, all
answered (finish "stop"), graded by script against each task's own patch.

| set | n | top-1 | any of 3 |
|---|---|---|---|
| SWE-rebench fresh (2025_10..2026_03) | 40 | 0.35 | 0.42 |
| SWE-bench Verified | 40 | 0.72 | 0.80 |

Gap 37.5 points (SE 10.3; 95% interval 17.3 to 57.7) against the rule's 15: **the fresh set reads less memorised; P28 uses
it.** Single-file tasks only (not decisive): fresh 0.12 of 17, Verified 0.75 of 32. Verified's 0.72 sits near the 76%
the Illusion paper reports for state-of-the-art models (arXiv 2506.12286).

Holds for this model, this prompt and these 80 tasks. It shows the fresh tasks are less recalled from issue text; it does
not prove they are absent from training (post-training data runs to May 2026). Dry control: run 44632695009 (0.50/0.50).
