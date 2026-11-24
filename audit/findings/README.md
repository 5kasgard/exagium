# Findings register (one file for every review series)

`register.jsonl`: one row per finding from every outside or in-chat review, built by `build.py` (Claude chat 2026-11-24).

**Why it exists:** five review series used their own `F001`-style numbers, so the same id named different defects. For example, GLM F001 is `router/decisions_probe.py`, while the 22 Nov review's F001 is the SWE grader. One report already mixed them up. Fixes were also made from chat sessions outside the queue, so no single place said what was still open.

**IDs are prefixed with their series:**

| Prefix | Review | Where the source is |
|---|---|---|
| `G1-` | GLM round 1 (83f06fd) | `audit/reviews/glm-20261120` |
| `G2-` | GLM round 2 new findings (5301510d) | owner's report; claims in `g2.tsv` |
| `Z1-` | Z.ai state audit (df7890aa) | owner's report and zip; claims in `z1.tsv` |
| `GM-` | Gemini rounds | `audit/reviews/gemini-20261120b` |
| `D1-` | 23 Nov delta audit | owner's report |

The in-chat rounds of 21 Nov (cited in commits as R1–R4, A, B) left no files. Their fixes appear only as commit messages.

**Status:**

| Status | Meaning |
|---|---|
| `fixed` | Cites a commit or PR. |
| `pending-pr` | Names the open PR or blocked branch. |
| `open` | No fix was found on main when the register was built. That is not proof the defect still exists. |
| `needs-read` | The code changed and nobody has read it yet. |
| `superseded` | Replaced by a narrower finding. |

- `dup_of` links the same defect found by more than one series.
- `checked` marks rows that an independent auditor read in the code.

**Accuracy, measured, 2026-11-24:**
- An independent audit read 31 sampled rows in the code.
- About 12% of `fixed` labels were wrong. About 5–10% of `open` labels were wrong.
- Every error it found is corrected here.
- Known limits:
  - G1 rows outside GLM's carried list are `fixed` on GLM round 2's word.
  - Fixes are found from commit-message ids, so a silent fix is missed.

**Counts at main 2d64e6b3 (unique rows):**

| Status | Count |
|---|---|
| open | 122 (4 high, 75 medium, 49 low) |
| fixed | 75 |
| pending-pr | 7 |
| needs-read | 2 |

46 of the open rows have a queue item.

**Next:** turn the open rows into queue items, grouped by file so related fixes land together. Rebuild with `python3 -I audit/findings/build.py REPO_CLONE OUT` after each fix wave.
