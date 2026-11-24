# P25 step 1: ten questions fixed before Gnomon or grep is run (2026-11-24)

Queue item P25 tests Gnomon (5kasgard/gnomon) as the Forge's knowledge store. Rule, fixed in the queue before this
file: load the Forge's results (transfer step 1, runner P2, P3 step 1, research cards R1-R8) as Gnomon claims with their
evidence files and run IDs; answer these 10 questions with Gnomon and with grep over the repo; a separate agent scores
both blind against `answers.json`. Adopt Gnomon as P11's base if it answers at least 8 of 10 correctly and none
wrongly, with no more than a day of glue code; otherwise build P11 minimal and carry Gnomon's ideas.

`answers.json` is computed by `answers.py` from the committed result files and cards (no model, no network); it is the
key. Whoever answers (Gnomon or grep) must not see `answers.json` or `answers.py`. The record is pinned at this file's
commit; Q2's answer depends on it, so both arms load the repo at this commit.

A numeric answer is correct when it matches the key to 4 decimals (accuracy differences, SE) or exactly (counts, names).
A list is correct when it has the same members. "I cannot answer from the store" counts as not correct, not wrong.

1. On P3 step 1's text task, which candidates have higher dev accuracy than label-linear, and by how much (paired
   difference and its SE over the 6,000 items)?
2. Which files in the repo cite Actions run 44486290814?
3. When Q54 was weakened (P3 step 1, rule 3), what was label-wisard's paired accuracy difference and SE against the
   better of label-linear and label-mlp16 on each task, and its instructions per item against that candidate's?
4. In transfer step 1, which family pairs change their accuracy order from one task to another?
5. In transfer step 1, how many order flips were found for each measure (accuracy, instructions, native time)?
6. In runner P2, how many of the 200 SAT dev instances did each candidate solve?
7. In P3 step 1, on each task, which candidate is cheapest in instructions per item among those within 1 SE of the most
   accurate (majority excluded)?
8. For research cards R1-R8: how many claims does each list under each evidence grade, and which card does not exist?
9. In P3 step 1, on which tasks does label-gbdt beat the better of label-linear and label-mlp16 by more than 2 paired SE?
10. Which of the cards R1-R8 that exist hold no claim graded "measured"?

Next step (step 2, free workers or Claude): load the same record into Gnomon (`store/gnomon-trial/load.py`, glue code
counted in hours), answer with Gnomon and with grep, then the blind scoring.
