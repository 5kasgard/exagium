# Can a decision model route an item between pipelines? (queue P31; owner 2026-11-20: route the system's interactions by decisions)
Fixed before any reply. Eight known cases from the record: each item's worker result just before the Claude tier ruled, and
the move the tier made (Q60 migration, chat and shift): E1, D16, E5 confirm; Q38 escalate (a guarantee's wording is an owner
decision); R4, R10, R11, R13 redo. One 'choice' question per case: confirm, redo or escalate. The model chooses among the
moves transitions.py allows from the item's state; it can never create a move.
Rule: routing by decision model may be tried live only if it gets at least 7 of 8 right in two separate runs, and any case
whose top option has probability under 0.6 counts as 'unsure' (escalated to the next tier, not counted as right).

## Result 1 (run 44667045470, Decider V1.3 free): fails the rule, 6 of 8
Right and confident: E1 confirm 0.98, Q38 escalate 0.81, R4/R10/R11/R13 redo 0.84-0.90. D16: confirm 0.48 vs redo 0.47,
unsure by the 0.6 band, so it would have gone up a tier (the band worked). E5: escalate 0.70, wrong (the guard was right
and the line stayed; a wasted escalation, in the safe direction). Not adopted for live routing. A second run is not needed
to settle the rule. Likely cause to test next: the state gave the worker's result but not the item's acceptable outcome or
which guard fired and why that guard exists, so 'confirm' for a guard block was unknowable from the state alone.
