# Can a decision model route an item between pipelines? (queue P31; owner 2026-11-20: route the system's interactions by decisions)
Fixed before any reply. Eight known cases from the record: each item's worker result just before the Claude tier ruled, and
the move the tier made (Q60 migration, chat and shift): E1, D16, E5 confirm; Q38 escalate (a guarantee's wording is an owner
decision); R4, R10, R11, R13 redo. One 'choice' question per case: confirm, redo or escalate. The model chooses among the
moves transitions.py allows from the item's state; it can never create a move.
Rule: routing by decision model may be tried live only if it gets at least 7 of 8 right in two separate runs, and any case
whose top option has probability under 0.6 counts as 'unsure' (escalated to the next tier, not counted as right).
