# Research checker: part-answers and overstatements (Claude shift 2026-11-19 23:10Z), fixed before any reply
Cause read from the artifacts of the escalated redo runs (majority-of-3 checker):
- R4 (worker run 44631291461): 3 script-verified claims, 0 of 3 votes for each. The checker is right: each claim adds an
  "indicating ..." conclusion its quote does not state (e.g. a win rate against *default* XGBoost read as "higher accuracy
  than tuned trees"). Fix in the reader's prompt: no added interpretation, comparison or conclusion.
- R10 (worker run 44631972990): 3 script-verified claims, 0 of 3 votes; all three notes say the claims do not answer the
  whole question (it lists several sub-questions). Fix in the checker's prompt: a claim answering one part counts.

Test of the checker change (new prompt, same model as the worker, Nemotron 3 Ultra through OpenRouter's free tier since
review.yml has no NIM key; 3 calls per prompt; a claim is kept by a strict majority, as in the worker):
- A-R10.md: claims 0 and 1 are R10's S12 claims (each restates its quote and answers one part: how RDF and property
  graphs relate) -> expected KEEP; claim 2 is an accurate TabPFN claim, off-topic for R10 -> expected DROP.
- B-R4.md: R4's three original overstated claims -> expected DROP (the change must not loosen fidelity).
Rule: the checker change stays only if all 6 expectations hold. The reader change is tested live on R4's redo (its claims
must not carry such clauses); it cannot make the checker looser.

## Result 1 (review runs 44633382554, 44633386797): the checker change fails its rule
B-R4: 0 of 3 votes on each claim, as expected. A-R10: 0 of 3 votes on every claim, including the two expected KEEPs; all
three notes say the claims do not address the (whole) question. 4 of 6 expectations held; the rule needed 6. The checker
prompt stays as it was (not committed). The model does not apply a "one part counts" instruction to a six-part question.

## Test 2, fixed before replies: is the question's breadth the cause?
A2-R10-narrow.md: the same three claims and the unchanged checker prompt, with one narrow question that R10's S12 claims
answer ("How do the property-graph and RDF data models relate, and can data be converted between them without loss?").
Expect KEEP 0 and 1, DROP 2. If all 3 hold, breadth is the cause and the fix is one question per card (R10 redone narrow;
multi-part cards split before they run); if not, the S12 claims are the problem and R10 escalates with this record.

## Result 2 (review run 44633642610): breadth is the cause
Claims 0 and 1 kept by 3 of 3 calls, the off-topic control dropped by 3 of 3; all 3 expectations held. R10's question
narrowed on the queue and R10 redone. R11's escalation (run 44632810471, old reader prompt) was R4's cause, not breadth:
5 claims, each adding detail its quote does not state. Not yet a rule for all cards: R9 passed with a three-part question.
Watch R13-R19: if no_claims outcomes with "does not answer the question" notes recur, split multi-part cards before they run.

## Test 3 (Claude chat, 2026-11-20, P31), fixed before any reply: can a decision model be the checker?
Decider V1.3 (OpenRouter Decisions API, free) on the same 6 known answers (A2-R10-narrow: KEEP, KEEP, DROP; B-R4: DROP x3),
one yes/no question per claim, keep if p >= 0.5; router/decisions_checker_probe.py. Rule: it may replace the Nemotron
majority-of-3 checker in a live trial only if all 6 hold in two separate runs. Decider Flash V1.3 runs alongside for the record.
