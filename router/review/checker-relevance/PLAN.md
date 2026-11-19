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
