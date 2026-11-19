# Grading rule for release test 2 acceptance, fixed before any grader reply (2026-11-19 19:55Z)
- Graders: three free models from families that did not draft, review or check the items (the worker used gpt-oss-120b and Nemotron):
  cloudflare llama-4-scout-17b, mistral-small-3.1-24b, qwen3-30b-a3b. Blind: they see the ask, the fixed acceptable outcome,
  how it was delivered, and the change; no worker names, run ids or the worker's own verdicts.
- An item is acceptable if at least 2 of the 3 graders say ACCEPTABLE. A grader reply that is missing or malformed for an item counts as NOT.
- Test 2's rule (worker/RELEASE-TEST-2.md, unchanged): pass on acceptance needs >= 25 of 30 (95% Wilson lower bound >= 0.65).
- Claude reads every disagreement and reports it, but does not overturn the majority; a suspected grader error goes in the results as a note.
