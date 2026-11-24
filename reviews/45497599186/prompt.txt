You are reviewing two proposed work items for a public research repository. In it, free AI models propose small learning programs (Python training plus C inference). Each proposal is compiled and run in a sandbox, then scored on a 956-item dev split against fixed baselines. For each item answer KEEP, DEFER, MERGE (with which) or CUT, give one sentence of reason, and name the single cheapest test that would show it is worth doing. Judge only the evidence given. Reply as a JSON list of {"item", "verdict", "reason", "cheapest_test"}.

Measured context:
- 1,097+ proposals have been scored, at one free model call each.
- About 59% are valid. It takes 1.6-1.7 calls per valid proposal.
- A held-out confirmation measured a winner's-curse excess of +0.0052 accuracy.
- The proposer never sees compile errors, instruction counts or which items it failed.

A. Staged selection: successive halving over rotating dev sub-folds of 60, 120, 240 and 480 items. Weak candidates are eliminated early with an anytime-valid e-value test against a fixed baseline. One pre-registered final paired test runs on a fresh procedural confirmation fold. It replaces one-pass scoring of every proposal on the full dev split.

B. A telemetry packet and a repair loop: each new prompt carries the parent proposal's compile errors, its instruction and cache-miss counts, the clusters of items it failed, and the current best design in the same cost range. A failed proposal gets at most two repair calls. It is measured as a new arm against the unchanged one-call baseline.
