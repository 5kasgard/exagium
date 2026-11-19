# P5b result: P5's top 5 on a held-out 20% of digits train (run 44628651975; plan in runner/proposals/confirm/PLAN.md)

- **Rule 1, above every family: 0 of 5.** The best (prop-44492370928-2, label-mlp2l) is +0.0017 ± 0.0046 over gbdt:
  indistinguishable. "Above every family" on dev was selection luck plus noise.
- **Rule 2, accuracy-cost front: 3 of 5 advance.** prop-44549931886-1 and prop-44601956368-1 (17,000-19,000 Ir/item) and
  label-mlp2l (36,203) are within 1 SE of gbdt's accuracy at a third or less of gbdt's 51,352 Ir/item, and beat mlp16 and
  linear by more than 1 SE. That is the useful finding: gbdt-level accuracy for a fraction of its instructions.
- **Rule 3, winner's curse: confirmed in direction.** The five dropped 0.0079 from dev to confirmation on average; the
  baselines 0.0027; excess 0.0052 (about 0.8 of one candidate's SE).
- Forecasts (made before the run): rule 1 holds for none (p 0.7: right); rule 2 at most 1 (p 0.6: wrong, 3); proposals drop
  more than baselines (p 0.75: right). Brier 0.17.
- Caveats: one task, one split, one host; the confirmation items were in train when P5 scored on dev; mlp16 itself reads
  0.9634 here against 0.9770 on dev, so splits differ and only the paired differences within this run compare.
