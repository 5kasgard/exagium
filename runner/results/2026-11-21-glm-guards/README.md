# Runner guards from GLM's review (2026-11-21, local, jail confine)

F051-F053 ported into `runner/run.py`, each checked against a known-bad candidate (`runner/controls/`):

| control | what it does | caught by |
|---|---|---|
| ctl-pos | answers depend on the item's position, not its content | items reordered every timed round; per-item scores changed |
| ctl-n0 | writes a record when given no items | N = 0 must write nothing |
| ctl-cnt | answers differently when LD_PRELOAD is set (as under valgrind) | output under the counter differs from native output |

Honest runs (`dev-sat.json`, `dev-optdigits.json`, rounds 2 and 1) pass, and their counts match the recorded runs:
sat-walksat Ir 702,716,557 / 136,982 exactly as `../2026-11-18/local/dev-sat.json`; optdigits differs from the CI
run `../2026-11-20-p5a` by the same 28 instructions at N and at 0 (another host), so Ir per item is unchanged.

Findings that changed the design:
- sat-walksat seeds by position, so a verifier task's answers legitimately change when items are reordered. The
  guard therefore compares answers first and, if they differ, each item's checker score (a label task's answer is
  its score). The result json marks `reordered_answers_differ`.
- loop.py's RATIO_BAND (30x) is reported, not enforced: honest trivial baselines reach 20-46x in past runner results
  (sat-zeros 38.5x here), because process startup dominates their time. See `ratio_flag`, `ns_per_ir_vs_lowest`.
