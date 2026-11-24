# 2026-11-24 hardening handoff

## Done in this session

No changes were merged to `main`. Three isolated PRs were opened and their GitHub Actions test suites passed:

- [#106](https://github.com/5kasgard/exagium/pull/106): automated tier now binds merge to the exact reviewed PR head SHA, rechecks that the PR remains open/non-draft, refreshes the queue before merge, and requires the queue item to still authorise that PR. Regression tests cover changed heads and queue cancellation. Actions run 184 passed.
- [#108](https://github.com/5kasgard/exagium/pull/108): usage recording stores a digest per run in the same commit as usage totals and `calls.jsonl`; exact retries are no-ops and conflicting payloads are refused. Actions run 185 passed.
- [#110](https://github.com/5kasgard/exagium/pull/110): stale OpenTimestamps explorer tips now produce `CantCheck`, not a false forgery verdict. Regression tests cover stale, missing and malformed tips. Actions run 186 passed.

## Open follow-ups

- [#109](https://github.com/5kasgard/exagium/issues/109): workflow-only fixes for experiment handoff validation/persistence, transfer pre-registration enforcement, counter false-green outcomes, stable reservation-operation IDs, and a whole-audit keeper fallback. These files were not changed because this integration is blocked from writing `.github/workflows/*`.
- [#111](https://github.com/5kasgard/exagium/issues/111): legacy loop / runner fallback resource containment. Includes stdout growth, timeout orphaning, confine-mode limits and cumulative disk quota.

## Important limitations

- The ledger's existing F002 test requires same-run/same-pool reservations to remain additive. PR #108 intentionally does not change this. A stable reservation-operation ID must be threaded through the workflow caller before reserve-side replay can be made idempotent safely.
- No live ambiguous-push, PR-head-race, or stale-explorer network fault injection was run. The three PRs have deterministic regression coverage and passing CI, not full adversarial integration proof.
- The repository's `review.yml` requires a manual workflow dispatch for an independent free-model review. The available GitHub integration exposed no workflow-dispatch action, so independent review remains outstanding. Do not merge these PRs until that review is recorded and its findings are checked.
- No queue branch changes were made. These are PRs/issues, not completed queue items.

## Next steps, in order

1. Run independent reviews of #106, #108 and #110 through `review.yml`; verify each finding against the source and tests.
2. Run targeted fault injection for ambiguous ledger pushes and PR head changes between review and merge.
3. Owner applies #109 workflow changes, preserving F002 additive reservation semantics with a stable operation ID.
4. Implement #111's sandbox fault-injection suite before changing resource limits; every new cap must be tied to measured workload requirements.
5. Reconcile the duplicate R10 PR lifecycle and queue lineage after inspecting both PR diffs; do not close or merge either blindly.
