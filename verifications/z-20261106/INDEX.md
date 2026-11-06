# Z's verification, 2026-11-06

An outside verifier (GLM-5.3 on "Z", a kata microVM on Intel Granite Rapids) ran `VERIFY.md` against `main` at `d801cea`. These are its files as received: `report.md` is the report, and `plan1-verdict-proposal.json` is its verdict on plan 1, which became ledger entry 4 unchanged.

The verdict names its evidence by label. Here is each label's file and sha256; anyone can recompute them with `sha256sum artifacts/*`:

| Evidence name | File |
|---|---|
| clean-checks-2026-11-06 | `artifacts/clean-checks-2026-11-06.txt` |
| monitor-clean-2026-11-06 | `artifacts/monitor-clean-2026-11-06.json` |
| check-anchors-rewrite-attack | `artifacts/check-anchors-rewrite-attack.json` |
| monitor-laundering-attack | `artifacts/monitor-laundering-attack.json` |
| plan2-remeasure | `artifacts/plan2-remeasure.json` |
| jail-analysis | `artifacts/jail-analysis.txt` |
| probe-full-seed1 | `artifacts/full.json` |
| probe-full-seed7 | `artifacts/full7.json` |
| probe-full-seed99 | `artifacts/full99.json` |

**Checked here before ingesting** (2026-11-06):
- Every sha256 above matches its file.
- The plan sha256 matches `ledger/entries/000001.json`, and the body passes the ledger's rules.
- The one new weakness reproduces: deleting `ledger/anchors/000003.ots` leaves `verify` and `check-anchors` at exit 0. It is fixed after entry 4; see `record/README.md`.
- The plan 2 re-measurement matches `loop/results/plan2/` exactly in accuracy and predictions for all five candidates. Ir is within 0.5% for four of them; linear is 10% higher on Z's gcc build.
