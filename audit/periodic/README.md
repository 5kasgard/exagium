# Whole-system auditor: weekly indicators (queue P10)

`indicators.py` measures debt and blind-spot signals across the whole repo, mechanically. Each indicator says how it is computed. Heuristics list what they flagged, so each flag can be checked. There are no fixed thresholds yet: the first run is the baseline, and later runs report the change against the previous one (`--previous`).

The planned design (queue P10):
- These indicators run weekly on CI. That workflow waits on Q07.
- A direction check by free models of another family compares the completed work with the README's "What this is for".
- A deeper audit by Claude agents runs monthly, or when an indicator trips.
- Reports go straight to the owner as a GitHub issue. They are not routed through the builders.

## Baseline, 2026-11-18 (`2026-11-18-baseline.json`, main at c8d4747)
- **Duplication:** 313 of 7,225 Python function lines (4.3%) are exact structural duplicates. Most are the frozen plan-2 candidate folders, plus the new runner label candidates.
- **Unreferenced code files:** 5, mostly one-off verifier and probe scripts. `router/analysis/design_variance.py` needs a look.
- **Workflow lines that swallow a failure:** 31, mostly in the parked bench workflows.
- **CI over the last 7 days:** router 7 of 16 runs not successful (most before the 2026-11-18 fix); runner 2 of 3 (jail work that failed closed).
- **README lines with a number and no evidence pointer:** 3.
- **Code churn over the last 7 days, by area:** about 7,400 lines in measuring and recording (loop, record, probe, counter), 2,600 in router, and 2,300 in runner, transfer and tasks, the parts that build and test AI. That is about one fifth on the project's core purpose. A direction signal to watch, not a verdict.
- **Not yet measured:** test coverage, lint warnings and dependency freshness. Each needs a tool installed.

**Review:** a free outside reviewer, gpt-oss-120b (Cloudflare; router-runs `reviews/44260921972`), raised 11 points.
- **Applied:** docstrings ignored in the duplicate check; exact 7-day window; full line counts; links counted as evidence; more fields in the weekly change; a churn indicator.
- **Rejected with evidence:**
  - **Shell quoting:** the commands are argument lists, not shell strings. The 0-day ages were a timezone bug, fixed separately.
  - **Matches inside longer names:** the lookarounds prevent them.
  - **Offset-less timestamps:** ISO-strict git dates always carry an offset.
  - **"Bare integers" in the claims check:** the pattern requires units.
  - **Exclusive output and failing on missing input:** both are deliberate. An audit record is never silently overwritten.

## Flow and connectors (added 2026-11-19)
- **flow_7d:** items that arrived and closed in the last 7 days, read from every version of the queue file; arrivals split by `origin` (`follow-up`: spawned by finishing or checking work; `new`: a new direction); median hours from first appearance to done; and Little's law, open items divided by closures per day, as an estimate of how long an item waits. Growth is expected (finishing work well spawns follow-ups); the warning signs are the waiting estimate rising week on week and new directions outpacing follow-ups. No thresholds until a few weeks of data set them. First reading (2026-11-19): 86 arrived, 37 closed, about 6 days' wait; the median of 0.6 hours to done is pulled down by audit items fixed within the hour they were filed.
- **connectors:** `connectors.json` lists each connector and the evidence that a real part has gone through it end to end; the script counts one as used only when every evidence path exists and is not empty. First reading: 5 of 11.
