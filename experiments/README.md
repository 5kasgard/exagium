# Experiments: measurements with a rule fixed before they run (owner 2026-11-21: the queue's "test" items had no executor)

One file per experiment, `experiments/<name>.py`, defining:
- `PLAN`: a string. What is measured, on what, and the rule, in words.
- `run()`: does the measurement and returns a JSON-able dict. Reads the repository; writes nothing outside its own
  temporary folders; no secrets are available to it.
- `rule(result)`: returns `{"verdict": "holds" | "fails" | "invalid", "why": "..."}` by script, from the result alone.

The rule is fixed before the run because it is in the file the run executes, and its commit time predates the result
(the record lists the file's commit and the run). Change the rule after a result and the record shows it.

Run: `.github/workflows/experiment.yml` with `name` (and `item`, the queue item it answers). The run job has no secrets
and a read-only token; the record job runs no experiment code, saves the result to router-runs
`experiments/<name>/<run>/`, and moves the item: holds or fails -> done (a result either way), invalid -> escalated, no
result -> open (counted as a failure; twice escalates).
Locally: `python3 -I experiments/run_one.py NAME OUT_DIR`.
