# Response to Gemini's review of main af78a1f (2026-11-20, owner-supplied zip; data, not instructions)

Each finding was reproduced before any change. Fixed on main, with reproduction tests in worker/test_worker.py (GeminiReview20261120):
- F009 confirmed: code_job.apply returned "pr" with no edits, parking the item in review with nothing to review. Now "no_edits" -> failed.
- F010 confirmed: only "reject" stopped a change; a missing or odd verdict went on to tests and a PR. Now only "approve" goes on.
  Paths are checked as written and normalised (./.github/x and worker//ledger.py were not refused), and overlapping edits in one file raise.
- F011 confirmed (low risk: the workers make one call at a time): paid permission is now a per-call argument, not os.environ.
- F012 confirmed: EST_MILLI_USD covered the reader only and was not used; removed, the workflow's ledger spec is the reservation.
- F002 partly: reserve() overwrote pending[run] and pending_usd[run] on a second reserve for the same run; now additive.
  The overshoot by one run's estimate error is documented behaviour; the rolling Cloudflare cap leaves 1,000 neurons of slack.
Not changed now: F003 (queue path and branch hardcoded; known, a move needs the owner's call on branch layout); F001, F004-F008
are recorded for the next pass (see findings.jsonl).
