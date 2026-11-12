<!-- cloudflare @cf/mistralai/mistral-small-3.1-24b-instruct status=200 latency=16.06s -->

1. **Ambiguous Rule for `provider_error`:**
   - **Line:** `if upstream_bad and not passed: return "provider_error"`
   - **Issue:** If a provider error occurs but the attempt passes, the outcome is ambiguous. The rule does not specify what happens in this case.
   - **Example:** An attempt fails due to a provider error but passes the hidden test.
   - **Fix:** Add a condition to handle the case where the attempt passes despite a provider error.

2. **Disagreement Between Prose and Code on `provider_error`:**
   - **Line:** `if upstream_bad and not passed: return "provider_error"`
   - **Issue:** The prose states that a pass despite a provider error still counts, but the code does not reflect this.
   - **Example:** An attempt passes the hidden test but encounters a provider error.
   - **Fix:** Ensure the code allows a pass despite a provider error.

3. **Unfair Penalization by the Seal Check:**
   - **Line:** `if not seal["ok"]: meter.terminate(); run["ending"] = "seal_failed"; ...`
   - **Issue:** The seal check can fail due to transient network issues, unfairly penalizing a harness.
   - **Example:** A harness is penalized for a temporary network glitch.
   - **Fix:** Add retries or a more robust network check to the seal verification.

4. **Statistical Mistake in Fisher Test:**
   - **Line:** `def fisher_less(a, n1, b, n2):`
   - **Issue:** The Fisher exact test is one-sided but should be two-sided to avoid bias.
   - **Example:** A harness with a slightly lower pass rate might be incorrectly labeled as "worse."
   - **Fix:** Use a two-sided Fisher exact test to ensure fairness.

5. **Unfair Advantage Due to Model Name Rewriting:**
   - **Line:** `req["model"] = MODEL`
   - **Issue:** Rewriting model names can change the behavior of harnesses that rely on specific models for side calls.
   - **Example:** A harness that uses a different model for summaries gets the same model, altering its performance.
   - **Fix:** Allow harnesses to specify different models for side calls without rewriting.

6. **Potential Void Run Due to Unchecked Control Failures:**
   - **Line:** `if any(c["passed"] for c in controls):`
   - **Issue:** The control must fail all three tasks, but the code does not explicitly check this.
   - **Example:** A control fails only two tasks, voiding the run.
   - **Fix:** Explicitly check that the control fails all three tasks before proceeding.

7. **Ambiguous Rule for `harness_crash`:**
   - **Line:** `if not run or run.get("ending") in (None, "start_failed"): return "harness_crash"`
   - **Issue:** The rule does not specify what constitutes a "harness crash," leading to potential ambiguity.
   - **Example:** A harness fails to start due to a missing dependency, but the rule does not cover this case.
   - **Fix:** Define clear criteria for what constitutes a "harness crash."

8. **Potential Bias in the Switch Rule:**
   - **Line:** `if v["cheaper"] is True and not v["worse"] and v["token_ratio"] is not None and v["token_ratio"] <= 1 / 3:`
   - **Issue:** The switch rule favors harnesses that use at most 1/3 of the base's tokens per pass, which may not always be the most efficient.
   - **Example:** A harness that uses 34% of the base's tokens per pass is not considered, even if it is more efficient in other ways.
   - **Fix:** Consider a range of token ratios rather than a fixed threshold.
