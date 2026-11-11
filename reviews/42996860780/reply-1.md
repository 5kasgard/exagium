<!-- cloudflare @cf/meta/llama-4-scout-17b-16e-instruct status=200 latency=25.07s -->

Here are 8 findings:

1. **Ambiguous rule**: 
   - Line: `if status.endswith("FINISHED") and not err and not codes:`
   - What goes wrong: An attempt can be both `finished` and have a `provider_error` that was not detected earlier, because `err` and `codes` only check a limited set of error sources. For example, if a `provider_error` occurs but does not result in an error message or code that matches the conditions for `provider_error`, it will be counted as `finished`.
   - Fix: Move `if PROVIDER.search(err + ...)` above `if status.endswith("FINISHED")`.

2. **Prose and code disagree**: 
   - Line (prose): `Counted attempts are endings 3, 4, 6, 7, 8 and 9.` 
   - Line (code): `COUNTED = {"timeout", "harness_crash", "iteration_cap", "stuck", "finished", "unclassified"}`
   - What goes wrong: The prose lists counted attempts as endings 3, 4, 6, 7, 8, and 9, but the code does not include `control`. However, the prose states that `control` is not counted. For example, an attempt ending in `control` will not be counted according to the prose but is not explicitly excluded in the code.
   - Fix: Ensure consistency by documenting that `control` is not counted.

3. **Result misreading**: 
   - Line: `s["arms"][a] = {"model": ARMS[a][0], "clean_passes": k, "n": len(rs), "low": round(lo, 3), "high": round(hi, 3), ...}`
   - What goes wrong: The interval (`lo`, `hi`) is calculated based on clean passes (`k`) over counted attempts (`len(rs)`), but it is not clear if `clean_passes` or `n` should be used for calculating `calls_per_clean_pass`. For example, if there are multiple attempts with different numbers of clean passes, using `n` instead of `k` could lead to incorrect calculations.
   - Fix: Clarify that `clean_passes` should be used.

4. **Statistical mistake**: 
   - Line: `return "holds" if 9 * k >= 7 * n else "fails" if 9 * k <= 4 * n else "inconclusive"`
   - What goes wrong: The comparison is done using integer arithmetic, which can lead to incorrect results if `k` or `n` are large. For example, if `k` is large and `n` is small, the comparison may not work as expected.
   - Fix: Ensure that the comparison is done using floating-point arithmetic.

5. **Void run**: 
   - Line: `if len(controls) == len(TASKS) and not any(c["passed"] for c in controls)`
   - What goes wrong: The run is only voided if all controls fail, but it does not check if there are any attempts that did not complete (e.g., due to `provider_cut`). For example, if there are attempts that were cut due to provider errors, the run may not be voided even if the controls pass.
   - Fix: Add a check for `provider_cut` attempts.

6. **Ambiguous rule**: 
   - Line: `if r.get("green_kept")`
   - What goes wrong: The rule for `stop_on_green` uses `green_kept` which may not be set for all attempts. For example, if an attempt does not have a `green_kept` value, it will not be considered when calculating `kept`.
   - Fix: Ensure that `green_kept` is set for all attempts.

7. **Result misreading**: 
   - Line: `s["ranking"] = [f"{x} above {y}" for x in A for y in A if A[x]["n"] and A[y]["n"] and A[x]["low"] > A[y]["high"]]`
   - What goes wrong: The ranking only considers arms with a significant difference in their intervals. For example, if two arms have overlapping intervals, they will not be ranked.
   - Fix: Clarify that the ranking only considers significant differences.

8. **Statistical mistake**: 
   - Line: `med = statistics.median(saving) if saving else None`
   - What goes wrong: The median is calculated only for attempts with a `saving` value. For example, if there are attempts with no `saving` value, they will not be considered when calculating the median.
   - Fix: Ensure that all attempts are considered when calculating the median.
