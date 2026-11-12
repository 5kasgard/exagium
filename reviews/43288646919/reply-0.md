<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=65.34s -->

**Finding 1 – Seal‑check is optional in the code but mandatory in the protocol**  
*Quote (protocol):* “Before each attempt, a probe … must reach the meter and be refused by 1.1.1.1 and github.com, **or the attempt doesn’t start**.”  
*Quote (code – attempt.py):*  

```python
if os.environ.get("BENCH4_SEAL_CHECK") == "1":
    …  # run the probe, abort on failure
```

If the environment variable is not set (the default), the probe is skipped and the attempt runs even when the container can reach the outside world.  
*Concrete problem:* A harness that silently contacts an external API will be allowed to run, giving it an unfair advantage (fewer “refused‑connection” penalties) while the paper claims every run is sealed.  
*Fix:* Make the seal check unconditional (`if True:`) or abort the run when `BENCH4_SEAL_CHECK` is missing, and raise an error if the probe fails.

---

**Finding 2 – `calls` field never populated, breaking all cost calculations**  
*Quote (status code):*  

```python
tok = lambda r: r["tokens"]["total"]
cal = lambda r: r["calls"]
…
v["tokens_per_pass"] = round(per_pass(rs, tok))
v["calls_per_pass"]   = round(per_pass(rs, cal), 1)
```

*Quote (attempt.py):*  

```python
run["meter"] = c          # c is {"calls":…, "refused":…}
…
json.dump(run, …)
```

The result JSON stores the call count inside `run["meter"]["calls"]`, **not** at the top‑level key `calls`. Consequently `r["calls"]` raises a `KeyError` (or evaluates as `None` → `0`) and every “calls per pass” and every bootstrap ratio that uses `calls` is computed from zeros, inflating the “cheaper” signal.  
*Concrete problem:* An arm that actually uses many calls will appear to have `calls_per_pass = 0`, possibly qualifying it for a router switch that it does not deserve.  
*Fix:* In `attempt.py` add `run["calls"] = c.get("calls", 0)` (and similarly for `refused`) before writing the JSON, or change the lambdas to read `r["meter"]["calls"]`.

---

**Finding 3 – `start_failed` ending is undocumented and un‑handled**  
*Quote (protocol ENDS list):* `["control", "skipped_budget", "seal_failed", "harness_crash", "provider_error", "call_cap", "timeout", "exited"]`  
*Quote (attempt.py, final assignment):*  

```python
run["ending"] = ending or ("exited" if run["harness_rc"] is not None else "start_failed")
```

`"start_failed"` never appears in `ENDS`, never in `COUNTED`, and `classify()` has no branch for it. When a container fails to start, the router will treat the result as `"harness_crash"` (the default fallback) or may crash when later code expects `run["end"]`.  
*Concrete problem:* A failing harness is silently counted as a crash rather than a distinct “start_failed” case, skewing the failure statistics and possibly triggering the futility stop incorrectly.  
*Fix:* Add `"start_failed"` to `ENDS` (and to `COUNTED` if it should be counted) and handle it explicitly in `classify()`.

---

**Finding 4 – Ambiguous classification when a provider error occurs together with a normal ending**  
*Quote (classify function):*  

```python
upstream_bad = any(isinstance(c.get("status"), int) and c["status"] != 200 for c in calls)
if upstream_bad and not passed:
    return "provider_error"
return {"call_cap": "call_cap", "timeout": "timeout", "exited": "exited"}.get(run["ending"], "harness_crash")
```

If a harness receives a non‑200 response **and** later exits cleanly (`run["ending"] == "exited"`), the function returns `"provider_error"` (first clause) **and** `"exited"` is a possible label in the ENDS list. The “first match wins” comment is violated because two different endings can both apply to the same attempt.  
*Concrete problem:* The same attempt could be counted both as a provider failure (penalised) and as a successful exit (rewarded), depending on the order of evaluation in downstream code, leading to inconsistent metrics.  
*Fix:* Decide a single precedence (e.g., keep `"provider_error"` and map any `"exited"` with upstream errors to `"provider_error"` explicitly) and remove `"exited"` from the fallback mapping for runs that have `upstream_bad`.

---

**Finding 5 – Bootstrap ratio treats an arm with zero passes as infinitely cheap**  
*Quote (boot_ratio):*  

```python
a = per_pass(sx, key)   # returns math.inf if passes == 0
b = per_pass(sb, key)
vals.append(math.inf if math.isinf(a) else (0.0 if math.isinf(b) else a / b))
```

When the **base** arm has zero passes (`b == math.inf`), the code stores `0.0` for the ratio, implying the test arm is *cheaper* even though no comparison is possible. The 95 % interval will then be `[0.0, 0.0]`, automatically satisfying `th < 1.0` and possibly triggering a router switch.  
*Concrete problem:* An arm that never solves any task can be declared “cheaper” simply because the base never solved, violating the study’s “without solving fewer” requirement.  
*Fix:* If either arm has `passes == 0`, set `token_ratio` to `None` and mark `cheaper = False` (or “unmeasured”) instead of forcing a numeric value.

---

**Finding 6 – Futility stop uses *all* rows, not only counted attempts**  
*Quote (futile function):*  

```python
rs = arm_rows(done, arm)
return arm != BASE and len(rs) >= FUTILITY_N and not any(r["passed"] for r in rs[:FUTILITY_N])
```

`arm_rows` returns **every** result for the arm, including `provider_error`, `seal_failed`, and other non‑counted endings. The protocol specifies “no pass in its first 6 **counted** attempts (repeats 1 and 2 of each task)”. Including non‑counted attempts can cause the futility condition to fire earlier than intended.  
*Concrete problem:* An arm that actually made six counted attempts with zero passes but also generated three `provider_error`s will have `len(rs) = 9`; the slice `rs[:6]` will contain some `provider_error`s, so the `any(r["passed"] ...)` check may see fewer than six counted attempts, potentially **preventing** futility when it should trigger, or vice‑versa.  
*Fix:* Filter `rs` to only counted endings (`r["end"] in COUNTED`) before applying the futility logic.

---

**Finding 7 – `call_cap` ending detection can be bypassed, leading to ambiguous outcomes**  
*Quote (attempt.py loop):*  

```python
if c.get("refused") and cap_seen is None:
    cap_seen = time.time()
if cap_seen and time.time() - cap_seen > 30:
    ending = "call_cap"
elif time.time() - t0 > WALL:
    ending = "timeout"
```

The code only declares `"call_cap"` **after** a 30 s grace period. If the container exits (or is killed) **before** those 30 s elapse, `ending` stays `None` and later falls back to `"exited"` (or `"timeout"`). Thus an attempt that actually hit the call cap can be recorded as `"exited"` instead of `"call_cap"`.  
*Concrete problem:* Two arms that both reach the 80‑call limit will be scored differently depending on whether they happen to finish quickly after the refusal, biasing the counted attempts and the token‑per‑pass metric.  
*Fix:* Immediately set `ending = "call_cap"` when `c.get("refused")` becomes true, and only use the 30 s timer to decide whether to kill the container, not to decide the label.

---

**Finding 8 – `plan()` computes `open_n` (remaining tasks) with `remaining = 0` in `status()`, making “complete” detection unreliable**  
*Quote (status):*  

```python
_, open_n = plan(results, 0)
s["complete"] = open_n == 0 and len(controls) == len(TASKS)
```

`plan()` expects `remaining` to be the number of free model calls left, but `status()` always passes `0`. Consequently `open_n` is computed as if no calls are left, which will almost always be non‑zero, causing `complete` to be `False` even when all attempts have finished. This prevents the router from ever switching, regardless of the statistical evidence.  
*Concrete problem:* The benchmark will never reach a “complete” state, making the switch rule moot and violating the experiment’s design.  
*Fix:* Pass the actual remaining call budget (e.g., read from the meter via `/meter/count`) to `plan()` when building the status, or redesign `complete` logic to base it on the absence of pending todo items.
