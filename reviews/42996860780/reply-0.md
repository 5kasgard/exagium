<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=44.37s -->

**1. KeyError in the seal check**  
*Quote* (line ≈ 115 of `status`):  

```python
"seal": "holds" if not any(r["outside_paths"] for r in counted) else "broken",
```  

*What goes wrong* – `outside_paths` is only added when the runner detects an outbound host path. For the majority of attempts the field is absent, so `r["outside_paths"]` raises a `KeyError` and the whole status‑generation crashes, making the whole benchmark void.  

*Fix* – use `r.get("outside_paths", [])` (or check with `in` before indexing) so missing fields are treated as empty.  

---

**2. Control‑failure rule is not enforced**  
*Quote* (line ≈ 106):  

```python
"control_ok": len(controls) == len(TASKS) and not any(c["passed"] for c in controls),
```  

*What goes wrong* – the specification says “the no‑agent control Z runs once per task and **must fail all 3**, or the run is void.” The code only marks the run *incomplete* when a control is missing or passes; it never produces a “void” verdict. An experiment in which the control is omitted (e.g., due to a provider cut) will silently be treated as “still running” instead of being declared invalid.  

*Fix* – add a separate `void` flag (e.g., `"void": any(c["passed"] for c in controls) or any(c is None for c in controls)`) and make the overall status `complete` false when `void` is true.  

---

**3. Futility‑stop ordering is not chronological**  
*Quote* (function `futile`):  

```python
return len(rs) >= FUTILITY_N and not any(r["passed"] and r["clean"] for r in rs[:FUTILITY_N])
```  

*What goes wrong* – `rs` is built from `sorted(done.items())`, which orders cells by `(task, arm, rep)`, not by the real time each attempt finished. An arm could have a clean pass in its first chronological attempt but appear later in the sorted list, preventing the futility stop and inflating its count.  

*Fix* – store a timestamp for each result and sort by that timestamp (or preserve insertion order) before applying the futility slice.  

---

**4. Provider‑error calls are omitted from the budget report**  
*Quote* (budget calculation):  

```python
calls = [r["calls"] for r in counted if r.get("calls") is not None]
s["budget"] = {"median_calls": statistics.median(calls) if calls else None,
               "total_calls": sum(calls)}
```  

*What goes wrong* – attempts that end in `provider_error` are **not** in `counted`, yet they still consume model‑call quota (the provider still charges for the request). The reported median/total calls therefore under‑estimate the true budget, allowing a run to appear cheaper than it actually is.  

*Fix* – include provider‑error attempts in the call aggregation (e.g., `all_calls = [r["calls"] for r in done.values() if r.get("calls")]`).  

---

**5. Extra “minimum‑n” guard in H1″ share‑verdict**  
*Quote* (function `share_verdict`):  

```python
if n < min_n:
    return "inconclusive"
```  

*What goes wrong* – the written rule for H1″ never mentions a minimum sample size; it should decide “holds/fails” purely on the 7/9 and 4/9 thresholds. With `min_n=5`, a run with 4 counted attempts that meets the 7/9 ratio (e.g., 3/4 = 75 % > 7/9) will be forced to “inconclusive”, contradicting the specification and giving an attacker a way to hide a strong result by simply running fewer attempts.  

*Fix* – remove the `min_n` check (or set `min_n=0`) so the verdict is based only on the ratio.  

---

**6. Ambiguous classification when a timeout also matches a provider error**  
*Quote* (in `classify`):  

```python
if "rc=124" in err or "rc=137" in err:
    return "timeout"
...
if PROVIDER.search(err + " " + " ".join(c + " " + str(e.get("detail")) for c, e in zip(codes, errs))):
    return "provider_error"
```  

*What goes wrong* – a provider‑rate‑limit response can include “rc=124” (e.g., a Docker‑style timeout wrapper) **and** the provider‑error pattern. Because the timeout check appears first, the attempt is labelled `timeout` and counted as a failure, even though the underlying cause is a provider cut that should be retried and excluded from the denominator.  

*Fix* – test for `PROVIDER` **before** the timeout check, or require that the timeout pattern is not also matched by `PROVIDER`.  

---

**7. “unclassified” is always treated as a crash**  
*Quote* (status construction):  

```python
ok = sum(1 for r in counted if r["end"] not in CRASH)
s["harness"] = {"no_crash": ok, "n": len(counted), "verdict": share_verdict(ok, len(counted))}
```  

*What goes wrong* – `CRASH` is defined as `{"harness_crash", "unclassified"}`. An `unclassified` result may be a benign non‑crash failure (e.g., a missing hidden test file) but is counted as a crash, inflating the crash rate and potentially causing H1″ to “fail” unfairly.  

*Fix* – split `unclassified` into a separate category (e.g., `UNKNOWN`) and exclude it from `CRASH`; treat it as a normal failure in the harness metric.  

---

**8. Batch planning can exceed the declared maximum agents**  
*Quote* (function `plan`):  

```python
n_agents = max(0, min(BATCH_MAX, (remaining - MARGIN) // CAP))
...
for c in todo:
    if c["arm"] == "Z":
        out.append(c)
    elif used < n_agents:
        out.append(c); used += 1
```

*What goes wrong* – control tasks (`arm == "Z"`) are added **unconditionally**, regardless of `n_agents`. If `BATCH_MAX` is 12 but the remaining budget allows only 2 agents, the batch will still contain all 3 control attempts plus up to 2 regular attempts, exceeding the intended per‑batch limit and potentially triggering provider‑rate‑limit cuts that invalidate the run.  

*Fix* – treat controls as part of the same `n_agents` budget (e.g., count them toward `used`) or enforce a separate hard cap on total tasks per batch.  
