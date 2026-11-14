<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=42.98s -->

**1. Capacity‑refusal definition contradicts the written rule**  
*Quote (PREREG):* “A server fault (5xx, timeout) is **not** a refusal …”  
*Quote (code, `REFUSAL` regex):* `...|Timeout|timed out...`  

**What goes wrong** – If an attempt ends with a timeout (e.g. a 504 Gateway Timeout that the SDK does not retry), `capacity()` will count it as a refusal and return **fails**, even though the pre‑registration states that time‑outs should be ignored. A run that would otherwise satisfy C (≥21 attempts, same UTC day) could be declared a failure simply because one call timed‑out.

**Fix** – Remove the “Timeout|timed out” alternatives from `REFUSAL` (or move them to a separate “server‑fault” regex that is ignored for C).

---

**2. `run_id` must be numeric but the code assumes it**  
*Quote (code, `capacity()`):* `first = min(agents, key=lambda r: int(r.get("run_id") or 0))["run_id"]`  

**What goes wrong** – `run_id` is a string supplied by GitHub (`"${GITHUB_RUN_ID}"`). If the repository ever changes to a non‑numeric identifier (e.g. a UUID) the `int()` conversion raises `ValueError` and the whole status generation crashes, rendering the benchmark unusable.

**Fix** – Compare run IDs as strings (lexicographically) or store a separate numeric batch counter; e.g. `first = min(agents, key=lambda r: r.get("run_id"))`.

---

**3. Capacity ignores attempts marked “skipped_budget”**  
*Quote (PREREG):* “C reads **every** agent attempt in the first batch, counted or not.”  
*Quote (code, `agents` filter):* `... and r.get("end") != "skipped_budget"`  

**What goes wrong** – If the provider returns a “budget‑exhausted” error that the SDK records as `skipped_budget`, that attempt is omitted from the batch size check. The batch could contain fewer than 21 *actual* attempts but still be judged “holds” because the skipped ones are invisible, inflating the apparent capacity.

**Fix** – Remove the `r.get("end") != "skipped_budget"` clause so that *all* attempts, even those that hit a budget limit, are counted for C.

---

**4. Mismatched refusal detection between `classify` and `capacity`**  
*Quote (code, `classify`):* uses `EXTRA_PROVIDER` **after** the base `b3.classify`.  
*Quote (code, `capacity`):* uses `REFUSAL` regex to spot refusals.  

**What goes wrong** – An attempt that triggers a credit‑error (`402 … insufficient credits`) will be labelled `provider_error` by `classify` (because `EXTRA_PROVIDER` matches) but **may not** be recognised as a refusal by `capacity` if the `_err` field does not contain the exact pattern used in `REFUSAL`. Consequently the batch could be declared “holds” even though a refusal occurred, letting a run pass the C rule while the human‑readable label says otherwise.

**Fix** – Consolidate the two patterns: define a single `REFUSAL` regex that includes the extra credit patterns, and have `classify` rely on the same regex (or share the compiled pattern).

---

**5. Parallelism is not validated**  
*Quote (PREREG):* “4 attempts in parallel … stays under 40 req/min unless calls take < 6 s.”  
*Quote (code, `capacity`):* `parallel = max((r.get("parallel") or 0) for r in b)` – no check against the 4‑agent limit.

**What goes wrong** – A malicious or buggy run could report `parallel: 10` in its metadata while actually launching only 4 agents. The capacity rule would record a higher parallel value (perhaps satisfying a hidden “≥4” requirement) and the benchmark would proceed, even though the real concurrency never exceeded the allowed limit. Conversely, a runner could launch 10 agents, claim `parallel: 4`, and still pass C.

**Fix** – Add a guard in `capacity()` (or in the workflow) that verifies `parallel <= 4` and fails the batch otherwise.

---

**6. `compare()` treats “A has zero attempts” as inconclusive, but the prose requires A to have at least one counted attempt for a hold**  
*Quote (PREREG):* “Holds if … A has at least one counted attempt …”  
*Quote (code, `compare()`):* `elif lo >= 0.5 and a_n > 0: v = "holds"` – otherwise falls to “inconclusive”.

**What goes wrong** – If arm A never produced any counted attempts (e.g., all were provider cuts), the code returns “inconclusive”. The written rule would also reject a hold, but the wording “at least one counted attempt” could be interpreted as a hard failure rather than “inconclusive”. Down‑stream logic (`second_pool`) treats any non‑“holds” as “undecided”, potentially leaving the benchmark in limbo even when the data clearly show N is usable.

**Fix** – Make the intention explicit: change the “inconclusive” branch to `"fails"` when `a_n == 0` (or add a comment clarifying that “inconclusive” is the intended outcome).

---

**7. Void detection does not enforce a *single* control per task**  
*Quote (PREREG):* “The no‑agent control Z runs **once per task** and must fail all 3, or the run is void.”  
*Quote (code, `status()`):* `void = any(c["passed"] for c in controls)` – only checks if *any* control passed.

**What goes wrong** – A runner could submit **multiple** Z controls for a task, making one of them pass while the others fail. The current check would flag the run as void, but it would not reveal that extra controls were submitted, which could be used to inflate the “open cells” count or to hide failures in the primary control.

**Fix** – Verify that exactly one control exists per task: `if len([c for c in controls if c["task"]==t]) != 1: void=True`.

---

**8. Plan does not guarantee that the first batch starts **after** the no‑agent controls**  
*Quote (PREREG):* “The no‑agent control Z runs once per task and must fail …” (implies it should be part of the first batch).  
*Quote (code, `plan()`):* Controls are added to `todo` first, but the workflow may skip them if `MODE != pilot` and `REGISTERED` is missing, or if the runner aborts early. Nothing in `capacity()` checks that the batch used for C actually contains the three Z controls.

**What goes wrong** – An attacker could delete or omit the Z controls from the first batch, run 21 N attempts, and have C return “holds”. The benchmark would then accept N as a second pool even though the required control verification never occurred.

**Fix** – In `capacity()` require that the batch contains three control attempts (one per task) and that all of them have `end == "control"` and `passed == False`; otherwise return `"fails"` or `"inconclusive"`.

--- 

These eight points expose the most serious gaps between the pre‑registration narrative, the implementation, and the statistical/operational safeguards. Addressing each with the suggested one‑line fix will make the benchmark run unambiguous, resistant to gaming, and faithful to the documented intent.
