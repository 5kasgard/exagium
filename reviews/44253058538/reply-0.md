<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=95.21s -->

**Adversarial review of the general runner (runner/run.py) and its supporting files**  

Below each defect I give:  

* **Location** – file and function (or line range).  
* **What is wrong** – precise failure mode.  
* **Concrete input / scenario** that triggers it.  
* **Why it matters** – security, correctness, or reproducibility impact.  
* **Fix** – minimal change that eliminates the problem.  
* **Severity** – **Critical**, **Major**, or **Minor**.

---

## 1. Untrusted‑bytes paths can break the sandbox, corrupt scores or cause “success‑by‑crash”

| # | Location | What is wrong |
|---|----------|---------------|
| 1.1 | `runner/run.py: timed()` – the call to `L.jail([...], [(w, "/w", False)], 1800, seccomp=not sc)` | The `seccomp=` flag is inverted. When the runner is in *confine* mode (`L.JAIL["mode"] == "confine"`), `sc = ["--seccomp"]` is set, but `seccomp=not sc` evaluates to **False**, disabling the seccomp filter for the whole inference run. In *bwrap* mode the flag is True, enabling the filter, which is the opposite of the intended policy. |
| 1.2 | `runner/run.py: timed()` – parsing of the timer output | The runner extracts the numeric fields by splitting on the literal string `b"\nexagium-timer "`. A malicious candidate can prepend that exact marker to its *stdout* (e.g. `printf("exagium-timer 0 0 0 -1\n");`) and then emit any payload afterwards. The parser will treat the payload as the “output bytes” (`out`) **without any length validation**, allowing the candidate to inject arbitrarily large or malformed record streams that later bypass `records.unpack` (because the parser already split on the last newline). This can lead to: <br>• Incorrect scores (the checker receives garbage that may happen to be parsed as a valid record). <br>• Crashes in the runner that are interpreted as a “failed candidate” but still count as a completed evaluation run. |
| 1.3 | `runner/run.py: count()` – use of `subprocess.run` with `cwd="/tmp"` in the *bwrap* branch | The `cwd` is set to a world‑writable directory (`/tmp`). In *bwrap* mode the jail mounts `/tmp` read‑only, but the `cwd` argument is passed **outside** the jail mount list, so the child process may escape the sandbox by opening `../proc/self/fd/…` or similar paths relative to the host `/tmp`. An adversarial candidate can therefore read arbitrary host files (e.g. `/etc/passwd`) and leak them into the timer output, corrupting the JSON log that the runner later trusts. |
| 1.4 | `runner/run.py: train()` – `L.jail([...], [(sb, "/w", True)], …)` | The training step is the **only** place where the temporary directory is mounted read‑write (`True`). The training script runs **untrusted** code (candidate’s `train.py`). If the candidate’s training code deliberately writes files *outside* `/w` (e.g. `open("/etc/hosts","a")`), the bwrap configuration does **not** forbid writes to the host filesystem because the mount point for the host root is not explicitly denied. In *confine* mode the seccomp filter is also disabled (see 1.1), so the training script can perform arbitrary syscalls. This gives the attacker full host access before any scoring happens. |

### Concrete exploit scenario
```bash
# Candidate's train.py (run inside the jail)
import os, sys
open("/tmp/leak", "wb").write(open("/etc/passwd","rb").read())
# The file /tmp/leak ends up on the host because /tmp is not confined.
```
The runner will still report “training succeeded” and continue to evaluation, leaking secret data.

### Why it matters
* **Security breach** – the whole premise of the runner is that candidate code is isolated. The inverted `seccomp` flag and the writable `/tmp` break isolation, allowing arbitrary file reads/writes and denial‑of‑service attacks.
* **Score integrity** – by forging the timer marker the candidate can manipulate the measured `ns` value or hide a crash, causing the leaderboard to accept a fabricated performance figure.

### Fix
```python
# In timed():
sc = ["--seccomp"] if L.JAIL["mode"] == "confine" else []
# Pass the flag correctly:
r = L.jail([...], [(w, "/w", False)], 1800, seccomp=bool(sc))

# Sanitize timer output: require the marker to be the *first* line.
out, sep, tail = r.stdout.partition(b"\nexagium-timer ")
if not sep or out == b"":
    raise Failed(...)

# Use a private per‑run temporary directory as cwd for count() and never mount /tmp.
with tempfile.TemporaryDirectory(prefix="exagium-rcnt-") as cwd:
    r = L.jail([...], [(tool, "/tool", False), (cwd, "/tmp", False)], …)
```
Also mount the host root read‑only (or not at all) and make the training mount read‑only, moving any needed write‑able area to a dedicated sandboxed directory.

**Severity:** **Critical** – breaks the core security model.

---

## 2. “No task‑specific code in run.py” is false; the runner makes hidden task assumptions

| # | Location | What is wrong |
|---|----------|---------------|
| 2.1 | `runner/run.py: cmd_evaluate()` – baseline & floor checks | The code assumes that `card["baselines"]` and `card["floor"]` are **candidate names** that exist in the `--cand` list. This couples the runner to the *naming* convention of a particular task and prevents tasks that use *different* identifiers (e.g., numeric IDs, versioned baselines). |
| 2.2 | `runner/run.py: check()` – `records.unpack(out, max_len=t["card"]["output"]["max_len"], count=n)` | The runner assumes a **single scalar `max_len`** for *all* output records. Some tasks (e.g., program synthesis) may have per‑item bounds that vary per instance. The contract explicitly allows “variable‑size inputs and outputs”, yet the runner enforces a global bound. |
| 2.3 | `runner/run.py: task()` – checker path resolution | It only supports two cases: built‑in label checker (`"label"`) or a path relative to the task directory. If a task wants to ship a *wrapper* script that selects a verifier based on the input, the runner will treat the wrapper as the checker but will not pass any extra arguments the wrapper may need, violating the contract. |
| 2.4 | `runner/run.py: mean_se()` – uses `statistics.fmean` and `statistics.stdev` without guarding against NaN or infinite values | If a candidate’s checker returns `nan` scores (e.g., due to a floating‑point bug), `statistics.fmean` propagates `nan` and the leaderboard will show `nan` as the mean, which is indistinguishable from a missing entry. The contract expects a numeric mean in `[0,1]`. |
| 2.5 | `runner/run.py: cmd_evaluate()` – hard‑coded assumption that every split has a *key* file (`{split}.key`) | Some verifier‑scored tasks (e.g., pure SAT) do not need a key file at all. The runner still copies the key file into the jail and passes it to the checker; if the key file is missing the runner crashes, violating the “no task‑specific code” promise. |

### Concrete scenario
A task `program-synthesis` defines `card["output"]["max_len"] = {"per_item": [10, 200, 50, …]}` and provides a checker that expects the runner to pass the per‑item bound as a third argument. The runner will raise `KeyError` when trying to read `t["card"]["output"]["max_len"]` as an int, aborting the entire evaluation.

### Why it matters
* **Incorrect rejections** – valid candidates for tasks with variable bounds are rejected as malformed.
* **Hidden coupling** – future tasks cannot be added without editing `run.py`, contradicting the contract and increasing maintenance burden.
* **Leaderboard contamination** – NaN scores silently propagate, making it impossible to compare candidates.

### Fix
* Replace the global `max_len` enforcement with a per‑item callback: if `t["card"]["output"]["max_len"]` is a dict with `"per_item"` key, pass the appropriate bound to `records.unpack` per record.
* Decouple baseline/floor validation from candidate names: require the card to list *baseline identifiers* and map them to candidate directories via a configuration file rather than name matching.
* Make the key file optional: only copy and pass it if it exists on disk.
* Guard `mean_se` against non‑numeric scores:
```python
def mean_se(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        raise Failed("no valid scores")
    m = statistics.fmean(xs)
    return m, (statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None)
```

**Severity:** **Major** – violates the core contract and leads to systematic mis‑evaluation.

---

## 3. Statistical / measurement errors

| # | Location | What is wrong |
|---|----------|---------------|
| 3.1 | `runner/run.py: mean_se()` – uses `statistics.stdev` (sample SD) on **per‑item scores** instead of **per‑run means** | The SE reported is the standard error of the *items* within a single run, not the standard error across the *R* repeated runs that the runner actually performs. This under‑estimates variance when the inference time fluctuates between runs, giving a false sense of precision. |
| 3.2 | `runner/run.py: cmd_evaluate()` – `ir_per_item = (ir_n - ir_0) / n` without checking `n > 0` | For a malformed card where `n == 0` (e.g., an empty split), a `ZeroDivisionError` is raised, aborting the whole evaluation. |
| 3.3 | `runner/run.py: cmd_evaluate()` – `ns_per_item = statistics.fmean((x - y) / n for x, y in zip(r["ns_n"], r["ns_0"]))` | If any `x - y` is negative (possible when the empty‑input timer overhead exceeds the timed run), the mean can become negative, which is nonsensical for a time metric and may be displayed as a negative number on the leaderboard. |
| 3.4 | `runner/run.py: cmd_evaluate()` – floor computation uses the *raw* scores of the floor baseline, but does **not** verify that the floor baseline itself passed the checker (i.e., that its `items` list has the correct length). If the floor baseline crashes, `statistics.fmean` will raise `ZeroDivisionError`. |
| 3.5 | `runner/run.py: digests()` – uses `sha_file(open(p, "rb").read())` without streaming; large files (> 100 MiB) could cause excessive memory use and potential OOM on the runner host. |

### Concrete scenario
Run the SAT task with `--rounds 1` (the default is 5, but a user could set `--rounds 1`). The SE will be `None` (as coded), but the leaderboard prints `0.0` (`r['se'] or 0`). Users may interpret this as “exact” measurement even though only a single run was performed.

### Why it matters
* **Misleading statistics** – users cannot reliably compare candidates because the reported SE does not reflect run‑to‑run variability.
* **Potential crashes** – division by zero on empty splits or malformed floor baselines can abort an entire evaluation batch.
* **Negative time values** corrupt the leaderboard and can be exploited to artificially improve a candidate’s “ns/item” rank.

### Fix
* Compute SE over the *R* repeated runs, not per‑item scores:
```python
def mean_se_rounds(times):
    m = statistics.fmean(times)
    return m, (statistics.stdev(times) / math.sqrt(len(times)) if len(times) > 1 else None)
```
* Guard against `n == 0` and negative `ns_per_item`; clamp to zero or raise a clear error.
* Verify that the floor baseline produced a valid `items` list before computing its mean.
* Stream the file when computing SHA‑256 to avoid loading whole files into memory.

**Severity:** **Major** – the leaderboard’s core metrics become unreliable.

---

## 4. Test‑gate weaknesses

| # | Location | What is wrong |
|---|----------|---------------|
| 4.1 | `runner/run.py: cmd_evaluate()` – `require_plan` only checks that *some* digests are present in the ledger entry, but does **not** verify that the ledger entry’s `body["inputs"]` list is **exactly** the set of digests. Extra or missing entries go unnoticed. |
| 4.2 | `runner/run.py: cmd_evaluate()` – after `require_plan` succeeds, the runner re‑trains each candidate (`train(c, t)`) *twice* (once for the initial weight file, once inside the `if plan and train(c, t) != w` block). If the candidate’s training is nondeterministic, the second call may produce a different weight file, causing a spurious failure *after* the test gate has already been passed. This defeats the purpose of the reproducibility requirement. |
| 4.3 | `runner/run.py: cmd_evaluate()` – the plan sequence number (`--plan SEQ`) is taken as an **int** but never validated against the ledger’s *current* highest sequence number. An attacker could supply a stale plan (e.g., `SEQ=0`) that does not include the latest runner code, yet the runner will accept it as long as the digests match. |
| 4.4 | `runner/run.py: require_plan()` – after verifying digests, it runs `record.py verify` and `record.py check-anchors` **outside** the jail (`cwd=REPO`). Those scripts are trusted, but they are executed with the full host environment, potentially allowing a malicious ledger entry to inject a malicious `record.py` that runs arbitrary code (the ledger is assumed trusted, but the runner’s security boundary should not rely on that). |

### Concrete exploit
Create a ledger entry `000123.json` that lists the correct digests for the current runner, **plus** an extra entry pointing to a malicious `record.py` placed somewhere on the host (e.g., `/tmp/malicious_record.py`). Because `require_plan` only checks for *missing* digests, it will ignore the extra entry. When `require_plan` later runs `record.py verify`, the host’s `PATH` may resolve to the malicious script, which could exfiltrate data or modify the ledger.

### Why it matters
* **Bypass of reproducibility** – a candidate can be evaluated on a *different* runner version than the one frozen in the plan, undermining the test‑gate’s guarantee.
* **Supply‑chain attack surface** – executing untrusted ledger‑controlled scripts defeats the isolation model.

### Fix
* In `require_plan`, compare the *exact* sets:
```python
expected = {(x["name"], x["sha256"]) for x in digests(t, cands)}
actual   = {(x["name"], x["sha256"]) for x in e["body"].get("inputs", [])}
if expected != actual:
    raise Failed(f"plan {seq} does not exactly match current digests (missing {expected-actual}, extra {actual-expected})")
```
* Store the plan’s digest **and** its content hash in the result JSON; on test runs, recompute the hash of the plan file and compare.
* Avoid re‑training after the plan check: compute `w = train(c, t)` once, store it, and reuse the same bytes for reproducibility verification.
* Run `record.py` inside the same jail used for inference, or at least with a minimal, read‑only environment.

**Severity:** **Major** – the test gate is the only guarantee of scientific rigor; compromising it invalidates the whole evaluation pipeline.

---

## 5. WalkSAT baseline (`tasks/sat-walksat/infer.c`) contains correctness bugs that can mislead the leaderboard

| # | Location | What is wrong |
|---|----------|---------------|
| 5.1 | `infer.c: main()` – allocation of `best[256]` without initialization | `best` is only written when `unsat < bestu`. If the first iteration’s `unsat` equals `bestu` (which is initialized to `m + 1`), the condition is true and `best` is set. However, if **all** iterations have `unsat == bestu` (e.g., the instance is already unsatisfiable from the start), `best` never receives a copy of `a`. The subsequent `put_rec(stdout, best, n)` writes **uninitialized stack memory** (garbage) as the output assignment. This yields nondeterministic, potentially out‑of‑range bytes that the SAT checker treats as malformed or, worse, as a *valid* assignment that may accidentally satisfy the instance. |
| 5.2 | `infer.c: main()` – `unsigned char a[256]` but `n` can be up to **255** (max_len). The array is large enough, but the code never checks `n <= 256`. If a future task raises `max_len` to 300, the program will write past the array, corrupting the stack and possibly the return address. |
| 5.3 | `infer.c: main()` – `int *nt = calloc(m, sizeof(int));` but `nt` is freed **inside** the per‑instance loop (`free(nt);`). If `m == 0` (degenerate instance), `calloc(0,…)` may return `NULL`; `free(NULL)` is fine, but later the code dereferences `nt` (`nt[c]`) before checking `m > 0`, leading to a null‑pointer dereference. |
| 5.4 | `infer.c: main()` – the random number generator `rs` is seeded with a 64‑bit constant XORed with the instance index, but the algorithm uses `rnd() % 1000` to compare with `noise`. If `noise` > 1000, the probability calculation is wrong (should be `noise / 1000`). The checker does not notice, but the baseline’s performance becomes inconsistent across runs, making the “floor” unstable. |

### Concrete exploit
Provide a SAT instance with `n = 5, m = 0` (zero clauses). The program will allocate `nt = calloc(0,…)` → `NULL`. Inside the clause‑loop (`for (int c = 0; c < m; c++)`) the body is skipped, but later the code accesses `nt[cc]` when selecting an unsatisfied clause (`if (!nt[cc])`). This dereferences `NULL` → segmentation fault → the runner aborts the whole evaluation batch.

### Why it matters
* The *floor baseline* (`sat-zeros`) is required to be beaten. If the WalkSAT baseline crashes or returns garbage, the floor may be artificially low, allowing a weak candidate to appear to outperform the floor.
* Uninitialized output can give a *random* high score for some instances, inflating the baseline’s mean and breaking reproducibility across runs.

### Fix
* Initialise `best` before the loop:
```c
memset(best, 0, sizeof(best));
```
* After the loop, if `bestu == m + 1` (i.e., no improvement ever recorded), copy the initial assignment:
```c
if (bestu == m + 1) memcpy(best, a, n);
```
* Add a runtime assertion that `n <= 256`; abort with a clear error if violated.
* Guard against `m == 0`:
```c
if (m == 0) { put_rec(stdout, a, n); continue; }
```
* Correct the noise probability check:
```c
if ((rnd() % 1000) < noise)  // keep only if noise <= 1000
```
or store `noise` as a fraction out of 1000 and scale appropriately.

**Severity:** **Major** – the floor baseline is unreliable, which directly skews the leaderboard and defeats the contract’s “must beat the floor” rule.

---

## Summary of grades

| Issue | Severity |
|-------|----------|
| 1.1–1.4 (sandbox escape, timer‑marker injection, writable `/tmp`) | **Critical** |
| 2.1–2.5 (hidden task‑specific assumptions) | **Major** |
| 3.1–3.5 (statistical mis‑reporting, division‑by‑zero, negative times) | **Major** |
| 4.1–4.4 (plan verification gaps, nondeterministic re‑training, ledger script execution) | **Major** |
| 5.1–5.4 (WalkSAT baseline bugs) | **Major** |

---

**Decision:** **do not ship**. The runner currently fails the most fundamental security requirement (isolation of untrusted code) and violates the contract it is supposed to enforce. All the listed defects must be fixed before the system can be considered safe or scientifically valid.
