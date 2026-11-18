<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=73.64s -->

**TL;DR – the experiment cannot be trusted in its current form.**  
The most serious problem is a *logic‑gate network (LGN) training bug* that flips the sign of every gradient update for all layers except the first.  This makes the LGN models essentially random and will either create spurious “rankings differ’’ or hide genuine flips.  Several smaller but still important issues also exist (duplicate flip counting, possible division‑by‑zero, nondeterministic BLAS, and a confound that the training budget changes with the task size).  All of these must be fixed **before the first run**.

Below is a systematic, adversarial audit.  For each defect I give:

* **Location** – file and function (or PLAN.md section) where the problem lives.  
* **What is wrong** – a concrete description of the bug / design flaw.  
* **Why it matters for the verdict** – how the error can produce a false “rankings differ’’ or a false “no flip found’’ (or otherwise invalidate the result).  
* **Concrete fix** – minimal code change or experimental‑design change that eliminates the problem.  
* **Severity** – **Critical** (must be fixed before any run), **Major** (must be fixed for a reliable verdict), **Minor** (nice‑to‑have, does not change the logical outcome).

---

## 1. Reversed gradient order in LGN training  
**Location:** `transfer/families.py`, function `train_lgn` – lines where `grads` is built and later passed to `adam_step`.

**What is wrong**  
```python
grads = []
...
for li in range(len(layers) - 1, -1, -1):
    ...
    grads = [glo, gla, glb] + grads
...
adam_step(params, grads, state, LGN_LR, t)
```
`params` is created **in forward layer order**:

```python
params = [L[k] for L in layers for k in ("lo","la","lb")]
```

but `grads` is built by **pre‑pending** each layer’s gradients, ending up in **reverse order** (last layer first).  Consequently every Adam update is applied to the wrong weight matrix (e.g. the gradient for layer 2 is added to the parameters of layer 1).  The LGN network never learns the intended function; its final exported weights are essentially noise.

**Why it matters**  
* The LGN family is a core comparison point (linear vs MLP vs LGN).  
* Incorrect training can make LGN appear dramatically slower or less accurate on some tasks, creating *false* flips in both accuracy and cost.  
* The “C‑output equals NumPy output’’ check will still pass because the same (incorrect) weights are used for both, so the error is invisible to the meter.

**Concrete fix**  
Collect gradients in the **same order** as `params`:

```python
grads = []
for L in layers:                     # forward order
    # compute glo, gla, glb for this layer as before
    grads.extend([glo, gla, glb])   # append, not prepend
# now grads matches params
adam_step(params, grads, state, LGN_LR, t)
```

Alternatively, keep the reverse‑order loop but reverse `params` as well, but the append‑style is clearer.

**Severity:** **Critical** – the experiment’s central hypothesis cannot be evaluated until LGN training is correct.

---

## 2. Duplicate counting of flips in `decide.py`  
**Location:** `transfer/decide.py`, function `main` – the nested loops that build `flips`.

**What is wrong**  
```python
for t1, t2 in itertools.permutations(sorted(ords), 2):
    if ... ords[t1][pair][m] == 1 and ords[t2][pair][m] == -1:
        flips.append(...)
```
Both `(taskA, taskB)` **and** `(taskB, taskA)` are examined, so a genuine flip between two tasks is recorded **twice** (once for each ordering).  The `flips_by_measure` summary therefore over‑states the number of flips, and the printed JSON contains duplicate entries.

**Why it matters**  
The verdict (`rankings differ` vs `no flip found`) is binary, so duplicate entries do not change the final decision, but they **inflate** the reported evidence and confuse downstream analysis (e.g., step 2 may treat the flip as occurring on three tasks instead of two).  It also makes debugging harder because the same flip appears multiple times.

**Concrete fix**  
Iterate over **unordered** task pairs:

```python
for t1, t2 in itertools.combinations(sorted(ords), 2):
    # check both directions
    if ords[t1][pair][m] == 1 and ords[t2][pair][m] == -1:
        flips.append(...)
    elif ords[t2][pair][m] == 1 and ords[t1][pair][m] == -1:
        flips.append(...)
```

Or keep the permutation loop but add a `seen` set to avoid duplicates.

**Severity:** **Minor** – does not affect the binary verdict but improves reporting.

---

## 3. Potential division‑by‑zero in `decide.py` when computing `ir_ratio`  
**Location:** `transfer/decide.py`, line constructing `flips`:

```python
ir_ratio = a["ir"] / b["ir"]
```

**What is wrong**  
If a family’s measured instructions per item (`ir`) is exactly zero (possible for a degenerate model that never reads input, e.g. a “majority‑class” baseline that could be added later), the division raises a `ZeroDivisionError` and aborts the script, marking the whole experiment invalid.

**Why it matters**  
Even if the current families never produce zero instructions, the code is brittle and will break if a future baseline is added or if a bug in the meter yields a zero count.

**Concrete fix**  
Guard against zero:

```python
ir_ratio = a["ir"] / b["ir"] if b["ir"] != 0 else float("inf")
```

Optionally store a flag indicating an undefined ratio.

**Severity:** **Minor** – safety‑net for future extensions; does not affect current verdict.

---

## 4. Training‑budget confound across tasks  
**Location:** `transfer/families.py`, function `epochs_for`; also described in **PLAN.md** (section *Limits, stated in advance*).

**What is wrong**  
The number of epochs is **task‑dependent** (`epochs = min(100, ceil(600 000 / n_train))`).  
* optdigits (n≈2.9 k) → 100 epochs (capped).  
* text (n≈30 k) → 20 epochs.  
* adult (n≈32 k) → 19 epochs.  

Thus the “same budget for all families’’ claim is false across tasks.  A flip may be caused simply by under‑training on a larger task rather than by a genuine change in family ordering.

**Why it matters**  
The hypothesis under test is *“Do the rankings of families change across task kinds?”*  
If the training budget changes, the experiment no longer isolates **family structure** as the only variable; budget differences become a systematic confound that can generate spurious flips (especially for LGN, which is more sensitive to epoch count).

**Concrete fix**  
*Option A – true‑budget fix:*  
Choose a **fixed epoch count** (e.g. 100 for every task) and adjust `epochs_for` to ignore `n_train`:

```python
def epochs_for(_):
    return MAX_EPOCHS   # 100 for all tasks
```

*Option B – analysis fix (if budget must stay variable):*  
Add a **post‑hoc correction** in `decide.py` that flags any pair where the two tasks have different epoch counts and either (i) excludes that pair from the flip test or (ii) reports the budget difference as a separate “budget‑flip’’ flag.  The simplest is to keep the fixed‑epoch version; it eliminates the confound entirely.

**Severity:** **Major** – without fixing, the primary scientific claim is ambiguous.

---

## 5. Nondeterministic BLAS / NumPy multithreading  
**Location:** `transfer/families.py` – any call that performs matrix multiplications (`@`), e.g. `a = z @ W1 + b1`, `h @ W2`, `np.einsum`, `np.dot` inside LGN training.

**What is wrong**  
NumPy (via OpenBLAS / MKL) may use nondeterministic floating‑point reduction order when multithreaded, leading to **tiny but non‑zero variations** between runs even with the same seed.  This inflates the *seed variance* estimate and can cause the accuracy SE to be over‑estimated, potentially turning a genuine flip into “no flip’’ (false negative) or vice‑versa.

**Why it matters**  
The decision rule relies on accurate SE estimates; an inflated SE makes the test too conservative, hiding real ranking changes.

**Concrete fix**  
Force **single‑threaded** linear algebra for the whole experiment:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Add these environment variables to the CI job script *before* any Python process starts, or set them at the top of `families.py`:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
```

**Severity:** **Major** – without it the SE calculation is unreliable; the fix is trivial and should be done.

---

## 6. Inadequate handling of instruction‑count noise  
**Location:** `transfer/meter.py`, function `ir`.

**What is wrong**  
The instruction count is taken **once** per artifact (`ir_n` and `ir_0`).  Valgrind’s cache‑grind output can vary by a few percent due to OS scheduling, page faults, or background activity.  The rule in `decide.py` uses a **fixed 10 % tolerance** (`Ir_A * 1.10 < Ir_B`).  If measurement noise exceeds ~5 % on either side, a true cost difference may be masked, yielding a false “no flip’’ for the cost measure.

**Why it matters**  
Cost flips (instructions or native time) are part of the verdict.  Under‑estimating measurement variance can hide legitimate cost flips, biasing the result toward “no flip”.

**Concrete fix**  
*Take multiple Valgrind runs* (e.g. 3 repetitions) and use the **median** or **mean** as the estimate, optionally computing an SE and widening the tolerance accordingly.  Minimal code change:

```python
def ir(prog, w, x, n, repeats=3):
    vals = []
    for _ in range(repeats):
        r = subprocess.run([...], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        m = re.search(r"I\s+refs:\s+([\d,]+)", r.stderr)
        vals.append(int(m.group(1).replace(",", "")))
    return statistics.median(vals)
```

Then adjust the tolerance in `decide.py` to, for example, `1.05` (5 %) if the median absolute deviation is <5 %.  The exact factor can be calibrated on a pilot run.

**Severity:** **Major** – cost flips are a primary outcome; noisy measurements can invalidate the verdict.

---

## 7. Missing seed‑variance term for *per‑item* differences  
**Location:** `transfer/decide.py`, function `orders`.

**What is wrong**  
The SE formula in the plan is:

\[
SE^2 = \frac{\operatorname{Var}(d_i)}{n} + \frac{\operatorname{Var}_{\text{seeds}}(A)}{s_A} + \frac{\operatorname{Var}_{\text{seeds}}(B)}{s_B}
\]

where \(d_i\) are per‑item differences **averaged over seeds**.  
`orders` computes `var = d.var(ddof=1) / len(d)` (first term) **and then adds** the seed‑variance of the *overall* accuracies (`acc_seeds`).  
However, the per‑item variance term already incorporates **within‑seed** variability; the seed‑variance of the *mean* should be computed from the **per‑seed item‑means**, not from the overall seed accuracies.  Using `acc_seeds` (which are already averages over items) double‑counts part of the variance and makes the SE **larger than it should be**.

**Why it matters**  
An oversized SE makes the accuracy test overly conservative, increasing the chance of a false “no flip’’ when a real accuracy flip exists.

**Concrete fix**  
Compute the seed‑variance of the *per‑seed item‑means*:

```python
# In task_summary, keep per‑seed item means:
out[f]["seed_item_means"] = np.stack([(b64(a["pred_b64"]) == y).astype(float) for a in arts])  # shape (seeds, items)

# In orders():
seed_means_A = np.mean(a["seed_item_means"], axis=1)   # mean over items per seed
seed_means_B = np.mean(b["seed_item_means"], axis=1)
if len(seed_means_A) > 1:
    var += statistics.variance(seed_means_A) / len(seed_means_A)
if len(seed_means_B) > 1:
    var += statistics.variance(seed_means_B) / len(seed_means_B)
```

If you prefer a minimal change, simply **remove** the two `if len(x["acc_seeds"]) > 1:` blocks; the per‑item variance already captures the within‑seed noise, and the seed‑variance term is optional given only three seeds.

**Severity:** **Major** – the accuracy decision rule becomes too lax, potentially hiding true flips.

---

## 8. Lack of explicit handling for missing `ns_rounds` (edge case)  
**Location:** `transfer/decide.py`, function `orders` (native‑time block) and `main` (table construction).

**What is wrong**  
If for any reason a round fails and `ns_rounds` stays empty, `rd.mean()` and `rd.std()` raise a `StatisticsError`.  The script would crash and the experiment would be marked **invalid**.

**Why it matters**  
A single failed timing (e.g., a transient OS hiccup) would invalidate the whole study, even though the data are otherwise fine.

**Concrete fix**  
Guard against empty lists:

```python
if len(rd) == 0:
    nD, nse = 0.0, float("inf")   # treat as no evidence
else:
    nD, nse = float(rd.mean()), float(rd.std(ddof=1) / math.sqrt(len(rd)))
```

**Severity:** **Minor** – defensive programming; does not affect the intended verdict when runs succeed.

---

## 9. Incomplete documentation of the “majority‑class baseline’’  
**Location:** `transfer/PLAN.md`, section *Families* (first bullet under *Families*).

**What is wrong**  
The plan says “Mandatory baselines: linear is in every comparison; the majority class is reported.”  
However, the code never creates a *majority‑class* artifact, never records its `family` name, and never includes it in `orders`.  Consequently the baseline cannot be used to sanity‑check that the ranking rule works (e.g., that the majority class is never declared “better’’ than a trained model).

**Why it matters**  
Without a trivial baseline, a hidden systematic error (e.g., all models accidentally receiving the same predictions) could go unnoticed, leading to a false “no flip’’ verdict.

**Concrete fix**  
Add a lightweight baseline generator (e.g., `families.py` flag `--baseline-majority`) that writes a dummy weight file containing the constant prediction and registers it as family `"majority"`.  Ensure `meter.py` and `decide.py` treat it like any other family.

**Severity:** **Minor** – does not change the main hypothesis but provides an important sanity check.

---

# Verdict

**✘ Do not run the experiment as‑is.**  
The critical LGN gradient‑order bug alone invalidates the scientific claim.  In addition, the epoch‑budget confound, the oversized SE calculation, and the nondeterministic BLAS issue each threaten the validity of the final verdict.  All **Critical** and **Major** problems must be resolved before any data are collected.

**Required actions before the first run**

1. Fix the gradient order in `train_lgn` (Critical).  
2. Adopt a **fixed epoch count** for all tasks (Major) **or** explicitly model the budget difference.  
3. Force single‑threaded NumPy/BLAS (Major).  
4. Correct the SE computation to avoid double‑counting seed variance (Major).  
5. (Optional but strongly recommended) average Valgrind instruction counts over several runs and tighten the tolerance (Major).  
6. Apply the minor patches listed above (duplicate flip counting, division‑by‑zero guard, empty‑round handling, baseline implementation).

Once these changes are merged and the repository is re‑tagged (the registration commit must be updated accordingly), the experiment can be re‑run safely.
