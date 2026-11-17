<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=69.47s -->

**Adversarial review of `loop/scale.py` and its documentation**  

Below is a line‑by‑line audit.  For each defect I give:

* **Location** – file and line (or function) where the problem lives.  
* **What is wrong** – the concrete bug, design flaw or misleading claim.  
* **Why it matters** – how the defect can corrupt the measurement, allow cheating, or mislead readers.  
* **Concrete fix** – a minimal change that eliminates the problem.  
* **Severity** – *Critical* (breaks the core claim), *Major* (substantially weakens credibility), *Minor* (annoyance or style).

---

### 1. Large‑N slope estimator is fragile
* **Location** – `summarise()` (lines 44‑53).  
  ```python
  big, prev = sizes[-1], sizes[-2]
  def marginal(r):
      return (r[big] - r[prev]) / (big - prev)
  ```
* **What is wrong** – The “large‑N marginal cost” is computed from *only the two largest run sizes*.  If either of those two points is an outlier (e.g. a single round where the timer slipped, thermal throttling, or OS jitter), the slope is biased.  The README presents this single‑point slope as a *sound estimator* without any robustness justification.  
* **Why it matters** – The whole ranking of candidates hinges on that number.  A single noisy round (the 8 ms t(0) shown for `lgn_blockread`) can swing the slope enough to flip the order of two close competitors, yet the reported SE is the standard error of the *means* and does **not** down‑weight the outlier.  
* **Fix** – Compute a per‑round marginal first, then take a **median** (or a trimmed‑mean) across rounds and report a robust dispersion (e.g. MAD → SE≈1.4826·MAD/√R).  Optionally fit an OLS line over *all* sizes ≥ 16 384 and use the fitted coefficient.  

  ```python
  def marginal(r):
      return (r[big] - r[prev]) / (big - prev)

  per_round = [marginal(r) for r in rounds]
  med = statistics.median(per_round)
  mad = statistics.median([abs(x - med) for x in per_round])
  se = 1.4826 * mad / math.sqrt(len(per_round))
  rec["marginal_ns_per_image_large_n"] = (med, se)
  ```

* **Severity** – **Critical** (the primary metric can be arbitrarily distorted).

---

### 2. Pairing by round does **not** control for order‑dependent effects
* **Location** – `measure()` (lines 84‑92).  

  ```python
  jobs = [(c, n) for c in cands for n in sizes]
  rng.shuffle(jobs)
  for c, n in jobs:
      pred, t[c["name"]][n] = L.timed(...)
  ```
* **What is wrong** – All candidates are run **sequentially** in a single shuffled list.  The order inside a round is therefore random, but the pairing of candidates for the “paired‑by‑round” statistics assumes the two measurements are taken under *identical* conditions.  In reality the first candidate in the list runs on a cold CPU, the last one runs after several seconds of heat, after the OS may have swapped pages, after the timer’s own memory‑mapped file has been touched, etc.  Those systematic drifts are *not* cancelled by the per‑round pairing.  
* **Why it matters** – A candidate that is slower when executed later will be penalised relative to a faster‑when‑first candidate, producing a spurious “large‑N advantage”.  This undermines the claim that “rounds are paired, therefore the comparison stays fair”.  
* **Fix** – Run each size **in its own mini‑round** so that every candidate experiences the *same* global state for that size, or shuffle the *candidate* order *per size* but keep the size order fixed.  Simpler:  

  ```python
  for n in sizes:                     # outer loop = size
      rng.shuffle(cands)              # shuffle candidates for this size only
      for c in cands:
          pred, t[c["name"]][n] = L.timed(...)
  ```
  This guarantees that for any given size all candidates are measured after the same amount of work has been done.  

* **Severity** – **Major** (creates a hidden source of variance that can change the ranking).

---

### 3. Nudged‑image pool is not a representative workload
* **Location** – `pool()` (lines 21‑38) and the README paragraph *“Each is either an original or a copy with 4 pixels moved by ±1.”*  
* **What is wrong** – Only **four** pixel perturbations are applied, and the label is *kept from the original* even if the perturbation would change the true digit.  This produces a set that is **highly correlated** with the original training set: many images are identical or differ by a single intensity change.  Candidates that cache the *exact* training rows (e.g. by keeping a hash table of seen rows) will see dramatically lower per‑image work, while a more realistic distribution (random rotations, scaling, noise) would expose the true cost.  Moreover, mis‑labelled nudged images inflate the measured accuracy (the test passes because the reference model also sees the same wrong label).  
* **Why it matters** – The “cost per image” is claimed to be “everything the candidate does per image”.  If the image set is artificially easy for a particular implementation, the reported ns/img is not portable to real‑world data and the comparison is unfair.  
* **Fix** –  
  1. Increase the perturbation budget (e.g. 8–12 random pixel changes, random Gaussian noise, small rotations).  
  2. Re‑label the nudged images by running the *reference* model on them **after** the perturbation and storing the *new* label, or discard any nudged image whose label changes.  
  3. Document the new procedure clearly in the README.  

* **Severity** – **Major** (the benchmark workload is not representative, allowing “cache‑cheating”).

---

### 4. Accuracy‑check is under‑powered and uses the same pool for both training and validation
* **Location** – `accuracy_check()` (lines 57‑70).  
* **What is wrong** –  
  * It compares only two aggregates: first 956 images vs. the rest.  A candidate can cheat on the middle segment (e.g. 10 000 – 20 000 images) and still pass the test.  
  * The threshold “more than 3 SE below” is computed from a *binomial* SE that assumes independent predictions; predictions from a deterministic model are perfectly correlated, so the SE is dramatically *under‑estimated*.  
  * Both halves are drawn from the **same pool** that the candidate was built on (training rows).  A model that simply memorises the training set will achieve 100 % on the whole pool, yet its *generalisation* cost is irrelevant.  
* **Why it matters** – The whole “cheating detection” mechanism can be bypassed (as the authors themselves demonstrated with the 1 000‑image cheat).  A malicious candidate could output correct labels for the first 956, then output a constant label for the rest, or degrade gradually, and the test would not flag it.  This directly invalidates the claim “the accuracy check catches all cheap‑but‑inaccurate candidates”.  
* **Fix** –  
  1. Use a **held‑out validation set** (the official `dev` split) that is *never* part of the pool.  
  2. Require monotonic non‑decrease of accuracy across **all** size thresholds (e.g. check 956, 4 096, 16 384, 65 536).  
  3. Replace the 3 SE rule with a *practical* tolerance, e.g. “accuracy must not drop more than 0.5 % absolute”.  
  4. Compute the tolerance using a *bootstrap* on the validation set rather than the binomial formula.  

* **Severity** – **Critical** (the primary safeguard against cheating is ineffective).

---

### 5. README makes undefined or overstated claims
* **Location** – README section *“Separation: the large‑N figure separates 23 of the 28 pairs at t(14).”*  
* **What is wrong** –  
  * The term **t(14)** is never defined; the only “t(14)” that appears elsewhere is a *threshold* of 2.145 ns/img mentioned in the review prompt, but the README does not explain how it is derived.  
  * The claim “separates 23 of 28 pairs” suggests a statistical test, yet no p‑value or confidence interval is provided.  
  * The statement “One noisy round: … carries one slow round” is contradictory – the noisy round is *included* in the reported SE, so the SE already reflects the outlier; no special handling is described.  
* **Why it matters** – Readers are led to believe that the reported numbers are *statistically significant* when, in fact, the significance criteria are undocumented.  This undermines trust in the whole results table.  
* **Fix** – Add a concise definition of **t(14)** (e.g. “the 14‑th percentile of the per‑round marginal differences”), and replace the informal “separates” language with an explicit hypothesis test (e.g. paired‑t test with α=0.05) or with a robust metric such as non‑overlapping confidence intervals.  Also either filter the outlier round before computing SE or report both *raw* and *outlier‑removed* statistics.  

* **Severity** – **Minor** (documentation flaw, but it propagates confusion).

---

### 6. SE computation assumes independence of rounds
* **Location** – `summarise()` lines 73‑80 (pairwise statistics).  

  ```python
  m, se = L.mean_se([p - q for p, q in zip(mx, my)])  # paired by round index
  ```
* **What is wrong** – `L.mean_se` presumably returns `mean ± std/√n`.  The *n* is the number of rounds, but the rounds share the **exact same image pool**, the same timer binary, and the same process environment, so
