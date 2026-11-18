<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=68.19s -->

**Adversarial security review – jail hardening change (Q10)**  

Only the code that is visible in the repository was traced.  Anything that cannot be followed from the shown functions is marked **“not traceable”**.

---

## 1.  Candidate code can run **before** the seccomp filter or raise limits **after** they are set  
**File / function**: `runner/jail.py :: jail` (the `seccomp=False` branch that is used for the trusted timer)

**Concrete attack**  
* The timer binary is started **without** the `--seccomp` flag, so `limits` runs only the r‑limit setup and `PR_SET_NO_NEW_PRIVS`.  
* The timer then **forks** its child and installs the filter *inside* the child.  
* An attacker who can influence the timer (e.g. by supplying a malicious `LD_PRELOAD` library, a crafted environment variable that the timer reads, or by abusing any code path that runs **before** the child is fork‑ed) can cause the **parent** process to execute arbitrary syscalls (fork, socket, unshare, etc.) **outside** the filter but still inside the bubblewrap namespace.

**Does it succeed against the current code?** – **Yes.**  
The parent runs with the full set of syscalls (no seccomp) and only the soft‑hard limits are in place.  Forking a new process, opening a network socket, or creating a user namespace is therefore possible before the child is filtered.

**Fix**  
* Always pass `--seccomp` to `limits` for the timer as well, i.e. do not special‑case `seccomp=False` for the timer.  
* If the timer must install the filter itself, do it **before** any `fork()` or other privileged operation (e.g. call `prctl(PR_SET_SECCOMP, …)` at the very start of `timer.c`).  

**Severity**: **Major** – the parent of the timer runs with essentially the same privileges as a candidate program and can leak data or affect the host.

---

## 2.  Counter run (`count.py` → Valgrind) gets **no seccomp filter**  
**File / function**: `runner/jail.py :: jail` (the `seccomp=False` path used for the counter run)

**Concrete attack**  
* `count.py` forks **valgrind**.  Because the `--seccomp` flag is omitted, `limits` only installs the rlimits; **no seccomp filter** is ever applied.  
* Valgrind uses `ptrace` to control the target binary, and it also opens raw sockets for its own diagnostics.  
* An adversarial candidate can embed malicious code in the *infer* binary (or in the data file) that, when executed under valgrind, uses the unrestricted `ptrace`, `socket`, `clone`, `unshare`, etc., to:
  * Read/write arbitrary memory of the host process tree,
  * Open a network connection and exfiltrate data,
  * Create additional processes or threads that survive after the valgrind run,
  * Perform a **fork bomb** that exhausts the PID table (the filter that would normally block `fork` is absent).

**Does it succeed against the current code?** – **Yes.**  
The sandbox that protects the candidate during the counter run consists only of the rlimits (8 GiB AS, 1 GiB file size, 256 fds).  All the dangerous syscalls that the seccomp filter would block are permitted.

**Fix**  
* Run the counter under the same full sandbox as the timed runs: invoke `limits --seccomp …` before `count.py`.  
* If valgrind itself must be run without a filter, place it **inside a nested bubblewrap** that still applies the seccomp filter to the *infer* process, or use a separate seccomp profile that allows only the handful of syscalls required by valgrind (e.g. `ptrace`, `mmap`, `rt_sigaction`).  

**Severity**: **Critical** – the missing filter re‑opens the entire attack surface that the original Q9 jail was designed to close.

---

## 3.  The jail‑probe (`check()`) does **not** verify the counter‑run sandbox  
**File / function**: `runner/jail.py :: check`

**Concrete attack**  
* `check()` launches a short Python probe that attempts `fork()`, `socket()`, `unshare(CLONE_NEWUSER)`, excessive allocations, and oversized file writes.  
* It is executed **only** via the normal `jail()` path (`seccomp=True`).  
* The counter run (`count.py`) deliberately calls `jail(..., seccomp=False)`.  Consequently the probe is **never** run in the environment that lacks the seccomp filter.

**Does it succeed against the current code?** – **Yes** (the missing protection is invisible to the probe).  
The probe will report “limited” even though the counter run is unfiltered, because the probe never exercises that code path.

**Fix**  
* Add a second probe that is invoked from `count.py` (or from the `J.jail` call that builds the counter run) to verify that the seccomp filter is present.  
* Alternatively, make the `seccomp=False` flag internal to `limits.c` only for the timer, and **force** `--seccomp` for every other entry point, including the counter.

**Severity**: **Major** – the test that is supposed to guarantee the sandbox’s integrity does not cover a whole class of executions.

---

## 4.  Optional `--size` flag on `bwrap` leaves **tmpfs unbounded** on older bubblewrap versions  
**File / function**: `runner/jail.py :: _bwrap_flags` and the construction of the `bwrap` command in `jail()`

**Concrete attack**  
* If the host’s `bwrap` binary does **not** advertise `--size`, the code falls back to `--tmpfs /tmp` **without** a size limit.  
* The candidate can write arbitrarily many files (up to the RLIMIT_FSIZE per‑file, but unlimited number of files) into `/tmp`, exhausting host RAM because the tmpfs is backed by memory.  
* This can trigger OOM kills of unrelated CI jobs or the host itself.

**Does it succeed against the current code?** – **Yes**, when `bwrap --help` does not list `--size`.  The fallback path is clearly present.

**Fix**  
* Enforce a size limit **regardless** of `bwrap` support: mount a tmpfs with `size=2G` via the `--tmpfs /tmp:size=2G` syntax (supported on recent `bwrap`) and, when unavailable, create a separate directory on the host and bind‑mount it read‑only into the sandbox, or use a user‑space quota tool.  
* Additionally, lower `RLIMIT_FSIZE` or add an `RLIMIT_FSIG`‑like quota on total bytes written (e.g. via a `cgroup` memory limit) to guarantee a hard bound.

**Severity**: **Major** – an attacker can cause denial‑of‑service on the shared CI runner.

---

## 5.  No limit on **process count** (`RLIMIT_NPROC`) – fork bomb possible when filter missing  
**File / function**: `runner/limits.c :: main` (sets RLIMIT_AS, RLIMIT_FSIZE, RLIMIT_NOFILE, RLIMIT_CPU but not RLIMIT_NPROC)

**Concrete attack**  
* In any code path where the seccomp filter is absent (timer parent, counter run), the candidate can repeatedly call `fork()` (or `clone`) until the PID table is exhausted.  
* Each child inherits the same rlimits, so the per‑process `RLIMIT_CPU` does not stop the overall CPU consumption; the attacker can saturate the scheduler.

**Does it succeed against the current code?** – **Yes**, in the two unfiltered paths identified above.

**Fix**  
* Add `setrlimit(RLIMIT_NPROC, …)` in `limits.c` (e.g. 32 processes) before exec.  
* Ensure the seccomp filter is always present, making the fork‑bomb impossible even if the rlimit is omitted.

**Severity**: **Major** – PID‑table exhaustion can halt the CI worker entirely.

---

## 6.  `prctl(PR_SET_NO_NEW_PRIVS)` is applied **after** rlimit setup  
**File / function**: `runner/limits.c :: main` (order: set limits → `prctl(NO_NEW_PRIVS)` → optional seccomp)

**Concrete attack**  
* A malicious binary could attempt to **raise** its own hard limits *before* `NO_NEW_PRIVS` is set, but the code already sets the limits first, so raising is impossible.  
* No practical exploit is visible from the current code.

**Does it succeed?** – **Not traceable**.

**Fix** – No change needed, but documenting the ordering would aid future reviewers.

**Severity**: **Minor** (informational).

---

## 7.  `exagium_seccomp` allows `prlimit64` on own PID (SELF_OR_ZERO) – could lower limits  
**File / function**: `loop/seccomp_filter.h :: exagium_seccomp`

**Concrete attack**  
* A candidate could call `prlimit64` to *lower* its own RLIMIT_AS or RLIMIT_FSIZE, causing subsequent allocations to fail earlier (self‑induced denial‑of‑service).  
* This does not increase privileges or break isolation.

**Does it succeed?** – **Not traceable** (no gain for attacker).

**Fix** – If desired, deny `prlimit64` altogether, but the impact is low.

**Severity**: **Minor**.

---

# Summary of findings

| # | File / function | Issue | Can be reproduced? | Fix | Severity |
|---|------------------|-------|--------------------|-----|----------|
|1|`runner/jail.py :: jail` (timer branch) | Timer parent runs without seccomp, can perform privileged syscalls before filter | Yes | Apply `--seccomp` to timer as well (or filter before any fork) | **Major** |
|2|`runner/jail.py :: jail` (counter branch) | Counter run lacks seccomp; Valgrind can use ptrace, sockets, fork, etc. | Yes | Run counter under `limits --seccomp` or sandbox Valgrind separately | **Critical** |
|3|`runner/jail.py :: check` | Probe does not cover counter path, so missing filter goes unnoticed | Yes | Add probe for counter run or enforce filter universally | **Major** |
|4|`runner/jail.py :: _bwrap_flags` / `jail` | When `bwrap` lacks `--size`, tmpfs is unbounded → OOM | Yes | Force a size limit via explicit mount options or host‑side quota | **Major** |
|5|`runner/limits.c :: main` | No `RLIMIT_NPROC`; fork bomb possible when filter missing | Yes (timer parent, counter) | Add `RLIMIT_NPROC` limit; keep filter always present | **Major** |
|6|`runner/limits.c :: main` | Order of `prctl(NO_NEW_PRIVS)` after limits (no exploit) | Not traceable | None needed | **Minor** |
|7|`loop/seccomp_filter.h :: exagium_seccomp` | Allows self‑prlimit64 (could lower limits) | Not traceable (no benefit) | Optionally deny `prlimit64` | **Minor** |

---

**Decision:** The changes introduce **critical** and **major** regressions (missing seccomp on the counter run, timer parent without filter, unbounded tmpfs, and lack of a process‑count limit).  These flaws re‑open large attack surfaces that the original Q9 jail deliberately closed.

**Recommendation:** **Do not ship** this version until the fixes above are applied and the test suite is extended to verify that every execution path (timed run, counter run, training run) is protected by both the rlimits **and** the seccomp filter.  

**Result:** **do not ship**.
