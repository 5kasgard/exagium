# Queue Q10 (jail hardening) and Q55 (weights across hosts), 2026-11-18

## Q10: the runner's jail now has the filter, the limits and no nested namespaces
**What was missing** (A-security m5): `loop.py`'s bubblewrap mode, the one CI uses, had no seccomp filter and allowed
nested user namespaces, and neither mode limited memory or disk.

**What changed** (`runner/jail.py`, `runner/limits.c`; `loop.py` not edited):
- every jailed command starts under `limits.c`, which sets 8 GiB of address space, 1 GiB per file, 256 open files
  and CPU seconds equal to the timeout, sets no_new_privs, and adds `loop.py`'s seccomp filter;
- the timer forks, so it gets the limits only and filters its child (`--seccomp`) in both modes; the runner now
  requires the timer to report a filtered child;
- the counter: `count.py` forks valgrind, so in bubblewrap mode valgrind, and with it the candidate, starts inside
  `loop.py`'s confine (Landlock and the same filter) within the jail;
- bubblewrap gets `--unshare-user --disable-userns` and a 2 GiB tmpfs (`--size`) where it supports them; CI's does;
- `jail.check()` fails closed unless, inside the jail:
  - all four limits read back as set and none can be raised, and no_new_privs is set;
  - fork, sockets, a new user namespace and a file over 1 GiB are all refused.

  In bubblewrap mode it also requires `--disable-userns` and `--size`. It proves a command without the filter is
  still refused a user namespace, and that the counter path refuses fork and sockets. (The read-back of the CPU
  limit, the raise attempt and these bubblewrap checks were added after the independent check; see below.)

**Evidence:**
- **Fails closed:** run 44257075574 stopped before any candidate ran (`run44257075574-failed-closed.txt`); bubblewrap
  needed an explicit `--unshare-user` for `--disable-userns`.
- **Probe passes with every protection,** in bubblewrap mode on CI (run 44260596412, `ci-44260596412/jail-*.json`)
  and in confine mode on this machine.
- **The probe catches the old jail:** run through `loop.py`'s own jail here (confine mode), it reported four
  missing rlimits, each of which could also be raised.
- **What changed in the measurements** against P2 (`runner/results/2026-11-18/ci/`):
  - outputs, dev scores and instructions per item are unchanged, except text mlp16 (see Q55);
  - raw instruction counts at N and at 0 are 29 higher per run on sat and text, so the per-item figure is equal. The
    likely cause is the added environment variable (hypothesis, not tested);
  - the weights' sha256 changed for 5 candidate–task cells, because training moved from each host's automatic
    kernel to the pinned one. Their outputs are identical.

**Reviews:**
- gpt-oss-120b (router-runs `reviews/44257135013`) raised 7 points; checked against the code:
  - **applied:** the counter run had no filter (critical in the review); it now runs under confine, and the probe
    covers it. The tmpfs could fill memory where `--size` is missing; CI's bubblewrap has it, and the gap is stated.
  - **rejected:** "the timer's parent is unfiltered" (the timer is trusted, bubblewrap clears the environment, and
    candidate code runs only in the filtered child). "Add RLIMIT_NPROC" (it counts every process of the host user,
    the harness included, so a low value breaks the runner; the filter refuses fork instead). Two points had no
    effect.
- **Remaining gaps:**
  - in confine mode the counter's valgrind run has the filter but not the rlimits;
  - training's working directory is a host directory, bounded per file only, not in total.
- **Independent check** (a separate agent): it confirmed every Q55 number and traced every candidate path. No path
  runs candidate code unfiltered in bubblewrap mode. It found overstated wording, fixed above, and the probe's gaps,
  closed above. It also found the runner compiles candidate C with gcc outside the jail. That is older than this
  work and outside Q10, so it is a new queue item, Q56: an `#include` of a host file could put that file into the
  binary and its output.

## Q55: the cause was OpenBLAS choosing its kernels per CPU; pinning one kernel type fixes it
**Test, fixed before running (`runner/q55.py`):**
- **mechanism:** on one host, weights differ between forced core types;
- **cause:** with the same forced type, weights are identical on hosts whose automatic weights differ.

**Both held.** Text, sha256 prefixes for label-linear / label-mlp16:

| forced type | this machine (Xeon, AVX-512, Py 3.13) | CI EPYC 7763 (AVX2, Py 3.12) | CI Xeon 6975P-C (Py 3.12) |
|---|---|---|---|
| automatic | 3657542e / c847e77d | 97ffdde4 / e686af06 | 3657542e / c847e77d |
| Prescott | 11ecbce2 / 1f37571d | 11ecbce2 / 1f37571d | 11ecbce2 / 1f37571d |
| Haswell | 97ffdde4 / e686af06 | 97ffdde4 / e686af06 | 97ffdde4 / e686af06 |
| SkylakeX | 3657542e / c847e77d | crashed: no AVX-512 | 3657542e / c847e77d |

Sources:
- `q55-local.json`;
- `q55-epyc7763-run44258892424.txt` (CI log lines; the crash lost only the Q55 file, and that run's text results
  are in `dev-text-epyc7763-run44258892424.json`, whose weights equal the pinned ones);
- `ci-44260596412/q55-text.json`.

**Fix:** the runner's jail pins `OPENBLAS_CORETYPE=Haswell` on x86 (AVX2, present on every x86 host seen so far).
With it, text trained in CI's jail (Xeon 6975P-C) and here (`reproduce-text-local-pinned.json`) give identical
weights for all three candidates.

**Effect on P2's numbers:** text mlp16's dev accuracy is now 0.3260. P2 reported 0.3278, trained on an EPYC 9V74 under
its automatic kernel. No other P2 score changed.

**Not covered:** ARM hosts keep OpenBLAS's own choice. On a CPU without AVX2 the pinned kernels would crash training
(SIGILL, as SkylakeX did on the EPYC); it fails loudly at train time, not at `check()`. Neither was seen.
