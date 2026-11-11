# Z's verification, 2026-11-12

An outside verifier (GLM-5.3 on "Z", a KVM guest on Intel Granite Rapids, kernel 5.10.134) ran `VERIFY.md` against `main` at `410a6c9`. `report.md` is its report as received. `artifacts/` holds its probe profiles and host details (`sha256sum artifacts/*` to recompute):

| File | sha256 |
|---|---|
| `full_11.json` | `dd156274426de3b52e1904781cf61955aa6e3595b11592380b415f2c56ecf14b` |
| `full_22.json` | `9b3d27a2fecca66ef12dbc285ee6346e0219c1cda77a7408b4cc732311e39ba1` |
| `full_33.json` | `183e70924f1a98d9eb5b54c26f2efc5852162bd3c64ffa66f90d91d229092ab1` |
| `quick.json` | `b13734588718da0ba80948ad8e520790ea15e56d6224c051cc8f78c6268e53be` |
| `lscpu.txt` | `c1d230a0d91a7a6ec091447e716ef9573231b36c122a163718127f1f171fe994` |
| `valgrind_version.txt` | `288ab4bfe4ceec05fede2b24f88faeb925b79a9353a5289c4fde0a104a5c5d86` |

The session that produced this report was a second attempt. An earlier Z session on the same day failed repeatedly; its bundle never arrived, but its chat summary did, and two of its findings are recorded below because the report here doesn't repeat them.

**Result:** every claim Z could run is confirmed. Loop claim 3 (attacking the jail) was not attempted: neither jail runs on kernel 5.10 (no Landlock; `/proc` mounts refused), as `loop/README.md` states. The loop was re-measured unjailed after reading the code.

**Checked here before filing** (2026-11-12):
- The five test accuracies are the stated counts out of 1,797 (1708, 1725, 1724, 1735, 1727) and match `loop/results/plan2/` exactly.
- **New minor finding, reproduced and fixed:** `count.py --check` exited 1 ("edited") on an intact record from an older schema. The card defines that case as 2 ("can't compare"). It now exits 2, with a test.
- **The earlier session's finding 1, CPU re-pin: true by the code, effect unmeasured.** The timer pins a candidate to one CPU, but the candidate may change its own affinity under the fallback, and under bubblewrap nothing stops it. On a host whose CPUs are unevenly loaded, it could move to a quieter one. Open; to be fixed by refusing affinity changes in the timed child under both jails.
- **The earlier session's finding 2, `capset` and file handles: not reproduced.** Run as root with every capability (kernel 6.18), `open_by_handle_at` on a file outside the Landlock allow-list fails with `EACCES`, like an ordinary open: Landlock checks the opened file itself, whatever path led to it.
- Z's other notes match known limits already on the cards: the probe's raw-forgery gap (documented), the rare inexact Python count in the counter tests (the counter refuses rather than reporting a wrong number), and `monitor` exiting 2 on GitHub rate limits.
