You are reviewing a small security change in a public research repo where untrusted, machine-generated C programs ("candidates", each an infer.c) are compiled and timed. Rule: candidate code must never be compiled by the host compiler, because a hostile #include of a host path (a key file, the runner's home) would copy that file into the binary and so into public outputs. runner/jail.py has compile_candidate(src, out): it copies only the source into a fresh directory bound at /w inside the jail (bubblewrap with namespaces, or a Landlock+seccomp "confine" fallback), runs gcc there with rlimits and without the seccomp filter (gcc forks), and copies the binary out. jail.check() selects the jail mode and fails closed unless resource limits hold inside the jail and the jailed compiler refuses to #include a planted host file. The trusted timer (the repo's own C file) is still built on the host by design.

The change below moves loop/scale.py (a cost-per-image meter at large run sizes) from host compilation to compile_candidate, switches its startup check from loop.check_jail() to jail.check(), adds a unit test, and adds a CI smoke step. Evidence offered: unit tests pass, the new test fails when the host build is restored (mutation check), objdump of host-built vs jail-built code is identical for all 5 candidates, and an end-to-end local run passed.

Review critically. Numbered findings only, each with severity (major/minor), the exact line, what is wrong, and the fix. Look for: any remaining path where candidate code or candidate-controlled files reach the host compiler or host tools; whether importing runner/jail.py into scale.py changes behaviour elsewhere (it sets OPENBLAS_CORETYPE=Haswell in loop's jail environment on x86 when imported); whether the test really pins the property; problems in the CI step. If nothing is wrong, say "no findings". No praise.

```diff
diff --git a/.github/workflows/loop.yml b/.github/workflows/loop.yml
index b77be47e..3cf98b40 100644
--- a/.github/workflows/loop.yml
+++ b/.github/workflows/loop.yml
@@ -51,6 +51,13 @@ jobs:
           shopt -s nullglob; dirs=(loop/baselines/*/ loop/candidates/*/); [ ${#dirs[@]} -gt 0 ] || { echo "no candidate directories"; exit 1; }
           set +e; "$RUNNER_TEMP/venv/bin/python" loop/loop.py evaluate "${dirs[@]}" --test dev --records out/records --out out/dev.json > out/dev.txt 2>&1; rc=$?; set -e
           cat out/dev.txt; echo "exit $rc" | tee -a out/dev.txt; exit $rc
+      - name: scale.py smoke (queue Q56b, candidates compiled in the jail)
+        if: ${{ !cancelled() }}
+        run: |
+          # small sizes, 2 rounds: checks the jailed compile, the reference predictions and the per-segment checks, not costs
+          set +e; "$RUNNER_TEMP/venv/bin/python" -I loop/scale.py loop/candidates/mlp16 loop/baselines/linear --sizes 0,956,4096 --rounds 2 \
+            --out out/scale-smoke.json > out/scale-smoke.txt 2>&1; rc=$?; set -e
+          cat out/scale-smoke.txt; echo "exit $rc" | tee -a out/scale-smoke.txt; exit $rc
       - name: score requested plans on the test set
         if: ${{ !cancelled() }}
         run: |
diff --git a/loop/scale.py b/loop/scale.py
index 51c485de..43b674b0 100644
--- a/loop/scale.py
+++ b/loop/scale.py
@@ -28,7 +28,8 @@ input (one big buffer, blocks, mmap). That is part of the candidate. Every candi
 
 Not measured: block steps. One extra image costs 0.03-0.3 us; single runs vary by 75-3,800 us here.
 
-Reuses loop.py (frozen by plan 2) by import; changes nothing in it.
+Reuses loop.py (frozen by plan 2) by import; changes nothing in it. Candidate C is compiled inside the jail by
+runner/jail.py (queue Q56b), never on the host; only the trusted timer is built on the host.
 Usage: python3 loop/scale.py DIR... [--sizes 0,956,4096,16384,65536] [--rounds 9] [--seed 1] [--out F]
        [--provenance TEXT]   (where candidates outside this repo came from: branch and commit)
 Exit codes: 0 done, no problems; 1 a check failed or problems were found; 3 failure.
@@ -38,6 +39,8 @@ import argparse, json, math, os, random, shutil, statistics, sys, tempfile
 HERE = os.path.dirname(os.path.abspath(__file__))
 sys.path.insert(0, HERE)  # python3 -I drops the script's directory
 import loop as L  # noqa: E402
+sys.path.insert(0, os.path.join(os.path.dirname(HERE), "runner"))
+import jail as J  # noqa: E402  (Q56b: compile candidates in the jail; imports loop itself, the same module)
 
 DEFAULT_SIZES = (0, 956, 4096, 16384, 65536)
 NUDGES = 4
@@ -153,7 +156,7 @@ def measure(dirs, sizes, rounds, seed):
             src = os.path.join(work, name, "infer.c")
             shutil.copyfile(os.path.join(d, "infer.c"), src)
             c = {"name": name, "dir": d, "exe": os.path.join(work, name, "infer")}
-            L.build(src, c["exe"])
+            J.compile_candidate(src, c["exe"])   # Q56b: never compile candidate code on the host
             c["ref"], _ = L.timed(c, imgs, timer, cpu)  # reference predictions on the whole pool (not a timing)
             acc[name] = accuracy_check(c["ref"], labels, sizes)
             if acc[name]["problem"]:
@@ -221,7 +224,7 @@ def main(argv=None):
         if a.rounds < 2:
             ap.error("--rounds must be >= 2")
         dirs = [L.candidate(d) for d in a.dirs]
-        L.check_jail()
+        J.check()   # loop's jail mode, then fail closed unless limits hold and the jailed compiler refuses host files
         s = measure(dirs, sizes, a.rounds, a.seed)
         s["provenance"] = a.provenance
         show(s)
diff --git a/loop/test_scale.py b/loop/test_scale.py
index 027011d6..8dffe73d 100644
--- a/loop/test_scale.py
+++ b/loop/test_scale.py
@@ -109,19 +109,31 @@ class AccuracyMiddle(unittest.TestCase):
 class Measure(unittest.TestCase):
     """measure() with the jail, compiler and timer faked: the checks must fire."""
 
-    def run_measure(self, behaviour):
+    def run_measure(self, behaviour, built=None):
         calls = {"n": 0}
+        built = [] if built is None else built
 
         def fake_timed(c, rows, timer, cpu):
             calls["n"] += 1
             return behaviour(rows, calls["n"]), 1_000_000 + 50 * len(rows)
 
-        with mock.patch.object(L, "build", lambda src, out: None), \
+        with mock.patch.object(L, "build", lambda src, out: built.append(("host", src))), \
+             mock.patch.object(scale.J, "compile_candidate", lambda src, out: built.append(("jail", src))), \
              mock.patch.object(L, "timed", fake_timed), \
              mock.patch.dict(L.JAIL, {"mode": "confine"}), \
              mock.patch.object(L, "versions", lambda: {}):
             return scale.measure([MLP16], [0, 956, 4096], 2, 1)
 
+    def test_candidates_compile_in_the_jail_and_only_the_timer_on_the_host(self):
+        """Q56b: candidate C is untrusted, so it never reaches the host gcc; the trusted timer still does."""
+        built = []
+        self.run_measure(lambda rows, k: self.labels_for(rows), built)
+        self.assertEqual([w for w, s in built if w == "host"], ["host"])
+        self.assertEqual([s for w, s in built if w == "host"], [L.TIMER])
+        jailed = [s for w, s in built if w == "jail"]
+        self.assertEqual(len(jailed), 1)
+        self.assertTrue(jailed[0].endswith(os.path.join("mlp16", "infer.c")))
+
     def labels_for(self, rows):
         imgs, labels = scale.pool(4096, 1)
         index = {tuple(r): y for r, y in zip(imgs, labels)}
```
