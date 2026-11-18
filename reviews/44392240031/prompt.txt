You are reviewing a GitHub Actions hardening change in a public repository. Be adversarial and concrete. You see only the diff below.

Context (facts, not claims to trust blindly):
- record.yml is the only writer of a hash-chained ledger; its `append` job signs Sigstore attestations (needs id-token and attestations write) and pushes to main. record-e2e.yml is a test copy whose `append` job must stay byte-identical to record.yml's.
- record.yml's `upgrade` job runs `ots upgrade` on proof files, pushes them, then runs `record.py check-anchors`, which may read attestations through the GitHub API.
- worker.yml is a queue worker: `claim` (writes the queue, no keys), `draft` (has model API keys), `land` (writes main). A file worker/PAUSE on main must stop everything. The new `chain` job makes a run start the next run via workflow_dispatch, because GitHub's scheduler dropped the hourly cron.
- anchor-test.yml, bench.yml, bench2.yml were also disabled in the repo settings (they are finished experiments).

Attack these specific questions, each with a verdict (holds / fails / cannot tell) and the diff line:
1. Does moving permissions from workflow level to job level change what record.yml's `append` job can do? Could `upgrade` now fail because it lacks a permission it used (id-token, attestations write, actions, anything)?
2. Is the `chain` job's loop guard sound? Can it run away (many runs per hour, or more than CHAIN_CAP per day), or never chain? Consider: the GitHub API date filter `created=>=YYYY-MM-DD`, pagination (per_page=100), `triggering_actor` of a GITHUB_TOKEN dispatch, runs racing each other (two runs both seeing 0 waiting and both dispatching), PAUSE check via the contents API, and whether `needs.claim.outputs.item` is empty when paused or out of budget.
3. Are the `zizmor: ignore[artipacked]` reasons true for each job as shown (does that job upload anything containing .git)?
4. Are the pinned SHAs consistent with the version comments (the author resolved them with git ls-remote: checkout v4 = v4.4.0 = 11d5960a..., upload-artifact v4 = v4.6.2 = ea165f8d..., download-artifact v4 = v4.3.0 = d3f86a10..., attest v2 = v2.4.0 = ce27ba3b...)?
5. Anything else that could break a push, a signature, or the pause switch.

Output: a numbered list of findings, each with severity (critical / major / minor), the line, the evidence, and a fix. End with one verdict: ready / ready with fixes / not ready. Do not praise.

DIFF:
diff --git a/.github/workflows/anchor-test.yml b/.github/workflows/anchor-test.yml
index 61ee3e1..e7b7a97 100644
--- a/.github/workflows/anchor-test.yml
+++ b/.github/workflows/anchor-test.yml
@@ -17,7 +17,7 @@ jobs:
         runner: [ubuntu-24.04, ubuntu-24.04-arm]
     runs-on: ${{ matrix.runner }}
     steps:
-      - uses: actions/checkout@v4
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
       - name: try each anchor route
         env:
           GH_TOKEN: ${{ github.token }}
@@ -32,7 +32,7 @@ jobs:
       - name: attest digest (Sigstore / Rekor)
         id: att
         continue-on-error: true
-        uses: actions/attest@v2
+        uses: actions/attest@ce27ba3b4a9a139d9a20a4a07d69fabb52f1e5bc  # v2.4.0
         with:
           subject-name: exagium-record-digest
           subject-digest: sha256:${{ steps.d.outputs.digest }}
@@ -43,7 +43,7 @@ jobs:
         run: |
           echo '{"route":"sigstore_attest","outcome":"${{ steps.att.outcome }}","bundle":"${{ steps.att.outputs.bundle-path }}","attestation_id":"${{ steps.att.outputs.attestation-id }}","url":"${{ steps.att.outputs.attestation-url }}"}' >> out/attempts.jsonl
           cp "${{ steps.att.outputs.bundle-path }}" out/attestation.jsonl 2>/dev/null || true
-      - uses: actions/upload-artifact@v4
+      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
         with:
           name: anchors-${{ matrix.runner }}
           path: out
@@ -52,7 +52,7 @@ jobs:
     if: always()
     runs-on: ubuntu-24.04
     steps:
-      - uses: actions/download-artifact@v4
+      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
         with:
           path: art
       - name: commit results to ci-runs
diff --git a/.github/workflows/bench.yml b/.github/workflows/bench.yml
index bd2ca89..e204bc3 100644
--- a/.github/workflows/bench.yml
+++ b/.github/workflows/bench.yml
@@ -90,7 +90,7 @@ jobs:
               "stderr_tail": err[-1500:]}))
           EOF
           cat "$out" | head -c 3000; echo; echo "exit 0"
-      - uses: actions/upload-artifact@v4
+      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
         if: always()
         with:
           name: bench1-${{ matrix.task }}-${{ matrix.arm.name }}
@@ -104,7 +104,7 @@ jobs:
     runs-on: ubuntu-24.04
     steps:
       - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
-      - uses: actions/download-artifact@v4
+      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
         with: {path: art, pattern: bench1-*}
       - name: commit results to router-runs
         run: |
diff --git a/.github/workflows/bench2.yml b/.github/workflows/bench2.yml
index 50f84be..a47d4e8 100644
--- a/.github/workflows/bench2.yml
+++ b/.github/workflows/bench2.yml
@@ -82,7 +82,7 @@ jobs:
           python3 -c "import json;r=json.load(open('$RUNNER_TEMP/out/result.json'));r.update(task='${{ matrix.task }}',arm='${{ matrix.arm.name }}',model='${{ matrix.arm.model }}',commit='$GITHUB_SHA');json.dump(r,open('$RUNNER_TEMP/out/result.json','w'));print({k:r[k] for k in ('task','arm','passed','end','llm_calls','actions','wall_s','clean','diff')})"
           git -C "$WS" diff --cached HEAD > "$RUNNER_TEMP/out/agent.diff" 2>/dev/null || true
           echo "exit 0"
-      - uses: actions/upload-artifact@v4
+      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
         if: always()
         with:
           name: bench2-${{ matrix.task }}-${{ matrix.arm.name }}
@@ -93,7 +93,7 @@ jobs:
     runs-on: ubuntu-24.04
     steps:
       - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
-      - uses: actions/download-artifact@v4
+      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
         with: {path: art, pattern: bench2-*}
       - name: commit results to router-runs
         run: |
diff --git a/.github/workflows/counter.yml b/.github/workflows/counter.yml
index db73edb..bdc5d3e 100644
--- a/.github/workflows/counter.yml
+++ b/.github/workflows/counter.yml
@@ -18,7 +18,9 @@ jobs:
         slot: [1, 2, 3, 4]
     runs-on: ${{ matrix.runner }}
     steps:
-      - uses: actions/checkout@v4
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
+        with:
+          persist-credentials: false   # this job only reads
       - name: install valgrind
         run: sudo apt-get update -qq && sudo apt-get install -y -qq valgrind
       - name: build workloads
@@ -38,7 +40,7 @@ jobs:
           for n in 20 0; do python3 ../count.py --reps 3 --input mlp.py --out-dir ../../runs/ci --keep-raw ../../runs/ci/raw -- python3 mlp.py $n || true; done
           f=$(ls ../../runs/ci/count-*.json | head -1); python3 ../count.py --check "$f"; echo "check exit $?" > ../../runs/ci/check.txt
           ls ../../runs/ci
-      - uses: actions/upload-artifact@v4
+      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
         with:
           name: runs-${{ matrix.runner }}-${{ matrix.slot }}
           path: runs/ci
@@ -49,7 +51,7 @@ jobs:
     permissions:
       contents: write
     steps:
-      - uses: actions/download-artifact@v4
+      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
         with:
           path: art
       - name: commit records to ci-runs
diff --git a/.github/workflows/digest.yml b/.github/workflows/digest.yml
index 9d48433..90318e4 100644
--- a/.github/workflows/digest.yml
+++ b/.github/workflows/digest.yml
@@ -62,7 +62,7 @@ jobs:
     permissions:
       contents: write
     steps:
-      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0  # zizmor: ignore[artipacked] this job pushes with the checkout token and uploads no workspace artifact
       - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
         with:
           name: facts
diff --git a/.github/workflows/loop.yml b/.github/workflows/loop.yml
index 1d150ff..b77be47 100644
--- a/.github/workflows/loop.yml
+++ b/.github/workflows/loop.yml
@@ -65,7 +65,7 @@ jobs:
             cat "out/test-plan$plan.txt"; echo "exit $rc" | tee -a "out/test-plan$plan.txt"; [ $rc -eq 0 ] || fail=1
           done
           exit $fail
-      - uses: actions/upload-artifact@v4
+      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
         if: ${{ !cancelled() }}
         with:
           name: loop-${{ matrix.runner }}-${{ matrix.slot }}
@@ -120,7 +120,7 @@ jobs:
           cat out/reproduce.txt; echo "exit $r1" | tee -a out/reproduce.txt
           cat out/dev.txt; echo "exit $r2" | tee -a out/dev.txt
           [ $r1 -eq 0 ] && [ $r2 -eq 0 ]
-      - uses: actions/upload-artifact@v4
+      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
         if: ${{ !cancelled() }}
         with:
           name: loop-docker-${{ matrix.runner }}
@@ -132,7 +132,7 @@ jobs:
     permissions:
       contents: write
     steps:
-      - uses: actions/download-artifact@v4
+      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
         with:
           path: art
       - name: commit results to ci-runs
diff --git a/.github/workflows/record-e2e.yml b/.github/workflows/record-e2e.yml
index a4c9755..436a0ec 100644
--- a/.github/workflows/record-e2e.yml
+++ b/.github/workflows/record-e2e.yml
@@ -25,18 +25,19 @@ env:
 concurrency:
   group: record-e2e-writer   # one writer at a time, across every ref
   cancel-in-progress: false
-permissions:
-  contents: write
-  id-token: write
-  attestations: write
+permissions: {}   # each job asks for what it needs (audit Q07)
 defaults:
   run:
     shell: bash   # bash -eo pipefail: a failing command in a pipe fails the step
 jobs:
   append:   # also on the daily schedule and by hand, so proposals stranded by a stepped-aside run are picked up
     runs-on: ubuntu-24.04
+    permissions:   # signs attestations and pushes the ledger
+      contents: write
+      id-token: write
+      attestations: write
     steps:
-      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0   # github.sha: the commit the certificate will name
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0   # github.sha: the commit the certificate will name  # zizmor: ignore[artipacked] this job pushes with the checkout token and uploads no workspace artifact
       - name: step aside if the branch has moved on
         id: current
         run: |
@@ -130,7 +131,7 @@ jobs:
       contents: write
       attestations: read
     steps:
-      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0  # zizmor: ignore[artipacked] this job pushes with the checkout token and uploads no workspace artifact
         with:
           ref: ${{ env.LEDGER_REF }}
       - name: verify the head with gh, pinned and unpinned
diff --git a/.github/workflows/record.yml b/.github/workflows/record.yml
index 124bdf5..0818982 100644
--- a/.github/workflows/record.yml
+++ b/.github/workflows/record.yml
@@ -27,18 +27,19 @@ env:
 concurrency:
   group: record-writer   # one writer at a time, across every ref
   cancel-in-progress: false
-permissions:
-  contents: write
-  id-token: write
-  attestations: write
+permissions: {}   # each job asks for what it needs (audit Q07)
 defaults:
   run:
     shell: bash   # bash -eo pipefail: a failing command in a pipe fails the step
 jobs:
   append:   # also on the daily schedule and by hand, so proposals stranded by a stepped-aside run are picked up
     runs-on: ubuntu-24.04
+    permissions:   # signs attestations and pushes the ledger
+      contents: write
+      id-token: write
+      attestations: write
     steps:
-      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0   # github.sha: the commit the certificate will name
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0   # github.sha: the commit the certificate will name  # zizmor: ignore[artipacked] this job pushes with the checkout token and uploads no workspace artifact
       - name: step aside if the branch has moved on
         id: current
         run: |
@@ -129,8 +130,11 @@ jobs:
     if: ${{ !cancelled() && github.event_name != 'push' }}
     needs: append
     runs-on: ubuntu-24.04
+    permissions:   # pushes upgraded proofs; reads attestations for the 48-hour check; signs nothing
+      contents: write
+      attestations: read
     steps:
-      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0  # zizmor: ignore[artipacked] this job pushes with the checkout token and uploads no workspace artifact
         with:
           ref: ${{ env.LEDGER_REF }}
       - name: complete pending OpenTimestamps proofs
diff --git a/.github/workflows/router.yml b/.github/workflows/router.yml
index 49c5d4f..184e601 100644
--- a/.github/workflows/router.yml
+++ b/.github/workflows/router.yml
@@ -19,7 +19,7 @@ jobs:
   probe:
     runs-on: ubuntu-24.04
     steps:
-      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0  # zizmor: ignore[artipacked] this job pushes with the checkout token and uploads no workspace artifact
       - name: probe every engine
         env:
           OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
diff --git a/.github/workflows/worker.yml b/.github/workflows/worker.yml
index e81f83c..cab829a 100644
--- a/.github/workflows/worker.yml
+++ b/.github/workflows/worker.yml
@@ -26,7 +26,7 @@ jobs:
       item: ${{ steps.out.outputs.item }}
       type: ${{ steps.out.outputs.type }}
     steps:
-      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0  # zizmor: ignore[artipacked] this job pushes with the checkout token and uploads only claim.json from RUNNER_TEMP, never .git
       - name: pause switch
         run: |
           if [ -f worker/PAUSE ]; then echo "paused by worker/PAUSE"; echo "PAUSED=1" >> "$GITHUB_ENV"; fi
@@ -147,7 +147,7 @@ jobs:
       TYPE: ${{ needs.claim.outputs.type }}
       GH_TOKEN: ${{ github.token }}
     steps:
-      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0  # zizmor: ignore[artipacked] this job pushes with the checkout token and uploads only job/ and note.json, never .git
         with:
           fetch-depth: 5
       - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
@@ -216,3 +216,32 @@ jobs:
           path: |
             job/
             note.json
+  chain:
+    # Queue Q57: GitHub's free scheduler delayed or dropped the hourly cron (0 of ~8 fired on 2026-11-18/19), so a run that
+    # found work starts the next run itself. A workflow_dispatch made with the job's own token does start a new run.
+    # Guards: not when paused or when nothing was claimed (no work or no budget: claim leaves item empty); not when a
+    # worker run is already waiting to start (at most one pending); at most CHAIN_CAP chained runs per UTC day. The cron stays as a backup.
+    needs: [claim, land]
+    if: ${{ always() && needs.claim.outputs.item != '' }}
+    runs-on: ubuntu-24.04
+    timeout-minutes: 5
+    permissions:
+      actions: write     # dispatch the next run and list today's runs
+      contents: read     # read worker/PAUSE
+    env:
+      GH_TOKEN: ${{ github.token }}
+      R: ${{ github.repository }}
+      CHAIN_CAP: 24
+    steps:
+      - name: start the next run if the guards allow
+        run: |
+          if gh api "repos/$R/contents/worker/PAUSE?ref=main" >/dev/null 2>&1; then echo "paused: not chaining"; exit 0; fi
+          waiting=$(gh api "repos/$R/actions/workflows/worker.yml/runs?per_page=50" \
+            -q '[.workflow_runs[] | select(.status=="queued" or .status=="waiting" or .status=="pending" or .status=="requested")] | length')
+          if [ "$waiting" -gt 0 ]; then echo "a worker run is already waiting ($waiting): not chaining"; exit 0; fi
+          today=$(date -u +%F)
+          chained=$(gh api "repos/$R/actions/workflows/worker.yml/runs?event=workflow_dispatch&created=>=$today&per_page=100" \
+            -q '[.workflow_runs[] | select(.triggering_actor.login=="github-actions[bot]")] | length')
+          if [ "$chained" -ge "$CHAIN_CAP" ]; then echo "daily cap reached ($chained of $CHAIN_CAP): not chaining"; exit 0; fi
+          gh workflow run worker.yml -R "$R" --ref main
+          echo "chained run $((chained + 1)) of $CHAIN_CAP today (UTC $today)"
