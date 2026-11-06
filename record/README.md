# record — spec card

**What it is.** The third brick: an append-only, hash-chained ledger of **plans** (a claim and the test fixed before it runs) and **verdicts** (the outcome, given by someone other than the claimant).
- **One writer.** Only `.github/workflows/record.yml` writes to it. Each run attests the exact range of entries it added: a Sigstore attestation in the public Rekor log, plus an OpenTimestamps proof in Bitcoin.
- **A stranger's checks:**
  - every entry is covered by the writer's attestations, with no gap;
  - every anchor is genuine;
  - every attestation the writer ever made, on any ref GitHub names, is in the ledger, or is explained by an orphan entry that publishes what it attested.
- **Why these anchors:** `experiments/anchors/RESULTS.md`, attacked by an outside verifier (`experiments/anchors/verifier1/`).

**Run.**
```
python3 record.py verify          # chain intact? also lists open and overdue plans
python3 record.py head            # head seq and sha256
python3 record.py check-anchors   # every anchor right; writer's ranges cover 1..head; each run's commit
python3 record.py monitor         # every Rekor attestation by record.yml on any ref, vs the ledger
python3 -m unittest -v test_record   # 21 tests, <1 s
```
- `verify` and `head` need only Python 3 (stdlib).
- `check-anchors` needs `sigstore==4.5.1` and `opentimestamps-client==0.8.0` (use a virtualenv), plus network access to blockstream.info and mempool.space.
- `monitor` needs network access to the GitHub API and Rekor. To judge orphan entries it needs a full (not shallow) git clone of the ledger branch; a shallow clone exits 2.

**Exit codes:**
- **0:** ok.
- **1:** something is wrong:
  - a ledger rule is broken;
  - an anchor is wrong;
  - the writer's ranges leave a gap or overlap, or don't reach the head;
  - an attestation is unexplained, or an orphan explanation is refused;
  - the ledger branch was force-pushed, or deleted (and recreated).
- **2:** can't check (missing library, network). `check-anchors` checks every anchor before deciding, so a problem in one doesn't hide a problem in another.
- **3:** unexpected failure. Wrong data never exits 3; a failure is never reported as a verdict.

## How it's written
Proposals go in `ledger/inbox/NAME.json` as `{"kind": "plan" | "verdict" | "orphan", "body": {...}}`. A push to `main` that touches the inbox runs `record.yml`, which:
1. **Runs exactly the commit that triggered it,** the one its certificate will name. If `main` has already moved on, it steps aside. The newer push's run, or failing that the daily run, takes the proposals. One writer runs at a time across all refs.
2. **Refuses to add anything unless the writer's earlier attestations cover every existing entry,** so an entry written by hand is never vouched for later.
3. **Ingests the proposals,** stamps the new head with OpenTimestamps, and refuses to attest if `main` has moved.
4. **Attests the range it added:** predicate `{from_seq, to_seq, prev_sha256, head_sha256}`, with the head entry as subject. Then it checks every anchor and commits.
5. **If it fails or is cancelled after attesting,** it pushes the bundle and the entries to the `ci-runs` branch (`orphans/<run>/`), so the orphaned attestation can be explained with its real statement and entries. That branch is owner-writable, so it's a convenience, not evidence.

The daily run also completes pending OpenTimestamps proofs.
- Steps run under `bash -eo pipefail`, so a failing command in a pipe fails the run.
- `record.py` runs under `python3 -I`, so a file next to it can't shadow the standard library.
- `checkout` and `attest` are pinned by commit SHA.
- The pip packages are pinned by version, not by hash.

**Entries.** `ledger/entries/NNNNNN.json`: `{seq, prev, kind, written_at, body}`.
- **An entry's identity** is the sha256 of its stored bytes, and `prev` is the previous entry's.
- **The bytes must be exactly RFC 8785 (JCS)** of the entry, with no trailing newline, so any language can recompute them. Floats are refused (write them as strings). Keys must be ASCII, integers within ±(2⁵³−1), and lone surrogates are refused. Under these rules Python's `json.dumps(sort_keys, separators, ensure_ascii=False)` equals JCS. Fuzzed against the `rfc8785` library: 0 mismatches in 20,000 random objects (a reviewer repeated it); the test runs 3,000.
- **Duplicate keys,** non-canonical bytes, stray files, symlinks and directories are refused.
- **`written_at` is the writer's clock,** a claim. The anchors say when an entry really existed.

**Rules `verify` enforces:**
- seq runs 1, 2, 3…, and `prev` links each entry to the one before;
- body fields are known and well-typed;
- `by` is a lowercase ASCII handle, so look-alike names can't pass as different people;
- `verdict_due` must be a real date;
- a verdict names an earlier plan by seq *and* sha256;
- **one verdict per plan:** a re-run needs a new plan, so re-running until a result looks good shows up as several plans;
- a verdict's `by` differs from its plan's `by`. Handles are claims, not identities.

`verify` lists **open** and **overdue** plans. Completeness can't come from cryptography, so this list is how a missing verdict becomes visible.

**Orphan entries.** An attestation the writer made but the ledger doesn't hold (a run that failed or was cancelled after attesting) is flagged by `monitor` until an `orphan` entry `{by, log_index, reason, statement_b64, entries_b64}` explains it.

The orphan entry must **publish what the run attested**: the exact statement, plus the hidden entries themselves. The entries must be canonical, consecutive from the attested `from_seq`, chained from the attested `prev_sha256`, and hash to the attested head. The explanation is refused if:
- the statement isn't the one Rekor holds for that log index (its payload hash);
- the entries don't reproduce the attested range;
- the attested head was ever published on the ledger branch and later removed (needs a full clone).

So hiding an attested verdict means publishing it inside the orphan entry. The `reason` is a claim; the entries are the evidence.

## What a stranger can check, and how
| Question | How |
|---|---|
| Entry unchanged since anchored? | `check-anchors`: Sigstore subject and OTS proof match its bytes |
| Every entry vouched for by the writer? | `check-anchors`: ranges cover 1..head, no gap or overlap |
| Existed by when? | Rekor `integratedTime`; Bitcoin block time (existence only) |
| Which code wrote each range? | `check-anchors`: each run's certified commit, and whether anything under `record/` or `.github/` changed since the last run |
| Runs held back, forked or deleted? | `monitor`: every writer attestation on any ref is in the ledger or published by an orphan entry; no force push or deletion of the ledger branch |
| Plans without a verdict? | `verify`: open and overdue plans |

**Verify an anchor without this code:**
- **Sigstore:**
  ```
  sigstore verify github --bundle ledger/anchors/N.sigstore.json \
    --cert-identity https://github.com/5kasgard/exagium/.github/workflows/record.yml@refs/heads/main \
    --repository 5kasgard/exagium --ref refs/heads/main sha256:<sha256 of entry N's file>
  ```
- **With `gh`:**
  ```
  gh attestation verify ledger/entries/N.json -R 5kasgard/exagium \
    --signer-workflow 5kasgard/exagium/.github/workflows/record.yml --source-ref refs/heads/main \
    --predicate-type https://github.com/5kasgard/exagium/blob/main/record/README.md
  ```
  Never with `-R` alone (Q1 below).
- **OpenTimestamps:** `ots verify` with a Bitcoin node, or the opentimestamps.org web page from a phone.

## Experiments (2026-11-05/06)
All ran on GitHub Actions with test copies of the writer, each with its own identity. The test copy kept on this branch is `.github/workflows/record-e2e.yml`, whose `append` job is identical to `record.yml`'s apart from names.

**Q1: does `gh attestation verify -R repo` accept other workflows and branches?** Yes.
- With `-R` alone, it accepted an attestation that `anchor-test.yml` made on `refs/heads/anchor-test`.
- Adding `--signer-workflow` or `--source-ref` refused it (`ci-runs` `q1/42092874356.txt`).
- Positive control: pinned to the right writer and ref, `gh` accepts the genuine e2e anchors; pinned to the wrong writer, it refuses (`ci-runs` `e2e/*.txt`, 6 runs).
- A predicate type containing `#` is rejected by GitHub's API (HTTP 422), so ours has none.

**Q2: does the monitor catch a run the owner never publishes?** Yes.
- **Setup:** a writer copy (`record-q2.yml`, since retired) attested a "hidden" entry on a shadow branch, and the run never published it.
- **Result:** `monitor`, reading only the stand-in main's ledger, flagged it (exit 1). It also flagged two earlier failed attempts that had attested before failing. Those weren't planted: even aborted runs leave a public trace.
- **Branch deletion** isn't tested, because this session can't delete branches. GitHub's activity log, which `monitor` reads by cursor, records creation and deletion.

**End to end** (`e2e-main`, 8 entries, ranges 1–1, 2–3, 4–5, 6–7, 8–8):
- **A, first entry:** written, attested, checked.
- **B, a hand-written entry pushed with a proposal:** the writer refused at its pre-check. Nothing was attested (run 42095360021), and the monitor later found no stray attestation.
- **C and D, two racing pushes:** one run took both proposals, and the other found an empty inbox.
- **E, two racing pushes with the final writer:** the older run saw `main` had moved and stepped aside. The newer one ran its own certified commit (`a431e68`) and took both. `check-anchors` reports the writer code changed before entries 6–7, which is true.
- **F, the final writer** (pinned actions, `python3 -I`): entry 8, run 42098215091.
- **Final state:** `check-anchors` exits 0 (writer attested through 8; code changed before 6–7 and before 8, both true). `monitor` exits 0: 5 Rekor entries, 5 anchors, 6 refs, no force push or deletion.

## Review
Three rounds by one independent adversarial agent.

**Round 1 found three demonstrated breaks:**
1. **Hand-written entries passed every check,** and the next CI run would vouch for them. Fixed by range attestations, full coverage, and the writer's pre-check.
2. **The monitor stayed red forever** because of old orphans. Fixed by orphan entries.
3. **Activity paging was broken** (the API pages by cursor). Fixed.

It also found:
- a forged OTS block height read as "can't check" (now wrong);
- a rejected proposal gave a green run (now pipefail);
- a race between pushes;
- OTS counted as coverage;
- several cases of wrong data exiting 3;
- look-alike handles;
- bogus dates;
- an unproven `gh` command.

Each fix has a test or an e2e run.

**Round 2 found two more breaks:**
1. **Laundering:** truncating published entries and explaining their attestation with an orphan. Now refused, by statement and git-history checks. Replayed in a clone: exit 1, "published on HEAD and later removed".
2. **The certificate named a different commit from the code that ran.** Now the writer runs exactly the triggering commit.

It also found that an explorer outage blocked the writer (now tolerated), and that a Rekor 404 read as "no entries" (now can't-check).

**Round 3 found:**
1. **A shallow clone accepted the laundering.** Now exit 2.
2. **A cancelled run's verdict could be hidden** behind an orphan explained only by its statement. Orphans must now publish the hidden entries; tested.
3. **A file next to `record.py` could change its behaviour unreported.** Now `python3 -I`, and the code-change report covers `record/` and `.github/`.
4. **A stepped-aside run could strand proposals.** The daily run now picks them up.
5. **Deleting and recreating the ledger branch wasn't flagged.** Now wrong; untested, since this session can't delete branches.

**Outside verifier Z** (GLM-5.3, 2026-11-06, `verifications/z-20261106/`) judged plan 1 **pass** (ledger entry 4). Every attack it tried exited 1: a hand-written entry, a fully re-referenced rewrite, orphan laundering with the real statement, and a corrupted signature with the network down. It verified an anchor with the sigstore CLI alone and recomputed the chain in Node.js. It found one weakness: deleting a range head's `.ots` proof left every check at exit 0, so the Bitcoin witness could vanish silently, though no entry could change. Fixed after entry 4: `check-anchors` requires a proof for every range head, which the writer always stamps before it attests (test `test_a_missing_ots_proof_for_a_range_head_is_caught`, and Z's case re-run gives exit 1).

## Trust model and known limits
- **No anchor proves completeness.** A run that never reaches the writer leaves no trace: a private run, a fork, code outside CI. Covered: every run of `record.yml` in this repo, on any ref the monitor can name, published or not. Pre-registered plans make a missing verdict visible. Re-measurement (item 4) is the backstop for content.
- **The owner controls the inbox and the code.** A proposal can be deleted before any run; only pre-registered plans and the overdue list catch a missing verdict. The writer runs the repo's own `record.py`, so a code change shows up only as `writer_code_changed: true` between two runs. A stranger has to read that diff; nothing judges it automatically. The pip packages are pinned by version, not by hash.
- **An attestation says** "record.yml at commit X added these entries". It doesn't say they are true.
- **Ref discovery depends on GitHub** (branches, tags, PRs, the activity log). Deleting the repo erases that list. Rekor keeps the attestations, but a stranger would need to scan the log (rekor-monitor). Rekor v2 has no search index; the public instance stays on v1 for now.
- **Orphans.** A failed push after attesting leaves an orphan that stays red until explained. That's intended: the failure is part of the record.
- **The monitor is a command, not a service.** Someone has to run it. A scheduled run from outside the owner's control (Z, or another account) is the next step.
- **OpenTimestamps** proofs are pending for hours. Verifying through the explorers trusts that two operators serve the real chain.
- **Size:** `record.py` is 630 lines, the tests 235, `record.yml` 127. Most of it is defences, as with the counter.

## Files
- `record.py`: the ledger, anchor checks and monitor.
- `test_record.py`: the tests.
- `../ledger/`: entries, anchors and inbox.
- `../.github/workflows/record.yml`: the only writer.
- `../.github/workflows/record-e2e.yml`: the test copy.
- `experiments/anchors/`: the anchor comparison, verifier round 1 and its adjudication, and `ots_check.py`.
