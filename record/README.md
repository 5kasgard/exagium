# record — spec card

**What it is.** The third brick: an append-only, hash-chained ledger of **plans** (a claim and the test fixed before it runs) and **verdicts** (the outcome, given by someone other than the claimant). Each new head is anchored outside the owner's control: a Sigstore attestation in the public Rekor log and an OpenTimestamps proof in Bitcoin. A monitor lists every attestation the ledger's workflow has ever made and flags any the ledger doesn't hold. Why these anchors: `experiments/anchors/RESULTS.md`, which an outside verifier attacked (`experiments/anchors/verifier1/`).

**Run.**
```
python3 record.py verify          # chain intact? also lists open and overdue plans
python3 record.py head            # head seq and sha256
python3 record.py check-anchors   # every anchor: right bytes, right signer, when it existed
python3 record.py monitor         # every Rekor attestation by record.yml on any ref, vs the ledger
python3 -m unittest -v test_record   # 14 tests, <1 s
```
`verify` and `head` need only Python 3 (stdlib). `check-anchors` needs `sigstore==4.5.1` and `opentimestamps-client==0.8.0`, ideally in a virtualenv, plus network access to blockstream.info and mempool.space. `monitor` needs network access to the GitHub API and Rekor.

**Exit codes:**
- **0:** ok.
- **1:** the ledger or an anchor is wrong, or `monitor` found an attestation the ledger doesn't hold (or the reverse).
- **2:** can't check (missing library, network).
- **3:** unexpected failure. A failure is never reported as a verdict.

## How it's written
**CI is the only writer.** Proposals go in `ledger/inbox/` as `{"kind": "plan" | "verdict", "body": {...}}`. A push to `main` that touches the inbox runs `.github/workflows/record.yml`, which:
1. ingests the proposals in file-name order;
2. stamps the new head with OpenTimestamps;
3. attests the head with Sigstore (`actions/attest`);
4. checks every anchor;
5. commits to `main`.

A daily job completes pending OpenTimestamps proofs. Anchors cover the head they were made on, and through the chain every entry before it.

**Entries.** `ledger/entries/NNNNNN.json`, one per file: `{seq, prev, kind, written_at, body}`.
- **An entry's identity** is the sha256 of its stored bytes, and `prev` is the previous entry's.
- **The bytes must be exactly RFC 8785 (JCS)** of the entry, with no trailing newline, so any language can recompute them. Floats are refused (write them as strings), keys must be ASCII, and integers must be within ±(2⁵³−1). Under those rules Python's `json.dumps(sort_keys, separators, ensure_ascii=False)` equals JCS. Fuzzed against the `rfc8785` library: 0 mismatches in 20,000 random objects, and a test repeats it.
- **Duplicate keys** and non-canonical bytes are refused.
- **`written_at` is the writer's clock,** a claim. The anchors say when an entry really existed.

**Rules `verify` enforces:**
- seq runs 1, 2, 3… with no gaps;
- `prev` links each entry to the one before;
- the body fields are known and well-typed;
- a verdict names an earlier plan by seq *and* sha256;
- **one verdict per plan:** a re-run needs a new plan, so re-running until a good result shows up as several plans;
- a verdict's `by` differs from its plan's `by`. These names are claims, not identities.

`verify` also lists **open plans** (no verdict yet) and **overdue plans** (past `verdict_due`). Completeness can't come from cryptography, so this list is how a missing verdict becomes visible.

## What a stranger can check, and how
| Question | How |
|---|---|
| Entry unchanged since anchored? | `check-anchors`: OTS proof and Sigstore subject must match its bytes |
| Existed by when? | earliest of Rekor `integratedTime` and the Bitcoin block time |
| Written by this repo's `record.yml` on `main`? | `check-anchors` pins the full identity (workflow, ref, repo, issuer) |
| Runs held back or forked? | `monitor`: every Rekor attestation by `record.yml` on any ref must be in the ledger |
| Plans without a verdict? | `verify`: open and overdue plans |

**Verify an anchor by hand, without this code:**
- **Sigstore:**
  ```
  sigstore verify github --bundle ledger/anchors/N.sigstore.json \
    --cert-identity https://github.com/5kasgard/exagium/.github/workflows/record.yml@refs/heads/main \
    --repository 5kasgard/exagium --ref refs/heads/main sha256:<sha256 of the entry file>
  ```
- **With `gh`:**
  ```
  gh attestation verify ledger/entries/N.json -R 5kasgard/exagium \
    --signer-workflow 5kasgard/exagium/.github/workflows/record.yml --source-ref refs/heads/main \
    --predicate-type https://github.com/5kasgard/exagium/blob/main/record/README.md
  ```
  Never with `-R` alone (Q1 below).
- **OpenTimestamps:** `ots verify` with a Bitcoin node, or the opentimestamps.org web page from a phone.

## Experiments behind it (2026-11-05/06)
**Q1: does `gh attestation verify -R repo` accept other workflows and branches?** Yes.
- With only `-R` (and the predicate type), it accepted an attestation that `anchor-test.yml` made on `refs/heads/anchor-test`.
- Adding `--signer-workflow` or `--source-ref` refused it ("expected SourceRepositoryRef to be refs/heads/q2-main, got refs/heads/anchor-test").
- Record: `ci-runs` branch, `q1/42092874356.txt`.
- Side finding: a predicate type containing `#` is rejected by GitHub's attestation API (HTTP 422), so ours has none.

**Q2: does the monitor catch a run the owner never publishes?** Yes.
- **Setup:** a test copy of the writer (`record-q2.yml`) ran on a stand-in main (`q2-main`) and on a shadow branch (`q2-shadow`) that added a "hidden" entry. The shadow run attested its head and then failed, so it never published.
- **Result:** `monitor`, reading only the stand-in main's ledger, found 4 Rekor entries for that workflow, matched 1, and flagged 3 (exit 1):
  - the shadow run;
  - two earlier failed attempts on `q2-main` that had attested before failing. These weren't planted. They show that even an aborted run leaves a public trace.
- **Branch deletion** isn't tested, because the session can't delete branches. GitHub's activity log records branch creation and deletion, and `monitor` reads it, so a deleted branch stays discoverable unless the whole repo is deleted.
- **Pull-request refs** (`refs/pull/N/merge`) appear in neither the branch list nor the activity log, so `monitor` also lists every PR.

**Found while building:**
- **A crash read as a verdict.** On the runner, pip's newer `cryptography` broke the system `pyOpenSSL`, and the sigstore CLI crashed. The first version reported that as "verification failed" (exit 1). Now the sigstore library is called directly: only a `VerificationError` is a verdict, anything else exits 3. CI installs its tools in a virtualenv.
- **Identical statements share a payload hash,** so `monitor` keys on Rekor log index, which is unique per attestation. Keying on payload hash would hide a re-run of the same head.

## Trust model and known limits
- **No anchor proves completeness.** A run that never reaches the workflow leaves no trace: a private run, a run on a fork, or code run outside CI. What *is* covered: every run of `record.yml` in this repo, on any ref the monitor can name, published or not. Pre-registered plans make a missing verdict visible. Re-measurement (item 4) is the backstop for content.
- **The owner writes the workflow.** An attestation says "record.yml at commit X vouched for this entry", not that the entry is true. A stranger can read the workflow at the commit named in the certificate.
- **Ref discovery.** `monitor` finds refs through the GitHub API (branches, tags, PRs, activity log). Deleting the repo erases that list. Rekor still holds the attestations, but a stranger would have to guess the refs or scan the log (rekor-monitor). Rekor v2 has no search index at all; the public instance stays on v1 for now.
- **Orphaned attestations.** If the commit fails after attesting, the attestation is orphaned and `monitor` flags it forever. That's intended: the failure is part of the record. The job orders steps so that most failures happen before attesting.
- **The monitor is a command, not yet a service.** Someone has to run it. A scheduled run from outside the owner's control (Z, or another account) is the next step.
- **OpenTimestamps** proofs stay pending for hours until the daily upgrade, and verifying with the explorers trusts that they serve the real chain (two operators must agree).
- **Names in `by`** are claims. Separation of claimant and judge is recorded, not proven.
- **Size.** `record.py` is 434 lines; the tests are 172.

## Files
- `record.py`: the ledger, anchor checks and monitor.
- `test_record.py`: the tests.
- `../ledger/`: entries, anchors and inbox.
- `../.github/workflows/record.yml`: the only writer.
- `experiments/anchors/`: the anchor comparison, verifier round 1 and its adjudication, `ots_check.py`, and the q2 test workflow (on the `anchor-test` branch).
