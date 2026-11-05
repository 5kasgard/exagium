# Where should the record's outside anchor live? (item 3, experiment 1)

An anchor has to show that a record's digest existed by a given time, in a way the owner can't quietly backdate, edit or delete. This compares the candidates on that, using measurements wherever possible. Run 2026-11-05.

**Method.**
- **Digest.** All routes anchor the same digest: the sha256 of `runs/count-2026-11-05T073638Z-0feb5e325c25.json`, which is `fe8d93a9…1de8`.
- **CI runs.** `try_anchors.sh` and `.github/workflows/anchor-test.yml` ran on the `anchor-test` branch: runs 41995168381 and 41995230106, 2 runners each (x86-64 and arm64), so 4 attempts per route. Results are on the `ci-runs` branch under `anchors/`.
- **Workspace checks.** Network reach, GH Archive, Software Heritage, offline verification of a CI token, and the public Rekor entry were checked from this workspace.
- **Caveat.** The first run's `attempts.jsonl` has invalid JSON numbers (`.22` with no leading zero, from `bc`). The values are readable and the script is now fixed.

## Measured
| Route | Result (4 attempts) |
|---|---|
| RFC 3161, 6 free TSAs | granted 4/4, verified 4/4 |
| RFC 3161, DFN | granted 4/4, chain not in system store |
| RFC 3161, Apple | no connection 4/4 |
| OpenTimestamps | submitted 4/4, pending |
| Sigstore attestation | 2/2 (second run only), in Rekor |
| Gist via CI token | HTTP 403, 4/4 |
| GH Archive | 0 of our events in 21 h scanned |
| Software Heritage | repo not archived |

**Details:**
- **RFC 3161.** The six that verified are FreeTSA, DigiCert, Sectigo, SSL.com, GlobalSign and Certum.
  - Each answered in 0.05–0.5 s with a 4.6–7.7 KB token.
  - Each token's time matched the runner clock to within 1 s.
  - Verification used the system CA store, plus FreeTSA's own CA for FreeTSA.
  - A DigiCert token re-verified offline in this workspace, and failed after one bit of the digest was flipped.
- **OpenTimestamps.** Stamping took 1.7–5.7 s and gave a 0.7–0.9 KB proof. It is pending until a Bitcoin block confirms it, usually within hours.
  - Full verification needs a Bitcoin node, or trust in a block explorer.
  - It works from this workspace too.
- **Sigstore.** `actions/attest@v2` signed the digest with the workflow's identity.
  - Rekor log index 3692913518 was fetched publicly from `rekor.sigstore.dev` at integratedTime 2026-11-05T08:32:34Z.
  - The certificate names the repo, the workflow file, the ref and commit `4ea7f2c…`. It is valid for 10 minutes.
- **Gist via CI token.** The request was refused with "Resource not accessible by integration". So a gist needs a personal token stored as a CI secret, or a manual step.
- **GH Archive.** It now holds about 4–15 k events per hour, a small sample of GitHub. None of our pushes appeared in the 21 hours scanned. Not a witness.
- **Software Heritage.** The repo isn't archived. A "save code now" request is possible but untested.
- **GitHub repo activity.** The API (`/repos/…/activity`) returns server-side push times without auth, including force pushes and branch creation. The anchor-test push showed as 08:31:46Z. Commit dates, by contrast, are set by the committer.

## What each route can and can't prove
| Route | Who can backdate | Who can delete or hide | Verifier needs |
|---|---|---|---|
| Git commit/tag | owner (commit dates) | owner: force push, or delete repo | the repo |
| Gist | owner | owner: force push, delete | the gist |
| RFC 3161 token | TSA only | owner (we hold the token) | token + CA roots |
| OpenTimestamps | nobody practical | owner (we hold the proof) | proof + Bitcoin headers |
| Sigstore / Rekor | Sigstore only | nobody (public append-only log) | bundle + Sigstore root |

**Notes on the table:**
- **Git.** A force push shows in the activity log, but only while the repo exists. Push times are GitHub's own.
- **Gist.** Revisions can be force-pushed away, and there's no activity log. CI can't write one.
- **RFC 3161.** The trust is in one company's key, but there are 6 independent TSAs to choose from. Validity after the TSA certificate expires is untested; long-term validation may need re-stamping.
- **OpenTimestamps.** Proofs are pending for hours.
- **Sigstore.** It says "this workflow at this commit vouched for D at T", not that D's content is true. The owner writes the workflow. The public-good instance has no SLA.

**Findings:**
- **Repo plus gist gives no independence.** Both sit under the same GitHub account and are rewritable by the same person. A gist is weaker than the repo: it has no activity log, and CI can't write to it.
- **Existence isn't completeness.** Every route proves that a digest existed. Only a public log stops the owner holding back an unfavourable run, because private proofs (TSA, OTS) can simply go unshown. The fix is to hash-chain the records, so each one includes the previous digest, and to anchor the chain head. A missing record then breaks the chain.

## Hypothesis for the next step (to test, not yet decided)
- **The log lives in git:** hash-chained JSON records, which are cheap, readable and diffable.
- **The chain head is anchored by two independent outside witnesses:**
  - a Sigstore attestation from CI, which is public, identity-bound and undeletable;
  - an RFC 3161 token from two TSAs, which is offline-verifiable with a standard CA store and needs no Bitcoin node.
- **OpenTimestamps is optional,** as a slow third witness with no company to trust.

Still to measure: whether `gh attestation verify` and a non-GitHub verifier (sigstore-python) both accept the bundle; whether TSA tokens verify after cert expiry; and whether OTS proofs complete and verify.
